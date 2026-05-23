import argparse
import os
import shlex
import subprocess
from typing import Dict, List


RUNNER_BY_DATASET = {
    "alpacaeval2": "scripts/run_alpacaeval2.py",
    "flask_hard": "scripts/run_flask.py",
    "mtbench": "scripts/run_mt_bench.py",
}


def build_variant_args(name: str, k: int) -> List[str]:
    if name == "full":
        return []
    if name == "no_offline_warm_start":
        return ["--warm_start", "random"]
    if name == "no_meta_vectors":
        return ["--no_meta_vectors"]
    if name == "no_query_embedding":
        return ["--no_query_embedding"]
    if name == "no_hadamard":
        return ["--no_hadamard"]
    if name == "no_cost_penalty":
        return ["--no_cost_penalty"]
    if name.startswith("k_"):
        return ["--k", str(k)]
    raise ValueError(f"Unknown variant: {name}")


def default_variants(k_values: List[int]) -> List[str]:
    variants = [
        "full",
        "no_offline_warm_start",
        "no_meta_vectors",
        "no_query_embedding",
        "no_hadamard",
        "no_cost_penalty",
    ]
    variants.extend([f"k_{k}" for k in k_values])
    return variants


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="alpacaeval2", choices=sorted(RUNNER_BY_DATASET))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-k", type=int, default=2)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--lam-cost", type=float, default=0.01)
    ap.add_argument("--out-dir", type=str, default="runs/ablations")
    ap.add_argument("--cache-dir", type=str, default="runs/ablation_cache")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--variants", type=str, nargs="*", default=None)
    args = ap.parse_args()

    runner = RUNNER_BY_DATASET[args.dataset]
    variants = args.variants or default_variants(args.k_values)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    commands: Dict[str, List[str]] = {}
    for variant in variants:
        k = args.base_k
        if variant.startswith("k_"):
            k = int(variant.split("_", 1)[1])
        out_log = os.path.join(args.out_dir, f"{args.dataset}_{variant}.jsonl")
        cache = os.path.join(args.cache_dir, f"{args.dataset}_{variant}.jsonl")
        cmd = [
            "python",
            runner,
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--k",
            str(k),
            "--lam_cost",
            str(args.lam_cost),
            "--out_log",
            out_log,
            "--cache",
            cache,
        ]
        if args.dataset == "flask_hard":
            cmd.extend(["--dataset", "flask_hard"])
        if args.dataset == "mtbench":
            cmd.extend(["--model_id", f"ces_{variant}"])
        cmd.extend(build_variant_args(variant, k))
        commands[variant] = cmd

    for variant, cmd in commands.items():
        print(f"[{variant}] {shell_join(cmd)}")
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
