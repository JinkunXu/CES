import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(REPO_ROOT)

from src.provider.openrouter_client import OpenRouterClient  # noqa: E402


DiskCache = None
build_prompt = None
default_expert_specs = None
load_objective_examples = None
parse_choice_letter = None
usage_cost_usd = None


ARC_SPLIT_ALIASES = {
    "eval": "validation",
    "evaluation": "validation",
    "val": "validation",
    "dev": "validation",
}


SYSTEM_PROMPT_SUFFIX = (
    "\n\nFor this benchmark, return the final answer as a single letter "
    "from the listed options."
)


def normalize_arc_split(split: str) -> str:
    return ARC_SPLIT_ALIASES.get(split.lower(), split)


def ensure_objective_helpers_loaded() -> None:
    global DiskCache
    global build_prompt
    global default_expert_specs
    global load_objective_examples
    global parse_choice_letter
    global usage_cost_usd

    if default_expert_specs is not None:
        return

    try:
        from run_objective_mcq import (  # noqa: WPS433
            DiskCache as _DiskCache,
            build_prompt as _build_prompt,
            default_expert_specs as _default_expert_specs,
            load_objective_examples as _load_objective_examples,
            parse_choice_letter as _parse_choice_letter,
            usage_cost_usd as _usage_cost_usd,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "datasets":
            raise RuntimeError(
                "The current Python environment does not have `datasets` installed. "
                "Activate the CES environment or install it before running evaluation."
            ) from exc
        raise

    DiskCache = _DiskCache
    build_prompt = _build_prompt
    default_expert_specs = _default_expert_specs
    load_objective_examples = _load_objective_examples
    parse_choice_letter = _parse_choice_letter
    usage_cost_usd = _usage_cost_usd


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def resolve_model_specs(
    requested: Optional[Iterable[str]],
    skipped: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    ensure_objective_helpers_loaded()
    specs = default_expert_specs()
    skip_set = {item.strip() for item in (skipped or []) if item.strip()}

    if requested:
        wanted = [item.strip() for item in requested if item.strip()]
    else:
        wanted = ["all"]

    if any(item.lower() == "all" for item in wanted):
        selected = list(specs)
    else:
        by_name = {spec["name"]: spec for spec in specs}
        by_model = {spec["model"]: spec for spec in specs}
        selected = []
        unknown = []
        for item in wanted:
            spec = by_name.get(item) or by_model.get(item)
            if spec is None:
                unknown.append(item)
            elif spec not in selected:
                selected.append(spec)
        if unknown:
            available = ", ".join(spec["name"] for spec in specs)
            raise ValueError(
                f"Unknown model selector(s): {', '.join(unknown)}. "
                f"Use a model name or API id from run_objective_mcq.py. "
                f"Available names: {available}"
            )

    if skip_set:
        selected = [
            spec
            for spec in selected
            if spec["name"] not in skip_set and spec["model"] not in skip_set
        ]

    if not selected:
        raise ValueError("No models selected.")

    return selected


def build_messages(spec: Dict[str, Any], prompt: str, append_suffix: bool) -> List[Dict[str, str]]:
    system_prompt = spec["sys"]
    if append_suffix:
        system_prompt = system_prompt.rstrip() + SYSTEM_PROMPT_SUFFIX
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def call_model(
    client: OpenRouterClient,
    cache: Any,
    spec: Dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    messages = build_messages(spec, prompt, args.append_system_suffix)
    key = {
        "type": "arc_challenge_single_model",
        "model": spec["model"],
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    cached = cache.get(key)
    if cached is not None:
        return {"cached": True, "latency_s": 0.0, "response": cached}

    started = time.perf_counter()
    out = client.chat(
        model=spec["model"],
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        backoff=args.backoff,
    )
    latency_s = time.perf_counter() - started

    if out.get("ok") or args.cache_errors:
        cache.put(key, out)

    return {"cached": False, "latency_s": latency_s, "response": out}


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    total_cost = sum(float(record.get("cost", 0.0) or 0.0) for record in records)
    return {
        "num_questions": n,
        "accuracy": (sum(float(record["correct"]) for record in records) / n) if n else 0.0,
        "parse_rate": (sum(float(record["parsed"]) for record in records) / n) if n else 0.0,
        "error_rate": (sum(0.0 if record.get("ok") else 1.0 for record in records) / n) if n else 0.0,
        "num_errors": sum(0 if record.get("ok") else 1 for record in records),
        "total_cost": total_cost,
        "cost_per_question": (total_cost / n) if n else 0.0,
        "avg_latency_s": (sum(float(record.get("latency_s", 0.0) or 0.0) for record in records) / n) if n else 0.0,
        "cache_hit_rate": (sum(1.0 if record.get("cached") else 0.0 for record in records) / n) if n else 0.0,
    }


def write_summary(summary_path: str, summaries: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def write_summary_csv(csv_path: str, summaries: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(csv_path)
    fieldnames = [
        "dataset",
        "split",
        "model_name",
        "model",
        "num_questions",
        "accuracy",
        "parse_rate",
        "error_rate",
        "num_errors",
        "total_cost",
        "cost_per_question",
        "avg_latency_s",
        "cache_hit_rate",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})


def evaluate_model(
    client: OpenRouterClient,
    cache: Any,
    spec: Dict[str, Any],
    examples: List[Dict[str, Any]],
    args: argparse.Namespace,
    wf,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    correct_hist: List[float] = []
    parsed_hist: List[float] = []
    cost_hist: List[float] = []

    desc = f"{spec['name']} arc_challenge"
    for idx, example in enumerate(tqdm(examples, desc=desc)):
        prompt = build_prompt(example["question"], example["choices"])
        call = call_model(client, cache, spec, prompt, args)
        response = call["response"]
        text = (response.get("text") or "").strip() if isinstance(response, dict) else str(response)
        pred = parse_choice_letter(text)
        ok = bool(response.get("ok", False)) if isinstance(response, dict) else False
        cost = usage_cost_usd(response)
        correct = float(pred == example["gold"])
        parsed = float(pred is not None)

        correct_hist.append(correct)
        parsed_hist.append(parsed)
        cost_hist.append(cost)

        record = {
            "t": idx,
            "dataset": args.dataset,
            "split": args.split,
            "requested_split": args.requested_split,
            "model_name": spec["name"],
            "model": spec["model"],
            "question": example["question"],
            "choices": example["choices"],
            "gold": example["gold"],
            "pred": pred,
            "correct": correct,
            "parsed": parsed,
            "ok": ok,
            "cached": call["cached"],
            "latency_s": call["latency_s"],
            "response": text,
            "usage": response.get("usage", {}) if isinstance(response, dict) else {},
            "cost": cost,
            "error_type": response.get("error_type") if isinstance(response, dict) else None,
            "error_msg": response.get("error_msg") if isinstance(response, dict) else None,
            "status_code": response.get("status_code") if isinstance(response, dict) else None,
            "request_id": response.get("request_id") if isinstance(response, dict) else None,
            "avg_accuracy": sum(correct_hist) / len(correct_hist),
            "avg_parse_rate": sum(parsed_hist) / len(parsed_hist),
            "avg_cost": sum(cost_hist) / len(cost_hist),
            "raw": example["raw"],
        }
        records.append(record)
        wf.write(json.dumps(record, ensure_ascii=False) + "\n")
        wf.flush()

        if args.sleep > 0 and not call["cached"]:
            time.sleep(args.sleep)

    summary = summarize_records(records)
    summary.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "requested_split": args.requested_split,
            "seed": args.seed,
            "limit": args.limit,
            "model_name": spec["name"],
            "model": spec["model"],
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate each CES expert LLM on ARC-Challenge with OpenRouter."
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=200, help="Use <= 0 to evaluate all labeled examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated CES expert names or OpenRouter API model ids. Use 'all' for every expert.",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        default="",
        help="Comma-separated expert names or API model ids to exclude.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.8)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--append-system-suffix", action="store_true")
    parser.add_argument("--cache-errors", action="store_true")
    parser.add_argument("--list-models", action="store_true", help="Print CES expert names and API ids, then exit.")
    parser.add_argument("--out-log", type=str, default="runs/objective_mcq/arc_challenge_models.jsonl")
    parser.add_argument("--summary-out", type=str, default="runs/objective_mcq/arc_challenge_model_summary.json")
    parser.add_argument("--summary-csv", type=str, default="runs/objective_mcq/arc_challenge_model_summary.csv")
    parser.add_argument("--cache", type=str, default="runs/objective_mcq/cache_arc_challenge_models.jsonl")
    args = parser.parse_args()

    args.dataset = "arc_challenge"
    args.requested_split = args.split
    args.split = normalize_arc_split(args.split)
    args.limit = None if args.limit is not None and args.limit <= 0 else args.limit
    args.out_log = resolve_repo_path(args.out_log)
    args.summary_out = resolve_repo_path(args.summary_out)
    args.summary_csv = resolve_repo_path(args.summary_csv)
    args.cache = resolve_repo_path(args.cache)

    ensure_objective_helpers_loaded()
    selected_specs = resolve_model_specs(
        requested=parse_csv_arg(args.models),
        skipped=parse_csv_arg(args.skip_models),
    )

    if args.list_models:
        for spec in selected_specs:
            print(f"{spec['name']}\t{spec['model']}")
        return

    ensure_parent_dir(args.out_log)
    cache = DiskCache(args.cache)
    client = OpenRouterClient()
    examples = load_objective_examples(args.dataset, args.split, args.limit, args.seed)

    summaries: List[Dict[str, Any]] = []
    with open(args.out_log, "w", encoding="utf-8") as wf:
        for spec in selected_specs:
            summary = evaluate_model(client, cache, spec, examples, args, wf)
            summaries.append(summary)
            write_summary(args.summary_out, summaries)
            write_summary_csv(args.summary_csv, summaries)
            print(
                f"[{spec['name']}] "
                f"accuracy={summary['accuracy']:.4f} "
                f"parse_rate={summary['parse_rate']:.4f} "
                f"errors={summary['num_errors']} "
                f"total_cost={summary['total_cost']:.6f}"
            )

    print(f"[DONE] wrote detail log to: {args.out_log}")
    print(f"[DONE] wrote summary JSON to: {args.summary_out}")
    print(f"[DONE] wrote summary CSV to: {args.summary_csv}")


if __name__ == "__main__":
    main()
