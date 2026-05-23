import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregators.llm_aggregator import AggregatorConfig, LLMAggregator
from src.embeddings.encoders import STTextEncoder
from src.experts.base import ExpertMeta
from src.experts.llm_expert import LLMExpertConfig, OpenRouterLLMExpert
from src.orchestration.router import Router


CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]
ANSWER_PATTERNS = [
    re.compile(r"^\s*([A-F])\s*$", re.I),
    re.compile(r"\b(?:ANSWER|FINAL ANSWER|FINAL|CHOICE)\s*[:\-]?\s*[\(\[]?\s*([A-F])\s*[\)\]]?\b", re.I),
    re.compile(r"\bTHE ANSWER IS\s*[\(\[]?\s*([A-F])\s*[\)\]]?\b", re.I),
]


class DiskCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.db[obj["key"]] = obj["val"]

    def _key(self, obj: Any) -> str:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, obj: Any) -> Optional[Any]:
        return self.db.get(self._key(obj))

    def put(self, obj: Any, val: Any) -> Any:
        key = self._key(obj)
        if key in self.db:
            return self.db[key]
        self.db[key] = val
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n")
        return val


def usage_cost_usd(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return 0.0
    return float(usage.get("cost", 0.0) or 0.0)


def init_mats_pca_warmstart(d_x: int, d_u: int, s_list: List[np.ndarray], seed: int = 0, eps: float = 1e-6):
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(d_u, d_u))
    q, _ = np.linalg.qr(mat)
    w_x = q[:d_x, :].astype(np.float64) * (1.0 / np.sqrt(max(d_u, 1)))

    s_mat = np.stack([np.asarray(s, dtype=np.float64) for s in s_list], axis=0)
    d_s = s_mat.shape[1]
    s_mean = s_mat.mean(axis=0)
    s_std = np.maximum(s_mat.std(axis=0), eps)
    z = (s_mat - s_mean) / s_std
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    k = min(d_x, d_s)
    w_m = np.zeros((d_x, d_s), dtype=np.float64)
    w_m[:k, :] = vt[:k, :] * (1.0 / np.sqrt(max(d_s, 1)))
    if d_x > k:
        w_m[k:, :] = rng.normal(size=(d_x - k, d_s)) * (1e-3 / np.sqrt(max(d_s, 1)))
    return w_x, w_m


def init_mats_random(d_x: int, d_u: int, d_s: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(d_u, d_u))
    q, _ = np.linalg.qr(mat)
    w_x = q[:d_x, :].astype(np.float64) * (1.0 / np.sqrt(max(d_u, 1)))
    w_m = rng.normal(size=(d_x, d_s)).astype(np.float64) * (1.0 / np.sqrt(max(d_s, 1)))
    return w_x, w_m


def parse_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(stripped)
        if match:
            return match.group(1).upper()
    match = re.search(r"(?<![A-Z0-9])([A-F])(?![A-Z0-9])", stripped.upper())
    return match.group(1) if match else None


def build_prompt(question: str, choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for idx, choice in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[idx]}. {choice}")
    lines.append("")
    lines.append("Think briefly if needed, then answer with a single letter only.")
    return "\n".join(lines)


def normalize_mmlu_example(example: Dict[str, Any]) -> Dict[str, Any]:
    answer = example["answer"]
    gold = CHOICE_LETTERS[int(answer)] if isinstance(answer, int) else str(answer).strip().upper()
    return {
        "question": example["question"],
        "choices": list(example["choices"]),
        "gold": gold,
        "subject": example.get("subject", "mmlu"),
        "raw": example,
    }


def normalize_arc_example(example: Dict[str, Any]) -> Dict[str, Any]:
    labels = list(example["choices"]["label"])
    texts = list(example["choices"]["text"])
    normalized_choices = []
    label_to_letter: Dict[str, str] = {}
    for idx, (label, text) in enumerate(zip(labels, texts)):
        letter = CHOICE_LETTERS[idx]
        label_to_letter[str(label).strip().upper()] = letter
        normalized_choices.append(text)
    answer_key = str(example.get("answerKey", "")).strip().upper()
    gold = label_to_letter.get(answer_key, answer_key if answer_key in CHOICE_LETTERS else None)
    return {
        "question": example["question"],
        "choices": normalized_choices,
        "gold": gold,
        "subject": example.get("id", "arc"),
        "raw": example,
    }


def normalize_commonsenseqa_example(example: Dict[str, Any]) -> Dict[str, Any]:
    labels = list(example["choices"]["label"])
    texts = list(example["choices"]["text"])
    normalized_choices = []
    label_to_letter: Dict[str, str] = {}
    for idx, (label, text) in enumerate(zip(labels, texts)):
        letter = CHOICE_LETTERS[idx]
        label_to_letter[str(label).strip().upper()] = letter
        normalized_choices.append(text)
    answer_key = str(example.get("answerKey", "")).strip().upper()
    gold = label_to_letter.get(answer_key, answer_key if answer_key in CHOICE_LETTERS else None)
    return {
        "question": example["question"],
        "choices": normalized_choices,
        "gold": gold,
        "subject": "commonsenseqa",
        "raw": example,
    }


def normalize_hellaswag_example(example: Dict[str, Any]) -> Dict[str, Any]:
    ctx_a = str(example.get("ctx_a", "")).strip()
    ctx_b = str(example.get("ctx_b", "")).strip()
    activity = str(example.get("activity_label", "")).strip()
    question = " ".join(part for part in [activity + ":" if activity else "", ctx_a, ctx_b] if part).strip()
    label = example.get("label")
    gold = CHOICE_LETTERS[int(label)] if label is not None and str(label).strip() != "" else None
    return {
        "question": question,
        "choices": list(example["endings"]),
        "gold": gold,
        "subject": "hellaswag",
        "raw": example,
    }


def _extract_choice_texts(choices: Any) -> Tuple[List[str], Dict[str, str]]:
    normalized_choices: List[str] = []
    label_to_letter: Dict[str, str] = {}

    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
        for idx, (label, text) in enumerate(zip(labels, texts)):
            if idx >= len(CHOICE_LETTERS):
                break
            letter = CHOICE_LETTERS[idx]
            label_to_letter[str(label).strip().upper()] = letter
            normalized_choices.append(str(text))
        return normalized_choices, label_to_letter

    if isinstance(choices, list):
        for idx, choice in enumerate(choices):
            if idx >= len(CHOICE_LETTERS):
                break
            letter = CHOICE_LETTERS[idx]
            if isinstance(choice, dict):
                label = choice.get("label", choice.get("key", letter))
                text = choice.get("text", choice.get("content", choice.get("value", "")))
                label_to_letter[str(label).strip().upper()] = letter
                normalized_choices.append(str(text))
            else:
                normalized_choices.append(str(choice))
        return normalized_choices, label_to_letter

    return normalized_choices, label_to_letter


def normalize_arc_agi2_example(example: Dict[str, Any]) -> Dict[str, Any]:
    question = (
        example.get("question")
        or example.get("prompt")
        or example.get("instruction")
        or example.get("input")
        or example.get("problem")
    )
    choices = (
        example.get("choices")
        or example.get("options")
        or example.get("candidates")
        or example.get("answers")
    )

    normalized_choices, label_to_letter = _extract_choice_texts(choices)
    if not question or not normalized_choices:
        raise ValueError(
            "ARC-AGI2 rows in this script must already be converted to multiple-choice format. "
            "Expected fields like question/prompt plus choices/options/candidates."
        )

    answer_key = (
        example.get("answerKey")
        or example.get("answer")
        or example.get("label")
        or example.get("target")
        or example.get("correct")
        or example.get("correct_option")
        or example.get("correct_answer")
    )
    if isinstance(answer_key, int):
        gold = CHOICE_LETTERS[answer_key] if 0 <= answer_key < len(normalized_choices) else None
    else:
        answer_key = str(answer_key or "").strip().upper()
        gold = label_to_letter.get(answer_key, answer_key if answer_key in CHOICE_LETTERS else None)

    return {
        "question": str(question),
        "choices": normalized_choices,
        "gold": gold,
        "subject": example.get("task_id", example.get("id", "arc_agi2")),
        "raw": example,
    }


def _arc_agi2_split_candidates(split: str) -> List[str]:
    if split in {"validation", "eval", "evaluation", "test"}:
        return [split, "eval", "evaluation", "test", "validation"]
    return [split]


def load_objective_examples(
    dataset_name: str,
    split: str,
    limit: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split=split)
        items = [normalize_mmlu_example(ex) for ex in dataset]

    elif dataset_name == "arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        items = [normalize_arc_example(ex) for ex in dataset]

    elif dataset_name == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        items = [normalize_arc_example(ex) for ex in dataset]

    elif dataset_name == "commonsenseqa":
        dataset = load_dataset("tau/commonsense_qa", split=split)
        items = [normalize_commonsenseqa_example(ex) for ex in dataset]

    elif dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split=split)
        items = [normalize_hellaswag_example(ex) for ex in dataset]

    elif dataset_name in {"arc_agi2", "arc-agi2"}:
        # ARC-AGI-2 的 HuggingFace split 通常只有:
        #   - train
        #   - evaluation
        #
        # 因此把常见的 validation/test/dev/eval 写法统一映射到 evaluation。
        split_aliases = {
            "train": "train",
            "training": "train",

            "evaluation": "evaluation",
            "eval": "evaluation",
            "validation": "evaluation",
            "val": "evaluation",
            "dev": "evaluation",
            "test": "evaluation",
        }

        arc_split = split_aliases.get(split.lower(), split)

        dataset_ids = [
            "arc-agi2",
            "ARC-AGI-2",
            "eturok/ARC-AGI-2",
            "vincentkoc/arc-agi-2",
        ]

        last_error = None
        items = None

        for dataset_id in dataset_ids:
            try:
                dataset = load_dataset(dataset_id, split=arc_split)
                items = [normalize_arc_agi2_example(ex) for ex in dataset]
                break
            except Exception as exc:
                last_error = exc

        if items is None:
            raise ValueError(
                f"Failed to load arc-agi2 dataset for split={split} "
                f"(mapped to {arc_split}): {last_error}"
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    items = [
        item
        for item in items
        if item.get("gold") in CHOICE_LETTERS[: len(item["choices"])]
    ]

    rng = random.Random(seed)
    rng.shuffle(items)

    if limit is not None:
        items = items[:limit]

    return items


def default_expert_specs() -> List[Dict[str, Any]]:
    return [
        {"name": "qwen-3", "model": "qwen/qwen3-vl-235b-a22b-instruct", "sys": "You are a precise expert reasoner. Solve the problem carefully and finish with the final answer letter.", "cost": 1.5},
        {"name": "deepseek-r1", "model": "deepseek/deepseek-r1", "sys": "You are a precise expert reasoner. Solve the problem carefully and finish with the final answer letter.", "cost": 1.2},
        {"name": "llama-3.1", "model": "meta-llama/llama-3.1-405b-instruct", "sys": "You are a precise expert reasoner. Solve the problem carefully and finish with the final answer letter.", "cost": 3.1},
        {"name": "deepseek-3", "model": "deepseek/deepseek-v3.2", "sys": "You are a STEM expert. Be accurate and end with the answer letter.", "cost": 0.32},
        {"name": "claude-4.5", "model": "anthropic/claude-sonnet-4.5", "sys": "You are an analytical expert. Be accurate and end with the answer letter.", "cost": 15.0},
        {"name": "gpt-4o", "model": "openai/gpt-4o", "sys": "You are an analytical expert. Be accurate and end with the answer letter.", "cost": 10.0},
        {"name": "Mistral-8x22B", "model": "mistralai/mixtral-8x22b-instruct", "sys": "You are a careful expert. Be accurate and end with the answer letter.", "cost": 1.2},
        {"name": "mistral-small-creative", "model": "mistralai/mistral-small-creative", "sys": "You are a careful expert. Be accurate and end with the answer letter.", "cost": 0.3},
        {"name": "grok-4.1-fast", "model": "x-ai/grok-4.1-fast", "sys": "You are a careful expert. Be accurate and end with the answer letter.", "cost": 0.5},
        {"name": "qwen-2.5-72b", "model": "qwen/qwen-2.5-72b-instruct", "sys": "You are a careful expert. Be accurate and end with the answer letter.", "cost": 0.39},
        {"name": "gemma-2-27b-it", "model": "google/gemma-2-27b-it", "sys": "You are a careful expert. Be accurate and end with the answer letter.", "cost": 0.65},
        {"name": "qwen-2.5-32b-instruct", "model": "qwen/qwen-2.5-32b-instruct", "sys": "You are an accurate expert. End with the answer letter.", "cost": 0.1},
        {"name": "glm-4-32b", "model": "z-ai/glm-4-32b", "sys": "You are an accurate expert. End with the answer letter.", "cost": 0.1104},
        {"name": "llama-3.1-70b-instruct", "model": "meta-llama/llama-3.1-70b-instruct", "sys": "You are an accurate expert. End with the answer letter.", "cost": 0.2},
        {"name": "gpt-3.5-turbo", "model": "openai/gpt-3.5-turbo", "sys": "You are an accurate expert. End with the answer letter.", "cost": 1.5},
    ]


def build_router(args) -> Tuple[Router, AggregatorConfig]:
    expert_specs = default_expert_specs()
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "mmlu_meta_10d.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    d_s = len(meta["cap_names"]) + 1
    profiles = meta["models"]
    costs = [float(spec["cost"]) for spec in expert_specs]
    cost_min, cost_max = min(costs), max(costs)

    experts = []
    s_list = []
    for spec in expert_specs:
        profile = profiles.get(spec["name"])
        if profile is None:
            raise KeyError(f"Missing meta profile for expert {spec['name']}")
        cost_norm = 0.0 if cost_max <= cost_min else (float(spec["cost"]) - cost_min) / (cost_max - cost_min)
        s_vec = np.array(profile["cap_acc_10d"] + [cost_norm], dtype=np.float64)
        s_list.append(s_vec)
        experts.append(
            OpenRouterLLMExpert(
                meta=ExpertMeta(name=spec["name"], s=s_vec, cost=float(spec["cost"])),
                cfg=LLMExpertConfig(
                    model=spec["model"],
                    system_prompt=spec["sys"],
                    temperature=0.2,
                    max_tokens=args.max_tokens,
                ),
            )
        )

    enc = STTextEncoder(model_name=args.text_model, instruction_prefix=args.instruction_prefix)
    d_u = enc.d_h + args.d_c
    if args.warm_start == "pca":
        w_x, w_m = init_mats_pca_warmstart(d_x=args.d_x, d_u=d_u, s_list=s_list, seed=args.seed)
    else:
        w_x, w_m = init_mats_random(d_x=args.d_x, d_u=d_u, d_s=d_s, seed=args.seed)

    aggregator_cfg = AggregatorConfig(
        model=args.aggregator_model,
        system_prompt="You are the Wise Integrator. Resolve disagreement among expert solutions and output only the best final answer letter.",
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    router = Router(
        W_x=w_x,
        W_m=w_m,
        experts=experts,
        text_model=args.text_model,
        instruction_prefix=args.instruction_prefix,
        d_c=args.d_c,
        alpha=args.alpha,
        lam=0.1,
        use_query_embedding=args.use_query_embedding,
        use_meta_vectors=args.use_meta_vectors,
        use_hadamard=args.use_hadamard,
    )
    router.aggregator = LLMAggregator(aggregator_cfg)
    return router, aggregator_cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate CES on objective multiple-choice benchmarks.")
    parser.add_argument("--dataset", type=str, default="arc_challenge", choices=["mmlu", "arc_challenge", "arc_easy", "commonsenseqa", "hellaswag", "arc_agi2", "arc-agi2"])
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--policy", type=str, default="bandit", choices=["bandit", "random", "cheapest"])
    parser.add_argument("--lam_cost", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=3.5)
    parser.add_argument("--out_log", type=str, default="runs/objective_mcq/ces_arc_challenge.jsonl")
    parser.add_argument("--cache", type=str, default="runs/objective_mcq/cache_ces_arc_challenge.jsonl")
    parser.add_argument("--warm_start", type=str, default="pca", choices=["pca", "random"])
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--d_c", type=int, default=8)
    parser.add_argument("--d_x", type=int, default=32)
    parser.add_argument("--text_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--instruction_prefix", type=str, default="query: ")
    parser.add_argument("--aggregator_model", type=str, default="qwen/qwen2.5-vl-72b-instruct")
    parser.add_argument("--use_query_embedding", dest="use_query_embedding", action="store_true")
    parser.add_argument("--no_query_embedding", dest="use_query_embedding", action="store_false")
    parser.add_argument("--use_meta_vectors", dest="use_meta_vectors", action="store_true")
    parser.add_argument("--no_meta_vectors", dest="use_meta_vectors", action="store_false")
    parser.add_argument("--use_hadamard", dest="use_hadamard", action="store_true")
    parser.add_argument("--no_hadamard", dest="use_hadamard", action="store_false")
    parser.add_argument("--use_cost_penalty", dest="use_cost_penalty", action="store_true")
    parser.add_argument("--no_cost_penalty", dest="use_cost_penalty", action="store_false")
    parser.set_defaults(
        use_query_embedding=True,
        use_meta_vectors=True,
        use_hadamard=True,
        use_cost_penalty=True,
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    cache = DiskCache(args.cache)
    examples = load_objective_examples(args.dataset, args.split, args.limit, args.seed)
    router, aggregator_cfg = build_router(args)

    cheapest_k_idx = np.argsort(router.costs)[: args.k]
    rng_select = np.random.default_rng(args.seed)
    effective_lam_cost = args.lam_cost if args.use_cost_penalty else 0.0

    accuracy_hist: List[float] = []
    parsed_hist: List[float] = []
    api_costs: List[float] = []
    rewards: List[float] = []

    with open(args.out_log, "w", encoding="utf-8") as wf:
        for t, ex in enumerate(tqdm(examples, desc=f"CES {args.dataset}")):
            prompt = build_prompt(ex["question"], ex["choices"])
            _, phis = router._phis_for_prompt(prompt)

            if args.policy == "bandit":
                chosen_idx = router.agent.select_topk(
                    phis,
                    k=args.k,
                    costs=router.costs if args.use_cost_penalty else None,
                    lam_cost=effective_lam_cost,
                )
            elif args.policy == "random":
                chosen_idx = rng_select.choice(len(router.experts), size=args.k, replace=False)
            else:
                chosen_idx = cheapest_k_idx.copy()

            chosen_experts = [router.experts[i] for i in chosen_idx]
            chosen_names = [e.meta.name for e in chosen_experts]
            answers = []
            actual_cost_usd = 0.0

            for expert in chosen_experts:
                key = {"type": "expert", "model": expert.cfg.model, "sys": expert.cfg.system_prompt, "prompt": prompt}
                cached = cache.get(key)
                if cached is None:
                    out = expert.infer_with_usage(prompt)
                    cached = {"text": out["text"], "usage": out.get("usage", {}), "id": out.get("id")}
                    cache.put(key, cached)
                actual_cost_usd += usage_cost_usd(cached)
                answers.append(cached["text"] if isinstance(cached, dict) else str(cached))

            agg_key = {"type": "agg", "model": aggregator_cfg.model, "prompt": prompt, "names": chosen_names, "answers": answers}
            final = cache.get(agg_key)
            if final is None:
                final = router.aggregator.aggregate_with_usage(prompt, chosen_names, answers)
                cache.put(agg_key, final)
            actual_cost_usd += usage_cost_usd(final)

            final_text = final["text"] if isinstance(final, dict) else str(final)
            pred = parse_choice_letter(final_text)
            correct = float(pred == ex["gold"])
            parsed = float(pred is not None)
            router_cost = float(router.costs[chosen_idx].sum())
            reward = float(correct - effective_lam_cost * router_cost)

            if args.policy == "bandit":
                z_sum = np.sum([phis[i] for i in chosen_idx], axis=0)
                router.agent.update(z_sum, reward)

            accuracy_hist.append(correct)
            parsed_hist.append(parsed)
            api_costs.append(actual_cost_usd)
            rewards.append(reward)

            record = {
                "t": t,
                "dataset": args.dataset,
                "split": args.split,
                "policy": args.policy,
                "question": ex["question"],
                "choices": ex["choices"],
                "gold": ex["gold"],
                "pred": pred,
                "correct": correct,
                "parsed": parsed,
                "chosen_idx": chosen_idx.tolist() if hasattr(chosen_idx, "tolist") else list(chosen_idx),
                "chosen_names": chosen_names,
                "answers": answers,
                "final": final_text,
                "cost": actual_cost_usd,
                "router_cost": router_cost,
                "reward": reward,
                "avg_accuracy": float(np.mean(accuracy_hist)),
                "avg_parse_rate": float(np.mean(parsed_hist)),
                "avg_cost": float(np.mean(api_costs)),
                "avg_reward": float(np.mean(rewards)),
                "raw": ex["raw"],
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()
            time.sleep(args.sleep)

    total_api_cost = float(np.sum(api_costs))
    num_questions = len(accuracy_hist)
    print(f"[DONE] wrote log to: {args.out_log}")
    print(f"accuracy={float(np.mean(accuracy_hist)):.4f}")
    print(f"parse_rate={float(np.mean(parsed_hist)):.4f}")
    print(f"avg_reward={float(np.mean(rewards)):.4f}")
    print(f"total_cost={total_api_cost:.6f}")
    print(f"num_questions={num_questions}")
    print(f"cost_per_question={(total_api_cost / num_questions) if num_questions else 0.0:.6f}")


if __name__ == "__main__":
    main()
