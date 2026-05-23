import argparse
import asyncio
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import load_dataset
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import scripts.run_kabb as rk
from kabb.llm_client import LLMClient
from kabb.utils import estimate_task_difficulty, logic_based_domain_inference
from scripts.run_kabb import get_domain_all_expert_responses, get_integrated_content_new, integrate_expert_responses


CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]
ANSWER_PATTERNS = [
    re.compile(r"^\s*([A-F])\s*$", re.I),
    re.compile(r"\b(?:ANSWER|FINAL ANSWER|FINAL|CHOICE)\s*[:\-]?\s*[\(\[]?\s*([A-F])\s*[\)\]]?\b", re.I),
    re.compile(r"\bTHE ANSWER IS\s*[\(\[]?\s*([A-F])\s*[\)\]]?\b", re.I),
]


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
    lines.append("Answer with a single letter only.")
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


def load_objective_examples(dataset_name: str, split: str, limit: Optional[int], seed: int) -> List[Dict[str, Any]]:
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
        last_error = None
        for dataset_id in ["arc-agi2", "ARC-AGI-2", "eturok/ARC-AGI-2", "vincentkoc/arc-agi-2"]:
            for split_name in _arc_agi2_split_candidates(split):
                try:
                    dataset = load_dataset(dataset_id, split=split_name)
                    items = [normalize_arc_agi2_example(ex) for ex in dataset]
                    break
                except Exception as exc:
                    last_error = exc
            else:
                continue
            break
        else:
            raise ValueError(f"Failed to load arc-agi2 dataset for split={split}: {last_error}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    items = [item for item in items if item.get("gold") in CHOICE_LETTERS[: len(item["choices"])]]
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit is not None:
        items = items[:limit]
    return items


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def force_single_domain(domain_label):
    if isinstance(domain_label, list) and domain_label:
        return domain_label[0]
    return domain_label


async def run_one_round(config: dict, llm_client: LLMClient, user_input: str, top_k_domain: int = 1) -> Tuple[str, str]:
    domain_settings = config.get("domain_inference_settings", {})
    experts_config = config.get("experts", {})
    integrator_config = config.get("integrator", {})
    system_prompt = config["system_prompts"]["default"]

    domain_label, cleaned_input = logic_based_domain_inference(user_input, domain_settings, top_k=top_k_domain)
    domain_label = force_single_domain(domain_label)
    default_domain = list(experts_config.keys())[0] if experts_config else "math"
    if domain_label not in experts_config:
        domain_label = default_domain

    _ = estimate_task_difficulty(cleaned_input, domain_settings.get(domain_label, {}))
    selected_experts = experts_config.get(domain_label, [])
    integrator_info = integrator_config.get("default")
    if not integrator_info:
        raise ValueError("Missing integrator.default in config.")
    if not selected_experts:
        raise ValueError(f"No experts defined for domain={domain_label}.")

    rk.reset_question_cost()
    all_responses = {
        domain_label: await get_domain_all_expert_responses(
            domain_label, selected_experts, system_prompt, llm_client, cleaned_input
        )
    }
    integrated_content = get_integrated_content_new(all_responses, selected_experts, use_contextual_expert=False)
    final_answer = await integrate_expert_responses(integrator_info, system_prompt, llm_client, cleaned_input, integrated_content)
    return final_answer.strip(), domain_label


async def main():
    parser = argparse.ArgumentParser(description="Evaluate KABB on objective multiple-choice benchmarks.")
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "config_template.yaml"))
    parser.add_argument("--dataset", type=str, default="arc_challenge", choices=["mmlu", "arc_challenge", "arc_easy", "commonsenseqa", "hellaswag", "arc_agi2", "arc-agi2"])
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k-domain", type=int, default=1)
    parser.add_argument("--out-log", type=str, default="runs/objective_mcq/kabb_arc_challenge.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    cfg = load_config(args.config)
    examples = load_objective_examples(args.dataset, args.split, args.limit, args.seed)

    llm_cfg = cfg.get("llm_api", {})
    provider = os.environ.get("LLM_PROVIDER") or llm_cfg.get("provider") or "openrouter"
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TOGETHER_API_KEY") or llm_cfg.get("api_key") or ""
    llm_client = LLMClient(api_key=api_key, provider=provider)

    correct_hist: List[float] = []
    parsed_hist: List[float] = []
    cost_hist: List[float] = []
    call_hist: List[int] = []

    with open(args.out_log, "w", encoding="utf-8") as wf:
        for idx, ex in enumerate(tqdm(examples, desc=f"KABB {args.dataset}")):
            prompt = build_prompt(ex["question"], ex["choices"])
            try:
                final_answer, domain_label = await run_one_round(cfg, llm_client, prompt, top_k_domain=args.top_k_domain)
                cost_q = float(rk.TOTAL_COST)
                calls_q = int(rk.TOTAL_CALLS)
            except Exception as exc:
                final_answer = f"[ERROR] {repr(exc)}"
                domain_label = "__error__"
                cost_q = 0.0
                calls_q = 0

            pred = parse_choice_letter(final_answer)
            correct = float(pred == ex["gold"])
            parsed = float(pred is not None)

            correct_hist.append(correct)
            parsed_hist.append(parsed)
            cost_hist.append(cost_q)
            call_hist.append(calls_q)

            record = {
                "t": idx,
                "dataset": args.dataset,
                "split": args.split,
                "question": ex["question"],
                "choices": ex["choices"],
                "gold": ex["gold"],
                "pred": pred,
                "correct": correct,
                "parsed": parsed,
                "domain": domain_label,
                "final": final_answer,
                "cost": cost_q,
                "num_calls": calls_q,
                "avg_accuracy": sum(correct_hist) / len(correct_hist),
                "avg_parse_rate": sum(parsed_hist) / len(parsed_hist),
                "avg_cost": sum(cost_hist) / len(cost_hist),
                "avg_num_calls": sum(call_hist) / len(call_hist),
                "raw": ex["raw"],
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()

    total_cost = sum(cost_hist)
    num_questions = len(correct_hist)
    print(f"[DONE] wrote log to: {args.out_log}")
    print(f"accuracy={(sum(correct_hist) / num_questions) if num_questions else 0.0:.4f}")
    print(f"parse_rate={(sum(parsed_hist) / num_questions) if num_questions else 0.0:.4f}")
    print(f"total_cost={total_cost:.6f}")
    print(f"num_questions={num_questions}")
    print(f"cost_per_question={(total_cost / num_questions) if num_questions else 0.0:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
