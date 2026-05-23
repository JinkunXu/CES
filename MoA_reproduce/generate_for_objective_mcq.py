import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from utils import generate_openai_with_usage, generate_openrouter_with_usage, generate_together_with_usage


CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]
ANSWER_PATTERNS = [
    re.compile(r"^\s*([A-F])\s*$", re.I),
    re.compile(r"\b(?:ANSWER|FINAL ANSWER|FINAL|CHOICE)\s*[:\-]?\s*[\(\[]?\s*([A-F])\s*[\)\]]?\b", re.I),
    re.compile(r"\bTHE ANSWER IS\s*[\(\[]?\s*([A-F])\s*[\)\]]?\b", re.I),
]

REFERENCE_SYSTEM_PROMPT = """You are one of several expert models.
Solve the multiple-choice question carefully.
End with the single best answer letter."""

AGGREGATOR_SYSTEM_PROMPT = """You are an aggregator model in a Mixture-of-Agents system.
You will be provided with responses from other models.
Synthesize them and output only the single best final answer letter."""


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


def get_generate_fn(provider: str):
    if provider == "together":
        return generate_together_with_usage
    if provider == "openai":
        return generate_openai_with_usage
    if provider == "openrouter":
        return generate_openrouter_with_usage
    raise ValueError(f"Unknown provider: {provider}")


def format_references(refs: List[str]) -> str:
    chunks = []
    for idx, ref in enumerate(refs, start=1):
        ref = (ref or "").strip()
        if ref:
            chunks.append(f"[Reference {idx}]\n{ref}")
    return "\n\n".join(chunks)


def run_chat(generate_fn, model: str, user_prompt: str, references: Optional[List[str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
    refs_text = format_references(references or [])
    system_content = AGGREGATOR_SYSTEM_PROMPT if refs_text else REFERENCE_SYSTEM_PROMPT
    if refs_text:
        system_content = system_content + "\n\n" + refs_text
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]
    return generate_fn(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)


def evaluate_one(example: Dict[str, Any], args) -> Dict[str, Any]:
    generate_fn = get_generate_fn(args.provider)
    prompt = build_prompt(example["question"], example["choices"])

    total_cost = 0.0
    total_calls = 0
    reference_cost = 0.0
    aggregator_cost = 0.0

    references: List[str] = []
    previous_round_refs: List[str] = []
    for _ in range(args.rounds):
        current_refs: List[str] = []
        for ref_model in args.reference_models:
            out = run_chat(
                generate_fn=generate_fn,
                model=ref_model,
                user_prompt=prompt,
                references=previous_round_refs,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            text = (out.get("text") or "").strip()
            if text:
                current_refs.append(text)
            call_cost = float(out.get("usage", {}).get("cost", 0.0) or 0.0)
            total_cost += call_cost
            reference_cost += call_cost
            total_calls += 1
        previous_round_refs = current_refs
        references = current_refs

    final = run_chat(
        generate_fn=generate_fn,
        model=args.model,
        user_prompt=prompt,
        references=references,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    final_text = (final.get("text") or "").strip()
    final_cost = float(final.get("usage", {}).get("cost", 0.0) or 0.0)
    total_cost += final_cost
    aggregator_cost += final_cost
    total_calls += 1

    pred = parse_choice_letter(final_text)
    correct = float(pred == example["gold"])
    parsed = float(pred is not None)
    return {
        "question": example["question"],
        "choices": example["choices"],
        "gold": example["gold"],
        "pred": pred,
        "correct": correct,
        "parsed": parsed,
        "final": final_text,
        "references": references,
        "cost": total_cost,
        "num_calls": total_calls,
        "reference_cost": reference_cost,
        "aggregator_cost": aggregator_cost,
        "raw": example["raw"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoA on objective multiple-choice benchmarks.")
    parser.add_argument("--dataset", type=str, default="arc_challenge", choices=["mmlu", "arc_challenge", "arc_easy", "commonsenseqa", "hellaswag", "arc_agi2", "arc-agi2"])
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", type=str, default="openrouter", choices=["together", "openai", "openrouter"])
    parser.add_argument("--model", type=str, required=True, help="Aggregator model id.")
    parser.add_argument("--reference-models", type=str, required=True, help="Comma-separated list of MoA reference models.")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out-log", type=str, default="runs/objective_mcq/moa_arc_challenge.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    args.reference_models = [model.strip() for model in args.reference_models.split(",") if model.strip()]

    examples = load_objective_examples(args.dataset, args.split, args.limit, args.seed)

    correct_hist: List[float] = []
    parsed_hist: List[float] = []
    cost_hist: List[float] = []
    call_hist: List[int] = []

    with open(args.out_log, "w", encoding="utf-8") as wf:
        for idx, ex in enumerate(tqdm(examples, desc=f"MoA {args.dataset}")):
            record = evaluate_one(ex, args)
            correct_hist.append(record["correct"])
            parsed_hist.append(record["parsed"])
            cost_hist.append(record["cost"])
            call_hist.append(record["num_calls"])
            record["t"] = idx
            record["dataset"] = args.dataset
            record["split"] = args.split
            record["avg_accuracy"] = sum(correct_hist) / len(correct_hist)
            record["avg_parse_rate"] = sum(parsed_hist) / len(parsed_hist)
            record["avg_cost"] = sum(cost_hist) / len(cost_hist)
            record["avg_num_calls"] = sum(call_hist) / len(call_hist)
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
    main()
