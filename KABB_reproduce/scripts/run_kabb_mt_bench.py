# scripts/run_kabb_mt_bench.py
import sys
import os

# Add project root to sys.path for kabb package import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
import uuid
import argparse
import asyncio
import yaml
from typing import List, Dict, Any, Tuple

from kabb.llm_client import LLMClient
from kabb.utils import logic_based_domain_inference, estimate_task_difficulty
import scripts.run_kabb as rk
from scripts.run_kabb import (
    get_domain_all_expert_responses,
    get_integrated_content_new,
    integrate_expert_responses,
)

DEFAULT_BENCH_NAME = "mt_bench"
DEFAULT_MODEL_ID = "kabb_openrouter"
DEFAULT_QUESTION_FILE = "MoA/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
DEFAULT_ANSWER_DIR = "KABB/runs/mt_bench/model_answer"

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_questions_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read FastChat mt_bench question.jsonl: each line is a JSON object.
    Expected keys: question_id, category, turns (list[str]) ...
    """
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs

def build_user_input_with_history(turns: List[str], answers: List[str], current_turn_idx: int) -> str:
    """
    Convert a multi-turn conversation into a single user_input string
    that your KABB pipeline can consume.
    - turns: original user turns
    - answers: assistant answers for previous turns (len == current_turn_idx)
    - current_turn_idx: index of the user turn to answer now
    """
    assert current_turn_idx < len(turns)
    if current_turn_idx == 0:
        return turns[0]

    parts = ["You are in a multi-turn conversation. Use the prior context to answer the follow-up.\n"]
    for i in range(current_turn_idx):
        parts.append(f"[Turn {i+1} - User]\n{turns[i]}\n")
        parts.append(f"[Turn {i+1} - Assistant]\n{answers[i]}\n")
    parts.append(f"[Turn {current_turn_idx+1} - User]\n{turns[current_turn_idx]}\n")
    return "\n".join(parts)

def force_single_domain(domain_label):
    """
    logic_based_domain_inference returns top-k list; we default to single domain.
    """
    if isinstance(domain_label, list) and len(domain_label) > 0:
        return domain_label[0]
    return domain_label

async def run_one_round(config: dict, llm_client: LLMClient, user_input: str, top_k_domain: int = 1) -> Tuple[str, float, int, str]:
    """
    Run one 'round' of KABB (experts -> integrator) for a single (possibly history-packed) user_input.
    Returns: (final_answer, round_cost, round_calls, domain_label_used)
    """
    domain_settings = config.get("domain_inference_settings", {})
    experts_config = config.get("experts", {})
    integrator_config = config.get("integrator", {})
    system_prompt = config["system_prompts"]["default"]

    # domain inference
    domain_label, cleaned_input = logic_based_domain_inference(user_input, domain_settings, top_k=top_k_domain)
    domain_label = force_single_domain(domain_label)

    # fallback domain
    default_domain = list(experts_config.keys())[0] if experts_config else "stem"
    if domain_label not in experts_config:
        domain_label = default_domain

    # difficulty (kept for consistency with original pipeline; not used unless you later enable bandit)
    _ = estimate_task_difficulty(cleaned_input, domain_settings.get(domain_label, {}))

    # experts + integrator
    selected_experts = experts_config.get(domain_label, [])
    integrator_info = integrator_config.get("default")
    if not integrator_info:
        raise ValueError("Missing integrator.default in config.")
    if not selected_experts:
        raise ValueError(f"No experts defined for domain={domain_label}.")

    rk.reset_question_cost()

    # call all experts (this is how your current code behaves)
    all_responses = {
        domain_label: await get_domain_all_expert_responses(
            domain_label, selected_experts, system_prompt, llm_client, cleaned_input
        )
    }

    integrated_content = get_integrated_content_new(all_responses, selected_experts, use_contextual_expert=False)

    final_answer = await integrate_expert_responses(
        integrator_info, system_prompt, llm_client, cleaned_input, integrated_content
    )

    return final_answer.strip(), float(rk.TOTAL_COST), int(rk.TOTAL_CALLS), domain_label

async def run_one_question(config: dict, llm_client: LLMClient, q: dict, max_turns: int = 2, top_k_domain: int = 1) -> Tuple[List[str], float, int, List[str]]:
    """
    Run MT-Bench question with N turns (standard is 2).
    Returns:
      answers_per_turn: List[str]
      cost_question: float
      calls_question: int
      domains_used: List[str] (per turn)
    """
    turns = q.get("turns", [])
    if not isinstance(turns, list) or len(turns) == 0:
        return [], 0.0, 0, []

    turns = turns[:max_turns]

    answers: List[str] = []
    domains: List[str] = []
    cost_q = 0.0
    calls_q = 0

    for t in range(len(turns)):
        user_input = build_user_input_with_history(turns, answers, t)
        ans, c, n, domain_used = await run_one_round(config, llm_client, user_input, top_k_domain=top_k_domain)
        answers.append(ans)
        domains.append(domain_used)
        cost_q += c
        calls_q += n

    return answers, cost_q, calls_q, domains

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to your KABB config yaml")
    ap.add_argument("--bench-name", type=str, default=DEFAULT_BENCH_NAME)
    ap.add_argument("--question-file", type=str, default=DEFAULT_QUESTION_FILE)
    ap.add_argument("--answer-dir", type=str, default=DEFAULT_ANSWER_DIR)

    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="model_id / generator name used by FastChat loader")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=2)
    ap.add_argument("--top-k-domain", type=int, default=1, help="Set 1 to run single-domain (recommended).")

    args = ap.parse_args()

    cfg = load_config(args.config)

    llm_cfg = cfg.get("llm_api", {})
    provider = os.environ.get("LLM_PROVIDER") or llm_cfg.get("provider") or "openrouter"
    api_key = os.environ.get("OPENROUTER_API_KEY") or llm_cfg.get("api_key") or "<YOUR_KEY_HERE>"
    llm_client = LLMClient(api_key=api_key, provider=provider)

    questions = load_questions_jsonl(args.question_file)
    if args.end is None:
        args.end = len(questions)
    args.end = min(args.end, len(questions))

    os.makedirs(args.answer_dir, exist_ok=True)
    out_path = os.path.join(args.answer_dir, f"{args.model_id}.jsonl")

    total_cost = 0.0
    total_calls = 0
    per_q_costs: List[float] = []
    per_q_calls: List[int] = []

    # Append mode is useful for resume; default overwrite each run for clean evaluation
    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx in range(args.start, args.end):
            q = questions[idx]
            qid = q.get("question_id", idx)
            cat = q.get("category", "")

            try:
                answers, cost_q, calls_q, domains = await run_one_question(
                    cfg, llm_client, q, max_turns=args.max_turns, top_k_domain=args.top_k_domain
                )
            except Exception as e:
                # If a question fails, still write a placeholder to keep ids consistent
                answers = [f"[ERROR] {repr(e)}"]
                cost_q, calls_q = 0.0, 0
                domains = ["__error__"]

            record = {
                "question_id": qid,
                "answer_id": str(uuid.uuid4()),
                "model_id": args.model_id,
                "choices": [{"index": 0, "turns": answers}],
                "tstamp": time.time(),

                # Extra fields (FastChat ignores unknown keys)
                "category": cat,
                "cost": cost_q,
                "num_calls": calls_q,
                "domains": domains,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_cost += cost_q
            total_calls += calls_q
            per_q_costs.append(cost_q)
            per_q_calls.append(calls_q)

            print(f"[{idx}] qid={qid} cat={cat} turns={len(answers)} calls={calls_q} cost={cost_q:.4f} domains={domains}")

    num_q = len(per_q_costs)
    avg_cost_q = (sum(per_q_costs) / num_q) if num_q > 0 else 0.0
    avg_calls_q = (sum(per_q_calls) / num_q) if num_q > 0 else 0.0

    print("\n==== MT-Bench Run Summary ====\n")
    print(f"Answer file: {out_path}")
    print(f"Total questions: {num_q}")
    print(f"Total calls: {total_calls}")
    print(f"Total cost: {total_cost:.6f}")
    print(f"Average cost per question: {avg_cost_q:.6f}")
    print(f"Average calls per question: {avg_calls_q:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
