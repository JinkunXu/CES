"""
moa/generate_for_mt_bench.py

Generate MT-Bench answers using MoA (reference models + aggregator model).
Supports providers:
- together   (via utils.generate_together)
- openai     (via utils.generate_openai)
- openrouter (OpenAI SDK with base_url=https://openrouter.ai/api/v1)

Example (smoke test 2 questions):
python moa/generate_for_mt_bench.py \
  --bench-name mt_bench \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --reference-models "qwen/qwen-2.5-72b-instruct,meta-llama/llama-3.1-70b-instruct,cohere/command-r-plus-08-2024" \
  --rounds 1 \
  --provider openrouter \
  --parallel 2 \
  --max-tokens 1024 \
  --question-begin 0 \
  --question-end 2 \
  --answer-file outputs/mt_bench/model_answer/moa_openrouter_test.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import time
from typing import List, Optional

import shortuuid
import tqdm
from loguru import logger

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.llm_judge.gen_model_answer import reorg_answer_file

# Keep Together/OpenAI compatibility if you still want it
from utils import generate_together, generate_openai, DEBUG  # noqa: F401

# OpenRouter (OpenAI-compatible)
from openai import OpenAI


# -----------------------------
# MoA prompt wrappers
# -----------------------------
REFERENCE_SYSTEM_PROMPT = """You are one of several expert models.
You may be provided with prior-round references from other models.
Write the best possible answer to the user. Be direct and helpful."""

AGGREGATOR_SYSTEM_PROMPT = """You are an aggregator model in a Mixture-of-Agents system.
You will be provided with a set of responses from other models ("references").
Synthesize them into a single high-quality answer for the user.
Rules:
- Do NOT mention other models or "references".
- Resolve disagreements.
- Provide a single coherent answer.
References from models:"""


def _format_references(refs: List[str]) -> str:
    chunks = []
    for i, r in enumerate(refs or [], start=1):
        r = (r or "").strip()
        if not r:
            continue
        chunks.append(f"[Reference {i}]\n{r}")
    return "\n\n".join(chunks)


def generate_with_references_local(
    *,
    model: str,
    messages: List[dict],
    references: Optional[List[str]],
    temperature: float,
    max_tokens: int,
    generate_fn,
) -> str:
    """
    Wraps references into a system message, then calls generate_fn(messages, model, temperature, max_tokens).
    """
    references = references or []
    refs_text = _format_references(references)

    if refs_text:
        system_content = AGGREGATOR_SYSTEM_PROMPT + "\n\n" + refs_text
    else:
        system_content = REFERENCE_SYSTEM_PROMPT

    wrapped_messages = [{"role": "system", "content": system_content}] + messages
    out = generate_fn(
        messages=wrapped_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (out or "").strip()


# -----------------------------
# Provider: OpenRouter
# -----------------------------
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _openrouter_headers():
    return {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "moa-mtbench"),
    }


def generate_openrouter(*, messages, model, temperature=0.7, max_tokens=1024) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=_openrouter_headers(),
        stream=False,
    )
    return (resp.choices[0].message.content or "")


def _pick_temperature(question: dict, force_temperature: Optional[float]) -> float:
    # Keep same logic as FastChat MT-Bench scripts
    if force_temperature is not None:
        return force_temperature
    if "required_temperature" in question:
        return question["required_temperature"]
    if question.get("category") in temperature_config:
        return temperature_config[question["category"]]
    return 0.7


def get_answer(
    question: dict,
    model: str,
    reference_models: List[str],
    num_choices: int,
    max_tokens: int,
    answer_file: str,
    rounds: int,
    provider: str,
    force_temperature: Optional[float],
):
    temperature = _pick_temperature(question, force_temperature)

    if provider == "together":
        generate_fn = generate_together
    elif provider == "openai":
        generate_fn = generate_openai
    elif provider == "openrouter":
        generate_fn = generate_openrouter
    else:
        raise ValueError(f"Unknown provider: {provider}")

    choices = []

    for i in range(num_choices):
        turns = []
        messages = []

        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({"role": "user", "content": qs})

            references: List[str] = []

            # Collect reference responses (MoA)
            if reference_models:
                prev_references: List[str] = []

                for i_round in range(rounds):
                    if "DEBUG" in globals() and DEBUG:
                        logger.info(
                            f"Q{question['question_id']} choice={i} turn={j} "
                            f"Round {i_round+1}/{rounds}: collecting reference responses"
                        )

                    cur_refs: List[str] = []
                    for reference_model in reference_models:
                        try:
                            ref = generate_with_references_local(
                                model=reference_model,
                                messages=messages,
                                references=prev_references,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                generate_fn=generate_fn,
                            )
                            if ref:
                                cur_refs.append(ref)
                        except Exception as e:
                            logger.warning(
                                f"Reference model failed: {reference_model} "
                                f"(q_id={question['question_id']}, turn={j}) err={e}"
                            )

                    prev_references = cur_refs
                    references = cur_refs  # last round refs used for aggregation

            # Aggregate (or baseline if no reference_models)
            try:
                output = generate_with_references_local(
                    model=model,
                    messages=messages,
                    references=references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    generate_fn=generate_fn,
                )
            except Exception as e:
                logger.error(
                    f"Aggregator model failed: {model} "
                    f"(q_id={question['question_id']}, turn={j}) err={e}"
                )
                output = ""

            messages.append({"role": "assistant", "content": output})
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, required=True, help="Aggregator model id.")
    parser.add_argument(
        "--reference-models",
        type=str,
        default=None,
        help="Comma-separated list of reference models for MoA.",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--provider",
        type=str,
        default="together",
        choices=["together", "openai", "openrouter"],
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )

    args = parser.parse_args()

    question_file = f"FastChat/fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    answer_file = args.answer_file or f"outputs/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")

    reference_models = []
    if args.reference_models:
        reference_models = [m.strip() for m in args.reference_models.split(",") if m.strip()]

    # Basic guardrails for OpenRouter
    if args.provider == "openrouter" and "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            futures.append(
                executor.submit(
                    get_answer,
                    question,
                    args.model,
                    reference_models,
                    args.num_choices,
                    args.max_tokens,
                    answer_file,
                    args.rounds,
                    args.provider,
                    args.force_temperature,
                )
            )

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()
