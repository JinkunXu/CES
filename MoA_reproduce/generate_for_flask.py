import json
import datasets
from fire import Fire
from functools import partial
from typing import List
from loguru import logger

from utils import (
    generate_together_with_usage,
    generate_openai_with_usage,
    generate_openrouter_with_usage,
    generate_with_references_with_usage,
    DEBUG,
)

# 仅新增：引入 openrouter 的 generate 函数（要求你已在 utils.py 里实现）
# from utils import generate_openrouter


def process_fn(
    item,
    model,
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
    provider="together",
):
    total_cost = 0.0
    total_calls = 0
    reference_cost = 0.0
    aggregator_cost = 0.0

    if provider == "together":
        generate_fn = generate_together_with_usage
    elif provider == "openai":
        generate_fn = generate_openai_with_usage
    elif provider == "openrouter":
        generate_fn = generate_openrouter_with_usage
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use together/openai/openrouter.")

    messages = [{"role": "user", "content": item["text"]}]

    references = item.get("references", [])

    if len(references) == 0 and len(reference_models) > 0:

        prev_references = []

        for i_round in range(rounds):

            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{rounds} to collecting reference responses."
                )

            references = []

            for reference_model in reference_models:
                payload = generate_with_references_with_usage(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    generate_fn=generate_fn,
                )
                reference = payload.get("text")

                if reference is not None:

                    references.append(reference)
                    call_cost = float(payload.get("usage", {}).get("cost", 0.0) or 0.0)
                    total_cost += call_cost
                    reference_cost += call_cost
                    total_calls += 1

            if i_round < rounds - 1:

                prev_references = references

                references = []

    payload = generate_with_references_with_usage(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
        generate_fn=generate_fn,
    )
    output = payload.get("text")
    call_cost = float(payload.get("usage", {}).get("cost", 0.0) or 0.0)
    total_cost += call_cost
    aggregator_cost += call_cost
    total_calls += 1

    return {
        "text": output,
        "cost": total_cost,
        "num_calls": total_calls,
        "reference_cost": reference_cost,
        "aggregator_cost": aggregator_cost,
    }


def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 16,
    provider: str = "together",
):

    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    eval_set = []
    with open("FLASK/evaluation_set/flask_evaluation.jsonl") as f:
        for line in f:
            if line.strip() == "":
                continue
            item = json.loads(line)
            eval_set.append({"question_id": item["idx"], "text": item["instruction"]})

    eval_set = datasets.Dataset.from_list(eval_set)

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(
                    reference_responses
                ):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(
                            reference_response["output"]
                        )

        eval_set = eval_set.add_column(f"references", references)

    elif len(reference_models):

        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=model,
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
            provider=provider,
        ),
        batched=False,
        num_proc=num_proc,
    )

    logger.info(f"Saving outputs to {output_path}.")

    eval_set.to_json(output_path)

    rows = list(eval_set)
    total_cost = sum(float(row.get("cost", 0.0) or 0.0) for row in rows)
    total_calls = sum(int(row.get("num_calls", 0) or 0) for row in rows)
    num_q = len(rows)
    avg_cost = total_cost / num_q if num_q else 0.0
    logger.info(f"Total cost: {total_cost:.6f}")
    logger.info(f"Average cost per question: {avg_cost:.6f}")
    logger.info(f"Total model calls: {total_calls}")


if __name__ == "__main__":

    Fire(main)
