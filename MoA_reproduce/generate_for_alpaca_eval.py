import os
import json
import datasets
from fire import Fire
from functools import partial
from loguru import logger

from openai import OpenAI

# -----------------------------
# OpenRouter client init
# -----------------------------
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


EXTRA_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
    "X-Title": os.getenv("OPENROUTER_X_TITLE", "moa-alpacaeval"),
}

# -----------------------------
# MoA prompts (system wrappers)
# -----------------------------
REFERENCE_SYSTEM_PROMPT = """You are one of several expert models. 
You may be provided with prior-round references from other models. 
Write the best possible answer to the user. Be direct and helpful."""

AGGREGATOR_SYSTEM_PROMPT = """You are an aggregator model in a Mixture-of-Agents system.
You will be provided with a set of responses from other models ("references").
Synthesize them into a single high-quality answer for the user.
- Do NOT mention that you saw other model outputs.
- Resolve disagreements.
- Keep the final answer coherent and complete.
References from models:"""

def _format_references(references):
    # references: List[str]
    chunks = []
    for i, r in enumerate(references, start=1):
        r = (r or "").strip()
        if not r:
            continue
        chunks.append(f"[Reference {i}]\n{r}")
    return "\n\n".join(chunks)

def openrouter_chat(
    model: str,
    user_messages,
    references=None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict:
    """
    user_messages: [{"role": "...", "content": "..."}]
    references: List[str] | None
    """
    references = references or []
    refs_text = _format_references(references)


    if refs_text:
        system_content = AGGREGATOR_SYSTEM_PROMPT + "\n\n" + refs_text
    else:
        system_content = REFERENCE_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_content}] + user_messages

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=EXTRA_HEADERS,
    )
    usage_obj = getattr(resp, "usage", None)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
    if usage_obj is not None:
        raw = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(getattr(usage_obj, "__dict__", {}) or {})
        usage = {
            "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(raw.get("total_tokens", 0) or 0),
            "cost": float(raw.get("cost", 0.0) or 0.0),
        }
    return {"text": (resp.choices[0].message.content or "").strip(), "usage": usage}


def process_fn(
    item,
    model,
    reference_models=None,
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
):
    total_cost = 0.0
    total_calls = 0
    reference_cost = 0.0
    aggregator_cost = 0.0
    if reference_models is None:
        reference_models = []

    messages = [{"role": "user", "content": item["instruction"]}]

   
    references = item.get("references", [])

    if (not references) and reference_models:
        prev_references = []

        for i_round in range(rounds):
            logger.info(f"Round {i_round+1}/{rounds}: generating reference responses")
            cur_refs = []

            for reference_model in reference_models:
                try:
                    ref = openrouter_chat(
                        model=reference_model,
                        user_messages=messages,
                        references=prev_references,   # 上一轮 refs（让 agents “看见”）
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    ref_text = ref.get("text")
                    if ref_text:
                        cur_refs.append(ref_text)
                    call_cost = float(ref.get("usage", {}).get("cost", 0.0) or 0.0)
                    total_cost += call_cost
                    reference_cost += call_cost
                    total_calls += 1
                except Exception as e:
                    logger.warning(f"Reference model failed: {reference_model} err={e}")

            prev_references = cur_refs
            references = cur_refs  
    try:
        output = openrouter_chat(
            model=model,
            user_messages=messages,
            references=references,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output_text = output.get("text") or ""
        call_cost = float(output.get("usage", {}).get("cost", 0.0) or 0.0)
        total_cost += call_cost
        aggregator_cost += call_cost
        total_calls += 1
    except Exception as e:
        logger.error(f"Aggregator model failed: {model} err={e}")
        output_text = ""

   
    return {
        "output": output_text,
        "generator": model + "-openrouter",
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
    num_proc: int = 4,  
    max_num_examples: int = None, 
):
    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = [m.strip() for m in reference_models.split(",") if m.strip()]

    eval_set = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
    )["eval"]

    eval_set = eval_set.remove_columns(["output", "generator"])

    if max_num_examples is not None:
        eval_set = eval_set.select(range(min(max_num_examples, len(eval_set))))

    if len(reference_paths):
        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(reference_responses):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(reference_response["output"])

        eval_set = eval_set.add_column("references", references)

    elif len(reference_models):
        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly via OpenRouter."
        )

    logger.info("Start generating outputs...")

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=model,
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
        ),
        batched=False,
        num_proc=num_proc,
    )

    logger.info(f"Saving outputs to {output_path}.")

    try:
        eval_set = eval_set.remove_columns("references")
    except Exception:
        pass

    with open(output_path, "w") as f:
        json.dump(list(eval_set), f, indent=2, ensure_ascii=False)

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
