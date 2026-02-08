# This script generates reference outputs and other single model outputs for the AlpacaEval dataset.
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.provider.openrouter_client import OpenRouterClient

MODEL_NAME = "deepseek/deepseek-r1"     
OUTPUT_PATH = "runs/alpacaeval/deepseek-r1.json"
TEMPERATURE = 0.0
SLEEP_BETWEEN_CALLS = 0.5

def main():
    # 1) load dataset
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
    eval_set = ds["eval"]
    instructions = [ex["instruction"] for ex in eval_set]

    # 2) init your client
    client = OpenRouterClient()

    # 3) generate
    reference_outputs = []
    total_cost = 0.0
    total_tokens = 0

    for inst in tqdm(instructions):
        resp = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": inst}],
            temperature=TEMPERATURE,
            retries=5,
        )

        if not resp.get("ok"):
            raise RuntimeError(
                f"OpenRouter call failed: {resp.get('error_type')} {resp.get('error_msg')}\n"
                f"status={resp.get('status_code')} request_id={resp.get('request_id')}\n"
                f"body={resp.get('response_body')}"
            )

        text = (resp.get("text") or "").strip()
        usage = resp.get("usage") or {}
        total_cost += float(usage.get("cost", 0.0) or 0.0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)

        reference_outputs.append({
            "instruction": inst,
            "output": text,
            "generator": MODEL_NAME
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    # 4) save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(reference_outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_PATH}")
    print(f"Total tokens: {total_tokens}, total cost: {total_cost:.6f}")

if __name__ == "__main__":
    main()
