# scripts/run_kabb_alpacaeval.py
import sys
import os

# Add project root to sys.path for kabb package import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import os, json, argparse, asyncio
import yaml
import datasets
import scripts.run_kabb as rk
from kabb.llm_client import LLMClient
from kabb.utils import logic_based_domain_inference, estimate_task_difficulty
from scripts.run_kabb import get_domain_all_expert_responses, get_integrated_content_new, integrate_expert_responses

DEFAULT_GEN_NAME = "kabb_openrouter"
# ---- Cost tracking for AlpacaEval ----
TOTAL_COST = 0.0
TOTAL_CALLS = 0
PER_QUESTION_COSTS = []

def reset_question_cost():
    global TOTAL_COST, TOTAL_CALLS
    TOTAL_COST = 0.0
    TOTAL_CALLS = 0

def add_cost(cost):
    global TOTAL_COST, TOTAL_CALLS
    try:
        c = float(cost)
    except Exception:
        c = 0.0
    TOTAL_COST += c
    TOTAL_CALLS += 1


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def run_one(config: dict, llm_client: LLMClient, user_input: str) -> str:
    domain_settings = config.get("domain_inference_settings", {})
    knowledge_graph = config.get("knowledge_graph", {})
    experts_config = config.get("experts", {})
    integrator_config = config.get("integrator", {})
    system_prompt = config["system_prompts"]["default"]

    domain_label, cleaned_input = logic_based_domain_inference(user_input, domain_settings, top_k=1)

   
    DEFAULT_DOMAIN = list(experts_config.keys())[0] if experts_config else "stem"
    if isinstance(domain_label, list):
        missing = [d for d in domain_label if d not in experts_config]
        if missing:
            domain_label = DEFAULT_DOMAIN
    else:
        if domain_label not in experts_config:
            domain_label = DEFAULT_DOMAIN

    if isinstance(domain_label, list):
        difficulty = estimate_task_difficulty(cleaned_input, domain_settings.get(domain_label[0], {}))
    else:
        difficulty = estimate_task_difficulty(cleaned_input, domain_settings.get(domain_label, {}))

   
    integrator_info = integrator_config.get("default")
    if isinstance(domain_label, list):
        selected_experts = []
        for label in domain_label:
            selected_experts.extend(experts_config.get(label, []))
    else:
        selected_experts = experts_config.get(domain_label, [])

    
    if isinstance(domain_label, list):
        all_responses = {}
        for label in domain_label:
            exps = [e for e in selected_experts if label in e.get("domain_expertise", [])]
            if exps:
                all_responses[label] = await get_domain_all_expert_responses(
                    label, exps, system_prompt, llm_client, cleaned_input
                )
    else:
        all_responses = {
            domain_label: await get_domain_all_expert_responses(
                domain_label, selected_experts, system_prompt, llm_client, cleaned_input
            )
        }

    integrated_content = get_integrated_content_new(all_responses, selected_experts, use_contextual_expert=False)
    final_answer = await integrate_expert_responses(integrator_info, system_prompt, llm_client, cleaned_input, integrated_content)
    return final_answer.strip()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=805)
    ap.add_argument("--generator", type=str, default=DEFAULT_GEN_NAME)
    args = ap.parse_args()

    cfg = load_config(args.config)

    llm_cfg = cfg.get("llm_api", {})
    provider = os.environ.get("LLM_PROVIDER") or llm_cfg.get("provider") or "openrouter"
    api_key = os.environ.get("OPENROUTER_API_KEY") or llm_cfg.get("api_key") or "<YOUR_KEY_HERE>"
    llm_client = LLMClient(api_key=api_key, provider=provider)

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]  # 805条:contentReference[oaicite:2]{index=2}

    outputs = []

    for i in range(args.start, args.end):
        instruction = eval_set[i]["instruction"]

        rk.reset_question_cost()
        answer = await run_one(cfg, llm_client, instruction)

        PER_QUESTION_COSTS.append(rk.TOTAL_COST)

        outputs.append({
            "instruction": instruction,
            "output": answer,
            "generator": args.generator,
            "cost": rk.TOTAL_COST,
            "num_calls": rk.TOTAL_CALLS
        })

        print(f"[{i}] done | cost={rk.TOTAL_COST:.4f} | calls={rk.TOTAL_CALLS}")


    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    num_q = len(PER_QUESTION_COSTS)
    total_cost_all = sum(PER_QUESTION_COSTS)
    avg_cost = total_cost_all / num_q if num_q > 0 else 0.0

    print("\n==== AlpacaEval Cost Summary ====\n")
    print(f"Total questions: {num_q}")
    print(f"Total cost: {total_cost_all:.6f}")
    print(f"num_instructions: {num_q}")
    print(f"cost_per_instruction: {avg_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
