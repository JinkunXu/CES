# This script runs the main loop for evaluating a Router-based system on the AlpacaEval 2.0 dataset.

import os
import json
import time
import hashlib
import argparse
import random
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Make repo importable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.orchestration.router import Router
from src.experts.base import ExpertMeta
from src.experts.llm_expert import OpenRouterLLMExpert, LLMExpertConfig
from src.aggregators.llm_aggregator import LLMAggregator, AggregatorConfig
from src.provider.openrouter_client import OpenRouterClient
from scripts.utils.meta_profiles import make_s_vec_ability10, ABILITY_KEYS_10
from src.data.dataset_loader import iter_dataset
from src.embeddings.encoders import STTextEncoder

# -------------------- Disk Cache --------------------

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
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, obj: Any) -> Optional[Any]:
        return self.db.get(self._key(obj))

    def put(self, obj: Any, val: Any) -> Any:
        k = self._key(obj)
        if k in self.db:
            return self.db[k]
        self.db[k] = val
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "val": val}, ensure_ascii=False) + "\n")
        return val


def usage_cost_usd(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return 0.0
    return float(usage.get("cost", 0.0) or 0.0)


def init_mats_pca_warmstart(
    d_x: int,
    d_u: int,
    S_list: List[np.ndarray],   # list of s_i, shape each (d_s,)
    seed: int = 0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (W_x, W_m) with:
      - W_x: orthogonal rows + scaled (variance-preserving)
      - W_m: PCA directions of standardized s_i (warm-start, interpretable)

    Shapes:
      W_x: (d_x, d_u)
      W_m: (d_x, d_s)
    """
    rng = np.random.default_rng(seed)

    # ---- W_x: orthogonal init (rows) ----
    # build square for QR, then take first d_x rows
    A = rng.normal(size=(d_u, d_u))
    Q, _ = np.linalg.qr(A)
    W_x = Q[:d_x, :].astype(np.float64)
    W_x *= (1.0 / np.sqrt(max(d_u, 1)))

    # ---- W_m: PCA on standardized S ----
    S = np.stack([np.asarray(s, dtype=np.float64) for s in S_list], axis=0)  # (M, d_s)
    d_s = S.shape[1]

    s_mean = S.mean(axis=0)
    s_std = np.maximum(S.std(axis=0), eps)
    Z = (S - s_mean) / s_std  # (M, d_s)

    # PCA via SVD: Vt rows are principal axes in R^{d_s}
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(d_x, d_s)

    W_m = np.zeros((d_x, d_s), dtype=np.float64)
    W_m[:k, :] = Vt[:k, :] * (1.0 / np.sqrt(max(d_s, 1)))

    # remaining rows small noise to avoid degeneracy if d_x > d_s
    if d_x > k:
        W_m[k:, :] = rng.normal(size=(d_x - k, d_s)) * (1e-3 / np.sqrt(max(d_s, 1)))

    return W_x, W_m


def init_mats_random(
    d_x: int,
    d_u: int,
    d_s: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d_u, d_u))
    Q, _ = np.linalg.qr(A)
    W_x = Q[:d_x, :].astype(np.float64)
    W_x *= (1.0 / np.sqrt(max(d_u, 1)))
    W_m = rng.normal(size=(d_x, d_s)).astype(np.float64)
    W_m *= (1.0 / np.sqrt(max(d_s, 1)))
    return W_x, W_m



def debug_phi(router, phis, names, topn=10):

    mus = np.array([float(phi @ router.agent.w_hat) for phi in phis])
    stds = np.array([float(np.sqrt(phi @ (router.agent.V_inv @ phi))) for phi in phis])

    idx = np.argsort(mus + router.agent.alpha * stds)[::-1][:topn]
    print("Top by UCB:")
    for i in idx:
        print(f"  {names[i]:25s} mu={mus[i]:.4f} std={stds[i]:.4f} ucb={(mus[i]+router.agent.alpha*stds[i]):.4f}")

    P = np.stack(phis)  # [M, p]
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    sim = Pn @ Pn.T
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    print(f"phi cosine sim: mean={off.mean():.4f} p90={np.quantile(off,0.9):.4f} max={off.max():.4f}")

import json, random, re
from typing import Optional

def judge_score_llm(
    judge_client,
    judge_model: str,
    prompt: str,
    answer: str,
    cache,                 # your DiskCache
    reference: Optional[str] = None,
    score_min: float = 0.0,
    score_max: float = 10.0,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> float:
    """
    Returns a scalar score in [0,1] by default (normalized from [score_min, score_max]).

    If reference is provided, judge compares answer to reference implicitly.
    If reference is None, judge scores answer standalone vs instruction.

    Cached by (prompt, answer, reference, judge_model).
    """
    key = {
        "type": "judge_score",
        "judge_model": judge_model,
        "prompt": prompt,
        "answer": answer,
        "reference": reference,
        "range": [score_min, score_max],
    }
    cached = cache.get(key)
    if cached is not None:
        return float(cached)

    sys_msg = "You are a strict evaluator. Return valid JSON only."

    if reference is None:
        user_msg = (
            "You will score an assistant answer for a user instruction.\n"
            "Criteria: helpfulness, correctness, completeness, clarity, and safety.\n"
            f"Return JSON ONLY with fields:\n"
            f'{{"score": number, "reason": string}}\n'
            f'Score must be between {score_min} and {score_max}.\n\n'
            f"INSTRUCTION:\n{prompt}\n\n"
            f"ANSWER:\n{answer}\n"
        )
    else:
        user_msg = (
            "You will score Output A for a user instruction, with Output B as a reference.\n"
            "Score Output A considering helpfulness, correctness, completeness, clarity, and safety.\n"
            f"Return JSON ONLY with fields:\n"
            f'{{"score": number, "reason": string}}\n'
            f'Score must be between {score_min} and {score_max}.\n\n'
            f"INSTRUCTION:\n{prompt}\n\n"
            f"OUTPUT A:\n{answer}\n\n"
            f"OUTPUT B (reference):\n{reference}\n"
        )

    out = judge_client.chat(
        model=judge_model,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (out.get("text") or "").strip()

    # robust JSON extraction
    score = None
    try:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            obj = json.loads(text[l:r+1])
            score = obj.get("score", None)
    except Exception:
        score = None

    if score is None:
        # fallback: extract first number
        m = re.search(r"(-?\d+(\.\d+)?)", text)
        score = float(m.group(1)) if m else score_min

    # clamp and normalize
    score = float(score)
    score = max(score_min, min(score_max, score))
    norm = (score - score_min) / max(score_max - score_min, 1e-12)

    cache.put(key, norm)
    return norm



# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_log", type=str, default="runs/alpacaeval2/CES.jsonl")
    ap.add_argument("--cache", type=str, default="runs/alpacaeval2/cache_CES.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lam_cost", type=float, default=0.01)
    # ap.add_argument("--selection_mode", type=str, default="set_ucb_greedy", choices=["sum_ucb", "set_ucb_greedy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", type=str, default="bandit",
                choices=["bandit", "random", "cheapest"],
                help="Expert selection policy for ablation.")
    ap.add_argument("--warm_start", type=str, default="pca", choices=["pca", "random"])
    ap.add_argument("--use_query_embedding", dest="use_query_embedding", action="store_true")
    ap.add_argument("--no_query_embedding", dest="use_query_embedding", action="store_false")
    ap.add_argument("--use_meta_vectors", dest="use_meta_vectors", action="store_true")
    ap.add_argument("--no_meta_vectors", dest="use_meta_vectors", action="store_false")
    ap.add_argument("--use_hadamard", dest="use_hadamard", action="store_true")
    ap.add_argument("--no_hadamard", dest="use_hadamard", action="store_false")
    ap.add_argument("--use_cost_penalty", dest="use_cost_penalty", action="store_true")
    ap.add_argument("--no_cost_penalty", dest="use_cost_penalty", action="store_false")


    # feature/embedding dims
    ap.add_argument("--d_s", type=int, default=24)
    ap.add_argument("--d_c", type=int, default=8)
    ap.add_argument("--d_x", type=int, default=32)

    #dataset
    ap.add_argument("--dataset", type=str, default="alpacaeval2",
                choices=["alpacaeval2", "mtbench", "flask_hard", "generic"])
    ap.add_argument("--dataset_hf_path", type=str, default=None)
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--dataset_split", type=str, default=None)
    ap.add_argument("--dataset_text_fields", type=str, default=None,
                    help="For --dataset generic/flask_hard: comma-separated fields to build prompt.")


    # encoder
    ap.add_argument("--text_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--instruction_prefix", type=str, default="query: ")

    # pacing
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--reference_model", type=str, default="qwen/qwen-2.5-72b-instruct",
                    help="Reference model for bandit scoring. Set to empty string to disable.")
    ap.add_argument("--reference_system_prompt", type=str,
                    default="You are a helpful assistant.Please answer the user's question thoroughly and accurately.")

    ap.set_defaults(
        use_query_embedding=True,
        use_meta_vectors=True,
        use_hadamard=True,
        use_cost_penalty=True,
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    cache = DiskCache(args.cache)

    # ---------- Experts (expand this list as you like) ----------
    expert_specs = [
        {
            "name": "qwen-3",
            "model": "qwen/qwen3-vl-235b-a22b-instruct",
            "sys": "You are a powerful expert in multiple regions. I need your assistance. Answer the question step-by-step and be precise.",
            "cost": 1.5,
            "slot": 0,
        },
        {
            "name": "deepseek-r1",
            "model": "deepseek/deepseek-r1",
            "sys": "You are a powerful expert in multiple regions. I need your assistance. Answer the question step-by-step and be precise.",
            "cost": 1.2,
            "slot": 1,
        },
        {
            "name": "llama-3.1",
            "model": "meta-llama/llama-3.1-405b-instruct",
            "sys": "You are a powerful expert in multiple regions. I need your assistance. Answer the question step-by-step and be precise.",
            "cost": 3.1,
            "slot": 2,
        },
        {
            "name": "deepseek-3",
            "model": "deepseek/deepseek-v3.2",
            "sys": "You are a math expert. I need your assistance with problems both related to math or not. Answer the question step-by-step and be precise.",
            "cost": 0.32,
            "slot": 3,
        },
        {
            "name": "claude-4.5",
            "model": "anthropic/claude-sonnet-4.5",
            "sys": "You are a coding expert. I need your assistance with problems both related to programming or not. Answer the question step-by-step and be precise.",
            "cost": 15,
            "slot": 4,
        },
        {
            "name": "gpt-4o",
            "model": "openai/gpt-4o",
            "sys": "You are a coding expert. I need your assistance with problems both related to programming or not. Answer the question step-by-step and be precise.",
            "cost": 10,
            "slot": 5,
        },
        {
            "name": "Mistral-8x22B",
            "model": "mistralai/mixtral-8x22b-instruct",
            "sys": "You are a helpful assistant with creativity. I need your assistance with problems both related to art or not. Answer the question with creativity and be precise.",
            "cost": 1.2,
            "slot": 6,
        },
        {
            "name": "mistral-small-creative",
            "model": "mistralai/mistral-small-creative",
            "sys": "You are a helpful assistant with creativity. I need your assistance with problems both related to art or not. Answer the question with creativity and be precise.",
            "cost": 0.3,
            "slot": 7,
        },
        {
            "name": "grok-4.1-fast",
            "model": "x-ai/grok-4.1-fast",
            "sys": "You are a language expert. I need your assistance. Answer the question and be precise.",
            "cost": 0.5,
            "slot": 8,
        },
        {
            "name": "qwen-2.5-72b",
            "model": "qwen/qwen-2.5-72b-instruct",
            "sys": "You are a language expert. I need your assistance. Answer the question and be precise.",
            "cost": 0.39,
            "slot": 9,
        },
        {
            "name": "gemma-2-27b-it",
            "model": "google/gemma-2-27b-it",
            "sys": "You are a language expert. I need your assistance. Answer the question and be precise.",
            "cost": 0.65,
            "slot": 10,
        },
        {
                "name": "qwen-2.5-32b-instruct",
                "model": "qwen/qwen-2.5-32b-instruct",
                "sys": "You are an accurate expert. Strive for the best possible answer.",
                "cost": 0.1,
                "slot": 11,
        },
        {
                "name": "glm-4-32b",
                "model": "z-ai/glm-4-32b",
                "sys": "You are a helpful expert. Strive for clarity and correctness in your answers.",
                "cost": 0.1104,
                "slot": 12,
        },
        {
                "name": "llama-3.1-70b-instruct",
                "model": "meta-llama/llama-3.1-70b-instruct",
                "sys": "You are a helpful expert. Strive for clarity and correctness in your answers.",
                "cost": 0.2,
                "slot": 13,
        },
        {
                "name": "gpt-3.5-turbo",
                "model": "openai/gpt-3.5-turbo",
                "sys": "You are a helpful expert. Strive for clarity and correctness in your answers.",
                "cost": 1.5,
                "slot": 14,
        },

        # You should add more experts to get a meaningful routing gain.
        # {"name": "...", "model": "...", "sys": "...", "cost": ..., "slot": ...},
    ]

    # ---- For ablation policies ----
    rng_select = np.random.default_rng(args.seed)

    # cheapest by ORIGINAL USD cost in expert_specs (NOT normalized)
    # Build mapping name -> usd_cost
    name2_usd = {spec["name"]: float(spec["cost"]) for spec in expert_specs}
    cheapest_sorted = sorted(range(len(expert_specs)), key=lambda i: name2_usd[expert_specs[i]["name"]])
    cheapest_k_idx = np.array(cheapest_sorted[:args.k], dtype=int)



    META_PATH = "artifacts/mmlu_meta_10d.json"
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    
    args.d_s = len(meta["cap_names"]) + 1
    # ---------- Aggregator ----------
    aggregator_cfg = AggregatorConfig(
        model="qwen/qwen-2.5-72b-instruct",
        system_prompt="You are the Wise Integrator. Combine expert answers and output the best final answer.",
        temperature=0.0,
        max_tokens=1024,
    )
    aggregator = LLMAggregator(aggregator_cfg)

    # ---------- Reference baseline model ----------
    ref_model = None
    if args.reference_model.strip():
        ref_model = OpenRouterLLMExpert(
            meta=ExpertMeta(name="ref", s=np.zeros(args.d_s, dtype=np.float64), cost=0.0),
            cfg=LLMExpertConfig(
                model=args.reference_model,
                system_prompt=args.reference_system_prompt,
                temperature=0.2,
                max_tokens=1024,
            ),
        )

    # ---------- Judge ----------
    judge_model = "openai/gpt-4o"
    judge_client = OpenRouterClient()


    # ---------- Build experts ----------
    def make_s_cap10_cost(cap10: List[float], cost_norm: float) -> np.ndarray:
        s10 = np.array(cap10, dtype=np.float64)          # (10,)
        s11 = np.concatenate([s10, [float(cost_norm)]])  # (11,)
        return s11


    def normalize_cost(cost: float, cmin: float, cmax: float) -> float:
        if cmax <= cmin:
            return 0.0
        return (cost - cmin) / (cmax - cmin)

    costs = [float(spec["cost"]) for spec in expert_specs]
    cmin, cmax = min(costs), max(costs)

    

    def get_cap10_from_meta(expert_name: str) -> List[float]:
        return meta["models"][expert_name]["cap_acc_10d"]

    experts: List[OpenRouterLLMExpert] = []
    for spec in expert_specs:
        cost_norm = normalize_cost(float(spec["cost"]), cmin, cmax)

        cap10 = get_cap10_from_meta(spec["name"])        # list length 10
        s_vec = make_s_cap10_cost(cap10, cost_norm)      # np length 11

        meta_obj = ExpertMeta(name=spec["name"], s=s_vec, cost=cost_norm)
        experts.append(
            OpenRouterLLMExpert(
                meta=meta_obj,
                cfg=LLMExpertConfig(
                    model=spec["model"],
                    system_prompt=spec["sys"],
                    temperature=0.2,
                    max_tokens=1024,
                ),
            )
        )

    # ---------- PCA warm-start matrices ----------
    enc = STTextEncoder(model_name=args.text_model, instruction_prefix=args.instruction_prefix)
    d_u = enc.d_h + args.d_c
    S_list = [e.meta.s for e in experts]   # each is (d_s,) e.g. 11

    if args.warm_start == "pca":
        W_x, W_m = init_mats_pca_warmstart(
            d_x=args.d_x,
            d_u=d_u,
            S_list=S_list,
            seed=args.seed,
        )
    else:
        W_x, W_m = init_mats_random(
            d_x=args.d_x,
            d_u=d_u,
            d_s=args.d_s,
            seed=args.seed,
        )

    # ---------- Router ----------
    router = Router(
        W_x=W_x,
        W_m=W_m,
        experts=experts,
        text_model=args.text_model,
        instruction_prefix=args.instruction_prefix,
        d_c=args.d_c,
        alpha=3.5,
        lam=0.1,
        use_query_embedding=args.use_query_embedding,
        use_meta_vectors=args.use_meta_vectors,
        use_hadamard=args.use_hadamard,
    )
    router.aggregator = aggregator  # attach
    effective_lam_cost = args.lam_cost if args.use_cost_penalty else 0.0

    scoress, rewards, costs = [], [], []

    with open(args.out_log, "w", encoding="utf-8") as wf:

        it = iter_dataset(
            dataset=args.dataset,
            limit=args.limit,
            seed=args.seed,
            dataset_hf_path=args.dataset_hf_path,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            dataset_text_fields=args.dataset_text_fields,
        )
        for t, (prompt, raw) in enumerate(tqdm(it, total=args.limit if args.limit else None)):
            
            actual_cost_usd = 0.0
            # 1) select top-k (policy)
            _, phis = router._phis_for_prompt(prompt)

            if args.policy == "bandit":
                if t % 5 == 0:
                    all_names = [e.meta.name for e in router.experts]
                    debug_phi(router, phis, all_names, topn=10)
                chosen_idx = router.agent.select_topk(
                    phis,
                    k=args.k,
                    costs=router.costs if args.use_cost_penalty else None,
                    lam_cost=effective_lam_cost,
                )

            elif args.policy == "random":
                chosen_idx = rng_select.choice(len(router.experts), size=args.k, replace=False)

            elif args.policy == "cheapest":
                chosen_idx = cheapest_k_idx.copy()

            else:
                raise ValueError(f"Unknown policy: {args.policy}")

            chosen_experts = [router.experts[i] for i in chosen_idx]
            chosen_names = [e.meta.name for e in chosen_experts]

            # 2) expert answers (cached)
            answers = []
            for e in chosen_experts:
                key = {"type": "expert", "model": e.cfg.model, "sys": e.cfg.system_prompt, "prompt": prompt}
                cached = cache.get(key)


                if cached is None:
                    out = e.infer_with_usage(prompt) 
                    ans = out["text"]
                    cache.put(key, {"text": ans, "usage": out.get("usage", {}), "id": out.get("id")})
                    actual_cost_usd += usage_cost_usd(out)
                    print (actual_cost_usd)
                else:
           
                    ans = cached["text"] if isinstance(cached, dict) else cached
                    actual_cost_usd += usage_cost_usd(cached)
                
                answers.append(ans)

            # 3) aggregate (cached)
            agg_key = {"type": "agg", "model": aggregator_cfg.model, "prompt": prompt, "names": chosen_names, "answers": answers}
            final = cache.get(agg_key)
            if final is None:
                final = router.aggregator.aggregate_with_usage(prompt, chosen_names, answers)
                cache.put(agg_key, final)
            actual_cost_usd += usage_cost_usd(final)

            final_text = final["text"] if isinstance(final, dict) else str(final)
            if args.policy == "bandit":
                # 4) reference baseline (cached)
                ref_text = None
                ref = None
                if ref_model is not None:
                    ref_key = {"type": "ref", "model": ref_model.cfg.model, "sys": ref_model.cfg.system_prompt, "prompt": prompt}
                    ref = cache.get(ref_key)
                    if ref is None:
                        ref = ref_model.infer_with_usage(prompt)
                        cache.put(ref_key, ref)
                    actual_cost_usd += usage_cost_usd(ref)
                    ref_text = ref["text"] if isinstance(ref, dict) else str(ref)

                # 5) judge score (cached)
            if args.policy == "bandit":
                score = judge_score_llm(
                    judge_client=judge_client,
                    judge_model=judge_model,
                    prompt=prompt,
                    answer=final_text,
                    reference=ref_text,
                    cache=cache,
                    score_min=0.0,
                    score_max=10.0,
                )

                # 6) reward + update
                cost = float(router.costs[chosen_idx].sum())
                reward = float(score - effective_lam_cost * cost)

                Z = np.sum([phis[i] for i in chosen_idx], axis=0)
                router.agent.update(Z, reward)

                scoress.append(score)
                costs.append(actual_cost_usd)
                rewards.append(reward)
            else:
                score = -1.0
                ref = None
                reward = 0.0
                costs.append(actual_cost_usd)

            rec = {
                "t": t,
                "dataset": "alpaca_eval_2.0",
                "policy": args.policy,
                "warm_start": args.warm_start,
                "use_query_embedding": bool(args.use_query_embedding),
                "use_meta_vectors": bool(args.use_meta_vectors),
                "use_hadamard": bool(args.use_hadamard),
                "use_cost_penalty": bool(args.use_cost_penalty),
                "lam_cost_effective": float(effective_lam_cost),
                "k": int(args.k),
                "prompt": prompt,
                "chosen_idx": chosen_idx.tolist(),
                "chosen_names": chosen_names,
                "answers": answers,
                "final": final,
                "reference": ref["text"] if isinstance(ref, dict) else ref,
                "score_final_vs_ref": float(score),
                "cost": float(actual_cost_usd),
                "reward": float(reward),
                "avg_score": float(np.mean(scoress)),
                "avg_cost": float(np.mean(costs)),
                "avg_reward": float(np.mean(rewards)),
                "raw": raw,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()

            time.sleep(args.sleep)

    total_cost = float(np.sum(costs))
    num_instructions = len(costs)
    cost_per_instruction = (total_cost / num_instructions) if num_instructions else 0.0
    print(f"[DONE] wrote log to: {args.out_log}")
    print(
        f"avg_score={float(np.mean(scoress)):.4f}, "
        f"avg_reward={float(np.mean(rewards)):.4f}, "
        f"total_cost={total_cost:.4f}, "
        f"num_instructions={num_instructions}, "
        f"cost_per_instruction={cost_per_instruction:.4f}"
    )


if __name__ == "__main__":
    main()
