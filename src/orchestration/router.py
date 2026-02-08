# src/orchestration/router.py
import json, os
import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embeddings.encoders import STTextEncoder, FeatEncoder
try:
    from src.embeddings.projectors import TaskProjector, MetaProjector, build_phi
except ImportError:
    from src.features.projectors import TaskProjector, MetaProjector, build_phi
from src.bandits.linucb_combinatorial import LinUCBSharedCombinatorial
from src.experts.base import BaseExpert, ExpertMeta
from src.experts.llm_expert import OpenRouterLLMExpert, LLMExpertConfig
from src.aggregators.llm_aggregator import LLMAggregator, AggregatorConfig

def _safe_load(path: str):
    if path and os.path.exists(path):
        return np.load(path)
    return None

class Router:
    def __init__(
        self,
        W_x: np.ndarray,
        W_m: np.ndarray,
        experts: List[BaseExpert],
        text_model: str,
        instruction_prefix: str,
        d_c: int,
        alpha: float = 1.0,
        lam: float = 1.0,
        W_x_path: str | None = None,
        b_x_path: str | None = None,
        W_m_path: str | None = None,
        b_m_path: str | None = None,
        V0_inv_path: str | None = None,
        b0_path: str | None = None,
        w0_path: str | None = None,
    ):


        if W_x_path is not None:
            W_x = np.load(W_x_path)
        if W_m_path is not None:
            W_m = np.load(W_m_path)

        b_x = _safe_load(b_x_path)   
        b_m = _safe_load(b_m_path)

        self.task_proj = TaskProjector(W_x, b_x=b_x)
        self.meta_proj = MetaProjector(W_m, b_m=b_m)



        self.experts = experts
        self.enc = STTextEncoder(model_name=text_model, instruction_prefix=instruction_prefix)
        self.feat = FeatEncoder(d_c=d_c)
        d_x = W_x.shape[0]
        self.p = 3 * d_x

        self.agent = LinUCBSharedCombinatorial(p=self.p, alpha=alpha, lam=lam)
        self.aggregator: LLMAggregator
        self.judge: LLMJudge
        #  v_i
        self.v_list = [self.meta_proj(e.meta.s) for e in experts]
        self.costs  = np.array([e.meta.cost for e in experts], dtype=float)

        V_inv = _safe_load(V0_inv_path)
        b = _safe_load(b0_path)
        w_hat = _safe_load(w0_path)
        if V_inv is not None or b is not None or w_hat is not None:
            self.agent.load_state(V_inv=V_inv, b=b, w_hat=w_hat)

    def _phis_for_prompt(self, q: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        h = self.enc(q)
        c = self.feat(q)
        x = self.task_proj(h, c)
        phis = [build_phi(x, v) for v in self.v_list]
        return x, phis

    def step_real(
        self,
        q: str,
        reference: Optional[str],
        k: int,
        lam_cost: float = 0.0,
        judge_mode: str = "llm",   # "llm" or "exact" etc.
        selection_mode: str = "set_ucb_greedy",
        )-> Dict:
        x, phis = self._phis_for_prompt(q)

        chosen_idx = self.agent.select_topk(phis, k=k, mode=selection_mode)
        chosen_experts = [self.experts[i] for i in chosen_idx]

        # 1) call experts
        answers = [e.infer(q) for e in chosen_experts]
        names = [e.meta.name for e in chosen_experts]

        # 2) aggregate by strong LLM
        final = self.aggregator.aggregate(q, names, answers)

        # 3) judge quality
        if judge_mode == "exact" and reference is not None:
            quality = 1.0 if final.strip() == reference.strip() else 0.0
        else:
            quality = float(self.judge.score(q, final, reference=reference))

        # 4) cost & reward
        cost = float(self.costs[chosen_idx].sum())
        reward = float(quality - lam_cost * cost)  

        # 5) update bandit
        Z = np.sum([phis[i] for i in chosen_idx], axis=0)
        self.agent.update(Z, reward)

        return {
            "chosen_idx": chosen_idx.tolist(),
            "expert_names": names,
            "answers": answers,
            "final": final,
            "quality": quality,
            "cost": cost,
            "reward": reward,
        }