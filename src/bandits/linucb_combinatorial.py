# src/bandits/linucb_combinatorial.py
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class LinUCBSharedCombinatorial:
    p: int                   # dim: p = 3 * d_x
    alpha: float = 1.0       # ucb coeff
    lam: float = 1.0         # λ

    def __post_init__(self):
        self.V_inv = np.eye(self.p) / self.lam
        self.b = np.zeros(self.p)
        self.w_hat = np.zeros(self.p)

    # single(expert i)'s UCB ： μ_i + α * σ_i
    def _item_ucb(self, phi: np.ndarray) -> float:
        mu = float(phi @ self.w_hat)
        s2 = float(phi @ (self.V_inv @ phi))
        return mu + self.alpha * np.sqrt(max(s2, 1e-12))

    # choosing top-k：
    def select_topk(self, phis: List[np.ndarray], k: int,
                costs: np.ndarray | None = None,
                lam_cost: float = 0.0) -> np.ndarray:
        m = len(phis)

        
        chosen, z = [], np.zeros(self.p)
        base_std = 0.0
        remaining = set(range(m))            
        for _ in range(k):
            best_i, best_val, best_new_std = None, -1e18, base_std
            for i in remaining:
                phi = phis[i]
                                  
                mu = float(phi @ self.w_hat)
                if costs is not None and lam_cost > 0:
                    mu = mu - lam_cost * float(costs[i])

                new_std = np.sqrt(max((z + phi) @ (self.V_inv @ (z + phi)), 1e-12))
                marg = mu + self.alpha * (new_std - base_std)
                if marg > best_val:
                    best_i, best_val, best_new_std = i, marg, new_std
            chosen.append(best_i)
            remaining.remove(best_i)
            z += phis[best_i]
            base_std = best_new_std
        return np.array(chosen, dtype=int)



    # update with observed reward r for chosen set Z
    def update(self, z: np.ndarray, r: float) -> None:
        Vz = self.V_inv @ z
        denom = 1.0 + float(z @ Vz)
        self.V_inv = self.V_inv - np.outer(Vz, Vz) / denom
        self.b += r * z
        self.w_hat = self.V_inv @ self.b

    # save/load state for checkpointing
    def load_state(self, V_inv: np.ndarray | None = None, b: np.ndarray | None = None, w_hat: np.ndarray | None = None):
        if V_inv is not None:
            assert V_inv.shape == (self.p, self.p)
            self.V_inv = V_inv.astype(np.float64)
        if b is not None:
            assert b.shape == (self.p,)
            self.b = b.astype(np.float64)
        if w_hat is not None:
            assert w_hat.shape == (self.p,)
            self.w_hat = w_hat.astype(np.float64)
