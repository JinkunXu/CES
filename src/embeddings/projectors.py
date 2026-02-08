# src/embeddings/projectors.py
import numpy as np
from typing import Optional

class TaskProjector:  # x = W_x [h;c] + b_x
    def __init__(self, W_x: np.ndarray, b_x: Optional[np.ndarray] = None):
        self.W_x = W_x.astype(np.float64)  # (d_x, d_u)
        d_x = self.W_x.shape[0]
        if b_x is None:
            b_x = np.zeros(d_x, dtype=np.float64)
        self.b_x = b_x.astype(np.float64)  # (d_x,)

    def __call__(self, h: np.ndarray, c: np.ndarray) -> np.ndarray:
        u = np.concatenate([h, c]).astype(np.float64)  # (d_u,)
        return (self.W_x @ u) + self.b_x               # (d_x,)

class MetaProjector:  # v_i = W_m s_i + b_m
    def __init__(self, W_m: np.ndarray, b_m: Optional[np.ndarray] = None):
        self.W_m = W_m.astype(np.float64)  # (d_x, d_s)
        d_x = self.W_m.shape[0]
        if b_m is None:
            b_m = np.zeros(d_x, dtype=np.float64)
        self.b_m = b_m.astype(np.float64)  # (d_x,)

    def __call__(self, s: np.ndarray) -> np.ndarray:
        s = s.astype(np.float64)
        return (self.W_m @ s) + self.b_m



def _l2n(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(a)
        return a / (n + eps)

def build_phi(x: np.ndarray, v: np.ndarray, beta: float = 3.0, gamma: float = 3.0) -> np.ndarray:
        x = x.astype(np.float64)
        v = v.astype(np.float64)

        xn  = _l2n(x)
        vn  = _l2n(v)
        xvn = _l2n(x * v)

       
        return np.concatenate([xn, beta * vn, gamma * xvn])  # phi_i = [ x ; v_i ; x ⊙ v_i ]


