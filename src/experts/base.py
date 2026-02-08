# src/experts/base.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ExpertMeta:
    name: str
    s: np.ndarray       # (d_s,) neta
    cost: float = 1.0   

class BaseExpert:
    def __init__(self, meta: ExpertMeta):
        self.meta = meta
    def infer(self, prompt: str) -> str:
        raise NotImplementedError
