# src/envs/synthetic_env.py
import numpy as np

class SyntheticExpertWorld:
    def __init__(self, n_arms: int, d: int, noise: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(n_arms, d))  
        self.noise = noise
        self.rng = rng

    def sample_context(self) -> np.ndarray:
        x = self.rng.normal(size=(self.W.shape[1],))
        return x / max(np.linalg.norm(x), 1e-9)

    def true_rewards(self, x: np.ndarray) -> np.ndarray:
        mean = self.W @ x
        y = mean + self.noise * self.rng.normal(size=mean.shape)
        
        y = 1 / (1 + np.exp(-y))
        return y
