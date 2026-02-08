import numpy as np
from typing import Dict, Optional

ABILITY_KEYS_10 = [
    "math",              # 0
    "physics_chem",      # 1
    "bio_medicine",      # 2
    "cs_ml_security",    # 3
    "law_politics",      # 4
    "business_econ",     # 5
    "history_humanities",# 6
    "psych_sociology",   # 7
    "logic_ethics",      # 8
    "geography_other",   # 9,
]

def make_s_vec_ability10(
    profile: Dict[str, float],
    cost: float,
    d_s: Optional[int] = None,
    cost_dim: Optional[int] = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Build expert meta vector s_i with:
      - 10 ability dims in fixed order (ABILITY_KEYS_10)
      - 1 cost dim (default: last dim)

    Args:
      profile: dict like {"math":0.8, "coding":0.6, ...}. Missing keys -> 0.0
      cost: float, arbitrary scale; recommended normalize to [0,1] or log-scale
      d_s: if None, set to 11 (10 abilities + cost)
      cost_dim: default last dim
      clip: clip abilities to [0,1]

    Returns:
      s: np.ndarray shape (d_s,)
    """
    if d_s is None:
        d_s = len(ABILITY_KEYS_10) + 1  # 11

    if d_s < len(ABILITY_KEYS_10) + 1:
        raise ValueError(f"d_s must be >= 11, got d_s={d_s}")

    s = np.zeros(d_s, dtype=np.float64)

    # abilities
    for j, k in enumerate(ABILITY_KEYS_10):
        v = float(profile.get(k, 0.0))
        if clip:
            v = max(0.0, min(1.0, v))
        s[j] = v

    # cost dim
    cd = (d_s - 1) if cost_dim is None else int(cost_dim)
    if cd < 0 or cd >= d_s:
        raise ValueError(f"cost_dim={cd} out of range for d_s={d_s}")
    if cd < len(ABILITY_KEYS_10):
        
        raise ValueError(f"cost_dim={cd} conflicts with ability dims [0..9]. Use cost_dim>=10 (e.g., last dim).")

    s[cd] = float(cost)
    return s
