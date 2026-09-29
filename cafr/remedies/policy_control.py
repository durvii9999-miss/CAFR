"""C7 Service-level integral controller (Rev 2 §7.1)."""

from __future__ import annotations

import numpy as np

__all__ = ["update_alpha"]

def update_alpha(
    alpha_t: float,
    alpha_target: float,
    c_t: float,
    n_t: int,
    gamma: float = 1.0,
) -> float:
    """Update the policy safety factor (R7).
    
    alpha_{t+1} = clip(alpha_t + gamma*(alpha_target - c_t), alpha_min, alpha_max)
    alpha_min = 0.50
    alpha_max = min(0.999, 1 - 1/(n_t + 1))
    """
    alpha_min = 0.50
    alpha_max = min(0.999, 1.0 - 1.0 / (n_t + 1.0))
    
    new_alpha = alpha_t + gamma * (alpha_target - c_t)
    return float(np.clip(new_alpha, alpha_min, alpha_max))
