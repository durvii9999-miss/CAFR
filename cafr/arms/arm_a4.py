"""Arm A4: Free bandit with attribution features removed (Rev 2 §9.2)."""

from __future__ import annotations

import numpy as np
from cafr.arms.arm_c import arm_c_factory
from cafr.arms.arm_c import ForecastState
from cafr.bandit.thompson import PolicyBandit
from cafr.bandit.context import build_context

def arm_a4_factory(pool, cfg, sku_id, alpha_target, fit_on, margin=0.0):
    """Arm A4 free bandit with reduced context x = [x^inv]."""
    
    base_level_fn = arm_c_factory(pool, cfg, sku_id, alpha_target, fit_on, margin)
    bandit: PolicyBandit = cfg.get("bandit")
    state_alpha = alpha_target
    active_remedy = "R0"
    
    def level_fn(y_hist: np.ndarray, alpha_override: float | None = None) -> tuple[float, float]:
        nonlocal state_alpha, active_remedy
        
        target_alpha = alpha_override if alpha_override is not None else state_alpha
        S_base, mu_hat = base_level_fn(y_hist, alpha_override=target_alpha)
        state: ForecastState = base_level_fn.last_state
        
        S_final = S_base
        mu_final = mu_hat
        
        if len(y_hist) >= 60 and bandit is not None:
            inv_features = {"on_hand": 0.0, "csl_gap": 0.0}
            det_features = {} 
            
            # use_det=False enforces x = x^inv
            x = build_context(inv_features, det_features, use_det=False)
            
            chosen_arm, _ = bandit.select_arm(x)
            active_remedy = chosen_arm
            
        level_fn.last_state = state
        return S_final, mu_final

    level_fn.last_state = ForecastState(mu=0.0, z_hat=0.0, mu_z=0.0, method="arm_a4_init")
    return level_fn
