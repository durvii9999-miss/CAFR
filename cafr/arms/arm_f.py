"""Arm (f): The free bandit (Rev 2 §9.2)."""

from __future__ import annotations

import numpy as np
from cafr.arms.arm_c import arm_c_factory
from cafr.arms.arm_c import ForecastState
from cafr.bandit.thompson import PolicyBandit
from cafr.bandit.context import build_context

def arm_f_factory(pool, cfg, sku_id, alpha_target, fit_on, margin=0.0):
    """Arm (f) free bandit with full context x = [x^inv, x^det]."""
    
    # Initialize the base forecaster
    base_level_fn = arm_c_factory(pool, cfg, sku_id, alpha_target, fit_on, margin=margin)
    
    # The global bandit is passed in via cfg or instantiated here for simulation.
    # In full rollout, the bandit is shared across SKUs. Here we assume it's attached to cfg.
    bandit: PolicyBandit = cfg.get("bandit")
    
    state_alpha = alpha_target
    active_remedy = "R0"
    
    def level_fn(t: int, y_observed: np.ndarray, censored: np.ndarray, inventory: dict) -> tuple[float, float]:
        nonlocal state_alpha, active_remedy
        
        target_alpha = state_alpha # could be overridden by bandit later
        
        # 1. Base forecast
        S_base, mu_hat = base_level_fn(t, y_observed, censored, inventory)
        state: ForecastState = base_level_fn.last_state
        
        S_final = S_base
        mu_final = mu_hat
        
        if len(y_observed) >= 60 and bandit is not None:
            # Build context (dummy features for this skeleton)
            inv_features = {"on_hand": 0.0, "csl_gap": 0.0}
            det_features = {"cause": "C0", "conf": 0.0}
            
            x = build_context(inv_features, det_features, use_det=True)
            
            # Select action
            chosen_arm, _ = bandit.select_arm(x)
            active_remedy = chosen_arm
            
            # (Apply remedy would go here)
            
        level_fn.last_state = state
        return S_final, mu_final

    level_fn.last_state = ForecastState(
        mu=0.0, z=0.0, sigma=0.0, method="arm_f_init"
    )
    
    return level_fn
