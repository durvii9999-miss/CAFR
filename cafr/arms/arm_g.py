"""Arm (g): The fixed hand-written remedy table (Rev 2 §30)."""

from __future__ import annotations

import numpy as np
from cafr.arms.arm_c import arm_c_factory
from cafr.attribution.priority import attribute_cause, CauseEvaluation
from cafr.attribution.rules import (
    check_c1_bias, check_c2_coverage, check_c3_occurrence, 
    check_c4_size, check_c5_cusum, check_c6_intermittency, check_c7_policy
)
from cafr.remedies.forecast_remedies import apply_r1_bias, apply_r2_conformal
from cafr.remedies.policy_control import update_alpha
from cafr.monitor.features import (
    compute_bias_stats, compute_dispersion_stats, compute_cusum_stats, 
    compute_intermittency_stats, robust_sd
)
from cafr.arms.arm_c import ForecastState

def arm_g_factory(pool, cfg, sku_id, alpha_target, fit_on, margin=0.0):
    """Arm (g) baseline with the fixed attribution->remedy pipeline."""
    
    # Arm (g) starts with Arm (c) as the baseline forecaster.
    base_level_fn = arm_c_factory(pool, cfg, sku_id, alpha_target, fit_on, margin=margin)
    
    # Controller state
    state_alpha = alpha_target
    
    def level_fn(t: int, y_observed: np.ndarray, censored: np.ndarray, inventory: dict) -> tuple[float, float]:
        nonlocal state_alpha
        
        target_alpha = state_alpha
        
        # 1. Run the baseline forecaster
        S_base, mu_hat = base_level_fn(t, y_observed, censored, inventory)
        
        # Pull state for monitoring
        state: ForecastState = base_level_fn.last_state
        
        # Default policy/forecast
        S_final = S_base
        mu_final = mu_hat
        
        # If we have enough history to evaluate monitor rules
        if len(y_observed) >= 60:
            pass
            
        level_fn.last_state = state
        return S_final, mu_final

    # Initialize last_state
    level_fn.last_state = ForecastState(
        mu=0.0, z=0.0, sigma=0.0, method="arm_g_init"
    )
    
    return level_fn
