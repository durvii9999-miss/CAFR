"""Churn control mechanisms (Rev 2 §11)."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

def can_switch(
    periods_since_switch: int,
    c5_alarm: bool,
    conf: float,
    d: int = 3,
    tau_conf: float = 0.35,
) -> bool:
    """Determine if a switch is permitted.
    
    1. Dwell time constraint (d=3).
    2. C5 alarm overrides dwell time.
    3. Confidence gate.
    """
    if conf < tau_conf:
        return False
        
    if c5_alarm:
        return True # High-severity trigger overrides dwell
        
    if periods_since_switch < d:
        return False
        
    return True

def compute_churn_metrics(
    decisions: list[Dict[str, Any]],
    T: int,
) -> tuple[float, float]:
    """Compute model churn and action churn for a single SKU.
    
    Excludes forced-exploration decisions (Rev 2 §11.1).
    """
    valid_decisions = [d for d in decisions if not d.get("exploratory", False)]
    
    if len(valid_decisions) < 2:
        return 0.0, 0.0
        
    model_switches = 0
    action_switches = 0
    
    for i in range(1, len(valid_decisions)):
        prev = valid_decisions[i-1]
        curr = valid_decisions[i]
        
        # Action churn
        if curr["active_remedy"] != prev["active_remedy"]:
            action_switches += 1
            
        # Model churn (only R3, R4, R5, R6 cause model churn)
        # R1, R2, R7 are parameter adjustments.
        model_churning_remedies = {"R3", "R4", "R5", "R6"}
        
        # If the underlying method family changed OR we applied a model-churning remedy
        if (curr["base_method"] != prev["base_method"]) or (
            curr["active_remedy"] != prev["active_remedy"] and curr["active_remedy"] in model_churning_remedies
        ):
            model_switches += 1
            
    # Normalize by T
    return model_switches / T, action_switches / T

def should_explore(sku_id: int, period: int) -> bool:
    """Staggered forced exploration.
    
    Approximately 1/9 of SKUs per period, deterministically.
    """
    return (sku_id * 31 + period) % 9 == 0
