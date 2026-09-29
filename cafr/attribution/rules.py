"""Attribution rules (M2, Rev 2 §6.3)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "check_c1_bias",
    "check_c2_coverage",
    "check_c3_occurrence",
    "check_c4_size",
    "check_c5_cusum",
    "check_c6_intermittency",
    "check_c7_policy",
]

def check_c1_bias(
    z_b: float,
    signs: list[float],
    n_usable: int,
    step_bic: float,
    drift_bic: float,
    min_evidence: int = 8,
) -> bool:
    """C1: Persistent directional bias.
    
    1. |z_b| > 1.96
    2. Sign stable across last 4 windows
    3. Drift model beats step model on BIC
    4. n >= 8
    """
    if n_usable < min_evidence:
        return False
    if abs(z_b) <= 1.96:
        return False
        
    # Sign stability: all last 4 signs are the same and non-zero
    if len(signs) < 4:
        return False
    last_4 = np.sign(signs[-4:])
    if not np.all(last_4 == last_4[0]) or last_4[0] == 0:
        return False
        
    if drift_bic > step_bic:
        return False
        
    return True

def check_c2_coverage(
    c_gap: float,
    p_val: float,
    z_b: float,
    r: float,
    tau_cov: float = 0.10,
) -> bool:
    """C2: Uncertainty mis-calibration.
    
    1. |c_hat - alpha| > tau_cov
    2. p_val < 0.05
    3. |z_b| <= 1.96
    4. r <= 1.5
    """
    if np.isnan(c_gap) or np.isnan(p_val) or np.isnan(z_b) or np.isnan(r):
        return False
        
    if abs(c_gap) <= tau_cov:
        return False
    if p_val >= 0.05:
        return False
        
    if abs(z_b) > 1.96:
        return False
    if r > 1.5:
        return False
        
    return True

def check_c3_occurrence(
    auc: float,
    fz: float,
    r: float,
) -> bool:
    """C3: Occurrence failure.
    
    1. AUC < 0.65 or FZ > 0.30
    2. r <= 1.5
    """
    if np.isnan(r):
        return False
        
    trigger = False
    if not np.isnan(auc) and auc < 0.65:
        trigger = True
    if not np.isnan(fz) and fz > 0.30:
        trigger = True
        
    if not trigger:
        return False
        
    if r > 1.5:
        return False
        
    return True

def check_c4_size(
    r_history: list[float],
    auc: float,
) -> bool:
    """C4: Size failure.
    
    1. r > 1.5 over 2 consecutive windows
    2. AUC >= 0.65
    """
    if len(r_history) < 2:
        return False
        
    if not (r_history[-1] > 1.5 and r_history[-2] > 1.5):
        return False
        
    if not np.isnan(auc) and auc < 0.65:
        return False
        
    return True

def check_c5_cusum(
    alarm_occ: bool,
    alarm_sz: bool,
    step_bic: float,
    drift_bic: float,
) -> bool:
    """C5: Level/regime shift.
    
    1. Either CUSUM alarms
    2. Step model beats drift model on BIC
    """
    if not (alarm_occ or alarm_sz):
        return False
        
    if step_bic >= drift_bic:
        return False
        
    return True

def check_c6_intermittency(
    cell_history: list[str],
    init_cell: str,
    bootstrap_stability: float,
) -> bool:
    """C6: Intermittency change.
    
    1. Cell differs from init_cell sustained for 3 consecutive windows
    2. Bootstrap stability >= 0.90
    """
    if len(cell_history) < 3:
        return False
        
    last_3 = cell_history[-3:]
    if any(c == init_cell for c in last_3):
        return False
    # Must also be the SAME new cell? The spec says "differs from the cell used at initialisation"
    # Wait, does it have to be consistently the SAME new cell? "sustained for m=3 consecutive windows"
    # Usually implies the same cell, but let's assume any differing cell sustained for 3 windows is a shift.
    if len(set(last_3)) != 1:
        return False
        
    if bootstrap_stability < 0.90:
        return False
        
    return True

def check_c7_policy(
    csl_gap: float,
    z_b: float,
    r: float,
    tau_svc: float = 0.05,
) -> bool:
    """C7: Inventory-policy mis-set.
    
    1. CSL gap < -tau_svc (under-service)
    2. Forecast is acceptable (e.g. z_b and r at control)
    """
    if np.isnan(csl_gap) or np.isnan(z_b) or np.isnan(r):
        return False
        
    if csl_gap > -tau_svc:
        return False
        
    if abs(z_b) > 1.96 or r > 1.5:
        return False
        
    return True
