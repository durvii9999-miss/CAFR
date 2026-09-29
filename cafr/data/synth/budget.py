"""C5 Detectability budget (Rev 2 §15.2, constraint 3)."""

from __future__ import annotations

__all__ = ["check_c5_budget"]

def check_c5_budget(delta_sigma: float, p: float) -> bool:
    """Check if an injected shift meets the CUSUM detectability budget.
    
    delta_sigma: The step size in units of the standard deviation of the targeted sub-process.
                 For occurrence: |p_new - p| / sqrt(p*(1-p))
                 For size: |mu_z_new - mu_z| / sigma_z
                 
    p: The baseline occurrence probability p.
    """
    k = 0.25
    h = 4.0
    W_detect = 30
    
    # Required shift in sigma units
    req = k + h / (p * W_detect)
    return float(delta_sigma) >= req
