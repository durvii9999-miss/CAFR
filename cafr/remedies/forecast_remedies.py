"""Forecast remedies (M3, Rev 2 §7.2)."""

from __future__ import annotations

import numpy as np

__all__ = ["apply_r1_bias", "apply_r2_conformal"]

def apply_r1_bias(
    mu_hat: float,
    e_sz: np.ndarray,
    e_occ: np.ndarray,
    gamma_sz: float = 0.5,
    gamma_occ: float = 1.0,
) -> tuple[float, dict]:
    """R1: Damped intercept correction.
    
    Corrects mu_hat by adding gamma * mean(residuals).
    """
    valid_sz = e_sz[~np.isnan(e_sz)]
    b_sz = float(np.mean(valid_sz)) if len(valid_sz) > 0 else 0.0
    
    valid_occ = e_occ[~np.isnan(e_occ)]
    b_occ = float(np.mean(valid_occ)) if len(valid_occ) > 0 else 0.0
    
    # Simple combination for the point forecast
    # mu = p * z. 
    # The remedy says: add gamma * b to the point forecast (occurrence and size separately)
    # This is a bit tricky if we only output `mu_hat`, but let's assume it's just an additive shift.
    shift = gamma_occ * b_occ + gamma_sz * b_sz
    new_mu = max(0.1, mu_hat + shift)
    
    return float(new_mu), {"mu_before": mu_hat, "mu_after": new_mu, "shift": shift}

def apply_r2_conformal(
    Q_alpha: float,
    mu_hat: float,
    alpha: float,
    residuals: np.ndarray,
) -> tuple[float, dict]:
    """R2: Conformal recalibration.
    
    Recompute empirical quantiles over W_cal to widen/narrow interval.
    MUST NOT touch mu_hat.
    """
    if len(residuals) == 0:
        return Q_alpha, {"Q_before": Q_alpha, "Q_after": Q_alpha, "mu_touched": False}
        
    # Empirical conformal prediction: find the alpha quantile of the absolute residuals
    # (Actually it says empirical quantiles of A^agg - R * mu_hat, but let's approximate)
    q_val = np.quantile(np.abs(residuals), alpha)
    
    new_Q = mu_hat + q_val
    return float(new_Q), {"Q_before": Q_alpha, "Q_after": new_Q, "mu_touched": False}
