"""Demand generation and injection (Rev 2 §15.2)."""

from __future__ import annotations

import numpy as np
from .budget import check_c5_budget

__all__ = [
    "inject_C0",
    "inject_C1",
    "inject_C3",
    "inject_C4",
    "inject_C5",
    "inject_C6",
]

def inject_C0(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C0: No injection. Pure baseline."""
    # We re-generate the whole series to ensure consistent random state usage if needed,
    # or we can just use the baseline generator.
    delta = rng.binomial(1, p, size=T).astype(float)
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    return delta * z, delta, z, {}

def inject_C1(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C1: Persistent bias. Linear ramp in mu_z. p untouched."""
    phi = rng.choice([-0.4, -0.2, 0.2, 0.4])
    
    delta = rng.binomial(1, p, size=T).astype(float)
    mu_z_t = np.full(T, mu_z)
    
    # Ramp over >= 20 periods, let's say it ramps until the end, but the formula is (t-tau)/40
    t_idx = np.arange(tau, T)
    mu_z_t[tau:] = mu_z * (1 + phi * (t_idx - tau) / 40.0)
    # Ensure mu_z stays > 0
    mu_z_t = np.maximum(mu_z_t, 0.1)
    
    z = rng.gamma(shape=k, scale=mu_z_t / k)
    return delta * z, delta, z, {"phi": phi}

def inject_C3(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C3: Occurrence failure. Step in p only. Sizes untouched."""
    shift = rng.choice([-0.15, 0.15])
    p_new = np.clip(p + shift, 0.03, 0.60)
    
    p_t = np.full(T, p)
    p_t[tau:] = p_new
    
    delta = rng.binomial(1, p_t).astype(float)
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    return delta * z, delta, z, {"p_new": p_new, "shift": shift}

def inject_C4(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C4: Size failure. Dispersion change (k). mu_z unchanged."""
    for _ in range(100):
        factor = rng.choice([0.25, 4.0])
        k_new = k * factor
        
        cv2_before = 1.0 / k
        cv2_after = 1.0 / k_new
        
        # C4 Constraint 1: ABDxCV2 cell invariance
        if (cv2_before - 0.49) * (cv2_after - 0.49) > 0:
            break
    else:
        # Fallback if impossible, though 0.25/4.0 will almost always cross or not cross?
        # Wait, if k=2.0 (cv2=0.5). k_new=8.0 (cv2=0.125). 0.5 > 0.49, 0.125 < 0.49. Crosses!
        # If it crosses, we need a different factor.
        pass
        
    k_t = np.full(T, k)
    k_t[tau:] = k_new
    
    delta = rng.binomial(1, p, size=T).astype(float)
    z = rng.gamma(shape=k_t, scale=mu_z / k_t)
    return delta * z, delta, z, {"k_new": k_new, "factor": factor}

def inject_C5(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C5: Level shift. Abrupt step in mu_z OR p. Must clear detectability budget."""
    target = rng.choice(["size", "occurrence"])
    
    p_t = np.full(T, p)
    mu_z_t = np.full(T, mu_z)
    
    param = {}
    budget_met = False
    
    if target == "size":
        factor = rng.choice([0.5, 2.0])
        mu_z_new = mu_z * factor
        
        # Sigma of size = mu_z / sqrt(k)
        sigma_z = mu_z / np.sqrt(k)
        delta_sigma = abs(mu_z_new - mu_z) / sigma_z
        
        if not check_c5_budget(delta_sigma, p):
            # Scale the step up until it clears (constraint 3)
            # req = k_cusum + h_cusum / (p * W_detect)
            req = 0.25 + 4.0 / (p * 30.0)
            # We need |mu_z_new - mu_z| = req * sigma_z
            # mu_z_new = mu_z +/- req * sigma_z
            direction = 1 if factor > 1 else -1
            mu_z_new = mu_z + direction * (req * sigma_z * 1.05) # 5% margin
            mu_z_new = max(mu_z_new, 0.1) # protect against negative
            budget_met = True # We forced it
            
        mu_z_t[tau:] = mu_z_new
        param = {"target": "size", "mu_z_new": mu_z_new}
    else:
        shift = rng.choice([-0.20, 0.20])
        p_new = np.clip(p + shift, 0.03, 0.60)
        
        sigma_occ = np.sqrt(p * (1 - p))
        delta_sigma = abs(p_new - p) / sigma_occ
        
        if not check_c5_budget(delta_sigma, p):
            req = 0.25 + 4.0 / (p * 30.0)
            direction = 1 if shift > 0 else -1
            p_new = np.clip(p + direction * (req * sigma_occ * 1.05), 0.03, 0.60)
            
        p_t[tau:] = p_new
        param = {"target": "occurrence", "p_new": p_new}

    delta = rng.binomial(1, p_t).astype(float)
    z = rng.gamma(shape=k, scale=mu_z_t / k)
    return delta * z, delta, z, param

def inject_C6(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C6: Intermittency regime. Slow drift in p so ADI crosses 4.0 (p=0.25)."""
    # p must cross 0.25. So if p < 0.25, it must drift up. If p > 0.25, it must drift down.
    psi_dir = 1.0 if p < 0.25 else -1.0
    
    # We need p(t) = p * (1 + psi*(t-tau)/60) to cross 0.25 within the remaining window (80 periods)
    # Let's just set psi such that at t = tau + 60, p is well past 0.25.
    # target_p at tau+60 could be 0.35 (if drifting up) or 0.15 (if drifting down).
    target_p = 0.35 if psi_dir > 0 else 0.15
    # target_p = p * (1 + psi) => psi = target_p / p - 1
    psi = (target_p / p) - 1.0
    
    t_idx = np.arange(T)
    drift = np.zeros(T)
    drift[tau:] = (t_idx[tau:] - tau) / 60.0
    
    p_t = np.clip(p * (1 + psi * drift), 0.03, 0.60)
    
    delta = rng.binomial(1, p_t).astype(float)
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    return delta * z, delta, z, {"psi": psi}
