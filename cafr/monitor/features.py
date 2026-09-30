"""Monitor features (Rev 2 §6.3, handoff §7.4)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

__all__ = [
    "compute_bias_stats",
    "compute_coverage_stats",
    "compute_occurrence_stats",
    "compute_dispersion_stats",
    "compute_cusum_stats",
    "compute_intermittency_stats",
]

def robust_sd(x: np.ndarray) -> float:
    """MAD-based robust standard deviation."""
    if len(x) < 2:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(mad * 1.4826, 1e-9))

def compute_bias_stats(e_sz: np.ndarray, usable: np.ndarray, min_evidence: int = 8) -> dict:
    """C1: b^sz, z_b."""
    if len(e_sz) != len(usable):
        raise ValueError("e_sz and usable must have same length")
        
    mask = usable & ~np.isnan(e_sz)
    e_val = e_sz[mask]
    n = len(e_val)
    
    if n < min_evidence:
        return {"b_sz": float("nan"), "z_b": float("nan"), "n_usable": n}
        
    b_sz = float(np.mean(e_val))
    # Standardised bias
    sigma = robust_sd(e_val)
    z_b = b_sz / (sigma / np.sqrt(n)) if sigma > 1e-9 else 0.0
    
    return {"b_sz": b_sz, "z_b": z_b, "n_usable": n}

def compute_coverage_stats(covered: np.ndarray, alpha: float, min_evidence: int = 30) -> dict:
    """C2: Coverage c_hat."""
    n = len(covered)
    if n < min_evidence:
        return {"c_hat": float("nan"), "c_gap": float("nan"), "p_val": float("nan")}
        
    c_hat = float(np.mean(covered))
    c_gap = c_hat - alpha
    
    # Binomial test
    k = int(np.sum(covered))
    p_val = binomtest(k, n, alpha).pvalue
    
    return {"c_hat": c_hat, "c_gap": c_gap, "p_val": float(p_val)}

def compute_occurrence_stats(delta: np.ndarray, z_hat: np.ndarray, usable: np.ndarray, min_evidence: int = 10) -> dict:
    """C3: Occurrence AUC and False Zero rate."""
    mask = usable
    d = delta[mask]
    z = z_hat[mask]
    
    n_pos = int(np.sum(d))
    if n_pos < min_evidence or n_pos == len(d):
        return {"auc": float("nan"), "fz": float("nan")}
        
    try:
        auc = float(roc_auc_score(d, z))
    except ValueError:
        auc = float("nan")
        
    fz = float(np.mean(z[d == 1] < 0.5)) if n_pos > 0 else float("nan")
    
    return {"auc": auc, "fz": fz}

def compute_dispersion_stats(
    y_now: np.ndarray,
    y_base: np.ndarray,
    *,
    min_evidence: int = 8,
    usable: np.ndarray | None = None,
) -> dict:
    """C4: size-dispersion ratio ``r``, measured on the LOG scale.

    **Why the log scale (Rev 2 §6.3, revised at Step 4).** C4 is a change in the
    dispersion of nonzero sizes at *fixed mean* (§15.2, constraint 2). For a Gamma
    size distribution ``z ~ Gamma(k, theta)`` the mean is ``k*theta`` and the sd is
    ``sqrt(k)*theta`` -- **proportional to the mean**. So *any* absolute dispersion
    measure is moved by any level intervention, which makes C4's statistic
    confounded by construction: measured on the pre-revision absolute ratio, C5's
    size step scored AUC **0.805** on C4's own statistic and C1's mean ramp scored
    **0.615** (gate 4.4 needs <= 0.65).

    The log scale removes the nuisance scale exactly::

        log z  =  log theta  +  log Gamma(k, 1)

    so ``sd(log z) = sqrt(trigamma(k))`` depends on the shape ``k`` alone and is
    **exactly invariant to theta**, i.e. to every level cause (C1's ramp, C5's size
    step, and the mean change C6 declares in §15.2). Measured after the change, at
    ``W_sz = 24`` and ``min_evidence = 8``: C4 detection AUC **0.939** and every
    confound <= 0.572.

    The statistic is computed on the **observed nonzero demand**, so it needs no
    forecast and no ground truth. Under ``observation_model: censored_sales`` the
    nonzero sizes are truncated by stock-outs; the C-guard (§6.3) is what suppresses
    C4 in that regime, not this function.

    ``below_budget`` reports constraint 9: with fewer than ``min_evidence`` nonzero
    observations in the window the ratio is not estimable and the SKU must be
    *flagged* (and excluded from the budget-restricted recall of §29.3), never
    silently dropped.
    """
    mask_now = np.ones(len(y_now), dtype=bool) if usable is None else usable
    a = np.asarray(y_now, dtype="float64")[mask_now]
    a = a[a > 0]
    b = np.asarray(y_base, dtype="float64")
    b = b[b > 0]

    n_now, n_base = int(a.size), int(b.size)
    below = n_now < min_evidence or n_base < min_evidence
    if below:
        return {
            "sd_log_sz": float("nan"),
            "r": float("nan"),
            "n_now": n_now,
            "n_base": n_base,
            "below_budget": True,
        }

    sd_now = float(np.log(a).std(ddof=1))
    sd_base = float(np.log(b).std(ddof=1))
    r = sd_now / sd_base if sd_base > 1e-9 else float("nan")
    return {
        "sd_log_sz": sd_now,
        "r": r,
        "n_now": n_now,
        "n_base": n_base,
        "below_budget": False,
    }

def compute_cusum_stats(e: np.ndarray, sigma: float, k: float = 0.25, h: float = 4.0) -> dict:
    """C5: Two-sided CUSUM."""
    if len(e) == 0 or sigma <= 1e-9:
        return {"S_pos": 0.0, "S_neg": 0.0, "alarm": False, "max_S": 0.0}
        
    S_pos = 0.0
    S_neg = 0.0
    alarm = False
    
    for val in e:
        S_pos = max(0.0, S_pos + val - k * sigma)
        S_neg = max(0.0, S_neg - val - k * sigma)
        if max(S_pos, S_neg) > h * sigma:
            alarm = True
            
    return {"S_pos": S_pos, "S_neg": S_neg, "alarm": alarm, "max_S": max(S_pos, S_neg) / max(sigma, 1e-9)}

def compute_intermittency_stats(delta: np.ndarray, usable: np.ndarray) -> dict:
    """C6: ADI."""
    mask = usable
    d = delta[mask]
    n = len(d)
    nz = int(np.sum(d))
    
    if nz == 0:
        return {"adi": float(n)}
        
    adi = n / nz
    return {"adi": adi}
