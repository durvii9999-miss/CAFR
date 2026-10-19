"""Monitor features (Rev 2 §6.3, handoff §7.4)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

__all__ = [
    "compute_bias_stats",
    "compute_coverage_stats",
    "compute_occurrence_stats",
    "compute_occurrence_gap",
    "conformal_interval",
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

def compute_occurrence_gap(
    delta: np.ndarray,
    z_hat: np.ndarray,
    usable: np.ndarray,
    min_evidence: int = 10,
) -> dict:
    """C3's calibrated occurrence gap: realized rate minus the model's own ``z_hat``.

    **Why this replaces the fixed 0.5 "false zero" threshold.** On this panel the
    occurrence probability is ``p in [0.05, 0.50]`` by construction, so ``z_hat < 0.5``
    is true on *every* period of an intermittent SKU: the measured false-zero rate was
    **1.000 on the C0 control**, which made C3 fire on the control by construction.
    A threshold can only discriminate when both sides of it are reachable.

    The gap is the same object C3's injection moves: a step in ``p`` leaves the
    forecaster's occurrence estimate behind for as long as it takes to adapt, so the
    realized rate sits above (or below) ``z_hat``. The binomial test uses the model's
    own mean occurrence probability as the null, which is the right null for
    "the occurrence forecast is calibrated".

    Returns ``occ_gap``, ``p_val``, ``n``. ``p_val`` is ``nan`` when the null is
    degenerate (``mean(z)`` at 0 or 1) or the evidence floor is unmet.
    """
    delta = np.asarray(delta, dtype="float64")
    z_hat = np.asarray(z_hat, dtype="float64")
    mask = np.asarray(usable, dtype=bool) & np.isfinite(z_hat)
    d, z = delta[mask], np.clip(z_hat[mask], 0.0, 1.0)
    n = int(d.size)
    if n < min_evidence:
        return {"occ_gap": float("nan"), "p_val": float("nan"), "n": n}

    p0 = float(np.mean(z))
    if not (0.0 < p0 < 1.0):
        return {"occ_gap": float("nan"), "p_val": float("nan"), "n": n}
    k = int(np.sum(d))
    return {
        "occ_gap": float(k / n - p0),
        "p_val": float(binomtest(k, n, p0).pvalue),
        "n": n,
    }


def conformal_interval(
    mu: float,
    residuals: np.ndarray,
    alpha: float = 0.90,
    *,
    scale: float = 1.0,
    min_evidence: int = 8,
) -> tuple[float, float]:
    """A two-sided one-period prediction interval from the arm's own residuals.

    **Why not the empirical 5 %/95 % quantiles.** C2's statistic is the realized
    coverage of this interval, so the interval must actually attain its nominal level
    on the control. Plug-in empirical quantiles do not: with ``n = 24`` calibration
    residuals and a zero-inflated, right-skewed error distribution, the empirical
    95th percentile is usually the second-largest draw, and the realized coverage
    measured **0.73-0.80 against a nominal 0.90 on the C0 control at Step 5** -- a
    bias the size of ``tau_cov`` itself, which made C2 fire on control SKUs.

    Conformal order statistics give finite-sample marginal coverage under
    exchangeability: with ``k_lo = floor((n+1)(1-alpha)/2)`` and
    ``k_hi = ceil((n+1)(1-(1-alpha)/2))``, ``P(r_(k_lo) <= r_new <= r_(k_hi))`` is at
    least ``alpha`` and at most ``alpha + 2/(n+1)``. It is conservative by at most
    ``2/(n+1)`` (0.08 at ``n = 24``), which is the honest price of a 24-point
    calibration set and is shared by every SKU alike.

    ``scale`` is C2's per-SKU interval-rescale ``gamma``: the interval is
    ``mu + scale * (r_lo, r_hi)``.
    """
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < min_evidence or not np.isfinite(mu):
        return float("nan"), float("nan")
    r = np.sort(r)
    k_lo = max(1, int(np.floor((n + 1) * (1.0 - alpha) / 2.0)))
    k_hi = min(n, int(np.ceil((n + 1) * (1.0 - (1.0 - alpha) / 2.0))))
    return (
        float(mu + scale * r[k_lo - 1]),
        float(mu + scale * r[k_hi - 1]),
    )


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

def compute_cusum_stats(
    e: np.ndarray,
    sigma: float,
    k: float = 0.25,
    h: float = 4.0,
    *,
    burn_in: int = 0,
) -> dict:
    """C5: Two-sided CUSUM, reset after an alarm and start after burn-in."""
    e = np.asarray(e, dtype="float64")[max(0, int(burn_in)) :]
    if len(e) == 0 or sigma <= 1e-9:
        return {
            "S_pos": 0.0, "S_neg": 0.0, "alarm": False,
            "max_S": 0.0, "final_S": 0.0, "alarm_score": 0.0,
        }
        
    S_pos = 0.0
    S_neg = 0.0
    diagnostic_pos = 0.0
    diagnostic_neg = 0.0
    alarm = False
    alarm_score = 0.0
    
    for val in e:
        diagnostic_pos = max(0.0, diagnostic_pos + val - k * sigma)
        diagnostic_neg = max(0.0, diagnostic_neg - val - k * sigma)
        S_pos = max(0.0, S_pos + val - k * sigma)
        S_neg = max(0.0, S_neg - val - k * sigma)
        current = max(S_pos, S_neg)
        alarm_score = max(alarm_score, current)
        if current > h * sigma:
            alarm = True
            S_pos = 0.0
            S_neg = 0.0

    return {
        "S_pos": S_pos,
        "S_neg": S_neg,
        "alarm": bool(alarm),
        "max_S": max(diagnostic_pos, diagnostic_neg) / max(sigma, 1e-9),
        "final_S": max(S_pos, S_neg) / max(sigma, 1e-9),
        "alarm_score": alarm_score / max(sigma, 1e-9),
    }

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
