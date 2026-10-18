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

def _guard_ok(value: float, bar: float) -> bool:
    """A guard statistic passes when it is *not estimable* as well as when it is small.

    Guards exist to stop one cause reading another cause's signature: C2 must not fire
    when a size change has moved ``r``, C3 must not fire when ``r`` moved, C7 must not
    fire when the forecast itself is biased. A NaN is **no evidence of a size change**,
    not evidence of one -- but ``nan > 1.5`` is False and ``nan <= 1.5`` is also False,
    so a bare comparison silently vetoes the rule on every SKU whose dispersion budget
    is unmet (measured at Step 5: ``r`` was estimable on only 7-18 % of checks, which
    vetoed C2, C3 and C7 on the remaining 80 %+ before any threshold was tested).
    Treating an unestimable guard as passing is the honest reading: the guard has
    nothing to object to.
    """
    return (not np.isfinite(value)) or value <= bar


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
    3. |z_b| <= 1.96   (a lifted mean must not read as bad width)
    4. r <= 1.5        (a size change must not read as bad width)
    """
    if np.isnan(c_gap) or np.isnan(p_val):
        return False

    if abs(c_gap) <= tau_cov:
        return False
    if p_val >= 0.05:
        return False

    if not _guard_ok(abs(z_b) if np.isfinite(z_b) else np.nan, 1.96):
        return False
    if not _guard_ok(r, 1.5):
        return False

    return True

def check_c3_occurrence(
    occ_gap: float,
    p_val: float,
    r: float,
    tau_occ: float = 0.10,
) -> bool:
    """C3: occurrence failure -- the occurrence forecast is mis-calibrated.

    **Restated at Step 5 (2026-10-18).** The pre-Step-5 rule was
    ``AUC < 0.65 or FZ > 0.30`` with ``FZ = mean(z_hat < 0.5 | delta = 1)``. On this
    panel ``z_hat in [0.05, 0.50]``, so ``FZ`` measured **1.000 on the C0 control** --
    the threshold was above the entire support of the statistic. The replacement is the
    calibrated occurrence gap ``mean(delta) - mean(z_hat)`` with a binomial test using
    the model's own ``mean(z_hat)`` as the null. It measures the same claim ("the
    occurrence prediction fails") on a scale where both sides of the threshold are
    reachable, and it is the object C3's injection (a step in ``p``) actually moves.
    The AUC is still computed and reported, but as a diagnostic, not as a trigger: an
    occurrence-rate forecast has AUC ~0.5 by construction and cannot separate causes.

    1. ``|occ_gap| > tau_occ``
    2. binomial ``p_val < 0.05``
    3. ``r <= 1.5`` (a size change must not read as an occurrence change)
    """
    if np.isnan(occ_gap) or np.isnan(p_val):
        return False

    if abs(occ_gap) <= tau_occ:
        return False
    if p_val >= 0.05:
        return False
    if not _guard_ok(r, 1.5):
        return False

    return True


def check_c4_size(
    r_history: list[float],
    auc: float,
) -> bool:
    """C4: Size failure -- the dispersion of nonzero sizes has changed.

    1. the dispersion ratio is out of band over 2 consecutive windows
    2. AUC >= 0.65

    **The band is two-sided.** ``r = sd(log z)_now / sd(log z)_base``, so C4's
    injection (``k -> k * factor`` with ``factor in {0.25, 4.0}``) moves ``r`` in
    *either* direction: ``k -> k/4`` raises the log-dispersion by ~2x, ``k -> 4k``
    lowers it by ~0.47x. The pre-Step-5 rule tested only ``r > 1.5``, which is
    blind to half of C4's own injections by construction -- measured at Step 5,
    ``c4_r`` crossed on **0 %** of C4 checks even though the log-dispersion had
    moved by a factor of two. Comparing ``max(r, 1/r)`` to 1.5 tests "the dispersion
    changed by more than 50 %", which is the claim C4 actually makes.
    """
    if len(r_history) < 2:
        return False
    lo, hi = r_history[-2], r_history[-1]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    if lo <= 0 or hi <= 0:
        return False
    if not (max(lo, 1.0 / lo) > 1.5 and max(hi, 1.0 / hi) > 1.5):
        return False

    if not np.isnan(auc) and auc < 0.65:
        return False

    return True

def check_c5_cusum(
    alarm_occ: bool,
    alarm_sz: bool,
    step_bic: float,
    drift_bic: float,
    occ_step_bic: float | None = None,
    occ_drift_bic: float | None = None,
) -> bool:
    """C5: Level/regime shift.

    1. Either CUSUM alarms
    2. The step model beats the drift model on BIC **for the sub-process that alarmed**

    The second condition is evaluated on the occurrence residual when the occurrence
    CUSUM is the trigger and on the size residual when the size CUSUM is. A step in
    ``p`` leaves the size residual untouched, so a size-BIC test cannot arbitrate an
    occurrence alarm; requiring it made C5's firing depend on an unrelated statistic.
    """
    if not (alarm_occ or alarm_sz):
        return False

    ok = False
    if alarm_occ and occ_step_bic is not None and occ_drift_bic is not None:
        if np.isfinite(occ_step_bic) and np.isfinite(occ_drift_bic):
            ok = ok or bool(occ_step_bic < occ_drift_bic)
    if alarm_sz:
        ok = ok or bool(step_bic < drift_bic)

    return ok

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
    if np.isnan(csl_gap):
        return False

    if csl_gap > -tau_svc:
        return False

    # Acceptability guard: the forecast must look healthy. Unestimable statistics are
    # not evidence of a bad forecast (see ``_guard_ok``).
    if not _guard_ok(abs(z_b) if np.isfinite(z_b) else np.nan, 1.96):
        return False
    if not _guard_ok(r, 1.5):
        return False

    return True
