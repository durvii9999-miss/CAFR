"""Detectability budgets for the injected panel (Rev 2 §15.2, constraints 3, 4 and 9).

A cause may only be claimed as *detectable* if the injected effect clears the
evidence its detector actually has. These three checks are the difference between a
panel that tests CAFR and a panel that tests its own injection design. Failing a
budget is never a reason to relax it: the SKU is flagged and excluded from the
budget-restricted recall of §29.3.
"""

from __future__ import annotations

import numpy as np

__all__ = ["check_c5_budget", "check_c4_budget", "check_c6_learnability", "check_c6_budget"]


def check_c5_budget(delta_sigma: float, p: float) -> bool:
    """Constraint 3 -- can the CUSUM see this step?

    ``delta_sigma`` is the step in units of the sd of the targeted sub-process
    (occurrence: ``|p_new - p| / sqrt(p(1-p))``; size: ``|mu_z_new - mu_z| / sigma_z``).

    Revision 1 injected ``0.50 sigma`` against a required ``1.33 sigma``.
    """
    k = 0.25
    h = 4.0
    W_detect = 30

    req = k + h / (p * W_detect)
    return float(delta_sigma) >= req


def check_c4_budget(n_nonzero: int, min_evidence: int) -> bool:
    """Constraint 9 -- are there enough nonzero sizes to estimate a dispersion ratio?

    C4's statistic is a ratio of log-scale dispersions, so it needs nonzero sizes in
    BOTH the evaluation window and the baseline window. At ``W_sz = 24`` and
    ``min_evidence = 8`` this requires ``p >= 0.333``, so **high-ADI SKUs cannot clear
    the floor from a 24-period window at all** (``p = 0.10`` yields 2.4 nonzero sizes).

    That is a real property of the statistic, not a defect to hide: such SKUs are
    flagged ``below_detection_budget`` and reported as *not evaluable at W_sz*, while
    the detector's AUC and recall are reported on the budget-restricted subset. The
    trade-off is recorded in Rev 2 §15.2 constraint 9 -- lengthening ``W_sz`` to 60
    raises the evaluable share of the panel from 35 % to 73 % at the cost of a slower
    detector.
    """
    return int(n_nonzero) >= int(min_evidence)


def check_c6_learnability(auc_c6: float, auc_c0: float, tol: float = 0.10) -> bool:
    """Constraint 4 -- is C6 still learnable, or has it degenerated into C3?

    ``AUC(C6 series) >= AUC(C0 series) - tol``. A drift in ``p`` that destroys the
    occurrence signal turns C6's panel into a second C3 panel and would show up later
    as attribution confusion that says nothing about CAFR.
    """
    return float(auc_c6) >= float(auc_c0) - float(tol)


def check_c6_budget(
    p_init: float,
    p_target: float,
    *,
    w_class: int = 30,
    adi_star: float = 4.0,
    min_sigma: float = 2.0,
) -> dict:
    """Heuristic detectability calculation for C6 intermittency transition.

    HEURISTIC METADATA ONLY: This calculation is a stylized Gaussian approximation
    on expected counts in a w_class window, NOT a finite-sample statistical guarantee
    (it does not account for joint CV^2 stability, transient forecaster lag, or finite-sample
    bootstrap variance). It is recorded purely as diagnostic design metadata. It must
    NEVER filter SKUs, change H-5 eligibility, alter monitor behavior, or affect attribution.

    Checks:
    1. Theoretical crossing: p_init and p_target strictly straddle p_star = 1.0 / adi_star (0.25).
       If either value equals p_star, the condition is not strictly crossing.
    2. Destination separation: In a w_class window, expected nonzeros mu_N = w_class * p_target
       has distance from the boundary N_star = w_class * p_star of at least min_sigma standard deviations:
       z_sep = |mu_N - N_star| / sqrt(w_class * p_target * (1 - p_target)) >= min_sigma.
    """
    try:
        p_init_f = float(p_init)
        p_target_f = float(p_target)
        w_class_i = int(w_class)
        adi_star_f = float(adi_star)
        min_sigma_f = float(min_sigma)
    except (TypeError, ValueError):
        return {
            "p_init": float("nan"),
            "p_target": float("nan"),
            "p_star": float("nan"),
            "crosses": False,
            "mu_N": 0.0,
            "N_star": 0.0,
            "sigma_N": 0.0,
            "z_sep": 0.0,
            "cleared": False,
        }

    if (
        not np.isfinite(p_init_f)
        or not np.isfinite(p_target_f)
        or not np.isfinite(adi_star_f)
        or not np.isfinite(min_sigma_f)
        or p_init_f <= 0.0
        or p_init_f >= 1.0
        or p_target_f <= 0.0
        or p_target_f >= 1.0
        or adi_star_f <= 0.0
        or w_class_i <= 0
        or min_sigma_f < 0.0
    ):
        p_star_safe = float(1.0 / adi_star_f) if np.isfinite(adi_star_f) and adi_star_f > 0 else float("nan")
        return {
            "p_init": p_init_f,
            "p_target": p_target_f,
            "p_star": p_star_safe,
            "crosses": False,
            "mu_N": 0.0,
            "N_star": 0.0,
            "sigma_N": 0.0,
            "z_sep": 0.0,
            "cleared": False,
        }

    p_star = 1.0 / adi_star_f
    # Strict straddling: product must be strictly negative
    crosses = bool((p_init_f - p_star) * (p_target_f - p_star) < 0.0)

    mu_N = float(w_class_i * p_target_f)
    N_star = float(w_class_i * p_star)
    var_N = float(w_class_i * p_target_f * (1.0 - p_target_f))
    sigma_N = float(np.sqrt(var_N)) if var_N > 0 else 0.0
    z_sep = float(abs(mu_N - N_star) / sigma_N) if sigma_N > 0 else 0.0

    clears = bool(crosses and z_sep >= min_sigma_f)
    return {
        "p_init": p_init_f,
        "p_target": p_target_f,
        "p_star": p_star,
        "crosses": crosses,
        "mu_N": mu_N,
        "N_star": N_star,
        "sigma_N": sigma_N,
        "z_sep": z_sep,
        "cleared": clears,
    }
