"""Detectability budgets for the injected panel (Rev 2 §15.2, constraints 3, 4 and 9).

A cause may only be claimed as *detectable* if the injected effect clears the
evidence its detector actually has. These three checks are the difference between a
panel that tests CAFR and a panel that tests its own injection design. Failing a
budget is never a reason to relax it: the SKU is flagged and excluded from the
budget-restricted recall of §29.3.
"""

from __future__ import annotations

__all__ = ["check_c5_budget", "check_c4_budget", "check_c6_learnability"]


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
