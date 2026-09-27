"""Evaluation, statistics, figures and tables. **Not implemented -- Steps 11-13.**

This package is the ONLY one allowed to open ``sku_attributes.parquet`` and the
synthetic ``ground_truth.parquet``. The policy, the monitor and every arm must be
unable to reach them -- ``adi_full`` / ``cv_squared_full`` are computed over all 60
periods and would hand the detector the post-regime answer, which is a direct
ground-truth leak, worse than a look-ahead.

Planned contents
----------------
``metrics.py``    cost, CSL, fill rate, PIS, NOS; the churn counts
``stats.py``      paired bootstrap clustered by SKU AND by generator parameter cell;
                  Wilson intervals (not the normal approximation); Holm-Bonferroni;
                  the ROPE at 2 %
``figures.py``    P1-P11
``tables.py``     T1-T8
``diagnostics.py`` the seven mandatory diagnostics

Sign convention, used without exception
---------------------------------------
The reported difference between arms X and Y is ``cost(X) - cost(Y)``.
**Positive means X is worse.** So ``(c) - (f) > 0`` means (f) is better.

Pre-registration
----------------
The success criteria in Rev 2 §29 are FROZEN before the single test-split run. The
test panel is confirmatory only; nothing is tuned on it.

What must be reported even when it is unwelcome
-----------------------------------------------
* every swept alpha, not only the favourable one
* the seed sweep (>= 4 of 5 in the same direction)
* the +/-20 % threshold perturbation
* per-cause recall TWICE: pooled, and restricted to SKUs where the C5 detectability
  budget was met. The budget-restricted numbers are the methodologically meaningful
  ones; the pooled numbers are reported for completeness.
* realised churn for every arm, so "matched churn" is checkable rather than asserted

Doing-when: handoff §13 Steps 11-13.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Steps 11-13"
