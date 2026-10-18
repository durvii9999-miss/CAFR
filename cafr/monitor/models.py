"""Step-vs-drift model selection for C1 and C5 (Rev 2 §6.3).

Both detectors need to tell a *step* from a *drift*, and they need opposite answers:

* C1 (persistent bias) is a **ramp**. ``check_c1_bias`` requires
  ``drift_bic <= step_bic`` -- the drift model must fit at least as well.
* C5 (regime shift) is a **step**. ``check_c5_cusum`` requires
  ``step_bic < drift_bic``.

The two models
--------------
step:   ``e_t = a + b * 1[t >= c]``     3 parameters (a, b, c)
drift:  ``e_t = a + b * t``             2 parameters

``c`` is profiled over an interior grid rather than estimated jointly; that is the
standard change-point profile likelihood and costs one extra parameter in the BIC
penalty, which is stated explicitly below. The Gaussian likelihood collapses to
``n * log(RSS / n)`` (the variance is profiled out), so the BIC is

    BIC = n * log(RSS / n) + k * log(n)

with ``RSS`` floored at ``1e-12`` so a perfect fit is finite rather than ``-inf``.

Nothing here reads a forecast or a cause: it is a pure function of an error series.
"""

from __future__ import annotations

import numpy as np

__all__ = ["step_drift_bic", "model_prefers_drift", "model_prefers_step"]

#: Interior change-point grid. Endpoints are excluded so the "step" cannot degenerate
#: into a one-parameter model at the first or last observation.
MIN_SPLIT_FRACTION = 0.20


def _bic(rss: float, n: int, k: int) -> float:
    rss = max(float(rss), 1e-12)
    return float(n * np.log(rss / n) + k * np.log(n))


def step_drift_bic(e: np.ndarray) -> tuple[float, float]:
    """Return ``(step_bic, drift_bic)`` for an error series.

    Both models are fitted to the same finite sample. A series shorter than
    ``max(8, 1/MIN_SPLIT_FRACTION)`` periods cannot support a profiled change point
    and returns ``(nan, nan)`` -- explicitly, so the rule that consumes it does not
    read a fabricated number.
    """
    e = np.asarray(e, dtype="float64")
    e = e[np.isfinite(e)]
    n = int(e.size)
    if n < 8:
        return float("nan"), float("nan")

    # -- drift: least squares on [1, t] ------------------------------------ #
    t = np.arange(n, dtype="float64")
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    rss_drift = float(np.sum((e - A @ coef) ** 2))
    bic_drift = _bic(rss_drift, n, 2)

    # -- step: profiled change point --------------------------------------- #
    lo = max(2, int(np.floor(MIN_SPLIT_FRACTION * n)))
    hi = min(n - 2, int(np.ceil((1.0 - MIN_SPLIT_FRACTION) * n)))
    if hi < lo:
        return float("nan"), bic_drift

    # Prefix/suffix sums give every split in O(n) rather than O(n^2).
    cs = np.concatenate([[0.0], np.cumsum(e)])
    cs2 = np.concatenate([[0.0], np.cumsum(e * e)])
    best_rss = np.inf
    for c in range(lo, hi + 1):
        n1, n2 = c, n - c
        s1, s2 = cs[c], cs[n] - cs[c]
        q1, q2 = cs2[c], cs2[n] - cs2[c]
        rss = (q1 - s1 * s1 / n1) + (q2 - s2 * s2 / n2)
        if rss < best_rss:
            best_rss = rss

    if not np.isfinite(best_rss):
        return float("nan"), bic_drift
    return _bic(best_rss, n, 3), bic_drift


def model_prefers_drift(e: np.ndarray) -> bool:
    """True when the drift model's BIC is no worse (C1's condition)."""
    step, drift = step_drift_bic(e)
    return bool(np.isfinite(step) and np.isfinite(drift) and drift <= step)


def model_prefers_step(e: np.ndarray) -> bool:
    """True when the step model's BIC is strictly better (C5's condition)."""
    step, drift = step_drift_bic(e)
    return bool(np.isfinite(step) and np.isfinite(drift) and step < drift)
