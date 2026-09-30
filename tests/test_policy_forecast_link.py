"""The policy must read the forecast — see ``docs/17_policy_forecast_channel_severed.md``.

``aggregate_order_up_to`` computes ``S = R*mu_hat + Q_alpha(aggregates - R*mu_hat)``.
Sample quantiles are translation-equivariant, so ``Q_alpha(x - c) = Q_alpha(x) - c`` and
the ``R*mu_hat`` cancels EXACTLY: ``S == Q_alpha(aggregates)``, a function of the
observed history alone. The forecaster therefore has no influence on the order-up-to
level, and arms (a), (b) and (c) are the same policy under three names.

The first test below states the REQUIREMENT ("S responds to the forecast"). It is
``xfail(strict=True)`` because the requirement is not met today: it fails now, and when
the estimator is fixed it will XPASS, which strict mode turns into a failure — forcing
whoever fixes it to delete the marker rather than leave a stale xfail behind.

The other two tests are not about the bug. They pin the surrounding behaviour that must
survive the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from cafr.forecasters.registry import build_pool
from cafr.sim.inventory import aggregate_order_up_to, rolling_aggregates
from cafr.utils.config import load_config

R = 3
WINDOW = 24


def _intermittent(seed: int = 0, n: int = 120, frac: float = 1.0 / 3.0) -> np.ndarray:
    """An intermittent series with roughly ``frac`` nonzero periods."""
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * frac)))
    y = np.zeros(n, dtype="float64")
    idx = rng.choice(n, size=k, replace=False)
    y[idx] = rng.gamma(2.0, 10.0, size=k)
    return y


@pytest.mark.xfail(
    strict=True,
    reason=(
        "docs/17: R*mu_hat cancels in S = R*mu_hat + Q_alpha(aggregates - R*mu_hat), so "
        "the order-up-to level ignores the forecast and every arm is the same policy. "
        "Delete this marker when the estimator is fixed."
    ),
)
def test_order_up_to_level_responds_to_the_forecast():
    """A 1000x change in mu_hat must move S *materially*. Today it does not move at all.

    The bar is relative, not exact: the cancellation is exact in real arithmetic but
    leaves ~1e-16 of floating-point noise, so ``S`` differs in the last bits and a bare
    ``len(set(...)) > 1`` would pass for the wrong reason. A forecast that changes a
    level by one part in 10^16 has not been read.
    """
    y = _intermittent()
    levels = [
        aggregate_order_up_to(y, mu, R, 0.95, window=WINDOW)[0]
        for mu in (0.5, 5.0, 50.0, 500.0)
    ]
    spread = (max(levels) - min(levels)) / max(abs(v) for v in levels)
    assert spread > 1e-6, (
        f"S moved by only {spread:.2e} relative when mu_hat moved 1000x: {levels}. "
        "The policy is reading the history, not the forecast."
    )


def test_the_cancellation_is_the_whole_story():
    """S is exactly the raw empirical aggregate quantile — the mechanism, not a symptom.

    This documents WHY the requirement above fails. It must be updated (not deleted) by
    whoever fixes §5 of docs/17. The tolerance is 1e-12 rather than exact because the
    cancellation leaves ~1e-16 of floating-point noise -- which is also why the
    requirement above is stated as a relative spread and not as exact equality.
    """
    y = _intermittent()
    S, _, _, fallback = aggregate_order_up_to(y, 7.0, R, 0.95, window=WINDOW)
    assert not fallback, "this series has plenty of aggregates; the fallback is not in play"
    agg = rolling_aggregates(y, R, WINDOW)
    # alpha_eff is capped by the nonzero count in the trailing window.
    n_cycles = int(np.count_nonzero(y[-WINDOW:]))
    alpha_eff = min(0.95, min(0.999, 1.0 - 1.0 / (n_cycles + 1.0)))
    expected = float(np.quantile(agg, alpha_eff, method="linear"))
    assert S == pytest.approx(expected, rel=1e-12), (
        "S is no longer the bare empirical quantile -- docs/17 may be fixed; re-read it"
    )


def test_the_fallback_branch_still_reads_the_forecast():
    """The one path that does depend on mu_hat, and the reason arms are not always equal.

    With fewer than ``min_aggregates`` complete aggregates the level is ``R * mu_hat``.
    That branch must keep working, because it is what a short-history SKU gets.
    """
    y = _intermittent(n=5)          # too short for min_aggregates complete aggregates
    S, alpha_eff, n_agg, fallback = aggregate_order_up_to(
        y, 3.0, R, 0.95, window=WINDOW, min_aggregates=4
    )
    assert fallback is True
    assert n_agg < 4
    assert S == pytest.approx(R * 3.0)
    assert alpha_eff == 0.95, "the fallback reports the REQUESTED alpha, uncapped"


def test_the_alpha_axis_of_the_frontier_is_live():
    """The fix must not break alpha sensitivity: the frontier's horizontal axis works.

    Measured today: mean S moves 8.04 -> 11.57 across alpha 0.70 -> 0.99 on RUF. If this
    breaks, the cost-service frontier collapses to a point.
    """
    y = _intermittent()
    s_lo = aggregate_order_up_to(y, 10.0, R, 0.70, window=WINDOW)[0]
    s_hi = aggregate_order_up_to(y, 10.0, R, 0.99, window=WINDOW)[0]
    assert s_hi > s_lo, f"S did not increase with alpha: {s_lo} -> {s_hi}"


def test_pool_construction_is_not_the_cause():
    """Guard against a tempting misdiagnosis: the arms share a pool by design.

    The arms producing identical results is NOT because they share a pool of fitted
    objects -- ``fit()`` is pure and returns a fresh state. It is the cancellation.
    """
    cfg = load_config("base.yaml")
    pool = build_pool(cfg)
    y = _intermittent(n=30)
    from cafr.forecasters.base import FittingInput

    a = pool["croston"].fit(FittingInput(y, np.zeros(30, bool)))
    b = pool["tsb"].fit(FittingInput(y, np.zeros(30, bool)))
    assert a.mu != b.mu, "two different models returned the same mu; the pool is broken"
