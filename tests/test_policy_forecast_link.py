"""The policy must read the forecast — see ``docs/17_policy_forecast_channel_severed.md``.

The original estimator was ``S = R*mu_hat + Q_alpha(aggregates - R*mu_hat)``. Sample
quantiles are translation-equivariant, so the ``R*mu_hat`` cancelled exactly and the
order-up-to level became a function of the observed history alone. Arms (a), (b) and (c)
were the same policy under three names.

The fix (``docs/17`` §5, option 1) is the residual-bootstrap predictive quantile::

    S = R * mu_hat_t  +  Q_alpha( sum_{k=0}^{R-1} e_{t-k} )

where ``e`` are the arm's own one-step-ahead residuals. The historical residuals were
produced by *different* past forecasts, so ``mu_hat`` does not cancel.

These tests are the regression guard for the fix. The first one is the requirement: a
1000x change in ``mu_hat`` must move ``S`` materially.
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


def test_order_up_to_level_responds_to_the_forecast():
    """A 1000x change in mu_hat must move S *materially*.

    The residuals are held fixed, so the only thing varying across the four levels is
    ``mu_hat``. The bar is relative, not exact: a forecast that changes a level by one
    part in 10^16 has not been read.
    """
    y = _intermittent()
    residuals = y - 5.0
    levels = [
        aggregate_order_up_to(y, mu, R, 0.95, residuals=residuals, window=WINDOW)[0]
        for mu in (0.5, 5.0, 50.0, 500.0)
    ]
    spread = (max(levels) - min(levels)) / max(abs(v) for v in levels)
    assert spread > 1e-6, (
        f"S moved by only {spread:.2e} relative when mu_hat moved 1000x: {levels}. "
        "The policy is reading the history, not the forecast."
    )


def test_safety_stock_is_the_quantile_of_residual_aggregates():
    """``S - R*mu_hat`` must be the alpha-quantile of the R-period residual sums.

    Computed independently of the library so this is not the code asserting itself.
    The residual aggregate ``sum_{k=0}^{R-1} e_{s-k}`` lives at the END of each
    R-period window; ``np.convolve(..., 'valid')`` is the same quantity.
    """
    y = _intermittent()
    mu = 7.0
    residuals = y - mu * 0.8          # a fixed residual history, independent of the mu below
    S, alpha_eff, _, fallback = aggregate_order_up_to(
        y, mu, R, 0.95, residuals=residuals, window=WINDOW
    )
    assert not fallback, "this series has plenty of aggregates; the fallback is not in play"

    resid_agg = np.array([
        float(residuals[i:i + R].sum()) for i in range(residuals.size - R + 1)
    ])[-WINDOW:]
    n_cycles = int(np.count_nonzero(y[-WINDOW:]))
    expected_alpha = min(0.95, min(0.999, 1.0 - 1.0 / (n_cycles + 1.0)))
    expected = R * mu + float(np.quantile(resid_agg, expected_alpha, method="linear"))

    assert alpha_eff == pytest.approx(expected_alpha)
    assert S == pytest.approx(expected, rel=1e-12), (
        "S is not R*mu_hat plus the residual-aggregate quantile — re-read docs/17 §5"
    )


def test_the_old_cancellation_is_gone():
    """The defect itself must stay gone: ``Q(x - c) == Q(x) - c`` no longer applies.

    ``S`` must NOT equal the bare empirical aggregate quantile (the severed-channel
    value). This is the negative control for the fix.
    """
    y = _intermittent()
    mu = 7.0
    residuals = y - mu * 0.8
    S, _, _, _ = aggregate_order_up_to(y, mu, R, 0.95, residuals=residuals, window=WINDOW)

    severed = float(np.quantile(rolling_aggregates(y, R, WINDOW), 0.95))
    assert abs(S - severed) > 1e-6 * max(abs(severed), 1.0), (
        f"S={S:.6f} still equals the model-free aggregate quantile "
        f"({severed:.6f}); the forecast channel is severed again"
    )


def test_residuals_are_required():
    """Forgetting the residuals must fail loudly, not silently restore the bug."""
    y = _intermittent()
    with pytest.raises(ValueError, match="residual"):
        aggregate_order_up_to(y, 7.0, R, 0.95, window=WINDOW)


def test_the_fallback_branch_still_reads_the_forecast():
    """The one path that does depend on mu_hat, and the reason arms are not always equal.

    With fewer than ``min_aggregates`` complete aggregates the level is ``R * mu_hat``.
    That branch must keep working, because it is what a short-history SKU gets.
    """
    y = _intermittent(n=5)          # too short for min_aggregates complete aggregates
    S, alpha_eff, n_agg, fallback = aggregate_order_up_to(
        y, 3.0, R, 0.95, residuals=np.zeros_like(y), window=WINDOW, min_aggregates=4
    )
    assert fallback is True
    assert n_agg < 4
    assert S == pytest.approx(R * 3.0)
    assert alpha_eff == 0.95, "the fallback reports the REQUESTED alpha, uncapped"


def test_the_alpha_axis_of_the_frontier_is_live():
    """The fix must not break alpha sensitivity: the frontier's horizontal axis works."""
    y = _intermittent()
    residuals = y - 5.0
    s_lo = aggregate_order_up_to(y, 10.0, R, 0.70, residuals=residuals, window=WINDOW)[0]
    s_hi = aggregate_order_up_to(y, 10.0, R, 0.99, residuals=residuals, window=WINDOW)[0]
    assert s_hi > s_lo, f"S did not increase with alpha: {s_lo} -> {s_hi}"


def test_pool_construction_is_not_the_cause():
    """Guard against a tempting misdiagnosis: the arms share a pool by design.

    The arms producing identical results was NOT because they share a pool of fitted
    objects -- ``fit()`` is pure and returns a fresh state. It was the cancellation.
    """
    cfg = load_config("base.yaml")
    pool = build_pool(cfg)
    y = _intermittent(n=30)
    from cafr.forecasters.base import FittingInput

    a = pool["croston"].fit(FittingInput(y, np.zeros(30, bool)))
    b = pool["tsb"].fit(FittingInput(y, np.zeros(30, bool)))
    assert a.mu != b.mu, "two different models returned the same mu; the pool is broken"
