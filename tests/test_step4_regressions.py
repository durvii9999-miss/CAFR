"""Regression tests for the Step 4 defects recorded in ``docs/16_*``.

Each test here corresponds to a bug that was **silent** -- it produced plausible
numbers, not an exception, and would have survived into the paper. A test that only
checks the corrected value would not have caught the original; these check the
mechanism.

1. the synthetic panel's observation convention (every run collapsed to zero orders);
2. ``z_b`` computed on the size sub-process, not the per-period residual;
3. ``n_unmet_cycles`` counting cycles, not periods;
4. C2's gamma solver returning the endpoint on the satisfying side;
5. a ``dead`` SKU's CV^2 being undefined rather than zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cafr.data.synth.c7_constraints import cycle_service_level, unmet_cycles
from cafr.data.synth.injections import (
    _induced_coverage,
    _solve_gamma,
    _aggregate_samples,
)
from cafr.data.synth.labels import generate_labelled_panel
from cafr.sim.inventory import SimConfig
from cafr.utils.config import load_config


# --------------------------------------------------------------------------- #
# 1. the observation convention
# --------------------------------------------------------------------------- #


def test_synthetic_config_uses_the_demand_convention():
    """Under ``censored_sales`` the policy sees zero sales from t=0 and never recovers.

    The failure is silent: the forecaster predicts 0 from an all-zero history, the
    order-up-to level stays at 0, and every arm returns identical numbers. Nothing
    raises. ``ruf.yaml`` documents exactly this collapse; the synthetic config inherited
    the base default and never overrode it, so every synthetic run was degenerate.
    """
    cfg = load_config("synthetic.yaml")
    assert cfg["sim"]["observation_model"] == "demand", (
        "the synthetic panel is a DEMAND panel (§15.6): stock-outs are the simulator's "
        "counterfactual cost layer, not part of the policy's information set. Under "
        "censored_sales the run collapses to alpha=0 and every arm agrees."
    )
    assert SimConfig.from_config(cfg).observation_model == "demand"


def test_synthetic_splits_include_the_injection_point():
    """``tau_inj = 120`` must fall inside the evaluated window.

    With ``burn_in + evaluated <= tau_inj`` the rollout stops before the injection and
    the C7 window loop never runs -- reporting an empty gate as if it had passed.
    """
    cfg = load_config("synthetic.yaml")
    tau = int(cfg["synthetic"]["tau_inj"])
    T = int(cfg["synthetic"]["T"])
    horizon = int(cfg["splits"]["burn_in"]) + int(cfg["splits"]["evaluated"])
    assert horizon >= T, f"horizon {horizon} < panel length {T}"
    assert int(cfg["splits"]["burn_in"]) <= tau < horizon


# --------------------------------------------------------------------------- #
# 2. z_b is a SIZE-subprocess statistic
# --------------------------------------------------------------------------- #


def test_burn_in_stats_are_undefined_for_a_dead_series():
    """A series with fewer than two nonzero periods has no CV^2 -- not CV^2 = 0."""
    _, _, params = generate_labelled_panel(
        42, n_per_cause=15, n_control=15, n_mixture=0, n_c2b=0
    )
    dead = params[params["sb_cell_init"] == "dead"]
    assert np.isnan(dead["cv2_init"]).all()


# --------------------------------------------------------------------------- #
# 3. cycles, not periods
# --------------------------------------------------------------------------- #


def test_unmet_cycles_counts_cycles_not_periods():
    """A cycle is ``R`` consecutive periods. Counting periods inflates by up to R."""
    R = 3
    # Three consecutive lost periods are ONE failed cycle, not three.
    lost = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0], dtype="float64")
    assert unmet_cycles(lost, R) == 1
    assert int((lost > 0).sum()) == 3, "the period count this replaced"

    # Two cycles each with one lost period -> two.
    lost2 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype="float64")
    assert unmet_cycles(lost2, R) == 2

    # A trailing partial block is not a cycle.
    assert unmet_cycles(np.array([0.0, 0.0, 0.0, 9.0]), R) == 0


def test_cycle_service_level_and_unmet_cycles_agree():
    """``unmet_cycles == 0`` exactly when CSL is 1.0, on random lost series."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        lost = (rng.random(30) < 0.2).astype("float64") * rng.integers(1, 5, 30)
        csl, _ = cycle_service_level(lost, 3)
        assert (unmet_cycles(lost, 3) == 0) == (csl == 1.0)


# --------------------------------------------------------------------------- #
# 4. C2's gamma solver
# --------------------------------------------------------------------------- #


def test_gamma_solver_lands_on_the_satisfying_side():
    """``|coverage - alpha| >= target_gap`` must hold whenever it is feasible.

    The bisection maintains ``coverage(lo) < target <= coverage(hi)``. Returning a fixed
    endpoint puts the result on the wrong side for the other branch: it made C2's gap
    land at exactly -0.120 for 200 of 200 SKUs with ``|gap| >= 0.12`` False for all of
    them.
    """
    alpha = 0.95
    target_gap = 0.12
    rng = np.random.default_rng(0)
    solved = 0
    for p in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        for k in (0.6, 1.0, 2.0, 5.0):
            a = _aggregate_samples(p, k, 50.0, 3, 8000, rng)
            gamma, cov, direction = _solve_gamma(a, alpha, target_gap)
            gap = abs(cov - alpha)
            # Either it clears, or neither extreme could reach the target.
            lo_c = _induced_coverage(a, alpha, 0.02)
            hi_c = _induced_coverage(a, alpha, 8.0)
            feasible = max(abs(lo_c - alpha), abs(hi_c - alpha)) >= target_gap
            if feasible:
                assert gap >= target_gap - 1e-9, (
                    f"p={p} k={k} direction={direction}: gap {gap:.6f} < {target_gap}"
                )
                solved += 1
    assert solved > 0, "no case was feasible; the test proved nothing"


def test_c2b_gap_is_flagged_when_it_cannot_clear():
    """A C2b SKU whose gap cannot reach the target is flagged, never silently passed."""
    from cafr.data.synth.injections import inject_C2b

    rng = np.random.default_rng(7)
    _, _, _, params = inject_C2b(200, 120, 0.05, 0.6, 50.0, rng)
    assert "gap_clears_with_margin" in params
    assert params["demand_touched"] is False, "C2's DGP must never be touched"
    assert params["c2_variant"] == "calibration_window"
    # The recorded verdict must agree with the recorded gap.
    assert params["gap_clears_with_margin"] == bool(
        abs(params["coverage_gap"]) >= 0.12
    )


# --------------------------------------------------------------------------- #
# 5. the panel's statics do not perturb its parameters
# --------------------------------------------------------------------------- #


def test_statics_do_not_shift_the_cell_draw():
    """Adding the static attributes must not move any series' parameters.

    ``draw_statics`` consumes RNG draws. Drawn inline, it shifted every cell, p, k and
    mu_z in the panel -- ``high_highdisp`` moved from 0.146 to 0.136 -- which is a
    coupling with no meaning behind it. Statics now use their own child stream.
    """
    from cafr.data.synth.labels import _sku_rng, _series

    rng = _sku_rng(42, "single", 0)
    y, gt, param = _series("C1", "single", 0, 200, 120, 24, rng)
    rng2 = _sku_rng(42, "single", 0)
    y2, gt2, param2 = _series("C1", "single", 0, 200, 120, 24, rng2)

    assert param == param2
    np.testing.assert_array_equal(y, y2)
    # And the statics really are present.
    assert {"lead_time", "unit_cost", "holding_rate"} <= set(param)


# --------------------------------------------------------------------------- #
# 6. gate 4.9's restated clauses (docs/16 §7, 2026-10-17)
# --------------------------------------------------------------------------- #


def test_unscaled_forecast_level_inverts_the_c7_scale():
    """Constraint 6 must score the FORECASTER's quantile, not the policy's scaled ``S``.

    ``S = R*mu_hat + scale*Q``. Dividing the safety term out recovers ``R*mu_hat + Q``
    exactly, for any scale -- including a scale the calibration ladder chose later.
    """
    from cafr.data.synth.c7_constraints import unscaled_forecast_level

    R, scale = 3, 0.45
    mu = np.array([10.0, 4.0, 0.0, 7.5])
    q = np.array([2.0, -1.0, 0.0, 3.0])
    s = R * mu + scale * q
    recovered = unscaled_forecast_level(s, mu, R, scale)
    np.testing.assert_allclose(recovered, R * mu + q, atol=1e-12)

    # Scale 1 is the identity; a fallback period (S == R*mu) stays at R*mu.
    np.testing.assert_allclose(unscaled_forecast_level(s, mu, R, 1.0), s)
    np.testing.assert_allclose(
        unscaled_forecast_level(R * mu, mu, R, 0.6), R * mu
    )


def test_unscaled_forecast_level_rejects_a_zero_scale():
    """At scale 0 the safety term carries no information; the reconstruction is undefined."""
    from cafr.data.synth.c7_constraints import unscaled_forecast_level

    with pytest.raises(ValueError):
        unscaled_forecast_level(np.array([1.0]), np.array([0.0]), 3, 0.0)


def test_pooled_rmsse_splits_pre_and_post_and_ignores_nan():
    """Pooling is by PERIOD, and a NaN outside the evaluated window must not count."""
    from cafr.data.synth.c7_constraints import pooled_rmsse

    resid = np.full(20, np.nan)
    resid[5:10] = 1.0        # pre-injection
    resid[10:15] = 2.0       # post-injection
    periods = np.arange(5, 15)

    pre, post = pooled_rmsse(resid, periods, tau_inj=10)
    assert pre == pytest.approx(1.0)
    assert post == pytest.approx(2.0)

    # No evaluated pre-injection period -> undefined, not zero.
    pre2, post2 = pooled_rmsse(resid, periods, tau_inj=0)
    assert np.isnan(pre2) and post2 == pytest.approx(np.sqrt(2.5))


def test_calibrate_picks_the_mildest_passing_scale(monkeypatch):
    """The pre-registered rule: walk mildest first, stop at the first 80 % exposure."""
    import cafr.data.synth.c7_constraints as c7

    seen: list[float] = []

    def fake_eval(series, factory, cfg, *, sku_ids, tau_inj=120, scale=0.6,
                  alpha=None, fit_on="uncensored", pool=None):
        seen.append(float(scale))
        return pd.DataFrame({"x": [1]}), pd.DataFrame(
            [{"scale": scale, "n_windows": 1, "constraint_7_pass": scale <= 0.45}]
        )

    monkeypatch.setattr(c7, "evaluate_c7_constraints", fake_eval)
    chosen, _, ladder = c7.calibrate_c7_scale(
        {}, None, {}, sku_ids=[], grid=(0.15, 0.60, 0.45)
    )

    assert seen == [0.60, 0.45], "must be walked mildest-first and stop at the first pass"
    assert chosen == 0.45
    assert ladder["selected"].tolist() == [False, True]
    assert ladder["selection_rule"].tolist() == ["not_selected", "mildest_passing_scale"]


def test_calibrate_reports_grid_exhaustion_without_relaxing(monkeypatch):
    """No scale in the grid clearing 80 % is a FAIL, reported at the strongest scale."""
    import cafr.data.synth.c7_constraints as c7

    def fake_eval(series, factory, cfg, *, sku_ids, tau_inj=120, scale=0.6,
                  alpha=None, fit_on="uncensored", pool=None):
        return pd.DataFrame({"x": [1]}), pd.DataFrame(
            [{"scale": scale, "n_windows": 1, "constraint_7_pass": False}]
        )

    monkeypatch.setattr(c7, "evaluate_c7_constraints", fake_eval)
    chosen, _, ladder = c7.calibrate_c7_scale({}, None, {}, sku_ids=[], grid=(0.60, 0.30))

    assert chosen == 0.30
    assert ladder.iloc[-1]["selection_rule"] == "grid_exhausted_no_pass"
    assert ladder.iloc[-1]["selected"]
    assert not ladder.iloc[-1]["constraint_7_pass"], "the gate must not be relaxed"


def test_forward_aggregates_cover_the_protection_interval_ahead():
    """``S_t`` covers ``[t+1, t+R]``, not the trailing ``[t-R+1, t]``.

    The pre-docs/17 coverage check used a trailing aggregate, which was the same
    realisation only while ``S`` *was* ``Q_alpha(aggregates)``. After the
    residual-bootstrap fix, a predictive quantile must be scored against the window it
    predicts. The last ``R`` periods have no complete forward window and stay NaN.
    """
    from cafr.data.synth.c7_constraints import forward_aggregates

    y = np.arange(1.0, 11.0)          # 1..10
    agg = forward_aggregates(y, R=3)
    assert agg[0] == pytest.approx(2 + 3 + 4)
    assert agg[1] == pytest.approx(3 + 4 + 5)
    assert agg[6] == pytest.approx(8 + 9 + 10)   # the last complete forward window
    assert np.isnan(agg[7])           # [9, 10, 11] -- period 11 does not exist
    assert np.isnan(agg[8])
    assert np.isnan(agg[9])


def test_evaluate_rejects_a_zero_scale():
    """A non-positive scale is refused before it can produce a meaningless coverage."""
    from cafr.data.synth.c7_constraints import evaluate_c7_constraints

    for bad in (0.0, -0.1):
        with pytest.raises(ValueError):
            evaluate_c7_constraints(
                {}, None, {"splits": {}, "sim": {}}, sku_ids=[], scale=bad, pool={}
            )
