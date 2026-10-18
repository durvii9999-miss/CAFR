"""Targeted correctness tests for the C2-only reliability-gated policy.

Five checks, matching the approved plan:

1. Closed gate is transparent: the run is identical to baseline.
2. Open gate resets C2's scale to 1.0 and leaves the point forecast untouched.
3. No look-ahead: the first outcome difference is at or after the first period a
   decision made at ``t`` can affect (``t + 1 + L``).
4. No action without a C2 attribution.
5. Fallback is logged once and the SKU is never re-queried.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "step6_c2_gated", ROOT / "experiments" / "step6_c2_gated.py"
)
step6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step6)

LOG_COLS = [
    "on_hand",
    "backorders",
    "order_qty",
    "demand_met",
    "demand_lost",
    "cost_period",
    "s_level",
    "alpha_t",
]


def _assert_logs_equal(a, b):
    for col in LOG_COLS:
        np.testing.assert_allclose(
            a[col].to_numpy(dtype=float),
            b[col].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            err_msg=f"column {col} differs",
        )


def _first_diff(x, y, atol=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = ~np.isclose(x, y, rtol=0.0, atol=atol, equal_nan=True)
    idx = np.flatnonzero(d)
    return int(idx[0]) if idx.size else None


@pytest.fixture(scope="module")
def ctx():
    cfg, readings, params, series = step6.load_context()
    model, tau, table, train, val = step6.fit_and_select(readings)
    pool = step6.build_pool(cfg, series, params)
    test = readings[readings["split"] == "test"]

    c2 = sorted(
        test.loc[
            (test["true_cause"] == "C2")
            & test["detected"]
            & (test["attributed_cause"] == "C2"),
            "sku_id",
        ]
    )[0]
    c0_non_c2 = sorted(
        test.loc[
            (test["true_cause"] == "C0")
            & test["detected"]
            & (test["attributed_cause"] == "C1"),
            "sku_id",
        ]
    )[0]
    c0_c2 = sorted(
        test.loc[
            (test["true_cause"] == "C0")
            & test["detected"]
            & (test["attributed_cause"] == "C2"),
            "sku_id",
        ]
    )[0]
    return {
        "cfg": cfg,
        "readings": readings,
        "params": params,
        "series": series,
        "model": model,
        "tau": tau,
        "pool": pool,
        "c2_id": c2,
        "c0_non_c2_id": c0_non_c2,
        "c0_c2_id": c0_c2,
    }


def _run(ctx, sid, mode, tau=None, trace=None):
    cfg = ctx["cfg"]
    p = ctx["params"].loc[sid]
    y = ctx["series"][sid]
    return step6.run_sku(
        sid, y, p, cfg, ctx["pool"], mode=mode,
        gate_model=ctx["model"], tau=tau, trace=trace,
    )


@pytest.fixture(scope="module")
def runs(ctx):
    out = {}

    tr = []
    out["c2_base"], out["c2_base_state"] = _run(
        ctx, ctx["c2_id"], "baseline", trace=tr
    )
    out["c2_base_trace"] = tr

    tr = []
    out["c2_open"], out["c2_open_state"] = _run(
        ctx, ctx["c2_id"], "gated", tau=0.0, trace=tr
    )
    out["c2_open_trace"] = tr

    tr = []
    out["c2_closed"], out["c2_closed_state"] = _run(
        ctx, ctx["c2_id"], "gated", tau=1.0 + 1e-9, trace=tr
    )
    out["c2_closed_trace"] = tr

    tr = []
    out["c0_non_c2_base"], _ = _run(
        ctx, ctx["c0_non_c2_id"], "baseline", trace=tr
    )
    out["c0_non_c2_base_trace"] = tr

    tr = []
    out["c0_non_c2_open"], out["c0_non_c2_state"] = _run(
        ctx, ctx["c0_non_c2_id"], "gated", tau=0.0, trace=tr
    )
    out["c0_non_c2_open_trace"] = tr

    out["c0_c2_open"], out["c0_c2_state"] = _run(
        ctx, ctx["c0_c2_id"], "gated", tau=ctx["tau"]
    )
    return out


# 1 ---------------------------------------------------------------------------
def test_closed_gate_is_transparent(ctx, runs):
    state = runs["c2_closed_state"]
    gamma = float(ctx["params"].loc[ctx["c2_id"], "gamma"])
    assert state["n_decisions"] == 1
    assert state["decision"]["action"] == "R0"
    assert state["action_t"] is None
    # The scale never leaves the injected value.
    assert state["scale"] == pytest.approx(gamma)
    _assert_logs_equal(runs["c2_base"].log, runs["c2_closed"].log)


# 2 ---------------------------------------------------------------------------
def test_open_gate_resets_scale_without_touching_the_forecast(ctx, runs):
    state = runs["c2_open_state"]
    dec = state["decision"]
    assert dec["action"] == "R2"
    assert dec["reason"] == "passed"
    assert dec["t"] == 120
    assert state["action_t"] == 121
    assert state["n_decisions"] == 1
    assert state["scale"] == 1.0

    tr_b = runs["c2_base_trace"]
    tr_g = runs["c2_open_trace"]
    mu_b = np.array([r["mu"] for r in tr_b], dtype=float)
    mu_g = np.array([r["mu"] for r in tr_g], dtype=float)
    np.testing.assert_allclose(mu_b, mu_g, rtol=0, atol=0, equal_nan=True)

    S_b = np.array([r["S"] for r in tr_b], dtype=float)
    S_g = np.array([r["S"] for r in tr_g], dtype=float)
    # The level is untouched while the policy has not acted...
    np.testing.assert_allclose(S_b[:121], S_g[:121], rtol=0, atol=0)
    # ...and moves exactly at the first call after the decision.
    assert _first_diff(S_b, S_g) == 121
    # The C2 scale multiplies a *signed* residual quantile, so resetting gamma
    # to 1.0 may raise or lower the level depending on the residual skew. The
    # remedy restores calibration; it is not a one-way safety-stock increase.
    assert not np.isclose(S_g[121], S_b[121])


# 3 ---------------------------------------------------------------------------
def test_no_look_ahead(ctx, runs):
    state = runs["c2_open_state"]
    decision_t = state["decision"]["t"]
    action_t = state["action_t"]
    L = int(step6.SimConfig.from_config(ctx["cfg"]).L)
    first_possible_cost_period = action_t + L

    log_b = runs["c2_base"].log
    log_g = runs["c2_open"].log

    # Nothing before the decision differs.
    for col in LOG_COLS:
        a = log_b[col].to_numpy(dtype=float)[: decision_t + 1]
        b = log_g[col].to_numpy(dtype=float)[: decision_t + 1]
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True,
                                   err_msg=f"pre-decision {col} differs")

    first_cost = _first_diff(log_b["cost_period"], log_g["cost_period"])
    assert first_cost is not None, "the intervention never changed a cost"
    assert first_cost >= first_possible_cost_period, (
        f"cost changed at {first_cost}, before an order placed at {action_t} "
        f"could arrive (t={first_possible_cost_period})"
    )
    first_on_hand = _first_diff(log_b["on_hand"], log_g["on_hand"])
    assert first_on_hand is None or first_on_hand >= first_possible_cost_period
    first_S = _first_diff(log_b["s_level"], log_g["s_level"])
    assert first_S == action_t


# 4 ---------------------------------------------------------------------------
def test_no_action_without_c2_attribution(runs):
    state = runs["c0_non_c2_state"]
    dec = state["decision"]
    assert dec is not None
    assert dec["attributed"] != "C2"
    assert dec["action"] == "R0"
    assert dec["reason"] == "not_C2"
    assert state["n_decisions"] == 1
    # Later checkpoints exist but no second decision is made.
    assert state["n_checks"] >= 2
    _assert_logs_equal(runs["c0_non_c2_base"].log, runs["c0_non_c2_open"].log)


# 5 ---------------------------------------------------------------------------
def test_fallback_is_logged_and_never_requeried(runs):
    state = runs["c0_c2_state"]
    dec = state["decision"]
    assert dec is not None
    assert dec["attributed"] == "C2"
    assert dec["action"] == "R0"
    assert dec["reason"] in {"low_reliability", "insufficient_evidence"}
    assert dec["t"] == 120
    assert state["n_decisions"] == 1
    assert state["n_checks"] == 1
    assert state["action_t"] is None
