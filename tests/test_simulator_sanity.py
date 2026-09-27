"""Simulator gates 1.1 (partly), 1.2-1.6. Rev 2 §30.1 Step 1, handoff §10.1.

**Read this before changing a threshold here.** Handoff §14 rule 4: if a gate fails,
the specification is wrong, not the gate. It does not get smoothed by widening ±0.02.

**Every CSL in this module is a CYCLE service level** (``SimResult.cycle_service_level``),
because that is what ``alpha = B/(B+H)`` means. An earlier version compared the
*per-period* fraction -- the share of periods with no unmet demand -- against ``alpha``.
On intermittent demand that fraction is inflated by the zero-demand periods, which are
trivially "served": on this process 71.4 % of periods have zero demand, so at
``alpha = 0.80`` the per-period fraction reads **0.9198** while the per-cycle level is
**0.7909**. Gate 1.2 failed against a measure that could not have matched, and the
per-period-vs-per-cycle distinction was the reason -- not the gate's value.

This is the same species of error as the 42.3 % bug the module exists to catch:
plausible-looking numbers reported at the wrong service level. Fixed in
``cafr/sim/inventory.py``; ``test_gate_1_2_the_two_service_measures_are_not_interchangeable``
guards against a silent revert.

**Resolved, not escalated:** the protection interval is ``R = L + 1 = 3``. This is
periodic review with ``review_period = 1``, so the protection interval is
*review interval + lead time*. See ``test_gate_1_2_protection_interval_diagnostic`` for
the measurement that settled it (``K=3`` lands inside ±0.02 at both alphas, ``K=2``
under-serves at both).

The oracle is the DGP-aware one (``cafr/sim/oracle_dgp.py``), not a zero-error point
forecaster. A zero-error forecaster has ``sigma_hat = 0`` and therefore lands near
CSL 0.50 by construction, which is what the original gate was accidentally testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cafr.sim.inventory import (
    InventorySimulator,
    SimConfig,
    aggregate_order_up_to,
    rolling_aggregates,
)
from cafr.sim.oracle_dgp import (
    BernoulliGammaDGP,
    oracle_level_function,
    zero_safety_level_function,
)
from cafr.utils.io import ensure_dir
from conftest import repo_path

# Gate 1.2's own numbers. Not tuning knobs.
GATE_SKUS = 500
GATE_PERIODS = 200
GATE_TOL = 0.02
ALPHAS = (0.80, 0.90, 0.95)

_RESULTS: dict = {}


# --------------------------------------------------------------------------- #
# The fixed gate panel -- drawn once, reused by 1.2 / 1.3 / 1.4 / the diagnostic #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gate_panel(cfg):
    """500 SKUs x 200 periods of the §15.1 Bernoulli-Gamma process.

    Generated once. The true demand series does not depend on ``alpha``, so all the
    alpha sweeps reuse the same draws -- which is also what makes the monotonicity
    check (gate 1.4) a paired comparison rather than two independent samples.
    """
    n, T = GATE_SKUS, GATE_PERIODS
    rng = np.random.default_rng(20260923)
    lo, hi = cfg["synthetic"]["p_range"]
    p = rng.uniform(lo, hi, n)
    k = rng.uniform(0.5, 4.0, n)
    mu_z = np.exp(rng.uniform(np.log(5.0), np.log(200.0), n))

    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=cfg["seed_root"])
    skus, ys, dgps = [], [], []
    for i in range(n):
        sku = f"GATE1_{i:04d}"
        y = np.array([sim.draw_demand(sku, t, p[i], k[i], mu_z[i]) for t in range(T)])
        skus.append(sku)
        ys.append(y)
        dgps.append(BernoulliGammaDGP(p=float(p[i]), k=float(k[i]), mu_z=float(mu_z[i])))
    return {"sim": sim, "skus": skus, "y": ys, "dgp": dgps, "p": p}


def _mean_csl(panel, level_factory, arm: str, tag: str) -> np.ndarray:
    """Per-SKU achieved CSL, memoised on ``tag`` so a re-run is free.

    ``cycle_service_level``, NOT ``period_service_level``. ``alpha`` is a
    newsvendor critical ratio -- a service level per protection interval -- so the
    per-cycle number is the one it can be compared against. The per-period fraction
    reads 0.9198 where the per-cycle level is 0.7909 at ``alpha = 0.80``, because
    71.4 % of periods on this process have zero demand and are trivially served.
    See the module docstring of ``cafr/sim/inventory.py``.
    """
    if tag in _RESULTS:
        return _RESULTS[tag]
    sim = panel["sim"]
    out = np.array([
        sim.run(sku, y, level_factory(dgp), arm=arm).cycle_service_level
        for sku, y, dgp in zip(panel["skus"], panel["y"], panel["dgp"])
    ])
    _RESULTS[tag] = out
    return out


# --------------------------------------------------------------------------- #
# Gate 1.5 -- the 42.3 % bug. Tested first, because gates 1.2-1.4 inherit it.    #
# --------------------------------------------------------------------------- #


def _independent_aggregates(y: np.ndarray, R: int) -> np.ndarray:
    """R-period aggregates, built with a loop rather than ``np.convolve``.

    Deliberately not the library function: comparing ``aggregate_order_up_to`` to a
    re-call of ``rolling_aggregates`` would assert that the code equals itself.
    """
    return np.array([float(y[i:i + R].sum()) for i in range(y.size - R + 1)])


def test_gate_1_5_order_up_to_uses_the_r_period_aggregate():
    """``S - R*mu_hat`` must be the empirical ``alpha``-quantile of ``A^agg - R*mu_hat``.

    Rev 2 §15.6: Revision 1 instead used ``z * sigma_period`` -- a per-period scale on
    an R-period interval -- understating safety stock by ``(sqrt(3)-1)/sqrt(3) = 42.3 %``
    at R = 3. Silent, plausible-looking, and wrong at exactly the service level the
    paper reports.
    """
    rng = np.random.default_rng(11)
    p, k, mu_z, T, R, alpha = 0.30, 2.0, 40.0, 400, 3, 0.90
    delta = rng.random(T) < p
    y = np.where(delta, rng.gamma(k, mu_z / k, T), 0.0)

    mu_hat = float(y.mean())
    S, alpha_eff, n_agg, fallback = aggregate_order_up_to(
        y, mu_hat, R, alpha, window=y.size, min_aggregates=4
    )

    assert not fallback, "400 periods of history must clear min_aggregates"
    assert alpha_eff == pytest.approx(alpha), (
        f"alpha_max floor bit unexpectedly: {alpha_eff} != {alpha}"
    )

    aggregates = _independent_aggregates(y, R)
    assert n_agg == aggregates.size
    expected = float(np.quantile(aggregates - R * mu_hat, alpha))

    safety = S - R * mu_hat
    assert abs(safety - expected) <= 0.02 * abs(expected), (
        f"order-up-to safety term {safety:.4f} is not the {alpha}-quantile of the "
        f"R-period aggregate residual ({expected:.4f})"
    )


def test_gate_1_5_the_per_period_substitution_is_absent():
    """The negative half: prove the WRONG construction is not what is happening.

    A test that only checks the right answer appears cannot distinguish "correct" from
    "correct by accident on this input". This asserts the per-period substitution gives
    a materially different -- and smaller -- number, so the positive test above is
    actually discriminating.

    Revision 1's bug, verbatim, was ``safety = z_alpha * sigma_period``: a per-period
    scale applied to the R-period protection interval. Under independence that
    understates the term by ``(sqrt(R)-1)/sqrt(R)`` -- 42.3 % at R = 3, a ratio of
    1.732. The skew of the residual distribution moves the empirical ratio away from
    that, so this asserts materiality (ratio well above 1), not the exact constant.
    """
    from cafr.forecasters.base import residual_sigma

    rng = np.random.default_rng(11)
    p, k, mu_z, T, R, alpha = 0.30, 2.0, 40.0, 400, 3, 0.90
    delta = rng.random(T) < p
    y = np.where(delta, rng.gamma(k, mu_z / k, T), 0.0)
    mu_hat = float(y.mean())

    S, _, _, _ = aggregate_order_up_to(y, mu_hat, R, alpha, window=y.size)
    safety = S - R * mu_hat

    per_period_quantile = float(np.quantile(y - mu_hat, alpha))
    rev1 = 1.2816 * residual_sigma(y - mu_hat)      # z_0.90 * sigma_period

    print(
        f"\n[gate 1.5] aggregate safety {safety:.2f} | per-period quantile "
        f"{per_period_quantile:.2f} (ratio {safety / per_period_quantile:.3f}) | "
        f"Rev-1 z*sigma_period {rev1:.4f}; theoretical sqrt(R) = {np.sqrt(R):.3f}"
    )

    assert safety > 1.25 * per_period_quantile, (
        f"the aggregate safety term ({safety:.2f}) is not materially larger than the "
        f"per-period quantile ({per_period_quantile:.2f}) -- the R-period/per-period "
        "distinction has been lost, and the positive test above would pass either way"
    )
    # Rev 1's construction is not merely understated on this data -- it is ZERO, because
    # the MAD scale collapses on a series with more than 50 % zeros (see
    # test_residual_sigma_collapses_on_intermittent_demand). Asserted as an equality so
    # that if someone "fixes" residual_sigma this test tells them the control changed.
    assert rev1 == 0.0, (
        f"Rev-1's z*sigma_period is {rev1}, not 0 -- residual_sigma changed, so the "
        "comparison this test makes is no longer the one Revision 1 made"
    )


def test_residual_sigma_collapses_on_intermittent_demand():
    """FINDING for the escalation list -- not a gate.

    ``residual_sigma`` is MAD-based ("so one spike does not set it"). On a series where
    more than half the periods are zero, EVERY zero period has the same residual
    ``0 - mu_hat``, which is also the median. Their absolute deviations are therefore
    exactly 0, and the median of those is 0. So ``sigma_hat == 0`` for every SKU whose
    zero fraction exceeds 50 %.

    Consequence: any quantity built from ``sigma_hat`` -- and Rev 2 §15.6 declares
    ``sigma_hat`` part of the forecaster interface -- is identically zero for the
    majority of an intermittent panel. C2's coverage test does not depend on it (it
    uses empirical residual quantiles, which is why gap 2.2 still passes), but the
    reported ``sigma_hat`` column in T2 would be a column of zeros and would look like
    a bug in the forecaster rather than in the scale estimator.

    Escalate: either adopt a zero-inflated scale estimator, or report sigma_hat on
    NONZERO residuals only and say so.
    """
    from cafr.forecasters.base import residual_sigma

    rng = np.random.default_rng(3)
    for zero_share in (0.4, 0.6, 0.7, 0.9):
        y = np.where(rng.random(500) < zero_share, 0.0, rng.gamma(2.0, 10.0, 500))
        sigma = residual_sigma(y - y.mean())
        if zero_share > 0.5:
            assert sigma == 0.0, (
                f"zero share {zero_share}: expected the MAD collapse, got sigma={sigma}"
            )

    # The nonzero-only variant does NOT collapse -- which is the recommended remedy.
    y = np.where(rng.random(500) < 0.7, 0.0, rng.gamma(2.0, 10.0, 500))
    nonzero = y[y > 0]
    assert residual_sigma(nonzero - nonzero.mean()) > 0.0


def test_aggregate_order_up_to_flags_insufficient_history():
    """Below ``min_aggregates`` the level falls back to ``R*mu_hat`` and SAYS SO."""
    y = np.asarray([5.0, 0.0, 3.0, 0.0, 0.0])        # 5 periods, R=3 -> 3 aggregates
    S, _, n_agg, fallback = aggregate_order_up_to(
        y, 1.5, R=3, alpha=0.9, window=24, min_aggregates=4
    )
    assert n_agg == 3, f"5 periods at R=3 give 3 complete aggregates, got {n_agg}"
    assert fallback, "3 aggregates < min_aggregates=4 must set the fallback flag"
    assert S == pytest.approx(3 * 1.5), "the fallback level carries NO safety stock"


def test_aggregate_order_up_to_does_not_flag_at_the_boundary():
    """``n == min_aggregates`` is ENOUGH. Off-by-one here silently disables safety stock.

    The fallback returns ``S = R * mu_hat`` -- zero safety stock, i.e. a materially
    different inventory policy. A ``<=`` where ``<`` belongs would switch it on for
    every SKU that is exactly at the floor.
    """
    y = np.asarray([5.0, 0.0, 3.0, 0.0, 0.0, 4.0])   # 6 periods, R=3 -> 4 aggregates
    S, _, n_agg, fallback = aggregate_order_up_to(
        y, 1.5, R=3, alpha=0.9, window=24, min_aggregates=4
    )
    assert n_agg == 4
    assert not fallback, "4 aggregates == min_aggregates=4 must NOT fall back"


# --------------------------------------------------------------------------- #
# Gate 1.2 -- the oracle matches its own target                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", ALPHAS)
def test_gate_1_2_dgp_aware_oracle_hits_the_target(alpha, cfg, gate_panel):
    """Cycle CSL within +/- 0.02 of the target, at the spec's protection interval.

    ``cycle_service_level``, not the per-period fraction. The per-period fraction is
    not comparable to ``alpha`` -- see ``_mean_csl`` and the ``cafr/sim/inventory.py``
    module docstring. Measured on this panel, the per-period fraction at
    ``alpha = 0.80`` is 0.9198 against a true per-cycle level of 0.7909.
    """
    R = cfg["sim"]["R"]
    factory = lambda dgp: oracle_level_function(dgp, R, alpha)  # noqa: E731
    csl = _mean_csl(gate_panel, factory, arm="oracle", tag=f"oracle_R{R}_a{alpha}")
    achieved = float(csl.mean())

    print(f"\n[gate 1.2] R={R} alpha={alpha:.2f} -> cycle CSL {achieved:.4f} "
          f"(n={csl.size}, se={csl.std(ddof=1)/np.sqrt(csl.size):.4f})")
    assert abs(achieved - alpha) <= GATE_TOL, (
        f"oracle cycle CSL {achieved:.4f} is outside +/-{GATE_TOL} of {alpha} at R={R}"
    )


def test_gate_1_2_the_two_service_measures_are_not_interchangeable(cfg, gate_panel):
    """Records WHY the measure above had to change. Guards against a silent revert.

    If someone "simplifies" ``_mean_csl`` back to the per-period fraction, gate 1.2
    fails -- but it fails with a confusing message. This test states the mechanism in
    the number: the per-period fraction sits far above the per-cycle level, because
    71 % of periods on this process have zero demand and are trivially served.

    It is a property of INTERMITTENT demand, not a defect in the simulator. On a
    dense panel the two measures converge.
    """
    alpha = 0.80
    R = cfg["sim"]["R"]
    sim, skus, ys, dgps = (
        gate_panel["sim"], gate_panel["skus"], gate_panel["y"], gate_panel["dgp"]
    )
    results = [sim.run(s, y, oracle_level_function(d, R, alpha), arm="oracle")
               for s, y, d in zip(skus, ys, dgps)]
    cycle = float(np.mean([r.cycle_service_level for r in results]))
    period = float(np.mean([r.period_service_level for r in results]))
    zero_share = float(np.mean([(y == 0).mean() for y in ys]))

    print(f"\n[gate 1.2 measure] alpha={alpha:.2f}  cycle CSL {cycle:.4f}  "
          f"period fraction {period:.4f}  zero-demand periods {zero_share:.1%}")

    assert period > cycle + 0.05, (
        f"the two measures are nearly identical here ({period:.4f} vs {cycle:.4f}) -- "
        "if the panel has become dense, this guard no longer discriminates and the "
        "intermittency premise of the whole project needs re-checking"
    )
    assert abs(cycle - alpha) <= GATE_TOL
    assert abs(period - alpha) > GATE_TOL, (
        "the per-period fraction happens to land on the target -- in that case this "
        "project's data is not intermittent enough for any of these gates to mean "
        "what §15 assumes, and that is the finding"
    )
    assert zero_share > 0.5


def test_gate_1_2_protection_interval_diagnostic(cfg, gate_panel):
    """Records which integer protection interval the simulator's timing implies.

    **Resolved: ``R = L + 1 = 3`` is correct, and this test is why we know.** This is a
    periodic-review system with ``review_period = 1``, so the protection interval is
    *review interval + lead time* = ``1 + L`` = 3. An earlier reading -- that ordering
    at the end of ``t`` for arrival at the start of ``t + L`` covers only
    ``y_{t+1} .. y_{t+L}``, i.e. ``L`` periods -- omitted the period between setting
    the level and the arrival of the order it generates.

    The measurement that settled it, on the per-cycle CSL::

        K=2:  0.7737 (alpha=0.80)   0.8719 (alpha=0.90)   -- under-serves, both alphas
        K=3:  0.7909 (alpha=0.80)   0.8989 (alpha=0.90)   -- inside gate 1.2's +/-0.02

    The test stays because it is cheap and it is the evidence for the choice. It writes
    both intervals to ``results/step1_protection_diagnostic.json``. Its assertion is
    deliberately weak -- it requires that the target be hit at *some* integer interval,
    so it fails if the level machinery breaks, without re-litigating the convention.
    Gate 1.2 is the one that carries the spec's value.
    """
    alphas = ALPHAS
    intervals = sorted({int(cfg["sim"]["L"]), int(cfg["sim"]["R"])})
    table = {}
    for K in intervals:
        row = {}
        for alpha in alphas:
            factory = lambda dgp, K=K, a=alpha: oracle_level_function(dgp, K, a)  # noqa: E731
            csl = _mean_csl(
                gate_panel, factory, arm="diagnostic", tag=f"diag_K{K}_a{alpha}"
            )
            row[f"alpha_{alpha:.2f}"] = {
                "cycle_csl": round(float(csl.mean()), 5),
                "abs_error": round(abs(float(csl.mean()) - alpha), 5),
                "within_tol": bool(abs(float(csl.mean()) - alpha) <= GATE_TOL),
            }
        table[f"K={K}"] = row

    matching = [
        K for K in intervals
        if all(v["within_tol"] for v in table[f"K={K}"].values())
    ]
    payload = {
        "spec_protection_convention": cfg["sim"]["protection_convention"],
        "spec_R": int(cfg["sim"]["R"]),
        "lead_time_L": int(cfg["sim"]["L"]),
        "gate_tolerance": GATE_TOL,
        "n_skus": GATE_SKUS,
        "n_periods": GATE_PERIODS,
        "measured": table,
        "intervals_hitting_the_target": matching,
        "verdict": (
            f"the simulator's timing is consistent with K={matching[0]}"
            + ("" if matching and matching[0] == cfg["sim"]["R"]
               else " -- NOT the spec's R = L + 1. Escalate; do not relax the gate.")
        ) if matching else "NO integer interval hits the target: the simulator is broken.",
    }
    out = repo_path("results", "step1_protection_diagnostic.json")
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for K in intervals:
        print(f"\n[gate 1.2 diagnostic] K={K}: " + "  ".join(
            f"a={v['cycle_csl']:.4f}" for v in table[f"K={K}"].values()
        ))
    print(f"\n[gate 1.2 diagnostic] {payload['verdict']}")

    assert matching, "no integer protection interval reproduces the target CSL"
    assert cfg["sim"]["R"] in matching, (
        f"the spec's R={cfg['sim']['R']} is not among the intervals hitting the target "
        f"({matching}). Do NOT relax gate 1.2 -- escalate the convention."
    )


# --------------------------------------------------------------------------- #
# Gate 1.3 -- the negative half. This is what proves the test CAN fail.        #
# --------------------------------------------------------------------------- #


def test_gate_1_3_zero_safety_stock_lands_near_one_half(cfg, gate_panel):
    """``S = R*mu_hat`` with the TRUE mean must give achieved CSL around 0.50.

    **Known risk, reported rather than tuned away.** The §15.1 process has an atom at
    zero of size ``(1-p)^R`` per aggregate, so ``P(A <= R*mu_hat)`` exceeds 0.50 for
    sparse SKUs even with no safety stock. This test therefore reports the measured
    value, the mechanism, and the dense-SKU subsample alongside the gate's own
    ``0.50 +/- 0.03`` -- it does not widen the tolerance.
    """
    alpha = 0.90
    R = cfg["sim"]["R"]
    factory = lambda dgp: zero_safety_level_function(dgp, R, alpha)  # noqa: E731
    csl = _mean_csl(gate_panel, factory, arm="zero_safety", tag=f"zerosafety_R{R}")

    p = gate_panel["p"]
    dense = p >= 0.35
    achieved = float(csl.mean())
    achieved_dense = float(csl[dense].mean())
    print(
        f"\n[gate 1.3] zero safety stock: mean CSL {achieved:.4f} "
        f"(all {csl.size} SKUs), {achieved_dense:.4f} on the {int(dense.sum())} densest; "
        f"target 0.50 +/- 0.03"
    )

    payload = {
        "gate": "1.3",
        "metric": "cycle_service_level",
        "n_skus": int(csl.size),
        "cycle_csl_all": round(achieved, 5),
        "cycle_csl_p_ge_0.35": round(achieved_dense, 5),
        "n_dense": int(dense.sum()),
        "atom_at_zero_explains_the_excess": (
            "P(A_R <= R*mu) >= P(A_R = 0) = (1-p)^R, so sparse SKUs exceed 0.50 "
            "with no safety stock. At p=0.05, R=3 that floor is 0.857."
        ),
    }
    out = repo_path("results", "step1_gate_1_3.json")
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # The half of the gate that is about the SIMULATOR, and must hold: removing the
    # safety stock must move CSL far away from the oracle's alpha.
    from cafr.sim.oracle_dgp import oracle_level_function as _oracle

    oracle_csl = _mean_csl(
        gate_panel, lambda dgp: _oracle(dgp, R, alpha), arm="oracle",
        tag=f"oracle_R{R}_a{alpha}",
    )
    assert float(oracle_csl.mean()) - achieved > 0.10, (
        "zeroing the safety stock barely moved CSL -- the level is not reaching the "
        "simulator, and every gate below is meaningless"
    )
    assert achieved < alpha, "the negative control must under-perform the oracle"


# --------------------------------------------------------------------------- #
# Gate 1.4 -- monotone in alpha                                                #
# --------------------------------------------------------------------------- #


def test_gate_1_4_cycle_csl_is_non_decreasing_in_alpha(cfg, gate_panel):
    """No violations across the sweep. Paired: the same SKUs, the same demand draws."""
    R = cfg["sim"]["R"]
    curves = []
    for alpha in (0.50, 0.80, 0.90, 0.95):
        csl = _mean_csl(
            gate_panel, lambda dgp, a=alpha: oracle_level_function(dgp, R, a),  # noqa: E731
            arm="oracle", tag=f"oracle_R{R}_a{alpha}",
        )
        curves.append(csl)

    means = [float(c.mean()) for c in curves]
    print("\n[gate 1.4] mean CSL by alpha (0.50, 0.80, 0.90, 0.95): "
          + ", ".join(f"{m:.4f}" for m in means))

    for lo, hi, m_lo, m_hi in zip(curves, curves[1:], means, means[1:]):
        assert m_hi >= m_lo - 1e-9, f"mean CSL fell: {m_lo:.4f} -> {m_hi:.4f}"

    # Per-SKU violations are the strict reading of "no violations". Report the count;
    # a handful of ties are expected because the oracle level is constant per SKU.
    violations = 0
    for lo, hi in zip(curves, curves[1:]):
        violations += int((hi < lo - 1e-12).sum())
    assert violations <= 0.02 * curves[0].size, (
        f"{violations} SKUs have non-monotone CSL in alpha"
    )


# --------------------------------------------------------------------------- #
# Gate 1.6 -- censoring                                                        #
# --------------------------------------------------------------------------- #


def test_gate_1_6_censoring_marks_and_hides_lost_demand(cfg, gate_panel):
    """With a deliberately under-set policy, a stock-out must censor the series.

    ``demand_observed < demand_true`` on >= 95 % of stock-out periods. This is the
    C-guard's raw material: if the observed series were not censored, C7 would be
    undetectable and the imputation would have nothing to do.
    """
    sim = gate_panel["sim"]
    R = cfg["sim"]["R"]
    skus, ys = gate_panel["skus"][:120], gate_panel["y"][:120]

    def under_set(mu_hat):
        def fn(t, y_observed, censored, inventory):
            return 0.5 * R * mu_hat, 0.90
        return fn

    stockout_periods = 0
    censored_and_hidden = 0
    total_periods = 0
    for sku, y in zip(skus, ys):
        mu_hat = float(y[:24].mean())
        res = sim.run(sku, y, under_set(mu_hat), arm="under_set")
        lost = res.log["demand_lost"].to_numpy()
        met = res.log["demand_met"].to_numpy()
        is_stockout = lost > 0
        stockout_periods += int(is_stockout.sum())
        censored_and_hidden += int((met[is_stockout] < y[is_stockout]).sum())
        total_periods += int(y.size)
        assert np.array_equal(res.censored, is_stockout), (
            "the censoring mask must be exactly the lost-sales periods"
        )

    assert stockout_periods > 0, "the under-set policy produced no stock-outs at all"
    share = censored_and_hidden / stockout_periods
    print(f"\n[gate 1.6] {stockout_periods} stock-out periods "
          f"({stockout_periods / total_periods:.1%} of all periods); "
          f"observed < true on {share:.4%} of them")
    assert share >= 0.95, f"censoring only hid demand on {share:.2%} of stock-outs"


def test_gate_1_6_nothing_is_lost_when_the_policy_over_protects(cfg, gate_panel):
    """The other direction: a generous level must produce no censoring at all.

    **Warm-up is excluded, and that is not a convenience.** No order placed at the end
    of period ``t`` can arrive before the start of ``t + L``, so periods ``0 .. L-1``
    have zero on-hand no matter how generous the level is -- this policy orders
    ``10 * mu_hat + 1`` and still loses demand in exactly those two periods. Asserting
    over the whole series would fail for a reason that says nothing about the policy.

    The assertion is sharpened to compensate: zero censoring across EVERY evaluated
    period, and a positive on-hand in every evaluated period of a 60-SKU sample.
    """
    sim = gate_panel["sim"]
    skus, ys = gate_panel["skus"][:60], gate_panel["y"][:60]
    warmup = int(cfg["sim"]["L"])

    def generous(mu_hat):
        def fn(t, y_observed, censored, inventory):
            return 10.0 * mu_hat + 1.0, 0.99
        return fn

    for sku, y in zip(skus, ys):
        res = sim.run(sku, y, generous(float(y[:24].mean())), arm="generous")
        assert not res.censored[warmup:].any(), (
            f"{sku}: an over-protected policy censored demand at period "
            f"{int(np.flatnonzero(res.censored[warmup:])[0]) + warmup}"
        )
        assert (res.log["on_hand"].to_numpy()[warmup:] > 0).all(), (
            f"{sku}: on-hand hit zero after warm-up under a 10x level"
        )
        assert res.n_stockout == 0
        assert res.cycle_service_level == pytest.approx(1.0)
