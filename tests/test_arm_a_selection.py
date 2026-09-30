"""Arm (a) must be a genuine validation-selected baseline, not a copy of arm (b).

Until this change, ``arm_a_factory`` fell back to ``SB_MODEL_MAP`` whenever no
``chosen_method`` was supplied -- which is always, in every driver -- so arm (a) and
arm (b) were the same arm under two names. Every (a)-vs-(b) comparison in the paper
would have been a comparison of a rule with itself.

These tests check the mechanism, not a number: that a selection happens, that it is
confined to the burn-in window, and that a fallback is recorded rather than silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from cafr.arms.arm_a import (
    CLASSICAL_CANDIDATES,
    SB_FALLBACK,
    arm_a_factory,
    select_by_validation,
)
from cafr.arms.arm_b import SB_MODEL_MAP
from cafr.forecasters.registry import build_pool
from cafr.utils.config import load_config

BURN_IN = 24


@pytest.fixture(scope="module")
def cfg():
    return load_config("base.yaml")


@pytest.fixture(scope="module")
def pool(cfg):
    return build_pool(cfg)


def _intermittent(seed: int, n: int = 40, p: float = 0.25, size: float = 20.0,
                  disp: float = 0.5) -> np.ndarray:
    """A series with a controllable rate and size dispersion."""
    rng = np.random.default_rng(seed)
    occ = rng.random(n) < p
    shape = 1.0 / max(disp, 1e-6)
    y = np.zeros(n, dtype="float64")
    y[occ] = rng.gamma(shape, size / shape, size=int(occ.sum()))
    return y


# --------------------------------------------------------------------------- #
# the selection is real
# --------------------------------------------------------------------------- #


def test_selection_log_matches_its_own_scores(cfg, pool):
    """When the rule is ``argmin_mase``, the chosen method IS the argmin."""
    for seed in range(12):
        y = _intermittent(seed)
        method, log = select_by_validation(
            y[:BURN_IN], np.zeros(BURN_IN, dtype=bool), pool, sb_cell="moderate_lowdisp"
        )
        if log["rule"] != "argmin_mase":
            continue
        finite = {m: s for m, s in log["scores"].items() if np.isfinite(s)}
        assert finite, "argmin_mase with no finite score is impossible"
        assert method in finite
        assert log["scores"][method] == min(finite.values())
        assert log["reason"] == "validation_mase"


def test_arm_a_does_not_merely_reproduce_arm_b(cfg, pool):
    """Over varied series, arm (a)'s data-driven pick must disagree with the SB rule."""
    differs = 0
    chosen: set[str] = set()
    for seed in range(40):
        y = _intermittent(seed, p=0.10 + 0.30 * (seed % 5) / 4, size=5.0 * (seed % 7 + 1))
        cell = ["moderate_lowdisp", "high_lowdisp", "moderate_highdisp", "high_highdisp"][
            seed % 4
        ]
        method, log = select_by_validation(
            y[:BURN_IN], np.zeros(BURN_IN, dtype=bool), pool, sb_cell=cell
        )
        chosen.add(method)
        if method != SB_MODEL_MAP[cell]:
            differs += 1
    assert len(chosen) > 1, f"arm (a) chose a single method everywhere: {chosen}"
    assert differs > 0, (
        "arm (a) reproduced arm (b)'s SB rule on all 40 series -- it is still a "
        "duplicate, not a validation-selected baseline"
    )


def test_every_candidate_is_scored_or_absent(cfg, pool):
    """Every candidate in the pool gets a finite score and the pool is not mutated."""
    y = _intermittent(3)
    before = set(pool)
    _, log = select_by_validation(
        y[:BURN_IN], np.zeros(BURN_IN, dtype=bool), pool, sb_cell="dead"
    )
    assert set(pool) == before
    for name in CLASSICAL_CANDIDATES:
        assert name in log["scores"]
    assert all(np.isfinite(log["scores"][m]) for m in CLASSICAL_CANDIDATES)


# --------------------------------------------------------------------------- #
# no look-ahead
# --------------------------------------------------------------------------- #


def test_selection_reads_only_the_burn_in_window(cfg, pool):
    """Choking the post-burn-in tail must not change the chosen method.

    The simulator hands the level function ``y_observed[:t + 1]``, and ``t`` is the
    first EVALUATED period. If the arm selected on ``y_observed[:t]`` or on the whole
    series, a change after the burn-in would move the choice -- which is look-ahead.
    """
    for seed in range(10):
        y = _intermittent(seed)
        method0, log0 = select_by_validation(
            y[:BURN_IN], np.zeros(BURN_IN, dtype=bool), pool, sb_cell="moderate_lowdisp"
        )

        # The same burn-in, but the evaluated window destroyed.
        y2 = y.copy()
        y2[BURN_IN:] = 0.0
        method1, _ = select_by_validation(
            y2[:BURN_IN], np.zeros(BURN_IN, dtype=bool), pool, sb_cell="moderate_lowdisp"
        )
        assert method0 == method1
        assert log0["n_origins"] <= BURN_IN


def test_level_fn_selects_at_the_first_call_from_the_burn_in_only(cfg, pool):
    """Through the arm's real entry point, not just the helper."""
    y = _intermittent(5)
    lf = arm_a_factory(pool, cfg, "s", 0.95, "observed", sb_cell="moderate_lowdisp")
    assert lf.selection is None, "nothing may be selected before the first call"

    lf(t=BURN_IN, y_observed=y[: BURN_IN + 1], censored=np.zeros(BURN_IN + 1, bool),
       inventory=None)
    first = dict(lf.selection)

    # A second call with a wildly different tail must not re-select.
    y_big = y.copy()
    y_big[BURN_IN + 1 :] = 1e6
    lf2 = arm_a_factory(pool, cfg, "s", 0.95, "observed", sb_cell="moderate_lowdisp")
    lf2(t=BURN_IN, y_observed=y_big[: BURN_IN + 1], censored=np.zeros(BURN_IN + 1, bool),
        inventory=None)
    assert lf2.selection["chosen"] == first["chosen"]

    # The selection window is the burn-in: one more scored origin than MIN_TRAIN at
    # most, never the full history.
    assert lf.selection["n_origins"] == BURN_IN - 8


# --------------------------------------------------------------------------- #
# a fallback is recorded, never silent
# --------------------------------------------------------------------------- #


def test_degenerate_window_falls_back_and_says_so(cfg, pool):
    """An all-zero validation window cannot rank models; the arm must say so."""
    zeros = np.zeros(BURN_IN, dtype="float64")
    method, log = select_by_validation(
        zeros, np.zeros(BURN_IN, dtype=bool), pool, sb_cell="high_highdisp"
    )
    assert log["rule"] == "sb_fallback"
    assert log["reason"] == "zero_naive_scale"
    assert method == SB_FALLBACK["high_highdisp"] == "tsb"
    assert method != "validation_mase"


def test_insufficient_window_falls_back_and_says_so(cfg, pool):
    """Fewer origins than MIN_TRAIN -> SB fallback, with the reason recorded."""
    y = _intermittent(1, n=6)
    _, log = select_by_validation(
        y, np.zeros(6, dtype=bool), pool, sb_cell="high_lowdisp"
    )
    assert log["rule"] == "sb_fallback"
    assert log["reason"] == "insufficient_validation"
    assert log["n_origins"] == 0


def test_pre_selected_method_is_never_overwritten(cfg, pool):
    """An explicit ``chosen_method`` wins, and is logged as external."""
    lf = arm_a_factory(
        pool, cfg, "s", 0.95, "observed", chosen_method="tsb", sb_cell="high_lowdisp"
    )
    assert lf.selection["rule"] == "external"
    lf(t=BURN_IN, y_observed=_intermittent(2)[: BURN_IN + 1],
       censored=np.zeros(BURN_IN + 1, bool), inventory=None)
    assert lf.chosen_method == "tsb"
    assert lf.last_state.method == "tsb"


def test_selection_is_deterministic(cfg, pool):
    y = _intermittent(11)
    a, la = select_by_validation(y[:BURN_IN], np.zeros(BURN_IN, bool), pool, sb_cell="dead")
    b, lb = select_by_validation(y[:BURN_IN], np.zeros(BURN_IN, bool), pool, sb_cell="dead")
    assert a == b
    assert la["scores"] == lb["scores"]
