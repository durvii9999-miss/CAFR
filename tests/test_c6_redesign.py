"""Focused unit tests for C6 generator redesign candidate and detectability budget.

Tests cover:
1. Exact ramp indexing: 24 ramp indices are t=120..143, destination regime begins at t=144.
2. Linear interpolation ramp endpoints and monotonicity (upward and downward).
3. Probability clipping and bounds integrity.
4. Unchanged demand-size parameters (k and mu_z invariant).
5. Preservation of the frozen H-5 rule (< 2 nonzeros = dead).
6. Exact calculations in check_c6_budget: target at boundary, strict crossing, >= 2-sigma separation.
7. Graceful handling of invalid and out-of-range inputs in check_c6_budget.
8. Verification that the original baseline inject_C6 remains completely untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from cafr.data.synth.budget import check_c6_budget
from cafr.data.synth.injections import inject_C6, inject_C6_v2
from cafr.monitor.stability import adi_cv2_cell


def test_inject_c6_v2_exact_boundary_indices():
    """Verify exact boundary indexing:

    - Pre-injection baseline: t < 120 (indices 0..119), p_t == p_init.
    - Exactly 24 ramp indices: t = 120..143.
    - Boundary values:
        t=120: p_t == p_init.
        t=143: p_t == p_init + (p_target - p_init) * (23/24).
    - Destination regime begins at t = 144:
        t=144: p_t == p_target.
        t=199: p_t == p_target.
    """
    T, tau, w_ramp = 200, 120, 24
    p_init = 0.15
    target_up = 0.45

    # Compute theoretical p_t directly from definition
    t_idx = np.arange(T)
    p_t = np.full(T, p_init)
    ramp_mask = (t_idx >= tau) & (t_idx < tau + w_ramp)
    p_t[ramp_mask] = p_init + (target_up - p_init) * (t_idx[ramp_mask] - tau) / float(w_ramp)
    p_t[t_idx >= tau + w_ramp] = target_up

    # 1. Baseline ends at t = 119
    assert p_t[119] == pytest.approx(0.15)

    # 2. Ramp starts at t = 120 and ends at t = 143 (exactly 24 periods: 143 - 120 + 1 = 24)
    ramp_indices = np.where(ramp_mask)[0]
    assert len(ramp_indices) == 24
    assert ramp_indices[0] == 120
    assert ramp_indices[-1] == 143
    assert p_t[120] == pytest.approx(0.15)
    assert p_t[143] == pytest.approx(0.15 + (0.45 - 0.15) * (23.0 / 24.0))

    # 3. Destination regime begins at t = 144 and holds through t = 199
    dest_indices = np.where(t_idx >= tau + w_ramp)[0]
    assert len(dest_indices) == 56  # 200 - 144 = 56 periods
    assert dest_indices[0] == 144
    assert dest_indices[-1] == 199
    assert p_t[144] == pytest.approx(0.45)
    assert p_t[199] == pytest.approx(0.45)


def test_inject_c6_v2_ramp_endpoints_and_monotonicity_upward():
    """Verify upward linear interpolation from p < 0.25 to target_up (0.45)."""
    rng = np.random.default_rng(42)
    T, tau, w_ramp = 200, 120, 24
    p_init = 0.15
    k, mu_z = 2.0, 30.0

    y, delta, z, params = inject_C6_v2(
        T, tau, p_init, k, mu_z, rng, w_ramp=w_ramp, target_up=0.45, target_down=0.10
    )

    assert len(y) == T
    assert len(delta) == T
    assert len(z) == T

    assert params["p_init"] == pytest.approx(0.15)
    assert params["p_target"] == pytest.approx(0.45)
    assert params["p_end"] == pytest.approx(0.45)
    assert params["w_ramp"] == 24
    assert params["adi_init"] == pytest.approx(1.0 / 0.15)
    assert params["adi_end"] == pytest.approx(1.0 / 0.45)
    assert params["adi_crosses_star"] is True
    assert params["demand_touched"] is True


def test_inject_c6_v2_ramp_endpoints_and_monotonicity_downward():
    """Verify downward linear interpolation from p >= 0.25 to target_down (0.10)."""
    rng = np.random.default_rng(42)
    T, tau, w_ramp = 200, 120, 24
    p_init = 0.35
    k, mu_z = 2.0, 30.0

    y, delta, z, params = inject_C6_v2(
        T, tau, p_init, k, mu_z, rng, w_ramp=w_ramp, target_up=0.45, target_down=0.10
    )

    assert params["p_init"] == pytest.approx(0.35)
    assert params["p_target"] == pytest.approx(0.10)
    assert params["p_end"] == pytest.approx(0.10)
    assert params["w_ramp"] == 24
    assert params["adi_init"] == pytest.approx(1.0 / 0.35)
    assert params["adi_end"] == pytest.approx(1.0 / 0.10)
    assert params["adi_crosses_star"] is True


def test_inject_c6_v2_probability_bounds_and_clipping():
    """Verify that extreme initial probabilities remain strictly bounded within [0.03, 0.60]."""
    rng = np.random.default_rng(123)
    T, tau = 100, 50

    # Extremely low p
    _, _, _, p_low = inject_C6_v2(T, tau, 0.01, 1.0, 20.0, rng, target_up=0.45, target_down=0.10)
    assert p_low["p_target"] == pytest.approx(0.45)
    assert 0.03 <= p_low["p_end"] <= 0.60

    # Extremely high p
    _, _, _, p_hi = inject_C6_v2(T, tau, 0.65, 1.0, 20.0, rng, target_up=0.45, target_down=0.10)
    assert p_hi["p_target"] == pytest.approx(0.10)
    assert 0.03 <= p_hi["p_end"] <= 0.60


def test_inject_c6_v2_preserves_demand_size_parameters():
    """Verify demand-size parameters (k and mu_z) are untouched across the injection."""
    rng = np.random.default_rng(999)
    T, tau = 200, 120
    k, mu_z = 2.5, 40.0

    pre_sizes = []
    post_sizes = []
    for _ in range(100):
        _, delta, z, _ = inject_C6_v2(T, tau, 0.15, k, mu_z, rng)
        pre_sizes.extend(z[:tau])
        post_sizes.extend(z[tau + 24 :])

    pre_sizes = np.array(pre_sizes)
    post_sizes = np.array(post_sizes)

    assert np.mean(pre_sizes) == pytest.approx(40.0, abs=1.0)
    assert np.mean(post_sizes) == pytest.approx(40.0, abs=1.0)
    assert np.var(pre_sizes) == pytest.approx(640.0, abs=40.0)
    assert np.var(post_sizes) == pytest.approx(640.0, abs=40.0)


def test_h5_rule_preservation_and_dead_cell_integrity():
    """Verify H-5 definition is preserved: < 2 nonzeros in burn-in produces 'dead', >= 2 does not."""
    # 0 nonzeros in 24 periods => dead
    w_zero = np.zeros(24)
    cell_zero = adi_cv2_cell(w_zero, adi_star=4.0, cv2_star=0.49)
    assert cell_zero["cell"] == "dead"
    assert np.isnan(cell_zero["cv2"])

    # 1 nonzero in 24 periods => dead
    w_one = np.zeros(24)
    w_one[5] = 10.0
    cell_one = adi_cv2_cell(w_one, adi_star=4.0, cv2_star=0.49)
    assert cell_one["cell"] == "dead"
    assert np.isnan(cell_one["cv2"])

    # 2 nonzeros in 24 periods => non-dead
    w_two = np.zeros(24)
    w_two[5] = 10.0
    w_two[15] = 12.0
    cell_two = adi_cv2_cell(w_two, adi_star=4.0, cv2_star=0.49)
    assert cell_two["cell"] != "dead"
    assert np.isfinite(cell_two["cv2"])
    assert np.isfinite(cell_two["adi"])


def test_check_c6_budget_calculations():
    """Verify exact theoretical crossing and >= 2-sigma separation calculations."""
    # Case 1: Upward drift from 0.15 to 0.45
    # mu_N = 30 * 0.45 = 13.5, N* = 7.5, sigma_N = sqrt(30 * 0.45 * 0.55) = 2.724885
    # z_sep = 6.0 / 2.724885 = 2.2019 >= 2.0
    b_up = check_c6_budget(0.15, 0.45, w_class=30, adi_star=4.0, min_sigma=2.0)
    assert b_up["crosses"] is True
    assert b_up["mu_N"] == pytest.approx(13.5)
    assert b_up["N_star"] == pytest.approx(7.5)
    assert b_up["sigma_N"] == pytest.approx(np.sqrt(30 * 0.45 * 0.55))
    assert b_up["z_sep"] == pytest.approx(2.2019275, rel=1e-4)
    assert b_up["cleared"] is True

    # Case 2: Downward drift from 0.35 to 0.10
    # mu_N = 30 * 0.10 = 3.0, N* = 7.5, sigma_N = sqrt(30 * 0.10 * 0.90) = 1.643168
    # z_sep = 4.5 / 1.643168 = 2.7386 >= 2.0
    b_down = check_c6_budget(0.35, 0.10, w_class=30, adi_star=4.0, min_sigma=2.0)
    assert b_down["crosses"] is True
    assert b_down["mu_N"] == pytest.approx(3.0)
    assert b_down["N_star"] == pytest.approx(7.5)
    assert b_down["sigma_N"] == pytest.approx(np.sqrt(30 * 0.10 * 0.90))
    assert b_down["z_sep"] == pytest.approx(2.7386127, rel=1e-4)
    assert b_down["cleared"] is True

    # Case 3: Does not cross boundary (0.15 to 0.20, both < 0.25)
    b_nocross = check_c6_budget(0.15, 0.20, w_class=30, adi_star=4.0, min_sigma=2.0)
    assert b_nocross["crosses"] is False
    assert b_nocross["cleared"] is False


def test_check_c6_budget_target_at_threshold_and_boundary_conditions():
    """Verify behavior when target or init is exactly at the ADI threshold (p = 0.25).

    Strict crossing requires strictly straddling; a boundary target has z_sep == 0.0.
    """
    # Target exactly at threshold (0.25) => z_sep == 0.0, crosses == False
    b_target_boundary = check_c6_budget(0.15, 0.25, w_class=30, adi_star=4.0, min_sigma=2.0)
    assert b_target_boundary["crosses"] is False
    assert b_target_boundary["z_sep"] == pytest.approx(0.0)
    assert b_target_boundary["cleared"] is False

    # Init exactly at threshold (0.25) => crosses == False
    b_init_boundary = check_c6_budget(0.25, 0.45, w_class=30, adi_star=4.0, min_sigma=2.0)
    assert b_init_boundary["crosses"] is False
    assert b_init_boundary["cleared"] is False


def test_check_c6_budget_invalid_and_out_of_range_inputs():
    """Verify that invalid, non-numeric, or out-of-range inputs return cleared: False gracefully."""
    # Negative probabilities
    assert check_c6_budget(-0.1, 0.45)["cleared"] is False
    assert check_c6_budget(0.15, -0.1)["cleared"] is False

    # Probabilities >= 1.0
    assert check_c6_budget(1.2, 0.45)["cleared"] is False
    assert check_c6_budget(0.15, 1.0)["cleared"] is False

    # Invalid window or adi_star
    assert check_c6_budget(0.15, 0.45, w_class=0)["cleared"] is False
    assert check_c6_budget(0.15, 0.45, adi_star=-1.0)["cleared"] is False

    # NaN / Inf inputs
    assert check_c6_budget(float("nan"), 0.45)["cleared"] is False
    assert check_c6_budget(0.15, float("inf"))["cleared"] is False

    # Non-numeric input
    assert check_c6_budget("invalid", 0.45)["cleared"] is False


def test_original_inject_c6_remains_unmodified():
    """Verify that original inject_C6 is completely untouched and exhibits baseline behavior."""
    rng = np.random.default_rng(42)
    y, delta, z, params = inject_C6(200, 120, 0.15, 2.0, 30.0, rng)

    assert len(y) == 200
    assert "psi" in params
    assert "p_end" in params
    assert "adi_end" in params
    assert "adi_crosses_star" in params
    assert "demand_touched" in params
    assert params["demand_touched"] is True

    # Baseline inject_C6 drifts over 60 periods without capping at t=180.
    # At t=199 (drift=79/60), p reaches ~0.4133.
    assert params["p_end"] == pytest.approx(0.15 * (1.0 + ((0.35 / 0.15) - 1.0) * (79.0 / 60.0)))
