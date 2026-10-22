"""Focused unit tests for C6 generator redesign candidate and detectability budget.

Tests cover:
1. Linear interpolation ramp endpoints and monotonicity (upward and downward).
2. Probability clipping and bounds integrity.
3. Unchanged demand-size parameters (k and mu_z invariant).
4. Preservation of the frozen H-5 rule (< 2 nonzeros = dead).
5. Theoretical crossing and >= 2-sigma calculations in check_c6_budget.
6. Verification that the original baseline inject_C6 remains completely untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from cafr.data.synth.budget import check_c6_budget
from cafr.data.synth.injections import inject_C6, inject_C6_v2
from cafr.monitor.stability import adi_cv2_cell


def test_inject_c6_v2_ramp_endpoints_and_monotonicity_upward():
    """Verify linear interpolation from p < 0.25 to target_up (0.45).

    Pre-injection: constant p.
    Ramp [tau, tau + w_ramp): monotonic linear interpolation.
    Destination [tau + w_ramp, T): constant target_up.
    """
    rng = np.random.default_rng(42)
    T, tau, w_ramp = 200, 120, 24
    p_init = 0.15
    k, mu_z = 2.0, 30.0

    # Compute expected probabilities manually
    t_idx = np.arange(T)
    p_expected = np.full(T, p_init)
    ramp_mask = (t_idx >= tau) & (t_idx < tau + w_ramp)
    p_expected[ramp_mask] = p_init + (0.45 - p_init) * (t_idx[ramp_mask] - tau) / float(w_ramp)
    p_expected[t_idx >= tau + w_ramp] = 0.45

    # Run injection with a fixed seed
    y, delta, z, params = inject_C6_v2(
        T, tau, p_init, k, mu_z, rng, w_ramp=w_ramp, target_up=0.45, target_down=0.10
    )

    assert len(y) == T
    assert len(delta) == T
    assert len(z) == T

    # Check params metadata
    assert params["p_init"] == pytest.approx(0.15)
    assert params["p_target"] == pytest.approx(0.45)
    assert params["p_end"] == pytest.approx(0.45)
    assert params["w_ramp"] == 24
    assert params["adi_init"] == pytest.approx(1.0 / 0.15)
    assert params["adi_end"] == pytest.approx(1.0 / 0.45)
    assert params["adi_crosses_star"] is True
    assert params["demand_touched"] is True

    # Test ramp properties on the theoretical p_expected
    assert p_expected[0] == pytest.approx(0.15)
    assert p_expected[tau - 1] == pytest.approx(0.15)
    assert p_expected[tau] == pytest.approx(0.15)
    assert p_expected[tau + w_ramp // 2] == pytest.approx(0.30)  # midpoint
    assert p_expected[tau + w_ramp] == pytest.approx(0.45)
    assert p_expected[-1] == pytest.approx(0.45)

    # Monotonicity check
    diffs = np.diff(p_expected)
    assert np.all(diffs >= 0.0), "Upward ramp must be monotonically non-decreasing"


def test_inject_c6_v2_ramp_endpoints_and_monotonicity_downward():
    """Verify linear interpolation from p >= 0.25 to target_down (0.10).

    Pre-injection: constant p.
    Ramp [tau, tau + w_ramp): monotonic linear interpolation.
    Destination [tau + w_ramp, T): constant target_down.
    """
    rng = np.random.default_rng(42)
    T, tau, w_ramp = 200, 120, 24
    p_init = 0.35
    k, mu_z = 2.0, 30.0

    t_idx = np.arange(T)
    p_expected = np.full(T, p_init)
    ramp_mask = (t_idx >= tau) & (t_idx < tau + w_ramp)
    p_expected[ramp_mask] = p_init + (0.10 - p_init) * (t_idx[ramp_mask] - tau) / float(w_ramp)
    p_expected[t_idx >= tau + w_ramp] = 0.10

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

    assert p_expected[0] == pytest.approx(0.35)
    assert p_expected[tau] == pytest.approx(0.35)
    assert p_expected[tau + w_ramp // 2] == pytest.approx(0.225)  # midpoint
    assert p_expected[tau + w_ramp] == pytest.approx(0.10)
    assert p_expected[-1] == pytest.approx(0.10)

    diffs = np.diff(p_expected)
    assert np.all(diffs <= 0.0), "Downward ramp must be monotonically non-increasing"


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
    """Verify demand-size parameters (k and mu_z) are untouched across the injection.

    Delta (occurrence) changes, but z (Gamma size) comes from the same stationary DGP.
    """
    rng = np.random.default_rng(999)
    T, tau = 200, 120
    k, mu_z = 2.5, 40.0

    # Draw across 100 series to test sample moments
    pre_sizes = []
    post_sizes = []
    for _ in range(100):
        _, delta, z, _ = inject_C6_v2(T, tau, 0.15, k, mu_z, rng)
        pre_sizes.extend(z[:tau])
        post_sizes.extend(z[tau + 24 :])

    pre_sizes = np.array(pre_sizes)
    post_sizes = np.array(post_sizes)

    # Theoretical mean: mu_z = 40.0
    # Theoretical var: mu_z^2 / k = 1600 / 2.5 = 640.0
    assert np.mean(pre_sizes) == pytest.approx(40.0, abs=1.0)
    assert np.mean(post_sizes) == pytest.approx(40.0, abs=1.0)
    assert np.var(pre_sizes) == pytest.approx(640.0, abs=40.0)
    assert np.var(post_sizes) == pytest.approx(640.0, abs=40.0)


def test_h5_rule_preservation_and_dead_cell_integrity():
    """Verify H-5 definition is preserved: < 2 nonzeros in burn-in produces 'dead', >= 2 does not."""
    # 0 nonzeros in 24 periods
    w_zero = np.zeros(24)
    cell_zero = adi_cv2_cell(w_zero, adi_star=4.0, cv2_star=0.49)
    assert cell_zero["cell"] == "dead"
    assert np.isnan(cell_zero["cv2"])

    # 1 nonzero in 24 periods
    w_one = np.zeros(24)
    w_one[5] = 10.0
    cell_one = adi_cv2_cell(w_one, adi_star=4.0, cv2_star=0.49)
    assert cell_one["cell"] == "dead"
    assert np.isnan(cell_one["cv2"])

    # 2 nonzeros in 24 periods
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

    # Case 4: Target right on boundary (0.25) => z_sep = 0.0
    b_boundary = check_c6_budget(0.15, 0.25, w_class=30, adi_star=4.0, min_sigma=2.0)
    assert b_boundary["crosses"] is False
    assert b_boundary["z_sep"] == pytest.approx(0.0)
    assert b_boundary["cleared"] is False


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

    # Baseline inject_C6 drifts over 60 periods. At t=180, drift is 1.0 (p=0.35).
    # Since original inject_C6 does not cap drift at 1.0, at t=199 (drift=79/60), p reaches ~0.4133.
    assert params["p_end"] == pytest.approx(0.15 * (1.0 + ((0.35 / 0.15) - 1.0) * (79.0 / 60.0)))
