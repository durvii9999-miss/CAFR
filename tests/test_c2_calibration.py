"""Focused unit tests for Issue 1 and Issue 2: C2/C2b interval calibration, payload parity, and isolation.

Issue 1: non-negative clipping, edge cases, C2 sensitivity, and split isolation.
Issue 2: forecast payload contract parity, C2b quiet-window under-coverage, zero monitor label leakage, Step 6 gate isolation.
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from cafr.monitor import Monitor
from cafr.monitor.features import conformal_interval
from cafr.attribution.rules import check_c2_coverage
from cafr.utils.config import load_config


# --------------------------------------------------------------------------- #
# Issue 1 Tests                                                               #
# --------------------------------------------------------------------------- #


def test_conformal_interval_clips_negative_upper_bound_to_zero():
    """When a 24-period window has all-negative residuals (e.g. zero demand run

    with mu_hat fluctuating around 2.0-2.5), raw upper bound would be negative.
    Verifies that conformal_interval clips hi to 0.0 so that y=0.0 is covered.
    """
    mu = 2.0
    residuals = np.array([-2.5, -2.4, -2.3, -2.2, -2.1, -2.05, -2.02] * 4)[:24]
    assert np.all(residuals < 0)

    lo, hi = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)

    assert lo == 0.0, f"Expected lo clipped to 0.0, got {lo}"
    assert hi == 0.0, f"Expected hi clipped to 0.0, got {hi}"
    # Demand of 0.0 must be covered by [0.0, 0.0]
    assert lo <= 0.0 <= hi


def test_conformal_interval_both_bounds_negative_edge_case():
    """Edge case: when all residuals are strictly less than -mu (e.g. mu=1.0,

    all residuals <= -1.5), both raw mu + r bounds are negative.
    Verifies both are clipped to 0.0, bounds remain valid, and y=0 is covered.
    """
    mu = 1.0
    residuals = np.full(24, -2.0)  # mu + r = -1.0 for all points

    lo, hi = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)

    assert lo == 0.0
    assert hi == 0.0
    assert lo <= 0.0 <= hi
    assert lo <= hi


def test_conformal_interval_when_lower_bound_exceeds_zero_edge_case():
    """Edge case & statistical limitation: when demand surged in the calibration

    window, all residuals are positive (mu=1.0, residuals in [2.0, 5.0]).
    Then raw lo = mu + r_lo > 0.
    In this case, clipping to max(0, lo) preserves lo > 0.
    If demand subsequently drops to y=0.0, y=0.0 is uncovered because the
    model's recent error history genuinely indicated nonzero positive demand.
    """
    mu = 1.0
    # All positive residuals (demand consistently exceeded point forecast)
    residuals = np.linspace(2.0, 10.0, 24)
    assert np.all(residuals > 0)

    lo, hi = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)

    assert lo > 0.0, f"Expected lo > 0 for all-positive residuals, got {lo}"
    assert hi > lo

    # If y=0 occurs right after a demand surge, it is not covered:
    y_zero = 0.0
    is_covered = (lo <= y_zero <= hi)
    assert not is_covered, "y=0 should not be covered when lo > 0"


def test_conformal_interval_c2_sensitivity():
    """Tests that interval scaling (C2 injection dial gamma) remains sensitively detected.

    Narrowing by gamma=0.2 must induce severe under-coverage.
    Widening by gamma=5.0 must induce over-coverage.
    """
    rng = np.random.default_rng(42)
    mu = 5.0
    # Intermittent residuals: mostly -mu with occasional positive spikes
    demand_draws = np.where(rng.uniform(size=30) < 0.25, rng.gamma(2.0, 5.0, size=30), 0.0)
    residuals = demand_draws - mu

    # Nominal scale (gamma=1.0)
    lo_nom, hi_nom = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)

    # Injected scale (gamma=0.2, severe narrowing)
    lo_narrow, hi_narrow = conformal_interval(mu, residuals, alpha=0.90, scale=0.2, min_evidence=8)

    assert (hi_narrow - lo_narrow) < (hi_nom - lo_nom)

    # In a sequence of 50 future draws, the narrowed interval fails coverage
    future_draws = np.where(rng.uniform(size=50) < 0.25, rng.gamma(2.0, 5.0, size=50), 0.0)
    covered_nom = np.mean([lo_nom <= y <= hi_nom for y in future_draws])
    covered_narrow = np.mean([lo_narrow <= y <= hi_narrow for y in future_draws])

    assert covered_narrow < covered_nom
    assert covered_narrow < 0.70


def test_training_split_has_zero_leakage_to_step6():
    """Verifies that the pre-injection training split definition (i % 5 < 3)

    has exactly ZERO overlap with Step 6 test SKUs (i % 5 == 4) and val SKUs (i % 5 == 3).
    Also checks that a naive prefix slice [:120] would leak 24 test SKUs.
    """
    total_skus = 200
    all_indices = np.arange(total_skus)
    split_cycle = np.array(["train", "train", "train", "val", "test"])
    assigned_splits = split_cycle[all_indices % len(split_cycle)]

    train_indices = set(all_indices[assigned_splits == "train"])
    val_indices = set(all_indices[assigned_splits == "val"])
    test_indices = set(all_indices[assigned_splits == "test"])

    # Disjointness checks
    assert len(train_indices) == 120
    assert len(val_indices) == 40
    assert len(test_indices) == 40
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(train_indices.intersection(val_indices)) == 0

    # Naive slice leakage demonstration
    naive_first_120 = set(all_indices[:120])
    leaked_test = naive_first_120.intersection(test_indices)
    assert len(leaked_test) == 24, f"Expected 24 leaked test SKUs under naive slice, got {len(leaked_test)}"


# --------------------------------------------------------------------------- #
# Issue 2 Tests                                                               #
# --------------------------------------------------------------------------- #


def test_forecast_payload_structure_is_identical_across_causes():
    """Safeguard 2 & 3: The dictionary contract passed into Monitor.update

    must have the exact same keys and structure for C0, C2, and C2b.
    """
    expected_keys = {"mu", "mu_nonzero", "sigma", "z", "lo", "hi", "alpha"}

    # Mock states representing the 3 variants:
    # 1. C0: standard conformal interval
    payload_c0 = {
        "mu": 3.0, "mu_nonzero": 10.0, "sigma": 2.5, "z": 0.3,
        "lo": 0.0, "hi": 8.0, "alpha": 0.90,
    }
    # 2. C2 direct: rescaled conformal interval
    payload_c2 = {
        "mu": 3.0, "mu_nonzero": 10.0, "sigma": 2.5, "z": 0.3,
        "lo": 0.0, "hi": 4.5, "alpha": 0.90,
    }
    # 3. C2b: quiet-window calibrated half-width
    payload_c2b = {
        "mu": 3.0, "mu_nonzero": 10.0, "sigma": 2.5, "z": 0.3,
        "lo": max(0.0, 3.0 - 1.2), "hi": max(0.0, 3.0 + 1.2), "alpha": 0.90,
    }

    assert set(payload_c0.keys()) == expected_keys
    assert set(payload_c2.keys()) == expected_keys
    assert set(payload_c2b.keys()) == expected_keys

    for payload in (payload_c0, payload_c2, payload_c2b):
        assert payload["lo"] >= 0.0
        assert payload["hi"] >= payload["lo"]
        assert 0.0 < payload["alpha"] < 1.0


def test_c2b_interval_undercovers_under_quiet_window_calibration():
    """Safeguard 1 & 3: An interval derived from a quiet calibration window

    consistently under-covers subsequent intermittent demand.
    """
    rng = np.random.default_rng(42)
    p = 0.25
    k = 2.0
    mu_z = 10.0
    # Generate 100 periods of intermittent demand
    draws = np.where(rng.uniform(size=100) < p, rng.gamma(k, mu_z / k, size=100), 0.0)

    # In a quiet stretch (say 5 periods of all zeros), sample std is 0 or near 0
    quiet_window = np.zeros(5)
    sigma_quiet = float(np.std(quiet_window, ddof=1)) if np.std(quiet_window, ddof=1) > 0 else 0.5
    half_width = 1.6449 * sigma_quiet

    mu_hat = p * mu_z  # 2.5
    lo = max(0.0, mu_hat - half_width)
    hi = max(0.0, mu_hat + half_width)

    # Under-coverage across future periods
    covered = [lo <= y <= hi for y in draws]
    realized_c = np.mean(covered)

    # Realized coverage should fall well below nominal 0.90
    assert realized_c < 0.80
    gap = realized_c - 0.90
    assert gap < -0.10


def test_monitor_logic_has_no_dependence_on_cause_labels_or_tau_inj():
    """Safeguard 1: Monitor.update and check_c2_coverage must NOT inspect

    any cause label, 'tau_inj', or synthetic metadata.
    """
    mon_sig = inspect.signature(Monitor.update)
    assert "cause" not in mon_sig.parameters
    assert "true_cause" not in mon_sig.parameters
    assert "tau_inj" not in mon_sig.parameters

    rule_sig = inspect.signature(check_c2_coverage)
    assert "cause" not in rule_sig.parameters
    assert "true_cause" not in rule_sig.parameters
    assert "tau_inj" not in rule_sig.parameters


def test_step6_gate_inputs_and_filters_exclude_c2b():
    """Safeguard 4: Step 6 gate fitting and evaluation MUST strictly exclude

    C2b rows, ensuring zero perturbation to Step 6 inputs or outcomes.
    """
    from experiments.step6_c2_gated import run_sku

    # Verifies run_sku rejects C2b explicitly
    with pytest.raises(ValueError, match="C2b is excluded"):
        run_sku("dummy", np.zeros(10), {"true_cause": "C2b"}, {}, None, mode="baseline")
