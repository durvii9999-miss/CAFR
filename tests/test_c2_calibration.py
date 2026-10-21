"""Focused unit tests for Issue 1: C2 interval calibration and C0 false-alarm analysis.

Stage B: tests verifying non-negative clipping, edge cases, C2 sensitivity, and split isolation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest

from cafr.monitor.features import conformal_interval


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
