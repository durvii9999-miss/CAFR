"""Focused unit tests for Issue 1: C2 interval calibration and C0 false-alarm analysis.

Stage A: tests and diagnosis only. Production implementation is NOT modified yet.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest

from cafr.monitor.features import conformal_interval


def test_current_unclipped_conformal_interval_produces_negative_upper_bound():
    """Documents the diagnosed defect: when a 24-period window has all zero demands

    and fluctuating mu_hat, residuals are all negative. In current unclipped
    conformal_interval, hi collapses below zero, so y=0 is falsely counted as uncovered.
    """
    mu = 2.0
    # Simulate a stretch of zero demands where point forecast fluctuated around 2.0-2.5
    residuals = np.array([-2.5, -2.4, -2.3, -2.2, -2.1, -2.05, -2.02] * 4)[:24]
    assert np.all(residuals < 0)

    lo, hi = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)

    # In current unclipped implementation, hi is negative!
    assert hi < 0.0, f"Expected hi < 0 in unclipped implementation, got {hi}"
    # Because hi < 0, zero demand y=0 is not covered:
    y_test = 0.0
    is_covered = (lo <= y_test <= hi)
    assert not is_covered, "Zero demand was unexpectedly covered by a negative interval"


def test_clipped_conformal_interval_guarantees_non_negative_bounds():
    """Tests the proposed non-negative clipping: bounds must be >= 0 and cover y=0."""
    mu = 2.0
    residuals = np.array([-2.5, -2.4, -2.3, -2.2, -2.1, -2.05, -2.02] * 4)[:24]

    raw_lo, raw_hi = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)
    lo_clip = max(0.0, raw_lo)
    hi_clip = max(0.0, raw_hi)

    assert lo_clip >= 0.0
    assert hi_clip >= 0.0
    # Demand of 0 must be covered
    assert lo_clip <= 0.0 <= hi_clip


def test_c2_sensitivity_under_interval_scaling():
    """Tests that interval scaling (C2 injection dial gamma) is sensitively detected.

    Narrowing by gamma=0.2 must induce severe under-coverage.
    Widening by gamma=5.0 must induce over-coverage.
    """
    rng = np.random.default_rng(42)
    mu = 5.0
    # Realistic intermittent residuals: mostly negative (-mu) with occasional large positive sizes
    demand_draws = np.where(rng.uniform(size=30) < 0.25, rng.gamma(2.0, 5.0, size=30), 0.0)
    residuals = demand_draws - mu

    # Nominal scale (gamma=1.0)
    lo_nom, hi_nom = conformal_interval(mu, residuals, alpha=0.90, scale=1.0, min_evidence=8)
    lo_nom = max(0.0, lo_nom)
    hi_nom = max(0.0, hi_nom)

    # Injected scale (gamma=0.2, severe narrowing)
    lo_narrow, hi_narrow = conformal_interval(mu, residuals, alpha=0.90, scale=0.2, min_evidence=8)
    lo_narrow = max(0.0, lo_narrow)
    hi_narrow = max(0.0, hi_narrow)

    assert (hi_narrow - lo_narrow) < (hi_nom - lo_nom)

    # In a sequence of 30 future draws, the narrowed interval must fail coverage significantly
    future_draws = np.where(rng.uniform(size=50) < 0.25, rng.gamma(2.0, 5.0, size=50), 0.0)
    covered_nom = np.mean([lo_nom <= y <= hi_nom for y in future_draws])
    covered_narrow = np.mean([lo_narrow <= y <= hi_narrow for y in future_draws])

    assert covered_narrow < covered_nom
    assert covered_narrow < 0.70  # Massive coverage gap induced


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
