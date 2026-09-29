import pytest
import numpy as np
from cafr.remedies.policy_control import update_alpha
from cafr.remedies.forecast_remedies import apply_r1_bias, apply_r2_conformal

def test_r7_updates_safety_factor():
    """R7 must change alpha and NOT touch mu."""
    # Under-serviced: target 0.95, actual 0.80
    new_alpha = update_alpha(0.95, 0.95, 0.80, n_t=30, gamma=1.0)
    assert new_alpha > 0.95 # policy widens
    
    # Over-serviced: target 0.95, actual 0.99
    new_alpha_over = update_alpha(0.95, 0.95, 0.99, n_t=30, gamma=1.0)
    assert new_alpha_over < 0.95 # policy narrows

def test_r2_updates_coverage_only():
    """R2 must widen/narrow the interval to restore coverage, without moving mu_hat."""
    mu_hat = 10.0
    Q_alpha_before = 15.0
    
    # Large residuals (under-coverage)
    residuals = np.array([10.0, 12.0, 11.0, 10.5, 9.5])
    Q_after, meta = apply_r2_conformal(Q_alpha_before, mu_hat, 0.95, residuals)
    
    assert Q_after > mu_hat # Interval exists
    assert meta["mu_touched"] == False
    
def test_r1_updates_bias():
    """R1 must correct mu_hat using the mean of residuals."""
    mu_hat = 10.0
    e_sz = np.array([2.0, 3.0, 4.0]) # Positive bias (under-forecast)
    e_occ = np.array([0.1, 0.2])
    
    new_mu, meta = apply_r1_bias(mu_hat, e_sz, e_occ, gamma_sz=0.5, gamma_occ=1.0)
    
    # b_sz = 3.0. gamma_sz = 0.5 -> shift = 1.5
    # b_occ = 0.15. gamma_occ = 1.0 -> shift = 0.15
    # Total shift = 1.65. New mu = 11.65
    assert new_mu == pytest.approx(11.65)
    assert meta["mu_before"] == 10.0
    assert meta["shift"] == pytest.approx(1.65)
