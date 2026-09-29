import pytest
from cafr.churn.rules import can_switch, compute_churn_metrics, should_explore

def test_can_switch_dwell_time():
    """Test dwell time constraint and override."""
    # dwell = 3. 2 periods elapsed. Not allowed.
    assert not can_switch(periods_since_switch=2, c5_alarm=False, conf=0.8, d=3)
    # 4 periods elapsed. Allowed.
    assert can_switch(periods_since_switch=4, c5_alarm=False, conf=0.8, d=3)
    # 2 periods elapsed, but C5 alarmed. Override.
    assert can_switch(periods_since_switch=2, c5_alarm=True, conf=0.8, d=3)
    # 4 periods elapsed, but low confidence. Blocked.
    assert not can_switch(periods_since_switch=4, c5_alarm=False, conf=0.1, tau_conf=0.35)
    
def test_compute_churn_metrics():
    """Test model churn vs action churn exclusions."""
    decisions = [
        {"active_remedy": "R0", "base_method": "SBA", "exploratory": False},
        {"active_remedy": "R0", "base_method": "Croston", "exploratory": False}, # Model churn 1
        {"active_remedy": "R1", "base_method": "Croston", "exploratory": False}, # Action churn 1, NOT model churn (R1)
        {"active_remedy": "R3", "base_method": "Croston", "exploratory": False}, # Action churn 2, Model churn 2 (R3)
        {"active_remedy": "R3", "base_method": "Croston", "exploratory": True},  # Ignored (exploratory)
        {"active_remedy": "R7", "base_method": "Croston", "exploratory": False}, # Action churn 3, NOT model churn (R7)
    ]
    T = 10
    model_churn, action_churn = compute_churn_metrics(decisions, T)
    
    assert action_churn == 3 / T
    assert model_churn == 2 / T
    
def test_should_explore():
    """Test exploration distribution."""
    # Staggered hash rule over 90 periods
    expl_count = sum(1 for t in range(90) if should_explore(sku_id=42, period=t))
    # Expect ~ 10 out of 90
    assert 8 <= expl_count <= 12
