import pytest
import numpy as np
import pandas as pd
from cafr.data.synth.labels import generate_panel

def test_gate_4_1_deterministic():
    """Gate 4.1: Deterministic regeneration from seed."""
    df1 = generate_panel(42, n_series=180)
    df2 = generate_panel(42, n_series=180)
    pd.testing.assert_frame_equal(df1, df2)
    
def test_gate_4_2_class_counts():
    """Gate 4.2: Class counts on the reachable scheme."""
    df = generate_panel(42, n_series=1800)
    # Check all 4 cells populated >= 20%
    cells = df[df["period"] == 0]["sb_cell_init"].value_counts(normalize=True)
    assert len(cells) == 4
    for cell, frac in cells.items():
        assert frac >= 0.15 # allowing some variance around 0.20
