import pytest
import os
import tempfile
import yaml
import pandas as pd
from cafr.cli import run_experiment
from cafr.data.schema import ACTIONS_COLUMNS, INVENTORY_LOG_COLUMNS

def test_cli_run_integration():
    """Test that the CLI can run a miniature E4 without crashing and satisfies the schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        out_dir = os.path.join(tmpdir, "outputs")
        
        cfg = {
            "n_skus": 2,
            "t_periods": 80,
            "splits": {"burn_in": 60},
            "sim": {"R": 3, "L": 2},
            "arms": ["(c)", "(f)", "A4", "(g)"],
            "out_dir": out_dir,
            "alpha_target": 0.95
        }
        
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
            
        # Run it
        run_experiment(config_path, seed=42)
        
        # Verify outputs exist
        actions_path = os.path.join(out_dir, "actions.parquet")
        inv_path = os.path.join(out_dir, "inventory_log.parquet")
        
        assert os.path.exists(actions_path)
        assert os.path.exists(inv_path)
        
        # Verify they can be read and have correct schemas
        act_df = pd.read_parquet(actions_path)
        inv_df = pd.read_parquet(inv_path)
        
        assert set(act_df.columns) == set(ACTIONS_COLUMNS.keys())
        assert set(inv_df.columns) == set(INVENTORY_LOG_COLUMNS.keys())
        
        # Number of execution periods = t_periods * n_skus * n_arms
        # = 80 * 2 * 4 = 640 (Inventory simulator returns full series including warmup)
        assert len(inv_df) == 640
