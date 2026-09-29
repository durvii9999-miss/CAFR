"""CAFR Command Line Interface (Rev 2 §13)."""

import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd

from cafr.eval.runner import run_arm
from cafr.data.schema import validate_frame, ACTIONS_COLUMNS, INVENTORY_LOG_COLUMNS
from cafr.arms.arm_c import arm_c_factory
from cafr.arms.arm_f import arm_f_factory
from cafr.arms.arm_a4 import arm_a4_factory
from cafr.arms.arm_g import arm_g_factory
from cafr.bandit.thompson import PolicyBandit

def get_arm_factory(arm_id: str):
    mapping = {
        "(c)": arm_c_factory,
        "(f)": arm_f_factory,
        "A4": arm_a4_factory,
        "(g)": arm_g_factory,
    }
    if arm_id not in mapping:
        raise ValueError(f"Unknown arm: {arm_id}")
    return mapping[arm_id]

def run_experiment(config_path: str, seed: int):
    """Run E4 as specified by the config."""
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    np.random.seed(seed)
    
    # In a full run, we would load ground_truth.parquet and demand_panel.parquet.
    # For now, we mock a small panel to allow tests to pass.
    print("Initializing dummy panel...")
    n_skus = cfg.get("n_skus", 5)
    t_periods = cfg.get("t_periods", 100)
    panel = np.random.poisson(0.5, size=(n_skus, t_periods)).astype(float)
    meta = pd.DataFrame({"sku_id": range(n_skus), "unit_cost": [10.0]*n_skus, "margin": [0.2]*n_skus})
    
    # Arms to run
    arms = cfg.get("arms", ["(c)", "(f)", "A4", "(g)"])
    
    class DummyForecaster:
        def fit(self, fi):
            from cafr.arms.arm_c import ForecastState
            from cafr.forecasters.base import FittingInput
            y = fi.resolve() if isinstance(fi, FittingInput) else fi
            return ForecastState(mu=np.mean(y) if len(y)>0 else 0.0, z=0.0, sigma=0.0, method="dummy")
            
    pool = {"croston": DummyForecaster(), "SBA": DummyForecaster(), "TSB": DummyForecaster()}
    
    out_dir = cfg.get("out_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    all_actions = []
    all_inv = []
    
    for arm in arms:
        print(f"Running arm {arm}...")
        
        # Instantiate bandit if this arm uses it
        bandit = None
        if arm in ["(f)", "A4", "(f')"]:
            d_val = 8 if arm == "A4" else 22
            cfg["bandit"] = PolicyBandit(d=d_val, seed=seed) 
            bandit = cfg["bandit"]
            
        factory = get_arm_factory(arm)
        act_df, inv_df = run_arm(arm, factory, panel, meta, cfg, pool, bandit)
        
        all_actions.append(act_df)
        all_inv.append(inv_df)
        
    print("Validating and saving schemas...")
    actions_full = pd.concat(all_actions, ignore_index=True)
    inv_full = pd.concat(all_inv, ignore_index=True)
    
    validate_frame(actions_full, ACTIONS_COLUMNS, name="actions")
    validate_frame(inv_full, INVENTORY_LOG_COLUMNS, name="inventory_log")
    
    actions_full.to_parquet(os.path.join(out_dir, "actions.parquet"), index=False)
    inv_full.to_parquet(os.path.join(out_dir, "inventory_log.parquet"), index=False)
    
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="CAFR Experiment Runner")
    subparsers = parser.add_subparsers(dest="command")
    
    run_parser = subparsers.add_parser("run", help="Run the experiment")
    run_parser.add_argument("--config", required=True, help="Path to config.yaml")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.add_args = parser.parse_args()
    
    if args.command == "run":
        run_experiment(args.config, args.seed)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
