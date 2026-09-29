"""E4 runner and SKU executor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any

from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.bandit.thompson import PolicyBandit

def run_sku(
    sku_id: int, 
    y: np.ndarray, 
    arm_name: str, 
    arm_factory: Callable,
    pool: Any, 
    cfg: dict, 
    sku_meta: dict,
    bandit: PolicyBandit | None = None
) -> tuple[list[dict], list[dict]]:
    """Runs a single SKU through the arm's level function.
    
    Returns lists of dicts for actions and inventory_log.
    """
    T = len(y)
    splits = cfg.get("splits", {})
    W_burn = splits.get("burn_in", 60)
    alpha_target = cfg.get("alpha_target", 0.95)
    
    # Initialize the simulator
    unit_cost = sku_meta.get("unit_cost", 10.0)
    holding_rate = sku_meta.get("holding_rate", 0.20)
    margin = sku_meta.get("margin", 0.20) # (1 - unit_cost/price)
    
    sim_cfg = SimConfig(L=cfg.get("lead_time", 1), unmet_model=cfg.get("unmet_model", "lost_sales"), H=holding_rate, B_over_H=(1-margin)/margin if margin > 0 else 19.0)
    sim = InventorySimulator(sim_cfg)
    
    level_fn = arm_factory(pool, cfg, sku_id, alpha_target, fit_on=cfg.get("fit_on", "observed"), margin=margin)
    
    # We use the official simulator loop. Note: bandit updates require a custom loop in a full implementation,
    # but for testing the CLI we mock it using the standard run.
    res = sim.run(str(sku_id), y, level_fn, arm=arm_name)
    
    # Extract inventory log
    inv_df = res.log
    if "warmup" in inv_df.columns:
        inv_df = inv_df.drop(columns=["warmup"])
    
    # Build dummy actions log
    actions_log = []
    for t in range(W_burn, T):
        actions_log.append({
            "sku_id": str(sku_id),
            "period": t,
            "arm": arm_name,
            "attributed_cause": "C0",
            "confidence": 1.0,
            "remedy": "R0",
            "chosen_by": "rule",
            "exploratory": False,
            "switched": False,
            "dwell_override": False,
            "context_block": "full",
            "window_full": True,
        })
        
    return actions_log, inv_df.to_dict("records")

def run_arm(
    arm_name: str, 
    arm_factory: Callable,
    panel: np.ndarray, 
    meta_df: pd.DataFrame, 
    cfg: dict, 
    pool: Any,
    bandit: PolicyBandit | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs all SKUs for an arm sequentially."""
    all_actions = []
    all_inv = []
    
    for i in range(panel.shape[0]):
        sku_id = meta_df.iloc[i]["sku_id"] if "sku_id" in meta_df.columns else i
        sku_meta = meta_df.iloc[i].to_dict()
        y = panel[i]
        
        actions, inv = run_sku(sku_id, y, arm_name, arm_factory, pool, cfg, sku_meta, bandit)
        all_actions.extend(actions)
        all_inv.extend(inv)
        
    return pd.DataFrame(all_actions), pd.DataFrame(all_inv)
