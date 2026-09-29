"""Synthetic panel generator and labeller (Rev 2 §15.4)."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .generator import get_cell_params, generate_baseline, CELLS
from .injections import inject_C0, inject_C1, inject_C3, inject_C4, inject_C5, inject_C6

__all__ = ["generate_panel"]

INJECTORS = {
    "C0": inject_C0,
    "C1": inject_C1,
    "C3": inject_C3,
    "C4": inject_C4,
    "C5": inject_C5,
    "C6": inject_C6,
}

def generate_panel(seed: int, n_series: int = 1800) -> pd.DataFrame:
    """Generate the synthetic ground truth panel.
    
    1800 series total:
    - 7 single causes (C1-C7) + C0 = 8 categories.
    Wait, C2 and C7 are causes too!
    Let's distribute n_series evenly. 1800 / 9 = 200 per category (including C2, C2b, C7, ambiguous).
    Let's just generate the demand DGP first.
    """
    rng = np.random.default_rng(seed)
    T = 200
    tau_inj = 120
    
    # 200 series for C0, C1, C3, C4, C5, C6, C7, C2, C2b? 
    # Spec says: Single-cause per cause = 200x7 = 1400. Control = 200. Ambiguous = 200. Total = 1800.
    causes = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "ambiguous"]
    series_per_cause = n_series // len(causes)
    
    rows = []
    
    for cause in causes:
        for i in range(series_per_cause):
            sku_id = f"{cause}_{i}"
            cell = rng.choice(CELLS)
            p, k, mu_z = get_cell_params(cell, rng)
            
            if cause in INJECTORS:
                y, delta, z, params = INJECTORS[cause](T, tau_inj, p, k, mu_z, rng)
            else:
                # C2, C7, ambiguous demand is just C0
                y, delta, z, params = inject_C0(T, tau_inj, p, k, mu_z, rng)
            
            # C2, C7 specific params
            if cause == "C2":
                params["gamma"] = rng.choice([0.5, 2.0])
            elif cause == "C7":
                params["safety_scale"] = 0.60
                
            for t in range(T):
                rows.append({
                    "sku_id": sku_id,
                    "period": t,
                    "demand_true": y[t],
                    "true_cause": cause,
                    "cause_active": int(t >= tau_inj),
                    "adi_full": 0.0, # Filled later or just kept for schema
                    "cv_squared_full": 0.0,
                    "sb_cell_init": cell,
                    # We can pack params into a JSON string if needed, or separate columns
                    "injection_param": str(params),
                })
                
    df = pd.DataFrame(rows)
    return df
