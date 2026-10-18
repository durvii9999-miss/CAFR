"""Ad-hoc diagnostic: print the monitor reading at each check for a few SKUs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.attribution import attribute
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor import Monitor
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import REPO_ROOT, load_config

DATA = REPO_ROOT / "data" / "synth"


def main(cause: str, n: int = 2, stride: int = 8) -> None:
    cfg = load_config("synthetic.yaml")
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    burn = int(cfg["splits"]["burn_in"])
    ids = sorted(params.index[params["true_cause"] == cause])[:n]
    series = {
        sid: g.sort_values("period")["demand_observed"].to_numpy(float)
        for sid, g in demand[demand["sku_id"].isin(ids)].groupby("sku_id")
    }
    statics = {
        sid: {
            "lead_time": float(params.loc[sid, "lead_time"]),
            "unit_cost": float(params.loc[sid, "unit_cost"]),
            "holding_rate": float(params.loc[sid, "holding_rate"]),
            "adi_init": float(params.loc[sid, "adi_init"]),
            "cv2_init": float(params.loc[sid, "cv2_init"]),
        }
        for sid in ids
    }
    train = pd.DataFrame(
        [{"sku_id": s, "period": t, "demand_observed": float(series[s][t]), **statics[s]}
         for s in ids for t in range(burn)]
    )
    pool = fit_global_pool(cfg, train, statics_by_sku=statics)

    for sid in ids:
        y = series[sid]
        sim = InventorySimulator(SimConfig.from_config(cfg))
        mon = Monitor(cfg, alpha_target=0.95, seed=0)
        base = arm_c_factory(pool, cfg, sid, 0.95, "observed")
        q_alpha = float(cfg["detection"]["c2_interval_alpha"])
        q_win = int(cfg["sim"]["quantile_window"])

        rows = []

        def level_fn(t, y_obs, cens, inv):
            S, a = base(t, y_obs, cens, inv)
            st = base.last_state
            res = base._residuals.to_array()
            rr = res[-q_win:] if res.size else res
            from cafr.monitor.features import conformal_interval
            lo, hi = conformal_interval(float(st.mu), rr, q_alpha, min_evidence=8)
            mon.update(t, float(y_obs[-1]), bool(cens[-1]),
                       {"mu": float(st.mu), "mu_nonzero": float(st.mu_nonzero),
                        "sigma": float(st.sigma),
                        "z": float(st.z), "lo": lo, "hi": hi, "alpha": q_alpha})
            if t >= burn and (t - burn) % stride == 0:
                rd = mon.read()
                att = attribute(rd, cfg)
                s = rd.statistics
                rows.append({
                    "t": t, "cause": att.cause, "conf": round(att.confidence, 2),
                    "fired": ",".join(att.fired),
                    "z_b": round(s["c1_z_b"], 2),
                    "step-drift": round(s["c1_step_bic"] - s["c1_drift_bic"], 2),
                    "c2": round(s["c2_c_gap"], 2),
                    "occ_gap": round(s["c3_occ_gap"], 2),
                    "r": round(s["c4_r"], 2), "r_prev": round(s["c4_r_prev"], 2),
                    "S_sz": round(s["c5_sz_max_s"], 2),
                    "S_occ": round(s["c5_occ_max_s"], 2),
                    "c6_chg": s["c6_changed"], "stab": round(s["c6_stability"], 2),
                    "c7": round(s["c7_csl_gap"], 2),
                    "signs": ",".join(f"{v:+.1f}" for v in rd.aux.get("signs", [])),
                    "cells": "/".join(rd.aux.get("cells", [])),
                    "init": rd.aux.get("init_cell", ""),
                })
            return S, a

        sim.run(sid, y, level_fn, arm="arm_c")
        print(f"\n=== {sid} ({cause}) adi_init={statics[sid]['adi_init']:.2f} "
              f"cv2_init={statics[sid]['cv2_init']:.2f}")
        print(pd.DataFrame(rows).to_string(index=False))
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C0",
         int(sys.argv[2]) if len(sys.argv) > 2 else 2)
