"""Pre-injection calibration of the size residual on C0 control SKUs."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cafr.arms.arm_c import arm_c_factory
from cafr.forecasters.pooled import fit_global_pool
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import REPO_ROOT, load_config

DATA = REPO_ROOT / "data" / "synth"

def main(n=12):
    cfg = load_config("synthetic.yaml")
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    burn = int(cfg["splits"]["burn_in"])
    ids = sorted(params.index[params["true_cause"] == "C0"])[:n]
    series = {s: g.sort_values("period")["demand_observed"].to_numpy(float)
              for s, g in demand[demand["sku_id"].isin(ids)].groupby("sku_id")}
    statics = {s: {k: float(params.loc[s, k]) for k in
                   ("lead_time","unit_cost","holding_rate","adi_init","cv2_init")} for s in ids}
    train = pd.DataFrame([{"sku_id": s, "period": t, "demand_observed": float(series[s][t]), **statics[s]}
                          for s in ids for t in range(burn)])
    pool = fit_global_pool(cfg, train, statics_by_sku=statics)
    print("sku      p_init  nz  mean_e/sd_base  sd_base  sd_post  frac_pos  cusum24")
    for sid in ids:
        y = series[sid]; sim = InventorySimulator(SimConfig.from_config(cfg))
        base = arm_c_factory(pool, cfg, sid, 0.95, "observed")
        rec = []
        def level_fn(t, y_obs, cens, inv):
            S, a = base(t, y_obs, cens, inv)
            st = base.last_state
            rec.append((t, float(y_obs[-1]), float(st.mu_nonzero)))
            return S, a
        sim.run(sid, y, level_fn, arm="arm_c")
        r = np.array(rec)
        e = np.where(r[:,1] > 0, r[:,1] - np.r_[np.nan, r[:-1,2]], np.nan)
        pre = e[24:120]; base24 = e[:24]
        base24f = base24[np.isfinite(base24)]
        pref = pre[np.isfinite(pre)]
        print(f"{sid}  {float(params.loc[sid,'p_req'] if 'p_req' in params.columns else np.nan):.3f} "
              f"{int(np.isfinite(pref).sum()):3d}  {np.nanmean(pref)/np.nanstd(base24f):8.2f} "
              f"{np.nanstd(base24f):8.2f} {np.nanstd(pref):8.2f} {np.mean(pref>0):8.2f} "
              f"{np.nansum(np.maximum(0,pref/np.nanstd(base24f)-0.25)):8.1f}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
