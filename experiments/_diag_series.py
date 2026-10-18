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
def main(sid):
    cfg = load_config("synthetic.yaml")
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    burn = int(cfg["splits"]["burn_in"])
    y = demand[demand.sku_id == sid].sort_values("period")["demand_observed"].to_numpy(float)
    statics = {sid: {k: float(params.loc[sid, k]) for k in
               ("lead_time","unit_cost","holding_rate","adi_init","cv2_init")}}
    train = pd.DataFrame([{"sku_id": sid, "period": t, "demand_observed": float(y[t]), **statics[sid]}
                          for t in range(burn)])
    pool = fit_global_pool(cfg, train, statics_by_sku=statics)
    sim = InventorySimulator(SimConfig.from_config(cfg)); base = arm_c_factory(pool, cfg, sid, 0.95, "observed")
    rec = []
    def level_fn(t, y_obs, cens, inv):
        S, a = base(t, y_obs, cens, inv); st = base.last_state
        rec.append((t, float(y_obs[-1]), float(st.mu_nonzero), float(st.sigma)))
        return S, a
    sim.run(sid, y, level_fn, arm="arm_c")
    r = np.array(rec)
    print("t   y   munz   sigma_model  e")
    for i in range(0, len(r), 4):
        t, yy, mz, sg = r[i]
        e = yy - (r[i-1,2] if i else np.nan)
        print(f"{int(t):3d} {yy:7.1f} {mz:8.2f} {sg:8.2f} {e:8.2f}")
if __name__ == "__main__":
    main(sys.argv[1])
