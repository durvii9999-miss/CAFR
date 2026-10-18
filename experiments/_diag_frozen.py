"""Is C1 undetectable because the forecaster absorbs it, or because the injection is small?

Runs a few SKUs through the real closed loop and computes C1's ``z_b`` two ways on the
SAME demand series:

  adaptive  ``e = z - mu_nonzero_hat(t-1)``   -- what the monitor sees
  frozen    ``e = z - mu_nonzero_hat(tau-1)`` -- the counterfactual "the forecaster did
                                                 not move", i.e. exactly the object
                                                 Step 4's gate 4.3 measured
                                                 (``e_sz = z - mu_z`` with the true,
                                                 constant pre-injection mean)

If the adaptive number is at the control and the frozen number is not, the injection is
real and the forecaster removed it. If both are at the control, the injection is too
small for the statistic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor.features import compute_bias_stats
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import REPO_ROOT, load_config

DATA = REPO_ROOT / "data" / "synth"


def main(n_per: int = 6, w: int = 24, floor: int = 8) -> None:
    cfg = load_config("synthetic.yaml")
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    tau = int(cfg["synthetic"]["tau_inj"])
    gap = 8  # checks are reported every `gap` periods after tau

    causes = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    ids: list[str] = []
    for c in causes:
        ids.extend(sorted(params.index[params.true_cause == c])[:n_per])
    series = {
        s: g.sort_values("period")["demand_observed"].to_numpy(float)
        for s, g in demand[demand.sku_id.isin(ids)].groupby("sku_id")
    }
    statics = {
        s: {k: float(params.loc[s, k]) for k in
            ("lead_time", "unit_cost", "holding_rate", "adi_init", "cv2_init")}
        for s in ids
    }
    burn = int(cfg["splits"]["burn_in"])
    train = pd.DataFrame(
        [{"sku_id": s, "period": t, "demand_observed": float(series[s][t]), **statics[s]}
         for s in ids for t in range(burn)]
    )
    pool = fit_global_pool(cfg, train, statics_by_sku=statics)

    rows = []
    for sid in ids:
        y = series[sid]
        sim = InventorySimulator(SimConfig.from_config(cfg))
        base = arm_c_factory(pool, cfg, sid, 0.95, "observed")
        rec: list[tuple[int, float, float]] = []

        def level_fn(t, y_obs, cens, inv):
            S, a = base(t, y_obs, cens, inv)
            st = base.last_state
            rec.append((int(t), float(y_obs[-1]), float(st.mu_nonzero)))
            return S, a

        sim.run(sid, y, level_fn, arm="arm_c")
        r = np.array(rec)
        # forecast stored at t-1 predicts t
        munz_prev = np.r_[np.nan, r[:-1, 2]]
        z = r[:, 1]
        frozen = float(r[tau - 1, 2]) if tau < len(r) else np.nan
        e_ad = np.where(z > 0, z - munz_prev, np.nan)
        e_fr = np.where(z > 0, z - frozen, np.nan)

        for t in range(tau, len(r), gap):
            a_now = e_ad[max(0, t - w + 1) : t + 1]
            f_now = e_fr[max(0, t - w + 1) : t + 1]
            ca = compute_bias_stats(a_now, np.ones(a_now.size, bool), min_evidence=floor)
            cf = compute_bias_stats(f_now, np.ones(f_now.size, bool), min_evidence=floor)
            rows.append({
                "sku": sid, "true": params.loc[sid, "true_cause"], "t": t,
                "z_adaptive": ca.get("z_b", np.nan),
                "z_frozen": cf.get("z_b", np.nan),
            })

    df = pd.DataFrame(rows)
    ok = lambda x: bool(np.isfinite(x) and abs(x) > 1.96)  # noqa: E731
    df["adaptive_cross"] = df["z_adaptive"].map(ok)
    df["frozen_cross"] = df["z_frozen"].map(ok)
    out = df.groupby("true")[["adaptive_cross", "frozen_cross"]].mean().round(3)
    out["median_z_adaptive"] = df.groupby("true")["z_adaptive"].median().round(2)
    out["median_z_frozen"] = df.groupby("true")["z_frozen"].median().round(2)
    print("\nC1 |z_b| > 1.96 crossing rate, adaptive vs frozen forecast")
    print(out.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
