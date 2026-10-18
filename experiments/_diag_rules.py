"""Diagnostic: which rule condition is binding, per cause, post-injection.

Runs a handful of SKUs per cause through the real pipeline and reports, over all
post-injection checks, how often each cause's trigger statistic crossed its bar and
how often the rule's extra conditions blocked it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor import Monitor
from cafr.monitor.features import conformal_interval
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import REPO_ROOT, load_config

DATA = REPO_ROOT / "data" / "synth"


def main(n_per: int = 20, stride: int = 4) -> None:
    cfg = load_config("synthetic.yaml")
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    burn = int(cfg["splits"]["burn_in"])
    tau = int(cfg["synthetic"]["tau_inj"])
    causes = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    ids: list[str] = []
    for c in causes:
        ids.extend(sorted(params.index[params["true_cause"] == c])[:n_per])
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

    rows = []
    for sid in ids:
        y = series[sid]
        import copy
        cfg_s = copy.deepcopy(cfg)
        cause = str(params.loc[sid, "true_cause"])
        if cause == "C2" and pd.notna(params.loc[sid, "gamma"]):
            cfg_s["sim"]["c2_quantile_scale"] = float(params.loc[sid, "gamma"])
        elif cause == "C2b" and pd.notna(params.loc[sid, "c2_calib_window"]):
            cfg_s["sim"]["quantile_window"] = int(params.loc[sid, "c2_calib_window"])
        elif cause == "C7":
            cfg_s["sim"]["c7_safety_factor_scale"] = float(params.loc[sid, "safety_scale"])
        sim = InventorySimulator(SimConfig.from_config(cfg_s))
        mon = Monitor(cfg_s, alpha_target=0.95, seed=0)
        base = arm_c_factory(pool, cfg_s, sid, 0.95, "observed")
        q_alpha = float(cfg_s["detection"]["c2_interval_alpha"])
        q_win = int(cfg_s["sim"]["quantile_window"])
        q_scale = float(cfg_s["sim"].get("c2_quantile_scale", 1.0))
        sku_records = []

        def level_fn(t, y_obs, cens, inv):
            S, a = base(t, y_obs, cens, inv)
            st = base.last_state
            res = base._residuals.to_array()
            lo, hi = conformal_interval(
                float(st.mu), res[-q_win:], q_alpha, scale=q_scale
            )
            mon.update(t, float(y_obs[-1]), bool(cens[-1]),
                       {"mu": float(st.mu), "mu_nonzero": float(st.mu_nonzero),
                        "sigma": float(st.sigma),
                        "z": float(st.z), "lo": lo, "hi": hi, "alpha": q_alpha})
            if t >= tau and (t - tau) % stride == 0:
                rd = mon.read()
                sku_records.append((t, rd))
            return S, a

        sim.run(sid, y, level_fn, arm="arm_c")
        for t, rd in sku_records:
            s, ev = rd.statistics, rd.evidence
            rows.append({
                "sku": sid,
                "true": params.loc[sid, "true_cause"],
                "t": t,
                # C1
                "c1_ev": ev["c1"],
                "c1_z": abs(s["c1_z_b"]) > 1.96,
                "c1_sign": bool(rd.aux["signs"]) and len(set(np.sign(rd.aux["signs"][-4:]))) == 1,
                "c1_drift": s["c1_drift_bic"] <= s["c1_step_bic"],
                # C2
                "c2_ev": ev["c2"],
                "c2_gap": abs(s["c2_c_gap"]) > 0.10,
                "c2_p": s["c2_p_val"] < 0.05,
                "_z": s["c1_z_b"],
                "_r": s["c4_r"],
                "_c2gap": abs(s["c2_c_gap"]),
                "_occgap": abs(s["c3_occ_gap"]),
                "_c4r": max(s["c4_r"], 1 / s["c4_r"])
                if (np.isfinite(s["c4_r"]) and s["c4_r"] > 0) else np.nan,
                "_S": max(abs(s["c5_sz_max_s"]), abs(s["c5_occ_max_s"])),
                "_c7gap": -s["c7_csl_gap"],
                # C3
                "c3_ev": ev["c3"],
                "c3_gap": abs(s["c3_occ_gap"]) > 0.10,
                "c3_p": s["c3_occ_p_val"] < 0.05,
                # C4
                "c4_ev": ev["c4"],
                "c4_r": max(s["c4_r"], 1 / s["c4_r"]) > 1.5
                and max(s["c4_r_prev"], 1 / s["c4_r_prev"]) > 1.5
                if (np.isfinite(s["c4_r"]) and np.isfinite(s["c4_r_prev"])
                    and s["c4_r"] > 0 and s["c4_r_prev"] > 0) else False,
                "c4_auc": (not np.isfinite(s["c3_auc"])) or s["c3_auc"] >= 0.65,
                # C5
                "c5_ev": ev["c5"],
                "c5_alarm": bool(s["c5_sz_alarm"]) or bool(s["c5_occ_alarm"]),
                "c5_step": s["c1_step_bic"] < s["c1_drift_bic"],
                # C6
                "c6_ev": ev["c6"],
                "c6_chg": bool(s["c6_changed"]),
                "c6_stab": s["c6_stability"] >= 0.90,
                # C7
                "c7_ev": ev["c7"],
                "c7_gap": s["c7_csl_gap"] < -0.10,
            })

    df = pd.DataFrame(rows)
    ok = lambda v, bar: (not np.isfinite(v)) or v <= bar  # noqa: E731
    df["c2_guard"] = [
        ok(abs(z) if np.isfinite(z) else np.nan, 1.96) and ok(r, 1.5)
        for z, r in zip(df["_z"], df["_r"])
    ]
    df["c3_guard"] = [ok(r, 1.5) for r in df["_r"]]
    cols = [c for c in df.columns if c not in ("sku", "true", "t", "_z", "_r")
            and not c.startswith("_")]
    agg = df.groupby("true")[cols].mean().round(2)
    print("\nMean condition-satisfaction rate over post-injection checks")
    print(agg.to_string())
    print("\nSKUs with at least one check meeting the FULL rule:")
    full = {
        "C1": df["c1_ev"] & df["c1_z"] & df["c1_sign"] & df["c1_drift"],
        "C2": df["c2_ev"] & df["c2_gap"] & df["c2_p"] & df["c2_guard"],
        "C3": df["c3_ev"] & df["c3_gap"] & df["c3_p"] & df["c3_guard"],
        "C4": df["c4_ev"] & df["c4_r"] & df["c4_auc"],
        "C5": df["c5_ev"] & df["c5_alarm"] & df["c5_step"],
        "C6": df["c6_ev"] & df["c6_chg"] & df["c6_stab"],
        "C7": df["c7_ev"] & df["c7_gap"] & df["c2_guard"],
    }
    for c, mask in full.items():
        sub = df[df["true"] == c]
        hit = sub.loc[mask.loc[sub.index]]
        print(f"  {c}: {hit['sku'].nunique()}/{sub['sku'].nunique()} SKUs fire at least once "
              f"(checks: {len(hit)}/{len(sub)})")

    # Closed-loop detectability: AUC of each cause's OWN trigger statistic, cause-c
    # checks vs C0 control checks, over the post-injection window. Step 4's gate 4.3 was
    # measured against an idealised non-adaptive residual; this is the same question in
    # the loop the monitor actually runs in.
    control = df[df["true"] == "C0"]
    stat_for = {"C1": "_z", "C2": "_c2gap", "C3": "_occgap",
                "C4": "_c4r", "C5": "_S", "C7": "_c7gap"}
    print("\nClosed-loop AUC of each cause's own statistic (cause vs C0)")
    for c, col in stat_for.items():
        pos = df.loc[df["true"] == c, col].to_numpy(float)
        neg = control[col].to_numpy(float)
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        if pos.size < 5 or neg.size < 5:
            print(f"  {c}: n_pos={pos.size} n_neg={neg.size} -- not estimable")
            continue
        # rank-based (Mann-Whitney) AUC, ties averaged
        allv = np.r_[pos, neg]
        order = allv.argsort(kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, allv.size + 1)
        # average ties
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        sums = np.zeros(cnt.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
        auc = (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (
            pos.size * neg.size
        )
        print(f"  {c}: AUC {auc:.3f}  (n_pos={pos.size}, n_neg={neg.size})")
    chg = df.groupby("true")["c6_chg"].mean().round(3)
    print("\nC6 change rate by truth:", chg.to_dict())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
