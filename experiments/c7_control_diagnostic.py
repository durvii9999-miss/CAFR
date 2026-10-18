"""C7 control diagnostic -- does constraint 6 fail WITHOUT the injection?

Gate 4.9's constraint 6 is meant to certify that the C7 injection did not destroy the
forecast's acceptability. But under ``observation_model = demand`` the forecaster never
sees the stock-outs the injection causes, so the C7 scale cannot move its residuals or
its predictive quantile (beyond floating-point boundary noise).

This runs the SAME real pipeline as ``build_synth_panel.py`` at ``scale = 1.0`` -- no
injection at all -- and writes the constraint-6 statistics. If they fail here too, then
constraint 6 is measuring the BASELINE forecaster's calibration, not the injection, and
the condition set cannot identify C7 as the cause.

This is a diagnostic, not a gate change: it changes no threshold and selects no scale.

    python experiments/c7_control_diagnostic.py --seed 42 --c7-skus 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.data.synth.c7_constraints import evaluate_c7_constraints
from cafr.forecasters.pooled import fit_global_pool
from cafr.utils.config import REPO_ROOT, load_config

OUT_DATA = REPO_ROOT / "data" / "synth"
OUT_RESULTS = REPO_ROOT / "results" / "synth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="synthetic.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c7-skus", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="1.0 = no injection; the control for constraint 6")
    args = ap.parse_args()

    cfg = load_config(args.config)
    syn = cfg["synthetic"]
    demand = pd.read_parquet(OUT_DATA / "demand_panel.parquet")
    params = pd.read_parquet(OUT_DATA / "generator_params.parquet")

    # Exactly the selection build_synth_panel.py uses.
    c7_ids = [
        s for s in sorted(params.loc[params["true_cause"] == "C7", "sku_id"])
    ][: args.c7_skus]
    series = {
        sid: grp.sort_values("period")["demand_true"].to_numpy(dtype="float64")
        for sid, grp in demand[demand["sku_id"].isin(c7_ids)].groupby("sku_id", sort=False)
    }

    burn = int(cfg["splits"]["burn_in"])
    statics_by_sku = {
        sid: {
            "lead_time": float(row["lead_time"]),
            "unit_cost": float(row["unit_cost"]),
            "holding_rate": float(row["holding_rate"]),
            "adi_init": float(row["adi_init"]),
            "cv2_init": float(row["cv2_init"]),
        }
        for sid, row in params.set_index("sku_id").iterrows()
        if sid in series
    }
    train_panel = pd.DataFrame(
        [
            {
                "sku_id": sid,
                "period": t,
                "demand_observed": float(v),
                **statics_by_sku[sid],
            }
            for sid, arr in series.items()
            for t, v in enumerate(arr[:burn])
        ]
    )
    pool = fit_global_pool(cfg, train_panel, statics_by_sku=statics_by_sku)
    factory = lambda pool_, cfg_, sku_id, alpha, fit_on: arm_c_factory(  # noqa: E731
        pool_, cfg_, sku_id, alpha, fit_on, margin=0.0
    )

    windows, summary = evaluate_c7_constraints(
        series,
        factory,
        cfg,
        sku_ids=sorted(series),
        tau_inj=int(syn["tau_inj"]),
        scale=float(args.scale),
        pool=pool,
    )

    row = summary.iloc[0].to_dict() if not summary.empty else {"n_windows": 0}
    row["control"] = "no_injection" if args.scale == 1.0 else "injected"
    row["seed"] = args.seed
    row["n_c7_skus_requested"] = args.c7_skus

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(
        OUT_RESULTS / f"c7_control_scale{args.scale:g}_windows.parquet", index=False
    )
    path = OUT_RESULTS / f"c7_control_scale{args.scale:g}.json"
    path.write_text(json.dumps([row], indent=2), encoding="utf-8")

    print(f"control (scale={args.scale:g}) on {row.get('n_skus', 0)} SKUs, "
          f"{row.get('n_windows', 0)} windows")
    for key in (
        "acceptability_pass_rate",
        "rmsse_pooled_pass_rate",
        "z_b_pass_rate",
        "coverage_pass_rate",
        "exposure_pass_rate",
        "constraint_6_pass",
        "constraint_7_pass",
    ):
        print(f"  {key:28s}: {row.get(key)}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
