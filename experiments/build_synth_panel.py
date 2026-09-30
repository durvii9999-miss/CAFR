"""Build the labelled synthetic panel and run Step 4's verification (Rev 2 §15.4-§15.5).

    python experiments/build_synth_panel.py --seed 42
    python experiments/build_synth_panel.py --seed 42 --c7-skus 60

Writes:

* ``data/synth/demand_panel.parquet``      -- the §27 demand contract
* ``data/synth/ground_truth.parquet``      -- the §27 ground-truth contract
* ``data/synth/generator_params.parquet``  -- per-SKU audit: parameters, cell, every
  constraint's flag
* ``results/synth/panel_report.json``      -- panel composition and constraint verdicts
* ``results/synth/c7_constraints.json``    -- gates 4.9 / 4.10 evidence

**Why a script and not a notebook.** Every number in the report must have a manifest.
A number produced by a notebook cell has none (handoff §14).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.data.synth.c7_constraints import evaluate_c7_constraints, save_c7_report
from cafr.data.synth.labels import generate_labelled_panel
from cafr.data.synth.injections import INJECTORS
from cafr.data.synth.budget import check_c4_budget
from cafr.forecasters.pooled import fit_global_pool
from cafr.utils.config import REPO_ROOT, config_hash, load_config

OUT_DATA = REPO_ROOT / "data" / "synth"
OUT_RESULTS = REPO_ROOT / "results" / "synth"


def _panel_report(
    demand: pd.DataFrame,
    ground_truth: pd.DataFrame,
    params: pd.DataFrame,
    seed: int,
) -> dict:
    """The panel's composition and every constraint's pass rate."""
    cells = params["sb_cell_init"].value_counts(normalize=True).to_dict()
    per_panel = params["panel"].value_counts().to_dict()
    per_cause = params["true_cause"].value_counts().to_dict()

    c4 = params[params["true_cause"] == "C4"]
    c5 = params[params["true_cause"] == "C5"]
    c7 = params[params["true_cause"] == "C7"]
    c2 = params[params["true_cause"].isin(["C2", "C2b"])]

    # Every constraint block has the SAME shape: ``checked`` / ``pass`` / ``share`` /
    # ``status`` / ``note``. The first version let each block carry its own key names
    # (``pass_with_margin``, ``p_shift_max``, a bare string), and the printer's
    # ``blk["pass"]`` raised KeyError on constraint 5. Uniform blocks make an absent
    # verdict impossible: a constraint that cannot be reduced to a pass count says so
    # in ``status`` instead of omitting the key.
    def _block(*, checked: int, n_pass: int | None, status: str, note: str, **extra):
        blk = {
            "checked": int(checked),
            "pass": int(n_pass) if n_pass is not None else None,
            "share": (float(n_pass) / checked) if (n_pass is not None and checked) else None,
            "status": status,
            "note": note,
        }
        blk.update(extra)
        return blk

    c1 = params[params["true_cause"] == "C1"]
    c5_occ = c5[c5["c5_target"] == "occurrence"] if "c5_target" in c5 else c5.iloc[:0]

    return {
        "seed": seed,
        "config_hash": config_hash(load_config("synthetic.yaml")),
        "n_series": int(len(params)),
        "n_periods": int(demand["period"].max()) + 1,
        "panel_sizes": per_panel,
        "cause_sizes": per_cause,
        "sb_cell_init_shares": cells,
        "constraints": {
            "1_c4_band_invariance": _block(
                checked=len(c4),
                n_pass=int(c4["c4_band_ok"].sum()),
                status="enforced",
                note=(
                    "C4 resamples k until the post-injection CV^2 stays in the same "
                    "SB band. This constraint is NEVER relaxed: violating it would let "
                    "the C4 detector separate on the band rather than on dispersion."
                ),
                n_k_resampled=int(c4["k_resampled_for_constraint_1"].sum())
                if "k_resampled_for_constraint_1" in c4
                else None,
            ),
            "3_c5_detectability_budget": _block(
                checked=len(c5),
                n_pass=int(c5["c5_budget_cleared"].sum()),
                status="scaled-then-flagged",
                note=(
                    "The injector scales the step up until it clears; SKUs that still "
                    "fail are FLAGGED in ground_truth, never dropped. The occurrence "
                    "branch is arithmetically infeasible for small p (the achievable "
                    "step in sd units is bounded by the range of p, while the required "
                    "step grows as 1/p), so those SKUs take the SIZE branch instead of "
                    "relaxing the budget."
                ),
                n_occurrence=len(c5_occ),
                n_occurrence_infeasible=int(
                    (~c5_occ["c5_occurrence_feasible"].eq(True)).sum()
                )
                if "c5_occurrence_feasible" in c5_occ
                else None,
            ),
            "5_c2_coverage_gap": _block(
                checked=len(c2),
                n_pass=int(c2["gap_clears_with_margin"].eq(True).sum()),
                status="enforced",
                note=(
                    "The coverage gap must clear tau_cov + margin on BOTH panels: C2 "
                    "(a direct interval injection, demand untouched) and C2b (a genuine "
                    "modelling error -- a per-period sigma applied to the R-period "
                    "protection interval, the documented 42.3 % understatement)."
                ),
                both_variants=sorted(c2["true_cause"].unique().tolist()),
            ),
            "9_c4_detectability_budget": _block(
                checked=len(c4),
                n_pass=int(c4["c4_budget_met"].sum()),
                status="flagged-not-dropped",
                note=(
                    "C4's log-dispersion statistic needs >= 8 nonzero sizes in "
                    "W_sz = 24, which requires p >= 0.333, so high-ADI SKUs cannot "
                    "clear it from a 24-period window. Reported as not evaluable at "
                    "W_sz, and flagged in ground_truth (Rev 2 §15.2, constraint 9)."
                ),
            ),
            "8_c1_occurrence_untouched": _block(
                checked=len(c1),
                n_pass=int(
                    (
                        (c1["c1_p_shift"] == 0.0)
                        & (c1["c1_k_shift"] == 0.0)
                    ).sum()
                ),
                status="enforced",
                note=(
                    "inject_C1 records p_shift = k_shift = 0.0: the occurrence rate and "
                    "the shape are untouched, so only the size distribution can carry "
                    "the change. A nonzero p_shift would move C1 onto C6's statistic."
                ),
            ),
            "6_7_c7_simulator_pre_flight": _block(
                checked=0,
                n_pass=None,
                status="see c7_constraints.json",
                note=(
                    "Constraints 6 and 7 are NOT generator-side: they require the real "
                    "simulator, so they are verified by the pre-flight below and "
                    "recorded in results/synth/c7_constraints.json."
                ),
            ),
        },
        "c7_parameters": {
            "n": int(len(c7)),
            "safety_scale": sorted(
                float(v) for v in c7["safety_scale"].dropna().unique()
            ),
            "fit_on": sorted(str(v) for v in c7["c7_fit_on"].dropna().unique()),
            "demand_touched": bool(c7["demand_touched"].any()),
        },
        "c2_gamma_range": [
            float(c2["gamma"].min()) if c2["gamma"].notna().any() else None,
            float(c2["gamma"].max()) if c2["gamma"].notna().any() else None,
        ],
        "dead_share": float(cells.get("dead", 0.0)),
        "n_c4_below_budget": int((~c4["c4_budget_met"]).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="synthetic.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c7-skus", type=int, default=60,
                    help="how many C7 series to run through the inventory pre-flight")
    ap.add_argument("--skip-c7", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    syn = cfg["synthetic"]
    t0 = time.time()

    print(f"generating panel (seed={args.seed}) ...")
    demand, ground_truth, params = generate_labelled_panel(
        args.seed,
        T=int(syn["T"]),
        tau_inj=int(syn["tau_inj"]),
        burn_in=int(cfg["splits"]["burn_in"]),
        n_per_cause=int(syn["n_per_cause"]),
        n_control=int(syn["n_control"]),
        n_mixture=int(syn["n_mixture"]),
        n_c2b=int(syn["n_c2b"]),
    )
    print(
        f"  {params['sku_id'].nunique()} series x {int(syn['T'])} periods "
        f"in {time.time() - t0:.1f}s"
    )

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(OUT_DATA / "demand_panel.parquet", index=False)
    ground_truth.to_parquet(OUT_DATA / "ground_truth.parquet", index=False)
    params.to_parquet(OUT_DATA / "generator_params.parquet", index=False)
    print(f"  wrote {OUT_DATA}")

    report = _panel_report(demand, ground_truth, params, args.seed)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "panel_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print("\npanel composition")
    for k, v in report["panel_sizes"].items():
        print(f"  {k:<10} {v}")
    print("  sb_cell_init:")
    for k, v in sorted(report["sb_cell_init_shares"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20} {v:.4f}")

    print("\nconstraints (generator-side)")
    for name, blk in report["constraints"].items():
        if blk["share"] is not None:
            print(
                f"  {name:<34} {blk['pass']}/{blk['checked']}  "
                f"({blk['share']:.4f})   [{blk['status']}]"
            )
        else:
            print(f"  {name:<34} {blk['status']}")
    c9 = report["constraints"]["9_c4_detectability_budget"]
    print(
        f"  C4 below budget (flagged, NOT dropped): "
        f"{report['n_c4_below_budget']}/{c9['checked']}"
    )
    c5_blk = report["constraints"]["3_c5_detectability_budget"]
    if c5_blk.get("n_occurrence_infeasible") is not None:
        print(
            f"  C5 occurrence branch infeasible (p too small): "
            f"{c5_blk['n_occurrence_infeasible']}/{c5_blk['n_occurrence']} "
            f"-> routed to the size branch"
        )

    # ------------------------------------------------------------------ #
    # C7 pre-flight -- gates 4.9 / 4.10. Runs the REAL pipeline.
    # ------------------------------------------------------------------ #
    if not args.skip_c7:
        c7_ids = [
            s for s in sorted(params.loc[params["true_cause"] == "C7", "sku_id"])
        ][: args.c7_skus]
        series = {
            sid: grp.sort_values("period")["demand_true"].to_numpy(dtype="float64")
            for sid, grp in demand[demand["sku_id"].isin(c7_ids)].groupby("sku_id", sort=False)
        }
        print(f"\nC7 pre-flight on {len(series)} series through arm (c) ...")
        # The pooled LightGBM is trained ONCE across the C7 SKUs, exactly as every other
        # driver does it. A bare build_pool() carries an untrained model and raises on
        # first use -- which is how this pre-flight first failed. The statics come from
        # generator_params, which the generator wrote before drawing any series.
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
        t1 = time.time()
        windows, summary = evaluate_c7_constraints(
            series,
            factory,
            cfg,
            sku_ids=sorted(series),
            tau_inj=int(syn["tau_inj"]),
            pool=pool,
        )
        path = save_c7_report(windows, summary)
        if summary.empty:
            print("  no windows produced -- check the panel length vs W_svc")
        else:
            row = summary.iloc[0]
            print(f"  windows={row['n_windows']} skus={row['n_skus']} ({time.time() - t1:.1f}s)")
            print(f"  acceptability pass rate : {row['acceptability_pass_rate']:.4f}  (gate 4.9 needs >= 0.80)")
            print(f"  exposure pass rate      : {row['exposure_pass_rate']:.4f}  (gate 4.10 needs >= 0.80)")
            print(f"  service-gap rate        : {row['service_gap_rate']:.4f}")
            print(f"  all three conditions    : {row['c7_fireable_rate']:.4f}")
            print(f"  mean achieved CSL       : {row['mean_achieved_csl']:.4f}")
            print(f"  constraint 6 : {'PASS' if row['constraint_6_pass'] else 'FAIL'}")
            print(f"  constraint 7 : {'PASS' if row['constraint_7_pass'] else 'FAIL'}")
            print(f"  gate 4.9     : {'PASS' if row['gate_4_9_pass'] else 'FAIL'}")
        print(f"  wrote {path}")

    print(f"\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
