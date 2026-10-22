"""Exploratory diagnostic comparing baseline inject_C6 vs candidate inject_C6_v2.

Evaluates on the 120 training SKUs only (i % 5 < 3).
Quarantines validation (i % 5 == 3) and test (i % 5 == 4) SKUs strictly.
Outputs results to results/exploratory/c6_redesign/.
Preserves all production code, configs, and frozen Step 5/Step 6 artifacts.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cafr.data.synth.budget import check_c6_budget
from cafr.data.synth.generator import CELLS, draw_statics, get_cell_params
from cafr.data.synth.injections import inject_C6_v2
from cafr.data.synth.labels import _sku_rng
from cafr.forecasters.pooled import fit_global_pool
from cafr.utils.config import load_config
from experiments.step5_monitor_gates import _run_one

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "c6_redesign"


def run_diagnostic():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config("synthetic.yaml")
    data_dir = REPO_ROOT / "data" / "synth"
    demand = pd.read_parquet(data_dir / "demand_panel.parquet")
    params = pd.read_parquet(data_dir / "generator_params.parquet").set_index("sku_id")

    burn = int(cfg["splits"]["burn_in"])
    T = int(cfg["synthetic"]["T"])
    tau_inj = int(cfg["synthetic"]["tau_inj"])
    check_from = burn
    detect_from = tau_inj
    stride = 4

    # 1. Identify 120 training SKUs for C0 and C6 (i % 5 < 3)
    c0_all = sorted(params.index[params["true_cause"] == "C0"])
    c6_all = sorted(params.index[params["true_cause"] == "C6"])

    assert len(c0_all) == 200, f"Expected 200 C0 SKUs, got {len(c0_all)}"
    assert len(c6_all) == 200, f"Expected 200 C6 SKUs, got {len(c6_all)}"

    c0_train_ids = [c0_all[i] for i in range(200) if i % 5 < 3]
    c6_train_ids = [c6_all[i] for i in range(200) if i % 5 < 3]

    assert len(c0_train_ids) == 120, f"Expected 120 C0 train SKUs, got {len(c0_train_ids)}"
    assert len(c6_train_ids) == 120, f"Expected 120 C6 train SKUs, got {len(c6_train_ids)}"

    print("=" * 70)
    print("C6 GENERATOR REDESIGN: TRAINING-ONLY EXPLORATORY DIAGNOSTIC")
    print(f"Training C0 SKUs: {len(c0_train_ids)} (quarantined: 40 val, 40 test)")
    print(f"Training C6 SKUs: {len(c6_train_ids)} (quarantined: 40 val, 40 test)")
    print("=" * 70)

    # 2. Build global pooled forecaster on burn-in (periods 0..24)
    # Using all panel series on burn-in to reproduce identical pool weights as Step 5
    all_panel_skus = sorted(params.index)
    statics_by_sku = {
        sid: {
            "lead_time": float(params.loc[sid, "lead_time"]),
            "unit_cost": float(params.loc[sid, "unit_cost"]),
            "holding_rate": float(params.loc[sid, "holding_rate"]),
            "adi_init": float(params.loc[sid, "adi_init"]),
            "cv2_init": float(params.loc[sid, "cv2_init"]),
        }
        for sid in all_panel_skus
    }
    series_map = {
        sid: grp.sort_values("period")["demand_observed"].to_numpy(dtype="float64")
        for sid, grp in demand[demand["sku_id"].isin(all_panel_skus)].groupby("sku_id", sort=False)
    }

    train_panel = pd.DataFrame(
        [
            {"sku_id": sid, "period": t, "demand_observed": float(series_map[sid][t]),
             **statics_by_sku[sid]}
            for sid in all_panel_skus
            for t in range(burn)
        ]
    )
    t0 = time.time()
    pool = fit_global_pool(cfg, train_panel, statics_by_sku=statics_by_sku)
    print(f"Global pool fitted on {len(all_panel_skus)} SKUs in {time.time()-t0:.1f}s")

    # 3. Use pre-calibrated CUSUM threshold from Step 5 training controls
    # (h = 10.642717449394974 calibrated on training C0 controls in results/step5/step5_summary.json)
    with open(REPO_ROOT / "results" / "step5" / "step5_summary.json") as f:
        s5_summary = json.load(f)
    calibrated_h = float(s5_summary["cusum_calibration"]["h"])
    cfg.setdefault("detection", {})["cusum_h"] = calibrated_h
    print(f"Using pinned CUSUM threshold h = {calibrated_h:.6g}")

    # 4. Run C0 controls (identical for baseline and candidate)
    print("\nEvaluating 120 C0 training controls...")
    c0_readings = []
    for sid in c0_train_ids:
        res, _ = _run_one(
            sid, series_map[sid], params.loc[sid], cfg, pool, 42,
            check_from=check_from, detect_from=detect_from, stride=stride,
        )
        c0_readings.append(res)
    df_c0 = pd.DataFrame(c0_readings)

    c0_to_c6_fp = int((df_c0["attributed_cause"] == "C6").sum())
    c0_raw_c6_fired = int(df_c0["rule_C6_any"].sum())
    c0_h5_dead = int((df_c0["c6_init_cell"] == "dead").sum())
    print(f"  C0 -> C6 false-positive attribution: {c0_to_c6_fp}/120 ({c0_to_c6_fp/120:.1%})")
    print(f"  C0 raw C6 rule-firing rate:         {c0_raw_c6_fired}/120 ({c0_raw_c6_fired/120:.1%})")
    print(f"  C0 H-5 dead SKUs (< 2 nonzeros):    {c0_h5_dead}/120 ({c0_h5_dead/120:.1%})")

    # 5. Run Baseline C6 (from stored demand_panel.parquet)
    print("\nEvaluating 120 Baseline C6 training SKUs...")
    c6_base_readings = []
    c6_base_crossing_count = 0
    c6_base_budget_cleared = 0

    for sid in c6_train_ids:
        # Load stored baseline series
        y_base = series_map[sid]
        param_row = params.loc[sid]

        # Audit crossing & budget on baseline
        p_init = float(param_row["p"])
        direction = 1.0 if p_init < 0.25 else -1.0
        p_target_base = 0.35 if direction > 0 else 0.15
        b_base = check_c6_budget(p_init, p_target_base, w_class=30, adi_star=4.0, min_sigma=2.0)
        if b_base["crosses"]:
            c6_base_crossing_count += 1
        if b_base["cleared"]:
            c6_base_budget_cleared += 1

        res, _ = _run_one(
            sid, y_base, param_row, cfg, pool, 42,
            check_from=check_from, detect_from=detect_from, stride=stride,
        )
        res["generator_variant"] = "baseline"
        res["budget_cleared"] = b_base["cleared"]
        res["budget_z_sep"] = b_base["z_sep"]
        c6_base_readings.append(res)
    df_c6_base = pd.DataFrame(c6_base_readings)

    # 6. Generate and run Candidate C6 (inject_C6_v2 with W_ramp=24, targets 0.45/0.10)
    print("\nEvaluating 120 Candidate C6 training SKUs (inject_C6_v2)...")
    c6_cand_readings = []
    c6_cand_crossing_count = 0
    c6_cand_budget_cleared = 0

    for sid in c6_train_ids:
        idx = int(sid.split(":")[-1])
        # Recreate exact per-SKU RNG state
        rng = _sku_rng(42, "single", idx)
        cell = str(rng.choice(CELLS))
        statics = draw_statics(rng.spawn(1)[0])
        p_draw, k_draw, mu_z_draw = get_cell_params(cell, rng)

        # Generate candidate series
        y_cand, delta_cand, z_cand, cand_info = inject_C6_v2(
            T, tau_inj, p_draw, k_draw, mu_z_draw, rng,
            w_ramp=24, target_up=0.45, target_down=0.10,
        )

        # Audit candidate crossing & budget
        p_init = cand_info["p_init"]
        p_target = cand_info["p_target"]
        b_cand = check_c6_budget(p_init, p_target, w_class=30, adi_star=4.0, min_sigma=2.0)
        if b_cand["crosses"]:
            c6_cand_crossing_count += 1
        if b_cand["cleared"]:
            c6_cand_budget_cleared += 1

        # Build candidate param Series with updated p_end
        cand_param = params.loc[sid].copy()
        cand_param["p_target"] = p_target
        cand_param["w_ramp"] = 24

        res, _ = _run_one(
            sid, y_cand, cand_param, cfg, pool, 42,
            check_from=check_from, detect_from=detect_from, stride=stride,
        )
        res["generator_variant"] = "candidate_v2"
        res["budget_cleared"] = b_cand["cleared"]
        res["budget_z_sep"] = b_cand["z_sep"]
        c6_cand_readings.append(res)
    df_c6_cand = pd.DataFrame(c6_cand_readings)

    # 7. Compute comparative metrics
    def summarize_c6(df: pd.DataFrame, label: str, crossing_count: int, budget_cleared: int) -> dict:
        n_total = len(df)
        recall = int((df["attributed_cause"] == "C6").sum())
        rule_c6_any = int(df["rule_C6_any"].sum())
        h5_dead = int((df["c6_init_cell"] == "dead").sum())

        # Sustained cell shifts and stability vetoes
        c6_changed = int((df["c6_changed"] == 1).sum())
        stability_veto = int(((df["c6_changed"] == 1) & (df["c6_stability"] < 0.90)).sum())
        stability_passed = int(((df["c6_changed"] == 1) & (df["c6_stability"] >= 0.90)).sum())

        # Attributions across all categories
        attr_counts = df["attributed_cause"].value_counts().to_dict()

        # Preemption breakdown among SKUs where rule_C6_any was True
        fired_mask = df["rule_C6_any"]
        df_fired = df[fired_mask]
        n_fired = len(df_fired)

        preempt_c2 = int((df_fired["attributed_cause"] == "C2").sum())
        preempt_c3 = int((df_fired["attributed_cause"] == "C3").sum())
        preempt_c5 = int((df_fired["attributed_cause"] == "C5").sum())
        preempt_c7 = int((df_fired["attributed_cause"] == "C7").sum())
        preempt_c4 = int((df_fired["attributed_cause"] == "C4").sum())
        preempt_c1 = int((df_fired["attributed_cause"] == "C1").sum())
        fired_c0 = int((df_fired["attributed_cause"] == "C0").sum())
        fired_c6 = int((df_fired["attributed_cause"] == "C6").sum())

        return {
            "variant": label,
            "denominator": n_total,
            "c6_recall": recall,
            "c6_recall_rate": recall / n_total,
            "rule_c6_any_firings": rule_c6_any,
            "rule_c6_any_rate": rule_c6_any / n_total,
            "h5_dead_abstentions": h5_dead,
            "c6_changed_sustained": c6_changed,
            "stability_vetoed": stability_veto,
            "stability_passed": stability_passed,
            "theoretical_crossing": crossing_count,
            "budget_cleared": budget_cleared,
            "attributions_overall": attr_counts,
            "preemptions_on_fired": {
                "total_fired": n_fired,
                "recalled_C6": fired_c6,
                "preempted_by_C2": preempt_c2,
                "preempted_by_C3": preempt_c3,
                "preempted_by_C5": preempt_c5,
                "preempted_by_C7": preempt_c7,
                "preempted_by_C4": preempt_c4,
                "preempted_by_C1": preempt_c1,
                "decayed_to_C0": fired_c0,
            },
        }

    summary_base = summarize_c6(df_c6_base, "Baseline (inject_C6)", c6_base_crossing_count, c6_base_budget_cleared)
    summary_cand = summarize_c6(df_c6_cand, "Candidate (inject_C6_v2)", c6_cand_crossing_count, c6_cand_budget_cleared)

    # 8. Build comparative results table
    table_rows = [
        ("SKU Population Evaluated", "120 training SKUs (i % 5 < 3)", "120 training SKUs (i % 5 < 3)"),
        ("Validation & Test SKUs", "Quarantined (untouched)", "Quarantined (untouched)"),
        ("C6 Training Recall", f"{summary_base['c6_recall']}/120 ({summary_base['c6_recall_rate']:.1%})",
         f"{summary_cand['c6_recall']}/120 ({summary_cand['c6_recall_rate']:.1%})"),
        ("C0 -> C6 False Positives", f"{c0_to_c6_fp}/120 ({c0_to_c6_fp/120:.1%})",
         f"{c0_to_c6_fp}/120 ({c0_to_c6_fp/120:.1%})"),
        ("C0 Raw C6 Firing Rate", f"{c0_raw_c6_fired}/120 ({c0_raw_c6_fired/120:.1%})",
         f"{c0_raw_c6_fired}/120 ({c0_raw_c6_fired/120:.1%})"),
        ("H-5 Dead Init Abstentions", f"{summary_base['h5_dead_abstentions']}/120",
         f"{summary_cand['h5_dead_abstentions']}/120"),
        ("Cell Changed (Sustained >= 3)", f"{summary_base['c6_changed_sustained']}/120",
         f"{summary_cand['c6_changed_sustained']}/120"),
        ("Stability Veto (c6_changed but stab < 0.90)", f"{summary_base['stability_vetoed']}/120",
         f"{summary_cand['stability_vetoed']}/120"),
        ("Stability Passed (c6_changed & stab >= 0.90)", f"{summary_base['stability_passed']}/120",
         f"{summary_cand['stability_passed']}/120"),
        ("Raw C6 Firings on True C6 (rule_C6_any)", f"{summary_base['rule_c6_any_firings']}/120",
         f"{summary_cand['rule_c6_any_firings']}/120"),
        ("Preempted by C2 (coverage gap)", f"{summary_base['preemptions_on_fired']['preempted_by_C2']}",
         f"{summary_cand['preemptions_on_fired']['preempted_by_C2']}"),
        ("Preempted by C3 (occurrence gap)", f"{summary_base['preemptions_on_fired']['preempted_by_C3']}",
         f"{summary_cand['preemptions_on_fired']['preempted_by_C3']}"),
        ("Preempted by C5 (CUSUM alarm)", f"{summary_base['preemptions_on_fired']['preempted_by_C5']}",
         f"{summary_cand['preemptions_on_fired']['preempted_by_C5']}"),
        ("Preempted by C7 (policy gap)", f"{summary_base['preemptions_on_fired']['preempted_by_C7']}",
         f"{summary_cand['preemptions_on_fired']['preempted_by_C7']}"),
        ("Decayed / Low-Confidence to C0", f"{summary_base['preemptions_on_fired']['decayed_to_C0']}",
         f"{summary_cand['preemptions_on_fired']['decayed_to_C0']}"),
        ("Theoretical Crossing (p crosses 0.25)", f"{summary_base['theoretical_crossing']}/120",
         f"{summary_cand['theoretical_crossing']}/120"),
        ("Heuristic Budget Cleared (>= 2 sigma)", f"{summary_base['budget_cleared']}/120",
         f"{summary_cand['budget_cleared']}/120"),
    ]

    df_table = pd.DataFrame(table_rows, columns=["Metric", "Baseline (inject_C6)", "Candidate (inject_C6_v2)"])

    # 9. Save outputs
    report_data = {
        "metadata": {
            "title": "C6 Generator Redesign Training-Only Diagnostic",
            "protocol": "modulo-5 zero-leakage training partition (i % 5 < 3)",
            "quarantined_splits": "val (i % 5 == 3, 40 SKUs), test (i % 5 == 4, 40 SKUs)",
            "n_train_c0": len(c0_train_ids),
            "n_train_c6": len(c6_train_ids),
            "seed": 42,
            "cusum_h": calibrated_h,
            "elapsed_seconds": time.time() - start_time,
        },
        "c0_controls": {
            "n_evaluated": len(c0_train_ids),
            "c0_to_c6_false_positives": c0_to_c6_fp,
            "c0_raw_c6_fired": c0_raw_c6_fired,
            "c0_h5_dead": c0_h5_dead,
            "all_attributions": df_c0["attributed_cause"].value_counts().to_dict(),
        },
        "c6_baseline": summary_base,
        "c6_candidate": summary_cand,
    }

    report_path = OUT_DIR / "c6_diagnostic_report.json"
    table_path = OUT_DIR / "c6_diagnostic_table.csv"
    readings_path = OUT_DIR / "c6_diagnostic_readings.parquet"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    df_table.to_csv(table_path, index=False)

    df_all_readings = pd.concat([
        df_c0.assign(cohort="c0_control"),
        df_c6_base.assign(cohort="c6_baseline"),
        df_c6_cand.assign(cohort="c6_candidate"),
    ], ignore_index=True)
    df_all_readings.to_parquet(readings_path, index=False)

    # 10. Print results
    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS SUMMARY")
    print("=" * 70)
    for row in table_rows:
        print(f"  {row[0]:<45} | {row[1]:<25} | {row[2]}")
    print("=" * 70)
    print(f"\nOutputs written strictly to:")
    print(f"  {report_path}")
    print(f"  {table_path}")
    print(f"  {readings_path}")
    print(f"Total diagnostic runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    run_diagnostic()

