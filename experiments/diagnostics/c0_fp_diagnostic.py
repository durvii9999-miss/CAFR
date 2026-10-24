"""CAFR Diagnostic Experiment: C0 False-Positive and Attribution Error Analysis.

Conducts a read-only, training-only diagnostic of CAFR's false-positive problem on C0 controls.
Investigates:
1. Why normal C0 controls falsely attribute to non-C0 causes (38/120, 31.7%).
2. The specific rule and feature failure mechanisms behind each cause (C6: 14, C7: 8, C2: 7, C1: 5, C5: 2, C4: 1, C3: 1).
3. Raw rule firings vs final attribution (e.g. C6 raw firing 42/120 vs final attribution 14/120).
4. Demand characteristics correlating with false alarms (intermittency, zero stretches, CV2, initialization).
5. Temporal persistence vs transience of false-alarm signals.

Outputs:
- results/exploratory/c0_fp_diagnostic/c0_fp_case_level.csv
- results/exploratory/c0_fp_diagnostic/c0_fp_case_level.parquet
- results/exploratory/c0_fp_diagnostic/c0_fp_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cafr.arms.arm_c import arm_c_factory
from cafr.attribution.pipeline import attribute, PRIORITY_ORDER, CauseEvaluation
from cafr.attribution.priority import attribute_cause, compute_confidence
from cafr.attribution.rules import (
    check_c1_bias,
    check_c2_coverage,
    check_c3_occurrence,
    check_c4_size,
    check_c5_cusum,
    check_c6_intermittency,
    check_c7_policy,
)
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor.features import conformal_interval
from cafr.monitor.reading import Monitor, MonitorReading
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import load_config
from experiments.step5_monitor_gates import _sku_cfg

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "c0_fp_diagnostic"
CAUSES_ALL = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")


def _calc_zero_stretch(y: np.ndarray) -> int:
    """Calculate maximum consecutive zero periods in demand array."""
    max_z = 0
    curr_z = 0
    for val in y:
        if val <= 1e-9:
            curr_z += 1
            if curr_z > max_z:
                max_z = curr_z
        else:
            curr_z = 0
    return max_z


def run_c0_diagnostic():
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

    # 1. Identify 120 training C0 controls (i % 5 < 3)
    c0_all = sorted(params.index[params["true_cause"] == "C0"])
    c0_train = [c0_all[i] for i in range(200) if i % 5 < 3]

    print("=" * 70)
    print("CAFR C0 FALSE-POSITIVE DIAGNOSTIC (TRAINING CONTROLS ONLY)")
    print(f"C0 Training Controls: {len(c0_train)}")
    print("=" * 70)

    # 2. Build global pooled forecaster on burn-in (periods 0..24)
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
    pool = fit_global_pool(cfg, train_panel, statics_by_sku=statics_by_sku)

    # Calibrated CUSUM threshold from Step 5
    with open(REPO_ROOT / "results" / "step5" / "step5_summary.json") as f:
        s5_summary = json.load(f)
    calibrated_h = float(s5_summary["cusum_calibration"]["h"])
    cfg.setdefault("detection", {})["cusum_h"] = calibrated_h

    # Threshold constants
    det = cfg.get("detection", {})
    z_crit = float(det.get("z_b_crit", 1.96))
    tau_cov = float(det.get("tau_cov", 0.10))
    tau_occ = float(det.get("tau_occ", 0.10))
    r_thr = float(det.get("c4_r_threshold", 1.5))
    h_cusum = float(det.get("cusum_h", 4.0))
    stab_thr = float(det.get("c6_bootstrap_stability", 0.90))
    tau_svc = float(det.get("delta_svc", 0.10))
    tau_conf = float(det.get("tau_conf", 0.35))

    case_records = []
    checkpoint_records = []

    for sid in c0_train:
        y = series_map[sid]
        param_row = params.loc[sid]
        sku_cfg = _sku_cfg(cfg, param_row)

        sim_base = SimConfig.from_config(sku_cfg)
        sim_cfg = SimConfig(
            L=sim_base.L,
            R=sim_base.R,
            review_period=sim_base.review_period,
            unmet_model=sim_base.unmet_model,
            quantile_window=sim_base.quantile_window,
            min_aggregates=sim_base.min_aggregates,
            quantile_method=sim_base.quantile_method,
            alpha_min=sim_base.alpha_min,
            alpha_max_cap=sim_base.alpha_max_cap,
            H=float(param_row.get("holding_rate", 0.20)),
            B_over_H=sim_base.B_over_H,
            c7_safety_factor_scale=sim_base.c7_safety_factor_scale,
            c2_quantile_scale=sim_base.c2_quantile_scale,
            interval_half_width=sim_base.interval_half_width,
            protection_convention=sim_base.protection_convention,
            observation_model=sim_base.observation_model,
        )
        sim = InventorySimulator(sim_cfg)
        monitor = Monitor(sku_cfg, alpha_target=0.95, seed=42)
        base = arm_c_factory(pool, sku_cfg, sid, 0.95, "observed")
        q_alpha = float(sku_cfg.get("detection", {}).get("c2_interval_alpha", 0.90))
        q_win = int(sku_cfg["sim"].get("quantile_window", 24))
        q_scale = float(sku_cfg["sim"].get("c2_quantile_scale", 1.0))

        detected = {"cause": "C0", "t": None, "reading": None}
        pre_flag_t = None
        checks = []
        rule_firings_by_check = {c: [] for c in CAUSES_ALL[1:]}

        def level_fn(t, y_obs, cens, inv):
            nonlocal pre_flag_t
            S, a = base(t, y_obs, cens, inv)
            st = base.last_state
            res = base._residuals.to_array()
            rr = res[-q_win:] if res.size else res
            half_w = sku_cfg['sim'].get('interval_half_width')
            if half_w is not None and pd.notna(half_w) and np.isfinite(float(half_w)):
                w = float(half_w)
                mu_f = float(st.mu)
                lo = max(0.0, mu_f - w) if np.isfinite(mu_f) else float('nan')
                hi = max(0.0, mu_f + w) if np.isfinite(mu_f) else float('nan')
            else:
                lo, hi = conformal_interval(
                    float(st.mu), rr, q_alpha, scale=q_scale, min_evidence=8
                )
            stockout = bool(inv.get("on_hand", 1.0) <= 1e-9 and float(y_obs[-1]) > 0.0)
            monitor.update(
                t,
                float(y_obs[-1]),
                bool(cens[-1]),
                {
                    "mu": float(st.mu),
                    "mu_nonzero": float(st.mu_nonzero),
                    "sigma": float(st.sigma),
                    "z": float(st.z),
                    "lo": lo,
                    "hi": hi,
                    "alpha": q_alpha,
                },
                stockout_t=stockout,
            )
            if int(t) >= check_from and (int(t) - check_from) % stride == 0:
                rd = monitor.read()
                att = attribute(rd, sku_cfg)
                checks.append((int(t), att))
                for cause in CAUSES_ALL[1:]:
                    if cause in att.fired:
                        rule_firings_by_check[cause].append(int(t))
                if att.cause != "C0" and int(t) < detect_from and pre_flag_t is None:
                    pre_flag_t = int(t)
                if detected["cause"] == "C0" and att.cause != "C0" and int(t) >= detect_from:
                    detected.update(cause=att.cause, t=int(t), reading=rd)
            return S, a

        sim.run(sid, y, level_fn, arm="arm_c")

        final_reading = monitor.read()
        final_att = attribute(final_reading, sku_cfg)
        detect_reading = detected["reading"] if detected["reading"] is not None else final_reading
        att = attribute(detect_reading, sku_cfg)

        # Demand characteristics
        pos_sizes = y[y > 1e-9]
        n_pos = len(pos_sizes)
        emp_adi = float(T / n_pos) if n_pos > 0 else float("inf")
        emp_cv2 = float((np.std(pos_sizes, ddof=1) / np.mean(pos_sizes))**2) if n_pos > 1 else float("nan")
        max_zero_run = _calc_zero_stretch(y)

        # Mechanism classification
        first_post120_fired = []
        for t_chk, att_chk in checks:
            if t_chk >= detect_from and att_chk.fired:
                first_post120_fired = list(att_chk.fired)
                first_post120_t = t_chk
                break
        else:
            first_post120_t = None

        is_fp = bool(att.cause != "C0")
        fired_at_det = list(att.fired)
        is_simultaneous = bool(len(fired_at_det) > 1)
        preempted_at_det = [c for c in fired_at_det if c != att.cause]

        # Persistence of attributed cause
        if is_fp and detected["t"] is not None:
            post_detect_firings = [t_chk for t_chk in rule_firings_by_check[att.cause] if t_chk >= detected["t"]]
            n_post_firings = len(post_detect_firings)
            persistent = (n_post_firings >= 3)
            transient = (n_post_firings <= 2)
        else:
            post_detect_firings = []
            n_post_firings = 0
            persistent = False
            transient = False

        # Read trigger statistics at detection
        s_det = detect_reading.statistics
        aux_det = detect_reading.aux

        case_row = {
            "sku_id": sid,
            "true_cause": "C0",
            "attributed_cause": att.cause,
            "is_false_positive": is_fp,
            "confidence": float(att.confidence),
            "detect_t": detected["t"],
            "pre_flag_t": pre_flag_t,
            "fired_at_detect": ",".join(fired_at_det),
            "n_fired_at_detect": len(fired_at_det),
            "is_simultaneous": is_simultaneous,
            "preempted_causes": ",".join(preempted_at_det),
            "is_terminal_fallback": bool(detected["t"] is None and final_att.cause != "C0"),
            "final_cause_t199": final_att.cause,
            "first_post120_t": first_post120_t,
            "n_firings_attributed_rule": n_post_firings,
            "is_persistent": persistent,
            "is_transient": transient,
            # Raw rule firing booleans across whole series
            "rule_C1_any": bool(len(rule_firings_by_check["C1"]) > 0),
            "rule_C2_any": bool(len(rule_firings_by_check["C2"]) > 0),
            "rule_C3_any": bool(len(rule_firings_by_check["C3"]) > 0),
            "rule_C4_any": bool(len(rule_firings_by_check["C4"]) > 0),
            "rule_C5_any": bool(len(rule_firings_by_check["C5"]) > 0),
            "rule_C6_any": bool(len(rule_firings_by_check["C6"]) > 0),
            "rule_C7_any": bool(len(rule_firings_by_check["C7"]) > 0),
            "total_rules_ever_fired": sum(bool(len(rule_firings_by_check[c]) > 0) for c in CAUSES_ALL[1:]),
            # Key statistics at detect_t
            "c1_z_b": float(s_det.get("c1_z_b", np.nan)),
            "c1_usable_sz": int(aux_det.get("n_usable_sz", 0)),
            "c2_c_hat": float(s_det.get("c2_c_hat", np.nan)),
            "c2_c_gap": float(s_det.get("c2_c_gap", np.nan)),
            "c2_p_val": float(s_det.get("c2_p_val", np.nan)),
            "c3_occ_gap": float(s_det.get("c3_occ_gap", np.nan)),
            "c3_occ_p_val": float(s_det.get("c3_occ_p_val", np.nan)),
            "c4_r": float(s_det.get("c4_r", np.nan)),
            "c5_sz_score": float(s_det.get("c5_sz_alarm_score", np.nan)),
            "c5_occ_score": float(s_det.get("c5_occ_alarm_score", np.nan)),
            "c6_init_cell": str(aux_det.get("init_cell", "")),
            "c6_changed": float(s_det.get("c6_changed", 0.0)),
            "c6_stability": float(s_det.get("c6_stability", np.nan)),
            "c6_adi": float(s_det.get("c6_adi", np.nan)),
            "c6_cv2": float(s_det.get("c6_cv2", np.nan)),
            "c7_csl_hat": float(s_det.get("c7_csl_hat", np.nan)),
            "c7_csl_gap": float(s_det.get("c7_csl_gap", np.nan)),
            "c7_unmet_cycles": int(aux_det.get("unmet_cycles", 0)),
            # Demand characteristics
            "param_p": float(param_row["p"]),
            "param_k": float(param_row["k"]),
            "param_mu_z": float(param_row["mu_z"]),
            "n_pos_demands": n_pos,
            "emp_adi": emp_adi,
            "emp_cv2": emp_cv2,
            "max_zero_run": max_zero_run,
        }
        case_records.append(case_row)

    df_cases = pd.DataFrame(case_records)

    # 3. Save Case-Level Datasets
    csv_path = OUT_DIR / "c0_fp_case_level.csv"
    parquet_path = OUT_DIR / "c0_fp_case_level.parquet"
    df_cases.to_csv(csv_path, index=False)
    df_cases.to_parquet(parquet_path, index=False)
    print(f"\nSaved case-level CSV to:     {csv_path}")
    print(f"Saved case-level Parquet to: {parquet_path}")

    # 4. Summary & Verification Calculations
    n_total = len(df_cases)
    c0_correct = int((df_cases["attributed_cause"] == "C0").sum())
    fp_total = int(df_cases["is_false_positive"].sum())
    fp_rate = fp_total / n_total

    fp_by_cause = df_cases[df_cases["is_false_positive"]]["attributed_cause"].value_counts().to_dict()
    raw_by_cause = {f"rule_{c}_any": int(df_cases[f"rule_{c}_any"].sum()) for c in CAUSES_ALL[1:]}

    # Verify baseline match
    exp_fp_total = 38
    exp_c6_fp = 14
    exp_c6_raw = 42

    match_overall = bool(fp_total == exp_fp_total)
    match_c6_fp = bool(fp_by_cause.get("C6", 0) == exp_c6_fp)
    match_c6_raw = bool(raw_by_cause["rule_C6_any"] == exp_c6_raw)

    print("\n" + "=" * 70)
    print("BASELINE REPRODUCIBILITY AUDIT")
    print(f"Overall C0 False Positives: {fp_total}/{n_total} ({fp_rate:.1%}) [Expected: 38/120] -> Match: {match_overall}")
    print(f"C0 Attributed to C6:        {fp_by_cause.get('C6', 0)}/{n_total} ({fp_by_cause.get('C6', 0)/n_total:.1%}) [Expected: 14/120] -> Match: {match_c6_fp}")
    print(f"Raw C6 Rule Firings:        {raw_by_cause['rule_C6_any']}/{n_total} ({raw_by_cause['rule_C6_any']/n_total:.1%}) [Expected: 42/120] -> Match: {match_c6_raw}")
    print("=" * 70)

    # 5. Mechanism Breakdown of False Positives
    df_fp = df_cases[df_cases["is_false_positive"]]
    simultaneous_count = int(df_fp["is_simultaneous"].sum())
    preempted_count = int((df_fp["preempted_causes"] != "").sum())
    terminal_count = int(df_fp["is_terminal_fallback"].sum())
    persistent_count = int(df_fp["is_persistent"].sum())
    transient_count = int(df_fp["is_transient"].sum())

    summary_report = {
        "metadata": {
            "title": "CAFR C0 False Positive and Attribution Error Diagnostic",
            "evaluated_split": "Training split only (i % 5 < 3)",
            "quarantined_splits": "Validation (i % 5 == 3) and Test (i % 5 == 4) strictly untouched",
            "n_evaluated": n_total,
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "baseline_verification": {
            "overall_fp_count": fp_total,
            "overall_fp_rate": fp_rate,
            "expected_overall_fp": exp_fp_total,
            "overall_matches": match_overall,
            "c6_fp_count": fp_by_cause.get("C6", 0),
            "expected_c6_fp": exp_c6_fp,
            "c6_fp_matches": match_c6_fp,
            "c6_raw_firings": raw_by_cause["rule_C6_any"],
            "expected_c6_raw": exp_c6_raw,
            "c6_raw_matches": match_c6_raw,
        },
        "false_positives_by_cause": {
            cause: {
                "attributed_count": int(fp_by_cause.get(cause, 0)),
                "attributed_rate": float(fp_by_cause.get(cause, 0) / n_total),
                "raw_firings_count": int(raw_by_cause[f"rule_{cause}_any"]),
                "raw_firings_rate": float(raw_by_cause[f"rule_{cause}_any"] / n_total),
                "conversion_to_fp_ratio": float(fp_by_cause.get(cause, 0) / raw_by_cause[f"rule_{cause}_any"]) if raw_by_cause[f"rule_{cause}_any"] > 0 else 0.0,
            }
            for cause in CAUSES_ALL[1:]
        },
        "triggering_mechanisms": {
            "total_false_positives": fp_total,
            "simultaneous_rules_at_detect": simultaneous_count,
            "priority_preempted_other_rules": preempted_count,
            "terminal_fallback_latch": terminal_count,
            "persistent_firings_3plus": persistent_count,
            "transient_firings_1to2": transient_count,
        },
        "top_feature_patterns": {
            "C6_false_positives": {
                "count": fp_by_cause.get("C6", 0),
                "mean_param_p": float(df_fp[df_fp["attributed_cause"] == "C6"]["param_p"].mean()),
                "mean_emp_adi": float(df_fp[df_fp["attributed_cause"] == "C6"]["emp_adi"].mean()),
                "median_stability": float(df_fp[df_fp["attributed_cause"] == "C6"]["c6_stability"].median()),
                "init_cell_distribution": df_fp[df_fp["attributed_cause"] == "C6"]["c6_init_cell"].value_counts().to_dict(),
            },
            "C7_false_positives": {
                "count": fp_by_cause.get("C7", 0),
                "mean_csl_gap": float(df_fp[df_fp["attributed_cause"] == "C7"]["c7_csl_gap"].mean()),
                "mean_unmet_cycles": float(df_fp[df_fp["attributed_cause"] == "C7"]["c7_unmet_cycles"].mean()),
                "mean_max_zero_run": float(df_fp[df_fp["attributed_cause"] == "C7"]["max_zero_run"].mean()),
            },
            "C2_false_positives": {
                "count": fp_by_cause.get("C2", 0),
                "mean_c_gap": float(df_fp[df_fp["attributed_cause"] == "C2"]["c2_c_gap"].mean()),
                "mean_p_val": float(df_fp[df_fp["attributed_cause"] == "C2"]["c2_p_val"].mean()),
            },
            "C1_false_positives": {
                "count": fp_by_cause.get("C1", 0),
                "mean_z_b": float(df_fp[df_fp["attributed_cause"] == "C1"]["c1_z_b"].mean()),
                "mean_usable_sz": float(df_fp[df_fp["attributed_cause"] == "C1"]["c1_usable_sz"].mean()),
            },
        },
    }

    json_path = OUT_DIR / "c0_fp_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved summary JSON to:       {json_path}")


if __name__ == "__main__":
    run_c0_diagnostic()

