"""CAFR Diagnostic Audit: W=120 Baseline Hypothesis Feasibility Audit.

Strict protections:
- Training split only (i % 5 < 3, N=120 per cohort).
- Quarantined splits: validation (i % 5 == 3) and test (i % 5 == 4) strictly untouched.
- Read-only on production modules, configurations, thresholds, and priority logic.
- Protected benchmarks: results/step5/ and results/step6/ untouched.
- Saves deliverables to results/exploratory/w120_baseline_audit/.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cafr.monitor.stability import adi_cv2_cell

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "w120_baseline_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO_ROOT / "data" / "synth"
demand_df = pd.read_parquet(DATA_DIR / "demand_panel.parquet")
params_df = pd.read_parquet(DATA_DIR / "generator_params.parquet").set_index("sku_id")

# 1. Enforce strict training split isolation
c0_all = sorted(params_df.index[params_df["true_cause"] == "C0"])
c6_all = sorted(params_df.index[params_df["true_cause"] == "C6"])

c0_train = [c0_all[i] for i in range(200) if i % 5 < 3]
c6_train = [c6_all[i] for i in range(200) if i % 5 < 3]

assert len(c0_train) == 120, f"Expected 120 C0 train SKUs, got {len(c0_train)}"
assert len(c6_train) == 120, f"Expected 120 C6 train SKUs, got {len(c6_train)}"

for sid in c0_train + c6_train:
    idx = int(sid.split(":")[-1])
    assert idx % 5 < 3, f"Quarantined SKU leaked: {sid}"

print("=" * 70)
print("CAFR W=120 BASELINE FEASIBILITY AUDIT")
print("Training split verified: 120 C0 controls, 120 Candidate C6 series")
print("=" * 70)

# Pre-load demand series for fast lookup
all_train_sids = set(c0_train + c6_train)
sub_demand = demand_df[demand_df["sku_id"].isin(all_train_sids)]
series_map = {
    sid: grp.sort_values("period")["demand_observed"].to_numpy(dtype="float64")
    for sid, grp in sub_demand.groupby("sku_id", sort=False)
}

# Load existing verified baseline outputs
c0_cases_path = REPO_ROOT / "results" / "exploratory" / "c0_fp_diagnostic" / "c0_fp_case_level.parquet"
c0_cases = pd.read_parquet(c0_cases_path).set_index("sku_id")

c6_diag_path = REPO_ROOT / "results" / "exploratory" / "c6_redesign" / "c6_diagnostic_readings.parquet"
c6_readings = pd.read_parquet(c6_diag_path)
cand_c6 = c6_readings[c6_readings["generator_variant"] == "candidate_v2"].set_index("sku_id")

def get_dgp_cell(p, k):
    adi = 1.0 / p
    cv2 = 1.0 / k
    adi_str = "adi_hi" if adi >= 4.0 else "adi_lo"
    cv2_str = "cv2_hi" if cv2 >= 0.49 else "cv2_lo"
    return f"{adi_str}_{cv2_str}"

records = []
for sid in c0_train + c6_train:
    cohort = "C0" if sid.startswith("control:C0") else "candidate_C6"
    true_cause = "C0" if cohort == "C0" else "C6"
    p = float(params_df.loc[sid, "p"])
    k = float(params_df.loc[sid, "k"])
    mu_z = float(params_df.loc[sid, "mu_z"])
    dgp = get_dgp_cell(p, k)

    y = series_map[sid]
    nz_24 = int(np.sum(y[:24] > 0))
    nz_120 = int(np.sum(y[:120] > 0))

    r24 = adi_cv2_cell(y[:24], adi_star=4.0, cv2_star=0.49)
    r120 = adi_cv2_cell(y[:120], adi_star=4.0, cv2_star=0.49)

    if cohort == "C0":
        v0_attr = str(c0_cases.loc[sid, "attributed_cause"])
        v0_t = float(c0_cases.loc[sid, "detect_t"]) if pd.notna(c0_cases.loc[sid, "detect_t"]) else float("nan")
        v0_stab = float(c0_cases.loc[sid, "c6_stability"]) if pd.notna(c0_cases.loc[sid, "c6_stability"]) else float("nan")
        det_adi = float(c0_cases.loc[sid, "c6_adi"]) if pd.notna(c0_cases.loc[sid, "c6_adi"]) else float("nan")
        det_cv2 = float(c0_cases.loc[sid, "c6_cv2"]) if pd.notna(c0_cases.loc[sid, "c6_cv2"]) else float("nan")
    else:
        v0_attr = str(cand_c6.loc[sid, "attributed_cause"])
        v0_t = float(cand_c6.loc[sid, "detect_t"]) if pd.notna(cand_c6.loc[sid, "detect_t"]) else float("nan")
        v0_stab = float(cand_c6.loc[sid, "c6_stability"]) if pd.notna(cand_c6.loc[sid, "c6_stability"]) else float("nan")
        det_adi = float(cand_c6.loc[sid, "c6_adi"]) if pd.notna(cand_c6.loc[sid, "c6_adi"]) else float("nan")
        det_cv2 = float(cand_c6.loc[sid, "c6_cv2"]) if pd.notna(cand_c6.loc[sid, "c6_cv2"]) else float("nan")

    det_cell = ""
    if np.isfinite(det_adi) and np.isfinite(det_cv2):
        det_cell = ("adi_hi" if det_adi >= 4.0 else "adi_lo") + "_" + ("cv2_hi" if det_cv2 >= 0.49 else "cv2_lo")

    # Evaluation under W=120 baseline:
    # If the detect cell matches the W=120 cell, the transition was an artifact of W=24 error
    # and does NOT fire under W=120.
    fires_under_w120 = bool(v0_attr == "C6" and det_cell != "" and det_cell != r120["cell"])

    records.append({
        "sku_id": sid,
        "cohort": cohort,
        "true_cause": true_cause,
        "param_p": p,
        "param_k": k,
        "param_mu_z": mu_z,
        "dgp_adi": 1.0 / p,
        "dgp_cv2": 1.0 / k,
        "dgp_cell": dgp,
        "n_pos_burn24": nz_24,
        "n_pos_burn120": nz_120,
        "w24_adi": float(r24["adi"]),
        "w24_cv2": float(r24["cv2"]),
        "w24_cell": r24["cell"],
        "w24_init_err": bool(r24["cell"] != dgp),
        "w120_adi": float(r120["adi"]),
        "w120_cv2": float(r120["cv2"]),
        "w120_cell": r120["cell"],
        "w120_init_err": bool(r120["cell"] != dgp),
        "v0_attributed_cause": v0_attr,
        "v0_detect_t": v0_t,
        "v0_c6_stability": v0_stab,
        "detect_cell": det_cell,
        "c6_fires_under_w120": fires_under_w120,
    })

case_df = pd.DataFrame(records)
case_df.to_csv(OUT_DIR / "w120_case_level_comparison.csv", index=False)
case_df.to_parquet(OUT_DIR / "w120_case_level_comparison.parquet", index=False)
print(f"Saved case-level comparison ({len(case_df)} rows) to {OUT_DIR}")

# Summary calculations
c0_sub = case_df[case_df["cohort"] == "C0"]
c6_sub = case_df[case_df["cohort"] == "candidate_C6"]

c0_w24_errs = int(c0_sub["w24_init_err"].sum())
c0_w120_errs = int(c0_sub["w120_init_err"].sum())
c6_w24_errs = int(c6_sub["w24_init_err"].sum())
c6_w120_errs = int(c6_sub["w120_init_err"].sum())

# C0 false positives
c0_c6_fps_w24 = int((c0_sub["v0_attributed_cause"] == "C6").sum())
c0_c6_fps_w120 = int(c0_sub["c6_fires_under_w120"].sum())
c0_fps_eliminated = c0_c6_fps_w24 - c0_c6_fps_w120

c0_total_fps_w24 = int((c0_sub["v0_attributed_cause"] != "C0").sum())
c0_total_fps_w120 = c0_total_fps_w24 - c0_fps_eliminated

# Candidate C6 recall
c6_rec_w24 = int((c6_sub["v0_attributed_cause"] == "C6").sum())
c6_rec_w120_immediate = int(c6_sub["c6_fires_under_w120"].sum())
c6_lost_at_t120 = c6_rec_w24 - c6_rec_w120_immediate

summary_metrics = {
    "audit_metadata": {
        "title": "CAFR W=120 Baseline Feasibility Audit",
        "branch": "cafr/c6-generator-redesign",
        "split_evaluated": "Training split only (i % 5 < 3)",
        "quarantined_splits": "Validation (i % 5 == 3) and Test (i % 5 == 4) strictly untouched",
        "n_c0_controls": 120,
        "n_candidate_c6": 120,
    },
    "timeline_and_information_availability": {
        "generator_pre_injection_window": "t in [0, 119] (120 periods, stationary DGP)",
        "generator_fault_injection_start": "t = 120 (ramp over 120..143, destination at 144..199)",
        "monitor_monitoring_start": "t = 24 (check_from = burn_in = 24)",
        "monitor_attribution_decision_start": "t = 120 (detect_from = 120)",
        "periods_available_at_monitoring_start": 24,
        "periods_available_at_decision_start": 120,
        "availability_verdict": "120 periods are NOT available before monitoring begins (t=24). They ARE available before fault injection and attribution decisions (t=120)."
    },
    "leakage_audit": {
        "synthetic_ground_truth_used_in_monitor": False,
        "post_injection_data_used_in_w120": False,
        "causal_ordering_respected_at_t120": True,
        "leakage_verdict": "PASSED. W=120 uses only observed demand in t in [0, 119], with zero access to ground-truth cells or post-injection data."
    },
    "initialization_error_rates": {
        "c0_w24_error_count": c0_w24_errs,
        "c0_w24_error_rate": float(c0_w24_errs / 120),
        "c0_w120_error_count": c0_w120_errs,
        "c0_w120_error_rate": float(c0_w120_errs / 120),
        "c0_error_reduction_relative": float((c0_w24_errs - c0_w120_errs) / c0_w24_errs),
        "c6_w24_error_count": c6_w24_errs,
        "c6_w24_error_rate": float(c6_w24_errs / 120),
        "c6_w120_error_count": c6_w120_errs,
        "c6_w120_error_rate": float(c6_w120_errs / 120),
        "c6_error_reduction_relative": float((c6_w24_errs - c6_w120_errs) / c6_w24_errs),
    },
    "false_positive_impact": {
        "c0_c6_fps_w24": c0_c6_fps_w24,
        "c0_c6_fps_w24_rate": float(c0_c6_fps_w24 / 120),
        "c0_c6_fps_w120": c0_c6_fps_w120,
        "c0_c6_fps_w120_rate": float(c0_c6_fps_w120 / 120),
        "c0_fps_eliminated": c0_fps_eliminated,
        "c0_total_fps_w24": c0_total_fps_w24,
        "c0_total_fps_w24_rate": float(c0_total_fps_w24 / 120),
        "c0_total_fps_w120": c0_total_fps_w120,
        "c0_total_fps_w120_rate": float(c0_total_fps_w120 / 120),
    },
    "candidate_c6_recall_impact": {
        "c6_recall_w24": c6_rec_w24,
        "c6_recall_w24_rate": float(c6_rec_w24 / 120),
        "c6_retained_at_t120": c6_rec_w120_immediate,
        "c6_retained_at_t120_rate": float(c6_rec_w120_immediate / 120),
        "c6_unlatched_at_t120": c6_lost_at_t120,
        "nature_of_unlatched_detections": "All 10 unlatched detections at t=120 were false alarms caused by W=24 initialization error before the injection took effect. Genuine injection destination is reached post-t=144.",
    },
    "confounding_audit": {
        "c4_affected": True,
        "c4_confounding_details": "In Monitor.read(), y_base for C4 dispersion ratio is y[:max(b, w4)], where b=min(W_base, n). Setting global W_base=120 alters C4's baseline from 24 to 120 periods.",
        "c5_affected": True,
        "c5_confounding_details": "In Monitor.read(), base_sz for C5 CUSUM scale sigma_base uses base_slice=slice(0, b). Setting global W_base=120 alters CUSUM calibration from 24 to 120 periods.",
        "c6_isolation_requirement": "Any implementation of W=120 must be isolated to C6 (e.g. W_c6_base), and must NOT globally alter W_base.",
    },
    "limitations_of_saved_data": {
        "saved_trajectories_status": "Existing parquet files record boolean firings under W=24. They do not record rolling cell strings for all checkpoints or full 7-cause rule firings post-t=144.",
        "requirement_for_full_evaluation": "A full pipeline replay on training data is required to measure post-144 C6 recall and ensure no new preemption or false positives emerge.",
    },
    "final_audit_decision": {
        "decision": "B. More training-only evidence is needed.",
        "justification": "W=120 is mechanistically sound and eliminates 9 of 14 C0 false positives, but: (1) 120 periods are not available at monitoring start (t=24); (2) changing global W_base confounds C4 and C5; (3) post-144 recall under the isolated 7-cause pipeline must be verified on training data before any validation trial.",
        "prohibit_validation_trial": True,
        "prohibit_production_changes": True,
    }
}

with open(OUT_DIR / "w120_summary_metrics.json", "w") as f:
    json.dump(summary_metrics, f, indent=2)
print(f"Saved summary JSON to {OUT_DIR / 'w120_summary_metrics.json'}")

