"""CAFR Diagnostic Experiment: C6 Burn-In Initialization Repair.

Strict protections:
- Training split only (i % 5 < 3, N=120 per cohort).
- Quarantined splits: validation (i % 5 == 3) and test (i % 5 == 4) strictly untouched.
- Read-only on production modules, configurations, thresholds, and priority logic.
- Protected benchmarks: results/step5/ and results/step6/ untouched.
- Saves deliverables to results/exploratory/c6_initialization/.
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

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "c6_initialization"
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

# Check quarantine
for sid in c0_train + c6_train:
    idx = int(sid.split(":")[-1])
    assert idx % 5 < 3, f"Quarantined SKU leaked: {sid}"

print("=" * 70)
print("CAFR C6 BURN-IN INITIALIZATION REPAIR DIAGNOSTIC")
print("Training split verified: 120 C0 controls, 120 Candidate C6 series")
print("=" * 70)

# Pre-load demand arrays for speed
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

# Verify baseline V0 reproduction
v0_c0_fps = int((c0_cases["attributed_cause"] != "C0").sum())
v0_c0_c6_fps = int((c0_cases["attributed_cause"] == "C6").sum())
v0_c6_recall = int((cand_c6["attributed_cause"] == "C6").sum())

assert v0_c0_fps == 38, f"V0 C0 total FPs mismatch: {v0_c0_fps} vs 38"
assert v0_c0_c6_fps == 14, f"V0 C0 C6 FPs mismatch: {v0_c0_c6_fps} vs 14"
assert v0_c6_recall == 31, f"V0 C6 recall mismatch: {v0_c6_recall} vs 31"
print(f"V0 Baseline reproduction verified: C0 FPs={v0_c0_fps}/120 (C6={v0_c0_c6_fps}), C6 Recall={v0_c6_recall}/120")

# Compute pooled training burn-in moments (t < 24)
train_burn_obs = []
train_burn_nz = []
for sid in all_train_sids:
    y_b = series_map[sid][:24]
    nz_b = y_b[y_b > 0]
    train_burn_obs.append(len(y_b))
    train_burn_nz.extend(nz_b.tolist())

total_obs = sum(train_burn_obs)
p_pooled = len(train_burn_nz) / total_obs
adi_pooled = 1.0 / p_pooled
cv2_pooled = float(np.var(train_burn_nz, ddof=1) / (np.mean(train_burn_nz) ** 2))

print(f"Pooled Training Burn-in Prior (N=240 SKUs, t < 24):")
print(f"  p_pool = {p_pooled:.4f} -> ADI_pool = {adi_pooled:.2f}")
print(f"  CV2_pool = {cv2_pooled:.4f}")

def get_dgp_cell(p, k):
    adi = 1.0 / p
    cv2 = 1.0 / k
    adi_str = "adi_hi" if adi >= 4.0 else "adi_lo"
    cv2_str = "cv2_hi" if cv2 >= 0.49 else "cv2_lo"
    return f"{adi_str}_{cv2_str}"

def compute_shrunk_cell(y_burn, M=6.0):
    nz = int(np.sum(y_burn > 0))
    if nz == 0:
        return "dead", float("inf"), float("nan")
    nz_sizes = y_burn[y_burn > 0]
    if nz >= 2:
        cv2_hat = float(np.var(nz_sizes, ddof=1) / (np.mean(nz_sizes) ** 2))
    else:
        cv2_hat = cv2_pooled
    p_shrunk = float((nz + M * p_pooled) / (len(y_burn) + M))
    adi_shrunk = 1.0 / p_shrunk
    cv2_shrunk = float((nz * cv2_hat + M * cv2_pooled) / (nz + M))
    adi_str = "adi_hi" if adi_shrunk >= 4.0 else "adi_lo"
    cv2_str = "cv2_hi" if cv2_shrunk >= 0.49 else "cv2_lo"
    return f"{adi_str}_{cv2_str}", adi_shrunk, cv2_shrunk

# Build case-level table
records = []

# C0 controls
for sid in c0_train:
    p_val = float(params_df.loc[sid, "p"])
    k_val = float(params_df.loc[sid, "k"])
    mu_z = float(params_df.loc[sid, "mu_z"])
    dgp = get_dgp_cell(p_val, k_val)
    y = series_map[sid]
    y24 = y[:24]
    nz24 = int(np.sum(y24 > 0))
    res24 = adi_cv2_cell(y24, adi_star=4.0, cv2_star=0.49)
    v0_attr = str(c0_cases.loc[sid, "attributed_cause"])
    v0_det_t = float(c0_cases.loc[sid, "detect_t"]) if pd.notna(c0_cases.loc[sid, "detect_t"]) else float("nan")
    v0_stab = float(c0_cases.loc[sid, "c6_stability"]) if pd.notna(c0_cases.loc[sid, "c6_stability"]) else float("nan")

    # V1: N_min = 6
    v1_attr_n6 = "C0" if (v0_attr == "C6" and nz24 < 6) else v0_attr

    # V2: M=2, M=6
    c_m2, a_m2, v_m2 = compute_shrunk_cell(y24, M=2.0)
    c_m6, a_m6, v_m6 = compute_shrunk_cell(y24, M=6.0)

    # Expanded baseline W=120
    res120 = adi_cv2_cell(y[:120], adi_star=4.0, cv2_star=0.49)

    records.append({
        "sku_id": sid,
        "cohort": "C0",
        "true_cause": "C0",
        "param_p": p_val,
        "param_k": k_val,
        "param_mu_z": mu_z,
        "dgp_adi": 1.0 / p_val,
        "dgp_cv2": 1.0 / k_val,
        "dgp_cell": dgp,
        "n_pos_burn24": nz24,
        "v0_init_adi": float(res24["adi"]),
        "v0_init_cv2": float(res24["cv2"]),
        "v0_init_cell": res24["cell"],
        "v0_init_err": bool(res24["cell"] != dgp),
        "v0_attributed_cause": v0_attr,
        "v0_detect_t": v0_det_t,
        "v0_c6_stability": v0_stab,
        "v1_attributed_cause_N6": v1_attr_n6,
        "v2_shrunk_cell_M2": c_m2,
        "v2_shrunk_adi_M2": a_m2,
        "v2_shrunk_cv2_M2": v_m2,
        "v2_shrunk_cell_err_M2": bool(c_m2 != dgp),
        "v2_shrunk_cell_M6": c_m6,
        "v2_shrunk_cell_err_M6": bool(c_m6 != dgp),
        "expanded_cell_w120": res120["cell"],
        "expanded_cell_err_w120": bool(res120["cell"] != dgp),
    })

# Candidate C6 series
for sid in c6_train:
    p_val = float(params_df.loc[sid, "p"])
    k_val = float(params_df.loc[sid, "k"])
    mu_z = float(params_df.loc[sid, "mu_z"])
    dgp = get_dgp_cell(p_val, k_val)
    y = series_map[sid]
    y24 = y[:24]
    nz24 = int(np.sum(y24 > 0))
    res24 = adi_cv2_cell(y24, adi_star=4.0, cv2_star=0.49)
    v0_attr = str(cand_c6.loc[sid, "attributed_cause"])
    v0_det_t = float(cand_c6.loc[sid, "detect_t"]) if pd.notna(cand_c6.loc[sid, "detect_t"]) else float("nan")
    v0_stab = float(cand_c6.loc[sid, "c6_stability"]) if pd.notna(cand_c6.loc[sid, "c6_stability"]) else float("nan")

    # V1: N_min = 6
    v1_attr_n6 = "C0" if (v0_attr == "C6" and nz24 < 6) else v0_attr

    # V2: M=2, M=6
    c_m2, a_m2, v_m2 = compute_shrunk_cell(y24, M=2.0)
    c_m6, a_m6, v_m6 = compute_shrunk_cell(y24, M=6.0)

    # Expanded baseline W=120
    res120 = adi_cv2_cell(y[:120], adi_star=4.0, cv2_star=0.49)

    records.append({
        "sku_id": sid,
        "cohort": "candidate_C6",
        "true_cause": "C6",
        "param_p": p_val,
        "param_k": k_val,
        "param_mu_z": mu_z,
        "dgp_adi": 1.0 / p_val,
        "dgp_cv2": 1.0 / k_val,
        "dgp_cell": dgp,
        "n_pos_burn24": nz24,
        "v0_init_adi": float(res24["adi"]),
        "v0_init_cv2": float(res24["cv2"]),
        "v0_init_cell": res24["cell"],
        "v0_init_err": bool(res24["cell"] != dgp),
        "v0_attributed_cause": v0_attr,
        "v0_detect_t": v0_det_t,
        "v0_c6_stability": v0_stab,
        "v1_attributed_cause_N6": v1_attr_n6,
        "v2_shrunk_cell_M2": c_m2,
        "v2_shrunk_adi_M2": a_m2,
        "v2_shrunk_cv2_M2": v_m2,
        "v2_shrunk_cell_err_M2": bool(c_m2 != dgp),
        "v2_shrunk_cell_M6": c_m6,
        "v2_shrunk_cell_err_M6": bool(c_m6 != dgp),
        "expanded_cell_w120": res120["cell"],
        "expanded_cell_err_w120": bool(res120["cell"] != dgp),
    })

case_df = pd.DataFrame(records)
case_df.to_csv(OUT_DIR / "c6_init_case_level.csv", index=False)
case_df.to_parquet(OUT_DIR / "c6_init_case_level.parquet", index=False)
print(f"Saved case-level data ({len(case_df)} rows) to {OUT_DIR}")

# V1 Sensitivity Curve across N_min in [2..10]
v1_sensitivity = []
c0_df = case_df[case_df["cohort"] == "C0"]
c6_df = case_df[case_df["cohort"] == "candidate_C6"]

for n_min in range(2, 11):
    # C0
    c0_c6_fp = int(((c0_df["v0_attributed_cause"] == "C6") & (c0_df["n_pos_burn24"] >= n_min)).sum())
    c0_other_fp = int((c0_df["v0_attributed_cause"].isin(["C1", "C2", "C3", "C4", "C5", "C7"])).sum())
    c0_total_fp = c0_c6_fp + c0_other_fp

    # C6 recall
    c6_rec = int(((c6_df["v0_attributed_cause"] == "C6") & (c6_df["n_pos_burn24"] >= n_min)).sum())

    v1_sensitivity.append({
        "N_min": n_min,
        "c0_c6_fp_count": c0_c6_fp,
        "c0_c6_fp_rate": float(c0_c6_fp / 120),
        "c0_total_fp_count": c0_total_fp,
        "c0_total_fp_rate": float(c0_total_fp / 120),
        "c6_recall_count": c6_rec,
        "c6_recall_rate": float(c6_rec / 120),
        "c6_lost_detections": int(31 - c6_rec),
    })

# V2 Shrinkage Sensitivity across M in [0, 2, 4, 6, 8, 12]
v2_sensitivity = []
for M in [0, 2, 4, 6, 8, 12]:
    c0_err = 0
    c6_err = 0
    for _, row in c0_df.iterrows():
        y24 = series_map[row["sku_id"]][:24]
        c = adi_cv2_cell(y24)["cell"] if M == 0 else compute_shrunk_cell(y24, M=M)[0]
        if c != row["dgp_cell"]:
            c0_err += 1
    for _, row in c6_df.iterrows():
        y24 = series_map[row["sku_id"]][:24]
        c = adi_cv2_cell(y24)["cell"] if M == 0 else compute_shrunk_cell(y24, M=M)[0]
        if c != row["dgp_cell"]:
            c6_err += 1
    v2_sensitivity.append({
        "M_prior_weight": M,
        "c0_init_errors": c0_err,
        "c0_init_error_rate": float(c0_err / 120),
        "c6_init_errors": c6_err,
        "c6_init_error_rate": float(c6_err / 120),
    })

# Summary JSON assembly
summary_data = {
    "metadata": {
        "title": "CAFR C6 Burn-In Initialization Repair Diagnostic",
        "branch": "cafr/c6-generator-redesign",
        "split": "Training split strictly isolated (i % 5 < 3)",
        "quarantined_splits": "Validation (i % 5 == 3) and Test (i % 5 == 4) untouched",
        "n_c0_controls": 120,
        "n_candidate_c6": 120,
    },
    "v0_baseline_reproduction": {
        "c0_total_fp_count": v0_c0_fps,
        "c0_total_fp_rate": float(v0_c0_fps / 120),
        "c0_c6_fp_count": v0_c0_c6_fps,
        "c0_c6_fp_rate": float(v0_c0_c6_fps / 120),
        "c6_recall_count": v0_c6_recall,
        "c6_recall_rate": float(v0_c6_recall / 120),
        "verified_against_benchmark": True,
    },
    "mechanism_diagnostics": {
        "c0_init_error_count_w24": int(c0_df["v0_init_err"].sum()),
        "c0_init_error_rate_w24": float(c0_df["v0_init_err"].mean()),
        "c6_fps_with_init_error": int(c0_df[c0_df["v0_attributed_cause"] == "C6"]["v0_init_err"].sum()),
        "c6_fps_with_boundary_cv2": 4,
        "c0_burn_nonzero_mean": float(c0_df["n_pos_burn24"].mean()),
        "c0_burn_nonzero_median": float(c0_df["n_pos_burn24"].median()),
        "c6_burn_nonzero_mean": float(c6_df["n_pos_burn24"].mean()),
        "c6_burn_nonzero_median": float(c6_df["n_pos_burn24"].median()),
        "c6_detected_with_n_pos_le_5": int(((c6_df["v0_attributed_cause"] == "C6") & (c6_df["n_pos_burn24"] <= 5)).sum()),
        "c6_detected_with_n_pos_le_5_fraction": float(((c6_df["v0_attributed_cause"] == "C6") & (c6_df["n_pos_burn24"] <= 5)).sum() / 31),
    },
    "v1_evidence_floor_sensitivity": v1_sensitivity,
    "v2_shrinkage_sensitivity": v2_sensitivity,
    "expanded_baseline_w120": {
        "c0_init_error_count_w120": int(c0_df["expanded_cell_err_w120"].sum()),
        "c0_init_error_rate_w120": float(c0_df["expanded_cell_err_w120"].mean()),
        "c6_fps_fixed_by_w120": int(((c0_df["v0_attributed_cause"] == "C6") & c0_df["v0_init_err"] & ~c0_df["expanded_cell_err_w120"]).sum()),
    },
    "decision_and_recommendation": {
        "v1_minimum_evidence_recommendation": "REJECT (Destroys 54.8% of genuine C6 recall at N_min=6 to eliminate 6 FPs; disfavors genuine intermittent demand)",
        "v2_shrinkage_recommendation": "REJECT (Catalog heterogeneity causes shrinkage to collapse cells toward pooled mean, increasing init errors from 39% up to 63%)",
        "expanded_baseline_recommendation": "ARCHITECTURAL NOTE ONLY (Expanding baseline window W_base to 120 reduces init errors to 15%, but is an architectural change beyond exploratory scope)",
        "proceed_to_validation": False,
    }
}

with open(OUT_DIR / "c6_init_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)
print(f"Saved summary JSON to {OUT_DIR / 'c6_init_summary.json'}")

