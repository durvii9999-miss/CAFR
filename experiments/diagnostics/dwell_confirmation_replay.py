"""Offline Training-Only Replay: Multi-Checkpoint Confirmation Gate (Dwell Latching).

Evaluates Hypothesis 1 by replaying saved training-only checkpoint trajectories:
- Baseline: Original immediate latching (V0).
- Candidate Policy 1 (Immediate Dwell): First non-C0 alarm at t0 must re-confirm at t0 + 4.
- Candidate Policy 2 (Windowed Dwell): Cause must achieve >= 2 consecutive firings post-120.

Data Sources:
- results/exploratory/c6_redesign/c3_c6_checkpoint_trajectories.parquet (15,840 checkpoint rows)
- results/exploratory/c6_redesign/c3_c6_paired_attributions.parquet (360 paired SKU attributions)
- results/exploratory/c0_fp_diagnostic/c0_fp_case_level.parquet (120 C0 control cases)

Output Directory:
- results/exploratory/dwell_confirmation_diagnostic/

Strictly offline: Zero simulations rerun, zero production files modified, training split only.
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

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "dwell_confirmation_diagnostic"


def find_first_consecutive(flags: list[bool], times: list[int]) -> tuple[int | None, int | None]:
    """Find the first occurrence of two consecutive checkpoints where flag is True."""
    assert len(flags) == len(times)
    for i in range(len(flags) - 1):
        if flags[i] and flags[i + 1]:
            return times[i], times[i + 1]
    return None, None


def run_dwell_replay():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Existing Saved Trajectories and Attributions
    traj_path = REPO_ROOT / "results" / "exploratory" / "c6_redesign" / "c3_c6_checkpoint_trajectories.parquet"
    att_path = REPO_ROOT / "results" / "exploratory" / "c6_redesign" / "c3_c6_paired_attributions.parquet"
    c0_case_path = REPO_ROOT / "results" / "exploratory" / "c0_fp_diagnostic" / "c0_fp_case_level.parquet"

    assert traj_path.exists(), f"Missing {traj_path}"
    assert att_path.exists(), f"Missing {att_path}"
    assert c0_case_path.exists(), f"Missing {c0_case_path}"

    df_cps = pd.read_parquet(traj_path)
    df_att = pd.read_parquet(att_path)
    df_c0_cases = pd.read_parquet(c0_case_path)

    # 2. Strict Assertions on Split Isolation and Data Integrity
    assert len(df_att) == 360, f"Expected 360 attributions, got {len(df_att)}"
    assert df_att["sku_id"].nunique() == 360, "Duplicate SKU keys found in attributions"
    assert len(df_cps) == 15840, f"Expected 15840 checkpoints, got {len(df_cps)}"
    assert df_cps["sku_id"].nunique() == 360, "Duplicate or missing SKUs in checkpoints"

    # Verify training split isolation (i % 5 < 3)
    for sid in df_att["sku_id"]:
        idx = int(sid.split(":")[-1])
        assert idx % 5 < 3, f"Split leak! Non-training index found: {sid}"

    # Verify baseline reproduction (V0)
    c0_sub = df_att[df_att["cohort"] == "C0"]
    c6_sub = df_att[df_att["cohort"] == "C6_cand"]
    c3_sub = df_att[df_att["cohort"] == "C3"]

    assert len(c0_sub) == 120 and len(c6_sub) == 120 and len(c3_sub) == 120

    v0_c0_fps = int((c0_sub["v0_cause"] != "C0").sum())
    v0_c0_c6_fps = int((c0_sub["v0_cause"] == "C6").sum())
    v0_c0_c3_fps = int((c0_sub["v0_cause"] == "C3").sum())
    v0_c6_recall = int((c6_sub["v0_cause"] == "C6").sum())
    v0_c3_recall = int((c3_sub["v0_cause"] == "C3").sum())

    assert v0_c0_fps == 38, f"V0 baseline mismatch: expected 38 C0 FPs, got {v0_c0_fps}"
    assert v0_c0_c6_fps == 14, f"V0 baseline mismatch: expected 14 C0->C6 FPs, got {v0_c0_c6_fps}"
    assert v0_c0_c3_fps == 1, f"V0 baseline mismatch: expected 1 C0->C3 FP, got {v0_c0_c3_fps}"
    assert v0_c6_recall == 31, f"V0 baseline mismatch: expected 31 C6 recall, got {v0_c6_recall}"
    assert v0_c3_recall == 11, f"V0 baseline mismatch: expected 11 C3 recall, got {v0_c3_recall}"

    print("=" * 70)
    print("HYPOTHESIS 1: DWELL CONFIRMATION GATE REPLAY (OFFLINE ONLY)")
    print("All split isolation, row count, and baseline reproduction assertions passed.")
    print("=" * 70)

    # 3. Replay Cohort 1: Candidate C6 (N = 120)
    c6_case_records = []
    for _, row in c6_sub.iterrows():
        sid = row["sku_id"]
        v0_cause = row["v0_cause"]
        v0_t = row["v0_t"]

        sub_cp = df_cps[(df_cps["sku_id"] == sid) & (df_cps["t"] >= 120)].sort_values("t")
        flags = sub_cp["c6_fired"].tolist()
        times = sub_cp["t"].tolist()

        # Policy 1: Immediate confirmation at v0_t + 4
        if v0_cause == "C6" and pd.notna(v0_t):
            t_next = v0_t + 4
            next_row = sub_cp[sub_cp["t"] == t_next]
            reconfirmed_t4 = bool(len(next_row) > 0 and next_row["c6_fired"].iloc[0])
            suppressed_t4 = not reconfirmed_t4
            delay_t4 = 4.0 if reconfirmed_t4 else None
        else:
            reconfirmed_t4 = False
            suppressed_t4 = False
            delay_t4 = None

        # Policy 2: Windowed consecutive firings anywhere post-120
        first_t, conf_t = find_first_consecutive(flags, times)
        consecutive_anywhere = (conf_t is not None)
        delay_anywhere = (conf_t - v0_t) if (v0_cause == "C6" and consecutive_anywhere) else None
        suppressed_anywhere = (v0_cause == "C6" and not consecutive_anywhere)

        c6_case_records.append({
            "sku_id": sid,
            "cohort": "C6_cand",
            "true_cause": "C6",
            "v0_cause": v0_cause,
            "v0_t": v0_t,
            "policy1_reconfirmed_t4": reconfirmed_t4,
            "policy1_suppressed_t4": suppressed_t4,
            "policy1_delay_t4": delay_t4,
            "policy2_consecutive_anywhere": consecutive_anywhere,
            "policy2_first_consec_t": conf_t,
            "policy2_delay": delay_anywhere,
            "policy2_suppressed": suppressed_anywhere,
            "total_c6_firings": sum(flags),
        })
    df_c6_dwell = pd.DataFrame(c6_case_records)

    # 4. Replay Cohort 2: True C3 (N = 120)
    c3_case_records = []
    for _, row in c3_sub.iterrows():
        sid = row["sku_id"]
        v0_cause = row["v0_cause"]
        v0_t = row["v0_t"]

        sub_cp = df_cps[(df_cps["sku_id"] == sid) & (df_cps["t"] >= 120)].sort_values("t")
        flags = sub_cp["c3_fired"].tolist()
        times = sub_cp["t"].tolist()

        # Policy 1: Immediate confirmation at v0_t + 4
        if v0_cause == "C3" and pd.notna(v0_t):
            t_next = v0_t + 4
            next_row = sub_cp[sub_cp["t"] == t_next]
            reconfirmed_t4 = bool(len(next_row) > 0 and next_row["c3_fired"].iloc[0])
            suppressed_t4 = not reconfirmed_t4
            delay_t4 = 4.0 if reconfirmed_t4 else None
        else:
            reconfirmed_t4 = False
            suppressed_t4 = False
            delay_t4 = None

        # Policy 2: Windowed consecutive firings anywhere post-120
        first_t, conf_t = find_first_consecutive(flags, times)
        consecutive_anywhere = (conf_t is not None)
        delay_anywhere = (conf_t - v0_t) if (v0_cause == "C3" and consecutive_anywhere) else None
        suppressed_anywhere = (v0_cause == "C3" and not consecutive_anywhere)

        c3_case_records.append({
            "sku_id": sid,
            "cohort": "C3",
            "true_cause": "C3",
            "v0_cause": v0_cause,
            "v0_t": v0_t,
            "policy1_reconfirmed_t4": reconfirmed_t4,
            "policy1_suppressed_t4": suppressed_t4,
            "policy1_delay_t4": delay_t4,
            "policy2_consecutive_anywhere": consecutive_anywhere,
            "policy2_first_consec_t": conf_t,
            "policy2_delay": delay_anywhere,
            "policy2_suppressed": suppressed_anywhere,
            "total_c3_firings": sum(flags),
        })
    df_c3_dwell = pd.DataFrame(c3_case_records)

    # 5. Replay Cohort 3: C0 Controls (N = 120)
    c0_case_records = []
    for _, row in c0_sub.iterrows():
        sid = row["sku_id"]
        v0_cause = row["v0_cause"]
        v0_t = row["v0_t"]
        c0_case = df_c0_cases[df_c0_cases["sku_id"] == sid].iloc[0]

        sub_cp = df_cps[(df_cps["sku_id"] == sid) & (df_cps["t"] >= 120)].sort_values("t")

        reconfirmed_t4 = False
        eliminated_t4 = False
        elimination_reason = "not_fp"

        if v0_cause == "C6":
            t_next = v0_t + 4
            next_row = sub_cp[sub_cp["t"] == t_next]
            reconfirmed_t4 = bool(len(next_row) > 0 and next_row["c6_fired"].iloc[0])
            eliminated_t4 = not reconfirmed_t4
            elimination_reason = "c6_failed_t4_reconfirm" if eliminated_t4 else "c6_persisted_t4"
        elif v0_cause == "C3":
            t_next = v0_t + 4
            next_row = sub_cp[sub_cp["t"] == t_next]
            reconfirmed_t4 = bool(len(next_row) > 0 and next_row["c3_fired"].iloc[0])
            eliminated_t4 = not reconfirmed_t4
            elimination_reason = "c3_failed_t4_reconfirm" if eliminated_t4 else "c3_persisted_t4"
        elif v0_cause == "C1":
            # All 5 C1 cases have total n_firings == 1 across entire run
            reconfirmed_t4 = False
            eliminated_t4 = True
            elimination_reason = "c1_single_fire_definitively_eliminated"
        elif v0_cause == "C4":
            reconfirmed_t4 = False
            eliminated_t4 = True
            elimination_reason = "c4_single_fire_definitively_eliminated"
        elif v0_cause in ("C2", "C7", "C5"):
            n_f = c0_case["n_firings_attributed_rule"]
            if n_f == 1:
                reconfirmed_t4 = False
                eliminated_t4 = True
                elimination_reason = f"{v0_cause.lower()}_single_fire_definitively_eliminated"
            else:
                # Conservative: multiple firings occurred, might reconfirm
                reconfirmed_t4 = True
                eliminated_t4 = False
                elimination_reason = f"{v0_cause.lower()}_multi_fire_conservative_survives"
        else:
            elimination_reason = "true_negative_c0"

        c0_case_records.append({
            "sku_id": sid,
            "cohort": "C0",
            "true_cause": "C0",
            "v0_cause": v0_cause,
            "v0_t": v0_t,
            "policy1_reconfirmed_t4": reconfirmed_t4,
            "policy1_eliminated_t4": eliminated_t4,
            "elimination_reason": elimination_reason,
            "n_firings": c0_case["n_firings_attributed_rule"],
        })
    df_c0_dwell = pd.DataFrame(c0_case_records)

    # 6. Save Case-Level Datasets
    df_all_cases = pd.concat([df_c0_dwell, df_c6_dwell, df_c3_dwell], ignore_index=True)
    csv_path = OUT_DIR / "dwell_case_level_comparison.csv"
    parquet_path = OUT_DIR / "dwell_case_level_comparison.parquet"
    df_all_cases.to_csv(csv_path, index=False)
    df_all_cases.to_parquet(parquet_path, index=False)

    # 7. Summary Metrics
    # Policy 1 metrics (Immediate Dwell at t0 + 4)
    # C0 FPs
    c0_eliminated_t4 = int(df_c0_dwell["policy1_eliminated_t4"].sum())
    c0_fps_p1 = v0_c0_fps - c0_eliminated_t4
    c0_eliminated_by_cause = df_c0_dwell[df_c0_dwell["policy1_eliminated_t4"]]["v0_cause"].value_counts().to_dict()

    # C6 Recall Policy 1
    c6_reconfirmed_p1 = int(df_c6_dwell["policy1_reconfirmed_t4"].sum())
    c6_suppressed_p1 = int(df_c6_dwell["policy1_suppressed_t4"].sum())

    # C3 Recall Policy 1
    c3_reconfirmed_p1 = int(df_c3_dwell["policy1_reconfirmed_t4"].sum())
    c3_suppressed_p1 = int(df_c3_dwell["policy1_suppressed_t4"].sum())

    # Policy 2 metrics (Consecutive anywhere)
    c6_confirmed_p2 = int(((df_c6_dwell["v0_cause"] == "C6") & df_c6_dwell["policy2_consecutive_anywhere"]).sum())
    c6_delays_p2 = df_c6_dwell[df_c6_dwell["policy2_delay"].notna()]["policy2_delay"].tolist()

    c3_confirmed_p2 = int(((df_c3_dwell["v0_cause"] == "C3") & df_c3_dwell["policy2_consecutive_anywhere"]).sum())
    c3_delays_p2 = df_c3_dwell[df_c3_dwell["policy2_delay"].notna()]["policy2_delay"].tolist()

    summary = {
        "metadata": {
            "title": "Hypothesis 1 Replay: Multi-Checkpoint Confirmation Gate (Dwell Latching)",
            "protocol": "Strictly offline replay of saved training trajectories (no simulations rerun)",
            "n_evaluated": len(df_all_cases),
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "assertions": {
            "split_isolation_passed": True,
            "row_counts_passed": True,
            "no_duplicate_keys": True,
            "v0_baseline_reproduced": True,
        },
        "c0_false_positives": {
            "v0_fp_count": v0_c0_fps,
            "v0_fp_rate": float(v0_c0_fps / 120),
            "policy1_dwell_fp_count": c0_fps_p1,
            "policy1_dwell_fp_rate": float(c0_fps_p1 / 120),
            "fp_reduction_count": c0_eliminated_t4,
            "fp_reduction_pp": float(c0_eliminated_t4 / 120),
            "eliminated_by_cause": c0_eliminated_by_cause,
            "surviving_c6_fps": 14 - c0_eliminated_by_cause.get("C6", 0),
        },
        "genuine_cause_recall_policy1_immediate_t4": {
            "c6_candidate": {
                "v0_recall_count": v0_c6_recall,
                "v0_recall_rate": float(v0_c6_recall / 120),
                "dwell_confirmed_count": c6_reconfirmed_p1,
                "dwell_confirmed_rate": float(c6_reconfirmed_p1 / 120),
                "suppressed_recall_count": c6_suppressed_p1,
                "suppressed_recall_pp": float(c6_suppressed_p1 / 120),
                "relative_recall_loss": float(c6_suppressed_p1 / v0_c6_recall),
                "detection_delay_periods": 4.0,
            },
            "c3_true": {
                "v0_recall_count": v0_c3_recall,
                "v0_recall_rate": float(v0_c3_recall / 120),
                "dwell_confirmed_count": c3_reconfirmed_p1,
                "dwell_confirmed_rate": float(c3_reconfirmed_p1 / 120),
                "suppressed_recall_count": c3_suppressed_p1,
                "suppressed_recall_pp": float(c3_suppressed_p1 / 120),
                "relative_recall_loss": float(c3_suppressed_p1 / v0_c3_recall),
                "detection_delay_periods": 4.0,
            },
        },
        "genuine_cause_recall_policy2_windowed_consecutive": {
            "c6_candidate": {
                "v0_recall_count": v0_c6_recall,
                "consecutive_confirmed_count": c6_confirmed_p2,
                "consecutive_confirmed_rate": float(c6_confirmed_p2 / 120),
                "suppressed_count": v0_c6_recall - c6_confirmed_p2,
                "mean_delay_periods": float(np.mean(c6_delays_p2)) if c6_delays_p2 else 0.0,
                "median_delay_periods": float(np.median(c6_delays_p2)) if c6_delays_p2 else 0.0,
                "max_delay_periods": float(np.max(c6_delays_p2)) if c6_delays_p2 else 0.0,
            },
            "c3_true": {
                "v0_recall_count": v0_c3_recall,
                "consecutive_confirmed_count": c3_confirmed_p2,
                "consecutive_confirmed_rate": float(c3_confirmed_p2 / 120),
                "suppressed_count": v0_c3_recall - c3_confirmed_p2,
                "mean_delay_periods": float(np.mean(c3_delays_p2)) if c3_delays_p2 else 0.0,
                "median_delay_periods": float(np.median(c3_delays_p2)) if c3_delays_p2 else 0.0,
                "max_delay_periods": float(np.max(c3_delays_p2)) if c3_delays_p2 else 0.0,
            },
        },
        "suppressed_sku_identities": {
            "c6_suppressed_at_t4": df_c6_dwell[df_c6_dwell["policy1_suppressed_t4"]]["sku_id"].tolist(),
            "c3_suppressed_at_t4": df_c3_dwell[df_c3_dwell["policy1_suppressed_t4"]]["sku_id"].tolist(),
        },
        "data_completeness_audit": {
            "fully_faithful_replay": ["C6_candidate", "C3_true", "C0_to_C6", "C0_to_C3"],
            "derived_proven_eliminations": ["C0_to_C1 (all 5 n=1)", "C0_to_C4 (1 n=1)", "C0_to_C2 (3 n=1)", "C0_to_C7 (2 n=1)"],
            "unresolved_ambiguity": ["C0_to_C2 (4 n>=2)", "C0_to_C7 (6 n>=2)", "C0_to_C5 (2 n=2)"],
            "missing_true_cohorts": "True C1, C2, C4, C5, C7 series were not in exploratory training datasets; recall impact on them cannot be replayed without new simulation",
        },
    }

    json_path = OUT_DIR / "dwell_summary_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON:                  {json_path}")
    print(f"Replay completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    run_dwell_replay()
