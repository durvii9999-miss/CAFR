"""Gate 5 report from a completed Step-5 run, with Wilson 95 % intervals.

Reads ``results/step5/step5_readings.parquet`` (one row per SKU) and
``step5_summary.json`` (for gate 5.4, which needs the live readings) and prints every
pre-registered gate with its denominator, point estimate and 95 % Wilson interval.

Nothing here changes a threshold. Every number is a re-tabulation of what the run
produced.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step5"
CAUSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
ALL = ["C0"] + CAUSES + ["C7"]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": float(k / n) if n else float("nan"),
            "ci95": [float(lo), float(hi)]}


def fmt(d: dict) -> str:
    if not np.isfinite(d["rate"]):
        return f"n/a (n=0)"
    return f"{d['rate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]  ({d['k']}/{d['n']})"


def main() -> None:
    df = pd.read_parquet(OUT / "step5_readings.parquet")
    before_df = pd.read_parquet(OUT / "e8_before" / "step5_readings.parquet")
    summary = json.loads((OUT / "step5_summary.json").read_text())
    params = pd.read_parquet(ROOT / "data" / "synth" / "generator_params.parquet")

    print("=" * 78)
    print("STEP 5 -- PRE-REGISTERED GATES (recall bar 0.60, FP bar 0.10, off-diag 0.40)")
    print("=" * 78)

    print("\n5.1 Recall, pooled (all SKUs of that cause)")
    for c in CAUSES + ["C7"]:
        sub = df[df["truth_for_gate"] == c]
        k = int((sub["attributed_cause"] == c).sum())
        tag = "" if c in CAUSES else "   [reported, not gated]"
        print(f"   {c}: {fmt(rate(k, len(sub)))} {tag}")

    print("\n5.1 Recall, budget-restricted (cause statistic had evidence after tau)")
    for c in CAUSES + ["C7"]:
        col = f"evany_c{c[1].lower()}"
        sub = df[df["truth_for_gate"] == c]
        if col in sub.columns:
            sub = sub[sub[col].astype(bool)]
        k = int((sub["attributed_cause"] == c).sum())
        print(f"   {c}: {fmt(rate(k, len(sub)))}")

    c2b = df[df["true_cause"] == "C2b"]
    if len(c2b):
        k = int((c2b["attributed_cause"] == "C2").sum())
        print(f"   C2b variant -> C2: {fmt(rate(k, len(c2b)))}")
    c2b_params = params[params["true_cause"] == "C2b"]
    if len(c2b_params):
        signed_gap = c2b_params["coverage_gap"]
        abs_gap = signed_gap.abs()
        print(f"   C2b induced coverage-gap median: {signed_gap.median():.6f}; "
              f"absolute-gap median: {abs_gap.median():.6f} "
              f"(n={abs_gap.notna().sum()})")
    c4_rows = df[df["true_cause"] == "C4"]
    c4_auc = c4_rows["c3_auc"].dropna()
    print(f"   C4 occurrence AUC diagnostic median: {c4_auc.median():.6f} "
          f"(estimable {len(c4_auc)}/{len(c4_rows)})")

    print("\n5.2 False positives on the C0 control panel (any post-tau non-C0 attribution)")
    c0 = df[df["true_cause"] == "C0"]
    k = int((c0["attributed_cause"] != "C0").sum())
    print(f"   FP rate: {fmt(rate(k, len(c0)))}   bar <= 0.10")
    print("   What C0 SKUs were called:")
    for cause, n in c0["attributed_cause"].value_counts().items():
        print(f"      {cause}: {n}/{len(c0)} = {n / len(c0):.3f}")
    pre = int(c0["pre_injection_flag_t"].notna().sum())
    print(f"   Pre-injection flag (diagnostic): {fmt(rate(pre, len(c0)))}")
    nz = c0["n_flagged_checks"].astype(float).sum()
    nd = c0["n_checks"].astype(float).sum()
    print(f"   Per-check flag rate: {nz:.0f}/{nd:.0f} = {nz / max(nd, 1):.3f}")

    print("\n5.3 Row-normalised confusion, largest off-diagonal per row")
    labels = [c for c in ALL if (df["truth_for_gate"] == c).any()]
    conf = pd.crosstab(df["truth_for_gate"], df["attributed_cause"]).reindex(
        index=labels, columns=ALL, fill_value=0
    )
    rown = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0)
    for c in labels:
        if c not in rown.index:
            continue
        row = rown.loc[c].drop(labels=[c], errors="ignore")
        mx = float(row.max())
        arg = row.idxmax()
        print(f"   {c}: max off-diagonal {mx:.3f} -> {arg}   {'PASS' if mx <= 0.40 else 'FAIL'}")
    print("\n   Full row-normalised table:")
    print(rown.round(3).fillna(0).to_string())

    print("\n5.4 Priority-order sensitivity (from summary.json)")
    s = summary["5.4_priority_sensitivity"]
    print(f"   max |macro-F1 diff| over 10 permutations: {s['value_max_abs_diff']:.3f}"
          f"   bar <= 0.10   base macro-F1 {s['base_macro_f1']:.3f}")

    print("\n5.6 Budget-restricted recall (generator's own budget flags)")
    for c, col in (("C4", "c4_budget_met"), ("C5", "c5_budget_cleared")):
        sub = df[(df["truth_for_gate"] == c) & df[col].astype(bool)]
        k = int((sub["attributed_cause"] == c).sum())
        print(f"   {c}: {fmt(rate(k, len(sub)))}")


    print("\nPer-rule firing on C0 controls")
    print("   scored reading (same one-row-per-SKU definition before and after):")
    for cause in CAUSES + ["C7"]:
        print(f"   {cause} before: {fmt(_c0_scored_firing(before_df, cause))}; "
              f"after: {fmt(_c0_scored_firing(df, cause))}")
    for cause in ("C4", "C5"):
        d = summary["c0_per_rule_firing"][cause]
        print(f"      after any-check: {d['sku_rate']:.3f} "
              f"[{d['sku_wilson95'][0]:.3f}, {d['sku_wilson95'][1]:.3f}] "
              f"({d['sku_hits']}/{d['sku_n']})")
    c5 = summary["c0_per_rule_firing"]["C5"]
    occ_any = int((c0["c5_occ_alarm_checks"] > 0).sum())
    print(f"   C5 occurrence CUSUM alarmed at least once: "
          f"{fmt(rate(occ_any, len(c0)))}")
    print(f"   C5 occurrence CUSUM check rate: {c5['occ_cusum_check_rate']:.4f} "
          f"[{c5['occ_cusum_check_wilson95'][0]:.4f}, "
          f"{c5['occ_cusum_check_wilson95'][1]:.4f}] "
          f"({c5['occ_cusum_check_hits']}/{c5['check_n']})")
    calibration = summary["cusum_calibration"]
    print("\nC5 threshold calibration")
    print(f"   h={calibration['h']:.6f}, k={calibration['k']:.2f}; "
          f"target={calibration['target_per_check_false_alarm_rate']:.3f}, "
          f"observed={calibration['calibration_per_check_false_alarm_rate']:.5f} "
          f"({calibration['calibration_n_checks']} checks)")
    print(f"   training data: {calibration['training_controls']} C0 controls, "
          f"periods [{calibration['periods'][0]}, {calibration['periods'][1]})")

    print("\nDetection diagnostics (not gated)")
    for c in CAUSES + ["C0", "C7"]:
        sub = df[df["truth_for_gate"] == c]
        hit = sub[sub["detected"].astype(bool)]
        lat = hit["detect_latency"].median() if len(hit) else float("nan")
        print(f"   {c}: detected {len(hit)}/{len(sub)} = {len(hit)/max(len(sub),1):.3f}"
              f"   median latency {lat if np.isfinite(lat) else float('nan'):.1f}")

    print("\nGate 5.1 pass (pooled):", summary["5.1_recall"]["pass"])
    print("Gate 5.1 pass (budget-restricted):", summary["5.1_recall"].get("pass_budget_restricted"))
    print("Gate 5.2 pass:", summary["5.2_false_positive"]["pass"])
    print("Gate 5.3 pass:", summary["5.3_off_diagonal"]["pass"])
    print("Gate 5.4 pass:", summary["5.4_priority_sensitivity"]["pass"])
    f1 = {}
    for cause in CAUSES:
        tp = int(conf.loc[cause, cause])
        fp = int(conf[cause].sum() - tp)
        fn = int(conf.loc[cause].sum() - tp)
        f1[cause] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    print("Minimum per-class F1:", min(f1.values()), f1)
    print("OVERALL:", summary["overall_pass"])


def _c0_scored_firing(df: pd.DataFrame, cause: str) -> dict:
    c0 = df[df["true_cause"] == "C0"]
    fired = c0["fired"].fillna("").map(lambda value: cause in value.split(","))
    return rate(int(fired.sum()), len(c0))


if __name__ == "__main__":
    main()
