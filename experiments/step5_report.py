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
    summary = json.loads((OUT / "step5_summary.json").read_text())

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
    print("OVERALL:", summary["overall_pass"])


if __name__ == "__main__":
    main()
