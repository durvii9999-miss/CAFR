#!/usr/bin/env python3
"""Build the paper results pack from frozen Step 5/6 artifacts.

This script reads existing result files only. It does not run the simulator,
regenerate data, modify CAFR source, or change any configuration.

Outputs:
  paper/figures/fig1_step5_attribution.png
  paper/figures/fig2_step6_paired_outcomes.png
  paper/tables/table1_step5_gates.{csv,tex}
  paper/tables/table2_step6_gate.{csv,tex}
  paper/tables/table3_step6_outcomes.{csv,tex}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGDIR = PAPER / "figures"
TABDIR = PAPER / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

Z95 = 1.959963984540054
CAUSES = [f"C{i}" for i in range(8)]


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (centre - half, centre + half)


def step6_split(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the deterministic per-true-cause split used by Step 6."""
    parts = []
    cycle = np.array(["train", "train", "train", "val", "test"])
    for _, sub in df[df["true_cause"] != "C2b"].groupby("true_cause"):
        sub = sub.sort_values("sku_id").copy()
        sub["split"] = cycle[np.arange(len(sub)) % len(cycle)]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Read frozen artifacts
# ---------------------------------------------------------------------------

s5 = pd.read_parquet(ROOT / "results/step5/step5_readings.parquet")
s5_summary = json.loads((ROOT / "results/step5/step5_summary.json").read_text())
s5_confusion = pd.read_parquet(ROOT / "results/step5/step5_confusion.parquet")
s5_split = step6_split(s5)

gate = json.loads((ROOT / "results/step6/c2_gate_summary.json").read_text())
compare = json.loads((ROOT / "results/step6/c2_compare_summary.json").read_text())
arms = pd.read_csv(ROOT / "results/step6/c2_compare_arms.csv")
paired = pd.read_csv(ROOT / "results/step6/c2_compare_paired.csv")


# ---------------------------------------------------------------------------
# Step 5 values, computed from the readings
# ---------------------------------------------------------------------------

def recall_row(
    label: str,
    subset: pd.DataFrame,
    attributed: str,
    bar: str,
    verdict: str,
    detail: str = "",
) -> dict:
    n = int(len(subset))
    k = int((subset["attributed_cause"] == attributed).sum())
    lo, hi = wilson_ci(k, n)
    return {
        "panel": "A",
        "cause_or_gate": label,
        "detail": detail,
        "n": n,
        "correct": k,
        "estimate": k / n if n else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "bar": bar,
        "verdict": verdict,
    }


panel_a = [
    recall_row("C1", s5[s5["truth_for_gate"] == "C1"], "C1", "0.60", "FAIL"),
    recall_row(
        "C2 (pooled)",
        s5[s5["truth_for_gate"] == "C2"],
        "C2",
        "0.60",
        "FAIL",
        "C2 + C2b; gated denominator",
    ),
    recall_row(
        "C2 (direct)",
        s5[s5["true_cause"] == "C2"],
        "C2",
        "0.60",
        "PASS",
        "C2 panel only",
    ),
    recall_row(
        "C2b (variant)",
        s5[s5["true_cause"] == "C2b"],
        "C2",
        "0.60",
        "FAIL (variant)",
        "C2b panel only; excluded from gate",
    ),
    recall_row("C3", s5[s5["truth_for_gate"] == "C3"], "C3", "0.60", "FAIL"),
    recall_row("C4", s5[s5["truth_for_gate"] == "C4"], "C4", "0.60", "FAIL"),
    recall_row("C5", s5[s5["truth_for_gate"] == "C5"], "C5", "0.60", "FAIL"),
    recall_row("C6", s5[s5["truth_for_gate"] == "C6"], "C6", "0.60", "FAIL"),
    recall_row(
        "C7 (reported)",
        s5[s5["true_cause"] == "C7"],
        "C7",
        "not gated",
        "not gated",
        "reported, not gated",
    ),
]
panel_a_df = pd.DataFrame(panel_a)

# Cross-check the computed counts against the stored summary.
expected = {"C1": 6, "C2 (pooled)": 209, "C3": 11, "C4": 0, "C5": 22, "C6": 29}
for cause, k in expected.items():
    row = panel_a_df[panel_a_df["cause_or_gate"] == cause].iloc[0]
    assert int(row["correct"]) == k, (cause, row["correct"], k)
assert int(panel_a_df[panel_a_df["cause_or_gate"] == "C2 (direct)"]["correct"].iloc[0]) == 191
assert int(panel_a_df[panel_a_df["cause_or_gate"] == "C2b (variant)"]["correct"].iloc[0]) == 18
assert int(panel_a_df[panel_a_df["cause_or_gate"] == "C7 (reported)"]["correct"].iloc[0]) == 56

# Gate 5.2: any non-C0 attribution on the C0 control panel.
c0 = s5[s5["truth_for_gate"] == "C0"]
c0_fp_k = int((c0["attributed_cause"] != "C0").sum())
c0_fp_n = int(len(c0))
c0_fp = c0_fp_k / c0_fp_n
c0_fp_lo, c0_fp_hi = wilson_ci(c0_fp_k, c0_fp_n)
assert c0_fp_k == 90 and c0_fp_n == 200

# Gate 5.3: row-normalised max off-diagonal, truth_for_gate rows.
ct = pd.crosstab(s5["truth_for_gate"], s5["attributed_cause"])
ct = ct.reindex(index=CAUSES, columns=CAUSES, fill_value=0)
# The stored 8x8 confusion file excludes C2b and matches the C0/C1/C3-C7 rows.
assert s5_confusion.shape == (8, 8)
for cause in CAUSES:
    if cause == "C2":
        continue
    assert (ct.loc[cause] == s5_confusion.loc[cause]).all(), cause
rn = ct.div(ct.sum(axis=1), axis=0)
offdiag = {}
for cause in CAUSES:
    row = rn.loc[cause].drop(labels=[cause])
    offdiag[cause] = (float(row.max()), str(row.idxmax()))

# C2 true-only detail: the readings' true_cause C2 row, not the pooled row.
ct_true = pd.crosstab(s5["true_cause"], s5["attributed_cause"])
ct_true = ct_true.reindex(index=CAUSES, columns=CAUSES, fill_value=0)
c2_true_rn = ct_true.loc["C2"] / ct_true.loc["C2"].sum()
c2_true_off = c2_true_rn.drop(labels=["C2"])
c2_true_off_v = float(c2_true_off.max())
c2_true_off_to = str(c2_true_off.idxmax())

# Verify the off-diagonal values used in the paper against the stored table.
assert math.isclose(offdiag["C1"][0], 0.545, abs_tol=1e-12)
assert math.isclose(offdiag["C4"][0], 0.560, abs_tol=1e-12)
assert math.isclose(offdiag["C6"][0], 0.480, abs_tol=1e-12)
assert math.isclose(offdiag["C2"][0], 0.335, abs_tol=1e-12)
assert math.isclose(c2_true_off_v, 0.040, abs_tol=1e-12)

gate54 = s5_summary["5.4_priority_sensitivity"]
assert math.isclose(gate54["value_max_abs_diff"], 0.011081653026416477, abs_tol=1e-15)

panel_b = [
    {
        "panel": "B",
        "cause_or_gate": "5.2",
        "detail": "C0 false-positive rate (any non-C0 call)",
        "n": c0_fp_n,
        "correct": "",
        "estimate": c0_fp,
        "ci_low": c0_fp_lo,
        "ci_high": c0_fp_hi,
        "bar": "<= 0.10",
        "verdict": "FAIL",
    }
]
for cause in CAUSES:
    value, target = offdiag[cause]
    detail = f"max off-diagonal to {target}"
    if cause == "C2":
        detail += f"; true C2 only {c2_true_off_v:.3f} to {c2_true_off_to}"
    panel_b.append(
        {
            "panel": "B",
            "cause_or_gate": f"5.3 {cause}",
            "detail": detail,
            "n": int(ct.loc[cause].sum()),
            "correct": "",
            "estimate": value,
            "ci_low": "",
            "ci_high": "",
            "bar": "<= 0.40",
            "verdict": "FAIL" if cause in {"C1", "C4", "C6"} else "pass",
        }
    )
panel_b.extend(
    [
        {
            "panel": "B",
            "cause_or_gate": "5.4",
            "detail": "max |macro-F1 difference|, 10 permutations",
            "n": "",
            "correct": "",
            "estimate": float(gate54["value_max_abs_diff"]),
            "ci_low": "",
            "ci_high": "",
            "bar": "<= 0.10",
            "verdict": "PASS",
        },
        {
            "panel": "B",
            "cause_or_gate": "Overall",
            "detail": "Step 5 gate verdict; E-8 unchanged",
            "n": "",
            "correct": "",
            "estimate": "",
            "ci_low": "",
            "ci_high": "",
            "bar": "",
            "verdict": "FAIL",
        },
    ]
)
table1 = pd.concat([panel_a_df, pd.DataFrame(panel_b)], ignore_index=True)
table1.to_csv(TABDIR / "table1_step5_gates.csv", index=False)


# ---------------------------------------------------------------------------
# Step 6 values
# ---------------------------------------------------------------------------

gated = arms[arms["arm"] == "gated"].copy()
assert len(gated) == 80
assert int((gated["action"] == "R2").sum()) == 38
assert int(((gated["action"] == "R2") & (gated["true_cause"] == "C2")).sum()) == 37
assert int(((gated["action"] == "R2") & (gated["true_cause"] == "C0")).sum()) == 1
assert int((gated["attributed"] == "C2").sum()) == 44
assert int(((gated["attributed"] == "C2") & (gated["true_cause"] == "C2")).sum()) == 37
assert int((gated["reason"] == "low_reliability").sum()) == 6
assert int((gated["reason"] == "not_C2").sum()) == 14
assert int((gated["decision_match"] == True).sum()) == 80

# Panel A: gate metrics by split, with candidate counts recomputed from readings.
split_rows = []
for split in ["train", "val", "test"]:
    stored = gate["gate"]["by_split"][split]
    sub = s5_split[s5_split["split"] == split]
    cand = sub[(sub["detected"] == True) & (sub["attributed_cause"] == "C2")]
    true_c2_cand = int((cand["true_cause"] == "C2").sum())
    assert len(cand) == int(stored["candidates"]), (split, len(cand), stored["candidates"])
    assert true_c2_cand == {"train": 115, "val": 39, "test": 37}[split]
    split_rows.append(
        {
            "panel": "A",
            "split_or_group": split,
            "n_rows": int(len(sub)),
            "candidates": int(stored["candidates"]),
            "true_c2_candidates": true_c2_cand,
            "acted": int(stored["acted"]),
            "true_c2_acted": int(stored["true_c2_acted"]),
            "precision": float(stored["precision"]),
            "recall_all_c2": float(stored["recall_all_c2_split"]),
            "coverage_all_rows": float(stored["coverage_all_rows"]),
            "coverage_candidates": float(stored["coverage_candidates"]),
            "true_c0_action_rate": float(stored["true_c0_action_rate"]),
            "auroc": float(stored["auroc"]),
        }
    )
assert [r["n_rows"] for r in split_rows] == [960, 320, 320]

# Panel B: live actions on the 80-SKU comparison.
comparison_rows = [
    ("Acted", int((gated["action"] == "R2").sum()), 80, "R2 decisions"),
    ("True C2 acted", 37, 40, "recall in C2 group"),
    ("True C0 acted", 1, 40, "false action; gamma=1 no-op"),
    ("C2-attributed rows", 44, 80, "37 true C2 + 7 true C0"),
    ("Precision", 37, 38, "true C2 among acted"),
    ("Coverage", 38, 80, "acted / all compared SKUs"),
    ("Decision match", 80, 80, "live decision equals stored-reading decision"),
]
comparison_rows = [
    {
        "panel": "B",
        "split_or_group": "C2+C0 comparison (80 SKUs)",
        "metric": metric,
        "numerator": num,
        "denominator": den,
        "value": num / den,
        "note": note,
    }
    for metric, num, den, note in comparison_rows
]
table2_a = pd.DataFrame(split_rows)
table2_b = pd.DataFrame(comparison_rows)
table2 = pd.concat([table2_a, table2_b], ignore_index=True)
for col in [
    "n_rows",
    "candidates",
    "true_c2_candidates",
    "acted",
    "true_c2_acted",
    "numerator",
    "denominator",
]:
    table2[col] = table2[col].astype("Int64")
table2.to_csv(TABDIR / "table2_step6_gate.csv", index=False)


# ---------------------------------------------------------------------------
# Table 3: paired outcomes
# ---------------------------------------------------------------------------

group_specs = [
    ("C2", paired[paired["true_cause"] == "C2"], compare["paired_C2"], 40),
    ("C0", paired[paired["true_cause"] == "C0"], compare["paired_C0"], 40),
    ("All", paired, compare["paired_all"], 80),
]
outcome_rows = []
for name, psub, psum, n in group_specs:
    if name == "All":
        base = arms[arms["arm"] == "baseline"]
        gt = arms[arms["arm"] == "gated"]
    else:
        base = arms[(arms["arm"] == "baseline") & (arms["true_cause"] == name)]
        gt = arms[(arms["arm"] == "gated") & (arms["true_cause"] == name)]

    b_cost = float(base["cost"].mean())
    g_cost = float(gt["cost"].mean())
    b_csl = float(base["csl"].mean())
    g_csl = float(gt["csl"].mean())
    b_psl = float(base["period_sl"].mean())
    g_psl = float(gt["period_sl"].mean())

    # The stored summary is the source of the bootstrap intervals.
    d_cost = float(psum["d_cost_mean"])
    d_cost_lo, d_cost_hi = map(float, psum["d_cost_ci95"])
    d_csl = float(psum["d_csl_mean"])
    d_csl_lo, d_csl_hi = map(float, psum["d_csl_ci95"])

    # Recompute checks against the per-SKU paired table.
    assert math.isclose(d_cost, float(psub["d_cost"].mean()), rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(d_csl, float(psub["d_csl"].mean()), rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(g_cost - b_cost, d_cost, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(g_csl - b_csl, d_csl, rel_tol=1e-9, abs_tol=1e-12)

    if name == "C0":
        assert d_cost == 0.0 and d_csl == 0.0
        assert float(psub["d_period_sl"].mean()) == 0.0

    outcome_rows.append(
        {
            "group": name,
            "n": n,
            "baseline_cost": b_cost,
            "gated_cost": g_cost,
            "delta_cost": d_cost,
            "ci_cost_low": d_cost_lo,
            "ci_cost_high": d_cost_hi,
            "pct_cost_change": 100.0 * d_cost / b_cost,
            "baseline_csl": b_csl,
            "gated_csl": g_csl,
            "delta_csl_pp": 100.0 * d_csl,
            "ci_csl_low_pp": 100.0 * d_csl_lo,
            "ci_csl_high_pp": 100.0 * d_csl_hi,
            "baseline_period_sl": b_psl,
            "gated_period_sl": g_psl,
            "delta_period_sl_pp": 100.0 * float(psub["d_period_sl"].mean()),
            "cost_up": int((psub["d_cost"] > 1e-12).sum()),
            "cost_down": int((psub["d_cost"] < -1e-12).sum()),
            "cost_equal": int(np.isclose(psub["d_cost"], 0.0).sum()),
            "csl_up": int((psub["d_csl"] > 1e-12).sum()),
            "csl_down": int((psub["d_csl"] < -1e-12).sum()),
            "csl_equal": int(np.isclose(psub["d_csl"], 0.0).sum()),
        }
    )
table3 = pd.DataFrame(outcome_rows)
table3.to_csv(TABDIR / "table3_step6_outcomes.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 1: Step 5 per-cause recall
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
    }
)

fig_rows = []
for row in panel_a:
    label = row["cause_or_gate"]
    if label == "C1":
        fig_rows.append({**row, "plot_label": f"C1 (n={row['n']})", "color": "#b2182b"})
    elif label == "C2 (pooled)":
        fig_rows.append({**row, "plot_label": f"C2 pooled (n={row['n']})", "color": "#b2182b"})
    elif label == "C2 (direct)":
        fig_rows.append({**row, "plot_label": f"C2 direct (n={row['n']})", "color": "#1b7837"})
    elif label == "C2b (variant)":
        fig_rows.append({**row, "plot_label": f"C2b variant (n={row['n']})", "color": "#b2182b"})
    elif label == "C7 (reported)":
        fig_rows.append({**row, "plot_label": f"C7 reported (n={row['n']})", "color": "#777777"})
    else:
        fig_rows.append({**row, "plot_label": f"{label} (n={row['n']})", "color": "#b2182b"})

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ys = np.arange(len(fig_rows))[::-1]
for y, row in zip(ys, fig_rows):
    x = row["estimate"]
    lo, hi = row["ci_low"], row["ci_high"]
    ax.errorbar(
        x,
        y,
        xerr=[[max(0.0, x - lo)], [max(0.0, hi - x)]],
        fmt="o",
        ms=5.5,
        color=row["color"],
        ecolor=row["color"],
        elinewidth=1.4,
        capsize=3.5,
        capthick=1.1,
        zorder=3,
    )
    ax.text(
        1.035,
        y,
        f"{row['correct']}/{row['n']}",
        va="center",
        ha="left",
        fontsize=7.2,
        color="#333333",
    )

ax.axvline(0.60, color="#333333", linestyle="--", linewidth=1.0, zorder=1)
ax.text(
    0.605,
    len(fig_rows) - 0.45,
    "predeclared bar = 0.60 (C1-C6)",
    rotation=90,
    va="top",
    ha="left",
    fontsize=7.2,
    color="#333333",
)
ax.set_yticks(ys, [r["plot_label"] for r in fig_rows])
ax.set_xlim(0.0, 1.18)
ax.set_ylim(-0.7, len(fig_rows) - 0.3)
ax.set_xlabel("Recall: correct cause attributed at final reading")
ax.set_title("Step 5 attribution recall by cause (seed 42 synthetic panel)")
ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

fig.text(
    0.01,
    0.015,
    "Whiskers: 95% Wilson intervals. Green = passes 0.60; red = fails / variant; gray = reported, not gated.\n"
    "C2 pooled = C2 + C2b (n=400); C2 direct = C2 panel only (n=200).\n"
    "C0 control false-positive rate = 0.450 (90/200; 95% CI 0.383-0.519), bar <= 0.10: FAIL. Overall Step 5: FAIL.",
    fontsize=7.0,
    color="#333333",
    va="bottom",
)
fig.subplots_adjust(left=0.23, right=0.98, top=0.91, bottom=0.22)
fig.savefig(FIGDIR / "fig1_step5_attribution.png", dpi=300)
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Step 6 paired outcomes
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.42})
plot_groups = [
    ("C2", 2, "#1b7837"),
    ("All", 1, "#4d4d4d"),
    ("C0", 0, "#999999"),
]
lookup = {row["group"]: row for row in outcome_rows}

# Cost panel.
ax = axes[0]
for name, y, color in plot_groups:
    row = lookup[name]
    mean = row["delta_cost"]
    lo, hi = row["ci_cost_low"], row["ci_cost_high"]
    ax.errorbar(
        mean,
        y,
        xerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
        fmt="o",
        ms=6,
        color=color,
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        capthick=1.1,
        zorder=3,
    )
    ax.annotate(
        f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]",
        xy=(mean, y),
        xytext=(14, 9) if name == "C0" else (0, 9),
        textcoords="offset points",
        ha="left" if name == "C0" else "center",
        va="bottom",
        fontsize=7.2,
        color=color,
    )
ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9, zorder=1)
ax.set_yticks([2, 1, 0], ["C2 (n=40)", "All (n=80)", "C0 (n=40)"])
ax.set_ylim(-0.7, 2.6)
ax.set_xlim(-110, 235)
ax.set_xlabel("Delta cost, gated - baseline\n(cost units per SKU, 68 periods)")
ax.set_title("(a) Paired cost difference")
ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

# CSL panel.
ax = axes[1]
for name, y, color in plot_groups:
    row = lookup[name]
    mean = row["delta_csl_pp"]
    lo, hi = row["ci_csl_low_pp"], row["ci_csl_high_pp"]
    ax.errorbar(
        mean,
        y,
        xerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
        fmt="o",
        ms=6,
        color=color,
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        capthick=1.1,
        zorder=3,
    )
    ax.annotate(
        f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]",
        xy=(mean, y),
        xytext=(14, 9) if name == "C0" else (0, 9),
        textcoords="offset points",
        ha="left" if name == "C0" else "center",
        va="bottom",
        fontsize=7.2,
        color=color,
    )
ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9, zorder=1)
ax.set_yticks([2, 1, 0], ["C2 (n=40)", "All (n=80)", "C0 (n=40)"])
ax.set_ylim(-0.7, 2.6)
ax.set_xlim(-1.0, 10.0)
ax.set_xlabel("Delta cycle CSL, gated - baseline\n(percentage points, 22 cycles)")
ax.set_title("(b) Paired cycle service-level difference")
ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

fig.text(
    0.01,
    0.015,
    "Points: mean paired difference; whiskers: 95% percentile bootstrap (10,000 resamples, seed 42).\n"
    "C0 is exactly 0/40 on both outcomes. Cost CIs for C2 and All include zero; CSL CIs exclude zero. Positive = higher cost / higher service.",
    fontsize=7.0,
    color="#333333",
    va="bottom",
)
fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.27)
fig.savefig(FIGDIR / "fig2_step6_paired_outcomes.png", dpi=300)
plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def f3(x) -> str:
    if x == "" or x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{float(x):.3f}"


def f2(x) -> str:
    if x == "" or x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{float(x):.2f}"


# Table 1.
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Step 5 attribution gates on the seed-42 synthetic panel. Panel A: per-cause recall (correct cause attributed at the final reading) with 95\% Wilson intervals. The predeclared bar is 0.60 for C1--C6. C2 pooled is the gated denominator (C2 + C2b); C2 direct is the C2 panel alone. Panel B: control false-positive rate, row-normalised maximum off-diagonal confusion, and priority-order sensitivity. C2b is a variant panel and is excluded from gate 5.3. Overall verdict: FAIL; E-8 unchanged.}",
    r"\label{tab:step5-gates}",
    r"\small",
    r"\begin{tabular}{lrrrrl}",
    r"\toprule",
    r"Cause & $N$ & Correct & Recall & 95\% Wilson CI & Verdict \\",
    r"\midrule",
]
for row in panel_a:
    label = row["cause_or_gate"]
    if label == "C2 (pooled)":
        label = r"C2 pooled"
    elif label == "C2 (direct)":
        label = r"C2 direct"
    elif label == "C2b (variant)":
        label = r"C2b variant"
    elif label == "C7 (reported)":
        label = r"C7 reported"
    verdict = row["verdict"].replace("FAIL (variant)", r"FAIL$^\dagger$")
    lines.append(
        f"{label} & {row['n']} & {row['correct']} & {f3(row['estimate'])} & "
        f"[{f3(row['ci_low'])}, {f3(row['ci_high'])}] & {verdict} \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\begin{tabular}{llrrrl}",
    r"\toprule",
    r"Gate & Detail & $n$ & Estimate & 95\% CI & Verdict \\",
    r"\midrule",
]
for row in panel_b:
    gate_name = row["cause_or_gate"]
    detail = row["detail"].replace("_", r"\_")
    n = row["n"] if row["n"] != "" else "--"
    if row["ci_low"] != "":
        ci = f"[{f3(row['ci_low'])}, {f3(row['ci_high'])}]"
    else:
        ci = "--"
    if gate_name == "Overall":
        lines.append(r"\multicolumn{5}{l}{Overall verdict (gates 5.1--5.5)} & \textbf{FAIL} \\")
    else:
        lines.append(
            f"{gate_name} & {detail} & {n} & {f3(row['estimate'])} & {ci} & {row['verdict']} \\\\"
        )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\footnotesize",
    r"$^\dagger$C2b is a variant panel; its recall is reported but it is not part of the six-cause gate. Gate 5.5 (outputs produced) passed; gate 5.6 budget-restricted recall does not change any verdict. C0 row in Panel B is the false-positive rate, not recall. C2 max off-diagonal is 0.335 to C0 on the pooled denominator; the true-C2-only value is 0.040 to C5.",
    r"\end{table*}",
]
(TABDIR / "table1_step5_gates.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Table 2.
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Step 6 reliability-gate performance. Panel A: stored Step 5 readings by split; ``Recall'' divides by all true C2 SKUs in the split, and ``Coverage (rows)'' divides by all non-C2b rows. The validation split selected $\tau=0.2446$; the test split was untouched. Panel B: live actions on the 80-SKU C2+C0 comparison. The comparison precision of 0.974 is not the all-cause test precision of 0.822.}",
    r"\label{tab:step6-gate}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{lrrrrrrrrrrr}",
    r"\toprule",
    r"Split & Rows & Candidates & True C2 cand. & Acted & True C2 acted & Precision & Recall & Coverage (rows) & Coverage (cand.) & C0 action & AUROC \\",
    r"\midrule",
]
for row in split_rows:
    lines.append(
        f"{row['split_or_group'].capitalize()} & {row['n_rows']} & {row['candidates']} & "
        f"{row['true_c2_candidates']} & {row['acted']} & {row['true_c2_acted']} & "
        f"{f3(row['precision'])} & {f3(row['recall_all_c2'])} & {f3(row['coverage_all_rows'])} & "
        f"{f3(row['coverage_candidates'])} & {f3(row['true_c0_action_rate'])} & {f3(row['auroc'])} \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\begin{tabular}{lr}",
    r"\toprule",
    r"Panel B: C2+C0 comparison & Value \\",
    r"\midrule",
]
for row in comparison_rows:
    lines.append(
        f"{row['metric']} & {row['numerator']}/{row['denominator']} ({f3(row['value'])}) \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\footnotesize",
    r"Notes: Candidates are rows with \texttt{detected=True} and \texttt{attributed\_cause=C2}. Panel B has 44 C2-attributed rows (37 true C2 + 7 true C0); 38 are acted (37 true C2 + 1 true C0). Six low-reliability fallbacks are all true C0; 14 \texttt{not\_C2} fallbacks comprise 3 true C2 attributed C5 and 11 true C0. The single false action was \texttt{control:C0:0099}, whose $\gamma=1.0$ made the scale reset a no-op.",
    r"\end{table*}",
]
(TABDIR / "table2_step6_gate.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Table 3.
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Step 6 paired inventory outcomes (seed 42; 40 C2 + 40 C0 test SKUs). Differences are gated minus baseline over the 68-period window $[132,200)$; cycle CSL uses 22 complete 3-period cycles. Cost CIs and CSL CIs are percentile bootstrap intervals (10,000 resamples, seed 42). C0 is exactly zero on all 40 SKUs. Period service-level CIs were not stored and are not inferred.}",
    r"\label{tab:step6-outcomes}",
    r"\small",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"\multicolumn{6}{l}{\textit{Panel A: cost}} \\",
    r"Group & $n$ & Baseline & Gated & $\Delta$ [95\% CI] & $\Delta$ (\%) \\",
    r"\midrule",
]
for row in outcome_rows:
    lines.append(
        f"{row['group']} & {row['n']} & {f2(row['baseline_cost'])} & {f2(row['gated_cost'])} & "
        f"${row['delta_cost']:+.2f}$ $[{row['ci_cost_low']:+.2f}, {row['ci_cost_high']:+.2f}]$ & "
        f"${row['pct_cost_change']:+.2f}$ \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\begin{tabular}{lrrrrrrr}",
    r"\toprule",
    r"\multicolumn{8}{l}{\textit{Panel B: service}} \\",
    r"Group & $n$ & Baseline CSL & Gated CSL & $\Delta$ CSL (pp) [95\% CI] & Baseline period SL & Gated period SL & $\Delta$ period SL (pp) \\",
    r"\midrule",
]
for row in outcome_rows:
    lines.append(
        f"{row['group']} & {row['n']} & {f3(row['baseline_csl'])} & {f3(row['gated_csl'])} & "
        f"${row['delta_csl_pp']:+.2f}$ $[{row['ci_csl_low_pp']:+.2f}, {row['ci_csl_high_pp']:+.2f}]$ & "
        f"{f3(row['baseline_period_sl'])} & {f3(row['gated_period_sl'])} & "
        f"${row['delta_period_sl_pp']:+.2f}$ \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\footnotesize",
    r"Notes: C2 cost direction: 21 up / 16 down / 3 equal; C2 cycle CSL: 27 up / 0 down / 13 equal. The C2 cost interval includes zero; the C2 CSL interval is entirely above zero. C0 is 0/40 changed on both outcomes. The overall cost interval also includes zero.",
    r"\end{table*}",
]
(TABDIR / "table3_step6_outcomes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Paper files present:")
for path in sorted(PAPER.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(ROOT))
