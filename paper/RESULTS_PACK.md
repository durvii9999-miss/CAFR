# Paper results pack — Step 5 attribution and Step 6 C2-only feasibility

**Scope:** synthetic panel only, seed 42. This pack is a re-presentation of frozen
Step 5 and Step 6 artifacts. It contains **no new experiment, no regenerated data,
no source or configuration change, and no new simulation or bootstrap**. The only
derived interval is the 95% Wilson interval computed from stored integer counts;
all other values are read from, or recomputed directly from, stored artifacts for
verification. The Step 5 overall **FAIL** and **E-8** stand unchanged. The original
claim that full CAFR beats accuracy-based selection at equal churn is **not**
supported and is not restated here.

**Built by:** `paper/make_results_pack.py` (reads existing artifacts only).
**Run date:** 2026-10-18.

**Key verified values**

- Step 5 overall: **FAIL**. Only C2 direct passes the 0.60 recall bar:
  **0.955 [0.917, 0.976]** (191/200). C2 pooled is **0.522 [0.474, 0.571]**
  (209/400) and fails. C0 false-positive rate is **0.450 [0.383, 0.519]**
  (90/200), against a bar of ≤ 0.10.
- Step 6 C2 group: cost **+71.48 [−20.01, 168.63]** (point estimate +2.85%;
  interval includes zero); cycle CSL **+6.14 pp [4.20, 8.30]**.
- Step 6 C0 group: exactly **0/40** changed on both cost and CSL.
- Step 6 all 80 compared SKUs: cost **+35.74 [−9.32, 85.77]**; CSL
  **+3.07 pp [1.93, 4.38]**.

These are single-seed feasibility results. They do not establish cost savings,
generalization, real-world validity, or publication novelty.

---

## 1. Figure 1 — Step 5 attribution recall

**File:** `paper/figures/fig1_step5_attribution.png`

**Caption.** Step 5 per-cause attribution recall on the seed-42 synthetic panel
(200 SKUs per cause; C2 pooled denominator 400). Points are recall at the final
reading; whiskers are 95% Wilson intervals. The dashed line is the predeclared
0.60 recall bar for C1–C6. C2 direct (200 SKUs) is the only cause that clears the
bar (0.955 [0.917, 0.976]); C2 pooled (C2 + C2b) is 0.522 [0.474, 0.571] and
fails. C2b variant, C1, C3, C4, C5, and C6 fail; C7 is reported but not gated.
The C0 control false-positive rate is 0.450 [0.383, 0.519] against a ≤ 0.10 bar.
Overall Step 5 verdict: FAIL; E-8 unchanged.

**Definitions.**

- **Recall:** number of SKUs whose `attributed_cause` equals the cause under test,
  divided by the number of SKUs in that panel. For C2 pooled, the denominator is
  `truth_for_gate == "C2"` (C2 + C2b). For C2 direct, C2b, and C7, the denominator
  is the corresponding `true_cause` panel.
- **95% Wilson interval:** computed from the integer count and denominator with
  `z = 1.959963984540054`.
- **0.60 bar:** the predeclared C1–C6 recall bar. It is not applied to C7, which
  is reported but not gated. C2b is a variant panel.

**Provenance.**

- Counts: `results/step5/step5_readings.parquet`, fields `true_cause`,
  `truth_for_gate`, and `attributed_cause`.
- Stored gate values: `results/step5/step5_summary.json` (`5.1_recall`,
  `5.2_false_positive`).
- Cross-check: `results/step5/step5_confusion.parquet` and
  `docs/18_step5_findings.md`.
- CIs are computed from the counts because the summary does not store the
  per-cause intervals; the rounded values match `docs/18`.

**Verified values plotted.**

| Cause | N | Correct | Recall | 95% Wilson CI | Bar / status |
|---|---:|---:|---:|---|---|
| C1 | 200 | 6 | 0.030 | [0.014, 0.064] | 0.60 — FAIL |
| C2 pooled (C2 + C2b) | 400 | 209 | 0.5225 | [0.474, 0.571] | 0.60 — FAIL |
| C2 direct (C2 only) | 200 | 191 | 0.955 | [0.917, 0.976] | 0.60 — PASS |
| C2b variant | 200 | 18 | 0.090 | [0.058, 0.138] | 0.60 — FAIL (variant) |
| C3 | 200 | 11 | 0.055 | [0.031, 0.096] | 0.60 — FAIL |
| C4 | 200 | 0 | 0.000 | [0.000, 0.019] | 0.60 — FAIL |
| C5 | 200 | 22 | 0.110 | [0.074, 0.161] | 0.60 — FAIL |
| C6 | 200 | 29 | 0.145 | [0.103, 0.200] | 0.60 — FAIL |
| C7 reported, not gated | 200 | 56 | 0.280 | [0.222, 0.346] | not gated |

**C0 control (annotation):** 90/200 = **0.450 [0.383, 0.519]**, bar ≤ 0.10 —
**FAIL**. The 90 false positives are C2 29, C6 27, C5 15, C7 13, C1 5, C3 1.

---

## 2. Figure 2 — Step 6 paired inventory outcomes

**File:** `paper/figures/fig2_step6_paired_outcomes.png`

**Caption.** Step 6 paired inventory outcomes for the 80 seed-42 test SKUs
(40 C2 + 40 C0). Points are mean paired differences (gated − baseline); whiskers
are 95% percentile bootstrap intervals (10,000 resamples, seed 42). (a) Cost over
the 68-period evaluation window [132, 200); positive means higher cost.
(b) Cycle service level over 22 complete 3-period cycles; positive means higher
service, in percentage points. C2 (n=40) cost +71.48 [−20.01, 168.63], interval
includes zero; C2 CSL +6.14 pp [4.20, 8.30]. C0 (n=40) is exactly zero on both
outcomes. All (n=80) cost +35.74 [−9.32, 85.77]; CSL +3.07 pp [1.93, 4.38].

**Definitions.**

- **Cost:** per-SKU sum of `cost_period` over periods [132, 200), where
  `cost_period = H * on_hand_end + B * unmet`, `B/H = 19`. The plotted value is
  the mean over SKUs in the group.
- **Cycle CSL:** fraction of complete 3-period cycles with zero lost demand,
  averaged over SKUs. The 68-period window gives 22 complete cycles.
- **Paired difference:** gated minus baseline for the same SKU, then averaged.
- **95% interval:** percentile bootstrap over SKUs, 10,000 resamples, seed 42.
  The interval is stored in `c2_compare_summary.json`; it is not recomputed for
  the figure.

**Provenance.**

- `results/step6/c2_compare_summary.json`: `paired_C2`, `paired_C0`,
  `paired_all` means and `ci95`.
- `results/step6/c2_compare_paired.csv`: per-SKU `d_cost`, `d_csl`,
  `d_period_sl`; used to verify the means and direction counts.
- `results/step6/c2_compare_arms.csv`: arm-level `cost`, `csl`, `period_sl`.
- `results/step6/c2_gate_summary.json`: `window.eval_start = 132`,
  `eval_periods = 68`.
- `docs/19_step6_c2_gated_findings.md`.

**Verified values plotted.**

| Group | n | Baseline cost | Gated cost | Δ cost [95% CI] | Baseline CSL | Gated CSL | Δ CSL (pp) [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 40 | 2507.26 | 2578.73 | +71.48 [−20.01, 168.63] | 0.8011 | 0.8625 | +6.14 [4.20, 8.30] |
| C0 | 40 | 2621.96 | 2621.96 | 0.00 [0.00, 0.00] | 0.8818 | 0.8818 | 0.00 [0.00, 0.00] |
| All | 80 | 2564.61 | 2600.35 | +35.74 [−9.32, 85.77] | 0.8415 | 0.8722 | +3.07 [1.93, 4.38] |

C2 cost change is **+2.85%** (+71.48 on 2507.26); the 95% interval includes
zero. No cost saving is demonstrated. C2 cycle CSL rose on 27 SKUs, fell on 0,
and was unchanged on 13; cost rose on 21, fell on 16, and was unchanged on 3.
C0 was exactly 0/40 changed on both outcomes.

**Period service level (secondary; point estimates only).** C2 0.9246 → 0.9493
(+2.46 pp); C0 0.9566 → 0.9566 (0.00); All 0.9406 → 0.9529 (+1.23 pp). No
bootstrap interval for period SL is stored in the artifacts, so no interval is
plotted or reported in Table 3.

---

## 3. Table 1 — Step 5 gate results

**Files:** `paper/tables/table1_step5_gates.csv`,
`paper/tables/table1_step5_gates.tex`.

**Caption.** Step 5 attribution gates on the seed-42 synthetic panel. Panel A:
per-cause recall (correct cause attributed at the final reading) with 95% Wilson
intervals. The predeclared bar is 0.60 for C1–C6. C2 pooled is the gated
denominator (C2 + C2b); C2 direct is the C2 panel alone. Panel B: control
false-positive rate, row-normalised maximum off-diagonal confusion, and
priority-order sensitivity. C2b is a variant panel and is excluded from gate 5.3.
Overall verdict: FAIL; E-8 unchanged.

**Panel A (recall).**

| Cause | N | Correct | Recall | 95% Wilson CI | Verdict |
|---|---:|---:|---:|---|---|
| C1 | 200 | 6 | 0.030 | [0.014, 0.064] | FAIL |
| C2 pooled | 400 | 209 | 0.522 | [0.474, 0.571] | FAIL |
| C2 direct | 200 | 191 | 0.955 | [0.917, 0.976] | PASS |
| C2b variant | 200 | 18 | 0.090 | [0.058, 0.138] | FAIL (variant) |
| C3 | 200 | 11 | 0.055 | [0.031, 0.096] | FAIL |
| C4 | 200 | 0 | 0.000 | [0.000, 0.019] | FAIL |
| C5 | 200 | 22 | 0.110 | [0.074, 0.161] | FAIL |
| C6 | 200 | 29 | 0.145 | [0.103, 0.200] | FAIL |
| C7 reported, not gated | 200 | 56 | 0.280 | [0.222, 0.346] | not gated |

**Panel B (other gates).**

| Gate | Detail | n | Estimate | 95% CI | Verdict |
|---|---|---|---:|---|---|
| 5.2 | C0 false-positive rate (any non-C0 call) | 200 | 0.450 | [0.383, 0.519] | FAIL |
| 5.3 C0 | max off-diagonal to C2 | 200 | 0.145 | — | pass |
| 5.3 C1 | max off-diagonal to C0 | 200 | 0.545 | — | FAIL |
| 5.3 C2 | max off-diagonal to C0; true C2 only 0.040 to C5 | 400 | 0.335 | — | pass |
| 5.3 C3 | max off-diagonal to C0 | 200 | 0.360 | — | pass |
| 5.3 C4 | max off-diagonal to C0 | 200 | 0.560 | — | FAIL |
| 5.3 C5 | max off-diagonal to C6 | 200 | 0.230 | — | pass |
| 5.3 C6 | max off-diagonal to C0 | 200 | 0.480 | — | FAIL |
| 5.3 C7 | max off-diagonal to C0 | 200 | 0.390 | — | pass |
| 5.4 | max absolute macro-F1 difference, 10 permutations | — | 0.011 | — | PASS |
| Overall | gates 5.1–5.5 | — | — | — | **FAIL** |

**Provenance.**

- `results/step5/step5_readings.parquet`: all counts and row-normalised
  confusion; recomputed in `paper/make_results_pack.py`.
- `results/step5/step5_confusion.parquet`: cross-check of the C0–C7 confusion
  matrix (8×8; excludes C2b).
- `results/step5/step5_summary.json`: gate 5.2 value, gate 5.4 value, and the
  stored C1–C6 recalls.
- `results/step5/step5_priority_sensitivity.parquet` and `run_full.log`.
- `docs/18_step5_findings.md`.

**Definitions.** “Max off-diagonal” is the largest row-normalised entry
outside the diagonal for that truth row. For C2 pooled, the row is
`truth_for_gate == "C2"`; the true-C2-only value uses `true_cause == "C2"`.
Gate 5.3's bar is no off-diagonal entry > 0.40. Gate 5.5 (outputs produced) and
gate 5.6 (budget-restricted recall) do not change any verdict.

---

## 4. Table 2 — Step 6 gate performance

**Files:** `paper/tables/table2_step6_gate.csv`,
`paper/tables/table2_step6_gate.tex`.

**Caption.** Step 6 reliability-gate performance. Panel A: stored Step 5 readings
by split; “Recall” divides by all true C2 SKUs in the split, and “Coverage
(rows)” divides by all non-C2b rows. The validation split selected τ = 0.2446;
the test split was untouched. Panel B: live actions on the 80-SKU C2+C0
comparison. The comparison precision of 0.974 is not the all-cause test
precision of 0.822.

**Panel A (stored readings by split).**

| Split | Rows | Candidates | True C2 cand. | Acted | True C2 acted | Precision | Recall | Coverage (rows) | Coverage (cand.) | C0 action | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 960 | 246 | 115 | 147 | 109 | 0.741 | 0.908 | 0.153 | 0.598 | 0.025 | 0.947 |
| Validation | 320 | 75 | 39 | 41 | 39 | 0.951 | 0.975 | 0.128 | 0.547 | 0.000 | 0.995 |
| Test | 320 | 81 | 37 | 45 | 37 | 0.822 | 0.925 | 0.141 | 0.556 | 0.025 | 0.979 |

**Panel B (live actions on the 80-SKU C2+C0 comparison).**

| Metric | Value |
|---|---:|
| Acted | 38/80 (0.475) |
| True C2 acted | 37/40 (0.925) |
| True C0 acted | 1/40 (0.025) |
| C2-attributed rows | 44/80 (0.550) |
| Precision | 37/38 (0.974) |
| Coverage | 38/80 (0.475) |
| Decision match | 80/80 (1.000) |

The 44 C2-attributed rows are 37 true C2 + 7 true C0. Of these, 38 were acted
(37 true C2 + 1 true C0). Six low-reliability fallbacks were all true C0; the
14 `not_C2` fallbacks were 3 true C2 attributed C5 plus 11 true C0. The single
false action was `control:C0:0099` at t = 120; its γ = 1.0, so resetting the
scale to 1.0 was a no-op and its cost/CSL differences are exactly zero.

**Provenance.**

- `results/step6/c2_gate_summary.json`: `gate.tau`, `by_split` metrics,
  `window`, and candidate counts.
- `results/step6/c2_compare_arms.csv`: live actions, reasons, attributed cause,
  and decision match on the 80 SKUs.
- `results/step5/step5_readings.parquet`: candidate counts recomputed under the
  deterministic Step 6 split (train/val/test cycle 3/1/1 per true cause).
- `results/step6/c2_compare_summary.json`: acted count.
- `docs/19_step6_c2_gated_findings.md`.

**Definitions.** A candidate is a row with `detected == True` and
`attributed_cause == "C2"`. “Precision” is true C2 among acted rows. “Recall”
is true C2 acted divided by all true C2 SKUs in the split (120/40/40).
“Coverage (rows)” is acted divided by all non-C2b rows in the split (960/320/320).
“Coverage (candidates)” is acted divided by candidates. Panel B's 0.974 precision
is on the C2+C0 comparison population only; the all-cause test precision is 0.822.

---

## 5. Table 3 — Step 6 paired outcomes

**Files:** `paper/tables/table3_step6_outcomes.csv`,
`paper/tables/table3_step6_outcomes.tex`.

**Caption.** Step 6 paired inventory outcomes (seed 42; 40 C2 + 40 C0 test
SKUs). Differences are gated minus baseline over the 68-period window [132, 200);
cycle CSL uses 22 complete 3-period cycles. Cost CIs and CSL CIs are percentile
bootstrap intervals (10,000 resamples, seed 42). C0 is exactly zero on all 40
SKUs. Period service-level CIs were not stored and are not inferred.

**Panel A (cost).**

| Group | n | Baseline | Gated | Δ [95% CI] | Δ (%) |
|---|---:|---:|---:|---|---:|
| C2 | 40 | 2507.26 | 2578.73 | +71.48 [−20.01, 168.63] | +2.85 |
| C0 | 40 | 2621.96 | 2621.96 | 0.00 [0.00, 0.00] | 0.00 |
| All | 80 | 2564.61 | 2600.35 | +35.74 [−9.32, 85.77] | +1.39 |

**Panel B (service).**

| Group | n | Baseline CSL | Gated CSL | Δ CSL (pp) [95% CI] | Baseline period SL | Gated period SL | Δ period SL (pp) |
|---|---:|---:|---:|---|---:|---:|---:|
| C2 | 40 | 0.801 | 0.863 | +6.14 [4.20, 8.30] | 0.925 | 0.949 | +2.46 |
| C0 | 40 | 0.882 | 0.882 | 0.00 [0.00, 0.00] | 0.957 | 0.957 | 0.00 |
| All | 80 | 0.841 | 0.872 | +3.07 [1.93, 4.38] | 0.941 | 0.953 | +1.23 |

C2 cost direction: 21 up / 16 down / 3 equal. C2 cycle CSL: 27 up / 0 down /
13 equal. The C2 cost interval includes zero; the C2 CSL interval is entirely
above zero. C0 is 0/40 changed. The overall cost interval also includes zero.
No cost saving is demonstrated.

**Provenance.**

- `results/step6/c2_compare_summary.json`: means and 95% bootstrap CIs.
- `results/step6/c2_compare_paired.csv`: per-SKU differences and direction
  counts; used to verify means and the C0 zero result.
- `results/step6/c2_compare_arms.csv`: baseline/gated arm means for cost, CSL,
  and period SL.
- `results/step6/c2_gate_summary.json`: evaluation window (start 132, 68
  periods).
- `docs/19_step6_c2_gated_findings.md`.

**Definitions.** Cost is the per-SKU sum of `H * on_hand_end + B * unmet` over
[132, 200), with `B/H = 19`. Cycle CSL is the fraction of complete 3-period
cycles with zero lost demand. Period SL is the fraction of evaluated periods with
zero lost demand. Differences are gated minus baseline; CSL and period SL
differences are reported in percentage points. Intervals are percentile
bootstrap (10,000 resamples, seed 42).

---

## 6. Verification performed

The build script asserts the following before writing any output:

- Step 5 counts: C1 6/200, C2 pooled 209/400, C2 direct 191/200, C2b 18/200,
  C3 11/200, C4 0/200, C5 22/200, C6 29/200, C7 56/200.
- Step 5 C0 false positives: 90/200.
- Step 5 max off-diagonal values used in Table 1: C1 0.545, C2 0.335 (pooled;
  true C2 0.040), C4 0.560, C6 0.480.
- Step 5 gate 5.4 value: 0.011081653026416477.
- Step 6 split rows: 960 train, 320 validation, 320 test.
- Step 6 candidates: 246 train, 75 validation, 81 test; true C2 candidates
  115/39/37; acted 147/41/45; true C2 acted 109/39/37.
- Step 6 comparison: 38 acted, 37 true C2, 1 true C0, 44 C2-attributed rows,
  6 low-reliability fallbacks, 14 `not_C2` fallbacks, 80/80 decision match.
- Step 6 paired means: C2 cost +71.4782266121367, C2 CSL +0.06136363636363634;
  All cost +35.73911330606835, All CSL +0.03068181818181817; C0 exactly 0.
- Arm means reproduce the paired differences; C0 is exactly 0 on all 40 SKUs.

All checks passed when the pack was generated. No plotted value was estimated or
interpolated.

---

## 7. Unresolved discrepancies and gaps

These are source-artifact issues, documented rather than silently corrected.
No source file, result file, or frozen configuration was modified.

1. **`step5_summary.json` confusion keys are transposed/mislabeled.**
   `confusion` and `confusion_rownormalised` are stored in column-oriented order
   and do not match the row-normalised table in `docs/18`. The pack uses
   `step5_confusion.parquet` (rows = truth, columns = attributed) and recomputes
   the row-normalised values from `step5_readings.parquet`. The recomputed values
   match `docs/18`. This is an artifact-key issue, not a change to any result.
2. **`step5_confusion.parquet` excludes C2b.** It is 8×8 (C0–C7). C2b counts and
   the C2 pooled denominator are taken from `step5_readings.parquet`.
3. **C2 direct and C2b intervals are not stored in `step5_summary.json`.** The
   summary stores the pooled C2 recall (0.5225) and C2b recall (0.09), but not
   the direct-panel count or either interval. The pack computes them from the
   readings; the rounded values match `docs/18`.
4. **Period SL has no stored bootstrap interval.** Table 3 reports period SL
   point estimates only. No CI is inferred or invented.
5. **`c2_compare_summary.json` `elapsed_s = 10833.3` is a machine-suspend
   artifact.** It is not compute time and is not used anywhere in this pack.
6. **Two different precisions are intentional.** Test all-cause precision is
   0.822 (45 acted, 37 true C2, 8 non-C2: 1 true C0 + 7 C1–C7). The C2+C0
   comparison precision is 0.974 (38 acted, 37 true C2, 1 true C0). The difference
   is the seven false actions on C1–C7 SKUs outside the inventory comparison. The
   pack labels both and does not conflate them.
7. **Single seed and small sample.** All Step 6 values are seed 42 with 40 C2 +
   40 C0 test SKUs. No generalization claim is made. The 95% CSL target is not
   reached (gated C2 CSL 0.8625; C0 0.8818).
8. **Tests were not re-run for this pack.** `docs/19` records 186 passing tests
   in 63.57 s. The results pack reads artifacts only and does not depend on a
   new test run.

No numerical discrepancy was found between the plotted/table values and their
source artifacts. The only discrepancies are the stored-label/orientation issues
above.

---

## 8. Reproduction

```
python paper/make_results_pack.py
```

The script requires the existing environment used for the project:
Python 3.13.5, `numpy` 2.5.3, `pandas` 2.3.3, `matplotlib` 3.11.2, and
`pyarrow` for Parquet input. It reads only `results/step5/` and
`results/step6/`; it writes only `paper/figures/` and `paper/tables/`.

**Files created**

```
paper/figures/fig1_step5_attribution.png
paper/figures/fig2_step6_paired_outcomes.png
paper/tables/table1_step5_gates.csv
paper/tables/table1_step5_gates.tex
paper/tables/table2_step6_gate.csv
paper/tables/table2_step6_gate.tex
paper/tables/table3_step6_outcomes.csv
paper/tables/table3_step6_outcomes.tex
paper/RESULTS_PACK.md
paper/make_results_pack.py
```
