# CAFR Diagnostic Report: C0 False Positives and Attribution Errors

**Date:** 2026-10-22
**Scope:** Training split only ($i \pmod 5 < 3$, $N = 120$ C0 controls)
**Status:** Read-only exploratory diagnostic
**Branch:** `cafr/c6-generator-redesign`
**Output Artifacts:**
- [`c0_fp_case_level.csv`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/c0_fp_diagnostic/c0_fp_case_level.csv)
- [`c0_fp_case_level.parquet`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/c0_fp_diagnostic/c0_fp_case_level.parquet)
- [`c0_fp_summary.json`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/c0_fp_diagnostic/c0_fp_summary.json)

---

## 1. Executive Summary & Baseline Reproducibility

An independent, training-only diagnostic of all 120 normal C0 control SKUs was conducted to identify the exact statistical and architectural mechanisms responsible for false-positive attributions.

### Baseline Verification Audit
The diagnostic reproduces all previously reported baseline control metrics with 100% precision:
- **Overall C0 False-Positive Attribution:** Exactly **38 / 120 (31.7%)** (82 / 120 correctly attributed to C0). Matches the C3–C6 overlap diagnostic V0 control.
- **C0 Attributed to C6:** Exactly **14 / 120 (11.7%)**. Matches the C6 diagnostic report.
- **Raw C6 Rule Firing:** Exactly **42 / 120 (35.0%)**. Matches the earlier C6 diagnostic.

```
+-----------------------------------------------------------------------------+
|                      C0 ATTRIBUTION OUTCOME (N = 120)                       |
+------------------------------------+----------------------------------------+
| Correctly Unflagged (C0):          | 82 / 120 (68.3%)                       |
| False Positives (Non-C0):          | 38 / 120 (31.7%)                       |
+------------------------------------+----------------------------------------+
|   • Attributed to C6 (Intermittency):| 14 / 120 (11.7%)  [36.8% of all FPs]   |
|   • Attributed to C7 (Policy Gap):  |  8 / 120  (6.7%)  [21.1% of all FPs]   |
|   • Attributed to C2 (Coverage Gap):|  7 / 120  (5.8%)  [18.4% of all FPs]   |
|   • Attributed to C1 (Bias):        |  5 / 120  (4.2%)  [13.2% of all FPs]   |
|   • Attributed to C5 (CUSUM):       |  2 / 120  (1.7%)  [ 5.3% of all FPs]   |
|   • Attributed to C4 (Size Var):    |  1 / 120  (0.8%)  [ 2.6% of all FPs]   |
|   • Attributed to C3 (Occur Gap):   |  1 / 120  (0.8%)  [ 2.6% of all FPs]   |
+------------------------------------+----------------------------------------+
```

---

## 2. Distinction Between Raw Rule Firings and Final Attributions

A critical architectural finding is the large discrepancy between **raw rule firings** across the time series and **final attributed causes** at first detection:

| Cause | Raw Rule Firings Across Series (Denominator = 120) | Final Attributed False Positives (Denominator = 120) | Conversion Ratio (Attributed / Fired) | Primary Preemption / Suppression Cause |
| :---: | :---: | :---: | :---: | :--- |
| **C1** | 17 / 120 (14.2%) | 5 / 120 (4.2%) | 29.4% | Pre-120 flag or transient decay |
| **C2** | 18 / 120 (15.0%) | 7 / 120 (5.8%) | 38.9% | Pre-120 flag or transient decay |
| **C3** | **40 / 120 (33.3%)** | **1 / 120 (0.8%)** | **2.5%** | Pre-120 firings (20 SKUs) or superseded by C6/C2/C1 |
| **C4** | 2 / 120 (1.7%) | 1 / 120 (0.8%) | 50.0% | Normal dispersion stability |
| **C5** | 8 / 120 (6.7%) | 2 / 120 (1.7%) | 25.0% | Calibrated $h = 10.64$ suppresses 6 cases |
| **C6** | **42 / 120 (35.0%)** | **14 / 120 (11.7%)** | 33.3% | Preempted by C2/C1/C7 or decayed ($\text{conf} < 0.35$) |
| **C7** | 24 / 120 (20.0%) | 8 / 120 (6.7%) | 33.3% | Lower priority in pipeline (`C7` is evaluated last) |

### Key Insights:
1. **The C3 Fire-and-Forget Paradox:** C3's rule fires on 33.3% of C0 controls across the full 200 periods, but yields only 1 final false positive (0.8%). This occurs because 20 firings happen pre-120 during initial forecaster convergence, and at $t \ge 120$, C3 firings are either preempted by higher priority or decay before latching.
2. **C6 High Conversion:** C6 has 42 raw firings and yields 14 final false positives. Once C6 triggers, its confidence score $\text{conf} = \min(1.0, \text{stability}/0.90 - 0.5 \times n_{\text{fired}})$ routinely exceeds $\tau_{\text{conf}} = 0.35$ because bootstrap stability on stationary series is high ($\sim 0.93$).

---

## 3. Case-Level Analysis of False-Positive Mechanisms

Every false positive was traced to its individual time series, parameters, and triggering metrics in [`c0_fp_case_level.csv`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/c0_fp_diagnostic/c0_fp_case_level.csv).

### Mechanism A: C6 False Positives (14 SKUs) — The Initialization Deficit & Boundary Noise
- **Root Cause 1: 24-Period Initialization Deficit (10 / 14 SKUs):**
  In [`reading.py`](file:///c:/Users/DELL/Minor_Project/CAFR/cafr/monitor/reading.py#L420-L422), `init_cell` is evaluated on `y[:24]` (burn-in). On intermittent demand with $p \approx 0.30$, a 24-period window contains only $\approx 7$ nonzero observations. The sample variance and sample inter-arrival times over 7 observations have severe finite-sample error, frequently misclassifying the initial cell relative to the true stationary DGP:
  - `control:C0:0001`: True cell is `adi_lo_cv2_hi`, but burn-in misclassified as `adi_hi_cv2_lo`.
  - `control:C0:0017`: True cell is `adi_lo_cv2_lo`, but burn-in misclassified as `adi_hi_cv2_lo`.
  - `control:C0:0027`: True cell is `adi_hi_cv2_lo`, but burn-in misclassified as `adi_lo_cv2_lo`.
  When the monitor later observes 30-period rolling windows, the SKU naturally settles into its true equilibrium cell. Because the true cell differs from the corrupted `init_cell`, `c6_changed` triggers for 3 consecutive windows.
- **Root Cause 2: Boundary Proximity & Sample Variance (4 / 14 SKUs):**
  For `control:C0:0010`, `0075`, `0092`, `0110`, the true dispersion parameter is $k \in [1.34, 1.83]$, meaning theoretical $CV^2 = 1/k \in [0.55, 0.74]$—directly adjacent to the quadrant boundary $CV^2 = 0.49$. In 30-period windows ($\sim 10$ nonzeros), sample $CV^2$ dipped below $0.49$ by chance ($0.13\dots0.32$), shifting the cell from `adi_lo_cv2_hi` to `adi_lo_cv2_lo`.
- **The Stability Illusion:** In all 14 cases, `c6_stability` was $\ge 0.90$ (median **0.9325**). Because the series is stationary in its true cell, bootstrap resamples over that cell are highly consistent, producing high stability on a completely unshifted series.

### Mechanism B: C7 False Positives (8 SKUs) — Intermittency-Induced Stockout Clustering
- **Root Cause:**
  All 8 SKUs have `c7_csl_gap` between $-0.117$ and $-0.150$ ($\tau_{\text{svc}} = 0.10$), with $\ge 4$ unmet replenishment cycles (mean 4.5).
  - 6 of the 8 SKUs have low occurrence probability ($p \le 0.20$, mean empirical ADI = 5.67).
  - Mean longest zero-demand run is **18.6 consecutive periods**.
  - Under base-stock inventory control on slow-moving parts, demand clustering following extended zero periods temporarily depletes stock. Over a rolling 60-period window ($W_{\text{svc}} = 60$), 4 or 5 stockout periods drive empirical CSL below 85%.
  - Because demand size is stationary ($|z_b| < 1.96, r \le 1.5$), C7's forecast health guard passes, falsely attributing normal inventory stockout noise to a faulty inventory policy.

### Mechanism C: C2 False Positives (7 SKUs) — Short-Window Conformal Undercoverage
- **Root Cause:**
  All 7 SKUs have `c2_c_gap = -0.1333` (or `-0.1667`) with binomial $p = 0.0258$ ($< 0.05$).
  - Over a rolling calibration window $W_{\text{cal}} = 30$, nominal 90% coverage expects 3 uncovered periods.
  - Due to random clustering, observing 7 uncovered periods ($23/30 = 76.7\%$ coverage) drops the gap to $-13.3\% < -10.0\%$, immediately passing the binomial significance test ($p < 0.05$).
  - **Transience:** 4 of the 7 cases were single-check or 2-check transient blips that vanished immediately.

### Mechanism D: C1 False Positives (5 SKUs) — Single-Period Small-Sample Spikes
- **Root Cause:**
  - **100% (5 / 5 SKUs) of C1 false positives fired at exactly ONE monitoring checkpoint** and never fired again.
  - The number of usable nonzero sizes was at or near the absolute floor: $n_{\text{usable\_sz}} \in [8, 13]$ (mean 10.2).
  - With only 8–10 observations, a single large demand size produces an instantaneous standardized error spike ($|z_b| \in [2.11, 6.32]$).
  - Because CAFR latches on the very first period where any rule fires, this single-check transient spike permanently attributes C1 to a normal SKU.

---

## 4. Triggering Mechanisms & Temporal Persistence

```
+-----------------------------------------------------------------------------+
|                     FALSE POSITIVE TRIGGER MECHANICS                        |
+------------------------------------+----------------------------------------+
| Total False Positives:             | 38 SKUs                                |
| Simultaneous Rules at Detection:   |  1 SKU   (control:C0:0192: C5 & C3)    |
| Priority Preemption at Detection:  |  1 SKU   (C5 preempted C3)             |
| Terminal Fallback ($t=199$):       |  0 SKUs  (All latched at t <= 192)     |
| Persistent Firings (>= 3 checks):  | 17 SKUs  (44.7%)                        |
| Transient Firings (1 to 2 checks): | 21 SKUs  (55.3%)                        |
+------------------------------------+----------------------------------------+
```

**Critical Finding on Transience:**
More than half (**55.3%**, 21 / 38) of all C0 false positives are **transient 1-check or 2-check blips**. The current architecture's "latch-on-first-fire" policy makes CAFR maximally vulnerable to finite-sample sampling fluctuations.

---

## 5. Ranked Hypotheses for the Next Experiment (Design Only)

We rank three prospective hypotheses to reduce C0 false positives without damaging true fault detection. **None of these are implemented yet.**

### Rank 1: Multi-Checkpoint Confirmation (Dwell Gate) Before Latching
- **Hypothesis:** Requiring that an attribution decision must be confirmed across at least 2 consecutive checkpoints (or an 8-period dwell window) before latching permanently will eliminate single-check transient spikes.
- **Evidence For:**
  - 21 of 38 false positives (55.3%) fired at only 1 or 2 checkpoints.
  - 100% of C1 false positives (5/5) and 57% of C2 false positives (4/7) fired at only 1 or 2 checkpoints.
  - In the C3–C6 overlap trial, deferring latching by 1–3 checks produced zero increase in C0 false positives.
- **Evidence Against / Risks:**
  - Increases detection latency by 1 check (4 periods).
  - Could reduce recall on genuine faults that are transient.
- **Minimal Training-Only Experiment:**
  - Replay the saved 15,840 checkpoint trajectories on training C0, C3, and C6 series requiring a 2-consecutive-check confirmation for C1, C2, and C7.
  - Monitored Metrics: C0 FP reduction (target: $-10$ to $-14$ FPs), C1/C2/C7 recall on training split, detection latency.
- **Stop Condition:** Stop if C0 FPs do not drop by at least 8 SKUs, or if genuine C1/C2/C7 recall drops by $> 2$ SKUs.

### Rank 2: Empirical Burn-In Initialization for C6 Cell Classification
- **Hypothesis:** Initializing C6's `init_cell` from the global pooled burn-in model (which aggregates observations across all SKUs) or requiring a minimum of 15 nonzero observations before locking `init_cell` will eliminate the 24-period small-sample initialization deficit.
- **Evidence For:**
  - C6 is the largest contributor to false positives (14 / 38 = 36.8%).
  - 10 of 14 C6 false positives were verified to be caused by 24-period small-sample initialization misclassifying the true stationary DGP.
- **Evidence Against / Risks:**
  - Shrinking initial cells toward global pool parameters could bias highly idiosyncratic SKUs.
- **Minimal Training-Only Experiment:**
  - Test initializing C6 using `adi_init` and `cv2_init` from static generator parameters or pooled burn-in estimates on training C0 and C6 series.
  - Monitored Metrics: C0 $\to$ C6 false positives (target: reduce from 14 to $\le 4$), candidate C6 recall.
- **Stop Condition:** Stop if candidate C6 recall drops by $> 3$ SKUs, or if C0 C6 false positives do not decrease by at least 7 SKUs.

### Rank 3: Intermittency-Scaled Threshold for C7 Stockout Gap
- **Hypothesis:** Requiring an intermittency-aware minimum exposure floor (e.g. minimum 10 positive demand events in $W_{\text{svc}}$) before evaluating C7's CSL gap will prevent slow-moving parts from falsely attributing cluster stockouts to policy failure.
- **Evidence For:**
  - C7 accounts for 8 of 38 false positives (21.1%).
  - 6 of the 8 SKUs have $p \le 0.20$ and extended zero stretches (mean 18.6 periods) where normal replenishment dynamics hit zero on-hand stock.
- **Evidence Against / Risks:**
  - Genuine inventory policy errors on slow-moving items might take longer to flag.
- **Minimal Training-Only Experiment:**
  - Test adding a minimum exposure requirement ($p \cdot W_{\text{svc}} \ge 10$) to C7's evidence floor on training C0 and C7 series.
  - Monitored Metrics: C0 $\to$ C7 false positives (target: reduce from 8 to $\le 3$), C7 recall.
- **Stop Condition:** Stop if true C7 recall drops by $> 2$ SKUs, or if C0 C7 false positives do not drop by at least 4 SKUs.

---

## 6. Single Recommendation for Next Step

**Recommended Action:** Execute **Hypothesis 1 (Multi-Checkpoint Confirmation Gate)** on the training split first.

**Scientific Justification:**
1. It directly addresses **55.3% of all observed C0 false positives** (21 SKUs) across C1, C2, and C7 without requiring changes to underlying statistical estimators or re-tuning thresholds.
2. It can be evaluated completely offline using the already-recorded checkpoint trajectories ([`c3_c6_checkpoint_trajectories.parquet`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/c6_redesign/c3_c6_checkpoint_trajectories.parquet)), requiring zero new simulation runs.
3. It preserves all mathematical definitions of C1, C2, C6, and C7 rules while testing a principled temporal persistence filter.

---

## 7. Protected-File and Artifact Audit

- **Production Integrity:** [`cafr/attribution/priority.py`](file:///c:/Users/DELL/Minor_Project/CAFR/cafr/attribution/priority.py), [`rules.py`](file:///c:/Users/DELL/Minor_Project/CAFR/cafr/attribution/rules.py), [`pipeline.py`](file:///c:/Users/DELL/Minor_Project/CAFR/cafr/attribution/pipeline.py), and `synthetic.yaml` have **0 uncommitted changes**.
- **Official Benchmarks:** [`results/step5/`](file:///c:/Users/DELL/Minor_Project/CAFR/results/step5/) and [`results/step6/`](file:///c:/Users/DELL/Minor_Project/CAFR/results/step6/) are **100% untouched**.
- **Working Tree Audit:**
  - Untracked exploratory files:
    - `experiments/diagnostics/c0_fp_diagnostic.py`
    - `results/exploratory/c0_fp_diagnostic/c0_fp_case_level.parquet`
    - `results/exploratory/c0_fp_diagnostic/c0_fp_summary.json`
    - `results/exploratory/c0_fp_diagnostic/c0_fp_diagnostic_report.md`
  - Gitignored files (by `.gitignore`):
    - `results/exploratory/c0_fp_diagnostic/c0_fp_case_level.csv`
- **Quarantine Compliance:** Validation ($i \pmod 5 == 3$) and test ($i \pmod 5 == 4$) splits were never loaded or analyzed.

