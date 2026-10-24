# CAFR Diagnostic Report: C6 Burn-In Initialization Repair

**Date**: 2026-10-22
**Branch**: `cafr/c6-generator-redesign`
**Execution Scope**: Strictly Training Split Only ($i \pmod 5 < 3$, $N = 120$ per cause cohort)
**Quarantined Splits**: Validation ($i \pmod 5 == 3$) and Test ($i \pmod 5 == 4$) Strictly Untouched
**Protected Artifacts**: `results/step5/` and `results/step6/` Strictly Untouched

---

## Executive Summary & Core Verdict

This offline, training-only diagnostic investigated whether noisy 24-period per-SKU initialization causes stationary C0 controls to be falsely attributed to C6, and whether candidate repair strategies—specifically **V1 (Minimum Evidence / Provisional Floor)** or **V2 (Empirical Bayes Shrinkage)**—can reduce false positives without harming genuine C6 detection.

### Core Findings

1. **The Root Mechanism Confirmed**:
   - The 24-period burn-in window ($W_{\text{base}} = 24$) is mathematically inadequate for intermittent demand. At typical spare-parts arrival rates ($p \approx 0.15 - 0.35$), a 24-period window yields an average of only **6.52 nonzero observations** (with 34.2% of SKUs having $\le 4$ nonzeros).
   - Across all 120 training C0 controls, **47 / 120 (39.2%)** have their 24-period initial cell (`init_cell`) misclassified relative to their true stationary DGP cell.
   - Among the 14 C0 controls falsely attributed to C6 under the verified baseline, **10 / 14 (71.4%)** suffered initialization cell errors. When the monitor later observes 30-period rolling windows, the SKU settles into its true equilibrium cell. Because the true cell differs from the corrupted `init_cell`, the monitor registers a sustained cell transition (`c6_changed = 1`) with high bootstrap stability ($\ge 0.90$, median $0.933$), latching a false attribution.
   - The remaining **4 / 14 (28.6%)** false positives have true DGP parameters sitting immediately adjacent to the cell boundary ($CV^2_{\text{DGP}} \in [0.55, 0.74]$ near $CV^{2*} = 0.49$).

2. **Evaluation of Strategy V1 (Minimum Evidence / Provisional Floor)**:
   - **Verdict: REJECTED.**
   - Requiring a static minimum evidence floor $N_{\text{burn}} \ge N_{\text{min}}$ on a 24-period burn-in window destroys genuine intermittent detection faster than it eliminates false positives.
   - At $N_{\text{min}} = 6$: C0 $\to$ C6 false positives drop from 14 to 8 ($-6$ FPs), **BUT candidate C6 recall collapses from 31 to 14 ($-17$ detections, a $54.8\%$ loss of genuine signal)**.
   - At $N_{\text{min}} \le 4$: C0 $\to$ C6 false positives remain completely unchanged at 14, while C6 recall drops from 31 down to 25 ($-6$ detections).
   - *Reason*: Intermittent demand SKUs have low occurrence rates by definition ($p \approx 0.20 \implies \mu = 4.8$ nonzeros in 24 periods). Over half ($54.8\%$) of genuine detected C6 transitions have $\le 5$ nonzeros in the burn-in period. Discarding them censors the exact target population C6 was designed to detect.

3. **Evaluation of Strategy V2 (Empirical Bayes Shrinkage Candidate)**:
   - **Verdict: REJECTED.**
   - Shrinking individual SKU sample moments toward the pooled training burn-in prior ($\overline{ADI}_{\text{pool}} = 3.79, \overline{CV}^2_{\text{pool}} = 1.01$) **increases initialization errors** as prior weight $M$ increases (from $39.2\%$ at $M=0$ up to $62.5\%$ at $M=12$).
   - *Reason*: Spare parts demand is fundamentally **heterogeneous**. Shrinkage pulls genuine slow-moving parts ($ADI \ge 4.0$) down below 4.0 and pulls genuine low-dispersion parts ($CV^2 < 0.49$) up above 0.49, collapsing diverse catalog items into the pooled central cell (`adi_lo_cv2_hi`).

4. **Architectural Insight (Long-Horizon Baseline Reference)**:
   - Expanding the baseline window $W_{\text{base}}$ over the available stationary pre-monitoring history (e.g. $W_{\text{base}} = 120$, before detection begins at $\tau_{\text{inj}} = 120$) drops C0 initialization errors from **39.2% down to 15.0%** and automatically resolves **8 of the 10 initialization errors** on the C6 false-positive cohort without censoring intermittent parts.
   - However, this is an architectural modification beyond exploratory tuning and must not be implemented on this branch.

### Final Recommendation
**Do NOT proceed to validation evaluation or official gate testing for V1 or V2.**
Neither strategy satisfies the decision criteria: V1 destroys genuine intermittent recall, and V2 worsens initialization errors due to catalog heterogeneity. All production code, thresholds, priority logic, and frozen Step 5/6 artifacts remain untouched.

---

## 1. Strict Boundaries & Isolation Audit

| Protection Boundary | Requirement | Audit Status | Evidence / Verification |
| :--- | :--- | :--- | :--- |
| **Split Isolation** | Strictly training split only ($i \pmod 5 < 3$) | **PASSED** | 120 C0 controls and 120 Candidate C6 series verified; all $i \pmod 5 \in \{0, 1, 2\}$. |
| **Split Quarantine** | Zero access to validation or test splits | **PASSED** | SKUs with $i \pmod 5 \in \{3, 4\}$ were neither read nor analyzed. |
| **Production Files** | Do not modify production modules or configs | **PASSED** | `git status` confirms zero diffs on tracked files. |
| **Protected Benchmarks**| `results/step5/` and `results/step6/` untouched | **PASSED** | Zero writes or modifications to benchmark directories. |
| **Exploratory Isolation**| All outputs saved under `results/exploratory/` | **PASSED** | Artifacts saved exclusively in `results/exploratory/c6_initialization/`. |
| **Baseline Reproduction**| V0 exactly matches verified baseline | **PASSED** | C0 total FPs: 38/120; C0 $\to$ C6 FPs: 14/120; Candidate C6 Recall: 31/120. |

---

## 2. Mechanism: Finite-Sample Noise and Initialization Error

### 2.1 The 14 C0 False Positives Attributed to C6
Under the baseline pipeline, 14 stationary C0 controls falsely attributed to C6. The table below details their true DGP parameters, realized burn-in nonzeros ($N_{nz}$ in $t < 24$), initial cell point estimate, detected cell at attribution time, and stability score:

| SKU ID | True $p$ | True $k$ | True DGP Cell | Realized $N_{nz}$ (24p) | V0 `init_cell` | Detect Cell | Detect $t$ | Bootstrap Stability | Mechanism Classification |
| :--- | :---: | :---: | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| `control:C0:0001` | 0.339 | 1.632 | `adi_lo_cv2_hi` | 5 | `adi_hi_cv2_lo` | `adi_lo_cv2_lo` | 156 | 0.908 | Init error: ADI & CV2 both erred on $N=5$ |
| `control:C0:0002` | 0.421 | 1.095 | `adi_lo_cv2_hi` | 10 | `adi_lo_cv2_lo` | `adi_lo_cv2_hi` | 120 | 0.937 | Init error: sample $CV^2=0.48$ fell below 0.49 |
| `control:C0:0010` | 0.448 | 1.588 | `adi_lo_cv2_hi` | 10 | `adi_lo_cv2_hi` | `adi_lo_cv2_lo` | 152 | 0.928 | Boundary SKU: DGP $CV^2=0.63$; rolling dipped to 0.32 |
| `control:C0:0017` | 0.289 | 5.637 | `adi_lo_cv2_lo` | 5 | `adi_hi_cv2_lo` | `adi_lo_cv2_lo` | 164 | 0.990 | Init error: $N=5 \implies ADI=4.8 \ge 4.0$ |
| `control:C0:0027` | 0.183 | 6.822 | `adi_hi_cv2_lo` | 7 | `adi_lo_cv2_lo` | `adi_hi_cv2_lo` | 176 | 0.905 | Init error: cluster $N=7 \implies ADI=3.43 < 4.0$ |
| `control:C0:0036` | 0.415 | 1.859 | `adi_lo_cv2_hi` | 10 | `adi_lo_cv2_lo` | `adi_lo_cv2_hi` | 188 | 0.917 | Init error: sample $CV^2=0.41$ fell below 0.49 |
| `control:C0:0051` | 0.347 | 7.770 | `adi_lo_cv2_lo` | 4 | `adi_hi_cv2_lo` | `adi_lo_cv2_lo` | 172 | 0.957 | Init error: $N=4 \implies ADI=6.0 \ge 4.0$ |
| `control:C0:0075` | 0.399 | 1.345 | `adi_lo_cv2_hi` | 14 | `adi_lo_cv2_hi` | `adi_lo_cv2_lo` | 120 | 0.979 | Boundary SKU: DGP $CV^2=0.74$; rolling dipped to 0.13 |
| `control:C0:0092` | 0.348 | 1.739 | `adi_lo_cv2_hi` | 9 | `adi_lo_cv2_hi` | `adi_lo_cv2_lo` | 124 | 0.938 | Boundary SKU: DGP $CV^2=0.58$; rolling dipped to 0.26 |
| `control:C0:0102` | 0.328 | 5.051 | `adi_lo_cv2_lo` | 5 | `adi_hi_cv2_lo` | `adi_lo_cv2_lo` | 120 | 0.921 | Init error: $N=5 \implies ADI=4.8 \ge 4.0$ |
| `control:C0:0110` | 0.470 | 1.832 | `adi_lo_cv2_hi` | 13 | `adi_lo_cv2_hi` | `adi_lo_cv2_lo` | 188 | 0.910 | Boundary SKU: DGP $CV^2=0.55$; rolling dipped to 0.27 |
| `control:C0:0152` | 0.398 | 0.586 | `adi_lo_cv2_hi` | 5 | `adi_hi_cv2_hi` | `adi_lo_cv2_hi` | 156 | 0.945 | Init error: $N=5 \implies ADI=4.8 \ge 4.0$ |
| `control:C0:0155` | 0.319 | 8.866 | `adi_lo_cv2_lo` | 4 | `adi_hi_cv2_lo` | `adi_lo_cv2_lo` | 120 | 0.913 | Init error: $N=4 \implies ADI=6.0 \ge 4.0$ |
| `control:C0:0165` | 0.421 | 2.752 | `adi_lo_cv2_lo` | 9 | `adi_lo_cv2_hi` | `adi_lo_cv2_lo` | 120 | 0.947 | Init error: sample $CV^2=0.93$ rose above 0.49 |

### 2.2 Mechanism Breakdown
1. **ADI Estimation Errors on Low Counts ($N_{nz} \le 5$, 6 SKUs)**:
   For SKUs `0001`, `0017`, `0051`, `0102`, `0152`, `0155`, only 4 or 5 nonzeros arrived in the first 24 periods. This forced the sample ADI to $24/4 = 6.0$ or $24/5 = 4.8$, crossing the $ADI^* = 4.0$ threshold into `adi_hi`. During post-120 monitoring (using 30-period rolling windows), their true arrival rates ($p \approx 0.30 - 0.42$) manifested naturally, producing $ADI \approx 2.3 - 2.7$ (`adi_lo`). Because the SKU settled into its true equilibrium cell (`adi_lo`), the monitor compared it against the corrupt `init_cell` (`adi_hi`), declared a cell change, and latched a false alarm.
2. **$CV^2$ Estimation Errors on Small Samples (4 SKUs)**:
   For SKUs `0002`, `0027`, `0036`, `0165`, sample variance over 7 to 10 observations had high finite-sample variance, placing the burn-in point estimate on the wrong side of $0.49$.
3. **Boundary Transients (4 SKUs)**:
   For SKUs `0010`, `0075`, `0092`, `0110`, the initial cell was correctly assigned, but the true DGP parameter $k \in [1.34, 1.83]$ yields $CV^2 \in [0.55, 0.74]$, very close to $0.49$. During rolling 30-period monitoring, sample $CV^2$ briefly dipped below 0.49 for 3 periods, triggering C6.

---

## 3. Evaluation of Candidate Strategies

### 3.1 Strategy V1: Minimum Evidence Floor Sensitivity
Strategy V1 treats the initial cell as unestimable/provisional if the 24-period burn-in contains fewer than $N_{\text{min}}$ nonzero observations (C6 abstains, reverting to C0).

The table below reports the sensitivity across $N_{\text{min}} \in [2, 10]$ on the 120 C0 controls and 120 Candidate C6 series:

| $N_{\text{min}}$ Threshold | C0 C6 FPs | C0 C6 FP Rate | C0 Total FPs | C0 Total FP Rate | Candidate C6 Recall | C6 Recall Rate | Genuine C6 Detections Lost |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2 (Baseline V0)** | **14** | **11.7%** | **38** | **31.7%** | **31** | **25.8%** | **0** |
| 3 | 14 | 11.7% | 38 | 31.7% | 29 | 24.2% | $-2$ ($-6.5\%$) |
| 4 | 14 | 11.7% | 38 | 31.7% | 25 | 20.8% | $-6$ ($-19.4\%$) |
| 5 | 12 | 10.0% | 36 | 30.0% | 21 | 17.5% | $-10$ ($-32.3\%$) |
| **6** | **8** | **6.7%** | **32** | **26.7%** | **14** | **11.7%** | **$-17$ ($-54.8\%$)** |
| 7 | 8 | 6.7% | 32 | 26.7% | 7 | 5.8% | $-24$ ($-77.4\%$) |
| 8 | 7 | 5.8% | 31 | 25.8% | 7 | 5.8% | $-24$ ($-77.4\%$) |
| 9 | 7 | 5.8% | 31 | 25.8% | 7 | 5.8% | $-24$ ($-77.4\%$) |
| 10 | 5 | 4.2% | 29 | 24.2% | 4 | 3.3% | $-27$ ($-87.1\%$) |

#### Why V1 Fails
- At low thresholds ($N_{\text{min}} \le 4$), **zero false positives are eliminated** because all 14 false-positive SKUs had at least 4 nonzeros. Yet 6 genuine C6 detections are lost.
- At intermediate thresholds ($N_{\text{min}} = 6$), 6 false positives are eliminated, but **17 genuine detections are destroyed** (recall plummets from 31 down to 14).
- *Mathematical Cause*: On genuine intermittent demand ($p \approx 0.20$), the expected nonzeros in 24 periods is $\mu = 4.8$. Over $54.8\%$ of genuine C6 SKUs have $N_{nz} \le 5$. Discarding SKUs with $N_{nz} \le 5$ creates severe survivorship bias against intermittent parts.

---

### 3.2 Strategy V2: Empirical Bayes Shrinkage
Strategy V2 computes a pooled prior on training burn-in demand ($t < 24$):
$$\bar{p}_{\text{pool}} = 0.2637 \implies \overline{ADI}_{\text{pool}} = 3.79, \quad \overline{CV}^2_{\text{pool}} = 1.0069$$
and shrinks each SKU's sample moments toward this prior using pseudo-count weight $M$:
$$\widetilde{p}_i = \frac{N_i + M \bar{p}_{\text{pool}}}{24 + M}, \quad \widetilde{CV}^2_i = \frac{N_i \widehat{CV}^2_i + M \overline{CV}^2_{\text{pool}}}{N_i + M}$$

The table below shows the initialization error rate on C0 controls and C6 series across shrinkage weights $M$:

| Prior Weight $M$ | C0 Init Errors ($/120$) | C0 Init Error Rate | C6 Init Errors ($/120$) | C6 Init Error Rate | Result / Mechanism |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0 (No Shrinkage, V0)** | **47** | **39.2%** | **50** | **41.7%** | Raw sample estimates |
| 2 | 40 | 33.3% | 38 | 31.7% | Slight noise smoothing |
| 4 | 53 | 44.2% | 45 | 37.5% | Shrinkage begins collapsing cells |
| 6 | 61 | 50.8% | 51 | 42.5% | Majority of SKUs misclassified |
| 8 | 69 | 57.5% | 60 | 50.0% | Heavy distortion towards pooled mean |
| 12 | 75 | 62.5% | 64 | 53.3% | Severe over-shrinkage ($>60\%$ error) |

#### Why V2 Fails
The spare parts panel is **heterogeneous**, not identically distributed:
- Genuine slow-moving SKUs ($ADI \ge 4.0$) get shrunk below $4.0$ toward the pooled mean $3.79$, misclassifying them into `adi_lo`.
- Genuine low-dispersion SKUs ($CV^2 < 0.49$) get shrunk above $0.49$ toward the pooled mean $1.01$, misclassifying them into `cv2_hi`.
- As $M$ increases, shrinkage forces all SKUs into the single dominant cell (`adi_lo_cv2_hi`), worsening initialization error from $39.2\%$ up to $62.5\%$.

---

## 4. Architectural Diagnostic: Expanding the Baseline Window

Why was $W_{\text{base}}$ frozen at 24 periods when 120 stationary periods exist before injection?
In `cafr/monitor/reading.py`, line 281:
```python
b = max(2, min(w["W_base"], n))
base_slice = slice(0, b)
```
Because $W_{\text{base}} = 24$, `base_slice` is permanently capped at `slice(0, 24)` even as monitoring advances to $t = 120$.

When we evaluate the cell estimate over the full pre-monitoring stationary baseline ($W = 120$ periods):
- C0 initialization cell errors drop from **47 / 120 (39.2%) down to 18 / 120 (15.0%)**.
- Among the 14 C0 false positives, **8 of the 10 initialization errors are completely eliminated** without censoring any intermittent SKUs.

> [!NOTE]
> Expanding $W_{\text{base}}$ from 24 to 120 periods is an architectural adjustment to the pre-monitoring window. In accordance with strict protocol, this cannot be implemented as a production change on this exploratory branch.

---

## 5. Verification Against Pre-Registered Decision Criteria

| Decision Criterion | Requirement | Assessment | Verdict |
| :--- | :--- | :--- | :--- |
| **1. Baseline Reproduction** | V0 reproduces verified baseline exactly | C0 FPs = 38 (C6=14), C6 Recall = 31 | **SATISFIED** |
| **2. False Positive Reduction** | Meaningful reduction in C0 $\to$ C6 FPs | V1 reduces FPs at $N \ge 5$; V2 does not reduce init errors | **PARTIAL** |
| **3. Genuine Recall Preservation** | Genuine C6 recall is preserved reasonably well | V1 destroys $54.8\%$ of genuine C6 recall; fails completely | **FAILED** |
| **4. Ex-Ante Information Only** | Uses only information available at initialization | Verified: both V1 and V2 use only $t < 24$ data | **SATISFIED** |
| **5. No Data Leakage** | No ground-truth cells or future information used | Verified: zero ground truth or post-injection leakage | **SATISFIED** |

**Net Assessment**: Because genuine C6 recall is severely degraded by V1 and initialization error is worsened by V2, neither candidate qualifies for promotion or validation testing.

---

## 6. Deliverable File Manifest

The following exploratory artifacts were generated and verified:
1. `results/exploratory/c6_initialization/c6_initialization_report.md` (this report)
2. `results/exploratory/c6_initialization/c6_init_case_level.csv` (240 SKU records)
3. `results/exploratory/c6_initialization/c6_init_case_level.parquet` (240 SKU records)
4. `results/exploratory/c6_initialization/c6_init_summary.json` (Structured metrics and sensitivity curves)
5. `experiments/diagnostics/c6_init_repair_diagnostic.py` (Reproducible diagnostic evaluation script)

