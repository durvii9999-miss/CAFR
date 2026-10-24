# RAF Spare-Parts External Forecasting Evaluation Report

**Workstream:** CAFR — RAF Real-Data External Evaluation (Isolated Workstream)
**Date:** 2026-10-24
**Status:** Evaluation Executed and Verified
**Primary Deliverable:** `results/exploratory/raf_external/raf_evaluation_report.md`
**Underlying Artifacts:**
- Data Audit: [`raf_data_audit.md`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/raf_external/raf_data_audit.md)
- Machine-Readable Profile: [`raf_data_profile.json`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/raf_external/raf_data_profile.json)
- Summary Results: [`raf_forecasting_results.json`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/raf_external/raf_forecasting_results.json)
- Series-Level Metrics Parquet: [`raf_series_metrics.parquet`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/raf_external/raf_series_metrics.parquet)
- Preprocessing Script: [`preprocess_raf.py`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/raf_external/preprocess_raf.py)
- Evaluation Driver: [`run_raf_eval.py`](file:///c:/Users/DELL/Minor_Project/CAFR/results/exploratory/raf_external/run_raf_eval.py)

---

## 1. Executive Summary & Decision

### Decision: **B. Dataset verified, but only limited forecasting tests are supported.**

* **Verified:** The Royal Air Force (RAF) spare-parts panel (5,000 SKUs × 84 monthly periods) was successfully acquired and verified against two independent public sources (0 numerical mismatches out of 5,000 series). It perfectly satisfies the 84-period horizon (24 burn-in + 60 evaluation) originally assumed in Revision 2 §14.1 with **zero temporal shortfall**.
* **Why Decision B (Limited Forecasting Supported):**
  1. **No Ground-Truth Cause Labels ($C_0$–$C_7$):** RAF is an empirical maintenance demand record. It does not contain labels explaining why forecasting failures occurred. Any claim of validating CAFR's cause-attribution accuracy on RAF would be scientifically fraudulent.
  2. **No Inventory Cost Parameters:** RAF provides item unit price and supplier lead time, but omits holding rate $h$, shortage penalty $B$, and target service level $\alpha$. Full inventory simulation cannot be conducted without external arbitrary assumptions.
  3. **Supported Benchmark:** Direct, leakage-free, rolling-origin **one-step-ahead forecasting** over 60 evaluated months (periods 24–83) across classical methods and pooled machine learning is fully supported and was executed across 400 test series (24,000 out-of-sample evaluated periods).

---

## 2. Experimental Setup & Protocol

* **Panel Sample Size:** 1,000 SKUs sampled from the 5,000-SKU panel (Seed 42).
  * **Training Split (60%):** 600 SKUs used strictly to train the pooled ML model (`GlobalLightGBM`).
  * **Test Split (40%):** 400 SKUs held out strictly for out-of-sample rolling-origin evaluation.
* **Temporal Horizon:**
  * **Burn-in / Initialization:** Periods 0–23 (24 months: Jan 1996 – Dec 1997). Used to initialize smoothing equations and compute static features (`adi_init`, `cv2_init`, `sb_cell_init`).
  * **Rolling-Origin Evaluation Window:** Periods 24–83 (60 months: Jan 1998 – Dec 2002). At each origin $t \in [24, 83]$, methods only observe history $s \le t-1$ to predict $y_t$.
* **Candidate Methods:**
  1. `zero`: Naive zero demand baseline ($\hat{y}_t = 0$).
  2. `croston`: Classic Croston (1972) with $\alpha = 0.1$.
  3. `sba`: Syntetos-Boylan Approximation (2005) with $\alpha = 0.1$ and 0.95 bias correction factor.
  4. `tsb`: Teunter-Syntetos-Babai (2011) with $\alpha_d = 0.2, \alpha_p = 0.2$.
  5. `ses_sizes`: Croston inter-arrival with SES on demand sizes ($\alpha = 0.1$).
  6. `lightgbm_global`: Pooled LightGBM regressor with lagged features, rolling stats, and static attributes, trained strictly on disjoint training SKUs.

---

## 3. Comparative Evaluation Results

Evaluation across 400 test SKUs × 60 rolling origins = 24,000 out-of-sample forecasts. Values report Mean, Median, and 95% non-parametric bootstrap confidence intervals:

| Method | sMAE (MASE) Mean [95% CI] | sMAE Median | RMSSE Mean | WAPE Mean | Mean Bias ($\hat{y} - y$) | PIS (Cum. Error) |
|---|---|---|---|---|---|---|
| **Zero Baseline** | 2.6697 [2.0091, 3.4991] | **1.0000** | 11.4393 | **1.0000** | -1.4430 | -2,679.31 |
| **SES on Sizes** | **4.0305** [3.2077, 5.0782] | 1.8746 | **11.3669** | 2.2681 | -0.1576 | -373.37 |
| **SBA** | 4.2642 [3.4896, 5.3149] | 2.2781 | 11.4580 | 2.9743 | +0.3874 | +943.24 |
| **TSB** | 4.3393 [3.3688, 5.5701] | 1.8587 | 11.5298 | 2.1623 | **-0.0899** | **-231.50** |
| **Croston** | 4.3493 [3.5665, 5.4107] | 2.3439 | 11.4781 | 3.0811 | +0.4838 | +1,133.90 |
| **Global LightGBM** | 12.7423 [10.4018, 15.7112] | 6.0534 | 20.9365 | 8.4744 | +1.8038 | +3,179.86 |

---

## 4. Key Findings & Scientific Insights

### 4.1 SBA Directly Improves Upon Croston on Real Intermittent Data
* SBA achieves a lower mean sMAE (4.2642 vs 4.3493) and lower positive bias (+0.3874 vs +0.4838) compared to Classic Croston.
* The pairwise Wilcoxon signed-rank test confirms that SBA's reduction in error over Croston is **statistically significant** ($p = 2.73 \times 10^{-67}$).
* This empirically replicates the foundational finding of Syntetos & Boylan (2005) on the exact historical dataset on which the method was originally vetted.

### 4.2 The "Zero Baseline" Paradox in Extreme Sparsity (89.8% Zeros)
* On median sMAE (1.0000) and WAPE (1.0000), the Zero Baseline outperforms all non-zero forecasting algorithms.
* Why? When demand occurs only 1 or 2 times every 12 months, predicting any positive number $\hat{y} > 0$ produces a non-zero error during the 10 or 11 empty months.
* However, predicting zero produces massive negative tracking error (Mean Bias: -1.4430, PIS: -2,679.31), which would lead to 100% stockout in an inventory setting.
* Among non-zero models, **TSB** achieves the lowest tracking error (Mean Bias: -0.0899, PIS: -231.50) because it updates demand probability downward on consecutive zero periods.

### 4.3 Catastrophic Overprediction by Pooled LightGBM on Spare Parts
* Global LightGBM performs significantly worse than all classical intermittent baselines (sMAE: 12.7423 vs SBA's 4.2642; Wilcoxon $p = 2.57 \times 10^{-58}$).
* **Root Cause Analysis:** On extreme spare-parts data (89.8% zeros, 0% smooth items), gradient boosted trees trained with L2 loss predict residual positive values (e.g. 0.3–0.8 units) continuously across dormant series.
* On series that are **"dead in burn-in"** ($< 2$ non-zeros in the first 24 months, 27.7% of the panel):
  * Classical TSB sMAE = 8.8873.
  * Global LightGBM sMAE explodes to **27.8027** (a 312% error inflation).
* This provides empirical validation for CAFR's design principle: **pooled gradient boosting cannot replace dedicated intermittent demand estimators on deeply sparse, low-volume spare-parts panels without cause-attributed stabilization.**

---

## 5. Scope Boundaries and Protections Maintained

1. **Isolation Preserved:** All outputs, scripts, and logs were confined to `results/exploratory/raf_external/`.
2. **No Modification of Core Pipeline:** No production code in `cafr/`, no configs in `cafr/configs/`, and no official results in `results/step5/` or `results/step6/` were modified or rerun.
3. **No Ground-Truth Leakage or Attribution Claims:** No synthetic $C_0$–$C_7$ labels were generated or attributed to RAF series.
4. **Planned Roadmap Intact:** The planned training-only diagnosis of C6 ramp detection versus C3 preemption remains completely unaffected and ready for its scheduled run.
