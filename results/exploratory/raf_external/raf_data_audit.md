# RAF Spare-Parts Dataset: Audit, Verification, and Evaluation Protocol

**Workstream:** CAFR — RAF Real-Data External Evaluation (Isolated Workstream)
**Date:** 2026-10-24
**Status:** Audit Complete — Phase 1 & Phase 2 Finalized
**Output Directory:** `results/exploratory/raf_external/`

---

## 1. Provenance and Integrity Verification

The historical **Royal Air Force (RAF)** spare parts demand dataset was retrieved and cross-verified from two independent public sources:

| Property | Candidate 1 (Primary Source) | Candidate 2 (Cross-Verification) |
|---|---|---|
| **Repository** | `https://github.com/danieldehaan96/spdf` (Branch: `main`) | `https://github.com/canerturkmen/gluon-ts` (Branch: `intermittent-datasets`) |
| **File Path** | `RAF data - 7 years demand  - 5000 items.xls` | `datasets/intermittent_raf.tar.gz` |
| **Direct URL** | [Raw GitHub URL](https://raw.githubusercontent.com/danieldehaan96/spdf/main/RAF%20data%20-%207%20years%20demand%20%20-%205000%20items.xls) | [Raw GitHub URL](https://raw.githubusercontent.com/canerturkmen/gluon-ts/intermittent-datasets/datasets/intermittent_raf.tar.gz) |
| **Retrieval Date** | 2026-10-24 | 2026-10-24 |
| **File Size** | 3,001,344 bytes | 204,987 bytes |
| **SHA-256** | `1165f2d54759bf7661becc39a2e80265626b59965c61f2ad8ad12a3d7940cf09` | `66d9674b964b32c0db7243d40cdf50401b3611f0e833a179d01e98d25bfa234c` |
| **Format** | Microsoft Excel 97-2004 (`.xls`) | Gzipped TAR containing JSON (`test/data.json`) |
| **Licence** | Historical academic distribution / No formal licence statement (OD-6) | Apache-2.0 (GluonTS repository) |

### 1.1 Cross-Verification Result
An exact element-by-element numerical comparison between Candidate 1 (`.xls`) and Candidate 2 (`data.json`) across all 5,000 series and 84 monthly periods resulted in **0 mismatches out of 5,000 series**. Both sources originate from the identical underlying British Ministry of Defence / RAF inventory study (Syntetos & Boylan, 2005; Teunter, Syntetos & Babai, 2011).

---

## 2. Dataset Properties & Empirical Characteristics

The dataset was audited by direct execution using `results/exploratory/raf_external/preprocess_raf.py`.

### 2.1 Dimensionality & Integrity
* **Number of SKUs:** Exactly **5,000** unique items (`Item Ref no` 1 to 5000).
* **Number of Periods:** Exactly **84** monthly periods spanning `JAN96` (January 1996) through `DEC02` (December 2002).
* **Missing Values:** Exactly **0** nulls across all 5,000 × 84 demand observations.
* **Non-numeric / Negative Values:** Exactly **0**. All values are non-negative, integer-valued floats (min: 0.0, max: 2,062.0).
* **Meaning of Zeros:** Zeros represent genuine non-demand months in military spare-parts logistics.
* **Overall Zero Fraction:** **89.83%** (377,305 zeros out of 420,000 total observations).

### 2.2 Metadata Supplied
Unlike the stripped JSON in GluonTS, the primary `.xls` file includes three physical logistics attributes:
1. **`DESCRIPTION`:** Item descriptions (e.g. `GOUGE FIRMER`, aircraft mechanical hardware).
2. **`Lead Time (months)`:** Supplier lead time:
   * Min: 0.0, p25: 5.0, **Median: 9.0**, Mean: 8.71, p75: 12.0, **Max: 33.0 months**.
3. **`PRICE (£)`:** Unit price in British Pounds:
   * Min: £0.0, p25: £1.92, **Median: £11.17**, Mean: £102.32, p75: £64.59, **Max: £9,131.99**.

### 2.3 Demand Sparsity and Intermittency Profile
| Metric | RAF (Historical Real Panel) | RUF v2 (Synthetic Surrogate) | Consequence for Analysis |
|---|---|---|---|
| **Series Length ($T$)** | **84 months** | 60 months | RAF fits Rev2 §14.1's 84-period plan (24 burn-in + 60 evaluation) **with 0 shortfall**. |
| **Zero Fraction** | **89.83%** | 70.08% | RAF is far sparser than RUF. |
| **Non-zero Periods per SKU** | Min 3, p25 6, **Median 8**, p75 10, **Max 20** | Min 1, p25 7, **Median 12**, p75 26, **Max 60** | Across 7 years, 90% of SKUs have $\le 12$ non-zero demands. |
| **Median ADI (Full Series)** | **10.5** (Inter-arrival time) | 5.0 | Average inter-arrival interval is nearly 1 year. |
| **Median $\text{CV}^2$ (Full Series)** | **0.460** | 0.317 | High demand-size variability. |
| **Syntetos–Boylan Breakdown** | **Intermittent:** 2,597 (51.9%)<br>**Lumpy:** 2,403 (48.1%)<br>**Smooth / Erratic:** 0 (0.0%) | Intermittent: 64.4%<br>Lumpy: 25.8%<br>Smooth: 6.1%<br>Erratic: 2.9% | **100%** of RAF series are intermittent or lumpy. Smooth forecasting methods are inapplicable. |
| **Reachable 2×2 (Full)** | High-Low: 2,597 · High-High: 2,403<br>Moderate: **0 (0.0%)** | All 4 cells populated (minimum cell share 13.4%) | Rev2's reachable band ($\text{ADI}^* = 4.0$) fails to populate the moderate band ($\text{ADI} < 4.0$) because min ADI on RAF is 4.2. |
| **Dead SKUs in Burn-in (24m)** | **1,383 (27.66%)** | 1,304 (26.08%) | 27.7% of SKUs have $< 2$ non-zeros in the first 24 months, forcing abstention by construction. |

### 2.4 Detector Evidence Floor Feasibility
Because RAF SKUs have so few non-zero periods (max 20 non-zeros across 84 months), CAFR's evidence floors are heavily constrained:
* **C1 ($W=24$, need $\ge 8$ non-zeros):** Only **1.22%** of SKUs ever reach this threshold.
* **C3 ($W=60$, need $\ge 10$ non-zeros):** Only **19.4%** of SKUs ever reach this threshold (and **0.0%** at $W=24$).
* **C4 ($W=24$, need $\ge 10$ non-zeros):** Exactly **0.0%** of SKUs reach this threshold.

---

## 3. Evaluation Protocol: What Can and Cannot Be Evaluated Honestly

To maintain strict scientific integrity, we explicitly partition CAFR's experimental pipeline into supported vs unsupported components:

```
+-----------------------------------------------------------------------------------------+
|                                 RAF EVALUATION MATRIX                                   |
+-------------------------------------------------------+---------------------------------+
| Component                                             | Status on RAF                   |
+-------------------------------------------------------+---------------------------------+
| 1. Pure Intermittent Demand Forecasting (Rolling/Split)| [SUPPORTED] Leakage-free        |
| 2. Point & Distributional Forecast Metrics            | [SUPPORTED] sMAE, RMSSE, WAPE   |
| 3. SKU Burn-In Classification (ADI/CV2)               | [SUPPORTED] Burn-in periods 0-23|
| 4. Inventory Simulation (End-to-End)                  | [UNSUPPORTED without assumptions|
| 5. Cause-Attribution Accuracy (C0-C7 Ground Truth)    | [BLOCKED] No ground-truth labels|
| 6. Bandits & Policy Action Selection                  | [BLOCKED] Requires cause labels |
+-------------------------------------------------------+---------------------------------+
```

### 3.1 Why Cause-Attribution ($C_0$–$C_7$) Cannot Be Evaluated on RAF
* RAF is an observational historical dataset from aircraft maintenance records. It **does not contain ground-truth labels** explaining why any specific order occurred or why a drop in demand happened.
* Inventing or assigning synthetic $C_0$–$C_7$ labels to RAF would be scientific fabrication.
* Attribution accuracy ($F_1$ score) remains validated exclusively on CAFR's ground-truth synthetic generator (`data/synth/`).

### 3.2 Why Inventory Simulation Is Limited Without External Assumptions
* RAF provides unit price (`PRICE (£)`) and supplier lead time (`Lead Time (months)`), but **does not provide**:
  1. Unit holding rate $h$ or capital opportunity cost.
  2. Stockout penalty or backorder cost $B$ (or target service levels $\alpha$).
  3. Stockout / censoring records (like RUF, censoring is present but unobserved).
* Furthermore, RAF lead times have a median of 9 months and reach 33 months. Simulating with $L=9$ to $33$ on an 84-month panel would consume 10%–40% of the entire temporal horizon in pipeline lead times.
* Consequently, end-to-end inventory cost numbers can only be generated by imposing external arbitrary assumptions.

### 3.3 The Supported Protocol: Temporal Split Forecasting Evaluation
We establish a predeclared, strictly leakage-free forecasting benchmark:

1. **Temporal Split:**
   * **In-sample Burn-in:** Periods 0–23 (24 months, `1996-01` to `1997-12`). Used strictly to initialize classical smoothing parameters and compute initial $\text{ADI}/\text{CV}^2$.
   * **Out-of-sample Evaluation Window:** Periods 24–83 (60 months, `1998-01` to `2002-12`). Exactly matches the 60-period evaluated span specified in Rev2 §14.1.
2. **SKU Split for Pooled Models:**
   * Disjoint SKU partitioning: 60% of SKUs (3,000 SKUs) reserved for training pooled ML models (Global LightGBM), 40% (2,000 SKUs) held out strictly for evaluation.
3. **Candidate Forecasting Methods Compared:**
   * **Zero Baseline:** Naive zero-demand forecast $\hat{y} = 0$.
   * **Croston Method:** Classical decomposition ($\alpha = 0.1$).
   * **Syntetos–Boylan Approximation (SBA):** Bias-corrected Croston ($\alpha = 0.1$).
   * **Teunter–Syntetos–Babai (TSB):** Demand probability updating on every period ($\alpha_d = 0.2, \alpha_p = 0.2$).
   * **SES on Sizes:** Croston interval with SES on demand sizes ($\alpha = 0.1$).
   * **Global LightGBM:** Pooled gradient boosted regression trees trained across disjoint series.
4. **Leakage Control:**
   * In-sample summary statistics (mean demand, variance) are computed strictly on periods 0–23.
   * Rolling origins advance strictly one period at a time ($t = 24, \dots, 83$) observing only past periods $s \le t$.
5. **Evaluation Metrics:**
   * **sMAE / MASE:** Mean Absolute Error scaled by in-sample mean demand:
     $$\text{sMAE} = \frac{\frac{1}{H}\sum_{t=1}^{H} |y_t - \hat{y}_t|}{\frac{1}{T_{\text{burn}}}\sum_{s=1}^{T_{\text{burn}}} y_s}$$
   * **RMSSE:** Root Mean Squared Scaled Error.
   * **WAPE:** Weighted Absolute Percentage Error ($\sum |y - \hat{y}| / \sum y$).
   * **Mean Bias / PIS Proxy:** Cumulative forecast error $\sum (\hat{y} - y)$.
   * *(Note: MAPE is omitted because 89.83% of actual values are zero, making MAPE mathematically undefined).*
