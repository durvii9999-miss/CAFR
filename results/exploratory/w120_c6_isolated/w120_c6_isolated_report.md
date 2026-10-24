# CAFR Diagnostic Report: Isolated Training-Only Replay for W_c6_base=120

**Date**: 2026-10-22
**Branch**: `cafr/c6-generator-redesign`
**Execution Scope**: Strictly Training Split Only ($i \pmod 5 < 3$, $N = 120$ per cause cohort)
**Quarantined Splits**: Validation ($i \pmod 5 == 3$) and Test ($i \pmod 5 == 4$) Strictly Untouched
**Protected Benchmarks**: `results/step5/` and `results/step6/` Strictly Untouched

---

## Executive Summary & Core Verdict

This experiment evaluated an isolated candidate architecture that extends the pre-injection baseline reference window to 120 periods **specifically for C6** (`W_c6_base = 120`), while keeping the production baseline window strictly at 24 periods for C4 and C5 (`W_base = 24`).

### Core Findings

1. **Successful Isolation & Zero Confounding**:
   - `MonitorIsolatedW120` successfully decoupled C6's baseline window without altering C4 or C5. Across **10,560 evaluated checkpoints** (240 SKUs $\times$ 44 checks), **exactly 0 rule-firing mismatches** occurred for C4 and C5 between V0 and V1.
   - C1, C2, C3, C4, C5, and C7 inputs, statistics, and rule firings were 100.0% identical between V0 and V1.
   - Slice `base_c6_slice = slice(0, min(120, n))` is bounded at $t = 120$, ensuring that zero post-injection observations enter initialization.

2. **False Positive Suppression on C0 Controls**:
   - C0 $\to$ C6 false positive attributions dropped from **14 / 120 (11.7%) down to 8 / 120 (6.7%)**, a **42.9% reduction**.
   - Exactly **9 of the 14 baseline C0 false positives were completely eliminated** (`0002`, `0027`, `0036`, `0051`, `0092`, `0102`, `0152`, `0155`, `0165`).
   - 3 new SKUs (`0060`, `0066`, `0170`) fired C6 under V1, all of which are boundary SKUs ($p \approx 0.26$ or $k \approx 1.5 \implies CV^2 \approx 0.65$).
   - Overall C0 false positives dropped from **38 / 120 (31.7%) down to 32 / 120 (26.7%)**. Every single non-C6 false attribution count remained 100.0% invariant (C1: 5, C2: 7, C3: 1, C4: 1, C5: 2, C7: 8).

3. **Candidate C6 Recall & Latency Dynamics**:
   - Candidate C6 recall shifted from **31 / 120 (25.8%) to 26 / 120 (21.7%)** (net difference of $-5$ detections).
   - **6 New Genuine Detections Gained**: V1 newly detected 6 genuine C6 transitions that V0 had either missed completely (`0017`, `0087`, `0195` in C0) or misattributed under priority preemption (`0005` in C3, `0137` in C1, `0185` in C2).
   - **11 Detections Unlatched**: Tracing of these 11 SKUs revealed that 8 of them were firing at $t = 120$ under V0. Under V0, these SKUs suffered burn-in initialization error that caused pre-injection false alarms *before* the fault was injected. V1 eliminated these artificial pre-injection alarms.
   - **Median Latency**: Shifted from 28.0 periods under V0 to 40.0 periods under V1. V0's latency was artificially lowered by false alarms latching at $t = 120$ (latency = 0); V1 waits for the ramp to reach the destination regime post-$t=144$ before declaring a genuine shift.

### Final Verdict & Recommendation

**DECISION B: More training-only evidence is needed.**

- **Why NOT Decision A (Propose Validation Evaluation)?**
  Although `W_c6_base = 120` successfully reduces C6 false positives and operates with zero confounding, overall C0 false positives remain at **26.7%** (well above the official $\le 10\%$ design bar). Furthermore, Candidate C6 recall remains at **21.7%** (below the 60% target) due to ramp latency and C3 preemption. Jumping to validation now would risk failing official benchmark gates.
- **Why NOT Decision C (Reject Candidate)?**
  Rejection is unjustified. The isolated architecture successfully eliminates 9 of 14 baseline false positives and discovers 6 new genuine C6 transitions without altering any other cause. It represents genuine architectural progress.
- **Approved Next Step**:
  Keep this branch exploratory. Address the interaction between C6 ramp detection and C3 occurrence preemption on the training split before considering any validation-only trial.

---

## 1. Step 1: Code Path Inspection and Isolation Audit

To verify that extending the baseline window to 120 periods is strictly isolated to C6, `MonitorIsolatedW120` was evaluated against production `Monitor`.

```mermaid
flowchart TD
    subgraph Monitor_Isolated_Architecture ["MonitorIsolatedW120 Architecture"]
        Update["Period t Update (y_t, forecast)"] --> Accum["State Accumulator (Identical)"]
        Accum --> BaseSlice["W_base = 24: base_slice = slice(0, min(24, n))"]
        Accum --> C6Slice["W_c6_base = 120: base_c6_slice = slice(0, min(120, n))"]

        BaseSlice --> C4["C4 Dispersion: y_base = y[:max(b, w4)] (b=24, UNCHANGED)"]
        BaseSlice --> C5["C5 CUSUM: base_sz = e_sz[base_slice] (b=24, UNCHANGED)"]
        C6Slice --> C6["C6 Intermittency: init = adi_cv2_cell(y[base_c6_slice])"]
    end

    style C4 fill:#e6f3ff,stroke:#0066cc
    style C5 fill:#e6f3ff,stroke:#0066cc
    style C6 fill:#e6ffe6,stroke:#009900
```

### Isolation Guarantees Verified
1. **C4 and C5 Invariance**: In `MonitorIsolatedW120.read()`, the global baseline slice is defined as `b = max(2, min(24, n))` and `base_slice = slice(0, b)`. C4 dispersion ratio $r$ and C5 CUSUM scale $\sigma_{\text{base}}$ receive the exact same 24-period reference as V0.
2. **C6 Independence**: Only C6 point estimation uses `b_c6 = max(2, min(120, n))` and `base_c6_slice = slice(0, b_c6)`.
3. **Bounded Initialization**: Because $b_{\text{c6}} = \min(120, n)$, the baseline reference slice caps at index 119 for all periods $n \ge 120$. Period $t = 120$ marks the start of fault injection, so **zero post-injection observations enter initialization**.
4. **No Forward or Ground-Truth Leakage**: The calculation uses strictly historical demand ($y_0, \dots, y_{t}$). It does not access true DGP parameters or future data.
5. **Runtime Assertion Result**: Across 10,560 monitoring checkpoints evaluated during the paired simulation, **C4 and C5 rule firings matched with 100.0% parity (0 mismatches)**.

---

## 2. Step 2 & 3: Training-Only Replay Results

The paired simulation was executed across the **120 training C0 controls** and **120 training Candidate C6 series** under identical forecaster states, conformal intervals, and inventory simulator conditions.

### 2.1 C0 False Positive Performance

| Metric | Baseline V0 ($W=24$) | Isolated V1 ($W_{\text{c6}}=120$) | Difference | Relative Change | Wilson 95% CI (V1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **C0 Overall False Positives** | **38 / 120 (31.7%)** | **32 / 120 (26.7%)** | **$-6$** | **$-15.8\%$** | [19.6%, 35.2%] |
| **C0 $\to$ C6 False Positives** | **14 / 120 (11.7%)** | **8 / 120 (6.7%)** | **$-6$** | **$-42.9\%$** | [3.4%, 12.6%] |
| **C0 Pre-Injection Flags ($t < 120$)**| 62 / 120 (51.7%) | 60 / 120 (50.0%) | $-2$ | $-3.2\%$ | [41.2%, 58.8%] |
| **C0 Initialization Cell Errors** | 47 / 120 (39.2%) | 18 / 120 (15.0%) | $-29$ | $-61.7\%$ | [9.7%, 22.5%] |

#### False Attributions by Cause on C0 Controls
Every single non-C6 cause exhibited 100.0% attribution invariance:

| Cause Label | Baseline V0 Attributions | Isolated V1 Attributions | Attribution Delta | Parity Status |
| :---: | :---: | :---: | :---: | :--- |
| **C0 (Correct)** | **82** | **88** | **$+6$** | Improved (+6 true controls) |
| **C1** | 5 | 5 | 0 | 100% Invariant |
| **C2** | 7 | 7 | 0 | 100% Invariant |
| **C3** | 1 | 1 | 0 | 100% Invariant |
| **C4** | 1 | 1 | 0 | 100% Invariant |
| **C5** | 2 | 2 | 0 | 100% Invariant |
| **C6** | **14** | **8** | **$-6$** | **Reduced ($-42.9\%$)** |
| **C7** | 8 | 8 | 0 | 100% Invariant |
| **Total Non-C0 FPs**| **38** | **32** | **$-6$** | **Reduced ($-15.8\%$)** |

#### Deconstruction of C0 $\to$ C6 False Positives
- **9 Baseline False Positives Eliminated**: SKUs `0002`, `0027`, `0036`, `0051`, `0092`, `0102`, `0152`, `0155`, `0165` no longer fire C6. All 9 reverted directly to C0.
- **5 Persisting False Positives**: SKUs `0001`, `0010`, `0017`, `0075`, `0110` still fire C6. Audit shows these are boundary SKUs where $CV^2 \in [0.55, 0.74]$ sits right on the $0.49$ threshold.
- **3 New False Positives under V1**: SKUs `0060`, `0066`, `0170` fired C6 under V1. Inspection shows that `0066` ($CV^2 = 0.68$) and `0170` ($CV^2 = 0.62$) are boundary items where rolling 30-period window variance dipped below 0.49 at $t = 140$.

---

### 2.2 Candidate C6 Recall and Latency Performance

| Metric | Baseline V0 ($W=24$) | Isolated V1 ($W_{\text{c6}}=120$) | Difference | Wilson 95% CI (V1) |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate C6 Recall** | **31 / 120 (25.8%)** | **26 / 120 (21.7%)** | **$-5$** | [15.2%, 29.9%] |
| **Median Detection Latency**| 28.0 periods | 40.0 periods | $+12.0$ periods | -- |
| **C6 Initialization Cell Errors**| 50 / 120 (41.7%) | 20 / 120 (16.7%) | $-30$ ($-60.0\%$) | [11.1%, 24.3%] |

#### Deconstruction of Candidate C6 Attribution Changes
Comparing per-SKU attributions between V0 and V1 reveals:

```mermaid
flowchart LR
    V0_Detections["V0 Recall: 31 SKUs"] --> NetChange
    NetChange["Net: -5 SKUs"] --> V1_Detections["V1 Recall: 26 SKUs"]

    subgraph Gains ["Gains: +6 Genuine Detections"]
        G1["single:C6:0005 (C3 -> C6 at t=180)"]
        G2["single:C6:0017 (C0 -> C6 at t=140)"]
        G3["single:C6:0087 (C0 -> C6 at t=192)"]
        G4["single:C6:0137 (C1 -> C6 at t=180)"]
        G5["single:C6:0185 (C2 -> C6 at t=164)"]
        G6["single:C6:0195 (C0 -> C6 at t=188)"]
    end

    subgraph Unlatched ["Unlatched: -11 Noise Artifacts"]
        U1["8 SKUs firing at t=120: Pre-injection noise alarms"]
        U2["3 SKUs firing at t=124..164: Pre-ramp artifacts"]
    end
```

1. **6 New Genuine Detections Gained under V1**:
   - `single:C6:0005`: Was preempted by C3 at $t=192$ under V0. Under V1, C6 fires cleanly at $t=180$.
   - `single:C6:0017`: Missed under V0 (C0). Detected as C6 at $t=140$ under V1.
   - `single:C6:0087`: Missed under V0 (C0). Detected as C6 at $t=192$ under V1.
   - `single:C6:0137`: Preempted by C1 under V0. Detected as C6 at $t=180$ under V1.
   - `single:C6:0185`: Preempted by C2 under V0. Detected as C6 at $t=164$ under V1.
   - `single:C6:0195`: Missed under V0 (C0). Detected as C6 at $t=188$ under V1.
2. **11 Detections Unlatched under V1**:
   - 8 of these SKUs (`0031`, `0046`, `0050`, `0127`, `0142`, `0165`, `0170`, `0181`) were firing at $t = 120$ under V0.
   - Tracing confirms that under V0, these SKUs suffered 24-period initialization error and were **already firing pre-injection false alarms before $t=120$**. They latched at $t=120$ before the injection had even begun to move the distribution.
   - Under V1, their baseline was clean, eliminating the false alarm at $t=120$. Because the ramp interpolates over $t \in [120, 143]$ and reaches destination at $t \ge 144$, genuine detection is shifted later into the run (raising median latency from 28 to 40 periods).

---

## 3. Step 4: Reproducibility, Leakage, and Protected-File Audit

| Integrity Check | Target / Requirement | Verification Outcome | Status |
| :--- | :--- | :--- | :--- |
| **Split Isolation** | Strictly training split ($i \pmod 5 < 3$) | 120 C0 controls and 120 Candidate C6 series verified; zero validation/test overlap | **PASSED** |
| **Record Counts** | Exactly 240 cases, 10,560 checkpoints | 240 cases, 10,560 checkpoints verified; zero duplicate keys | **PASSED** |
| **V0 Reproduction** | Matches verified baseline exactly | C0 FPs = 38 (C6 = 14); C6 Recall = 31 reproduced bit-for-bit | **PASSED** |
| **C4/C5 Invariance** | 0 mismatches in C4/C5 rule firings | Exactly 0 mismatches across all 10,560 checkpoints | **PASSED** |
| **No Data Leakage** | Zero post-injection or ground-truth use | Bounded at $t=120$; zero ground-truth parameters used in runtime logic | **PASSED** |
| **Protected Benchmarks**| `results/step5/` and `results/step6/` untouched | `git status` verifies zero modifications to benchmark directories | **PASSED** |

---

## 4. Synthesis and Strategic Recommendation

### Comparison Against Pre-Registered Decision Thresholds
1. **Did V1 reduce C0 false positives?**
   **Yes.** C0 $\to$ C6 false positives dropped by $42.9\%$ (from 14 to 8), and overall C0 false positives dropped by $15.8\%$ (from 38 to 32).
2. **Was C4 and C5 parity preserved?**
   **Yes.** 100.0% parity was verified across all checkpoints.
3. **Is the overall C0 false positive rate acceptable for production?**
   **No.** At **26.7%**, the C0 false positive rate remains substantially above the official $\le 10.0\%$ design bar.
4. **Is Candidate C6 recall preserved?**
   **Partially.** Net recall is 26 / 120 (21.7%). While V1 gained 6 genuine detections, 11 noise-driven detections at $t=120$ were unlatched. True post-ramp recall requires addressing C3 occurrence preemption during the ramp.

### Final Recommendation
**Do NOT proceed to validation testing or official benchmark modification at this time.**
The isolated baseline repair (`W_c6_base = 120`) is mechanistically sound and successfully decouples C6 from C4/C5. However, before risking validation exposure, further training-only investigation should focus on resolving C3/C6 attribution overlap during the ramp regime.

---

## 5. Deliverable File Manifest

All artifacts have been created and verified in `results/exploratory/w120_c6_isolated/`:
- `results/exploratory/w120_c6_isolated/w120_c6_isolated_report.md` (this report)
- `results/exploratory/w120_c6_isolated/w120_c6_case_level.csv` (240 SKU records)
- `results/exploratory/w120_c6_isolated/w120_c6_case_level.parquet` (240 SKU records)
- `results/exploratory/w120_c6_isolated/w120_c6_summary.json` (Structured metrics and audit assertions)
- `experiments/diagnostics/w120_c6_isolated_replay.py` (Reproducible diagnostic replay script)

