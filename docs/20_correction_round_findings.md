# E-8 correction round findings

**Run date:** 2026-10-19  
**Branch:** `cafr/e8-corrections`  
**Panel command:** `python experiments/build_synth_panel.py --seed 42 --c7-skus 40`  
**Measurement command for every Step 5 row below:** `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42`

The pre-registered bars and frozen settings were not changed. Before values are from the archived run in `results/step5/e8_before/`; after values are from the Step 5 rerun. Wilson intervals use the run's per-SKU counts. C0 firing is the matched one-row-per-SKU `fired` reading before and after; the any-check rate is additionally reported for C4 and C5.

## Before and after

| Cause | Recall before [95% Wilson] | Recall after [95% Wilson] | C0 rule firing before -> after [95% Wilson] | Command |
|---|---|---|---|---|
| C1 | 0.030 [0.014, 0.064] (6/200) | 0.025 [0.011, 0.057] (5/200) | 0.025 [0.011, 0.057] -> 0.030 [0.014, 0.064] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C2 direct | 0.955 [0.917, 0.976] (191/200) | 0.985 [0.957, 0.995] (197/200) | 0.145 [0.103, 0.200] -> 0.145 [0.103, 0.200] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C2 pooled (C2 + C2b) | 0.522 [0.474, 0.571] (209/400) | 0.535 [0.486, 0.583] (214/400) | 0.145 [0.103, 0.200] -> 0.145 [0.103, 0.200] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C2b variant | 0.090 [0.058, 0.138] (18/200) | 0.085 [0.054, 0.132] (17/200) | C2 rule: 0.145 [0.103, 0.200] -> 0.145 [0.103, 0.200] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C3 | 0.055 [0.031, 0.096] (11/200) | 0.060 [0.035, 0.102] (12/200) | 0.005 [0.001, 0.028] -> 0.010 [0.003, 0.036] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C4 | 0.000 [0.000, 0.019] (0/200) | 0.075 [0.046, 0.120] (15/200) | 0.000 [0.000, 0.019] -> 0.015 [0.005, 0.043] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C5 | 0.110 [0.074, 0.161] (22/200) | 0.035 [0.017, 0.070] (7/200) | 0.085 [0.054, 0.132] -> 0.015 [0.005, 0.043] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C6 | 0.145 [0.103, 0.200] (29/200) | 0.165 [0.120, 0.223] (33/200) | 0.195 [0.146, 0.255] -> 0.195 [0.146, 0.255] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |
| C7 (reported, not gated) | 0.280 [0.222, 0.346] (56/200) | 0.285 [0.227, 0.351] (57/200) | 0.070 [0.042, 0.114] -> 0.070 [0.042, 0.114] | `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42` |

The matched scored-reading C0 firing rates are directly tabulated from the archived and rerun `step5_readings.parquet` files. For the corrected rules, the rerun also measured any-check firing: C4 was 6/200 (0.030, 95% Wilson [0.014, 0.064]); C5 was 10/200 (0.050, 95% Wilson [0.027, 0.090]). C4's scored-reading rate was 3/200 (0.015); its any-check rate is the more conservative per-SKU measure.

C0's overall false-positive rate was 90/200 (0.450, 95% Wilson [0.383, 0.519]) before and 81/200 (0.405, 95% Wilson [0.339, 0.474]) after. It remains above its registered bar. The rerun minimum per-class F1 was 0.044643; the off-diagonal gate also remains failed. No failure was tuned away.

## Fix 1: C2b interval reaches inventory

`inject_C2b` reported `half_width`, but the generated SKU metadata previously discarded it. The parameter now survives panel generation in [`labels.py`](../cafr/data/synth/labels.py#L185), Step 5 forwards it in [`step5_monitor_gates.py`](../experiments/step5_monitor_gates.py#L81), and arm (c) passes it through in [`arm_c.py`](../cafr/arms/arm_c.py#L157) to [`inventory.py`](../cafr/sim/inventory.py#L239). The inventory function replaces only the residual safety quantile with the supplied half-width at [`inventory.py`](../cafr/sim/inventory.py#L293); it does not scale by `sqrt(R)`. The order-up-to point term `R * mu_hat` is unchanged. The optional argument defaults to `None`, preserving C0 and direct C2 behavior. The panel generator continues to return `demand_touched: False`.

The focused regression in [`test_step4_regressions.py`](../tests/test_step4_regressions.py#L162) pins C0 and C2 order-up-to outputs to frozen values, verifies the C2b width differs on at least 90% of sampled SKUs, and checks demand remains untouched. The metadata integration test is at [`test_step4_regressions.py`](../tests/test_step4_regressions.py#L198). The rebuilt panel's signed generator coverage-gap median was -0.176075 and its absolute median was 0.176075 across the 200 C2b SKUs, meeting the intended 0.10 gap-magnitude check. C2b attribution recall was 17/200 (0.085, 95% Wilson [0.054, 0.132]): this correction makes the aggregate policy interval real, but the Step 5 C2 monitor still evaluates its one-period conformal interval. **C2b detector verdict: FAIL**, not a claimed gate pass.

## Fix 2: C4 AUC is diagnostic only

[`rules.py`](../cafr/attribution/rules.py#L130) no longer requires occurrence AUC to fire C4. The two-sided dispersion ratio above 1.5 in two consecutive windows is unchanged. AUC remains computed in the monitor and is reported only as a diagnostic: the C4-panel median was 0.434884 and it was estimable for 96/200 SKUs. It is an occurrence-ranking measure, not a size-dispersion guard, so no substitute occurrence guard was added; the existing C4 evidence floor and separate occurrence/size rules remain intact. This removes a structurally unrelated veto, not a threshold on dispersion.

C4 fired on 15/200 C4 SKUs (0.075, 95% Wilson [0.046, 0.120]) and on 6/200 C0 SKUs at any check (0.030, [0.014, 0.064]). Its recall was measured on all 200 C4 SKUs. The recall bar is still missed. The AUC-independent regression and fixed-sample unaffected-rule predicates are in [`test_step4_regressions.py`](../tests/test_step4_regressions.py#L208) and [`test_step4_regressions.py`](../tests/test_step4_regressions.py#L216).

## Fix 3: C5 control calibration and resets

[`features.py`](../cafr/monitor/features.py#L231) resets positive and negative CUSUM accumulators at each alarm and exposes the reset-aware excursion separately from the legacy continuous diagnostic. [`reading.py`](../cafr/monitor/reading.py#L143) uses the configured burn-in; size and occurrence residuals remain separate streams in [`reading.py`](../cafr/monitor/reading.py#L389). The Step 5 calibration in [`step5_monitor_gates.py`](../experiments/step5_monitor_gates.py#L567) uses only the sorted first 60% of C0 control SKU IDs (120 SKUs), and only pre-injection periods [24, 120). It takes the higher 95th percentile of the maximum reset-aware standardized excursion across the size and occurrence CUSUMs. No injected period, C2/C2b SKU, validation row, or test row is used. `k` stayed at 0.25.

The run selected `h = 10.642717` from 2,880 train-control checks; the measured calibration false-alarm rate was 0.04896 against the 0.05 target. On held-out C0 controls, C5 rule firing was 10/200 at any check (0.050, 95% Wilson [0.027, 0.090]); occurrence CUSUM alarms occurred at least once on 40/200 (0.200, [0.150, 0.261]), and the occurrence per-check rate was 616/8,800 (0.0700, [0.0649, 0.0755]). The prior occurrence-CUSUM any-check rate was 99.0%. C5 recall was 7/200 (0.035, 95% Wilson [0.017, 0.070]), so the C5 attribution verdict is **FAIL** despite the control calibration behaving as targeted. The reset/burn-in regression is at [`test_step4_regressions.py`](../tests/test_step4_regressions.py#L237).

## Verdicts

| Cause | Verdict | Basis |
|---|---|---|
| C1 | FAIL | Recall 5/200; remains generated below the frozen detector's detectability. |
| C2 direct | PASS | Recall 197/200. |
| C2 pooled | FAIL | Pooling C2b gives 214/400. |
| C2b | FAIL | Aggregate gap-magnitude acceptance is met, but monitor recall is 17/200. |
| C3 | FAIL | Recall 12/200; remains generated below the frozen detector's detectability. |
| C4 | FAIL | AUC veto removed and firing now measured, but recall is 15/200. |
| C5 | FAIL | Control CUSUM calibration met its target; recall is 7/200. |
| C6 | FAIL | Recall is 33/200. |
| C7 | FAIL / documented negative | Reported recall is 57/200; not a gated cause and remains a documented negative. |

**Overall Step 5: FAIL.** The full panel still fails the registered recall, C0 false-positive, off-diagonal confusion, and minimum per-class F1 bars. The run output, not this report, is the authority for each result.

## What this does not fix

- C1 remains undetectable as generated.
- C3 remains undetectable as generated.
- C7 remains a documented negative; this does not change the Step 4.9 finding.
- The full-CAFR equal-churn claim remains untested.

## Verification

- `python -m pytest -q` before panel rebuild: 190 passed, 4 warnings.
- `python experiments/build_synth_panel.py --seed 42 --c7-skus 40`: completed successfully.
- `python experiments/step5_monitor_gates.py --per-cause 200 --seed 42`: completed on the full panel; Step 5 remained FAIL.
- `python experiments/step6_c2_gated.py --stage compare`: unchanged: C2 cost +71.48 [-20.01, 168.63], CSL +6.14 pp [4.20, 8.30], C0 0/40.
- `python -m pytest -q` after rerun: 191 passed, 4 warnings.

The protected Step 6 outcome figures are unchanged. Its reliability-gate fit metadata was not byte-identical: the regenerated Step 5 readings yielded 251 training and 77 validation rows and selected `tau = 0.242753`, versus the prior 246/75 rows and `tau = 0.2446`. No Step 6 code or threshold was adjusted; the specified paired inventory outcomes and C0 comparison reproduced exactly.
