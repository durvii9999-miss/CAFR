# Step 6 findings — C2-only reliability-gated CAFR (seed-42 feasibility experiment)

**Experiment:** `experiments/step6_c2_gated.py --stage compare`
**Panel:** the committed seed-42 synthetic panel; T = 200, `tau_inj` = 120
**Result files:** `results/step6/` (`c2_gate_summary.json`,
`c2_compare_summary.json`, `c2_compare_arms.csv`, `c2_compare_paired.csv`,
`c2_compare_run.log`, plus the smoke and timing outputs)
**Status:** a single-seed feasibility experiment. It does not reopen the Step 5
attribution verdict. The Step 5 overall **FAIL** and **E-8** in `ESCALATIONS.md`
stand unchanged; no Step 5 result file, frozen configuration, or prior result was
modified. The only existing module touched is the opt-in hook in
`cafr/arms/arm_c.py` (§6.1).

---

## 1. Research question and motivation

Step 5 measured the attribution gates on the full panel. In the archived
pre-correction readings used to fit this gate, only **C2 direct** passed its
recall bar (0.955, 191/200); the corrected Step 5 rerun gives 0.985 (197/200).
The overall verdict was FAIL in both cases, and the causes were recorded in
`docs/18_step5_findings.md`, corrected in
`docs/20_correction_round_findings.md`, and carried in E-8. That result leaves a
narrower, testable question:

> When the live monitor makes its first post-injection C2 attribution, can a
> reliability gate trained on the Step 5 readings decide whether to apply the C2
> remedy, and what happens to inventory cost and service relative to the untouched
> baseline?

The experiment is deliberately C2-only and single-decision. It is not a six-cause
claim, not a re-run of Step 5, and not a test of a new attribution rule. The
reliability screen reported earlier was exploratory; this is the first paired
inventory comparison of that policy.

---

## 2. What is injected, gated, repaired, and compared

### 2.1 C2: interval miscalibration, demand untouched

`cafr/data/synth/injections.py::inject_C2` does **not** change the demand draw. It
returns the baseline `delta * z` and records `demand_touched: False`. The
miscalibration is a per-SKU scale `gamma` on the forecaster's reported interval,

```
Q~_alpha = mu_hat + gamma * (Q_alpha - mu_hat)
```

`_solve_gamma` (through `_c2`) bisects on Monte Carlo coverage with target
`tau_cov + margin = 0.10 + 0.02 = 0.12` at `alpha = 0.95`, `R = 3`, and gamma
bounds (0.02, 8.0). The target is not always reachable: **193/200** C2 SKUs clear
0.12; 4 land in [0.10, 0.12); 3 fall below 0.10 (minimum 0.0561). All seven
shortfalls hit the 0.02 gamma floor and are recorded with
`gap_clears_with_margin: False`. In the 40-SKU test group, 2 are in [0.10, 0.12)
and none are below 0.10. All 200 C2 SKUs have `gamma < 1`: range
**0.020–0.560**, median **0.453**; the 40 C2 test SKUs span **0.020–0.556**,
median **0.459**. The generator's 95 % coverage-gap median is **−0.120**. The live monitor is Step 5's
C2 detector, which checks a **90 %** conformal interval (`c2_interval_alpha = 0.90`)
scaled by the same `gamma`; on the pre-correction readings used to fit this gate,
the Step 5 measurement of that detector was C2 direct recall **0.955** (191/200);
the corrected rerun gives **0.985** (197/200). The two coverage objects (the
generator's 95 % solve and the monitor's 90 % check) are not the same statistic
and are not equated here.

### 2.2 The reliability gate

The gate is trained only on Step 5 training-split rows with
`detected == True` and `attributed_cause == "C2"`. The label is
`true_cause == "C2"`. Undetected final-reading rows and C2b are excluded.

**Training rows:** 246 (115 true C2, 131 false C2 attributions).
**Validation rows:** 75 (39 true C2, 36 non-C2).
**Test candidates:** 81 (37 true C2, 44 non-C2).

The nine features are the reading at the decision event:

| Feature | Definition |
|---|---|
| `margin_C2` | C2 rule margin |
| `n_fired` | number of rules that fired |
| `ev_c2` | C2 evidence floor met |
| `window_full` | monitor window is full |
| `abs_gap` | `|c2_c_gap|` |
| `neglog10_p` | `-log10(clip(c2_p_val, 1e-12, 1))` |
| `abs_z_b` | `|c1_z_b|` |
| `r_strength` | `max(r, 1/r)` from `c4_r` |
| `confidence` | attribution confidence |

Pipeline: `SimpleImputer(median)` → `StandardScaler` →
`LogisticRegression(max_iter=3000, C=1.0)`.

**Threshold selection.** The lowest validation threshold with precision ≥ 0.90 and
at least 20 admitted true-C2 validation SKUs. The selected threshold is
**τ = 0.2446**. On validation it admits 41 rows, of which 39 are true C2
(precision **0.951**). The test split is never used for selection. If no threshold
had qualified, the rule-only variant (act on every detected C2 attribution) would
have been used; it was not needed.

### 2.3 The R2 remedy

The policy uses `cafr/arms/arm_c.py` with an optional
`c2_scale_provider = f(t) -> scale`. The default is `None`, which preserves every
earlier experiment. The remedy is one assignment:

```
scale(t+1) := 1.0
```

So `aggregate_order_up_to` computes

```
S = R * mu_hat + 1.0 * Q_alpha(R-period sums of signed residuals)
```

The point forecast `mu_hat` is not touched; the monitor sees the same scale the
policy uses. The check in `tests/test_c2_gated.py` compares the per-period `mu`
traces of the two arms and finds them identical. This is **not**
`apply_r2_conformal`; that function computes an absolute-residual quantile, which
is not the policy's signed aggregate quantile (see the Step 6 plan and
`cafr/remedies/forecast_remedies.py`).

Because the safety term is a **signed** residual quantile, `gamma < 1` can move
`S` either up or down. Resetting to 1.0 restores nominal calibration; it is not a
one-way safety-stock increase.

### 2.4 Baseline comparison

Both arms use the same pooled forecaster, the same demand series, the same
simulator configuration, and the same seed. The baseline fixes
`c2_scale_provider(t) = gamma` for the whole run, exactly as Step 5 did. The gated
arm starts at `gamma` and switches to `1.0` only if the single decision is R2.
All comparisons are paired by SKU.

---

## 3. Design

### 3.1 Panel and split

The panel has 1,800 SKUs: C0 control 200, C1–C7 200 each, and C2b 200. C2b is
excluded everywhere. T = 200, `tau_inj` = 120, burn-in 24, seed 42.

The split is the deterministic per-true-cause split used by the reliability
screening: sort by `sku_id`, then cycle
`train, train, train, val, test`. That gives 120 train, 40 validation, and 40 test
SKUs per cause, disjoint by construction.

The 80-SKU comparison uses **all 40 test C2 SKUs and all 40 test C0 SKUs**. No
training or validation SKU enters the outcome comparison.

### 3.2 One decision per SKU

The policy makes at most one decision:

- checkpoints: `t >= burn_in` and every `stride = 4` periods;
- first eligible decision: `t >= tau_inj = 120`;
- at the first **non-C0** attribution, call `decide` once;
- R2 only if the attribution is C2, `ev_c2` is true, and the gate score ≥ τ;
- otherwise R0. The SKU is never queried again (`n_decisions` is at most 1).

The decision is made **after** the order-up-to level for period `t` has been
computed and the monitor has been updated with `y_t`. The new scale is therefore
first used by the call at `t + 1`, which places an order that arrives at
`t + 1 + L` with `L = 2`. There is no look-ahead.

### 3.3 Timing actually observed

| | C2 | C0 |
|---|---|---|
| Test SKUs | 40 | 40 |
| Decisions made | 40 | 19 |
| Decision checkpoint | 38 at t = 120; 2 at t = 124 | 120–192 (median 138) |
| Action start (`action_t`) | 35 at 121; 2 at 125 | 1 at 121 (false action) |
| R0 (`not_C2`) | 3 (all attributed C5 at t = 120) | 12 |
| R0 (`low_reliability`) | 0 | 6 |
| No decision | 0 | 21 (no non-C0 attribution at any checkpoint) |

The three true-C2 non-actions were **attribution misses** (the first non-C0
checkpoint named C5), not threshold misses. The six low-reliability fallbacks were
all true C0. This distinction is visible in `c2_compare_arms.csv` and is corrected
relative to the earlier chat summary.

### 3.4 Evaluation window

The window was fixed before the comparison from the **training** split only:

- `EVAL_START = tau_inj + 12 = 132`;
- evaluated periods: **[132, 200) = 68**;
- 22 complete non-overlapping 3-period cycles (66 periods); the last two periods
  contribute to cost and period service level but not to cycle CSL.

Of 120 detected train C2 SKUs, **95.8 %** have
`detect_t + 1 + lead_time <= 132`; the 5 late detectors are reported in
`c2_gate_summary.json`. In the comparison run all 40 C2 decisions were at t = 120
or 124, so the 37 actions started at 121 or 125 and the first period an affected
order could arrive was 123 or 127 — all before 132. The intervention can therefore
affect the evaluated periods.

### 3.5 Outcomes and inference

- **Cost:** sum of `cost_period` over [132, 200). The simulator uses
  `cost_period = H * on_hand_end + B * unmet`, with `B / H = 19`; `H` is the
  per-SKU `holding_rate`.
- **Cycle CSL:** fraction of complete 3-period cycles with no lost demand.
- **Period SL:** fraction of evaluated periods with no lost demand.
- **Paired bootstrap:** 10,000 resamples over SKUs, percentile 2.5/97.5,
  seed 42, computed in `paired_bootstrap`.

---

## 4. Paired results

Mean per SKU over the 68-period window; differences are gated − baseline.

| Group | n | Baseline cost | Gated cost | Δ cost [95 % CI] | Baseline CSL | Gated CSL | Δ CSL (pp) [95 % CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 40 | 2507.26 | 2578.73 | **+71.48 [−20.01, 168.63]** | 0.8011 | 0.8625 | **+6.14 [4.20, 8.30]** |
| C0 | 40 | 2621.96 | 2621.96 | 0.00 [0.00, 0.00] | 0.8818 | 0.8818 | 0.00 [0.00, 0.00] |
| All | 80 | 2564.61 | 2600.35 | +35.74 [−9.32, 85.77] | 0.8415 | 0.8722 | +3.07 [1.93, 4.38] |

**Cost in the C2 group increased by 2.9 %** (+71.48 on 2507.26), and the 95 %
interval includes zero. No cost saving is demonstrated. The overall cost interval
also includes zero.

**Service improved in the C2 group**: cycle CSL +6.14 pp, interval entirely above
zero. Period SL moved from 0.9246 to 0.9493 (+2.46 pp); the overall period SL
moved from 0.9406 to 0.9529 (+1.23 pp). The C0 group is exactly unchanged on all
40 SKUs (maximum absolute difference 0.0 on both outcomes).

Direction of the C2 effect, per SKU:

| Outcome | Up | Down | Equal |
|---|---:|---:|---:|
| Cost | 21 | 16 | 3 |
| Cycle CSL | 27 | 0 | 13 |

The cost changes range from −886.87 to +1129.07; the CSL changes range from 0.0 to
+0.3182. The mixed cost direction is consistent with resetting a **signed**
quantile: on some SKUs the nominal level is higher, on others lower. CSL did not
fall on any C2 SKU in this seed.

---

## 5. Gate performance and control results

### 5.1 Gate diagnostics on the stored Step 5 readings

“Recall (all true C2)” divides by all true C2 SKUs in the split (120 / 40 / 40),
not just those attributed C2. “Coverage (all rows)” divides by all non-C2b rows in
the split (960 / 320 / 320).

| Split | Candidates | True C2 candidates | Acted | True C2 acted | Precision | Recall (all true C2) | Coverage (all rows) | Coverage (candidates) | True-C0 action rate | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 246 | 115 | 147 | 109 | 0.741 | 0.908 (109/120) | 0.153 | 0.598 | 0.025 | 0.947 |
| Validation | 75 | 39 | 41 | 39 | **0.951** | 0.975 (39/40) | 0.128 | 0.547 | 0.000 | 0.995 |
| Test | 81 | 37 | 45 | 37 | 0.822 | 0.925 (37/40) | 0.141 | 0.556 | 0.025 | 0.979 |

The validation row is the one used to select τ; the test row is untouched. The
test precision of 0.822 includes eight non-C2 actions: one true C0 and seven true
C1–C7 SKUs outside the 80-SKU comparison. It must not be read as the precision on
the C2 + C0 comparison population.

### 5.2 Actions on the 80 compared SKUs

| Metric | Value |
|---|---|
| Acted | **38 / 80** |
| True C2 acted | **37 / 40 = 0.925** |
| True C0 acted | **1 / 40 = 0.025** |
| Precision | **37 / 38 = 0.974** |
| Coverage of all compared SKUs | 38 / 80 = 0.475 |
| C2-attributed rows in the comparison | 44 (37 true C2 + 7 true C0) |
| Acted among C2-attributed | 38 (37 true C2 + 1 true C0) |
| Low-reliability fallbacks | 6 (all true C0) |
| `not_C2` fallbacks | 15 (3 true C2 attributed C5 + 12 other C0 attributions) |
| Live decision matches the stored-reading decision | 77 / 80 |

The one false action was `control:C0:0099` at t = 120 (score 0.630). Its `gamma`
is 1.0, so resetting the scale to 1.0 was a no-op; its cost and CSL differences
are exactly zero. This is why the C0 group is bit-identical even though a control
was acted on. A false action on a genuinely miscalibrated non-C2 cause would not
be a no-op; that case was not in this experiment.

The gate sent six of the seven false C2 attributions on controls to
`low_reliability`; the seventh was the no-op action. Three true C2 SKUs were lost
to attribution (C5 at the first checkpoint), not to the threshold; the
one-decision, no-requery design cannot recover them.

---

## 6. Tests and implementation files

### 6.1 Files changed

| File | Change |
|---|---|
| `cafr/arms/arm_c.py` | optional `c2_scale_provider` on `_ArmCLevelFn` and `arm_c_factory`; default `None` preserves prior behaviour |
| `experiments/step6_c2_gated.py` | new experiment: split, gate, threshold, live policy, paired comparison, diagnostics |
| `tests/test_c2_gated.py` | new: five targeted correctness tests |
| `results/step6/` | new result artifacts (this experiment only) |

No Step 5 result file, `docs/18_step5_findings.md`, `ESCALATIONS.md`, or frozen
configuration was edited.

### 6.2 Tests

| Test | What it checks | Result |
|---|---|---|
| `test_closed_gate_is_transparent` | τ > 1 forces R0; the full simulator log is exactly equal to baseline | pass |
| `test_open_gate_resets_scale_without_touching_the_forecast` | R2 at t = 120, `action_t = 121`, scale 1.0, identical `mu` traces, `S` identical through t = 120 and different at 121 | pass |
| `test_no_look_ahead` | pre-decision logs identical; first cost difference at or after `action_t + L`; first `S` difference at `action_t`; on-hand difference not earlier | pass |
| `test_no_action_without_c2_attribution` | a non-C2 attribution gets R0 `not_C2`, one decision, later checkpoints exist, log identical to baseline | pass |
| `test_fallback_is_logged_and_never_requeried` | a C2 attribution below τ gets R0 `low_reliability`, `n_decisions = 1`, no action | pass |

The first run of the second test failed because the test asserted that
`gamma < 1` always raises `S`. It does not: the scale multiplies a signed
quantile. The assertion was corrected to the verified property (the level changes
at the first call after the decision; the point forecast is unchanged). This is a
substantive property of the remedy, not a cosmetic test fix.

Commands run:

```
python experiments/step6_c2_gated.py --stage gate
python experiments/step6_c2_gated.py --stage smoke      # 4 SKUs, 2.47 s
python experiments/step6_c2_gated.py --stage timing     # 20 SKUs, 13.53 s
python -m pytest tests/test_c2_gated.py -q              # 5 passed
python experiments/step6_c2_gated.py --stage compare    # 80 SKUs
python -m pytest -q                                     # 186 passed in 63.57 s
```

The frozen comparison JSON records `elapsed_s = 57.63`, the wall time of the
comparison run. It is not used as a compute-time estimate. The clean 20-SKU
timing is 13.53 s.

---

## 7. Limitations

1. **Single seed, single synthetic panel.** Seed 42 only. The 40 C2 + 40 C0 test
   SKUs are a small sample; this is a feasibility result, not a confirmation or a
   generalisation.
2. **Cost improvement was not demonstrated.** The C2 point estimate is a 2.9 %
   increase, with a 95 % interval that includes zero. There is no cost-saving claim
   in this document.
3. **Pooled-forecaster exposure.** To reproduce Step 5's live readings, the global
   pooled forecaster is fitted on the first 24 periods of **all 1,800 SKUs**,
   including the 80 test SKUs. The gate model itself is trained only on train-split
   rows, but the forecaster's burn-in exposure to test SKUs is mild leakage; its
   effect on gate robustness is not measured. A leak-free replication (fit the pool
   on train SKUs only, or freeze it at the burn-in) is the clean follow-up.
4. **Attribution misses are out of the gate's reach.** Three true C2 test SKUs were
   attributed C5 at the first checkpoint. The one-decision, no-requery policy cannot
   recover them. This is an attribution limitation, not a threshold result.
5. **The 95 % service target is not reached.** Gated C2 CSL is 0.8625 and C0 is
   0.8818. The scale reset repairs the C2-specific miscalibration, but a residual
   gap is shared with the control and is not repaired here. The 68-period window
   gives only 22 cycles, so cycle CSL is noisy.
6. **The remedy is one per-SKU scale reset.** It does not adapt over time and does
   not address other causes. `C2b` is excluded, and no multi-cause claim is made.
7. **Single cost model.** Holding plus lost-sales penalty, no fixed ordering cost,
   one synthetic DGP. Results are conditional on that simulator.
8. **Exploratory inference.** Two outcomes are reported; the CSL interval is not
   adjusted for multiplicity. The bootstrap resamples 40 C2 SKUs, so the interval
   is coarse.
9. **Global test precision is population-dependent.** The all-cause test precision
   is 0.822; the C2 + C0 comparison precision is 0.974. The difference is seven
   false actions on C1–C7 SKUs that were not part of the inventory comparison. Do
   not quote 0.974 as the full-panel precision.
10. **Step 5 verdict unchanged.** This experiment conditions on C2 attribution. It
    does not change the Step 5 overall FAIL, the E-8 findings, or the status of
    C1, C3, C4, C5, C6, or C7.
11. **Seven C2 injections hit the gamma floor.** 7/200 C2 SKUs cannot reach the
    0.12 margin at `gamma = 0.02`; 4 have |gap| in [0.10, 0.12) and 3 are below
    0.10. Two of the seven are in the 40-SKU test group (both in [0.10, 0.12)).
    The Step 5 recall was measured with them, but those injections are weaker than
    the constraint intends.

---

## 8. Interpretation and next step

In this seed-42 feasibility experiment, **service improved in the C2 group**
(+6.14 pp cycle CSL, 95 % CI [4.20, 8.30]) and **cost increased by 2.9 %**, with a
95 % interval that includes zero (−20.01 to 168.63). The reliability gate acted on
37 of 40 true C2 SKUs, sent six of the seven false C2 attributions on controls to
`low_reliability`, and the seventh was a no-op because the control's scale was
already 1.0.
Non-C2 SKUs were untouched: the 40 C0 SKUs are exactly identical in both arms.

The result is consistent with the remedy doing what it was specified to do —
restoring the nominal C2 scale without changing the point forecast — and with the
service gain being a calibration effect rather than a uniform stock increase. It
does **not** establish a cost saving, a generalisation beyond this seed and panel,
real-world validity, or publication novelty. The 95 % CSL target is still not
reached.

**Recommendation.** Keep the Step 5 FAIL and E-8 as the attribution verdict and
report this as a seed-42 feasibility result, not a confirmation. If the team wants
to pursue C2 repair, the smallest defensible next step is a pre-registered
multi-seed replication with a larger test set and a leak-free pooled forecaster
(fit on train SKUs only), with the objective fixed in advance: a cost-neutral
service improvement (what this experiment found) is a different target from cost
reduction (not found here). No further experiment was run for this document.
