# Step 5 findings — the attribution gates, measured on the pre-registered panel

**Run:** `experiments/step5_monitor_gates.py --per-cause 200 --seed 42`
= 1,800 SKUs (C0 control 200, C1–C7 200 each, C2b variant 200), T = 200, `tau_inj` = 120.
**Completed:** 2026-10-18, exit 0, **1,800 SKUs ran, 0 errors**, 869.5 s.
**Config:** `base.yaml` as frozen, including `monitor.windows.W_sz: 24`.
**Result files:** `results/step5/{step5_readings.parquet, step5_summary.json,
step5_confusion.parquet, step5_priority_sensitivity.parquet}`.

This document reports what the monitor actually did. It changes no threshold and no
frozen parameter. Where a gate fails, the finding is the failure.

---

## 1. Gate results

### 5.1 Per-cause recall (bar 0.60 for C1–C6)

| Cause | Pooled recall [95 % Wilson] | Budget-restricted | Verdict |
|---|---|---|---|
| C1 | 0.030 [0.014, 0.064] (6/200) | 0.050 [0.023, 0.105] (6/120) | **FAIL** |
| C2 — pooled C2 + C2b | 0.522 [0.474, 0.571] (209/400) | 0.522 (209/400) | **FAIL as pooled** |
| C2 — direct panel only | **0.955 [0.917, 0.976] (191/200)** | — | **PASS** |
| C2b — variant panel | 0.090 [0.058, 0.138] (18/200) | — | **FAIL** |
| C3 | 0.055 [0.031, 0.096] (11/200) | 0.076 [0.043, 0.132] (11/144) | **FAIL** |
| C4 | 0.000 [0.000, 0.019] (0/200) | 0.000 [0.000, 0.090] (0/39) | **FAIL** |
| C5 | 0.110 [0.074, 0.161] (22/200) | 0.324 [0.224, 0.442] (22/68) | **FAIL** |
| C6 | 0.145 [0.103, 0.200] (29/200) | 0.158 [0.112, 0.217] (29/184) | **FAIL** |
| C7 (reported, not gated) | 0.280 [0.222, 0.346] (56/200) | 0.286 (56/196) | — |

**The C2 denominator.** `truth_for_gate` maps the C2b variant into C2, so the pooled figure
is 209/400. The C2b panel is a *variant* required by §15.2 constraint 5 and is reported
separately (§15.3: "evaluate separately … do not fold it into the macro-F1"). On its own
panel the direct C2 detector recovers **191/200** SKUs. Both numbers are stated; the gate
is reported as failing on the pooled denominator, which is the conservative reading.
Gate 5.1 fails either way, because C1, C3, C4, C5 and C6 fail.

### 5.2 False positives on the C0 control panel (bar ≤ 0.10)

**0.450 [0.383, 0.519] (90/200) — FAIL.** What the controls were called:

| Called | n/200 | Rate |
|---|---|---|
| C0 (abstain, correct) | 110 | 0.550 |
| C2 | 29 | 0.145 |
| C6 | 27 | 0.135 |
| C5 | 15 | 0.075 |
| C7 | 13 | 0.065 |
| C1 | 5 | 0.025 |
| C3 | 1 | 0.005 |

Pre-injection flag rate (diagnostic): **0.630 (126/200)**. Per-check flag rate:
1,258/8,800 = **0.143**.

### 5.3 Row-normalised confusion (bar: no off-diagonal > 0.40)

| Truth | Max off-diagonal | Goes to | Verdict |
|---|---|---|---|
| C0 | 0.145 | C2 | pass |
| C1 | 0.545 | **C0 (abstain)** | **FAIL** |
| C2 | 0.335 (truth_for_gate) / 0.040 (true_cause) | C0 / C5 | pass |
| C3 | 0.360 | C0 | pass |
| C4 | 0.560 | **C0 (abstain)** | **FAIL** |
| C5 | 0.230 | C6 | pass |
| C6 | 0.480 | **C0 (abstain)** | **FAIL** |
| C7 | 0.390 | C0 | pass |

**Every 5.3 failure is abstention, not misattribution.** C1, C4 and C6 are labelled C0 at
0.55, 0.56 and 0.48. There is almost no cross-cause confusion anywhere in the matrix: the
largest true confusion is C5 → C6 at 0.230. The monitor is not mixing causes up; it is
failing to name them.

Full row-normalised table:

```
attributed_cause     C0     C1     C2     C3   C4     C5     C6     C7
truth_for_gate
C0                0.550  0.025  0.145  0.005  0.0  0.075  0.135  0.065
C1                0.545  0.030  0.140  0.015  0.0  0.085  0.130  0.055
C2                0.335  0.000  0.522  0.002  0.0  0.060  0.060  0.020
C3                0.360  0.015  0.255  0.055  0.0  0.100  0.170  0.045
C4                0.560  0.015  0.145  0.005  0.0  0.100  0.150  0.025
C5                0.200  0.015  0.160  0.200  0.0  0.110  0.230  0.085
C6                0.480  0.010  0.100  0.110  0.0  0.070  0.145  0.085
C7                0.390  0.010  0.130  0.005  0.0  0.075  0.110  0.280
```

### 5.4 Priority-order sensitivity (bar ≤ 0.10) — **PASS**

Max |macro-F1 difference| over 10 permutations: **0.011**. Base macro-F1 over C1–C7:
**0.150**. The ordering is not what is deciding the result; the detectors are.

### 5.5 Outputs produced — **PASS** (confusion matrix and statistic distributions written).

### 5.6 Budget-restricted recall (generator's own budget flags)

- C4: **0.000 [0.000, 0.051] (0/72)** — FAIL
- C5: **0.110 [0.074, 0.161] (22/200)** — FAIL

### Overall: **FAIL**

---

## 2. Why it failed — the monitor does not discriminate

The decisive diagnostic is not recall but **how often each rule fires on a SKU carrying
its own cause versus on a C0 control**. A detector that fires equally often on controls
and on injections carries no information regardless of its threshold.

| Cause | Firing rate on own cause | Firing rate on C0 | Two-proportion z |
|---|---|---|---|
| C1 | 0.030 | 0.025 | **+0.31** |
| C2 | 0.540 (direct: 0.955) | 0.145 | **+9.28** |
| C3 | 0.055 | 0.005 | **+2.93** |
| C4 | 0.000 | 0.000 | — |
| C5 | 0.110 | 0.085 | **+0.84** |
| C6 | 0.180 | 0.195 | **−0.38** |
| C7 (not gated) | 0.305 | 0.070 | **+6.02** |

Only C2 (strongly), C7 (strongly) and C3 (weakly) separate from the control. C1, C5 and
C6 fire at the control's base rate; C4 never fires at all.

**40 % of the panel never triggers any rule.** 717/1,800 SKUs have an empty firing set at
the final reading: C0 107, C1 105, C2 128 (all C2b), C3 64, C4 109, C5 35, C6 94, C7 75.
The confidence gate is not the binding constraint where a rule does fire: among SKUs with
one cause firing, only 3.4 % abstain; with two, 1.9 %.

### Which conjunct is unreachable

Each rule is a conjunction. The table below shows which conjunct never passes, measured
over the whole panel.

| Rule | Conjunct | Measured reachability |
|---|---|---|
| C1 | `|z_b| > 1.96` | 7.4 % of all SKUs |
| C1 | drift BIC < step BIC | 69.6 % |
| C4 | two-window `max(r,1/r) > 1.5` | **10 / 1,800** |
| C4 | `AUC ≥ 0.65` (occurrence) | **1.6 % overall; 0 % of C4 SKUs** |
| C5 | occurrence CUSUM alarm | **98.4 % overall; 99.0 % of C0** |
| C5 | occurrence step BIC < drift BIC | **3.9 %** |
| C6 | bootstrap stability ≥ 0.90 | 14.2 % |
| C7 | `csl_gap < −0.10` | C7 30.5 % vs C0 7.0 % |

Two of these are *structural*, not power problems:

- **C4's `AUC ≥ 0.65` guard is unsatisfiable on this panel.** The occurrence AUC is
  estimated on only 50.6 % of SKUs (the evidence floor), and its median is **0.44**
  (10th–90th percentile 0.32–0.58). It reaches 0.65 on **1.6 %** of all SKUs and **0 %** of
  C4 SKUs. The ten SKUs that clear C4's dispersion and evidence conditions are all blocked
  by this single clause. The AUC of an i.i.d. Bernoulli occurrence process against a
  smoothed rate forecast is ≈ 0.5 by construction; a 0.65 bar was written for a case where
  the occurrence model is *worse* than a constant predictor, which the panel never
  produces. **C4 cannot fire under §6.3 as written.**
- **C5's occurrence CUSUM is saturated.** It alarms on 99.0 % of controls, so the alarm no
  longer carries information, and the step-vs-drift BIC that must arbitrate it passes on
  3.9 % of SKUs. The size CUSUM is better (alarm 0.236 overall, 0.200 on C0) but still not
  calibrated to the 10 % false-alarm bar.

---

## 3. Per-cause verdicts: injection design vs detector

### C1 — **unidentifiable injection** (not a detector limitation)

The injection applies a linear ramp φ ∈ {±0.2, ±0.4} to `μ_z` over 40 periods. With a
*frozen* forecast the expected bias statistic is `z_b = φ·√k·√n ≈ 3.48 φ`, i.e. 0.70–1.39 —
below the 1.96 bar even with an oracle decoder. With the live adaptive forecaster the
tracking model absorbs what remains. Measured `|z_b| > 1.96` crossing rates (6 SKUs per
cause, adaptive vs frozen on the same series):

| Truth | adaptive (monitor sees this) | frozen (oracle) |
|---|---|---|
| C1 | 0.000 | 0.017 |
| C0 | 0.067 | 0.117 |

The C1 crossing rate is *below* the control's. No threshold change recovers a signal the
statistic does not contain.

### C2 — **supported** (direct panel); C2b is an **injection-design failure**

Direct C2: recall **0.955**, firing 0.540 vs 0.145 on control (z = +9.28), coverage gap
median **−0.567** vs +0.000 on control. The detector works.

C2b: recall **0.090**; coverage gap median **+0.033**, q10 −0.100, q90 +0.100; binomial
`p < 0.05` on 9.5 % of SKUs versus 14.5 % of controls. **The C2b panel is statistically
indistinguishable from the control.** `inject_C2b` draws the baseline demand and leaves it
untouched (`delta, z = _baseline_draw(...)`, returned unmodified); the mis-calibration
lives only in the generator's metadata. The monitored pipeline therefore never experiences
a miscalibrated interval. This is a generator defect, not a detector failure, and the
detector could not detect it with any threshold.

### C3 — **unidentifiable injection as generated** (small residual signal)

Median injected `Δp` is **+0.031** (q10 −0.167, q90 +0.188). One sampling sd of an
occurrence rate over 36 periods near p = 0.2 is ≈ 0.048 and `tau_occ` is 0.10, so the
injected step is within one standard error of its own estimate for most SKUs. The
forecaster's occurrence model also tracks a step in ~5 periods, leaving ~5/36 of it over
`W_occ = 36`. The rule is nonetheless highly specific — fires 0.055 on C3 vs 0.005 on C0
(z = +2.93) — but that is 11 SKUs, far below 0.60.

### C4 — **detector/specification limitation** (strong injection, unsatisfiable rule)

The injection changes the Gamma shape `k → k·{0.25, 4.0}` at fixed `μ_z`, a 4× dispersion
change; `r` separates from control (C4 median 0.62, q90 3.14 vs C0 median 0.97, q90 1.55).
The failure is mechanical:

- the generator's own budget flag marks **72/200 C4 SKUs** as clearing constraint 9;
- the monitor's evidence floor at the final reading is met on **27/200 (13.5 %)**;
- both `W_sz` windows are in band on **5/200**;
- the occurrence-AUC guard then blocks **all five**.

The rule fired on **0 of 1,800 SKUs**. No threshold in `rules.py` fixed the AUC guard;
only a specification change can.

### C5 — **detector calibration limitation** (strong injection)

`Δp` median **0.204**; every C5 SKU clears the generator's detectability budget
(`c5_budget_cleared = 1.0`). The occurrence CUSUM alarms on 99.0 % of controls — at
`k = 0.25`, `h = 4.0` the nominal `ARL0 ≈ exp(2kh)/(2k²) ≈ 59` over a 96-period horizon,
so a per-window control alarm is expected by construction. The step-vs-drift BIC then
passes on 3.9 %. The detector is saturated, not underpowered: strong injections and
controls alike alarm.

### C6 — **both**

The cell moves from init for 47.1 % of SKUs, but the bootstrap-stability guard passes on
only 14.2 %, and the rule fires on 0.195 of controls versus 0.180 of C6 SKUs (z = −0.38).
ADI estimated on 30 periods of an intermittent series is too noisy to carry the cell.
The generator does not require the cell to cross either.

### C7 — **documented negative for the real panel; the synthetic policy detector carries signal**

Not gated. Recall 0.280, and the firing rate separates strongly (0.305 on C7 vs 0.070 on
C0, z = +6.02). This does not reverse Step 4.9: constraint 6 was a *forecast-diagnostic*
test, and it failed because under `observation_model: demand` the forecaster never sees the
stock-outs C7 causes. The monitor's CSL-gap statistic is a *policy-outcome* statistic, and
on this synthetic panel it does see the scaled safety factor. On RUF (no stock-out record,
H-3) it remains unidentifiable. Report C7 as a documented negative with this nuance; it is
still far below the 0.60 bar.

---

## 4. Restatements made to the monitor before this run

Each is a place where the literal spec text was impossible to implement or demonstrably
wrong under measurement, recorded so the paper can state its own rules.

| # | Spec text | What the code does | Why |
|---|---|---|---|
| R-1 | §6.3 C1: bias of size residuals | `e_sz = y − mu_nonzero_hat(t−1)`, nonzero periods only | `mu_nonzero` was not plumbed through the arm state |
| R-2 | §6.3 C2: interval coverage | one-period conformal interval at `c2_interval_alpha` | the direct C2 injection clears a 90 % interval by construction |
| R-3 | §6.3 C3: occurrence calibration | signed occurrence gap vs `tau_occ = 0.10`; AUC reported as diagnostic | the original `FZ > 0.30` measured 1.000 on the C0 control (threshold outside the statistic's support) |
| R-4 | §6.3 C4: dispersion ratio | two-sided `max(r, 1/r) > 1.5` | the injection moves dispersion either way |
| R-5 | §6.3 C4: "sustained over 2 windows" | two adjacent non-overlapping `W_sz` blocks (n ≥ 168 at `W_sz = 24`) | the only reading under which "sustained" is meaningful |
| R-6 | §6.3 C6: "sustained for 3 windows" | three consecutive monitor periods | three non-overlapping `W_class` blocks need 210 > T |
| R-7 | §6.3 C5: CUSUM scale | size CUSUM on trailing realised sd; occurrence CUSUM on `sqrt(p̂(1−p̂))` | first-`W_base` sd alarms on 94 % of pre-injection controls; fitted model `sigma` on 100 % |
| R-8 | §6.3 C5: step-vs-drift | BIC on the same sub-process as the alarm | the size-BIC blocked ~85 % of occurrence alarms on an untouched series |
| R-9 | §6.3 C5: CUSUM inputs | residual aligned to the period the forecast was made | a one-period misalignment leaks forecast movement |
| R-10 | §6.3 guards | a NaN guard passes | NaN is no evidence, and must not veto a cause that fired |
| R-11 | §6.1 detect start | `detect_from = tau_inj` | the monitor cannot see the injection before it occurs |

**Not restated:** `W_sz = 24`, `W_occ = 36`, `tau_conf = 0.35`, `c4_r_threshold = 1.5`,
`cusum_k/h = 0.25/4.0`, all evidence floors, and every gate threshold.

---

## 5. Recommended smallest defensible correction

None of these is a threshold change made to pass a gate. Items 1–2 are specification
defects that make a rule unreachable; item 3 is a generator defect; items 4–5 are
measurement/reporting.

1. **C4 — fix the specification, not the detector.** §6.3's `AUC ≥ 0.65` conjunct is
   unreachable on an i.i.d. Bernoulli panel (measured 1.6 %, 0 % of C4 SKUs). The
   defensible replacement is the C3/C4 discriminator the spec already defines in the next
   paragraph — occurrence MAE on the mis-classified cells — or a calibrated guard
   `|occ_gap| ≤ tau_occ`, which is the object that actually says "occurrence is fine".
   Keeping `AUC ≥ 0.65` guarantees C4 recall 0. **This is a §6.3 revision requiring
   sign-off.**
2. **C5 — calibrate the occurrence CUSUM.** It alarms on 99 % of controls: `k`/`h` were
   co-designed for the size sub-process, not for a Bernoulli residual standardised at
   `sqrt(p̂(1−p̂))`. The defensible route is to measure `ARL0` on the train control split
   and set `h` from it (a spec decision, since `k`/`h` are frozen), or to reset the CUSUM
   after an alarm. Do not lower either bar in `rules.py`.
3. **C2b — fix the generator.** It must apply the modelling error to the object the
   monitor scores (the forecaster's interval), not to generator metadata. As generated,
   the C2b panel is the control panel with a different label. This is a §15.2 change.
4. **C1, C3 — report as undetectable as generated.** The only honest correction is a
   generator redesign with a stated detectability budget (C1: φ large enough that the
   statistic clears 1.96 under a live forecaster; C3: `|Δp|` above the sampling sd over
   `W_occ`). That is a new panel and a new experiment, not a fix to this one.
5. **C6 — state the detector limitation and the generator's missing constraint.** Require
   the injected cell to cross (the analogue of constraint 3 for C5), and/or report C6 on
   the subset where the cell is estimable.

### Impact on CAFR's contribution

One cause is validated on its intended panel (**C2, 0.955**). C2b, C3 and C4 fail at the
generator/specification boundary; C1 and C5 fail as detectable classes under the frozen
parameters; C6 fails on both sides.

The honest claim is therefore **not a six-cause taxonomy**. It is a two-part result:

- CAFR's attribution machinery works where the cause is identifiable, and the identifiability
  boundary is measurable and reportable — which is a stronger, more useful claim than a
  taxonomy that silently assumes all seven causes are visible;
- the co-primary hypothesis H1′ (attribution vs no attribution at equal churn) remains
  testable on the causes that are attributable, with the rest reported as negatives.

The paper must present the detectability analysis as a result: which forecast-error causes
a live adaptive forecaster leaves visible, with the measurement that establishes each
boundary. It must **not** claim six-cause attribution.
