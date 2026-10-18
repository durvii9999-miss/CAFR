# ESCALATIONS — things this repo will NOT decide for you

**Read this before you run anything.** Every item here is a question the code refuses to
answer on its own. They are team decisions, written down so they cannot get lost.

Two rules from the handoff govern all of them:

> **If a gate fails, the specification is wrong, not the gate.** (Handoff §14 rule 4)
>
> **Silently resolving H-1 … H-5 is on the "do not do" list.** (Handoff §13, item 23)

So: when something below bites, **report it**. Do not widen a tolerance and do not pick
the convenient reading.

---

## Part A — Carried over from the handoff (§17.3). Not new.

These five come from the dataset verification. Rev 2 does not contain them.

| ID | The problem | Handoff's recommendation | Who decides |
|---|---|---|---|
| **H-1** | RUF ships **60 periods, not 84**. Rev 2 §14.1 assumed 24 burn-in + 60 evaluated. Both rungs of its ladder now fail. What are burn-in / evaluated / `W_occ`? | burn-in 24, evaluated 36, `W_occ = 36`; report C3's recall on the SKUs that fit | **The team.** It changes the evaluated window that goes into T1 |
| **H-2** | Rev 2 §6.1's `W_occ`: must the window be *full*, or does the minimum-evidence floor govern? | **Reading B** — use the available history, floor on the nonzero count, log partial windows | The team (a reading of the spec) |
| **H-3** | RUF records no stock-outs, so `fit_on = "uncensored"` is impossible there | **Accept and state as a limitation.** On RUF, `fit_on` is forced to `"observed"` — it is not a preference | The team (reporting) |
| **H-4** | Use the pre-filtered 250-item M5 panel instead of Kaggle M5? | **Adopt it** — removes the account dependency and the filtering step | The team (changes E5's provenance) |
| **H-5** | **26.1 % of RUF SKUs (1,304 / 5,000)** have fewer than 2 nonzero periods at burn-in 24 → CV² undefined → `sb_cell_init = "dead"` → **C6 can never fire for them** | **Keep them.** Label them `dead`, let them abstain, report the count in T1. **Do not drop them** — dropping SKUs to suit the method is a reportable integrity failure | The team (reporting) |

**RESOLVED 2026-10-17 — team sign-off (the project owner).** All five were accepted as the
handoff recommends:

| ID | Decision |
|---|---|
| **H-1** | **burn-in 24, evaluated 36, `W_occ = 36`.** Report C3 recall on the SKUs where the horizon fits, alongside the pooled figure. The evaluated window (36) goes into the methods section and T1. |
| **H-2** | **Reading B** — window = `min(W_occ, available history)`, abstain below 10 nonzero periods. Log full vs partial per decision point and report it. |
| **H-3** | **Accepted as a limitation.** On RUF `fit_on = "observed"` is forced. Record in T7. The C-guard's value is measured on the labelled synthetic panel (A6a). |
| **H-4** | **Adopt the pre-filtered 250-item M5 panel** in `dataset_release/m5_full38_panel/`. No Kaggle account. Record the provenance change for E5. |
| **H-5** | **Keep all 5,000 SKUs.** The 1,304 `dead` SKUs abstain by construction and are reported in T1. Report per-cause recall pooled and with `dead` excluded. Do not drop them. |

`cafr/configs/ruf.yaml` already sets burn-in 24 / evaluated 36 with H-1's reasoning inline;
that now matches the signed-off decision. H-2 and H-5 are implementation/reporting checks
to confirm when their code paths are next exercised.

---

## Part B — Open decisions from Rev 2 (§17.4)

| ID | Question | Status | Blocks |
|---|---|---|---|
| **OD-1** | Six causes or seven in the contribution sentence? | OPEN — build all seven regardless | Nothing |
| **OD-5** | What the `2601.21844` generator emits | OPEN — but the generator is now ours, so it no longer blocks Step 4 | Nothing |
| **OD-6** | RAF licence — use or drop | OPEN — **Dr. V. K. Chawla** | Nothing. **Do not download RAF until a licence is confirmed in writing**; three panels suffice |
| **OD-8** | Source of the metric formulas | OPEN — adopt `2609.13840`'s definitions and record the source of each formula | Step 6+ |

**Frozen — do not reopen:** OD-2, OD-3, OD-4, OD-7, OD-9. There is no OD-10.

---

## Part C — Raised while building Steps 0–2

These are **new**. They came out of writing the code and the tests, not out of the
documents. Each one is a place where the specification and the machinery disagree, or where
the specification is silently unsatisfiable.

### E-1 · The protection interval is declared `R = L + 1 = 3` but the timing implies `L`

- Rev 2 §15.6 and handoff §5.5 both fix **`L = 2`, `R = L + 1 = 3`**.
- The simulator's timing (an order placed at the end of `t` arrives at the start of
  `t + L`) covers **`L` = 2** periods of demand, not 3.
- Both conventions are implemented behind `sim.protection_convention`.
- `tests/test_simulator_sanity.py` **measures both** and writes the verdict to
  `results/step1_protection_diagnostic.json` rather than picking one quietly.

**Why it matters.** `R` sets the length of the aggregate the order-up-to quantile is taken
over. Get it wrong and the safety stock is wrong by roughly the ratio of the two intervals —
the same failure mode as the 42.3 % bug the project is built to avoid.

**Decision needed:** which convention is the project's?

### E-2 · Gate 2.1 has no external referent for mSBA and mTSB

- Gate 2.1 validates every forecaster against `statsforecast` at relative error ≤ 1e-6.
- **`statsforecast` implements neither mSBA nor mTSB.** There is nothing to validate against.
- What was done instead: both were given definitions with **exact algebraic reductions**
  (mSBA → SBA at `alpha_z = alpha_p`; mTSB → TSB at `beta = 0`, `phi = 1`), and the
  *reductions* are asserted at ≤ 1e-12 against the validated parents. Anyone can then
  re-derive them from two forecasters that *are* externally validated.

**This is a substitute, not an equivalent.** A wrong mSBA that still reduces correctly would
pass. Escalate so the team can accept the substitute knowingly.

### E-3 · `residual_sigma` returns exactly 0.0 on most intermittent series

- Rev 2 §15.6 makes `sigma_hat` part of the forecaster interface, and T2 has a `sigma_hat`
  column.
- The MAD-based `residual_sigma` returns **identically 0.0 whenever more than half the
  residuals are equal** — which any series with > 50 % zeros always produces.
- So **T2's `sigma_hat` column would be all zeros for the majority of the panel.**
- Gate 2.2 is unaffected: C2's coverage test uses empirical residual quantiles, not
  `sigma_hat`.

**Decision needed:** adopt a zero-inflated scale estimator, or report `sigma_hat` on the
nonzero residuals only **and say so in the paper**. Either is defensible; silently shipping a
column of zeros is not.

### E-4 · Gate 1.3's `0.50 ± 0.03` has a floor that is not 0.50

- Gate 1.3 checks that with **zero** safety stock the achieved CSL lands near 0.50.
- But when demand is intermittent, `P(A_R ≤ R·μ̂) ≥ (1 − p)^R`. At `p = 0.05, R = 3` that
  floor is **0.857** — no safety stock at all, and the service level is already 0.857.
- The test therefore reports **two** numbers: all SKUs, and dense SKUs only (`p ≥ 0.35`),
  and writes both to `results/step1_gate_1_3.json`.

**This is not a reason to widen the tolerance.** It is the reason gate 1.3 may need to be
*restated* — as a statement about dense SKUs, with the sparse-SKU atom reported separately.

### E-5 · The order-up-to level did not read the forecast — **RESOLVED 2026-10-17**

- `S = R*mu_hat + Q_alpha(aggregates - R*mu_hat)` and the `R*mu_hat` cancelled exactly,
  so arms (a), (b), (c) were the same policy.
- **Resolution:** docs/17 §5 option 1, the residual-bootstrap predictive quantile
  `S = R*mu_hat + Q_alpha(sum of R-period residual sums)`. `aggregate_order_up_to` now
  requires `residuals` and raises without them.
- Full record: `docs/17_policy_forecast_channel_severed.md` §9. Tests:
  `tests/test_policy_forecast_link.py`.

### E-6 · Arm (a) selected on one period of history — **RESOLVED 2026-10-17**

- The level function is called from `t = 0`, but arm (a) resolved its method on the
  first call. `n_origins` was 0, so every SKU silently took the SB fallback and arm (a)
  remained a copy of arm (b) even after E-5 was fixed.
- **Resolution:** selection waits until `len(y_observed) >= burn_in` and reads
  `y_observed[:burn_in]` only. Locked by
  `test_selection_waits_for_the_burn_in_to_be_complete`.

### E-7 · Gate 4.9 — measured 2026-10-17: constraint 7 now PASSES, constraint 6 still FAILS, and it fails without the injection

- Two of constraint 6's three clauses were unusable **as written**: the per-window
  RMSSE comparison is P(pass) = 0.5 under the null, and the coverage clause's object was
  ambiguous between the policy's `S` and the forecaster's quantile.
- **Restated before the reporting run** (docs/16 §7): **(1)** pooled per-SKU RMSSE with a
  pre-registered 10 % band; **(2)** coverage against the forecaster's **unscaled**
  predictive quantile; **(3)** constraint 7 calibrated over a pre-declared scale grid
  `(0.60, 0.45, 0.30, 0.15, 0.05)`, mildest passing scale first; **(4)** constraint 5's
  high-ADI tail flagged and reported separately, C2 not redefined, no SKUs dropped.
- **Result (40 SKUs, 2,040 windows, seed 42):** constraint 7 **PASSES** at scale 0.15
  (0.8265). Constraint 6 **FAILS** at 0.3240 — RMSSE pooled 0.6250, coverage 0.5441,
  `z_b` 0.9877. **Gate 4.9: FAIL.**
- **The decisive diagnostic:** the same pipeline at scale 1.0 (no injection) scores
  constraint 6 at **0.3201**, with identical RMSSE and `z_b`. Constraint 6 therefore does
  not measure the injection at all: under `observation_model = demand` the forecaster
  never sees the stock-outs C7 causes. It measures the baseline forecaster's absolute
  calibration, and on this sample that is 0.32.
- **Consequence:** §15.2's stop rule applies — **report C7 as not separately
  identifiable in this simulator** and publish the six-cause taxonomy. Do not lower the
  bars. A difference-in-difference redesign of constraint 6 is a team decision, not
  implemented.
- Full record: `docs/16_step4_gate_4_9_findings.md` §7 (rules) and §8 (result).
  Code: `cafr/data/synth/c7_constraints.py`; control:
  `experiments/c7_control_diagnostic.py`; tests: `tests/test_step4_regressions.py`.

### E-8 · Gate 5 — measured 2026-10-18 on the full pre-registered panel: **FAIL**, and the cause is specification/generator, not thresholds

- **Run:** `step5_monitor_gates.py --per-cause 200 --seed 42`, 1,800 SKUs, 0 errors,
  869.5 s, `base.yaml` frozen including `W_sz = 24`. Seed 42.
- **Result:** **5.1 FAIL** (C1 0.030, C2 0.522 pooled / **0.955 direct**, C3 0.055,
  C4 **0.000**, C5 0.110, C6 0.145; all with Wilson intervals in docs/18 §1).
  **5.2 FAIL** (FP 0.450 [0.383, 0.519], 90/200; pre-injection flag 0.630).
  **5.3 FAIL** — but every failing row is **abstention to C0** (C1 0.545, C4 0.560,
  C6 0.480); there is almost no cross-cause confusion (largest true confusion C5→C6
  0.230). **5.4 PASS** (0.011). **5.5 PASS.** **5.6 FAIL** (C4 0.000 on 72 budget-cleared
  SKUs; C5 0.110 on 200). **OVERALL FAIL.**
- **The decisive measurement — firing rate on own cause vs C0 control:** C1 0.030 vs
  0.025 (z = +0.31), C2 0.540 vs 0.145 (**z = +9.28**), C3 0.055 vs 0.005 (z = +2.93),
  C4 0.000 vs 0.000, C5 0.110 vs 0.085 (z = +0.84), C6 0.180 vs 0.195 (**z = −0.38**).
  **40 % of the panel (717/1,800) never triggers any rule.** Only C2 separates.
- **Three defects are structural, not tuning:**
  1. **C4 cannot fire under §6.3 as written.** Its `AUC ≥ 0.65` conjunct (occurrence AUC)
     is met on **1.6 % of all SKUs and 0 % of C4 SKUs** (panel median 0.44); all ten SKUs
     clearing the dispersion + evidence conditions are blocked by it. The injection is
     strong (`k → k·{0.25, 4.0}`).
  2. **C5's occurrence CUSUM alarms on 99.0 % of controls** and its step-vs-drift BIC
     passes on 3.9 %. The alarm is saturated; `k`/`h` were co-designed for the size
     sub-process.
  3. **The C2b panel is the control panel with a different label.** Its coverage gap is
     median +0.033 (q10 −0.10, q90 +0.10), `p < 0.05` on 9.5 % vs 14.5 % of controls,
     because `inject_C2b` never modifies the demand or the forecaster; the miscalibration
     exists only in generator metadata. Recall 0.090 is therefore **not** a detector
     result.
- **C1 and C3 are undetectable injections:** C1's expected `z_b = 3.48φ ≤ 1.39` even with
  a frozen forecast, and its adaptive crossing rate (0.000) is below the control's (0.067);
  C3's median `Δp` is +0.031, within one sampling sd (≈0.048) and well below `tau_occ`.
- **C7 (not gated):** recall 0.280, firing 0.305 vs 0.070 on control (z = +6.02). Does not
  reverse E-7 — constraint 6 was a forecast-diagnostic test; this is a policy-outcome
  statistic. On RUF it remains unidentifiable.
- **Nothing was changed to pass a gate.** No threshold, no frozen parameter, no edit to
  `attribution/rules.py`.
- **Decisions requested (smallest defensible correction, per docs/18 §5):**
  1. **C4** — approve a §6.3 revision of the `AUC ≥ 0.65` conjunct (replace with the
     spec's own C3/C4 discriminator, or `|occ_gap| ≤ tau_occ`). Until then C4 recall is 0
     by construction.
  2. **C5** — approve calibrating `h` on the train control split (report the resulting
     ARL0), or a CUSUM reset after alarm.
  3. **C2b** — approve fixing `inject_C2b` to apply the error to the forecaster's interval.
  4. **C1/C3** — approve reporting as undetectable, or a new panel with a detectability
     budget (new experiment).
  5. **C6** — approve a generator constraint that the injected cell must cross.
- **Consequence if none is approved:** report C2 as the single validated cause and C1, C3,
  C4, C5, C6, C7 as negatives. The paper's claim becomes "attribution works where the cause
  is identifiable, and the identifiability boundary is measured" — **not** a six-cause
  taxonomy. Do not lower the 0.60 / 0.10 / 0.40 bars.
- Full record: `docs/18_step5_findings.md`. Result files: `results/step5/`.

---

## How to escalate, in practice

1. Say which item, by its ID.
2. Say what the code measured — quote the number, do not summarise it.
3. Say what you changed **nothing** to work around it.
4. Ask for the decision.

**Never** make a gate pass by editing the gate. `Gate 4.9 is never relaxed, full stop.`

---

## Status of this file

Parts A and B are copied from the handoff and are stable.
**Part C is live** — it grows as Steps 3–13 get built. Add to it; do not delete from it.
