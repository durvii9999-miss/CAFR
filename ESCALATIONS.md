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

**What the repo does about these.** `cafr/configs/ruf.yaml` sets burn-in 24 / evaluated 36
and carries H-1's reasoning inline. That is the *handoff's recommendation*, wired in so the
code runs — not a resolution. **H-1 still needs sign-off**, because it fixes the window
every T1 number is computed over.

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
