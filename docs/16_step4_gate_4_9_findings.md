# Step 4 findings — gate 4.9 FAILS, and five implementation bugs it exposed

**Status: SETTLED AND MEASURED (2026-10-17).** Constraint 7 now passes at a
pre-registered scale (0.15). Constraint 6 still fails (0.3240) — and fails **identically
with no injection at all** (0.3201), so it measures baseline forecast calibration, not
C7. **Gate 4.9: FAIL.** Per §15.2's stop rule, C7 is reported as not separately
identifiable. No threshold was moved (§7, §8).

Date: 2026-09-30. Branch: `cafr/step3-10-completion`. Evidence: `results/synth/*.json`,
`data/synth/generator_params.parquet`, and the reproduction command at the bottom.

---

## 1. The headline

**Gate 4.9 FAILS honestly.** Measured through the real pipeline (arm (c) via
`rollout_sku`, safety factor scaled to 0.60), 40 C7 SKUs, 2,040 windows:

| quantity | measured | required |
|---|---|---|
| constraint 6 — acceptability pass rate | **0.1961** | ≥ 0.80 |
| constraint 7 — exposure pass rate | **0.6225** | ≥ 0.80 |
| gate 4.9 | **FAIL** | — |

The gate has **not** been relaxed, and the C7 injection has **not** been retuned. §15.2
constraint 6 is a genuine gate; §15.2's own stop rule applies.

But the run also exposed five **implementation** defects, four of which were silently
corrupting numbers. Those are fixed (§3). They are reported separately from the gate
failure because they are different kinds of problem: the defects were mine, the gate
failure is a property of the specification.

---

## 2. Why constraint 6 fails — two of its three clauses are not usable as written

Constraint 6 is the conjunction of three clauses, so the breakdown is what matters:

| clause | pass rate | verdict |
|---|---|---|
| `\|z_b\| ≤ 1.96` | **0.9892** | works, once `z_b` is computed correctly |
| RMSSE below its own baseline | 0.5760 | **ill-posed — a coin flip** |
| coverage within `τ_cov` | 0.4877 | **object is ambiguous** |

### 2a. The RMSSE clause is a coin flip by construction

The clause compares the RMSSE of the 30-period test window against a 30-period
baseline window. Both are estimates of the *same* quantity when nothing has changed, so
under the null P(window ≤ baseline) = 0.5 and the clause is rejected half the time.
Measured: the ratio has median **0.912**, q10 0.475, q90 2.074 — i.e. the forecast error
is *not* rising after the injection, but a per-window test at 80 % cannot see that.

For this clause to pass 80 % of the time on an unchanged series, the median ratio would
have to sit near 0.5, which would require the forecast to *improve* substantially after
a stock-out. **No implementation of "the error did not rise" can satisfy it.** This is a
specification defect and needs a decision:

- compare *pooled* RMSSE across all windows per SKU (one estimate, not 51), which has
  the variance to support an 80 % bar; or
- replace the clause with a tolerance band (RMSSE ≤ (1+δ)·baseline), which is a
  different statement and must be written down as such; or
- drop the clause and rely on `z_b`, which does work.

### 2b. The coverage clause's object is ambiguous, and the two readings disagree

The clause says "coverage is within `τ_cov`". §6.3's C2 row is explicit that coverage is
"measured against the interval as REPORTED to the policy (`Q̃_α`)". C7's injection
scales the **safety factor inside `aggregate_order_up_to`** — the inventory level `S` —
and never touches the forecaster. So:

- measured against the **forecaster's interval**, C7 cannot move coverage at all and the
  clause is vacuous;
- measured against the **policy's `S`** (what the code does), C7 moves coverage a lot,
  and C7 becomes indistinguishable from C2 — which is exactly the identifiability
  problem constraint 6 exists to detect.

Neither reading is obviously right, and they give opposite verdicts. This needs §6.3's
C7 condition set to say which object it means.

### 2c. Constraint 7 is a magnitude problem in §15.2's C7 row

At `safety_scale = 0.60` the mean achieved CSL is 0.8013 against a target of 0.95 — a
real, large service gap. But the exposure clause needs **≥ 2 unmet cycles** in ≥ 80 % of
windows, and `achieved_csl < 0.85` ⇔ ≥ 2 failed cycles out of 10. Only 62.25 % of
windows reach that. The injection is working; it is not working *strongly enough* to
guarantee two failed cycles in four windows out of five.

Per §15.2's stop rule the honest options are (i) a larger scale in the injection table,
(ii) a longer exposure window, or (iii) §15.2's documented alternative — **report C7 as
not separately identifiable** and publish the six-cause taxonomy. **Not** a relaxed gate.

---

## 3. Five implementation defects found and fixed

These were mine, not the specification's. Each is fixed, tested, and committed.

### 3.1 `observation_model` was never set for the synthetic panel — every run collapsed

`synthetic.yaml` inherits `censored_sales` from `base.yaml` and never overrode it. Under
that convention the policy sees only what it sold, which is 0 from `t = 0` because it
starts with no stock and no history. The forecaster then correctly predicts 0 forever
and `alpha_eff = 1 − 1/(n_cycles+1) = 0` forever.

**This was silent.** No error, no NaN — just a run in which every arm returns identical
zeros. It was found by reading `alpha_t` in the C7 log: it was **0.0 in every window**
while the gate compared it against α = 0.95. `ruf.yaml` sets `observation_model: demand`
with a long comment describing this exact collapse; the synthetic config simply never
did. Fixed, with the same reasoning recorded in the config.

### 3.2 The C7 pre-flight stopped before the injection

`rollout_sku` truncates to `splits.burn_in + splits.evaluated` (= 84), which is short of
`tau_inj = 120`. The window loop starts at `tau_inj + W_svc = 150`, so it never
executed and the pre-flight would have reported an **empty gate as a pass**. The
pre-flight now simulates the whole series, and `synthetic.yaml` sets
`evaluated: 176` so the injection is inside the evaluated window everywhere.

### 3.3 `z_b` and RMSSE were computed on the demand level, not on the residuals

§6.3 line 255 defines `z_b = b^sz / (σ_sz/√n)` on the **size-subprocess error**
`e^sz = z − μ_nonzero`. The code used `y − μ̂` on nonzero periods, whose mean is
`μ_z(1−p)` — large and positive for *any* intermittent series. The clause therefore
rejected almost everything for a reason that had nothing to do with forecast
acceptability (0.51 pass, now **0.989**).

`μ_nonzero` was not recorded by the rollout at all, so `RolloutResult` now carries it in
`extra`. `res.residuals` covers the evaluated window while the inventory log covers all
`T`; the two are now aligned by **period**, not by position, which would have offset
every residual by `burn_in`.

### 3.4 C2's γ solver returned the endpoint on the wrong side

The bisection maintains `coverage(lo) < target ≤ coverage(hi)`. For **over**-coverage
the satisfying side is `hi`; for **under**-coverage it is `lo`. The code returned a
fixed endpoint, so the achieved gap landed exactly on `−0.120` for 200 of 200 C2 SKUs
and `|gap| ≥ 0.12` was False for every one of them. **Constraint 5: 0.3675 → 0.9525.**

### 3.5 `n_unmet_cycles` counted periods, not cycles

§6.3 condition 4 requires "≥ 2 unmet *cycles*", and a cycle is `R = 3` consecutive
periods. Counting periods with lost demand inflates the count by up to 3× and made the
exposure condition look satisfied on windows that had not completed a single failed
cycle.

### 3.6 Also fixed

- `draw_statics` consumed RNG before the parameter draw, shifting every cell in the
  panel. Statics now use their own child stream (`rng.spawn(1)[0]`).
- `PooledLightGBM` lived inside `experiments/run_ruf.py`, so any other driver building a
  pool through `build_pool()` got an untrained `GlobalLightGBM` and died on first use.
  Moved to `cafr/forecasters/pooled.py`, which is how the C7 pre-flight failed.
- The constraint report had heterogeneous key names and the printer raised `KeyError`.
  All blocks now share one shape, so an absent verdict is impossible.
- C5's occurrence branch is **arithmetically infeasible** for `p < 0.058`: the required
  step grows as `1/p` while the achievable step in sd units is bounded by the range of
  `p`. Those SKUs now take the size branch, which is always satisfiable. Constraint 3:
  **0.6400 → 1.0000**. The budget was not relaxed.

### 3.7 Coverage was scored against the trailing aggregate — fixed 2026-10-17

`S_t` is set at the end of period `t` and covers the protection interval **ahead**,
`[t+1, t+R]`. The coverage check compared it against the **trailing** aggregate
`[t-R+1, t]`.

That was correct in the pre-docs/17 code, where `S = Q_alpha(aggregates)` was a
statement about the same observed data. After docs/17 replaced it with the
residual-bootstrap predictive quantile, the comparison became a predictive quantile
scored against the *wrong realisation* — a real quantity, but not coverage. Fixed by
`forward_aggregates`: period `t` is scored against the demand it will actually have to
cover. The last `R` periods have no complete forward window and are excluded from the
coverage count. This is implementation, not specification: the clause always meant the
forward interval.

---

## 4. What now passes, with manifests

| constraint | measured | note |
|---|---|---|
| 1 — C4 band invariance | **200/200** | never relaxed |
| 3 — C5 detectability budget | **200/200** | scale up, else flag |
| 5 — C2 coverage gap | **381/400 (0.9525)** | C2 193/200, C2b 188/200 |
| 8 — C1 occurrence untouched | **200/200** | `p_shift = k_shift = 0.0` |
| 9 — C4 evidence budget | 72/200 | flagged, never dropped (by design) |

Panel: 2,000 series × 200 periods. Cells among live SKUs: `high_lowdisp` 0.333,
`moderate_lowdisp` 0.268, `moderate_highdisp` 0.190, `high_highdisp` 0.146; `dead` 0.0815.

### 4a. Constraint 5 is infeasible on the high-ADI tail — a spec finding

The 19 remaining failures all have `p ∈ [0.05, 0.091]`, i.e. **ADI ≥ 11**. Their maximum
achievable gap is below 0.12 for every one. The reason is structural: the R-period
aggregate is zero with probability `(1−p)³ ≥ 0.75`, so `P(a ≤ mean)` is already ≈ 0.83,
and any interval centred at or above the mean covers at least that much. The induced
under-coverage cannot reach `α − (τ_cov + margin)` **no matter how the interval is
built**. Those SKUs are flagged in `generator_params.parquet`.

This needs a decision: flag them (current behaviour), or define C2's coverage on the
per-period interval rather than the R-period aggregate.

---

## 5. Reproduce

```bash
python experiments/build_synth_panel.py --seed 42 --c7-skus 40
python -m pytest tests/ -q          # 149 passed
```

Every number above comes from that command and the files it writes. No notebook-sourced
numbers, no hand-entered values.

---

## 6. What is needed from the team

1. **Constraint 6, RMSSE clause** — pooled-vs-per-window, a tolerance band, or drop it
   (§2a). All three change what the gate *means*, so it is not my call.
2. **Constraint 6, coverage clause** — which object: the forecaster's interval or the
   policy's `S` (§2b).
3. **Constraint 7** — retune the C7 row's scale, lengthen the exposure window, or take
   §15.2's documented alternative and report C7 as not separately identifiable (§2c).
4. **Constraint 5** — confirm the high-ADI flagging rule (§4a).

---

## 7. Decisions — signed off 2026-10-17

The project owner delegated all four decisions ("choose best for paper project"). What
follows is written down **before** the run that reports them, so the restatement cannot
be reverse-engineered from the result. The rule that governs all four: **a clause that
cannot express its own intent is restated; a gate that fails a restated clause is not
relaxed.**

### 7.1 Constraint 6, RMSSE clause — pooled per-SKU, 10 % tolerance band

**Decision.** Replace the per-window comparison with the **pooled per-SKU** RMS of the
rolling one-step residuals:

```
rmsse_pre  = RMS( e_t : burn_in <= t < tau_inj )
rmsse_post = RMS( e_t : tau_inj <= t < T )
pass       = rmsse_post <= (1 + RMSSE_TOL) * rmsse_pre      # RMSSE_TOL = 0.10
```

The per-window ratio is still computed and reported (`rmsse`, `rmsse_base`,
`rmsse_window_ok`) for transparency, but it does not set the verdict.

**Why this is a restatement, not a relaxation.** §2a: under the null, a per-window
"post ≤ baseline" comparison has P(pass) = 0.5 *by construction*. An 80 % bar was
therefore unreachable by any correct implementation — the clause as written did not
assert "the error did not rise", it asserted a coin flip. Pooling turns the quantity
into one estimate per SKU of the **ratio** it was always meant to be about; the 10 %
band is the explicit statement of how much sampling noise counts as "did not rise".
Before the change, 57.6 % of windows passed; that number measured sampling noise.

**What it does not do.** It does not let a forecast degrade arbitrarily: the ratio is
still one-sided, still against the SKU's own pre-injection error, and the 10 % band was
chosen from the sampling spread of an 80–96-period RMS, not from the result.

### 7.2 Constraint 6, coverage clause — the forecaster's UNSCALED quantile

**Decision.** Coverage is measured against the forecast's predictive quantile **before**
C7's safety-factor scaling:

```
S              = R * mu_hat + scale * Q_alpha(A^e)
S_forecast     = R * mu_hat + Q_alpha(A^e) = R * mu_hat + (S - R * mu_hat) / scale
coverage       = mean( A^agg <= S_forecast )
pass           = |coverage - alpha_used| <= TAU_COV
```

`S_forecast` is reconstructed from the logged `S`, `mu_hat` and the scale
(`unscaled_forecast_level`). A fallback period (`S == R*mu_hat`) reconstructs to the same
value without a special case.

**Why.** §2b: measured against the policy's scaled `S`, C7's injection *is* the
coverage gap, so the clause becomes C2 and C7 is not identifiable — the exact failure
constraint 6 exists to detect. Measured against the unscaled forecast, the clause asks
the question it was written to ask: *did the forecast stay acceptable while the policy
was mis-set?* This is also the only reading under which C7 differs from C2.

**Consequence.** The coverage pass rate is now invariant to the injection scale, as it
must be: C7 moves the policy, not the forecaster. `scale > 0` is enforced, because at 0
the safety term carries no information and the reconstruction is undefined.

### 7.3 Constraint 7 — pre-registered scale grid, mildest first

**Decision.** The injection scale is chosen over a **grid fixed in advance**:

```
grid = (0.60, 0.45, 0.30, 0.15, 0.05)        # mildest first
```

Walk the grid in that order and use the **first** scale whose exposure pass rate
(`n_usable >= 20` and `n_unmet_cycles >= 2`) is at least 0.80. If no scale reaches it,
use the strongest scale (0.05), record `grid_exhausted_no_pass`, and **gate 4.9 fails**.

Selection reads **constraint 7 only**. Constraint 6 is invariant to the scale under
`observation_model = demand` — the forecaster sees demand, not sales — so letting it
steer the pick would be selecting on noise.

**Why 0.05 and not 0.0.** The unscaled quantile has to be recovered by dividing the
safety term by the scale (§7.2). At scale 0 that is undefined and the policy carries no
information about the forecast. 0.05 is the no-safety-stock regime in practice.

**Why this is allowed.** §15.2's stop rule names retuning the injection table as its
first option. The scale is a **generator parameter**, not a gate threshold. What would
not be allowed — and is not done — is lowering the 0.80 pass rate or the ≥ 2 unmet
cycles.

### 7.4 Constraint 5 — high-ADI tail stays flagged, not redefined

**Decision.** Confirmed: keep `gap_clears_with_margin = False` for the SKUs on which the
gap is arithmetically infeasible, report them separately (count and `p` range in
`panel_report.json`), and **do not** drop them or redefine C2 onto the per-period
interval.

**Why.** §4a: for `p` in `[0.05, 0.091]`, `P(A_R = 0) = (1-p)^3 >= 0.75`, so no interval
centred at or above the mean can under-cover by `tau_cov + margin`. That is a property
of the demand DGP, not of the injection. Redefining the object to make the constraint
pass would change what C2 means and make it non-comparable to C2b. Flagging is the
honest report; the drop is forbidden by handoff §13 item 23.

---

## 8. Measured outcome under the restated gate

**Run: 2026-10-17, 40 C7 SKUs, 2,040 windows, seed 42.**

```bash
python experiments/build_synth_panel.py --seed 42 --c7-skus 40
python experiments/c7_control_diagnostic.py --seed 42 --c7-skus 40   # scale 1.0
```

Artifacts: `results/synth/c7_constraints.json`, `c7_constraints.parquet`,
`c7_windows.parquet`, `c7_scale_ladder.parquet`, `c7_control_scale1.json`, and the
console log `c7_preflight_run.log`.

### 8.1 The verdict

| quantity | measured | required | verdict |
|---|---|---|---|
| constraint 6 — acceptability | **0.3240** | ≥ 0.80 | **FAIL** |
| constraint 7 — exposure | **0.8265** | ≥ 0.80 | **PASS** |
| gate 4.9 | — | both | **FAIL** |
| constraint 5 (C2 gap) | 0.9525 (381/400) | unchanged | PASS, 19 high-ADI flagged |

Constraint 7 is now settled: the pre-registered grid chose **scale 0.15**, the mildest
scale whose exposure reached 80 %.

| scale | acceptability | exposure | mean achieved CSL | selected |
|---|---|---|---|---|
| 0.60 | 0.3324 | 0.5794 | 0.8216 | no |
| 0.45 | 0.3201 | 0.6833 | 0.7878 | no |
| 0.30 | 0.3314 | 0.7686 | 0.7607 | no |
| **0.15** | 0.3240 | **0.8265** | **0.7240** | **yes** |

**No threshold was moved.** The 0.80 pass rates, the ≥ 2 unmet cycles, `TAU_COV`, the
10 % RMSSE band and the gate itself are exactly as pre-registered in §7.

### 8.2 Constraint 6, clause by clause

| clause | pass rate | note |
|---|---|---|
| `\|z_b\| ≤ 1.96` | **0.9877** | passes; the size-error bias is not the problem |
| RMSSE pooled ≤ 1.10 × baseline | **0.6250** | 15 of 40 SKUs exceed the band |
| coverage within `tau_cov` of `alpha_used` | **0.5441** | mean gap **−0.070**; 38.1 % under by > 0.10, 7.5 % over |

The pooled RMSSE ratios are centred where expected — median **1.04**, q10 0.61, q90 1.40
— but the spread is wide enough that 15/40 SKUs cross a 10 % band. Coverage is
systematically low: mean `alpha_used` 0.799, mean coverage 0.729.

### 8.3 The decisive finding: constraint 6 fails WITHOUT the injection

The same pipeline at **scale 1.0** — no injection at all — gives:

| clause | injected (0.15) | control (1.0) | difference |
|---|---|---|---|
| acceptability | 0.3240 | **0.3201** | +0.0039 |
| RMSSE pooled | 0.6250 | **0.6250** | 0 |
| `z_b` | 0.9877 | **0.9877** | 0 |
| coverage | 0.5441 | **0.5402** | +0.0039 |

**Constraint 6 fails just as hard with no C7 in the panel.** This is not noise about the
injection; it is the baseline forecaster's calibration. The reason is structural: under
`observation_model = demand` the forecaster never sees the stock-outs the injection
causes, so C7's scale cannot move its residuals, its `z_b`, or its predictive quantile.
The tiny differences are floating-point boundary effects in reconstructing the unscaled
quantile.

Therefore **constraint 6 cannot identify C7**. It is a test of whether arm (c)'s
forecast is acceptable in absolute terms, and on this sample it is not (0.32). No
injection scale can fix that, and no restatement of the clause's object can either —
which is why the restatements in §7, though correct, were not sufficient.

### 8.4 What this means for the paper

§15.2's stop rule is explicit: when the pre-flight cannot certify the injection, the
documented alternative is to **report C7 as not separately identifiable in this
simulator** and publish the six-cause taxonomy. That is the honest outcome here.

This is a **negative finding about the condition set**, not about the attribution
method: the monitor and classifier read the rolling residuals, which do depend on the
forecast, so C7's detectability is a separate question. But the C7 *injection* cannot
be certified by these constraints, so:

- **Do not** claim C7 identification from this pre-flight.
- **Do not** quote an arm-level inventory result as evidence about C7.
- **Do not** lower the 0.80 bar, widen `TAU_COV`, or relax the 10 % band. Any of those
  would turn a measured negative into an unmeasured claim.
- A **redesign** of the condition set — e.g. a difference-in-difference against the
  scale-1.0 control per SKU — is a team decision about what the gate should mean. It is
  not implemented here, because it would be a new gate chosen after seeing this result.

### 8.5 State saved for resumption

Results and the control at `results/synth/`; log at `results/synth/c7_preflight_run.log`;
the diagnostic driver at `experiments/c7_control_diagnostic.py`. Test suite: **181
passed** (includes the new clause and calibration tests). Gate 4.9 remains the blocker
for Steps 5–14 and M2.


