# Step 4 findings — gate 4.9 FAILS, and five implementation bugs it exposed

**Status: OPEN — needs a team decision. Nothing here has been resolved by changing a threshold.**

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
