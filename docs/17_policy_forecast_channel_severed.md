# Finding 17 — the policy does not read the forecast; every arm is the same arm

**Status: OPEN — escalated, not resolved.** Date: 2026-09-30. Branch:
`cafr/step3-10-completion`. This is a **more serious** blocker than gate 4.9, because it
does not fail a gate — it makes the comparison the gate exists to protect *vacuous*.

---

## 1. The finding

The order-up-to level is **mathematically independent of the forecast**. A 1000× change
in `mu_hat` leaves `S` bit-for-bit identical:

```
mu_hat=    0.50  S=   42.875125
mu_hat=    5.00  S=   42.875125
mu_hat=   50.00  S=   42.875125
mu_hat=  500.00  S=   42.875125
```

Consequently arms (a), (b), (c) — a fixed validation-selected classical method, the
Syntetos-Boylan rule, and the pooled LightGBM — produce **identical policies**. Measured
on the real RUF panel, `S` differed between arm (a) and arm (c) in **0 of 720 periods**,
and arms (a) and (b) agree to the sixth decimal on `mean_cost`, `cycle_csl`, `fill_rate`,
`n_stockout` and `mean_S`.

## 2. The proof (this is arithmetic, not a judgement)

`cafr/sim/inventory.py`, normal branch of `aggregate_order_up_to`:

```python
residuals = aggregates - R * mu_hat
safety    = np.quantile(residuals, alpha_eff, method=method) * safety_factor_scale
return R * mu_hat + safety, ...
```

so

```
S = R*mu_hat + Q_alpha(aggregates - R*mu_hat)
```

Sample quantiles are translation-equivariant: `Q_α(x − c) = Q_α(x) − c` exactly, for any
sample and any `method`. Therefore

```
S = R*mu_hat + Q_alpha(aggregates) - R*mu_hat  ==  Q_alpha(aggregates)
```

**`R*mu_hat` cancels.** `aggregates = rolling_aggregates(y_hist, R, window)` reads `y`
and nothing else — it has no model input — so `S` is a function of the observed history
alone. In the normal branch the forecaster has **zero** influence on the policy.

The only `mu_hat`-dependent path left is the **fallback** branch
(`aggregates.size < min_aggregates`), which returns `S = R * mu_hat` with no safety
stock. That is why the arms are not *always* identical: a handful of early periods take
the fallback and carry a trace of the model. This is the entire explanation for arm (c)
differing from (a)/(b) by 1 stock-out period in 32 SKUs in
`results/ruf_abc_n80_seed42.json` — a rounding-scale artefact of the fallback, not a
policy difference.

**A trap for whoever fixes this.** The cancellation is exact in real arithmetic but not
in floating point: `agg - R*mu` then `np.quantile` then `+ R*mu` reintroduces rounding
error that depends on `mu`, so `S` *does* differ in the last ~16 significant digits. A
naive regression test — "S moves when `mu_hat` moves" via `len(set(levels)) > 1` — passes
on that noise alone. `tests/test_policy_forecast_link.py` therefore asserts a **relative
spread > 1e-6**, not exact equality. It was written the naive way first and XPASSed; the
noise is real and will fool the next reader too.

## 3. What this invalidates, and what it does not

**Invalidated.** Every inventory outcome comparison between arms — cost, CSL, fill rate,
stock-outs — is a comparison of an arm against itself. The project's central claim is
that *cause-attributed repair of the forecast* improves the inventory cost–service
frontier. Under this code there is **no channel** through which a forecast could improve
anything. Arm (c), the key comparator, is not being tested.

**Not invalidated.** The monitor and the attribution classifier read the **rolling
one-step residuals** `e_t = y_t - mu_hat_t`, which do depend on the model. So M1/M2 and
the per-cause recall numbers are unaffected. The service target `alpha` is also live
(`S` moves 8.04 → 11.57 across `alpha` 0.70 → 0.99), so the frontier's *horizontal* axis
works; it is the arms that do not move.

**Scope: global.** Both panels, every arm, every caller of `aggregate_order_up_to`. The
C7 pre-flight (§16) ran through this path, so its coverage numbers were measured on a
model-free `S` — which also explains why §16's coverage clause looked vacuous when read
against the forecaster's interval.

## 4. Why it happened

The code is a **faithful implementation of R-8 as written**. R-8 replaced
`S = R·μ̂ + z·σ̂` with "`S = Q̂_α`, the empirical α-quantile of the R-period aggregate",
and that is exactly what the code computes. The defect is therefore in the
**specification's wording**, not in the transcription of it: an "empirical quantile of
the aggregate" is by construction a statement about the data, not about the model.

R-8's justification — that the old form understated safety stock by 42.3 % at R = 3 —
was measured against the *old* formula. It does not carry over to this one, which is a
different estimator with different variance. That justification needs re-deriving
whichever way this is decided.

Note also that `base.py`'s own docstring says residuals must come from the rolling loop
and not from inside `fit`. `R * mu_hat` subtracted from every historical aggregate is a
single current forecast applied to the whole past — the same category of error.

## 5. Recommended fix (concrete, and implementable as-is)

Use the arm's **own rolling one-step residuals**, which each level function already
computes, to build the aggregate's predictive distribution:

```
S = R * mu_hat_t  +  Q_alpha( sum_{k=0}^{R-1} e_hat_{t-k} )
```

where `e_hat` are the rolling out-of-sample residuals `y_s - mu_hat_s` the level function
produced at its own past calls. `mu_hat` no longer cancels, because the historical
`mu_hat_s` differ from the current `mu_hat_t`. This is a residual-bootstrap predictive
quantile — standard practice for intermittent demand, and the direct analogue of R-8's
intent ("use the quantile of the aggregate, not `mu + z·sigma`") while keeping the
forecast in the loop.

The plumbing is small: `aggregate_order_up_to` takes a residual series instead of
subtracting a scalar, and each arm passes the deque it already maintains (arm (a) keeps
`self._residuals`; arm (b) needs three lines to do the same).

**This must not be applied unilaterally.** It changes every inventory number in the
project and redefines `Q̂_α`. Alternatives the team may prefer:

1. **Residual bootstrap** (recommended above).
2. **Scale-based safety stock**: `S = R·μ̂ + z_α · σ̂_R`, with `σ̂_R` the model's own
   estimate of R-period aggregate error. Simpler, but reintroduces the normal
   approximation R-8 was moving away from.
3. **Accept a model-free policy** and re-scope the paper: report the severed channel as
   a negative result and drop the inventory-outcome claim. This is a coherent paper, but
   it is a different paper from the one §11 specifies.

## 6. How this was found

Not by a gate — every gate passed. It was found because arm (a) was rewritten to select
its model honestly (§17.1), and the honest arm produced *identical* results to arm (b).
The coincidence was the symptom. This is the sixth silent defect in the Step 3–4 work,
and the same pattern as the other five: **plausible numbers, no exception**.

## 7. Reproduce

```bash
python -m pytest tests/test_policy_forecast_link.py -q      # 1 xfail(strict), 2 pass
python experiments/run_ruf.py --arms a,b,c --n-skus 80 --seed 42
```

`results/ruf_abc_n80_seed42.json` is the manifest: arms (a) and (b) identical, (c)
differing only through the fallback branch.

## 8. Decision needed

Which of §5's three options. Option 1 is recommended and is ready to implement. **Until
this is settled, no arm-level inventory result should be produced or reported** — the
numbers would be real, reproducible, and meaningless.

---

### 17.1 For the record: arm (a) is now a genuine baseline

Fixed in the same commit. `arm_a_factory` previously fell back to `SB_MODEL_MAP`
whenever no `chosen_method` was passed — which is every driver — so arm (a) and arm (b)
were the same arm under two names *even before* §1's cancellation was found. Arm (a) now
scores every classical candidate by rolling-origin one-step MASE on the burn-in window
(`y_observed[:burn_in]`, strictly pre-evaluation) and holds the argmin fixed. Fallbacks
are recorded with their reason, never silent. `RolloutResult.selection` carries the log.
Locked in by `tests/test_arm_a_selection.py` (9 tests).
