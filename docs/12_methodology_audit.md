# 12 — FORMAL METHODOLOGY AUDIT OF THE CAFR SPECIFICATION

**Date:** 2026-09-23
**Audited document:** `11_technical_specification.md` (Rev. 1, 1168 lines), plus `08`, `09`, `10`.
**Auditor stance:** journal/conference reviewer. No encouragement by default. Negative findings are acceptable outcomes.
**Status:** **`11_technical_specification.md` was NOT edited by this audit.** This file proposes; the team decides.

**No experiments were run. No results appear here. Nothing in this document assumes CAFR works.**

---

## A. OVERALL VERDICT

# NEEDS MAJOR METHODOLOGY CHANGES

**Not a redesign.** The research question, the four-part seam, the novelty framing, the architecture
(M0–M5), the arms and the evaluation philosophy are all sound and should be kept. But **four
findings are pre-coding blockers**, and one of them means a headline claim is currently near-tautological.

**The four blockers:**

1. **§29's primary hypothesis H1 is close to true by construction.** (f) optimises inventory cost; (c)
   optimises MASE. Comparing them on cost asks "does optimising the objective beat not optimising it?"
   — not the research question. The paper's actual claim lives in ablation **A4**, which is currently
   a mid-table row.
2. **C2, C4 and C6 are the same intervention on the same knob.** All three change the Gamma shape `k`
   at fixed `μ_z`. Three causes, one manipulation. C4 is shadowed by C2 in the priority order; C6 has
   no distinct injection at all.
3. **C6's stated injection is mathematically impossible.** ADI = 1/p, and p ∈ [0.05, 0.50] gives
   ADI ∈ [2, 20]. ADI can never cross 1.32. The alternative (CV² crossing 0.49) is C4's injection.
4. **C7 cannot fire, for three independent reasons**, and the C7 controller (§7.1) is written in a
   different error-distribution convention from the simulator (§15.6). C7/R7 is one of the paper's
   two genuinely distinctive contributions, so this is not a detail.

Anything marked **[BEFORE CODING]** must be resolved before Himanshi writes the first line, because
each one changes what the code is.

---

## B. CRITICAL ISSUES

### C-1 · §29.1 / §21 — H1 is nearly a tautology; A4 must be promoted

**Exact problem.** H1 pre-registers "(f) beats (c) on the cost–service frontier". Arm (f) is a bandit
whose reward *is* standardised inventory cost (§9.3). Arm (c) selects models by rolling MASE (§12).
The comparison therefore measures whether an agent that directly optimises the evaluation objective
outperforms one that optimises a different, known-uncorrelated proxy. Any competent learner wins.

Worse, the spec itself supplies the evidence that this is expected: §1 cites arXiv `2609.13840` —
accuracy rank vs service rank Spearman ρ = **−0.555**. If accuracy is *anti*-correlated with service,
(f) > (c) is the predicted outcome before any experiment runs.

**Why it matters.** A reviewer's first sentence will be "your headline result is guaranteed by
construction." That is fatal for the *scientific* claim even though the *practical* claim is real.
The paper's novel content is **attribution**, and attribution is tested by A4 (bandit with attribution
features removed from the context) — which §21 currently places fifth in a table with no hypothesis
attached to it.

**Proposed fix.**

- **Add H1′ as co-primary, pre-registered:** *at matched churn, (f) — the bandit with mechanism
  attribution in its context — beats A4 — the same bandit, same action space, same cost reward, with
  attribution features removed.* This is the test of the paper's actual claim.
- **Demote H1 to a precondition.** Report it, but frame it explicitly as "necessary but not
  sufficient — it establishes that the operational objective is the right one, which §1's literature
  already predicts. The scientific claim is H1′."
- **Attach a stated failure condition to H1′:** if A4 ≈ (f), the attribution is decorative and the
  contribution is the reward signal alone. That is a publishable but *different and weaker* paper.
  §29 must say so in advance.
- Keep A4's current wording ("This is a hard test of the paper's core claim") — it is correct. Move it
  to the front of §21 and give it a hypothesis ID.

**[BEFORE CODING]** — yes. It changes what E4 *is*, and therefore what step 10 builds.

---

### C-2 · §15.2 / §6.4 — C2, C4 and C6 are one intervention with three labels

**Exact problem.** For `Gamma(shape=k, scale=μ_z/k)`: `mean = μ_z`, `var = μ_z²/k`, `CV² = 1/k`.

| Cause | Stated injection | Effect on `k` | Effect on `CV²` |
|---|---|---|---|
| C2 mis-calibration | "Multiply the size variance by γ ∈ {0.33, 3.0} at fixed mean (adjust k at constant μ_z)" | `k → 3k` or `k → k/3` | ×1/3 or ×3 |
| C4 size failure | "`k → k·{0.25, 4.0}`, μ_z unchanged" | `k → 4k` or `k → k/4` | ×1/4 or ×4 |
| C6 intermittency change | "CV² crosses 0.49" | requires changing `k` | crosses 0.49 |

**These are the same manipulation.** C2 and C4 differ only in the multiplier (3 vs 4); C6's only
reachable form is the same knob again. The spec asserts in §15.2 that each injection "deliberately
leaves the others alone" — it does not. All three move **size-distribution dispersion at fixed mean**,
which moves, simultaneously:

- the size-model error → **C4's** signature
- the interval coverage → **C2's** signature
- the SB class via CV² = 1/k → **C6's** signature

**Then the priority order (§6.4) decides the outcome, not the mechanism.** Order is
`C5 → C1 → C2 → C3/C4 → C6 → C7`. C2 sits *ahead* of C4. C2's only mean-guard is `|z_b| ≤ 1.96`,
which a fixed-mean `k` change satisfies by construction. So:

- On a **C2** series → C2 fires. Correct.
- On a **C4** series → coverage has also moved → **C2 fires**. C4 is misattributed.
- On a **C6** series → identical manipulation → C2 fires. C6 never fires.

**Consequence if shipped:** the confusion matrix would measure the priority ordering, not the
detectors. Precisely the failure §15.2's own rationale says it wants to avoid ("the confusion matrix
would say more about the injection design than about CAFR"). The spec identified the risk for C1/C4/C5
and did not notice it applies to C2/C4/C6.

**Proposed fix — decouple the three:**

| Cause | Must perturb | Must NOT perturb |
|---|---|---|
| **C2** | Interval width **only** — perturb the *quantile mapping*, not the data | the size distribution; the point forecast; coverage must be the sole moving statistic |
| **C4** | Size **level/location** at fixed shape — e.g. a scale shift on nonzero sizes applied only to the *later* part of the series, or a size-distribution change the forecaster cannot track | CV² (hold `k` fixed); occurrence |
| **C6** | Requires a **new construction** — see C-3 | — |

The cleanest C2 injection is not a DGP change at all: **leave the demand process untouched and
deliberately mis-set the *interval* the forecaster reports** (e.g. multiply the reported predictive
standard deviation by 0.5). Coverage then moves and *nothing else does*. This is legitimate — C2 is a
statement about the *forecaster's uncertainty*, not about demand — and it makes C2 uniquely
identifiable. It also makes C2 the natural twin of C7 (a deliberately mis-set policy), which
strengthens the design's internal logic rather than weakening it.

**[BEFORE CODING]** — yes. Three of seven causes are currently not separately inducible, and the
injection table is the ground truth for every downstream number.

---

### C-3 · §15.2 / §6.3 — C6's ADI crossing is mathematically impossible

**Exact problem.** `ADI = (number of periods) / (number of nonzero periods)`. Under i.i.d.
`Bernoulli(p)`, `E[ADI] = 1/p`. The stated range is `p ∈ [0.05, 0.50]`, so `E[ADI] ∈ [2, 20]`.

The Syntetos–Boylan cut-off is `ADI = 1.32`. ADI ≥ 1.32 ⟺ p ≤ 0.7576. **Every value of p in the
specified range is already above the cut-off, permanently.** "Slow drift in p so that ADI crosses
1.32" cannot happen. To cross it, p would have to exceed 0.758 — which is fast-moving demand, not
intermittent demand, and contradicts §1's entire framing.

The spec's fallback, "or CV² crosses **0.49**", is C4's and C2's injection (see C-2).

**Why it matters.** C6 is one of seven causes. As specified it is (a) unreachable via ADI, (b)
indistinguishable via CV², and (c) additionally guarded by a bootstrap-stability requirement
("stable in ≥ 90 % of resamples") that a 12-period window can essentially never satisfy. C6 is dead
on three counts. Its ground-truth series will be labelled C2 by the monitor, and the reported C6
recall will be ~0.

**Proposed fix — pick one, explicitly:**

- **(a) Redefine C6 on the reachable boundary.** In this domain ADI ≥ 1.32 is a *definition of
  entry*, not a boundary to cross. Replace C6 with "**intermittency-regime change**" defined on the
  pair (ADI, CV²) crossing a **domain-relevant** boundary: e.g. `ADI` crossing 2.0, or moving between
  the intermittent and lumpy quadrants. State the new cut-offs and justify them from the panel's own
  ADI distribution (T1), not from the textbook values that do not apply here.
- **(b) Merge C6 into C4** and present six causes. Defensible, and OD-1 already contemplates
  collapsing classes.
- **(c) Widen `p`** to include `p > 0.758` so the ADI cut-off is reachable. **Not recommended** —
  it changes the project's subject matter to make the methodology easier, which is exactly what this
  audit was told not to do.

**Recommendation: (a).** It keeps seven causes, keeps the SB machinery, and is honest about the
domain — spare-parts panels live entirely inside ADI ≥ 1.32, and saying so is itself informative.

**[BEFORE CODING]** — yes.

---

### C-4 · §15.1 — The four SB classes cannot be generated

**Exact problem.** §15.1 requires sampling `(p, μ_z, k)` "so that all four Syntetos–Boylan classes are
represented — target roughly equal counts of smooth / erratic / intermittent / lumpy." With
`p ∈ [0.05, 0.50]`:

- ADI ≥ 2 > 1.32 always ⇒ **smooth** (ADI < 1.32, CV² < 0.49) is **unreachable**
- ADI ≥ 2 > 1.32 always ⇒ **erratic** (ADI < 1.32, CV² ≥ 0.49) is **unreachable**
- Only **intermittent** (CV² < 0.49, i.e. `k > 2.04`) and **lumpy** (CV² ≥ 0.49, i.e. `k < 2.04`)
  exist.

**Corroborating inconsistency:** §16 requires "a panel that is > 70 % *smooth* is the wrong panel."
Under §15.1's own parameters, 0 % of any panel can be smooth. The two sections cannot both be right.

**Why it matters.** §15.1 asks the generator to produce a stratification that is impossible, and
§15.2's C6 injection depends on the class machinery. Himanshi would build it, find two classes, and
have no way to tell whether the generator or the requirement is wrong.

**Proposed fix.** State what is true: **in this domain only the intermittent and lumpy quadrants are
inhabited, and that is the point** — ADI ≥ 1.32 *is* what intermittent demand means. Then:

- Require the panel to span **both ADI sub-ranges within intermittent/lumpy** (e.g. ADI ∈ [2, 4] and
  ADI ∈ [4, 20]) and **both CV² sub-ranges** (k > 2.04 and k < 2.04), giving four genuinely
  populated cells.
- Restate §16's panel-suitability rule on the *reachable* classes: a panel that is > 90 % one
  (ADI-band, CV²-band) cell is the wrong panel.
- Record the true class as a 2×2 (ADI-band × CV²-band) label rather than the four-class SB label.

This is a small change to the text and removes an impossible requirement.

**[BEFORE CODING]** — yes, jointly with C-3, since both concern the class machinery.

---

### C-5 · §6.3 — The CUSUM is run on the wrong series, and C5's step is below its own slack

Two independent defects in the same detector.

**(i) The CUSUM is run on the pooled residual `e_t = y_t − μ̂_t`, which is dominated by the zero
structure, not by level shifts.**

For a zero period, `e_t = −μ̂_t ≈ −p·μ_z`. For a nonzero period, `e_t = z_t − p·μ_z`. Over a window
with p = 0.1, roughly nine of every ten residuals are a moderate negative constant. The recursion

```
S⁻_t = max(0, S⁻_{t−1} − e_t − k)
```

therefore accumulates steadily through any run of zeros, **regardless of whether a level shift
occurred**. Long zero runs are the defining feature of intermittent demand, so the CUSUM's false-alarm
rate is governed by `p`: the *lower* the p, the *more* false C5 alarms. The spec cites §6.2's
C-guard as the protection — but the guard only excludes *censored* periods, and structural zeros are
not censored. **The guard does not cover this.**

This is exactly the failure mode the audit was asked to look for: *a statistic moving for a reason
other than the intended mechanism*, and here it moves as a direct function of the intermittency the
project is about.

**(ii) C5's stated step magnitude is at or below the CUSUM slack, for typical `k`.**

The slack is `k = 0.5·σ_e`. Take `μ_z = 100` and `Gamma` shape `k_shape = 1` (well inside the sampled
range, and CV² = 1, i.e. lumpy — the regime the paper targets). Then `sd(size) = μ_z/√k_shape = 100`,
so `σ_e ≈ 100` before adding the forecaster's own error (which only makes it worse).

C5's injection is a multiplicative step `{0.6, 1.5}` on `μ_z`:

| Step | Shift in `μ_z` | Standardised shift | vs slack `k = 0.5σ_e` | Consequence |
|---|---|---|---|---|
| ×1.5 | +50 | **0.50 σ** | exactly equal to the slack | zero drift in the CUSUM; ARL₁ ≈ ARL₀ ≈ 465. **Never alarms.** |
| ×0.6 | −40 | **0.40 σ** | **below** the slack | CUSUM drifts the *wrong way*. **Never alarms.** |

And it is worse than that, because only nonzero periods carry size information: at p = 0.2 the ARL
must be multiplied by ~5 in calendar periods. With `T = 200` and `τ_inj = 120`, C5 is undetectable
for most of the sampled parameter space.

**Why it matters.** C5 is the cause the priority order places *first* — it pre-empts C1, and
"abrupt shift invalidates the window" is the stated rationale for that. A C5 detector that false-alarms
on zero runs and cannot see its own injection is the worst possible combination: it will absorb
C1/C6/C7 series as false positives while missing genuine shifts.

**Proposed fix.**

1. **Split the CUSUM into two, one per sub-process, each on the correct series:**
   - **Occurrence CUSUM** on the Bernoulli sequence `δ_t` against `ẑ_t` — detects a step in `p`.
   - **Size CUSUM** on `e^sz` over **nonzero periods only** — detects a step in `μ_z`.
   Never on the pooled residual. This also makes C5 consistent with the C3/C4 sub-process
   decomposition the paper already claims as its sharpest sub-piece.
2. **Co-design the CUSUM parameters with the injection magnitudes.** Either reduce the slack
   (`k = 0.25·σ`) and the threshold (`h = 3·σ`), or **increase C5's step to ≥ 2σ_e** (e.g. ×2.0 / ×0.5).
   Do not tune one without the other. This must become part of OD-7.
3. **Re-derive the detection-delay budget** and state it: with the chosen `k`, `h`, `p` and `W`, what
   standardised shift is detectable within `W` periods? If the answer excludes C5's injection, the
   injection is wrong.
4. **Add the confound test** (see the Step-4 gate in §F): the CUSUM statistic must separate C5 series
   from C0 **and must not** separate C1/C6 series from C0.

**[BEFORE CODING]** — yes. The CUSUM's input series is a code-level decision.

---

### C-6 · §6.3 / §15.2 / §15.6 — C7 cannot fire, for three independent reasons

C7 is defined by four conditions. Three of them are currently unsatisfiable or self-defeating.

**(i) The exposure condition contradicts the window.**
§6.3 C7 condition 4 requires "**≥ 20 periods in the window**". The monitor window is `W = 12` (§6.1).
`12 < 20`. **C7 fails on arithmetic.** (§7.1 repeats the same error independently: "hence C7's
requirement of `W ≥ 20`." The spec knows the requirement and sets `W` below it.)

**(ii) The C-guard can starve the window.**
C7's injection sets the safety factor to 60 % of correct, producing frequent stock-outs. Every
stock-out period is removed from `U_i(t)` by the C-guard (§6.2). If, say, 35 % of periods are
censored, `|U_i(t)|` over a 12-period window is ~8. C7's own condition 4 then fails harder, and
conditions 1–2 lose the evidence they need.

**(iii) The forecast-acceptability gate is destroyed by the injection's own consequences —
unless the forecaster's input is specified, which it is not.**
C7 condition 2 requires the forecast to be acceptable (RMSSE below baseline, `|z_b| ≤ 1.96`, coverage
within `τ_cov`). In a C7 series the policy is mis-set → stock-outs → **censored demand** → the
forecaster, if it fits on `demand_observed`, learns a systematically **understated** demand level →
its error rises → **the acceptability gate fails** → **C7 never fires.**

This is a genuine loop, and it is precisely the mechanism ISNCC 2026 documents and the C-guard exists
to mitigate. But §6.2's guard protects only the *bias and CUSUM statistics*. It does **not** protect
the forecaster's training input, which §14.3 and §15.6 leave entirely unspecified. §15.6 says "log
both censored and true demand… the policy sees only the observed series" — it says nothing about what
the **forecaster** is fitted on.

There is a further irony: if the forecaster *does* fit on censored demand, then C7 series will
present a genuine C1/C5 signature, and the correct answer for the monitor may well be C1 — making C7
not merely undetectable but arguably *not the right label*. The audit cannot resolve that from the
spec; it must be decided (see §E, OD-2).

**Why it matters.** C7/R7 is, with the C3/C4 decomposition, one of the two genuinely distinctive
contributions — it is the row that makes "non-forecast remedies" more than a slogan, and it is the
cleanest instance of RQ3. As written it will almost certainly report near-zero recall, and the paper
will be unable to distinguish "the mechanism is undetectable" from "our code never let it run".

**Proposed fix.**

1. **Raise `W` above 20** — see C-7. C7's exposure condition is the binding constraint on the whole
   monitor design.
2. **Add a mandatory pre-flight check to step 4:** on C7 series, the forecast-acceptability gate must
   pass on **≥ 80 % of post-injection windows**. If it does not, the C7 condition set is
   self-defeating and must be revised — not the code.
3. **Specify the forecaster's fitting input explicitly**, and make it a config switch:
   - `fit_on: observed` (realistic, but destroys C7's gate)
   - `fit_on: true` (isolates C7's claim cleanly; declare it as an idealisation)
   - `fit_on: uncensored` (impute lost sales before fitting — the realistic middle path, and the
     natural home for OD-2's "C8" machinery if the team wants it)
   Whichever is chosen, **the choice must be applied identically across all arms and all panels**, and
   stated in the paper. The audit's recommendation is `uncensored` — it is defensible, it is what a
   competent planner does, and it lets C7 fire without an idealisation.
4. **Restate the C-guard's scope** in §6.2 to say explicitly what it does and does not protect.

**[BEFORE CODING]** — yes. Items 1 and 3 are structural.

---

### C-7 · §6.3 / §6.1 — The monitor window `W = 12` contradicts four detectors' own evidence requirements

The spec fixes one window, `W = 12` (§6.1), then asks the detectors to work over four different
horizons:

| Detector | Stated requirement | Effective horizon needed | vs `W = 12` |
|---|---|---|---|
| C1 condition 3 | "sign is stable across the last **k = 4** consecutive windows" | 4 × 12 = **48 periods** | 4× over |
| C2 | "requires **n ≥ 30** coverage observations" (one per period) | **30 periods** | **2.5× over — unreachable** |
| C4 | "sustained over **2 consecutive windows**" | **24 periods** | 2× over |
| C7 condition 4 | "**≥ 20** periods in the window" | **20 periods** | **unreachable** |
| C3 | "**≥ 10 nonzero periods** in the window" | at p = 0.1, **100 periods** | **8× over — unreachable** |

C2, C3 and C7 are therefore **dead by internal arithmetic** before any code exists. C1 and C4 are
reachable only if "window" silently means something larger than `W`.

**Why it matters.** Three of seven causes cannot fire. The failure presents as "attribution macro-F1
is low" — which §29.4 interprets as *"the mechanisms are not separable"*, an RQ1 negative finding.
**The audit's conclusion is the opposite: the mechanisms may separate perfectly, and the code still
returns a null result because the windows are too short.** A false negative produced by a parameter
contradiction is the most damaging possible outcome here, because it is reported as a scientific
finding.

**Proposed fix.** Replace the single `W` with an **explicit multi-scale monitor**, documented in one
table:

| Statistic | Trailing horizon | Minimum evidence | Rationale |
|---|---|---|---|
| Bias `b`, `z_b` | `W_short = 24` | ≥ 8 nonzero periods | short-window bias |
| Coverage `ĉ` (C2) | `W_cal = 30` | ≥ 30 coverage obs | C2's own requirement |
| Occurrence AUC (C3) | `W_occ = 60` | ≥ 10 nonzero | needs enough positives |
| Size error `r` (C4) | `W_sz = 24` | ≥ 10 nonzero | 2 × 12 |
| CUSUM (C5) | recursive, no window | — | naturally cumulative |
| ADI / CV² (C6) | `W_class = 30` | ≥ 30 periods | bootstrap stability |
| CSL gap (C7) | `W_svc = 30` | ≥ 20 periods, ≥ 2 stock-outs | C7's own requirement |
| Bias sign stability (C1) | 4 × `W_short` | — | C1's own requirement |

Then check the panels can supply it: **T = 200 with τ_inj = 120 leaves 80 post-injection periods — fine
for the synthetic panel.** But §14.1's real-panel split is 24 burn-in + 60 evaluated = 84, so a
60-period occurrence horizon leaves almost no evaluated window. **Either extend the real panels'
usable length or reduce the longest horizon.** This interaction must be resolved before coding.

**[BEFORE CODING]** — yes.

---

### C-8 · §7.1 vs §15.6 — The C7 controller and the simulator use different error-distribution conventions

**Exact problem.**

- **§7.1 (the controller):** `z_{t+1} = clip(z_t + κ·g, 0, Φ⁻¹(0.999))`. `z` is explicitly "in
  **standard-normal units**", and the update is linear in the CSL gap.
- **§15.6 (the simulator):** "`σ̂` estimated from the **empirical error distribution**, **not assumed
  Gaussian** (intermittent demand errors are not normal…)" and `S = R·μ̂ + z·σ̂`.

**These cannot both hold.** If the order-up-to level is built from empirical quantiles — which is the
correct construction for intermittent demand, and §15.6 is right to insist on it — then `z` is not a
standard-normal quantile and the controller's update rule has no defined meaning. `κ = 1.0` "in
standard-normal quantile units per unit of CSL error" is undefined when the quantile map is empirical.

**Why it matters.** This is R7 — the remedy that makes C7 more than a label, and the paper's only
"leave the forecast alone" action. If (a) C7 can never fire (C-6) and (b) R7's controller is
undefined, then the paper's most distinctive claim has no working implementation.

**Proposed fix.** Pick one convention and propagate it. The audit recommends **empirical throughout**:

```
Controller (empirical form):
    Let  F̂_t  = the empirical CDF of the protection-interval demand error, estimated on W_svc periods
    Let  q_t   = F̂_t⁻¹(α_target)             # the target service quantile
    Let  ĉ     = achieved CSL over W_svc
    g          = α_target − ĉ
    S_{i,t+1}  = R·μ̂_{i,t} + q_t + κ · g · (q_{0.95} − q_{0.50}) / 0.05     # κ in quantile-units per unit gap
```

i.e. keep the integral controller's *structure* (it is sound, and `κ = 1.0` is a sensible default)
but express the increment in the units the simulator actually uses. Alternatively, keep `z` in
standard-normal units **and** build `S` from the normal quantile — but then §15.6's "not assumed
Gaussian" instruction must be retracted, which would be a real methodological regression.

Whichever is chosen, **§7.1 and §15.6 must be edited together, and R7's unit test (step 7) must
assert the actual CSL response to a known CSL gap.**

**[BEFORE CODING]** — yes. It is the definition of a remedy, not a tuning detail.

---

### C-9 · §15.6 / §28.6 — The `(S,S)` formula is mis-scaled, and the simulator sanity test cannot pass as written

**(i) The safety-stock term is on the wrong scale.**

§15.6: `S = R·μ̂ + z·σ̂`, with `R = L + 1` and `σ̂` the empirical error sd. But `σ̂` is estimated from
**per-period** forecast errors, while `S` must cover the **R-period** protection interval. Under
independence the protection-interval sd is `σ_R ≈ √R · σ_period`. For the default `L = 2` (`R = 3`),
`√R ≈ 1.73` — **the safety stock is understated by ~42 %**, so achieved CSL lands well below target
for every arm.

The spec contains the correct object and does not use it: §6.1 defines `A^{agg}` (aggregate demand
over the next R periods) and `Q̂_α` (its predicted α-quantile), and §6.3 C2 uses `Q̂_α` correctly.
**The fix is to use `Q̂_α` in §15.6's order-up-to rule**, which also resolves C-8 in one move.

**(ii) The simulator sanity test cannot pass as written.**

§28.6: *"with a **known-perfect forecaster** and a correctly-set policy, achieved CSL matches the
target within Monte-Carlo error."*

A forecaster with *zero error* has `σ̂ = 0`, hence safety stock `= 0`, hence `S = R·μ̂`. Demand exceeds
the mean in roughly half of all cycles ⇒ **achieved CSL ≈ 0.50**, against a target of 0.80–0.95.
**The test fails by construction**, and the failure is in the test's specification, not the simulator.
Himanshi would build the simulator, watch the gate fail, and have no way to tell which side is wrong.

**Proposed fix.**

- Rewrite the gate as: *"with a forecaster that supplies the **true predictive distribution** of
  protection-interval demand (i.e. a DGP-aware oracle, not a zero-error point forecaster), achieved
  CSL matches target within ±0.02"* — see §F, Step 1 for the full test set.
- Adopt `S_{i,t} = Q̂_α(i,t)` directly (the `α`-quantile of the R-period aggregate), with the C7
  controller modulating `α` rather than `z`.
- State whether an ordering cost is included (§18 leaves this open — decide, and say so in the paper).

**[BEFORE CODING]** — yes. This is step 1, the first gate.

---

### C-10 · §14.3 / §15.6 — What the forecaster is fitted on is unspecified, creating an unmeasured policy→forecaster feedback loop

**Exact problem.** §14.3 governs what the **policy** may see. §15.6 governs what the **simulator**
logs. **Neither specifies what the forecaster is fitted on.** The `Forecaster.fit(y)` signature in
§25 does not say whether `y` is `demand_observed` or `demand_true`.

This is not a minor omission. It creates a **feedback loop from the policy to the forecaster** that
is present in every arm and every cause:

```
policy sets safety factor → stock-outs → demand censored → forecaster learns a lower demand level
   → forecast degrades → the monitor sees bias/error → attribution fires (possibly wrongly)
   → policy changes the safety factor ...
```

The loop is *realistic* — it is exactly what ISNCC 2026 documents — but it is **uncontrolled,
unmeasured and unstated** in the current spec, and it silently couples the seven causes:

- On **C7** series the loop destroys C7's own acceptability gate (C-6 iii).
- On **C1/C5** series the loop *mimics* their signatures, so C1/C5 false-positive rates rise for
  reasons the C-guard does not address (the guard handles censoring in the *statistics*, not in the
  *training data*).
- Ablation **A6** ("C-guard removed") would attribute the entire loop's effect to the guard, which
  only covers part of it. **A6's measured effect would be misattributed.**

**Why it matters.** This is the single most load-bearing unstated assumption in the design. Every
confusion-matrix cell depends on it, and no reader could reproduce the experiment without guessing.

**Proposed fix.**

1. **Make it an explicit config switch** with three settings: `observed`, `uncensored` (impute lost
   sales before fitting), `true`. See C-6, item 3.
2. **Fix it identically across arms and panels.** Any variation between arms invalidates the
   comparison.
3. **Report it in table T7** alongside the other frozen hyperparameters.
4. **Re-scope A6** to state precisely which is ablated — the statistical guard, the fitting input, or
   both — and consider splitting A6 into A6a (statistical guard) and A6b (fitting input).
5. **Add a measured diagnostic**: log how often the loop changes the attributed cause. A single
   number — "the censoring loop changed the attributed cause on X % of windows" — would be a genuinely
   useful contribution and is cheap to produce.

**[BEFORE CODING]** — yes.

---

### C-11 · §27 / §14.3 — A static feature file leaks the post-injection class, and the no-look-ahead test cannot catch it

**(i) `sku_meta.parquet` leaks.** §27 lists `class_adi` (str) and `class_cv2` (str) as static per-SKU
attributes. If these are computed over the **full series**, then:

- they leak the future into every decision (a §14.3 rule-3 violation in spirit, and rule 3's wording
  — "all **rolling** statistics use trailing windows" — does not obviously cover a static input file);
- on the **synthetic panel** the full-series class is computed *after* injection, so a C6 (or C4/C2,
  per C-2) series would carry a class label reflecting the *post-injection* regime — **a direct
  ground-truth label leak into the feature vector.**

C6's detector (§6.3) explicitly compares "the class at t" against "the class used at initialisation",
so the initialisation class **is** a legitimate input — but only if computed on the burn-in window.

**(ii) The no-look-ahead unit test cannot catch this.** §14.3 rule 5 asserts the policy's inputs
contain no column matching `*_true*` or `*_cause*`. Against the actual schema of §27:

| Column | Matches `*_true*`? | Matches `*_cause*`? | Leaks? |
|---|---|---|---|
| `true_cause` | ✅ caught | — | yes |
| `demand_true` | ✅ caught | — | yes |
| `cause_active` | ❌ **missed** | ❌ **missed** | yes — reveals exactly when the injection is live |
| `injection_param` | ❌ **missed** | ❌ **missed** | yes — reveals the cause and its magnitude |
| `class_adi`, `class_cv2` | ❌ missed | ❌ missed | yes (if full-series) |

**Two of the three ground-truth columns, and both class columns, pass the test that §14.3 calls "the
single highest-value test in the repository."** A blacklist of name fragments is the wrong design.

**Proposed fix.**

1. **Replace the name-pattern test with an exact column whitelist.** The policy's input frame must
   have columns *exactly equal* to the declared feature set — set equality, not substring absence.
   Any extra column fails the test. This is shorter to write than the blacklist and cannot be evaded
   by naming.
2. **Have `ground_truth.parquet` and `demand_panel.parquet` be separate files that the policy loader
   never opens** — enforced by the loader's signature taking only the observed-demand frame, not by a
   runtime check.
3. **State that `sku_meta`'s class columns are computed on the burn-in window only**, and add an
   assertion comparing them against a burn-in-window recomputation.
4. **Merge `class_adi`/`class_cv2` into one `sb_class_init` column** plus the numeric `adi_init`,
   `cv2_init`. The SB class is a *joint* function of the two; two independent string columns is a
   schema error waiting to happen.

**[BEFORE CODING]** — yes. It is three lines of schema and one test.

---

## C. IMPORTANT BUT NON-CRITICAL ISSUES

These should be fixed before the **test-split run**, not necessarily before coding starts.

### I-1 · §9.4 — The burn-in biases the headline against arm (f)

"First `W_burn = 12` periods of the **evaluated window**, force uniform-random arm selection." The
evaluated window is 60 periods, so **20 % of (f)'s scored periods are deliberately random**, while
arm (c) selects normally throughout. The headline comparison is therefore run with (f) handicapped
by design.

**Fix (recommended): stagger the forced exploration across SKUs** — at any given period only ~1/9 of
SKUs are exploring, so no period is systematically degraded for (f), and the posterior still fills
quickly (5,000 SKUs × 12 periods = 60,000 forced decisions is far more than needed).
**Alternative:** add a 12-period policy warm-up excluded from scoring for **all** arms, reducing the
evaluated window to 48 and requiring the real panels to supply 24 + 12 + 48 = 84 periods (§14.1 is
already exactly 84 — it would still fit).

### I-2 · §9.1 — Stationary Thompson sampling against a deliberately non-stationary problem

C5 and C6 *are* non-stationarity injections. A standard linear Thompson sampler converges and then
stops adapting; its posterior variance collapses, so it under-explores exactly when the world has
changed. **CAFR would systematically under-perform on the causes it exists to handle**, and the
CUSUM-triggered dwell override (§11.2) addresses *switching*, not *posterior staleness*.

**Fix.** Add one of: (a) a per-period discount on the posterior precision (`γ ≈ 0.99`); (b) a
sliding-window likelihood over the last `N` updates; (c) **an explicit posterior reset on a C5 CUSUM
alarm** — the most defensible, since it ties the adaptation to the diagnosis. Report the choice in T7
and sweep it in E6.

### I-3 · §21 — Ablation A5 is confounded by reward scale

A5 is "the paper's cleanest result" (§21) — same action space, accuracy reward `r_t = −|e_t|`. But
(f)'s reward is **per-SKU standardised** (`/σ_{C,i}`) and A5's is **raw**. Thompson sampling's
exploration is scale-sensitive (the prior precision interacts with the reward scale), so an A5-vs-(f)
difference would be attributable to reward *content* or reward *scale*, indistinguishably.

**Fix.** `r_t = −|e_t| / σ_{e,i}` — per-SKU standardised absolute error — so A5 differs from (f) **only**
in the reward's content. Without this the sharpest test in the paper does not test what it claims.

### I-4 · §29.1 criterion 3 — Too strong; it will report real effects as failures

"At **every** swept target α ∈ {0.80, 0.90, 0.95}, (f) is weakly better than (c)" — ANDed with a
bootstrap-CI criterion, a 2 % effect-size floor, a 5-seed sweep, and a ±20 % threshold perturbation.
A conjunction of five conditions makes SUPPORTED very hard, so a substantively real effect is
reported as **FAILED**. Note also the internal inconsistency: "weakly better" admits +0.01 %, while
criterion 2 declares anything under 2 % "no practical difference".

**Fix.** Replace criterion 3 with: **"at ≥ 2 of the 3 target α values, and at no α is (c)
significantly better than (f) after Holm correction"** — or define dominance as the **area under the
(cost, CSL) frontier**, which is the quantity §18 already says is the right reporting convention.
Also **unify the practical-equivalence threshold**: §22.3 says 1 %, §29.1 says 2 %. Pick one (2 % is
the more defensible for a cost metric) and use it in both places.

### I-5 · §19 / §29.2 — The macro-F1 denominator is undefined, and including C0 creates a perverse incentive

§19 defines "attribution macro-F1 … macro-averaged" over a "9×9 (7 causes + C0 + ambiguous)" matrix.
§15.3 says do not fold the ambiguous panel into the macro-F1. §29.2 requires macro-F1 ≥ 0.70 with no
stated denominator.

If **C0 is a class in the macro-F1**, a detector that abstains on everything scores a perfect C0 F1
while failing every other class — the metric rewards blanket abstention, which is precisely what §8
and §19's "abstention precision" exist to prevent.

**Fix.** State the denominator explicitly: **macro-F1 over the 7 real causes only.** Report C0
separately as false-positive rate and abstention precision (§19), and the ambiguous panel separately
as abstention rate (§15.3). Add a **per-class F1 floor** so a single collapsed class cannot hide
behind a good macro average — the `min` per-class F1 is the honest headline number.

### I-6 · §29.2 — H4 is descriptive, not a hypothesis

"≥ 25 % of the cost improvement attributed to R2/R7/R0" — this is an *outcome* of the data. If a
panel genuinely needs mostly forecast-side repairs, the share is low and H4 "fails" for a reason that
is not a design failure.

**Fix.** Report the RQ3 decomposition descriptively with a bootstrap CI. If a hypothesis is wanted:
*"the non-forecast share is greater than zero, with a CI excluding zero"* — i.e. non-forecast remedies
contribute at all. That is genuinely falsifiable and genuinely informative.

### I-7 · §6.4 / §29.2 — The ≤ 10 % false-positive criterion is a tuning target, not a test

§6.4: "`τ_conf` must be tuned on the validation split to hit a target false-positive rate **≤ 10 %**."
§29.2 H2 then requires "false-positive rate ≤ 10 % on the C0 control panel." Tuning to a target and
then testing against the same target on a different split is weakly circular.

**Fix.** Tune `τ_conf` on **train**, report on **validation** (that is the honest number), and treat
the test-panel FP rate as confirmatory only. Alternatively, state the FP rate as a *design
characteristic* and drop it from H2, leaving H2 as the macro-F1 bar alone. Also report the full
abstention-vs-FP ROC (§6.4 already asks for it — good).

### I-8 · §15.7 — The oracle-regret horizon is unspecified

"Fork the simulator from a decision point and run every arm from the same state" — over what horizon?
R7 (a safety-factor change) pays off over many periods; R0's benefit is one period. A one-period
regret systematically **flatters R0 and penalises R7**, biasing the very comparison the metric was
added to strengthen.

**Fix.** Specify a horizon of at least `R = L + 1` periods, use a discounted multi-period cost, and
state the horizon and discount in the paper. **Also state explicitly** that the future demand draws
must be *identical* across arms (guaranteed by the `(panel_seed, sku_id, period)` seeding of §15.6,
but it must be asserted as a requirement of the oracle, not left implicit — a naive implementation
using a running RNG stream would silently break it).

### I-9 · §20 — E6 is a 1,728-cell full factorial

Six factors: `4 (B/H) × 3 (α) × 3 (L) × 4 (d) × 3 (τ_conf) × 4 (κ) = 1,728` combinations × 4 arms.
At even one minute per run that is over 100 hours. §20 says "sweeps" without specifying the design.

**Fix.** Specify **one-factor-at-a-time** sweeps around the frozen default (6 sweeps × 4 arms ≈ 24
runs), with at most one or two two-way heatmaps for the interactions P8 actually plots
(`B/H × α` and `L × d`). Say so in §20 and in T8.

### I-10 · §30 — Four step gates have no numeric pass criterion

| Step | Current "done when" | Problem |
|---|---|---|
| 1 | "the simulator sanity test passes" | the test itself is unpassable (C-9) |
| 2 | "match `statsforecast` within **tolerance**" | tolerance unspecified |
| 5 | "statistics **visibly** separate" | not decidable by an implementer |
| 7 | "**measurably** changes what it claims to change" | no threshold |

**Fix.** §F below supplies numbers for steps 1–5. Steps 7 and 2 need the same treatment; proposed:
step 2 tolerance = relative `1e-6` on a fixed series with a fixed seed; step 7 = each remedy must move
its target statistic by **≥ 1 standard deviation of that statistic on the C0 control panel**, in the
predicted direction.

### I-11 · §15.3 — The ambiguous panel's injection gap is shorter than any monitor window

Injections at `τ` and `τ + 6`, against monitor horizons of 24–60 periods. The monitor cannot see the
two events as separate; it sees a mixture. So the panel tests *"can you tell a mixture from a pure
cause"*, not *"can you detect ambiguity between two distinguishable mechanisms"* — which is what §8's
"ambiguous / multi-cause" no-action rule claims.

**Fix.** Either increase the gap to `≥ 2 × W_class` (≈ 60 periods, which requires extending `T`), or
restate the panel's purpose honestly as a **mixture-detection** panel. The audit recommends the
restatement — it is cheaper and the mixture case is arguably the more realistic one.

### I-12 · §22.1 / §15.1 — The bootstrap must cluster on the generator's stratification cell, not only on SKU

§22.1 correctly clusters the bootstrap by SKU. But §15.1 samples `(p, μ_z, k)` **stratified** to fill
the class quadrants — which means multiple SKUs share a small set of parameter cells. SKU-clustering
handles within-SKU serial correlation but **not** within-parameter-cell correlation. With a modest
number of distinct parameter cells, the effective sample size is far below 5,000, and the reported
CIs will be too narrow.

**Fix.** Record the generator's parameter cell per SKU, and either cluster on the **cell** (the
conservative choice) or report both clustered-by-SKU and clustered-by-cell CIs and state which the
headline uses.

### I-13 · §2 / §29 — Residual novelty-collapse risk that the spec does not flag

The audit checked each element of the four-part seam against the cited prior art and agrees the
*combination* is uncovered. But it found a collapse risk the spec does not name:

**If the paper's headline is (f) vs (c) — cost-optimising bandit vs accuracy-based selection — it
collapses into "cost-aware model selection", which is adjacent to Mikkonen & Saarinen (cost-weighted
loss, §2's cite list) and AI-BA (service-aligned quantile selection).** A reviewer will find those.

Separately, **R1 is intercept correction** (Clements & Hendry, and Lyubchik & Grinberg's adaptive
bias-corrected Croston — both already in `10` §2), and **R6 is online Syntetos–Boylan reclassification**.
Their novelty is thin and the spec leans on them less than it could, which is correct — but the paper
must not present them as contributions.

**Fix (framing, not redesign).** The defensible core is:
1. **C7/R7** — a *non-forecast* remedy triggered by a *diagnosis*, learned from service feedback;
2. **the C3/C4 sub-process decomposition** (GAP-3), which the spec already correctly calls its
   sharpest sub-piece;
3. **the equal-churn evaluation methodology** (H1′), including the matched-churn construction itself;
4. **the C-guard** as the response to censoring.

State this hierarchy explicitly in §3 and §29. Do not let the paper's framing rest on (f) > (c).

---

## D. WHAT IS ALREADY METHODOLOGICALLY STRONG — do not change these

The audit looked for reasons to attack each of these and could not find one. **Leave them alone.**

1. **The C-guard concept (§6.2).** Correct, necessary, and the direct technical answer to ISNCC 2026.
   The *scope* is wrong (C-6, C-10), but the idea and its placement as a mandatory pre-step are right.
2. **The novelty discipline (§2.2, §3.1).** The scope sentence, the refusal to claim "first ever",
   the explicit "do not claim the taxonomy concept" for Clements & Hendry, and the readme-style
   honesty about the remedies not being novel ("CAFR need not invent the remedies — only the selection
   of them") are exactly right. **This is better prior-art handling than most published papers.**
3. **The A5 ablation (§21).** The *concept* — same action space, accuracy reward, inside our own
   harness — is the correct and cleanest way to make the paper's central contrast measurable rather
   than argued. It needs the scale fix (I-3) but not a rethink.
4. **A4 exists at all (§21).** The spec already identifies it as "a hard test of the paper's core
   claim". The audit only asks that it be *promoted* (C-1), not invented.
5. **The negative control (§22.4 item 6).** "Arm (c) vs itself, different seeds → null difference.
   Run this before trusting any positive result." Correct instinct, correctly placed.
6. **The equal-churn methodology (§11.3).** Matching on validation, comparing on test; imposing the
   constraint on the *comparator*; reporting the realised rather than the target churn; and the second
   assumption-free matched-switch-count method. The spec's "why this is not circular" paragraph is
   precisely the paragraph a reviewer looks for, and it is already written.
   *(Recommendation: make the per-SKU matched-switch-count method **primary** and mean-matching the
   robustness check — matching a mean does not match a skewed distribution, and (f)'s and (c)'s churn
   distributions differ in shape.)*
7. **The no-look-ahead rules as a *concept* (§14.3).** Rules 1–4 are correct and well chosen. Only
   rule 5's implementation is wrong (C-11).
8. **The pre-registration discipline (§14.4, §29).** Writing the criteria down before the test run and
   refusing to re-tune afterwards is exactly right, and §29's "if the experiment does not show this,
   the claim fails and the paper says so" is the sentence that makes this science.
9. **The two implementation checkpoints (§30).** Stopping for the simulator sanity test and stopping
   again to plot M1 feature separation *before* writing M2 are the right two places, for the right
   reasons. They only need numeric criteria (I-10).
10. **"Do not handicap arm (c)" (§12).** Correct, and it is the note most likely to save the paper.
11. **"Report all ablations, including null results" (§21).** Correct.
12. **The C3/C4 sub-process decomposition (§6.3).** Genuinely the sharpest technical sub-piece, and
    the spec correctly identifies it as such. It survives this audit intact.
13. **The reproducibility requirements (§28).** Single entry point, pinned deps, seeded everywhere,
    config+results travelling together, `DATA.md` with SHA-256 and licence text, "no notebook-sourced
    numbers". Nothing to add.
14. **The freeze/no-GPU/OSS-only stack and the CPU feasibility claim (§26).** Correct: 5,000 SKUs ×
    84 periods × 9 arms is small. The only infeasibility found was E6's factorial (I-9), not the stack.
15. **The honest-citation section for the no-action precedent (§8).** Citing Zenodo
    `10.5281/zenodo.18364003` as "the one precedent for the *idea*. Do not claim it as our idea." is
    exactly the right call.

---

## E. FOUR DECISIONS THAT MUST BE FROZEN BEFORE CODING

Each is stated as a **freeze**, with the audit's reasoning. The team may overrule; it may not leave
these open.

### OD-2 — C8 vs censoring guard → **FREEZE: guard only. No C8.**

**Why.** (a) C7 already claims the policy root; a separate C8 double-counts the same mechanism and
would make C7 and C8 mutually unfalsifiable. (b) Adding an eighth cause changes the macro-F1
denominator, the priority order, the ablation set and the arm count (9 → 10) — all of which are
already specified. (c) The guard's value is already measured by ablation A6, so C8 adds no new
measurable claim.

**But freeze it *together with* the C-10 fix.** The audit's finding is that the guard as specified is
**incomplete**: it protects the statistics, not the forecaster's training input. So the freeze is:

> **Censoring is handled by (i) the statistical C-guard on bias/CUSUM/size statistics, and (ii) a
> specified, identical fitting input for every forecaster in every arm — recommended `uncensored`
> (impute lost sales before fitting). No C8 cause is added. A6 is split into A6a (statistical guard)
> and A6b (fitting input).**

Without part (ii), freezing OD-2 as "guard only" freezes a guard that does not cover the loop.

### OD-3 — Reward formulation → **FREEZE: default standardised cost, headline; option (b) as a robustness condition.**

> `r_{i,t} = − [ ( H·Ī_{i,t+1} + B·B̄_{i,t+1} ) − C̄_i ] / σ_{C,i}`

**Add two clauses:**

1. **Floor `σ_{C,i}`** — `σ_{C,i} ← max(σ_{C,i}, 0.05 · C̄_i)` — or the reward explodes for any SKU
   with near-constant cost, which will happen on stable, low-variance SKUs.
2. **The same standardisation must be applied in A5** (I-3), or the paper's sharpest ablation is
   scale-confounded.

Option (b) (service-gap-minus-cost-penalty) stays as a **pre-registered robustness condition**: if
the headline claim holds only under the default reward shape, the paper reports that as a finding.

### OD-4 — Free vs constrained bandit → **FREEZE: free bandit (f) as headline; (f′) constrained and A4 no-attribution as *required* comparators.**

The audit's change: §9.2 frames OD-4 as a two-way choice whose resolution "changes the meaning of the
headline result". That is true, but C-1 shows the **more important** comparator is not (f′) — it is
**A4, the free bandit with attribution features removed from the context**.

> **Freeze: (f) = free bandit with attribution context. Required comparators, all pre-registered:
> (f′) = attribution-constrained bandit; A4 = free bandit, no attribution context, same reward, same
> action space; (g) = fixed table. Report policy–table agreement (§19) for all three.**

A4 is what makes the headline interpretable. Without it, (f) > (c) says only "cost beats accuracy",
which §1's own literature predicts.

### OD-9 — Oracle regret → **FREEZE: include, at 5,000 points, with a specified horizon.**

**Conditions on the freeze:**

1. **Horizon:** ≥ `R = L + 1` periods, discounted; state both in the paper. A one-period horizon
   flatters R0 and penalises R7 (I-8).
2. **Identical future demand draws across arms** — asserted as an explicit requirement of the oracle,
   not assumed from the seeding rule.
3. **Stratify the 5,000 sampled points by cause and by panel**, and log them so the sample is auditable.
4. If compute blocks, reduce to 1,000 points **before** dropping the metric. The metric is worth the
   compute: it converts "CAFR beats (c)" into "CAFR is *this far* from optimal", which is a materially
   stronger and more credible claim, and it is unavailable on real data.

**Note:** OD-9 should also be **renamed in the paper**. "Regret" against an oracle over a *sampled*
set of decision points with a *finite* horizon is a **sampled finite-horizon regret estimate**, not
regret in the online-learning sense. Saying so pre-empts an easy reviewer objection.

---

## F. STEP 1–5 GO/NO-GO CHECKLIST

Pass criteria are numeric and decidable by an implementer. **A step is not "done" until every box in
its row passes.** Where a criterion is a *new* requirement created by this audit, it is marked
**[audit]**.

### Step 1 — Inventory simulator (M5)

| # | Criterion | Threshold |
|---|---|---|
| 1.1 | Determinism: same config + seed twice | **byte-identical** `inventory_log.parquet` |
| 1.2 | **[audit]** DGP-aware oracle forecaster (supplies the *true* predictive distribution of R-period aggregate demand), correctly-set policy | achieved CSL within **±0.02** of target, for each α ∈ {0.80, 0.90, 0.95}, over ≥ 500 SKUs × 200 periods |
| 1.3 | **[audit]** *Negative half of the same test* — safety stock set to zero | achieved CSL ≈ **0.50 ± 0.03**. *This proves the test can fail.* |
| 1.4 | Censoring behaves | with a deliberately under-set policy, `demand_lost > 0` and `demand_observed < demand_true` on ≥ 95 % of stock-out periods |
| 1.5 | **[audit]** Order-up-to uses the **R-period aggregate** quantile, not `z·σ_period` | `S` at α = 0.90 exceeds `R·μ̂` by the empirical 90th percentile of `A^agg − R·μ̂` within **±2 %** |
| 1.6 | **[audit]** Fitting input switch exists and is logged | `fit_on ∈ {observed, uncensored, true}` appears in `config.yaml` and in T7 |

**Do not start step 2 until 1.1–1.6 pass.** §30 is right that everything downstream inherits these bugs.

### Step 2 — Forecaster pool (M0)

| # | Criterion | Threshold |
|---|---|---|
| 2.1 | Croston / SBA / TSB / mSBA / mTSB / SES vs `statsforecast`, fixed 200-period series, fixed seed | relative error **≤ 1e-6** |
| 2.2 | Point forecasts non-negative; quantiles monotone in α | no violations over the suite |
| 2.3 | **[audit]** TSB's occurrence probability decays toward zero on an all-zero series | final `ẑ < 0.01` after 200 zero periods |
| 2.4 | LightGBM is deterministic | fixed seed + `n_jobs=1` (or a documented deterministic reduction); two runs byte-identical |

### Step 3 — Baselines (a), (b), (c), (e)

| # | Criterion | Threshold |
|---|---|---|
| 3.1 | All four arms run end-to-end on RUF through the rolling-origin harness | T2's forecast block produced, no NaNs |
| 3.2 | **[audit]** Arm (c) is genuinely adaptive — guard against silent degeneration to arm (a) | at margin `m = 0`, **≥ 30 % of SKUs switch at least once** |
| 3.3 | **[audit]** Arm (b)'s SB mapping is not degenerate | no single SB class contains **> 90 %** of RUF SKUs; mapping table printed in T7 |
| 3.4 | **Negative control** (§22.4 item 6) | arm (c) vs itself, seeds A ≠ B → cost difference **95 % CI includes zero** |
| 3.5 | **[audit]** Censoring-feedback diagnostic logged | `% of windows where the censoring loop changed the attributed cause` reported (C-10, item 5) |

### Step 4 — Labelled synthetic panel

This is the gate that the current §15.2 **fails**, so it is stated most fully.

| # | Criterion | Threshold |
|---|---|---|
| 4.1 | Deterministic regeneration from seed | byte-identical `ground_truth.parquet` |
| 4.2 | **[audit]** Class counts on the *corrected* class definition (C-4) | all four (ADI-band × CV²-band) cells **≥ 20 %**; print the exact split. **The "four SB classes" requirement is dropped — it is impossible at p ≤ 0.5** |
| 4.3 | **Targeted separation** — for every cause C, C's targeted statistic separates C's series from the C0 control | **AUC ≥ 0.80** |
| 4.4 | **Confound test [audit] — the criterion the current design fails.** For every cause C and every statistic listed in §15.2's "deliberately does NOT move" column, that statistic must **not** separate C's series from C0 | **AUC ≤ 0.65** |
| 4.5 | **[audit]** Cross-cause separation: each cause's targeted statistic must not separate *other* causes' series from C0 above the same bar | **AUC ≤ 0.65** for all C′ ≠ C |
| 4.6 | **[audit]** C5-specific: the CUSUM's standardised shift for C5's injection exceeds the slack `k` | **shift ≥ 2·k**; recompute for the full sampled `(p, μ_z, k_shape)` range and report the worst case |
| 4.7 | **[audit]** C7-specific: the forecast-acceptability gate passes post-injection | on **≥ 80 %** of C7 post-injection windows |
| 4.8 | **[audit]** The censoring rate on C7 series does not starve the window | `|U_i(t)|` ≥ 20 on **≥ 80 %** of C7 windows under the corrected multi-scale horizons (C-7) |

**If 4.3–4.8 do not pass, STOP. Revise §15.2's injection table. Do not proceed to M1/M2 and do not
compensate by tuning detectors** — that is fitting code to a broken ground truth, and it produces a
confusion matrix that measures the injection design.

### Step 5 — M1 feature separation

| # | Criterion | Threshold |
|---|---|---|
| 5.1 | Priority-ordered detector, per-cause **recall** on the labelled panel | **≥ 0.60** for every cause C1–C7 |
| 5.2 | False-positive rate on the C0 control panel | **≤ 10 %** |
| 5.3 | **[audit]** No cause absorbs another: no off-diagonal cell in the row-normalised confusion matrix exceeds | **40 %** |
| 5.4 | **[audit]** Priority-order diagnostic: re-run with a shuffled priority order and report the macro-F1 difference | reported; if the difference is **> 0.10**, the attribution is measuring the ordering, not the detectors — revise §6.4 |
| 5.5 | **[audit]** Plot P3 (confusion matrix) and the per-cause statistic distributions | produced; **"visibly separates" is replaced by 5.1–5.4** |

**Deliberate note:** 0.60 per-cause recall is *below* H2's 0.70 macro-F1 bar. That is intentional —
step 5 is a go/no-go on the **injection + monitor**, not the final result. Do not raise it.

**If step 5 does not pass, STOP and revise §15.2 — not `attribution/rules.py`.** §30 says this and it
is correct; the audit only supplies the number.

---

## G. FINAL IMPLEMENTATION INSTRUCTION

### What Himanshi implements first

**Two workstreams, in parallel, both gated.**

**Workstream 1 — steps 0 and 1, in this order:**

1. Repo skeleton, `config.yaml`, `requirements.txt`, and **the corrected schema of §27** (C-11: exact
   column whitelist; `ground_truth.parquet` in a separate file the policy loader cannot open;
   `sb_class_init` + numeric `adi_init`/`cv2_init` instead of two class strings).
2. **The inventory simulator**, implementing `Q̂_α`-based order-up-to (C-9), the `fit_on` switch
   (C-10), and the corrected sanity test (Step 1.2/1.3).
3. **Stop at the Step-1 gate.** Run all six criteria. Do not proceed on a partial pass.

**Workstream 2 — step 4's generator, which has no dependency on the simulator:**
Once C-3 (C6's redefinition) and C-2 (decoupling C2/C4/C6) are resolved on paper, build the generator
and the injection table as pure data-generation code, plus the separation and confound tests
(Step 4.3–4.6). This can run while the simulator is being fixed.

That is the correct ordering because **the injection table is the ground truth for every number in
the paper**, and it currently contains an impossibility (C-3) and a three-way collapse (C-2).

### What she must NOT implement until the gates pass

| Must not build | Until | Why |
|---|---|---|
| **M2 attribution** (§30 step 6) | Step-4 gate 4.3–4.8 passes | If the injections do not separate, no detector can. Tuning rules here is fitting code to a broken ground truth. |
| **M4 bandit** (step 9) | OD-3 and OD-4 frozen **and** the burn-in bias (I-1) and posterior-staleness (I-2) fixes are specified | Otherwise the headline comparison is handicapped against (f) and uninterpretable |
| **E4 / any headline run** (step 10) | **A4 exists** (C-1) | Without A4, (f) > (c) says only "cost beats accuracy", which §1's own literature predicts |
| **The pre-registration file** (step 14, §14.4) | §29 is rewritten with the corrected H1 / H1′ structure | Pre-registering the current H1 pre-registers a tautology |
| **R7 / the C7 controller** (step 7) | §7.1 and §15.6 are made coherent (C-8) | The remedy is currently undefined in the simulator's units |
| **Any C5/C7 result** | C-5's CUSUM split and C-6's three fixes | Both will silently return ~0 recall, and §29.4 would report that as "the mechanisms are not separable" — a false negative stated as a scientific finding |

### The one-line summary for her

> Build the simulator and the generator first, and make their gates pass **numerically**. Do not write
> a single line of attribution or bandit code until the injections demonstrably separate — because if
> they do not, the code cannot fix it, and the paper will report a broken generator as a scientific
> negative result.

### Escalations that are not hers to decide

Everything in §E, plus **C-2, C-3, C-4 and C-6**, which change what the ground truth *is* and
therefore what the paper claims. These are team/guide decisions. She should be told explicitly that
**the injection table she is handed may change after this audit**, and that she should build the
generator so the injection parameters are config-driven rather than hard-coded — so that a
redefinition is a YAML edit, not a rewrite.

---

## AUDIT SUMMARY TABLE

| ID | Section | Issue | Severity | Before coding? |
|---|---|---|---|---|
| C-1 | §29.1, §21 | H1 is near-tautological; A4 must be co-primary | Critical | **Yes** |
| C-2 | §15.2, §6.4 | C2/C4/C6 are one intervention, three labels | Critical | **Yes** |
| C-3 | §15.2, §6.3 | C6's ADI crossing is mathematically impossible | Critical | **Yes** |
| C-4 | §15.1 | The four SB classes cannot be generated | Critical | **Yes** |
| C-5 | §6.3 | CUSUM on the pooled residual false-alarms on zero runs; C5's step is below its own slack | Critical | **Yes** |
| C-6 | §6.3, §15.2, §15.6 | C7 cannot fire — three independent causes | Critical | **Yes** |
| C-7 | §6.3, §6.1 | `W = 12` contradicts four detectors' own evidence requirements | Critical | **Yes** |
| C-8 | §7.1 vs §15.6 | The C7 controller and the simulator use different error-distribution conventions | Critical | **Yes** |
| C-9 | §15.6, §28.6 | `(S,S)` mis-scaled; the sanity test is unpassable as written | Critical | **Yes** |
| C-10 | §14.3, §15.6 | The forecaster's fitting input is unspecified — an unmeasured policy→forecaster loop | Critical | **Yes** |
| C-11 | §27, §14.3 | Full-series `sku_meta` classes leak; the no-look-ahead test misses `cause_active` | Critical | **Yes** |
| I-1 | §9.4 | Burn-in biases the headline against (f) | Important | No — before E4 |
| I-2 | §9.1 | Stationary TS against a non-stationary problem | Important | No — before E4 |
| I-3 | §21 | A5 is confounded by reward scale | Important | No — before ablations |
| I-4 | §29.1 | Criterion 3 is too strong; ROPE thresholds inconsistent (1 % vs 2 %) | Important | No — before pre-registration |
| I-5 | §19, §29.2 | Macro-F1 denominator undefined; including C0 rewards blanket abstention | Important | No — before E1 |
| I-6 | §29.2 | H4 is descriptive, not a hypothesis | Important | No — before pre-registration |
| I-7 | §6.4, §29.2 | The ≤ 10 % FP criterion is a tuning target, not a test | Important | No |
| I-8 | §15.7 | Oracle-regret horizon unspecified; identical-draws requirement unstated | Important | No — before E7 |
| I-9 | §20 | E6 is a 1,728-cell full factorial | Important | No — before E6 |
| I-10 | §30 | Four step gates have no numeric criterion | Important | Partly (steps 1, 2) |
| I-11 | §15.3 | Ambiguous-panel gap is shorter than any monitor window | Important | No |
| I-12 | §22.1 | Bootstrap must cluster on the generator's stratification cell, not only SKU | Important | No |
| I-13 | §2, §29 | Residual novelty-collapse risk into cost-aware model selection is unflagged | Important | No — framing |

**Statement of audit limits.** This audit examined the specification's internal consistency, its
mathematics, and its methodological exposure to leakage, confounding, circularity and evaluation
invalidity. It did **not** run any code, generate any data, or verify any citation — and it did not
re-search the literature, per the standing instruction. The arithmetic findings (C-3, C-4, C-5, C-7)
are verifiable by hand from the stated parameters. The design findings (C-1, C-2, C-6, C-8, C-10)
require a team decision, not a proof. **No part of this audit assumes CAFR works, and several findings
suggest that, uncorrected, it would appear not to.**

---

*Audit prepared 2026-09-23 against `11_technical_specification.md` Rev. 1. `11_technical_specification.md`
was not modified. No experiments were run; no results appear in this document.*
