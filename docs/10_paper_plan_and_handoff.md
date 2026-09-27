# 10 — PAPER PLAN & HANDOFF

**Date:** 2026-09-23 · **For:** the next chat, and then the research paper
**Read first:** `08_final_novelty_search.md` (analysis) → `09_round3_confirmation_search.md` (evidence log) → this file (what to do next)

---

## 0. STATUS — THE SEARCH IS COMPLETE

**The novelty search is FINISHED, including IEEE.** Do not re-run it. 64 OpenAlex Boolean queries, **14 IEEE queries**, ~27 web searches, Crossref DOI checks.

| Source | Status |
|---|---|
| OpenAlex M21–M38 | ✅ Complete (only **M32** never returned — treat as partially run, never a clean zero) |
| **IEEE Xplore, 8 round-3 queries** | ✅ **Complete** — 5 clean zeros against a 255-hit positive control |
| Web search | ✅ Complete — nothing remains here |
| **Scopus / Web of Science** | ❌ **Never searched — no access on this machine. Do not claim them in the paper.** |

**Nothing is outstanding. The next step is writing the paper — §3 below.**

**Note for future re-runs only:** IEEE Xplore detects **headless** Chrome and returns HTTP 418. Run `ieee_round2.py` with a **visible** window (`--headless` must stay removed). No login needed.

---

## 1. THE ONE-SENTENCE CONTRIBUTION (do not weaken or inflate this)

> **We introduce a cause-attributed forecast-repair layer for intermittent spare-parts demand that classifies each forecast failure by mechanism — separating occurrence failure, size failure, level shift, intermittency change, uncertainty mis-calibration, and inventory-policy mis-set — and selects the matched corrective action, where the cause→remedy mapping is learned from downstream inventory service-level feedback rather than from forecast accuracy.**

**Falsifiable claim (the paper's centrepiece):** *at equal model churn, cause-attributed repair beats accuracy-based model selection on the inventory cost–service frontier.*

If the experiment does not show this, **the claim fails — say so in the paper**. Agree this in advance; it is what makes the work science rather than a demo.

**Honest scoping sentence for the introduction:**
> No directly matching work was found in the searched databases (OpenAlex, IEEE Xplore, Crossref, web search); Scopus and Web of Science were not searched. The individual components exist separately — in econometric forecast-error taxonomy (Clements & Hendry), intermittent-demand theory (Syntetos–Boylan, Croston), decision-aware benchmarking (arXiv 2609.13840, ICDM-26), bandit-driven forecast correction (PtC, arXiv 2607.16354), and root-cause-to-action inventory patents (US20100125489) — and each of those holds at most two of CAFR's four elements.

**Never write** "100% unique" or "nobody has done this".

---

## 2. RELATED WORK — THE FULL CITE-AND-DISTINGUISH LIST

**⚠️ Verify every DOI against the publisher before submission.** Items marked *(r2)* were gathered in the round-2 session and have **not** been re-checked in round 3. Items marked *(r3)* were verified this session.

### 2a. The four that must be cited FIRST (a reviewer will raise these)

| Work | DOI / ID | Must say |
|---|---|---|
| **PtC** — *A Predict-then-Correct Loop Based on Few-Shot Continuous Contextual Bandit for Demand Forecasting* (Lei, Ma & Jackson, 2026) *(r3)* | arXiv `2607.16354` | The closest neighbour. Action = **one continuous multiplier** clipped to [−1,2]; reward = **forecast error** (`r_t = τ(e_ML,t − e_CB,t)`); **no attribution**; **lumpy SKUs excluded**. It is a degenerate single-action case of our action space. |
| **Patent US20100125489** — *System and Method for Root Cause Analysis and Early Warning of Inventory Problems* *(r3)* | US `20100125489` | Cause → forecast **and** "inventory policy parameters (such as safety stock targets)" → "self-tuning". But: not intermittent/spare parts; root-cause only in **dependent** claim 7/14; human confirms by consensus; rule-based. |
| **Clements & Hendry** forecast-error taxonomies *(r2)* | `10.1007/978-1-4614-1653-1_9` (+ Clements & Hendry 2000; Hendry & Mizon 2011) | 9 error sources; structural change is the prime culprit; **intercept corrections** are an established cause-matched remedy. **Do not claim the taxonomy concept** — we *operationalise* it for intermittent demand + inventory. |
| **Manary, Willems & Shihata**, *Interfaces* 2009 (Intel) *(r2)* | — | Diagnose-then-fix in demand planning. But: not intermittent, no cause taxonomy, no automated loop, no inventory feedback, human-authored. |

### 2b. The "accuracy ≠ service" cluster (this is taken — cite, do not claim)

| Work | DOI / ID | Note |
|---|---|---|
| *Accuracy Is Not Service: A Decision-Aware Benchmark for Intermittent-Demand Forecasting* *(r3)* | arXiv `2609.13840` — **accepted ICDM-26** | Spearman ρ = −0.555 accuracy rank vs contractual service target across 38 methods. Ships **RUF** + the fix worth +14.5/+17.1 pts fill. **Peer-reviewed — cite as such.** |
| *Bridging Forecast Accuracy and Inventory KPIs* (Alabdallah et al., 2026) *(r2)* | arXiv `2601.21844`; also `10.1007/978-3-032-23833-7_32` | Closed-loop simulation; accuracy gains do not transfer; Simpson's paradox. Open source generator — **needed for our ground-truth-cause panel**. |
| *Cost-Aligned Deep Learning for Intermittent Spare-Parts Demand and Inventory* (Mikkonen & Saarinen, 2025) *(r2)* | SSRN `5554492` | Asymmetric cost-weighted loss, CSL sweep 75–95 %. |
| **AI-BA** — regime-specific adaptive ensemble *(r2)* | `10.1145/3789982.3790048` | ADI/CV² → quantile forecasts aligned to service/reorder. |
| **LOWDII** *(r3)* | arXiv `2607.19835` | Error-distribution diagnostic → safety stock → target service at 5.1–22.6 % less stock. Statistical, **not causal**; FMCG. Closest thing to our *cause-7* machinery. |
| **Stock-Out Distortion** *(r3, IEEE)* | `10.1109/ISNCC70543.2026.11693859` | **Best "the field is starting to ask why" citation.** Diagnoses stock-out censoring → under-forecasting across SARIMA/XGBoost/CNN-LSTM. Then **stops at the diagnosis**. Use its own finding that the effect **weakens on intermittent SKUs** — it applies where CAFR does not. |

### 2c. Adaptation / drift / churn machinery (taken as machinery — usable only as a control variable)

| Work | DOI / ID | Note |
|---|---|---|
| **EMC** — dual-threshold drift (δ, η), hysteresis, graded regimes *(r2)* | — | Drift detection with churn control already exists. |
| **AHSIV / AHS** — adaptive automatic model selection *(r3)* | arXiv `2602.13939` | RMSSE/MAE/sMAPE/BIAS + structural features. No non-forecast actions, **no churn control**, not intermittent. **This is our experiment arms (c)/(d).** |
| **LCMA** — LLM multi-agent, spare parts *(r3)* | `10.1016/j.knosys.2026.116334` | Classifies **demand patterns**, not failures; action space is models only. **The paradigm we argue against** — and its design *contravenes* churn control. Use as the foil. |
| **CoSFan** (ICLR 2025) *(r2)* | — | Meta-learns *what, how & when* to adapt. Not demand, not causes, not remedies. |
| **After-sales diagnostic loop** *(r3)* | arXiv `2510.01006` | "Ranked root causes", bias decomposition — but **no action set is ever enumerated**; WMAPE objective. |
| **Killi 2026** *(r2)* | `10.51483/ijaiml.6.5s.2026.548-555` | **0 references — low-quality venue. Cite descriptively only, do not build on it.** |
| **AMIT 2025** closed-loop prediction-inventory-scheduling *(r3)* | `10.62177/amit.v1i4.507` | Low-quality venue. Cite descriptively only. |

### 2d. Diagnose-then-correct in adjacent fields (the loop shape exists elsewhere)

| Work | DOI / ID | Note |
|---|---|---|
| Exception-based forecasting *(r2)* | `10.2753/mis0742-1222310209`; `10.1108/ijpdlm-02-2014-0017` | Flag → **human** review. No automated cause attribution. |
| MPC prediction-error diagnosis & remedies *(r2)* | `10.1002/asjc.782`; `10.1109/acc.2012.6314957` | Same loop shape, control engineering. Diagnosis is of the *plant*, not the forecast. |
| IJPE 2019 service-BOM alerts *(r2)* | `10.1016/j.ijpe.2019.08.001` | Corrects data/fault records with expert validation. |

### 2e. Method building blocks (ours to use, not to claim)

| Work | DOI / ID | Use |
|---|---|---|
| **NNARMA** *(r3)* | `10.1016/j.ejor.2026.06.009` | Prediction intervals for intermittent demand suitable for stock control. |
| **Sensors 2023** tensor-optimisation interval prediction *(r3)* | `10.3390/s23167182` | Interval widening exists off the shelf → **we need not invent our remedies, only the selection of them.** |
| **SHOS** *(r3)* | `10.1038/s41598-026-57129-6` | Occurrence/size as *feature engineering*, not diagnosis. |
| **Zenodo decision-aware** *(r3)* | `10.5281/zenodo.18364003` | The **only** prior statement of a no-action option ("better managed through expert oversight"). Non-peer-reviewed. |
| **Lyubchyk & Grinberg 2024** *(r2)* | SSRN `10.2139/ssrn.4608877` | Adaptive bias-corrected Croston. Closest single hit — **but no diagnosis step.** |
| Contextual bandits for inventory control *(r3)* | arXiv `2310.16096` | Confirms the bandit itself is not novel. |
| RL inventory control, non-stationary demand *(r3)* | SSRN `10.2139/ssrn.5398041` | RL over *replenishment* ≠ RL over *forecast repair*. |
| Syntetos–Boylan (ADI × CV²); Croston; SBA/TSB/mSBA/mTSB; MASE; MASEII; SPEC; PIS/NOS; BDD; Rožanec types *(r2)* | standard | Background + our classical baselines. |

### 2f. Industry practice (evidence the market wants this — strengthens, not weakens, our case)

RELEX Solutions "AI-assisted diagnostics" (product launch, Nov 2025) · ToolsGroup (diagnose long-tail drivers before changing methods; limit overrides to stop human overreaction) · PTC/Servigistics "Performance Analytics and Intelligence" · Blue Yonder "Install Base & Causal Forecasting" · Verdantis MRO360.
**No vendor publishes a method**, and none defines a failure-mechanism taxonomy or a learned cause→remedy map.

---

## 3. PAPER STRUCTURE

| Section | Content | Source |
|---|---|---|
| **1. Introduction** | Intermittent spare-parts problem → accuracy ≠ service (cite 2609.13840) → the gap: nobody asks *why* a forecast failed and picks a matched fix | file `08` §A, §D |
| **2. Related work** | Exactly §2 above. Lead with PtC and the patent. | this file §2 |
| **3. Problem setup** | Periodic review, order-up-to (S,S), target CSL ∈ {80, 90, 95 %}, cost sweep | file `08` §H |
| **4. Method (CAFR)** | M0 base pool → M1 outcome monitor → M2 attribution → M3 remedy library → M4 selection policy → M5 harness | file `08` §F |
| **5. The attribution table** | The 8 rows: cause / observable signature / matched remedy / what is new | **file `08` §F — this is the technical core** |
| **6. Datasets** | RUF primary; generator panel for ground truth; M5 second profile; RAF only if licensed | file `08` §G, `09` §2.13 |
| **7. Experiments** | Arms (a)–(h); metrics; equal-churn headline | file `08` §H |
| **8. Results** | The cost–service frontier, (c) vs (f) at matched churn | to produce |
| **9. Ablations** | (g) fixed remedy table → isolates RQ2; (h) no no-action arm → isolates churn control | file `08` §H |
| **10. Limitations & threats** | From the risk register, written honestly | file `08` §J |
| **11. Conclusion** | Restate the falsifiable claim and whether it held | — |

### The three research sub-questions (from file `08` §D)
- **RQ1 (attribution):** can these mechanisms be separated reliably from residual + inventory signals alone, without knowing the true data-generating process?
- **RQ2 (learning):** does a *learned* cause→remedy mapping beat a fixed, hand-written one?
- **RQ3 (decomposition):** how much of the gain comes from **non-forecast** remedies vs forecast-model changes?

---

## 4. TITLE

**Option 1** — keeps the submitted shape:
> *Cause-Attributed Forecast Repair for Intermittent Industrial Spare-Parts Demand: A Feedback-Driven Framework for Forecasting and Inventory Optimization*

**Option 2** — sharper:
> *Learning Which Correction to Apply: Mechanism-Attributed Forecast Failure Repair for Intermittent Spare-Parts Inventory*

**What to tell Dr. V. K. Chawla:** the scope is unchanged — still AI/ML demand forecasting and inventory optimization for industrial spare parts. What changed is the *contribution statement*: one testable sentence instead of a framework description. The submitted title does **not** need re-approval; this sharpens it.

---

## 5. VENUE

**Target a conference proceedings paper.** India-based IEEE conferences have low fees, and this work is conference-length.

**₹0 constraint:** most good journals charge an APC. The realistic no-APC route is **conference proceedings + arXiv preprint**. **Do not promise a no-APC journal.** (Preference memory: software-only, ₹0, publishable without APC.)

---

## 6. WORK PLAN FOR THREE STUDENTS

| Track | Owner | Deliverable |
|---|---|---|
| **A — Simulator + baselines** | 1 student | Periodic-review (S,S) inventory simulator; arms (a), (b), (c), (e); rolling-origin backtest over 84 periods |
| **B — Attribution + remedies** | 1 student | M1 monitor features; the rule-based attribution (auditable first); M3 remedy library; per-cause precision/recall on the synthetic panel |
| **C — Selection policy + evaluation** | 1 student | M4 contextual bandit; M5 harness; churn logging; the equal-churn comparison and the frontier figure |

**Order of work (do not skip ahead):**
1. Simulator + arms (a)–(c) working end-to-end on **RUF**.
2. **Ground-truth panel** from the 2601.21844 generator — without it, RQ1 cannot be measured at all.
3. Rule-based attribution (M2) + fixed remedy table (M3) → this is arm **(g)**, a complete paper on its own.
4. Bandit (M4) → arm **(f)**. Compare **(f)** vs **(c)** at matched churn.
5. Ablations (h).

**If time runs short, stop at step 3.** Arm (g) plus the attribution table is still a publishable, honest contribution; the learned mapping is the increment, not the whole.

---

## 7. RISK REGISTER (paper-facing — from file `08` §J)

| Risk | Severity | Defence in the paper |
|---|---|---|
| "PtC already does this" | **High** | Its four-way distinction, stated in the introduction, not a footnote |
| Patent US20100125489 | **High** | Cite; distinguish on automation + mechanism taxonomy + intermittency |
| "Just a taxonomy on an adaptive framework" | **High** | The equal-churn falsifiable claim is the defence |
| Clements–Hendry pre-emption | **High** | Frame as *operationalising* an econometric taxonomy. Never claim the concept. |
| No ground truth for attribution on real data | **Medium** | The labelled synthetic panel |
| The bandit is not novel | **Medium** | Declare it openly; novelty is the action space + reward signal |
| M32 never completed | **Low** | Report as partially run |

---

## 8. FILES

| File | What it is |
|---|---|
| `01`–`07` | Earlier rounds (gap verification, candidate hunts, database verification) |
| `08_final_novelty_search.md` | **The analysis** (A–J). Updated in round 3. |
| `09_round3_confirmation_search.md` | **The round-3 evidence log** — every query, every count, every verdict |
| **`10_paper_plan_and_handoff.md`** | **This file** — the plan for the paper |
| `11_technical_specification.md` | **Revision 1 — superseded.** 30-section technical spec + the Handoff Package for Himanshi. **Retained unmodified on disk** for the audit trail. Do not hand this version to Himanshi |
| **`11_technical_specification_rev2.md`** | **★ THE IMPLEMENTATION CONTRACT — THIS IS THE FILE TO HAND OVER.** Revision 1 with every correction in file `13` §E applied: the four blockers resolved, all 16 R-items written in, the exact §B injection table, the exact §C §29 hypotheses, and the exact §D Step 1–5 gates. Ends with a **REVISION 2 CHANGELOG** listing every section modified. **Read §5–§7** (taxonomy, detection maths, remedies — §6.3, §6.4 and §7.1 are rewritten), **§14.3** (no-look-ahead — rules 5, 6 and 7 are new), **§15** (§15.1, §15.2, §15.3, §15.6 rewritten), **§29** (rewritten), **§30.1** (the Step 1–5 gates with numbers). No results anywhere — status is still pre-implementation |
| `11_spec_page.html` | Revision 1 rendered as a **web page**. **Now out of date — regenerate from `11_technical_specification_rev2.md` before sending anything** |
| **`12_methodology_audit.md`** | **The pre-implementation review of file `11`.** Verdict: **NEEDS MAJOR METHODOLOGY CHANGES** — 11 critical issues (4 are pre-coding blockers), 13 important issues, plus the four decisions to freeze and the Step 1–5 go/no-go gates with numbers. **Read this before implementing anything.** File `11` itself was not modified |
| **`13_methodology_revision_proposal.md`** | **The fixes.** Resolves all 4 blockers + 7 critical issues (R-1…R-16), each with current spec / problem / corrected spec / formal definition / what it affects. Ends with **A** the four frozen decisions, **B** the revised §15.2 injection table, **C** the revised §29 hypotheses, **D** the revised Step 1–5 gates, **E** the exact section-by-section change list for `11`, **F** verdict: **READY TO FREEZE, conditional on §E being applied to `11`**. File `11` itself was still not modified |
| `oa_round3.py` / `oa_round3_raw.json` | OpenAlex round-3 script and raw results |
| `ieee_round2.py` | The 8 IEEE queries — **working method: visible window, never headless** |

---

*Prepared 2026-09-23. Verify all DOIs against the publisher before submission. Items marked (r2) were gathered in the round-2 session and not re-checked in round 3.*
