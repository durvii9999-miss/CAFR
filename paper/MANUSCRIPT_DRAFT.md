# Cause-Attributed Forecast Repair for Intermittent Demand: Attribution Detectability and a C2-Only Reliability-Gated Feasibility Study on a Synthetic Panel

**Authors:** [AUTHOR 1 NAME], [AUTHOR 2 NAME], [AUTHOR 3 NAME]

**Affiliations:** [DEPARTMENT, INSTITUTION, CITY, COUNTRY]

**Corresponding author:** [NAME, EMAIL ADDRESS]

**Manuscript status:** First complete draft, prepared 18 October 2026. Author, affiliation, funding, competing-interest, and repository details are placeholders to be completed before submission.

---

## 2. Abstract and keywords

**Abstract**

Intermittent demand makes forecast failure difficult to diagnose: a bad forecast can arise from a biased size process, a changed occurrence rate, a level shift, a mis-calibrated uncertainty interval, a mis-set inventory policy, or simply noise. Cause-attributed forecast repair (CAFR) proposes diagnosing the mechanism behind a forecast failure and applying a matched corrective action, including actions outside the forecasting model. This paper reports two executed experiments on a labelled synthetic panel of intermittent demand (seed 42), using the accuracy-based adaptive-selection policy (arm c) as the base policy. The first experiment measured whether a live monitor and priority-ordered attribution rule could name seven injected causes at predeclared recall, false-positive, confusion, and priority-sensitivity bars. The overall verdict was failure. Only the direct C2 panel (interval miscalibration) passed the 0.60 recall bar, with recall 0.955 (191/200; 95% Wilson interval [0.917, 0.976]). The C0 control false-positive rate was 0.450 [0.383, 0.519], far above the 0.10 bar. C1 and C3 were undetectable as generated; C4 was blocked by an unsatisfiable rule guard; C5's occurrence CUSUM was saturated; C6 failed on both generator and detector; C7 remained a documented negative. The C2b variant was a generator defect and its 0.090 recall is not a detector result. The second experiment asked whether a reliability gate trained on the Step 5 readings could decide when to apply a one-time C2 scale reset. On 40 C2 and 40 C0 test SKUs, the gate acted on 37 of 40 true C2 SKUs and one control; the control action was a no-op. Cycle service level in the C2 group rose by 6.14 percentage points (95% percentile bootstrap interval [4.20, 8.30]), while cost rose by 71.48 units, or 2.85%, with an interval that includes zero ([−20.01, 168.63]). The C0 group was exactly unchanged. Gated C2 cycle service level was 0.8625, still below the 95% target. The experiments therefore provide evidence that a C2-type calibration error can be repaired in this synthetic setting, but they do not demonstrate a cost reduction, generalisation, real-world validity, or the original full-CAFR claim that cause-attributed repair beats accuracy-based selection at equal churn. That claim was not tested and is not supported here.

**Keywords:** intermittent demand; spare-parts inventory; forecast error; cause attribution; inventory control; cycle service level; conformal prediction; reliability gating; negative results; synthetic panel

---

## 3. Introduction and research questions

Intermittent demand is common in industrial spare-parts planning: many periods have zero demand, and nonzero demands are irregular and often lumpy. Forecasting methods for this setting have a long history, from Croston's separation of occurrence and size (Croston, 1972) to the Syntetos–Boylan classification and accuracy measures (Syntetos & Boylan, 2005; Syntetos, Boylan & Croston, 2005). Inventory decisions add a second layer: the same forecast error can have very different cost and service consequences depending on lead time, holding cost, shortage penalty, and the review policy (Teunter, Syntetos & Babai, 2011; Silver, Pyke & Thomas, 2016; Zipkin, 2000).

A recurring finding in both forecasting practice and the recent literature is that forecast accuracy and inventory service are not the same objective. A model that wins on an accuracy measure can lose on cost or cycle service level, and removing a bias for accuracy's sake can sometimes reduce service (Chin, Cheng & Gunawan, 2026; Fukuhara, Alabdallah, Gunasekara & Nowaczyk, 2026; Mikkonen & Saarinen, 2025). This divergence motivates a different question from "which model has the lowest error?" It motivates asking *why* a forecast failed and whether the appropriate response is to change the model, widen an interval, adjust an inventory policy parameter, or do nothing.

Cause-attributed forecast repair (CAFR) is the name used in this project for a repair layer built around that question. In its full form, CAFR would (i) classify a forecast failure by mechanism, (ii) map each mechanism to a matched corrective action, including non-forecast actions, (iii) learn that mapping from downstream service or cost feedback rather than forecast accuracy, and (iv) control model churn explicitly. The original project plan included a contextual bandit, a churn constraint, and a comparison between full CAFR and accuracy-based adaptive selection at equal churn. That full design was **not executed as a whole**. The present paper reports the two experiments that were run, and it is explicit about what they can and cannot support.

The first experiment, Step 5, asked which injected causes the live monitor and attribution pipeline can actually name on a labelled synthetic panel. The panel contains 1,800 stock-keeping units (SKUs): 200 controls (C0), 200 SKUs for each of seven single causes (C1–C7), and 200 SKUs for a C2 variant (C2b). Each SKU has 200 periods, with the injection beginning at period 120. The pipeline is the full arm (c) pipeline: a pooled forecaster, an accuracy-based adaptive model selection rule, a rolling periodic-review inventory simulator, a monitor, and a priority-ordered attribution rule. The gates were fixed in advance: recall at least 0.60 for C1–C6, a C0 false-positive rate at most 0.10, no row-normalised off-diagonal confusion above 0.40, and a priority-sensitivity check. The overall verdict was **failure**. Only the direct C2 panel passed its recall bar. This is a substantive result about detectability, not a threshold that was relaxed after the fact.

The second experiment, Step 6, asked a narrower question. Conditional on the live monitor making a first post-injection C2 attribution, can a logistic reliability gate trained on the Step 5 readings decide whether to apply a one-time C2 scale reset? And what happens to inventory cost and cycle service level relative to the same policy with the C2 miscalibration left in place? The experiment is C2-only, one decision per SKU, and paired by SKU. It is not a six-cause repair experiment, not a test of the bandit, and not a test of full CAFR against arm (c). On the 80 test SKUs, the gate acted on 37 of 40 true C2 SKUs and one control. Cycle service level rose in the C2 group, while cost rose with an interval that includes zero. The C0 group was exactly unchanged. The result is a feasibility signal, not a confirmation.

The paper makes three contributions, each proportional to the evidence:

1. **An attribution-detectability result.** On the labelled synthetic panel, only C2 direct passes the predeclared recall bar (0.955, 191/200). The remaining causes fail for identifiable and different reasons: two injections are undetectable as generated, one rule has an unsatisfiable guard, one detector is saturated, one cause fails on both generator and detector, and one is a documented negative. The paper reports these boundaries rather than collapsing them into a single "the monitor does not work" statement.
2. **A C2-only feasibility result.** A reliability gate trained on Step 5 readings can select most true C2 SKUs in the test comparison (37/40) with one false action, and a one-time scale reset is associated with a 6.14-percentage-point cycle-CSL gain. The same intervention is associated with a 2.85% cost increase whose interval includes zero. No cost saving is demonstrated.
3. **A negative/mixed-results record.** The original full-CAFR hypothesis—cause-attributed repair beats accuracy-based selection at equal churn—is retired. The paper states what was not run, what failed, and what would be needed to test the larger claim.

The research questions are:

- **RQ1 (Step 5):** On the labelled synthetic panel, which injected causes does the live monitor and attribution pipeline detect and name at the predeclared bars, and where are the detectability boundaries?
- **RQ2 (Step 6):** When the live monitor attributes C2, can a gate trained on Step 5 readings decide whether to reset C2's interval scale, and what happens to inventory cost and cycle service level relative to the untouched baseline?

The remainder of the paper is organised as follows. Section 4 reviews related work and states the research gap. Section 5 describes the synthetic panel, injections, pipeline, and monitor. Section 6 gives the experimental design, gates, outcomes, and inference. Section 7 reports results. Section 8 discusses implications. Section 9 states limitations and threats to validity. Section 10 concludes. Section 11 covers reproducibility and data/code availability. Section 12 lists references.

---

## 4. Related work and research gap

### 4.1 Forecast-error taxonomies

Econometrics already has a general forecast-error taxonomy. Hendry (2000) organises forecast error by model component and by problem type, distinguishing sources such as estimation uncertainty, mis-specification, and change in the data-generating process. Hendry and Mizon (2013) develop an open-model version of the taxonomy. A central theme in that literature is that structural change is a major source of forecast failure, and that a matched response such as an intercept correction can be more useful than re-estimating the whole model. This is an important precedent: the idea of pairing an error source with a matched remedy is not new in general.

What the econometric taxonomy is not is a taxonomy for intermittent demand. It was developed for macroeconomic series that are typically continuous, dense, and observed without the zero-inflation and lumpiness of spare-parts demand. It also does not address inventory service, lost sales, or an online attribution loop operating on a live forecaster. The present study therefore does not claim to introduce a forecast-error taxonomy. It asks a narrower operational question: which mechanisms remain visible, to a live monitor, in an intermittent-demand panel with a known injection?

### 4.2 Intermittent demand: classification, accuracy, and inventory

Croston (1972) proposed separating the occurrence process from the nonzero-size process, a structure that remains central to intermittent-demand forecasting. Syntetos, Boylan and Croston (2005) introduced the ADI × CV² classification, which sorts series into smooth, erratic, intermittent, and lumpy cells using average demand interval (ADI) and squared coefficient of variation (CV²). Syntetos and Boylan (2005) examined the accuracy of intermittent-demand estimates and highlighted the limitations of percentage-error measures in this setting. Hyndman and Koehler (2006) provide a broader treatment of forecast-accuracy measures, including scaled errors.

These classifications describe the *series*, not the *failure*. They are used to initialise or select a method, not to explain why a particular forecast went wrong. The present study uses the same broad intermittent-demand structure but adds labelled injections so that attribution can be scored against a known cause.

On the inventory side, Teunter, Syntetos and Babai (2011) link intermittent-demand forecasting to inventory obsolescence, and standard treatments of periodic-review order-up-to policies relate the safety term to a quantile of the lead-time demand distribution (Silver, Pyke & Thomas, 2016; Zipkin, 2000). The present study uses a periodic-review, order-up-to simulator with lead time two, review period one, a protection interval of three periods, lost sales, and a cost function combining holding and shortage penalties.

### 4.3 Accuracy is not service

A growing body of work shows that forecast accuracy and inventory performance can diverge. Chin, Cheng and Gunawan (2026) report a decision-aware benchmark for intermittent-demand forecasting and find a negative rank correlation between accuracy rank and contractual service target across a large set of methods. Fukuhara, Alabdallah, Gunasekara and Nowaczyk (2026) present a simulation-based software framework that connects forecast accuracy to inventory key performance indicators, and report that accuracy gains do not automatically transfer to cost or service. Mikkonen and Saarinen (2025) study cost-aligned deep learning for intermittent spare-parts demand, using an asymmetric cost-weighted loss and a service-level sweep.

This literature is the reason the present paper evaluates the C2 remedy on both cost and cycle service level, and the reason it does not treat a forecast-accuracy improvement as sufficient evidence of operational benefit. The Step 6 result is consistent with the broader message: a calibration-oriented remedy was associated with improved service while the cost difference had an interval that includes zero.

### 4.4 Adaptive model selection and drift response

Adaptive selection is a strong baseline in this setting. González and Parada (2026) propose adaptive automatic model selection that combines observed predictive performance with structural demand information. Xiong et al. (2026) describe an LLM-empowered multi-agent framework for large-scale spare-parts forecasting in which a model-selection agent matches forecasting models to demand patterns. Nunkaew and Vitiyutt (2025) present a regime-specific adaptive ensemble for spare-parts inventory management. Lyubchyk and Grinberg (2024) propose an adaptive bias-corrected variant of Croston's method.

These methods adapt the forecasting model, but they do not attribute a failure to a mechanism and then choose among remedies that may lie outside the model. The distinction matters for the present study: arm (c), the accuracy-based adaptive-selection policy, is the base policy in both experiments. The paper does not compare CAFR with arm (c); it studies attribution and a C2 remedy *within* arm (c).

Drift detection and churn control also have established machinery, including sequential monitoring and hysteresis. The present study does not test a churn-control mechanism and makes no claim about model switching. The one-decision design in Step 6 is deliberately simpler: it asks whether a single gated action can change the outcome, not whether a full switching policy can be improved.

### 4.5 Diagnose-then-correct and non-forecast actions

Several literatures contain a diagnose-then-correct loop. Manary, Willems and Shihata (2009) describe a supply-chain optimisation effort at Intel in which heterogeneous and biased forecast error is diagnosed and a targeted correction is applied. This is an academic precedent for diagnose-then-fix in demand planning, but it is not intermittent-demand-specific, does not use an automated cause taxonomy, and does not learn the remedy from inventory feedback.

Dong, Huang, Sinha and Xu (2014) and Alftan, Kaipia, Loikkanen and Spens (2015) study exception-based forecasting and exception management. In those designs, an exception is flagged and a human reviews it; the loop is not an automated cause-attribution-and-remedy loop.

Lei, Ma and Jackson (2026) propose a predict-then-correct loop based on a few-shot continuous contextual bandit for demand forecasting. The action is a continuous adjustment factor applied to the model prediction, the reward is an accuracy improvement, and the evaluation includes M5 and an industrial beverage panel. The paper is a close neighbour and should be cited as such. It does not attribute a failure to a mechanism, its action space is one continuous correction, and its learning signal is accuracy rather than inventory service or cost. Contextual bandits have also been applied to inventory-policy evaluation and improvement (Foster, Jia & Madeka, 2023); the bandit itself is not a novel component of the present study.

A patent application, US20100125489A1, describes root-cause analysis and early warning of inventory problems, with actions that include adjustments to demand forecasts and to inventory policy parameters. The root-cause discussion appears in dependent claims and the description, the setting is not specific to intermittent demand, and the described process involves planner confirmation. It is prior art for a diagnose-then-adjust loop, but not for an automated, mechanism-level, intermittent-demand-specific repair layer.

Other recent work holds individual pieces of the puzzle. Fernández-Palacios, Ceballos and Muñoz-Ocaña (2026) propose a diagnostic that evaluates the influence of historical forecast errors on an empirical uncertainty distribution and feeds safety-stock calibration; it is a statistical refinement rather than a cause attribution, and it is evaluated in a fast-moving consumer-goods setting. Amarta and Zarlis (2026) diagnose stock-out censoring as a source of under-forecasting in multi-SKU sales forecasting and stop at the diagnosis. Sbrana and Babai (2026) study non-negative ARMA models for intermittent demand with prediction intervals intended for stock control. Hong, Ren, Li, Mao and Gao (2023) study robust interval prediction for intermittent spare-parts demand using tensor optimisation. Nathan et al. (2026) use event-preserving feature engineering for intermittent-demand forecasting. Prakash (2026) notes, in a computational notebook, that some operational processes may be better managed through expert oversight than automated forecasting. In the works reviewed here, none combines mechanism-level attribution, a matched remedy set that includes non-forecast actions, learning from inventory feedback, and explicit churn control.

### 4.6 Interval calibration and conformal prediction

Conformal prediction provides a distribution-free way to construct prediction intervals with finite-sample coverage guarantees under exchangeability (Vovk, Gammerman & Shafer, 2005). Interval calibration is therefore an established remedy for a mis-calibrated uncertainty statement, and the present study treats it as one remedy rather than as a novel contribution.

The C2 injection in this study is deliberately a calibration error: the demand process is untouched, and a per-SKU scale γ is applied to the forecaster's reported interval. The Step 6 remedy resets that scale to 1.0. This is a narrow, well-defined intervention. It is not a general conformal recalibration algorithm, and the paper does not claim that it is.

### 4.7 Research gap and positioning

The literature search for this study covered OpenAlex, IEEE Xplore, Crossref, and web search; Scopus and Web of Science were not searched. No directly matching work was found in the searched databases. That statement is a statement about the search, not proof of novelty. The individual components exist separately: econometric forecast-error taxonomies (Hendry, 2000; Hendry & Mizon, 2013), intermittent-demand theory and classification (Croston, 1972; Syntetos & Boylan, 2005; Syntetos, Boylan & Croston, 2005), decision-aware benchmarking (Chin et al., 2026; Fukuhara et al., 2026), bandit-based forecast correction (Lei et al., 2026), interval calibration (Vovk et al., 2005; Hong et al., 2023), and root-cause-to-action inventory patents (US20100125489A1). Each holds at most a part of the proposed CAFR loop.

The gap that motivated the project was the intersection of four elements: (i) mechanism-level attribution of *why* an intermittent-demand forecast failed; (ii) a matched remedy set that includes non-forecast actions; (iii) learning or selecting the remedy from inventory service or cost feedback; and (iv) explicit control of model churn. The executed experiments do not test that full intersection. They test the first element directly (Step 5) and one remedy-selection question inside the second element (Step 6); the third and fourth elements are not tested. The paper therefore describes the gap as motivation, not as a demonstrated contribution.

---

## 5. Materials and methods

### 5.1 Study scope

Two experiments were run on a synthetic panel. Step 5 measured attribution gates on the full labelled panel. Step 6 measured a C2-only, reliability-gated repair on a held-out subset of that panel. Both experiments use arm (c), the accuracy-based adaptive-selection policy, as the base policy. Neither experiment compares full CAFR with arm (c), and neither experiment tests the bandit, churn control, multi-cause remedy selection, or real-data transfer.

### 5.2 Synthetic panel

The panel is generated by the project's synthetic data generator. The baseline demand process for each SKU is intermittent:

$$
\delta_t \sim \mathrm{Bernoulli}(p), \qquad z_t \sim \mathrm{Gamma}(\mathrm{shape}=k,\ \mathrm{scale}=\mu_z/k), \qquad y_t = \delta_t z_t .
$$

Each SKU is assigned to one of four reachable ADI × CV² cells. For the moderate-ADI cells, $p$ is drawn from 0.2501–0.50; for the high-ADI cells, $p$ is drawn from 0.05–0.25. The Gamma shape $k$ is drawn from the corresponding low- or high-dispersion range, and $\mu_z$ is uniform on [5, 200]. Per-SKU statics are lead time in {1, 2, 4}, unit cost from a lognormal(2, 0.5), and holding rate uniform on [0.05, 0.30].

The panel has 200 periods per SKU, with the injection time $\tau_{\mathrm{inj}} = 120$ and a burn-in of 24 periods. The seed is 42. The generator produced 2,000 SKUs: 200 controls (C0), 200 for each of C1–C7, 200 for the C2b variant, and 200 for a C3 × C2 mixture. The 200 mixture SKUs are excluded from Step 5 because they are ambiguous for single-cause scoring. The analysed panel therefore contains 1,800 SKUs.

Each series is generated from `SeedSequence(seed, spawn_key=(panel_id, index))`, so its draws do not depend on how many other series are generated. The panel is a demand panel: `demand_observed = demand_true`, and `stockout_flag` is False at the panel level. Stock-outs are created only in the simulator's counterfactual cost layer. The policy observes demand, and the unmet-demand model is lost sales.

The pooled forecaster is fitted once on the first 24 periods of all 1,800 Step 5 SKUs. This includes the SKUs later used as test SKUs in Step 6. Step 5 has no train/validation/test split; all 1,800 SKUs are measured. The exposure of the pooled forecaster to test SKUs is a limitation and is treated as such in Section 9.

### 5.3 Injected causes

The gate results are reported in Section 7; the injections themselves are summarised here.

| Cause | What the injection changes | Key parameters |
|---|---|---|
| C0 | Nothing; control | No injection |
| C1 | Size mean $\mu_z$: linear ramp over 40 periods after $\tau$ | $\phi \in \{-0.4,-0.2,+0.2,+0.4\}$; $p$ and $k$ untouched |
| C2 | Forecaster's reported interval, not demand: $\tilde{Q} = \hat{\mu} + \gamma(Q_\alpha - \hat{\mu})$ | $\gamma$ solved per SKU; target coverage gap 0.12 at $\alpha=0.95$, $R=3$; bounds (0.02, 8.0) |
| C2b | Intended modelling-error variant of C2 | Per-period sigma applied without the $\sqrt{R}$ correction; longest calibration window that clears the 0.12 target. Generator defect; see below |
| C3 | Occurrence probability $p$: step at $\tau$ | $p_{\mathrm{new}} = \mathrm{clip}(p \pm 0.15, 0.03, 0.60)$; sizes untouched |
| C4 | Size dispersion: $k \to k \times \mathrm{factor}$ at fixed $\mu_z$ | factor $\in \{0.25, 4.0\}$; $k$ resampled to stay in the same CV² band |
| C5 | Abrupt step in $\mu_z$ or $p$ at $\tau$ | Step scaled to the detectability budget; occurrence branch used only when feasible |
| C6 | Slow drift in $p$ over 60 periods after $\tau$ | Ends near 0.35 if $p<0.25$, else near 0.15; $\mu_z$ untouched; ADI crosses 4 |
| C7 | Policy's safety factor only | `safety_scale = 0.60`; demand untouched |

For C2, the generator solves $\gamma$ by bisection on Monte-Carlo coverage (20,000 aggregate draws) at $\alpha=0.95$, $R=3$, with target coverage gap $\tau_{\mathrm{cov}} + \mathrm{margin} = 0.10 + 0.02 = 0.12$. The target is not always reachable: 193 of 200 C2 SKUs clear 0.12, four land in [0.10, 0.12), and three fall below 0.10; all seven shortfalls hit the 0.02 gamma floor. All 200 C2 SKUs have $\gamma < 1$, with range 0.020–0.560 and median 0.453. The 40 C2 test SKUs span 0.020–0.556 with median 0.459. The generator's 95% coverage-gap median is −0.120.

The monitor checks a 90% conformal interval (`c2_interval_alpha = 0.90`), scaled by the same $\gamma$. The generator's 95% solve and the monitor's 90% check are different coverage objects and are not equated in this paper.

**C2b generator defect.** The C2b injection was intended to represent a modelling-error variant of C2. In the executed generator, `inject_C2b` returns the untouched baseline demand and records the intended per-period-sigma modelling error only in metadata. The Step 5 pipeline applies the generator's calibration window to the simulator's quantile window, not to the forecaster's reported interval. The C2b panel is therefore statistically indistinguishable from the control: its coverage-gap median is +0.033 (10th–90th percentile −0.100 to +0.100), and the binomial $p<0.05$ rate is 9.5% versus 14.5% on controls. The 0.090 C2b recall is a property of the generator defect, not evidence about the detector. This is stated explicitly wherever C2b appears.

### 5.4 Pipeline and arm (c)

Each SKU is run through arm (c):

- the pooled forecaster supplies $\hat{\mu}$;
- arm (c) selects the active pool member by trailing mean absolute scaled error (MASE);
- the simulator computes
  $$
  S = R\,\hat{\mu} + Q_\alpha\!\left(\text{R-period sums of signed residuals}\right),
  $$
  with $R = L + 1 = 3$;
- the monitor is updated every period;
- attribution is priority-ordered and returns a cause plus the fired rules, evidence, margins, and confidence.

The inventory settings are: lead time $L=2$, review period 1, protection interval $R=3$, lost sales, demand observation, $\alpha=0.95$, $B/H=19$, quantile window 24, and a minimum of four aggregates. The monitor starts at the burn-in (period 24) and checks every `stride=4` periods, giving 44 checks over periods 24–196.

The monitor uses windows $W_{\mathrm{short}}=24$, $W_{\mathrm{sign}}=96$, $W_{\mathrm{cal}}=30$, $W_{\mathrm{occ}}=36$, $W_{\mathrm{sz}}=24$, $W_{\mathrm{class}}=30$, $W_{\mathrm{svc}}=30$, $W_{\mathrm{cusum}}=96$, and $W_{\mathrm{base}}=24$. Evidence floors are C1 8 nonzero observations, C3 10, C4 10, C2 30 coverage observations, C6 30 periods, C7 30 periods or 2 unmet cycles, and C5 8 baseline nonzero observations. Detection settings include $|z_b|>1.96$, binomial $p=0.05$, $\tau_{\mathrm{cov}}=0.10$, $\tau_{\mathrm{occ}}=0.10$, a 0.90 C2 interval level, $\delta_{\mathrm{svc}}=0.10$, $\mathrm{ADI}^*=4.0$, $\mathrm{CV^2}^*=0.49$, cumulative sum (CUSUM) parameters $k=0.25$ and $h=4.0$, C3 area under the curve (AUC) 0.65 and false-zero rate 0.30, C4 ratio threshold 1.5 with two windows and an occurrence-AUC guard of 0.65, C6 three windows with 1,000 bootstrap draws and stability 0.90, and a confidence gate $\tau_{\mathrm{conf}}=0.35$.

No threshold or frozen parameter was changed to make a gate pass after the run.

### 5.5 Detection reading

For each SKU, the detection reading is the first check at $t \ge \tau_{\mathrm{inj}} = 120$ whose attribution is non-C0. If no check after $\tau_{\mathrm{inj}}$ is non-C0, the final reading is used. The reported `attributed_cause` is the attribution at that reading. Consequences:

- a C0 control is a false positive if **any** post-$\tau$ check names a non-C0 cause;
- a detected cause is scored on the first non-C0 name, not on a later or final re-reading;
- undetected SKUs fall back to the final reading.

The Step 5 results-pack captions describe recall as being measured at the "final reading". That is shorthand. The implementation uses the first post-injection non-C0 reading, with the final reading as fallback. The counts reported here are the implementation's counts. This is a caption imprecision, not a numerical discrepancy.

---

## 6. Experimental design and evaluation metrics

### 6.1 Step 5: attribution gates

Step 5 runs the whole 1,800-SKU panel once. There is no train/validation/test split and no model selection on this panel.

The predeclared gates are:

- **Gate 5.1 (recall):** fraction of SKUs in a cause's panel whose attributed cause equals that cause, with a bar of 0.60 for C1–C6. The gate requires both the pooled recall and the budget-restricted recall to clear 0.60. The C2 gate denominator is `truth_for_gate == "C2"`, which maps C2b into C2 and gives 400 SKUs. C2b and C2 direct are also reported separately. C7 is reported but not gated under the accepted six-cause scope.
- **Budget-restricted recall:** the same recall computed only on SKUs where the cause's evidence floor was met at least once after $\tau_{\mathrm{inj}}$.
- **Gate 5.2 (control false positives):** fraction of C0 SKUs with any non-C0 attribution at a check at $t \ge \tau_{\mathrm{inj}}$, with a bar of at most 0.10.
- **Gate 5.3 (confusion):** row-normalised confusion over C0–C7; the gate fails if any off-diagonal entry exceeds 0.40. C2b is excluded from this matrix. The stored gate value for C2 is on the true-C2 row only; a pooled C2 row including C2b is reported descriptively.
- **Gate 5.4 (priority sensitivity):** 10 random permutations of the attribution priority order, compared with the base order by macro-F1 over C1–C7; the gate is the maximum absolute difference, with a bar of 0.10.
- **Gate 5.5:** the confusion matrix and statistic distributions were produced.
- **Gate 5.6:** budget-restricted recall is reported. It is a reporting gate, not an input to the overall pass.
- **Overall pass:** the conjunction of 5.1 (pooled and budget-restricted), 5.2, 5.3, and 5.4.

The 95% Wilson intervals shown for Step 5 are descriptive confidence intervals (CIs), computed from the integer counts (Wilson, 1927). The gate bars are applied to point estimates.

Two denominator notes matter for interpretation. First, the C2 pooled denominator of 400 includes the C2b variant. The pooled C2 recall therefore mixes a real C2 panel with a panel that is, in effect, a control. Second, the budget-restricted recall in gate 5.1 uses evidence availability: C4 has $n=39$ and C5 has $n=68$. The Step 5 findings document also reports the generator's own budget-flag denominators (C4 $n=72$, C5 $n=200$). These are not the same denominator and should not be compared directly. Both sets fail the 0.60 bar.

### 6.2 Step 6: panel, split, and reliability gate

Step 6 uses the same seed-42 panel. C2b is excluded everywhere. The split is deterministic within each true cause: sort by `sku_id`, then cycle `train, train, train, validation, test`. Each cause contributes 120 train, 40 validation, and 40 test SKUs. The test split contains 40 C2 and 40 C0 SKUs. Only these 80 test SKUs enter the inventory comparison. No training or validation SKU enters the outcome comparison.

The reliability gate is trained only on Step 5 readings satisfying `detected == True`, `attributed_cause == "C2"`, and `true_cause != "C2b"`. The label is `true_cause == "C2"`. These are the first post-$\tau$ checkpoint readings, the exact event at which the policy makes its one decision. Undetected final-reading rows are excluded.

| Split | Rows | Candidates | True-C2 candidates |
|---|---:|---:|---:|
| Train | 960 | 246 | 115 |
| Validation | 320 | 75 | 39 |
| Test | 320 | 81 | 37 |

Nine features are taken from the decision reading: `margin_C2`, `n_fired`, `ev_c2`, `window_full`, `abs_gap`, `neglog10_p`, `abs_z_b`, `r_strength`, and `confidence`. The model is a pipeline of median imputation, standardisation, and logistic regression with `max_iter=3000` and `C=1.0`.

The threshold is the lowest validation threshold with precision at least 0.90 and at least 20 admitted true-C2 validation SKUs. The selected threshold is $\tau = 0.24462755647681794$. On validation it admits 41 rows, 39 of them true C2 (precision 0.951). The test split is never used for selection. If no threshold had qualified, a rule-only variant would have acted on every detected C2 attribution; it was not needed.

### 6.3 Step 6: policy and C2 remedy

The policy checks at $t \ge$ burn-in 24, every `stride=4` periods, and makes at most one decision, at the first non-C0 attribution with $t \ge \tau_{\mathrm{inj}}=120$. The decision rule is:

```
if attributed_cause != C2:            R0, reason not_C2
elif not ev_c2:                       R0, reason insufficient_evidence
elif gate_score < tau:                R0, reason low_reliability
else:                                 R2, reset the C2 scale
```

The remedy is a single assignment:

$$
\mathrm{scale}(t+1) := 1.0,
$$

so the order-up-to level becomes

$$
S = R\,\hat{\mu} + 1.0 \cdot Q_\alpha\!\left(\text{R-period sums of signed residuals}\right).
$$

The point forecast $\hat{\mu}$ is not touched. The monitor sees the same scale the policy uses. This is not `apply_r2_conformal`; that function computes an absolute-residual quantile, which is not the policy's signed aggregate quantile. Because the safety term is a signed residual quantile, $\gamma<1$ can move $S$ either up or down. Resetting to 1.0 restores nominal calibration; it is not a one-way safety-stock increase.

### 6.4 Baseline and timing

The baseline is the same arm (c) on the same demand series, with the C2 scale fixed at the generator's $\gamma$ for the whole run, exactly as in Step 5. The gated arm starts at $\gamma$ and switches to 1.0 only if the single decision is R2. Both arms use the same pooled forecaster, simulator configuration, and seed. All comparisons are paired by SKU.

A decision is made after the level for period $t$ is computed and the monitor is updated with $y_t$. The new scale is first used by the call at $t+1$. The order placed at the end of $t+1$ arrives at the start of $t+1+L$, with $L=2$. There is no look-ahead.

The outcome window was fixed from the training split only: `EVAL_START = tau_inj + 12 = 132`, giving evaluated periods [132, 200), or 68 periods. Of 120 detected train C2 SKUs, 95.8% have `detect_t + 1 + lead_time <= 132`; the five late detectors are reported. In the comparison, all 40 C2 decisions occurred at $t=120$ or $t=124$; 37 of the 40 decisions led to action, and actions began at 121 or 125. The first affected orders arrived at 123 or 127, before 132. The 68-period window contains 22 complete 3-period cycles (66 periods). The last two periods contribute to cost and period service level but not to cycle CSL.

### 6.5 Outcomes and inference

- **Cost:** per-SKU sum of `cost_period` over [132, 200), where `cost_period = H * on_hand_end + B * unmet`, $B/H=19$, and $H$ is the per-SKU holding rate. The base config's $H=1.0$ is overridden by each SKU's recorded holding rate.
- **Cycle CSL:** fraction of complete 3-period cycles with zero lost demand, averaged over SKUs; this is the cycle service level (CSL).
- **Period service level:** fraction of evaluated periods with zero lost demand, denoted period SL. This is secondary, and no bootstrap interval is stored.
- **Paired difference:** gated minus baseline for the same SKU, then averaged. CSL and period-SL differences are reported in percentage points (pp).
- **95% interval:** percentile bootstrap over SKUs, 10,000 resamples, seed 42, percentiles 2.5 and 97.5 (Efron & Tibshirani, 1994).

The two outcomes are reported without multiplicity adjustment. This is an exploratory feasibility analysis, not a confirmatory test.

### 6.6 Software and reproducibility settings

The results pack records Python 3.13.5, NumPy 2.5.3, pandas 2.3.3, Matplotlib 3.11.2, and pyarrow for Parquet input. The Step 5 run used the frozen `base.yaml` and `synthetic.yaml`; Step 6 uses the same configuration with the C2 scale provider. Seeds and settings are listed in Section 11.

---

## 7. Results

### 7.1 Step 5: only direct C2 passes the recall bar

Table 1 and Figure 1 report the Step 5 gates. The overall verdict is **FAIL**.

**Table 1.** Step 5 attribution gates on the seed-42 synthetic panel. Panel A: per-cause recall at the detection reading, with 95% Wilson intervals. The predeclared bar is 0.60 for C1–C6. C2 pooled is the gated denominator (C2 + C2b); C2 direct is the C2 panel alone. Panel B: control false-positive rate, row-normalised maximum off-diagonal confusion, and priority-order sensitivity. C2b is a variant panel and is excluded from gate 5.3. Overall verdict: FAIL; E-8 unchanged. Source: `paper/tables/table1_step5_gates.tex`.

| Cause | N | Correct | Recall | 95% Wilson CI | Verdict |
|---|---:|---:|---:|---|---|
| C1 | 200 | 6 | 0.030 | [0.014, 0.064] | FAIL |
| C2 pooled | 400 | 209 | 0.522 | [0.474, 0.571] | FAIL |
| C2 direct | 200 | 191 | 0.955 | [0.917, 0.976] | PASS |
| C2b variant | 200 | 18 | 0.090 | [0.058, 0.138] | FAIL (variant) |
| C3 | 200 | 11 | 0.055 | [0.031, 0.096] | FAIL |
| C4 | 200 | 0 | 0.000 | [0.000, 0.019] | FAIL |
| C5 | 200 | 22 | 0.110 | [0.074, 0.161] | FAIL |
| C6 | 200 | 29 | 0.145 | [0.103, 0.200] | FAIL |
| C7 reported, not gated | 200 | 56 | 0.280 | [0.222, 0.346] | not gated |

| Gate | Detail | n | Estimate | 95% CI | Verdict |
|---|---|---:|---|---|---|
| 5.2 | C0 false-positive rate (any non-C0 call) | 200 | 0.450 | [0.383, 0.519] | FAIL |
| 5.3 C0 | max off-diagonal to C2 | 200 | 0.145 | — | pass |
| 5.3 C1 | max off-diagonal to C0 | 200 | 0.545 | — | FAIL |
| 5.3 C2 | max off-diagonal to C0; true C2 only 0.040 to C5 | 400 | 0.335 | — | pass |
| 5.3 C3 | max off-diagonal to C0 | 200 | 0.360 | — | pass |
| 5.3 C4 | max off-diagonal to C0 | 200 | 0.560 | — | FAIL |
| 5.3 C5 | max off-diagonal to C6 | 200 | 0.230 | — | pass |
| 5.3 C6 | max off-diagonal to C0 | 200 | 0.480 | — | FAIL |
| 5.3 C7 | max off-diagonal to C0 | 200 | 0.390 | — | pass |
| 5.4 | max absolute macro-F1 difference, 10 permutations | — | 0.011 | — | PASS |
| Overall | gates 5.1–5.5 | — | — | — | **FAIL** |

Note: the `.tex` source caption uses "final reading" as shorthand. The implementation and the counts in this paper use the first post-injection non-C0 reading, with the final reading as fallback (Section 5.5). E-8 is the project's internal escalation identifier for the Step 5 attribution failure; it is unchanged.

![Figure 1. Step 5 per-cause attribution recall on the seed-42 synthetic panel. Points are recall at the detection reading; whiskers are 95% Wilson intervals. The dashed line is the predeclared 0.60 bar for C1–C6. C2 direct is the only cause that clears the bar. The C0 control false-positive rate is 0.450 against a bar of at most 0.10.](figures/fig1_step5_attribution.png)

Only C2 direct passes: 191/200, recall 0.955, 95% Wilson interval [0.917, 0.976]. C2 pooled, the conservative gate denominator that includes the C2b variant, is 209/400 = 0.522 [0.474, 0.571] and fails. C2b on its own is 18/200 = 0.090 [0.058, 0.138]. As explained in Section 5.3, the C2b panel is a generator defect: the demand is untouched and the intended miscalibration is recorded only in metadata. The C2b recall is therefore not a detector failure.

The budget-restricted recalls also fail the 0.60 bar: C1 0.050 (6/120), C2 0.522 (209/400), C3 0.076 (11/144), C4 0.000 (0/39), C5 0.324 (22/68), and C6 0.158 (29/184). The gate fails on both denominators, so the verdict does not depend on the C2b mapping.

### 7.2 Step 5: false positives and confusion

The C0 false-positive rate is 90/200 = 0.450 [0.383, 0.519], against a bar of at most 0.10. The 90 false positives were attributed to C2 (29), C6 (27), C5 (15), C7 (13), C1 (5), and C3 (1). The pre-injection flag rate, a diagnostic, was 0.630 (126/200); the per-check flag rate was 1,258/8,800 = 0.143.

Every gate 5.3 failure is abstention to C0, not cross-cause confusion. C1, C4, and C6 are labelled C0 at 0.545, 0.560, and 0.480. The largest true cross-cause confusion is C5 → C6 at 0.230. The monitor is not mixing causes up; it is largely failing to name them.

The priority-sensitivity gate passes: the maximum absolute macro-F1 difference over ten permutations is 0.0111, against a bar of 0.10, and the base macro-F1 over C1–C7 is 0.1498. The ordering is not what decides the result; the detectors are.

For C2, the stored gate value is 0.040 on the true-C2 row (to C5); the pooled C2 row including C2b has a maximum off-diagonal of 0.335 (to C0). Both are below the 0.40 bar. The gate itself used the true-C2 value.

### 7.3 Step 5: why the failures occurred

The failures have different causes, and separating them is the main scientific value of Step 5.

**C1: undetectable injection as generated.** The linear ramp with $\phi \in \{\pm 0.2, \pm 0.4\}$ is too small for the bias statistic. With a frozen forecast, the expected $|z_b|$ is approximately $3.48|\phi|$, at most 1.39, below the 1.96 bar. With the live adaptive forecaster, the tracking model absorbs what remains. The measured $|z_b|>1.96$ crossing rate was 0.000 on C1 versus 0.067 on C0. No threshold change can recover a signal the statistic does not contain.

**C2: supported on the direct panel.** Direct C2 recall is 0.955, with a rule firing rate of 0.540 versus 0.145 on control (two-proportion $z=+9.28$), and a coverage-gap median of −0.567 versus 0.000 on control. The detector works on the object it was designed to detect.

**C2b: generator defect, not a detector result.** The C2b panel is statistically indistinguishable from control, as described in Section 5.3. Its 0.090 recall is not evidence that the detector cannot detect a modelling-error variant; the panel does not contain the intended error.

**C3: undetectable injection as generated.** The median injected change in $p$ is +0.031 (10th–90th percentile −0.167 to +0.188). One sampling standard deviation of an occurrence rate over 36 periods near $p=0.2$ is about 0.048, and $\tau_{\mathrm{occ}}=0.10$. The step is within one standard error for most SKUs, and the forecaster tracks the step within about five periods. The rule is highly specific—firing 0.055 on C3 versus 0.005 on C0 ($z=+2.93$)—but that is 11 SKUs, far below 0.60.

**C4: specification limitation.** The injection is strong, changing the Gamma shape $k \to k \times \{0.25, 4.0\}$ at fixed $\mu_z$. The rule's occurrence-AUC guard, however, is unsatisfiable on this panel: occurrence AUC is 0.65 or more on 1.6% of all SKUs and 0% of C4 SKUs. The rule fired on 0 of 1,800 SKUs. The generator's budget flag marks 72/200 C4 SKUs as clearing its constraint; the monitor's evidence floor is met on 27/200 at the final reading, and both size windows are in band on 5/200. All five are blocked by the AUC guard.

**C5: detector calibration limitation.** The injection is strong: the median change in $p$ is 0.204, and every C5 SKU clears the generator's detectability budget. But the occurrence CUSUM alarms on 99.0% of controls, so the alarm is saturated and carries little information. The step-versus-drift Bayesian information criterion (BIC) that must arbitrate it passes on 3.9% of SKUs. The size CUSUM is better (alarm 0.236 overall, 0.200 on C0) but is still not calibrated to the 10% false-alarm bar.

**C6: both generator and detector.** The ADI/CV² cell moves from its initial value on 47.1% of SKUs, but the bootstrap-stability guard passes on only 14.2%. The rule fires on 0.195 of controls versus 0.180 of C6 SKUs ($z=-0.38$). ADI estimated on 30 periods of an intermittent series is too noisy to carry the classification, and the generator does not require the cell to cross either.

**C7: documented negative.** Recall is 0.280, and the rule is not gated. The policy-outcome service statistic separates C7 from control on this synthetic panel (0.305 versus 0.070, $z=+6.02$), but this does not reverse the earlier forecast-diagnostic negative: under demand observation, the forecaster never sees the stock-outs that C7 causes. The earlier negative therefore stands.

Across the panel, 717 of 1,800 SKUs (about 40%) never trigger any rule at the final reading: C0 107, C1 105, C2 128 (all C2b), C3 64, C4 109, C5 35, C6 94, and C7 75. The confidence gate is not the binding constraint where a rule does fire: among SKUs with one cause firing, only 3.4% abstain; with two, 1.9%.

The Step 5 overall verdict is **FAIL**. The E-8 escalation in the project's `ESCALATIONS.md` remains the authoritative escalation, and no gate was relaxed or reopened.

### 7.4 Step 6: reliability-gate performance

Table 2 reports the gate diagnostics on the stored Step 5 readings and on the 80-SKU comparison.

**Table 2.** Step 6 reliability-gate performance. Panel A: stored Step 5 readings by split. "Recall" divides by all true C2 SKUs in the split; "Coverage (rows)" divides by all non-C2b rows. The validation split selected $\tau = 0.2446$; the test split was untouched. Panel B: live actions on the 80-SKU C2+C0 comparison. The comparison precision of 0.974 is not the all-cause test precision of 0.822. Source: `paper/tables/table2_step6_gate.tex`.

| Split | Rows | Candidates | True-C2 candidates | Acted | True-C2 acted | Precision | Recall | Coverage (rows) | Coverage (cand.) | C0 action | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 960 | 246 | 115 | 147 | 109 | 0.741 | 0.908 | 0.153 | 0.598 | 0.025 | 0.947 |
| Validation | 320 | 75 | 39 | 41 | 39 | 0.951 | 0.975 | 0.128 | 0.547 | 0.000 | 0.995 |
| Test | 320 | 81 | 37 | 45 | 37 | 0.822 | 0.925 | 0.141 | 0.556 | 0.025 | 0.979 |

| Metric | Value |
|---|---:|
| Acted | 38/80 (0.475) |
| True C2 acted | 37/40 (0.925) |
| True C0 acted | 1/40 (0.025) |
| C2-attributed rows | 44/80 (0.550) |
| Precision | 37/38 (0.974) |
| Coverage | 38/80 (0.475) |
| Decision match | 80/80 (1.000) |

The gate is a candidate filter: a candidate is a row with `detected == True` and `attributed_cause == "C2"`. On the test split, 45 of 81 candidates were acted on; 37 were true C2 and 8 were non-C2 (one true C0 plus seven true C1–C7 SKUs outside the inventory comparison). The test all-cause precision is therefore 0.822. The comparison precision on the 80 C2+C0 SKUs is 0.974. These two numbers are intentional and must not be conflated: seven false actions fall outside the inventory comparison. AUROC in Table 2 is the area under the receiver-operating characteristic curve.

### 7.5 Step 6: actions and attribution misses

On the 80 compared SKUs, 44 were attributed to C2: 37 true C2 and 7 true C0. Of these, 38 were acted on (37 true C2 and 1 true C0). Six low-reliability fallbacks were all true C0. Fourteen `not_C2` fallbacks comprised 3 true C2 SKUs attributed C5 and 11 true C0 SKUs. Twenty-two SKUs had no decision, all true C0. The live decision matched the stored-reading decision on 80/80 SKUs.

The single false action was `control:C0:0099` at $t=120$, with a gate score of 0.630. Its $\gamma$ was 1.0, so resetting the scale to 1.0 was a no-op; its cost and CSL differences are exactly zero. This is why the C0 group is exactly unchanged even though a control was acted on. A false action on a genuinely miscalibrated non-C2 cause would not be a no-op; that case was not in this experiment.

Three true C2 SKUs were attribution misses: their first non-C0 name was C5 at the first checkpoint. They were not threshold misses. The one-decision, no-requery policy cannot recover them. The gate sent six of the seven false C2 attributions on controls to `low_reliability`; the seventh was the no-op action.

### 7.6 Step 6: paired inventory outcomes

Table 3 and Figure 2 report the paired outcomes.

**Table 3.** Step 6 paired inventory outcomes (seed 42; 40 C2 + 40 C0 test SKUs). Differences are gated minus baseline over the 68-period window [132, 200); cycle CSL uses 22 complete 3-period cycles. Cost CIs and CSL CIs are percentile bootstrap intervals (10,000 resamples, seed 42). C0 is exactly zero on all 40 SKUs. Period service-level CIs were not stored and are not inferred. Source: `paper/tables/table3_step6_outcomes.tex`.

| Group | n | Baseline cost | Gated cost | Δ cost [95% CI] | Δ (%) |
|---|---:|---:|---:|---|---:|
| C2 | 40 | 2507.26 | 2578.73 | +71.48 [−20.01, +168.63] | +2.85 |
| C0 | 40 | 2621.96 | 2621.96 | +0.00 [+0.00, +0.00] | +0.00 |
| All | 80 | 2564.61 | 2600.35 | +35.74 [−9.32, +85.77] | +1.39 |

| Group | n | Baseline CSL | Gated CSL | Δ CSL (pp) [95% CI] | Baseline period SL | Gated period SL | Δ period SL (pp) |
|---|---:|---:|---:|---|---:|---:|---:|
| C2 | 40 | 0.801 | 0.863 | +6.14 [+4.20, +8.30] | 0.925 | 0.949 | +2.46 |
| C0 | 40 | 0.882 | 0.882 | +0.00 [+0.00, +0.00] | 0.957 | 0.957 | +0.00 |
| All | 80 | 0.841 | 0.872 | +3.07 [+1.93, +4.38] | 0.941 | 0.953 | +1.23 |

![Figure 2. Step 6 paired inventory outcomes for the 80 seed-42 test SKUs (40 C2 + 40 C0). Points are mean paired differences (gated minus baseline); whiskers are 95% percentile bootstrap intervals (10,000 resamples, seed 42). (a) Cost over the 68-period window [132, 200); positive means higher cost. (b) Cycle service level over 22 complete 3-period cycles; positive means higher service, in percentage points. C0 is exactly zero on both outcomes.](figures/fig2_step6_paired_outcomes.png)

In the C2 group, cost rose from 2507.26 to 2578.73, a mean paired increase of 71.48 units, or 2.85%, with a 95% interval of [−20.01, 168.63] that includes zero. Cycle CSL rose from 0.8011 to 0.8625, a gain of 6.14 percentage points, with a 95% interval of [4.20, 8.30] that is entirely above zero. The C0 group was exactly unchanged on all 40 SKUs. The all-80 group shows a cost increase of 35.74 [−9.32, 85.77] and a CSL gain of 3.07 pp [1.93, 4.38].

Period service level, a secondary outcome with no stored interval, moved from 0.9246 to 0.9493 in the C2 group (+2.46 pp), from 0.9566 to 0.9566 in C0, and from 0.9406 to 0.9529 overall (+1.23 pp). No bootstrap interval is available for period SL, so no interval is reported.

The direction of the C2 effect is mixed on cost and one-sided on service. Cost rose on 21 SKUs, fell on 16, and was unchanged on 3. Cycle CSL rose on 27 SKUs, fell on 0, and was unchanged on 13. The cost changes ranged from −886.87 to +1129.07; the CSL changes ranged from 0.0 to +0.3182. The mixed cost direction is consistent with resetting a signed quantile: on some SKUs the nominal level is higher, on others lower. CSL did not fall on any C2 SKU in this seed.

Gated C2 cycle CSL is 0.8625, and control cycle CSL is 0.8818. The 95% service target is not reached. The scale reset repairs the C2-specific miscalibration; a residual service gap is shared with the control and is not repaired by this intervention.

---

## 8. Discussion and implications

### 8.1 Attribution is the binding constraint

The Step 5 result is not a single failure but a set of distinct detectability boundaries. Only C2 direct passes. For C1 and C3, the injected signal is too small relative to the statistic's noise for the live forecaster; these are generator-design failures as much as detector failures. For C4, the generator produces a strong dispersion change, but the rule contains an occurrence-AUC guard that is effectively unsatisfiable on an i.i.d. Bernoulli occurrence process: the AUC of such a process against a smoothed rate forecast is near 0.5 by construction, and a 0.65 bar was written for a different case. For C5, the occurrence CUSUM alarms on almost all controls, so the alarm is saturated. For C6, the classification cell is too noisy at 30 periods and the detector fails on both generator and detector. C7 remains a documented negative.

This matters for CAFR as a research programme. A repair layer can only be as good as the attribution that triggers it. The present results suggest that, under a live adaptive forecaster, the set of causes that remain visible is much smaller than the set that can be injected. That is a useful and reportable finding: it says where a mechanism-attribution claim must be tested before it is made. It also explains why the paper does not claim a six-cause taxonomy.

The C2b panel is a separate lesson. A generator defect made a cause panel statistically indistinguishable from control. If the defect had not been found, the low C2b recall would have been misread as a detector failure. The paper reports the defect explicitly so that the C2b number is not used as evidence either way.

### 8.2 The C2 repair trades service for cost in this seed

The Step 6 result is mixed but interpretable. The gate selects most true C2 SKUs (37/40) and almost no controls (1/40, and that action is a no-op). The C2 scale reset is associated with a clear cycle-CSL gain of 6.14 percentage points, with an interval entirely above zero, and with a cost increase of 2.85% whose interval includes zero. The C0 group is bit-identical, which is the expected result when no effective action is taken on controls.

This is not a cost-saving result. It is also not evidence that the remedy is harmful: the cost interval includes zero, and cost fell on 16 of 40 C2 SKUs. The most defensible reading is that the one-time scale reset is consistent with restoring nominal calibration, which is associated with higher service in this setting, and that the cost consequence is uncertain at this sample size. The 95% service target is still not met, so the remedy is partial even on its own terms.

The direction counts are consistent with the mechanism. Because the safety term is a signed residual quantile, a scale below one can move the order-up-to level either up or down depending on the sign of the aggregate residual quantile. Resetting to one is not a uniform safety-stock increase. That explains why cost rises on some SKUs and falls on others, while cycle CSL never falls in this seed.

### 8.3 Relation to prior work

The result is consistent with the accuracy–service literature. Chin et al. (2026) and Fukuhara et al. (2026) show that accuracy rankings do not automatically translate into service or cost performance; the present experiment adds a case in which a calibration-oriented action was associated with improved service without a demonstrated cost benefit. Mikkonen and Saarinen (2025) frame the problem as cost-aligned learning; the present result suggests that the cost and service objectives may need to be reported separately rather than collapsed into a single reward.

The experiment is also narrower than the diagnose-then-correct precedents. Manary et al. (2009) diagnose and correct forecast error in an industrial planning process, but with human-authored corrections and without an intermittent-demand mechanism taxonomy. The patent US20100125489A1 describes a diagnose-and-adjust loop for inventory problems, but with planner confirmation and without a mechanism-level, intermittent-demand-specific design. Lei et al. (2026) learn a correction with a bandit, but the action is a single continuous adjustment and the reward is accuracy. The present study differs on all of those dimensions, but it does not establish that the difference produces a better operational outcome. It establishes only that a C2-specific calibration error can be detected and that a scale reset is associated with higher service in this synthetic panel.

### 8.4 Implications for research and practice

For research, the main implication is methodological. Attribution studies need explicit detectability budgets for each injected cause, and they need to report generator defects and unsatisfiable rule guards as part of the result. A gate that fails because the injection is too small is not the same as a gate that fails because the detector is bad. The Step 5 analysis separates these cases.

For practice, the result is a caution and a narrow signal. A C2-type calibration error—where the demand process is fine but the reported uncertainty is wrong—may be repairable with a simple scale reset, and the service effect can be material. But the cost effect is uncertain, the service target is not reached, and the result comes from one seed and 40 C2 test SKUs. Deployment would require multi-seed replication, a leak-free forecaster, a larger test set, and a real-data transfer study.

For the CAFR programme, the original claim that full CAFR beats accuracy-based selection at equal churn is retired. The present experiments do not test that claim. They test attribution detectability and one C2 remedy inside arm (c). A future test of the full claim would need the bandit, the churn constraint, the multi-cause remedy set, and a leak-free evaluation.

---

## 9. Limitations and threats to validity

1. **Synthetic only.** Every result is on a generated intermittent-demand panel. No external benchmark panel (such as RUF or M5) or real dataset was used. There is no real-world validation and no demonstrated transfer.
2. **Single seed.** The panel and both experiments use seed 42. The generator's held-out seed 43 was not used. No multi-seed replication or seed-level uncertainty is reported.
3. **Small test set.** Step 6 uses 40 C2 and 40 C0 test SKUs, with 22 complete cycles per SKU. The bootstrap resamples 40 C2 SKUs, so its intervals are coarse. This is a feasibility result, not a confirmation.
4. **No demonstrated cost reduction.** The C2 point estimate is a 2.85% cost increase, and the 95% interval includes zero. The all-80 estimate is +1.39%, also with an interval that includes zero. No cost saving, cost neutrality, or cost-effectiveness claim is supported.
5. **Service is below the 95% target.** Gated C2 cycle CSL is 0.8625; C0 is 0.8818. The scale reset repairs the C2-specific miscalibration only; a residual service gap is shared with the control and is not repaired.
6. **Pooled-forecaster exposure and possible leakage.** The global pooled forecaster is fitted on the first 24 periods of all 1,800 SKUs, including the 40 C2 and 40 C0 test SKUs. The gate model is trained only on train-split rows, but the forecaster's burn-in exposure to test SKUs is mild leakage. Its effect on gate robustness is not measured. A leak-free replication would fit the pool on train SKUs only or freeze it at burn-in.
7. **Step 5 attribution is largely unvalidated.** Only C2 direct passes the recall bar. C1 and C3 are undetectable as generated; C4's rule is blocked by an unsatisfiable guard; C5's CUSUM is saturated; C6 fails on both generator and detector; C7 is a documented negative. The C2b panel is a generator defect, so its 0.090 recall is not a detector result. No six-cause attribution claim is possible.
8. **C2-only, one-time remedy.** Step 6 repairs one C2 scale, once per SKU. It does not adapt over time, does not address C2b or any other cause, and does not test a multi-cause policy. The result is conditional on C2 attribution.
9. **No full-CAFR or churn comparison.** Both arms use arm (c). The bandit, churn control, and the original full-CAFR-versus-accuracy-selection hypothesis were not run. Step 6 isolates a C2 scale reset, not CAFR's full decision layer.
10. **Gate limitations.** The threshold was selected on 75 validation candidates; the test split has 81 candidates. All-cause test precision is 0.822, while the C2+C0 comparison precision is 0.974; the difference is seven false actions on C1–C7 SKUs outside the comparison. The gate missed three true C2 SKUs through attribution, and six of seven false C2 attributions on controls were filtered as low reliability. The 0.974 figure must not be quoted as full-panel precision.
11. **Inference limitations.** Two outcomes are reported without multiplicity adjustment. Period SL has no stored interval. Cycle CSL is based on 22 cycles and is noisy. The bootstrap is over SKUs, not over seeds or generator cells.
12. **Single cost model.** Cost is holding plus lost-sales penalty, with no fixed ordering cost and one synthetic data-generating process. $H$ varies by SKU and $B/H=19$; the results are conditional on that simulator.
13. **C2 injection weakness.** Seven of 200 C2 SKUs cannot reach the 0.12 margin at the 0.02 gamma floor; four are in [0.10, 0.12) and three are below 0.10. Two of the seven are in the 40-SKU test group, both in [0.10, 0.12). Those injections are weaker than the constraint intends.
14. **Different coverage objects.** The generator solves $\gamma$ at 95% coverage; the monitor checks a 90% conformal interval. They are related by the same $\gamma$ but are not the same statistic, and this paper does not equate them.
15. **Historical tests.** The 186 passing tests recorded in the Step 6 findings were not rerun for the results pack or this manuscript. They are historical evidence, not a fresh test result.

Additional reporting notes, recorded rather than silently corrected:

- The `step5_summary.json` confusion keys are stored in a column-oriented/mislabeled form. The results pack uses `step5_confusion.parquet` (rows = truth, columns = attributed) and recomputes row-normalised values from `step5_readings.parquet`; the recomputed values match the Step 5 findings document.
- `step5_confusion.parquet` is 8×8 and excludes C2b. C2b counts and the C2 pooled denominator come from `step5_readings.parquet`.
- Step 5 recall intervals are not stored in `step5_summary.json`; they are computed from the integer counts and match the rounded values in the Step 5 findings document.
- The `c2_compare_summary.json` field `elapsed_s = 10833.3` is a machine-suspend artifact, not compute time. It is not used anywhere in this paper.
- The Step 6 `live_actions_by_cause` key counts attributed causes (C2 44, C6 7, C5 5, C1 1, C7 1), not R2 actions. The 80-SKU comparison has 38 R2 actions. Action counts in this paper come from `c2_compare_arms.csv`.

---

## 10. Conclusion and future work

This paper reported a synthetic-panel study of cause-attributed forecast repair for intermittent demand. The study had two parts. Step 5 measured whether a live monitor and priority-ordered attribution rule could name seven injected causes at predeclared bars. The answer was largely no: only direct C2 passed the recall bar (0.955), the control false-positive rate was 0.450, and the other causes failed for a mixture of generator, specification, and detector reasons. Step 6 asked whether a reliability gate trained on those readings could decide when to apply a one-time C2 scale reset. On 40 C2 and 40 C0 test SKUs, the gate acted on 37 true C2 SKUs and one control, and the reset was associated with a 6.14-percentage-point cycle-CSL gain and a 2.85% cost increase whose interval includes zero. The C0 group was exactly unchanged. Gated service remained below 95%.

The experiments do not establish that full CAFR beats accuracy-based selection at equal churn. That claim was not tested, and it is retired. They do establish a narrower set of findings: attribution detectability is a real and measurable constraint; a C2-type calibration error can be detected and repaired in this synthetic setting; and the service and cost consequences of that repair need to be reported separately.

Future work should proceed in the order the evidence demands. First, the attribution study should be repeated with injections that have explicit detectability budgets, a fixed C4 guard, a calibrated C5 CUSUM, and a corrected C2b generator. Second, the C2 repair should be replicated across multiple seeds, with a leak-free pooled forecaster fitted only on training SKUs and a larger test set. Third, the remedy set should be extended beyond C2, but only for causes that pass their attribution gates. Fourth, the full CAFR comparison—including the bandit, the churn constraint, and non-forecast actions—should be run only after the attribution layer is validated. Fifth, the work should be transferred to a real panel with a documented licence and a pre-registered analysis plan.

---

## 11. Reproducibility and data/code availability

### 11.1 Artifacts

The analysis artifacts are held in the project repository. The principal files are:

| Artifact | Role |
|---|---|
| `data/synth/demand_panel.parquet` | Demand panel used by both experiments |
| `data/synth/generator_params.parquet` | Per-SKU generator parameters, including `gamma`, cause, statics, and constraint flags |
| `data/synth/ground_truth.parquet` | Per-period cause labels; evaluation only |
| `results/step5/step5_readings.parquet` | 1,800 rows × 66 columns; detection readings, attributed cause, features |
| `results/step5/step5_summary.json` | Step 5 gate values, confusion, latency, check schedule |
| `results/step5/step5_confusion.parquet` | 8×8 C0–C7 confusion matrix (excludes C2b) |
| `results/step5/step5_priority_sensitivity.parquet` | 10 priority permutations |
| `results/step5/run_full.log` | Step 5 run log: 1,800 SKUs, 0 errors, 869.5 s |
| `results/step6/c2_gate_summary.json` | $\tau$, split diagnostics, evaluation window |
| `results/step6/c2_compare_summary.json` | Arm means, paired means, and bootstrap CIs |
| `results/step6/c2_compare_arms.csv` | 160 rows: 80 SKUs × 2 arms, actions/reasons/outcomes |
| `results/step6/c2_compare_paired.csv` | 80 rows: per-SKU differences and actions |
| `results/step6/c2_compare_run.log` | Comparison log; contains the suspend-time artifact |
| `results/step6/c2_smoke_*`, `c2_timing_*` | Smoke and timing checks |
| `docs/18_step5_findings.md` | Step 5 narrative |
| `docs/19_step6_c2_gated_findings.md` | Step 6 narrative |
| `ESCALATIONS.md` E-8 | Step 5 FAIL escalation; unchanged |
| `paper/RESULTS_PACK.md` | Frozen results pack and provenance |
| `paper/figures/`, `paper/tables/` | Publication figures and tables |
| `paper/make_results_pack.py` | Results-pack build; reads artifacts only |

### 11.2 Commands

The historical commands, recorded for provenance, are:

```
python experiments/build_synth_panel.py --seed 42
python experiments/step5_monitor_gates.py --per-cause 200 --seed 42
python experiments/step6_c2_gated.py --stage gate
python experiments/step6_c2_gated.py --stage smoke
python experiments/step6_c2_gated.py --stage timing
python experiments/step6_c2_gated.py --stage compare
python -m pytest tests/test_c2_gated.py -q
python -m pytest -q
python paper/make_results_pack.py
```

These commands were not rerun for this manuscript. The results pack reads existing artifacts only; it does not rerun experiments, regenerate data, or run tests. The 186-test result is the historical record in the Step 6 findings document.

### 11.3 Seeds and settings

| Item | Value |
|---|---|
| Panel generator seed | 42 |
| Held-out generator seed | 43 (not used) |
| Step 5 CLI seed | 42 |
| Step 6 seed | 42 |
| Priority-permutation seed | 20261017 |
| Bootstrap seed / resamples | 42 / 10,000 |
| Step 5 periods | 200; $\tau_{\mathrm{inj}}=120$; burn-in 24 |
| Step 6 evaluation | [132, 200); 68 periods; 22 complete 3-period cycles |
| Inventory | $L=2$, $R=3$, $B/H=19$, $\alpha=0.95$, lost sales, demand observation |
| Gate threshold $\tau$ | 0.24462755647681794 |
| Step 5 C2 monitor level | 0.90 conformal interval |
| Step 5 C2 generator target | 0.12 gap at 95% coverage, $R=3$ |

### 11.4 Environment and compute time

The results pack records Python 3.13.5, NumPy 2.5.3, pandas 2.3.3, Matplotlib 3.11.2, and pyarrow for Parquet input. The Step 5 run took 869.5 s for 1,800 SKUs with 0 errors. The Step 6 gate run and the clean 20-SKU timing run are recorded in the Step 6 findings document (13.53 s for 20 SKUs; pool fit 1.4 s). The `c2_compare_summary.json` field `elapsed_s = 10833.3` is a machine-suspend artifact and must not be cited as compute time.

### 11.5 Data and code availability

The synthetic panel, derived results, and analysis scripts are held in the project repository at [REPOSITORY URL TO BE INSERTED]. No public deposit has been made at the time of this draft. The corresponding author can provide the artifacts on request, subject to [INSTITUTIONAL POLICY TO BE CONFIRMED]. The generator and pipeline are software-only and require no paid data or GPU.

---

## 12. References

Alftan, A., Kaipia, R., Loikkanen, L., & Spens, K. (2015). Centralised grocery supply chain planning: Improved exception management. *International Journal of Physical Distribution & Logistics Management*, 45(3), 237–259. https://doi.org/10.1108/IJPDLM-02-2014-0017

Amarta, I. M. D., & Zarlis, M. (2026). Stock-out distortion and under-forecasting: An inventory-aware diagnosis of demand-signal bias in multi-SKU sales forecasting. In *2026 International Symposium on Networks, Computers and Communications (ISNCC)* (pp. 1–6). IEEE. https://doi.org/10.1109/ISNCC70543.2026.11693859

Chin, J. E., Cheng, S.-F., & Gunawan, A. (2026). *Accuracy is not service: A decision-aware benchmark for intermittent-demand forecasting* (arXiv:2609.13840). arXiv. https://arxiv.org/abs/2609.13840 [AUTHOR VERIFICATION REQUIRED: acceptance at the ICDM-26 Applied Track is reported in the arXiv record; the proceedings version was not independently checked for this draft.]

Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Journal of the Operational Research Society*, 23(3), 289–303. https://doi.org/10.1057/jors.1972.50

Dong, Y., Huang, X., Sinha, K. K., & Xu, K. (2014). Collaborative demand forecasting: Toward the design of an exception-based forecasting mechanism. *Journal of Management Information Systems*, 31(2), 245–284. https://doi.org/10.2753/MIS0742-1222310209

Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. Chapman and Hall/CRC. https://doi.org/10.1201/9780429246593

Fernández-Palacios, L., Ceballos, M., & Muñoz-Ocaña, Y. (2026). *Beyond recency and magnitude: Learning-aware uncertainty estimation for safety stock under evolving forecast-error distributions* (arXiv:2607.19835). arXiv. https://arxiv.org/abs/2607.19835

Foster, D., Jia, R., & Madeka, D. (2023). *Contextual bandits for evaluating and improving inventory control policies* (arXiv:2310.16096). arXiv. https://arxiv.org/abs/2310.16096

Fukuhara, S., Alabdallah, A., Gunasekara, N., & Nowaczyk, S. (2026). Bridging forecast accuracy and inventory KPIs: A simulation-based software framework. In *Advances in Intelligent Data Analysis XXIV* (Lecture Notes in Computer Science, pp. 438–452). Springer. https://doi.org/10.1007/978-3-032-23833-7_32

González, A., & Parada, V. (2026). *Adaptive automatic model selection for demand forecasting under heterogeneous demand patterns* (arXiv:2602.13939). arXiv. https://arxiv.org/abs/2602.13939

Hendry, D. F. (2000). *A general forecast-error taxonomy*. Econometric Society World Congress 2000 Contributed Papers, No. 0608. [AUTHOR VERIFICATION REQUIRED: conference-paper citation; no DOI was located for this item.]

Hendry, D. F., & Mizon, G. E. (2013). Open-model forecast-error taxonomies. In *Recent advances and future directions in causality, prediction, and specification analysis* (pp. 219–240). Springer. https://doi.org/10.1007/978-1-4614-1653-1_9

Hong, K., Ren, Y., Li, F., Mao, W., & Gao, X. (2023). Robust interval prediction of intermittent demand for spare parts based on tensor optimization. *Sensors*, 23, 7182. https://doi.org/10.3390/s23167182

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001

Lei, Z., Ma, B. J., & Jackson, I. (2026). *A predict-then-correct loop based on few-shot continuous contextual bandit for demand forecasting* (arXiv:2607.16354). arXiv. https://arxiv.org/abs/2607.16354

Lyubchyk, L., & Grinberg, G. (2024). *Intermittent demand forecasting with adaptive bias correction based on the modified Corston method*. SSRN. https://doi.org/10.2139/ssrn.4608877 [AUTHOR VERIFICATION REQUIRED: the SSRN metadata spells the method "Corston"; the intended term is presumably "Croston". The title is reproduced as deposited.]

Manary, M. P., Willems, S. P., & Shihata, A. F. (2009). Correcting heterogeneous and biased forecast error at Intel for supply chain optimization. *Interfaces*, 39(5), 415–427. https://doi.org/10.1287/inte.1090.0452

Mikkonen, A., & Saarinen, L. (2025). *Cost-aligned deep learning for intermittent spare-parts demand and inventory*. SSRN. https://doi.org/10.2139/ssrn.5554492

Nathan, B. S., Reddy B, V. S., Sastry, C. C., Salunkhe, S., & Cep, R. (2026). Event-preserving feature engineering for intermittent demand forecasting using SHOS. *Scientific Reports*, 16, 27022. https://doi.org/10.1038/s41598-026-57129-6

Nunkaew, W., & Vitiyutt, N. (2025). An AI-business analysis (AI-BA) forecasting approach using a regime-specific adaptive ensemble for spare-parts inventory management. In *Proceedings of the 2025 8th Artificial Intelligence and Cloud Computing Conference* (pp. 510–518). ACM. https://doi.org/10.1145/3789982.3790048

Prakash, J. (2026). *Decision-aware stochastic consumption forecasting under covariate scarcity* [Computational notebook]. Zenodo. https://doi.org/10.5281/zenodo.18364003

Sbrana, G., & Babai, M. Z. (2026). Non-negative autoregressive moving average models for intermittent demand: Forecast accuracy and inventory implications. *European Journal of Operational Research*. https://doi.org/10.1016/j.ejor.2026.06.009 [AUTHOR VERIFICATION REQUIRED: volume and page numbers were not available in the Crossref record at the time of audit.]

Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). *Inventory and production management in supply chains* (4th ed.). CRC Press. https://doi.org/10.1201/9781315374406

Surendra, S., Ghosh, M., Najmi, A., Banerjee, A., & Singh, K. P. (2010). *System and method for root cause analysis and early warning of inventory problems* (US Patent Application US20100125489A1). https://patents.google.com/patent/US20100125489A1 [AUTHOR VERIFICATION REQUIRED: application publication; grant status and current legal status were not checked for this draft.]

Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303–314. https://doi.org/10.1016/j.ijforecast.2004.10.001

Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495–503. https://doi.org/10.1057/palgrave.jors.2601841

Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 214(3), 606–615. https://doi.org/10.1016/j.ejor.2011.05.018

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer. https://doi.org/10.1007/b106715

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

Xiong, P., Zhang, L., Wu, D., Zhang, Z., Ruan, M., & Zhang, Y. (2026). An LLM-empowered collaborative multi-agent framework for intermittent demand forecasting of large-scale spare parts in smart manufacturing. *Knowledge-Based Systems*, 347, 116334. https://doi.org/10.1016/j.knosys.2026.116334

Zipkin, P. H. (2000). *Foundations of inventory management*. McGraw-Hill.

---

### Reference audit note

The references above were checked against publisher pages, Crossref DOI records, arXiv records, Google Patents, or Zenodo records where those were available. The following items carry explicit `AUTHOR VERIFICATION REQUIRED` notes in their entries: Chin et al. (proceedings acceptance), Hendry (2000) (conference citation, no DOI), Lyubchyk and Grinberg (SSRN title spelling), Sbrana and Babai (missing volume/pages in Crossref), and US20100125489A1 (application status). The 2000 general forecast-error taxonomy is authored by Hendry alone; it is sometimes mis-cited as "Clements and Hendry" and is cited correctly here. The simulation-framework paper is led by Fukuhara, not by a later-listed author.

Several sources located during the project's novelty search were not used as load-bearing citations in this draft because their bibliographic details could not be independently verified to the same standard: an after-sales diagnostic-loop preprint (arXiv:2510.01006), an EMC drift paper without a complete citation, a CoSFan ICLR 2025 item without full author/venue details, Killi (2026) in a low-quality venue, and several DOI-only items from the search log. They are therefore omitted rather than cited as confirmed. This omission does not affect the numerical results or the claims made in this paper.
