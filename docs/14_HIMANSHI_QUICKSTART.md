# 14 — QUICKSTART

**For:** Himanshi
**Read this page. Then open `14_HIMANSHI_COMPLETE_HANDOFF.md`.**

---

## The project in three lines

We built a method called **CAFR**. It asks *why* an intermittent-demand forecast failed, picks
a **matched fix** — sometimes a fix that does not touch the forecast at all — and learns which
fix to use from **inventory cost and service**, not from accuracy.

The claim we are testing: **at the same amount of model churn, cause-attributed repair beats
accuracy-based model selection on the inventory cost–service frontier.**

If it does not, we say so. That is a result too.

---

## Your contract

**`11_technical_specification_rev2.md` is the source of truth.** Not this page.
Where Rev 2 and any other file disagree, Rev 2 wins — and tell us the disagreement exists.

---

## Read these, in this order — before writing any code

| # | File | Why |
|---|---|---|
| 1 | `12_methodology_audit.md` | **Why each rule exists.** Read before you are tempted to simplify one. |
| 2 | `11_technical_specification_rev2.md` §5–§7 | The taxonomy, the detection maths, the remedies. The technical core. |
| 3 | `11_technical_specification_rev2.md` §14 | Splits and the no-look-ahead rules. **Read twice.** |
| 4 | `11_technical_specification_rev2.md` §15 | The injection table. **This table IS the ground truth.** |
| 5 | `11_technical_specification_rev2.md` §29 | The pre-registered success criteria. Read **before** the test run, not after. |
| 6 | `11_technical_specification_rev2.md` §30.1 | The Step 1–5 gates, with numbers. |
| 7 | `14_HIMANSHI_COMPLETE_HANDOFF.md` §3–§5 | The dataset, and what is already built for you. |

---

## The dataset is DONE. Do not re-download anything.

`data/DATA.md` has the provenance, licence and SHA-256.
`data/README.md` explains every file.

The project-ready panels already exist:

```
data/ruf/demand_panel.parquet       300,000 rows  — the input contract
data/ruf/sku_meta.parquet            5,000 SKUs   — static features, burn-in only
data/ruf/sku_attributes.parquet      5,000 SKUs   — ⛔ EVALUATION ONLY, never a policy input
data/ruf/panel_profile.json                        — the T1 numbers
```

**Verify them once, then move on:**

```bash
cd data
python -c "import hashlib;print(hashlib.sha256(open('raw/RUF_synthetic_benchmark_5000_v2.zip','rb').read()).hexdigest())"
# expect d6f441077d922ee986a2b41a9a6ff500b0f9a0154f529ff5786bc334fb21dc61
python build_ruf_panel.py --burn-in 24
```

- **Licence: CC-BY-4.0.** Redistribution of derived files is allowed **with attribution** —
  the attribution block is in `data/DATA.md` §4. It travels with the files.
- **RUF has 60 periods, not 84.** Rev 2 assumed 84. **Escalate this (decision H-1) — do not
  decide it yourself.** Handoff §3.4 has the options and the recommendation.
- ⛔ **`sku_attributes.parquet` contains full-series ADI/CV². Never load it in the policy or
  the monitor.** It is a direct ground-truth leak if it reaches the feature vector.

---

## Do these five things first

**1. Read the files above. (Day 1.)**

**2. Escalate H-1 … H-5 to the team. (Day 1.)**
They come from the dataset verification and are **not** in Rev 2. The table is in
`14_HIMANSHI_COMPLETE_HANDOFF.md` §17.3. **Do not resolve them alone.** H-1 in particular
changes the evaluated window that goes into T1.

**3. Step 0 — set up. (Day 1–2.)**
Repo skeleton (handoff §12), `configs/base.yaml` with every parameter in handoff §5.7,
`configs/features.yaml` with the `policy_features` whitelist, pinned `requirements.txt`.
**Write `tests/test_no_lookahead.py` FIRST** — the **exact column whitelist with set equality**,
not a name blacklist — and confirm it **fails** when you add an undeclared column.

**4. Step 1 — build the inventory simulator. (Day 2–4.)**
`sim/inventory.py`, exactly as handoff §7.1.
**The order-up-to level is `S = Q̂_α`, the R-period aggregate quantile — NOT `R·μ̂ + z·σ̂`.**
The wrong form understates safety stock by **42 %** and produces plausible-looking results at
the wrong service level. **This is the silent bug the whole project can die from.**
Then run the **three-part sanity test** (gates 1.1–1.7). **Do not proceed until all seven pass.**

**5. Step 2 — build the forecaster pool. (Day 4–6.)**
Croston, SBA, TSB, mSBA, mTSB, SES-on-sizes, and one **pooled** LightGBM.
**Validate every one against `statsforecast` — relative error ≤ 1e-6** (gate 2.1).
A silently wrong Croston invalidates everything downstream and is very easy to ship unnoticed.

Then Steps 3 → 8. **Step 8 is the safe stopping point.**

---

## The five rules that matter most

1. **Never let the policy see the future, or the ground truth.**
   Use the **exact column whitelist with set equality**. Rev 1's name blacklist missed
   `cause_active` and `injection_param` — the two columns that give the answer away.

2. **Never tune anything against the test split.**
   Tune on validation. Freeze every threshold before the single test run.

3. **Never change a gate or a threshold to make an experiment pass.**
   **If a gate fails, the specification is wrong, not the gate.** Escalate it as a team
   decision. Gate 4.9 in particular is **never** relaxed.

4. **Never handicap arm (c).** It is the paper's comparator. Same for **A4** — violating any of
   its four hard constraints invalidates the paper's central claim (H1′).

5. **Never report a number that came from a notebook.**
   Everything reported comes from `cafr/cli.py` with a config and a seed.

---

## If you run out of time

**Stop at Step 8.** Arm (g) — the fixed, hand-written cause→remedy table running against the
baselines — is **a complete, honest, publishable paper.** Rev 2 §30 says so in exactly those
words. **Stop there and say so clearly.**

Below that, in order of degradation:
- **Step 5** — attribution validated on the injected panel, baselines running. A shorter
  methods-and-diagnosis paper.
- **Step 3** — baselines only on real panels. A benchmark note, not this paper.

**Never cut:** the gates, the negative control (arm (c) vs itself, different seeds — it must
show a null), the no-look-ahead test, the determinism test, **A4**, **A5**, or the
pre-registration file.

---

## The one sentence to remember

> **At equal model churn, cause-attributed repair beats accuracy-based model selection on the
> inventory cost–service frontier.**

And the pre-commitment, from Rev 2 §29.4:

> **If H1′ fails while H1 holds, the paper's contribution is the reward signal and the action
> space, not the attribution — and the paper says so in exactly those words.**

---

## Where everything is

| File | What |
|---|---|
| `11_technical_specification_rev2.md` | **the contract** |
| `12_methodology_audit.md` | why each rule exists |
| `13_methodology_revision_proposal.md` | the corrections, with derivations |
| `14_HIMANSHI_COMPLETE_HANDOFF.md` | **the working document — 17 sections** |
| `14_HIMANSHI_QUICKSTART.md` | this page |
| `data/DATA.md` | dataset provenance, licence, SHA-256 |
| `data/README.md` | every data file explained |
| `08`/`09`/`10` | novelty search, evidence log, paper plan |

**Start with `14_HIMANSHI_COMPLETE_HANDOFF.md` §13 — it is the numbered step-by-step.**
