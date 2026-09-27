# CAFR — Cause-Attributed Forecast Repair

B.Tech minor project. Intermittent / spare-parts demand forecasting, and the question
of whether *repairing the cause of a forecast failure* beats *picking a better model*.

**The claim under test**

> At equal model churn, cause-attributed repair beats accuracy-based model selection on
> the inventory cost–service frontier.

Nothing in this repository is novel until that sentence is supported by a run of
`cafr.cli` with a config and a seed. Everything else is machinery built to test it.

---

## 1. Setup (5 minutes, ₹0, CPU only)

```bash
git clone <this repo> && cd CAFR
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

No GPU, no paid API, no cloud account. Every dependency is free and open-source.

**Check it works:**

```bash
python -m cafr.cli doctor         # environment + data present
python -m cafr.cli panels          # the T1 dataset numbers
python -m cafr.cli verify          # run the 100+ test suite (the gates)
```

`doctor` should print `PASS`. `verify` should print all tests passing — see
`STATUS.md` for the exact counts and for the two gates that are known to need a
team decision.

---

## 2. Where the project's thinking lives

**Read these in this order.** The code is downstream of them; reading the code first
is what makes it look arbitrary.

| # | document | what it is |
|---|---|---|
| 1 | `docs/00_START_HERE.md` | 5-page orientation. **Start here.** |
| 2 | `docs/01_COMPLETE_HANDOFF.md` | the full handoff, 17 sections |
| 3 | `docs/02_TECHNICAL_SPEC_REV2.md` | the specification, and the §-references the code cites |
| 4 | `docs/03_PAPER_WRITING_PACK.md` | paper outline, 38-entry IEEE citation list, draft prose |
| 5 | `data/DATA.md` | dataset provenance, licence, SHA-256 |

Every code comment that says `(Rev 2 §7.1)` or `(handoff §5.6)` points at one of these.

---

## 3. What is built, and what is not

Steps 0, 1 and 2 of the 15-step plan are implemented **and verified by tests**:

| step | module | status |
|---|---|---|
| 0 | `cafr/data/schema.py` | no-look-ahead whitelist, enforced by set equality |
| 1 | `cafr/sim/inventory.py` | inventory simulator, periodic review `(S,S)` |
| 2 | `cafr/forecasters/` | the seven-member forecaster pool |

Steps 3–14 are **not** implemented. Their modules exist as documented interfaces —
each `cafr/<package>/__init__.py` states the interface, the config keys, and the gate
that decides when it is done. `cafr.cli run --steps 3` fails on purpose and names the
handoff section.

This is deliberate. A half-built pipeline that produces plausible numbers is worse
than an empty one.

---

## 4. Running it

```bash
python -m cafr.cli run --config base.yaml --seed 42 --steps 0,1,2
python -m cafr.cli gates            # gate table + last measured diagnostics
python -m cafr.cli verify -k 1_2    # just gate 1.2
```

Every run writes `results/run_seed<N>/manifest.json` with the config hash. That hash
is T8. **A number with no manifest is a number with no provenance** — and no
notebook-sourced number goes in the paper.

---

## 5. Repository layout

```
CAFR/
├── cafr/
│   ├── cli.py             THE entry point. Every reported number comes from here.
│   ├── configs/           base.yaml (every parameter) + features.yaml (the whitelist)
│   ├── data/
│   │   ├── schema.py      §27 contracts + build_policy_input() -- the leak barrier
│   │   ├── loaders/ruf.py observed vs. ground-truth loaders, separated by signature
│   │   └── synth/impute.py  fit_on="uncensored" imputation (observables only)
│   ├── sim/               inventory.py (DONE) · oracle_dgp.py (DONE) · loop.py (Step 3)
│   ├── forecasters/       the pool (DONE) -- Croston, SBA, TSB, mSBA, mTSB, SES, LGBM
│   ├── monitor/           M1 detectors                      (Step 5)
│   ├── attribution/       M2 cause classifier               (Step 6)
│   ├── remedies/          M3 remedies R0-R7                 (Step 7)
│   ├── policy/            M4 Thompson-sampling bandit       (Step 8)
│   ├── arms/              the ten arms (a)..(h), A4         (Steps 8-10)
│   ├── eval/              metrics, statistics, P1-P11, T1-T8 (Steps 11-13)
│   └── utils/             config, seeding, io
├── data/                  RUF v2 panel (committed, 3 MB) + the rebuild script
├── docs/                  the five documents above
├── tests/                 the gate suite -- one file per gate family
└── results/               generated. Never edit by hand.
```

---

## 6. The four rules that must not be broken

1. **No gate is ever relaxed.** If a gate fails, the *specification* is wrong.
   Escalate it as a team decision; do not widen a tolerance. (Handoff §14 rule 4.
   Gate 4.9 is never relaxed, full stop.)
2. **No look-ahead.** The policy sees the output of `build_policy_input()` and nothing
   else. The whitelist is enforced by *set equality*, not by a name blacklist — a
   blacklist fails open, and Revision 1's missed the two columns that give the answer
   away.
3. **No number from a notebook.** Config + seed + `cafr.cli`, or it does not go in
   the paper.
4. **Do not handicap arm (c)**, and do not violate A4's four hard constraints. Arm (c)
   is the comparator and A4 carries the co-primary hypothesis; weakening either makes
   the result vacuous.

---

## 7. Before the test run

The success criteria (H1, H1′, H1″, H2) are **pre-registered**. Read them in
`docs/01_COMPLETE_HANDOFF.md` §10.6 before touching the test split. The one sentence
that matters most:

> **If H1′ fails while H1 holds, the paper's contribution is the reward signal and the
> action space, not the attribution — and the paper says so in exactly those words.**

A clearly-stated negative result on a well-posed question is a legitimate paper. A
positive result produced by a relaxed gate is not.
