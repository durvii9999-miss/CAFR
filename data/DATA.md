# DATA.md — dataset provenance, licences and integrity

**Project:** CAFR — Cause-Attributed Forecast Repair (B.Tech minor project)
**Created:** 2026-09-23
**Scope:** every dataset this project uses, with its exact source, licence, SHA-256, and
what was actually verified by execution (as opposed to read from a README).

> **Rule this file exists to satisfy.** Rev2 §28.8 requires *"the exact download URL and
> SHA-256 of every dataset, recorded in `DATA.md`. Verify the licence text is present
> before use."* Nothing below is copied from a claim that was not checked against the
> actual artefact.

---

## 1. Primary panel — RUF v2 (the RUF release associated with arXiv:2609.13840)

### 1.1 What it is

The **RUF** benchmark release that accompanies *"Accuracy Is Not Service: A
Decision-Aware Benchmark for Intermittent-Demand Forecasting"* (Chin, Cheng & Gunawan;
ICDM 2026, Applied Track; arXiv **2609.13840**, 12 Sep 2026).

The paper's own industrial panel is **confidential**. RUF is its **fidelity-certified
synthetic surrogate**: 5,000 materials × 60 monthly periods, constructed to reproduce the
real panel's marginal and dependence structure. The paper's Section IV-G results
(ρ = −0.47) run on the **v2 parametric panel**, which is the one used here.

### 1.2 Exact source

| Item | Value |
|---|---|
| Repository | `https://github.com/sfcheng-research/icdm-2026-reproduction` |
| Repository description | "Reproduction package for ICDM 2026 paper: Accuracy Is Not Service: A Decision-Aware Benchmark for Intermittent-Demand Forecasting" |
| File used | `dataset_release/RUF_synthetic_benchmark_5000_v2.zip` |
| Local copy | `data/raw/RUF_synthetic_benchmark_5000_v2.zip` |
| **SHA-256** | `d6f441077d922ee986a2b41a9a6ff500b0f9a0154f529ff5786bc334fb21dc61` |
| Size | 2,056,485 bytes |
| Zenodo concept DOI (from the archive's `.zenodo.json`) | **`10.5281/zenodo.20405841`** |
| Version | 2.0.0 |
| Generator seed | **42** (the release ships seed-42 panels; the repo's `results_release/` also contains seed-robustness runs for seeds **43–47**) |

Repository existence was confirmed via the GitHub API (repo id 1362074174, owner
`sfcheng-research`) — it was **not** assumed from the citation.

The **v1 structural panel** (`RUF_synthetic_benchmark_v1.zip`, SHA-256 recorded in the
archive's own `checksums.sha256`) is also present in `data/raw/`. It is **not** used: the
paper's camera-ready Section IV-G runs on v2, and v1 is cited only by the originally
submitted version. Do not mix them.

### 1.3 Licence — verified, not assumed

| Artefact | Licence | Where the text actually lives |
|---|---|---|
| **The dataset (RUF panels)** | **CC-BY-4.0** | Stated in **three** independent places inside the releases: `DATASET_README.md` ("License: **CC-BY-4.0**"), `DATASHEET.md` ("License **CC-BY-4.0**"), and the v2 archive's `.zenodo.json` (`"license": "cc-by-4.0"`) |
| The code in the repository | **MIT** | Root `LICENSE` — "Copyright (c) 2026 Joo Ern Chin, Shih-Fen Cheng, Aldy Gunawan" |

> **Two dead ends, recorded so nobody re-checks them.** (i) There is **no
> `dataset_release/LICENSE`** file — the URL returns literally `404: Not Found`.
> (ii) The **v2 archive's own `CITATION.cff` says `version: "1.0.0"`** while its
> `.zenodo.json` says `version: "2.0.0"`, and the CFF carries a provisional-author note.
> The `.zenodo.json` is the one that matches the archive's own `checksums.sha256`, so
> **cite version 2.0.0**. This is an inconsistency in the upstream release, not in our use
> of it.

**Consequence for this project:** CC-BY-4.0 permits redistribution and derivative works
**with attribution**. The derived, project-ready files in `data/ruf/` are therefore legal
to create, commit and share, provided the attribution in §4 below travels with them.

### 1.4 What was verified by actually loading the files

Everything in this table was produced by loading the CSVs and computing, not by reading a
description.

| Property | Value |
|---|---|
| Shape | **5,000 SKUs × 60 monthly periods** (`2018-09-01` … `2023-08-01`) |
| Zero fraction | **0.7008** |
| Total demand | 619,941 units |
| Values | non-negative, integer-valued, **no NaN** |
| Nonzero periods per SKU | min 1, p25 7, **median 12**, p75 26, max 60 |
| ADI (full series) | min 1.0, p25 2.31, **median 5.00**, p75 8.57, max 60 |
| CV² (full series) | median **0.317**; **undefined for 39 SKUs** (exactly one nonzero period) |
| Syntetos–Boylan (full series) | intermittent 3,222 · lumpy 1,289 · smooth 303 · erratic 147 · **dead 39** |

**The Rev2 §15.1 2×2 is populated on the real panel.** Using Rev2's reachable bands
(`ADI* = 4.0`, `CV²* = 0.49`) on the full series:

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| **ADI < 4** | **1,221** (24.4 %) | **764** (15.3 %) |
| **ADI ≥ 4** | **2,304** (46.1 %) | **672** (13.4 %) |

All four cells are populated, minimum share **13.4 %**. This directly answers the audit's
C-3 blocker: the reachable 2×2 is not an artefact of our synthetic design — the real panel
satisfies it. (Rev2 §15.1's "≥ 20 % per cell" bar is a requirement on **our generator**
(§30 gate 4.2), not on RUF, which must instead satisfy §16's "at least two cells
populated". It satisfies all four.)

### 1.5 ⚠️ What RUF does NOT provide — and the consequence for CAFR

These are the deliberate absences. Each is a stated limitation, not something to paper
over.

| Missing field | Why it matters | Resolution adopted |
|---|---|---|
| **No stock-out / censoring record** | The C-guard (§6.2) needs a stock-out flag. RUF is a *demand* panel, not an inventory transaction log, so **censoring is present but unobserved** | §16 already anticipates exactly this: it "materially weakens the C-guard and must be admitted as a limitation". An unobserved stock-out cannot be imputed, which **forces `fit_on = observed` on RUF** (§14.3 rule 6). The C-guard is exercised on the **labelled synthetic panel**, where the simulator knows the truth. **State this in the paper.** UCI Online Retail has the same problem. |
| **No ground-truth cause labels** | RQ1 (attribution accuracy) cannot be measured here | Expected, and already Rev2 §13's stated methodological point: *"You cannot validate a failure-attribution method on RUF or RAF alone… The generator panel is not a convenience — it is the only way to measure the new claim."* Attribution is measured on **our** labelled panel (Step 4); RUF tests whether the gain **transfers**. |
| **No backorder cost `B`** | The reward and the cost model need `B/H` | Derivable from RUF's own `service_level_target` via the newsvendor critical ratio `B/H = α/(1−α)`. See §1.6. |
| **60 periods, not 84** | Rev2 §14.1 assumes 24 burn-in + 60 evaluated = **84** | **This is the one genuine design decision RUF forces.** See §1.7. |

### 1.6 What RUF *does* supply that Rev2 did not anticipate having

`synthetic_inventory_planning_params_5y.csv` is a genuine planning-parameter table, and it
is better than the synthetic defaults Rev2 proposed:

| Rev2 constant | Rev2 default | RUF's actual value | Match |
|---|---|---|---|
| Lead time `L` | 2 periods | `planned_lead_time_months`: median **2** (1→2,428; 2→2,149; 3+→471; max 23) | ✅ median matches |
| Protection interval `R = L + 1` | 3 | `protection_period_months` == lead + 1 **exactly, for all 5,000 SKUs** (verified) | ✅ |
| Review period | 1 (monthly) | `review_period_months` = **1** for all SKUs | ✅ |
| Holding rate | — | `holding_rate_monthly` = **0.02**; `holding_cost_per_unit_month` = `unit_cost` × 0.02 | ✅ usable directly |
| `B/H` sweep | {4, 9, 19, 49} | `service_level_target` ∈ {0.95 (3,584), 0.90 (1,345), 0.80 (39), 0.85 (32)} → implied `B/H` = **{19, 9, 4, 5.67}** | ✅ Rev2's sweep is **principled on this panel**, not arbitrary |
| Incumbent policy | not anticipated | `safety_stock_units`, `reorder_point_units`, `min_stock_units`, `max_stock_units` ship per SKU | ✅ usable as a **real-world incumbent baseline** (an additional comparator, not required by Rev2) |

**Do not use RUF's raw lead times as the main-run `L`.** They reach 23 months against a
60-month panel. Rev2 §15.6 fixes `L = 2` as the default and sweeps `L ∈ {1, 2, 4}` in E6.
RUF's real value is preserved in `data/ruf/sku_attributes.parquet` as `lead_time_ruf` for a
documented realism variant — but the headline run keeps Rev2's `L = 2` so the comparison
stays pre-registered.

### 1.7 ⚠️ THE PANEL-LENGTH FINDING — a design decision, not a patch

**What Rev2 assumes.** §14.1: burn-in 24 + evaluated 60 = **84 periods**, "RUF has 84, so
24 + 60 = 84 exactly". §14.1's resolution ladder says: (1) check whether the panel supplies
more than 84 — attempt first; (2) **otherwise reduce `W_occ` to 48**, "a 48-period window
fits the 60-period evaluated span with margin".

**What RUF actually has.** **60 periods total** — not 60 evaluated plus 24 burn-in.

**Therefore both rungs of §14.1's ladder fail:**
- Rung 1 fails: no RUF panel supplies 84 periods. v2 is 60; v1 is also 60.
- Rung 2 fails: `24 + 48 = 72 > 60`. A 48-period window does **not** fit once burn-in is
  reserved.

This is precisely the contingency the handoff brief anticipated ("If RUF cannot provide the
required fields for CAFR, explicitly identify the missing fields and give the scientifically
justified solution"). It is **not** patched silently here. The decision, the options and a
recommendation are in **`14_HIMANSHI_COMPLETE_HANDOFF.md` §3.4 and §16**, and the
supporting arithmetic is in `data/ruf/panel_profile.json`.

**Also verified: the panel cannot be regenerated.** The v2 archive ships
`generate_synthetic_replicable_dataset_v2.py`, and it looks re-runnable — but **it is not**.
It reads the confidential real panel at module level:

```
REAL_DEMAND = PROJECT / "results/material_characterization/demand_matrix.csv"
REAL_FEAT   = PROJECT / "results/material_characterization/material_features_full.csv"
CATALOGUE   = PROJECT / "results/synthetic_replicable_dataset/_real_param_catalogue.csv"
```

Executing it fails immediately:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '.../results/material_characterization/demand_matrix.csv'
```

None of those three inputs ships in the public release. **So the published 5,000 × 60 panel
is the only publicly obtainable RUF artefact: it cannot be made longer, and it cannot be
regenerated with a different seed.** Anyone planning around "we'll just regenerate a longer
panel" should read this paragraph twice.

### 1.8 Other files in the v2 archive

Present and inspected, none of them needed for the core pipeline but all recorded:

| File | Contents | Use here |
|---|---|---|
| `synthetic_demand_matrix_5y.csv` | the 5,000 × 60 panel | **the primary input** |
| `synthetic_inventory_planning_params_5y.csv` | 32 planning columns | lead time, `R`, holding cost, target service → `sku_attributes.parquet` |
| `synthetic_material_metadata_5y.csv` | 39 per-material demand statistics | ADI/CV²/SB class → panel profile |
| `synthetic_full_data_generic_5y.csv` | order-line event log, 13 columns (`quantity_reserved` per line) | not needed — it is a **disaggregation of the same panel**, and it carries **no shipment or stock-out field**, confirming the censoring finding |
| `synthetic_material_catalog_5y.csv` | catalogue = metadata + id | not needed |
| `synthetic_context_materials_5y.csv` | tokenized ERP-like categoricals | not needed (all neutral labels) |
| `synthetic_map_prices.csv` | `material_id`, `map_price` | **= `unit_cost` in the planning params** |
| `synthetic_material_fidelity_5y.csv` | per-material fidelity audit | reference only |
| `period_lookup_5y.csv` | `period_index` → `calendar_month` | period labelling |
| `checksums.sha256` | 17 embedded checksums | integrity — verify on extract |

### 1.9 The M5 panel also ships (recorded, not yet adopted)

`dataset_release/m5_full38_panel/` in the same repository is a **250-item monthly M5
panel** (`demand_matrix.csv`, `material_features_full.csv`, `material_segments.csv`,
`synthetic_map_prices.csv`), aggregated from the public M5 competition data and
**stratified toward M5's sparsest items** — 56.8 % classify Syntetos–Boylan intermittent or
lumpy; item ids anonymised `M5_0001`…; prices synthetic.

**This is a materially better M5 option than Rev2 §13/§16 assumed.** Rev2 said "M5
(Kaggle) — requires a Kaggle account… Restrict to the top-N SKUs by coverage. M5 is *not*
intermittent by nature — it must be filtered". This pre-filtered, pre-stratified panel
needs **no Kaggle account** and arrives already intermittent. Adopting it removes a
blocking dependency and the "report N and why it was chosen" requirement.

---

## 2. Labelled synthetic panel — built by us (Step 4)

**Not downloaded.** Rev2 §13 lists "intermittent-demand generator released with arXiv
`2601.21844`" as the labelled source. **Do not build on that.** Rev2 §15 specifies our own
generator (`p`, `μ_z`, `k`, the seven injections, C0, ambiguous), which is **ours**, needs
**no external dataset**, and is the only way to get ground-truth causes that match the
seven-cause taxonomy. Rev2 already states this: *"Build it (Step 4). Do not wait for RUF —
the generator is ours."*

- **Parameters:** `T = 200`, `τ_inj = 120`; `p ∈ [0.05, 0.50]`; `μ_z ∈ [5, 200]`;
  `δ_t ~ Bernoulli(p)`; `z_t ~ Gamma(shape = k, scale = μ_z/k)`.
- **Panels:** 200 × 7 single-cause (1,400) + 200 C0 + 200 ambiguous = **1,800 series**,
  split **60/20/20** (1,080 / 360 / 360), stratified by cause.
- **Seeds:** **42** for train/val/test; **43** held out entirely (§22.4 item 5).
- **Ground truth:** `ground_truth.parquet` (§27) — every column of it is unreachable by the
  policy, enforced by an **exact column whitelist**, not a name blacklist (§14.3 rule 5).
- **OD-5** ("what does the `2601.21844` generator actually emit?") **ceases to block Step 4**
  once the generator is ours. OD-5 should be closed or narrowed.

---

## 3. Panels deliberately NOT used

| Dataset | Decision | Reason |
|---|---|---|
| **RAF** (5,000 SKUs × 84 months, via CRAN `fable.intermittent` / GitHub `gluon-ts`) | **Do NOT download** | **No stated licence** (OD-6, open — Dr. V. K. Chawla). Rev2 §13: *"Do not publish results computed on an unlicensed dataset."* Note the irony: RAF's 84 periods are exactly what §14.1 wanted — but licence wins over convenience. If Dr. Chawla confirms a licence in writing **and** the panel-length decision in §1.7 needs revisiting, RAF becomes the natural 84-period panel. |
| **Kaggle M5 raw** | Superseded | The pre-filtered M5 panel in §1.9 removes the account requirement. |
| **The confidential industrial panel** | Unavailable | Not released. RUF is its certified surrogate. |

---

## 4. Required attribution (must travel with any redistribution of `data/ruf/`)

```
This project uses the RUF synthetic benchmark panel (v2.0.0), released under
CC-BY-4.0 with the paper:

  Joo Ern Chin, Shih-Fen Cheng, Aldy Gunawan.
  "Accuracy Is Not Service: A Decision-Aware Benchmark for Intermittent-Demand
  Forecasting." IEEE ICDM 2026 (Applied Track). arXiv:2609.13840.

  Dataset: https://github.com/sfcheng-research/icdm-2026-reproduction
  DOI:     10.5281/zenodo.20405841
  Licence: CC-BY-4.0 (data) / MIT (code)

The files in data/ruf/ are DERIVED from that panel: reshaped to the CAFR input
contract, with burn-in ADI/CV^2 cells computed. The demand values are unmodified.
```

---

## 5. Integrity check — run this before trusting anything

```bash
cd data
python - <<'PY'
import hashlib
h = hashlib.sha256(open('raw/RUF_synthetic_benchmark_5000_v2.zip','rb').read()).hexdigest()
want = 'd6f441077d922ee986a2b41a9a6ff500b0f9a0154f529ff5786bc334fb21dc61'
print('OK' if h == want else 'MISMATCH', h)
PY
```

Then rebuild the derived panels and confirm the profile still matches:

```bash
python build_ruf_panel.py --burn-in 24
```

`build_ruf_panel.py` is a **pure function of the input CSVs** — no randomness, so a
re-run must reproduce `panel_profile.json` byte-for-byte. If it does not, the input is not
the artefact recorded here.

---

## 6. Summary of what this file changes about the plan

1. **RUF is confirmed, licence-clean, and usable** — CC-BY-4.0, GitHub source verified by
   API, SHA-256 recorded, contents inspected.
2. **RUF supplies almost every simulator input** (lead time, `R`, holding cost, target
   service) — better than Rev2 assumed, and it makes `B/H`'s sweep principled rather than
   arbitrary.
3. **RUF has 60 periods, not 84** — §14.1's plan and its stated fallback both fail. This is
   the one decision the user must make. It affects the **real-panel transfer** evaluation
   only: the labelled synthetic panel is ours, is unaffected, and still carries the
   attribution and the headline comparisons.
4. **RUF has no censoring record and no cause labels** — expected absences, each with a
   stated resolution, and each already anticipated by Rev2 §16 and §13.
5. **Censoring being unobserved forces `fit_on = observed` on RUF**, which weakens the
   C-guard there — a limitation to state, and a reason the C-guard's value is measured on
   the labelled panel (ablation A6a).
6. **The v2 generator cannot be re-run publicly** — verified by execution. The 5,000 × 60
   panel is the only obtainable RUF artefact.
7. **A pre-filtered, no-account M5 panel exists** in the same repository — adopt it and drop
   the Kaggle dependency.
