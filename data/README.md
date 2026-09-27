# `data/` — the CAFR data folder

Everything here is built from the **RUF v2** benchmark panel. Full provenance, SHA-256 and
licence text: **`DATA.md`**. Read that first.

> **Attribution.** This project uses the RUF synthetic benchmark panel (v2.0.0), released
> under **CC-BY-4.0** with Chin, Cheng & Gunawan, *"Accuracy Is Not Service: A Decision-Aware
> Benchmark for Intermittent-Demand Forecasting"*, IEEE ICDM 2026 (Applied Track),
> arXiv:2609.13840. DOI `10.5281/zenodo.20405841`. **This attribution must travel with any
> redistribution of `ruf/`.**

---

## What is here

```
data/
├── DATA.md                     provenance, licences, SHA-256, integrity checks
├── README.md                   this file
├── build_ruf_panel.py          the build script — pure function, no randomness
├── raw/
│   ├── RUF_synthetic_benchmark_5000_v2.zip    the source archive (unmodified)
│   ├── RUF_synthetic_benchmark_v1.zip         v1 — present, NOT used
│   ├── REPO_LICENSE_MIT.txt                   the repo's MIT code licence
│   ├── REPRODUCTION_REPO_*.md / *.cff         the repo's own docs, as downloaded
│   └── v2/                                    the nine CSVs, extracted
└── ruf/                        ← THE PROJECT-READY FILES
    ├── demand_panel.parquet
    ├── sku_meta.parquet
    ├── sku_attributes.parquet
    └── panel_profile.json
```

---

## The four project-ready files

### 1. `ruf/demand_panel.parquet` — 300,000 rows

The Rev 2 §27 demand contract, long format. **The primary input to the simulator.**

| Column | dtype | Meaning |
|---|---|---|
| `sku_id` | object (str) | `SYN5Y_0001` … `SYN5Y_5000` |
| `period` | int32 | 0 … 59 (monthly, 2018-09 … 2023-08) |
| `demand_observed` | float64 | **what the policy may see.** Integer-valued, ≥ 0 |
| `demand_true` | float64 | ⛔ **ground truth — must NEVER reach the policy** |
| `stockout_flag` | bool | ⛔ **`False` everywhere** (see the note below) |
| `panel` | object (str) | `"ruf"` |

Example rows:

| sku_id | period | demand_observed | demand_true | stockout_flag | panel |
|---|---|---|---|---|---|
| `SYN5Y_0001` | 0 | 0.0 | 0.0 | False | `ruf` |
| `SYN5Y_0001` | 1 | 0.0 | 0.0 | False | `ruf` |
| `SYN5Y_0001` | 2 | 0.0 | 0.0 | False | `ruf` |

> **`stockout_flag` is `False` everywhere and that is correct.** RUF is a **demand** panel,
> not an inventory transaction log — it has no stock-out record, so censoring is present but
> **unobserved**. Inventing stock-outs here would be fabricating data. Censoring is generated
> downstream by the **inventory simulator** when a policy under-protects. Consequence:
> **`fit_on = observed` is forced on RUF** (Rev 2 §14.3 rule 6). See handoff §3.6.

### 2. `ruf/sku_meta.parquet` — 5,000 rows

Static per-SKU attributes. **Exactly the seven columns Rev 2 §27 declares.** Safe as a policy
input — these are static and computed from the **burn-in only**.

| Column | dtype | Meaning |
|---|---|---|
| `sku_id` | object (str) | join key |
| `lead_time` | int64 | **always 2** — Rev 2 §15.6's default, not RUF's own |
| `unit_cost` | float64 | RUF's `map_price` |
| `holding_rate` | float64 | **0.02** monthly |
| `adi_init` | float64 | **ADI over the first 24 periods only** |
| `cv2_init` | float64 | **CV² over the first 24 periods only.** **NaN** where < 2 nonzero |
| `sb_cell_init` | object (str) | one of `moderate_lowdisp`, `high_lowdisp`, `moderate_highdisp`, `high_highdisp`, **`dead`** |

Example rows:

| sku_id | lead_time | unit_cost | holding_rate | adi_init | cv2_init | sb_cell_init |
|---|---|---|---|---|---|---|
| `SYN5Y_0001` | 2 | 262.96 | 0.02 | 24.0 | **NaN** | **`dead`** |
| `SYN5Y_0002` | 2 | 20.24 | 0.02 | 6.0 | 0.181070 | `high_lowdisp` |
| `SYN5Y_0003` | 2 | 21.81 | 0.02 | 1.0 | 0.481521 | `moderate_lowdisp` |

> **`SYN5Y_0001` is not a bug.** It has fewer than 2 nonzero periods in its first 24 months,
> so CV² is undefined. The build script writes **NaN**, not `0.0` — writing `0.0` would file
> the SKU into a `_lowdisp` cell it does not belong to. **Do not "fix" it.** At burn-in 24,
> **1,304 of 5,000 SKUs (26.1 %) are `dead`**; they abstain by construction and must be
> reported in T1, not dropped. See handoff §3.11.

### 3. `ruf/sku_attributes.parquet` — 5,000 rows

RUF's **real** planning parameters. ⛔ **Evaluation and reporting ONLY. Never a policy input.**

| Column | Meaning |
|---|---|
| `sku_id` | join key |
| `lead_time_ruf` | RUF's real planned lead time (median 2, **max 23**) |
| `protection_period_ruf` | `== lead_time_ruf + 1`, for all 5,000 (verified) |
| `review_period_months` | always 1 |
| `holding_cost_per_unit_month` | `unit_cost × 0.02` |
| `service_level_target` | ∈ {0.95 (3,584), 0.90 (1,345), 0.80 (39), 0.85 (32)} |
| `safety_stock_units` | RUF's **incumbent** policy — a real-world comparator |
| `reorder_point_units` | RUF's **incumbent** policy |
| `syntetos_boylan` | SB class over the **full series** |
| `adi_full` | ADI over the **full series** |
| `cv_squared_full` | CV² over the **full series** (NaN for the 39 dead SKUs) |

> **⚠️ THIS IS THE ONE FILE THAT CAN SILENTLY RUIN THE PROJECT.**
> `adi_full` and `cv_squared_full` are computed over **all 60 periods**. If either reaches the
> feature vector, the detector is handed the post-regime answer — a **direct ground-truth
> leak**, worse than an ordinary look-ahead. That is exactly why Rev 2 §27 replaced Rev 1's
> `class_adi`/`class_cv2` with burn-in-only `adi_init`/`cv2_init`/`sb_cell_init`. **Use this
> file for evaluation, realism variants and the paper's tables. Never load it in the policy or
> the monitor.** The `policy_features` whitelist test is what enforces it.

### 4. `ruf/panel_profile.json`

T1's headline numbers, the detector-support arithmetic, and the honesty notes. **Documentation
only — not an input to anything.**

Key values: `n_skus` 5000, `n_periods` 60, `burn_in_periods` 24, `evaluated_periods` 36,
**`panel_length_shortfall` 24**, `zero_fraction` 0.70079, `total_demand_units` 619941.0,
`dead_skus_in_burn_in` 1304, `censoring_observed` **false**, `generator_rerunnable_publicly`
**false**, `detector_evidence_support_fraction` {C1 W=24 **0.5366**, C3 W=60 0.6074 / W=48
0.573 / W=36 0.5158, C4 W=24 **0.4228**}.

---

## How to rebuild

`build_ruf_panel.py` is a **pure function of the input CSVs — no randomness anywhere.** A
re-run must reproduce `panel_profile.json` byte-for-byte. If it does not, the input is not the
artefact recorded in `DATA.md`.

```bash
cd data

# 1. verify the source archive
python -c "import hashlib;print(hashlib.sha256(open('raw/RUF_synthetic_benchmark_5000_v2.zip','rb').read()).hexdigest())"
# expect d6f441077d922ee986a2b41a9a6ff500b0f9a0154f529ff5786bc334fb21dc61

# 2. rebuild the derived panels
python build_ruf_panel.py --burn-in 24

# options
#   --raw ./raw        source folder
#   --out ./ruf        output folder
#   --burn-in 24       periods used for adi_init/cv2_init (Rev 2 §14.1)
#   --panel-name ruf   value written to the `panel` column
```

**If the archives are missing**, download them:

```bash
curl -L -o raw/RUF_synthetic_benchmark_5000_v2.zip \
  https://github.com/sfcheng-research/icdm-2026-reproduction/raw/main/dataset_release/RUF_synthetic_benchmark_5000_v2.zip
```

---

## The three findings this folder exists to record

1. **RUF has 60 periods, not 84.** Rev 2 §14.1 assumed 24 burn-in + 60 evaluated = 84, and its
   stated fallback (`W_occ = 48` → 72) **also does not fit**. Both rungs fail. This is
   decision **H-1** in the handoff (§3.4) and it needs the team's sign-off.
2. **Censoring is unobserved**, which forces `fit_on = observed` on RUF. Decision **H-3**
   (§3.6). A stated limitation, not a choice.
3. **The generator cannot be re-run publicly** — verified by execution, not assumed. It reads
   the confidential real panel at module level and dies with `FileNotFoundError`. So the
   published 5,000 × 60 panel is the **only** obtainable RUF artefact: it cannot be made
   longer, and it cannot be regenerated with a different seed.

**Panels deliberately NOT used:** RAF (no stated licence — OD-6), Kaggle M5 raw (superseded by
the pre-filtered 250-item panel in the same repository — decision **H-4**), and the
confidential industrial panel (not released).
