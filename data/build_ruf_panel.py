#!/usr/bin/env python3
"""
build_ruf_panel.py — build the CAFR project-ready panels from the RUF v2 release.

Source  : RUF_synthetic_benchmark_5000_v2.zip
          https://github.com/sfcheng-research/icdm-2026-reproduction
          data/dataset_release/RUF_synthetic_benchmark_5000_v2.zip
Licence : CC-BY-4.0 (data) / MIT (code) — see ../DATA.md
SHA-256 : recorded in ../DATA.md, verified on extract.

Outputs (all under ./ruf/):
    demand_panel.parquet    the §27 input contract, long format
    sku_meta.parquet        the §27 contract: EXACTLY the seven declared columns
    sku_attributes.parquet  RUF's real planning attributes (evaluation-only,
                            never a policy input — see the header note)
    panel_profile.json      the T1 dataset-summary numbers + detector support

Determinism: pure function of the input CSVs. No randomness anywhere.

Usage:
    python build_ruf_panel.py --raw ./raw --out ./ruf --burn-in 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --- Revision 2 constants (11_technical_specification_rev2.md) ---------------
ADI_STAR = 4.0        # §15.1 / §6.3 C6 — the reachable boundary, NOT 1.32
CV2_STAR = 0.49       # §15.1 — the SB dispersion cut-off
REV2_PANEL_PERIODS = 84   # what §14.1 assumed (24 burn-in + 60 evaluated)
REV2_W_OCC = 60           # §6.1
REV2_W_OCC_FALLBACK = 48  # §30 gate 3.5 option 2

# Detector evidence floors, §6.1 + §6.4. (cause -> (horizon, min_nonzero))
DETECTOR_FLOORS = {
    "C1": (24, 8),    # W_short, ≥8 nonzero periods
    "C3": (None, 10), # W_occ (swept below), ≥10 nonzero periods
    "C4": (24, 10),   # W_sz, ≥10 nonzero periods
}
# C2/C6/C7 count ALL periods, not nonzero ones (coverage observations /
# window periods / usable periods), so they are satisfied by t >= horizon on any
# panel and are not sparsity-limited. They are therefore not tabulated here.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_extracted(raw: Path) -> Path:
    """Extract the RUF v2 archive into raw/v2/ (idempotent) and return the dir."""
    v2 = raw / "v2"
    marker = v2 / "synthetic_demand_matrix_5y.csv"
    if marker.exists():
        return v2
    zips = list(raw.glob("*.zip"))
    if not zips:
        raise SystemExit(
            f"No .zip in {raw}. Download "
            "data/dataset_release/RUF_synthetic_benchmark_5000_v2.zip "
            "from the reproduction repo into that folder first."
        )
    z2 = [z for z in zips if "v2" in z.name.lower()]
    if not z2:
        raise SystemExit(f"No *_v2*.zip found in {raw}; found {[z.name for z in zips]}")
    v2.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z2[0]) as zf:
        zf.extractall(v2)
    if not marker.exists():
        raise SystemExit(f"{z2[0].name} did not contain {marker.name}")
    return v2


def sb_cell(adi: float, cv2: float) -> str:
    """The reachable ADI-band x CV2-band 2x2 of Rev2 §15.1."""
    if not np.isfinite(cv2):
        return "dead"     # < 2 nonzero periods: CV^2 is undefined, no cell exists
    band = "moderate" if adi < ADI_STAR else "high"
    disp = "lowdisp" if cv2 < CV2_STAR else "highdisp"
    return f"{band}_{disp}"


def burn_in_stats(vals: np.ndarray, burn: int) -> tuple[float, float]:
    """
    ADI and CV^2 on the first `burn` periods ONLY (Rev2 §27 — never the full series).

    A SKU with fewer than 2 nonzero periods in the burn-in has an UNDEFINED CV^2
    (sample sd of one observation). That is returned as NaN, not 0.0 — silently
    writing 0.0 would file the SKU into a `_lowdisp` cell it does not belong to.
    Such SKUs are labelled `dead` and abstain by construction (Rev2 §8: they can
    never reach C1's n>=8 or C3/C4's n>=10 evidence floor).
    """
    w = vals[:burn]
    nz = w[w > 0]
    adi = len(w) / len(nz) if len(nz) else float(len(w))
    if len(nz) >= 2 and nz.mean() > 0:
        cv2 = float((nz.std(ddof=1) / nz.mean()) ** 2)
    else:
        cv2 = float("nan")
    return float(adi), cv2


def supportable(nz: np.ndarray, horizon: int, need: int) -> np.ndarray:
    """
    Per SKU: is there ANY trailing window of length `horizon` (over the whole
    observable series, burn-in included — Rev2 §14.3 rule 1 permits s <= t)
    containing >= `need` nonzero periods?

    `nz` is the 2-D (n_sku x T) boolean occurrence matrix. The cumsum MUST be
    taken along axis=1; a bare np.cumsum flattens the matrix and silently
    returns a per-period statistic instead of a per-SKU one.

    Reported as the fraction of the panel for which the detector can EVER fire.
    """
    n_sku, T = nz.shape
    if horizon > T:
        return np.zeros(n_sku, dtype=bool)
    c = np.cumsum(
        np.concatenate([np.zeros((n_sku, 1), dtype=np.int64), nz.astype(np.int64)], axis=1),
        axis=1,
    )
    counts = c[:, horizon:] - c[:, :-horizon]      # (n_sku, T-horizon+1)
    return (counts >= need).any(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path(__file__).parent / "raw")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "ruf")
    ap.add_argument("--burn-in", type=int, default=24,
                    help="periods used to initialise + compute adi_init/cv2_init (Rev2 §14.1)")
    ap.add_argument("--panel-name", type=str, default="ruf")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    v2 = ensure_extracted(args.raw)
    src_zip = next(z for z in args.raw.glob("*.zip") if "v2" in z.name.lower())
    print(f"source archive : {src_zip.name}")
    print(f"sha256         : {sha256(src_zip)}")

    dm = pd.read_csv(v2 / "synthetic_demand_matrix_5y.csv")
    ip = pd.read_csv(v2 / "synthetic_inventory_planning_params_5y.csv")
    md = pd.read_csv(v2 / "synthetic_material_metadata_5y.csv")

    id_col = dm.columns[0]
    period_cols = [c for c in dm.columns if c != id_col]
    T = len(period_cols)
    vals = dm[period_cols].to_numpy(dtype="float64")
    n_sku = vals.shape[0]
    print(f"panel          : {n_sku} SKUs x {T} monthly periods "
          f"({period_cols[0]} .. {period_cols[-1]})")

    # ---------------- 1. demand_panel.parquet (Rev2 §27 contract) -------------
    # RUF is a DEMAND panel, not an inventory transaction log: it carries no
    # stock-out record, so censoring is present-but-UNOBSERVED (Rev2 §16).
    # Therefore stockout_flag is False everywhere and demand_true == demand_observed.
    # This is the honest encoding — do NOT invent stock-outs here. The inventory
    # simulator (M5) generates censoring downstream when a policy under-protects.
    long = pd.DataFrame({
        "sku_id": np.repeat(dm[id_col].to_numpy(), T),
        "period": np.tile(np.arange(T, dtype="int32"), n_sku),
        "demand_observed": vals.reshape(-1),
        "demand_true": vals.reshape(-1),
        "stockout_flag": np.zeros(n_sku * T, dtype=bool),
        "panel": np.full(n_sku * T, args.panel_name, dtype=object),
    })
    long.to_parquet(args.out / "demand_panel.parquet", index=False)
    print(f"wrote demand_panel.parquet   {len(long):,} rows")

    # ---------------- 2. sku_meta.parquet (EXACTLY the §27 seven) -------------
    need = ["sku_id", "lead_time", "unit_cost", "holding_rate",
            "adi_init", "cv2_init", "sb_cell_init"]
    ip_by = ip.set_index(ip.columns[0])
    rows = []
    for i, sku in enumerate(dm[id_col].to_numpy()):
        adi, cv2 = burn_in_stats(vals[i], args.burn_in)
        rows.append({
            "sku_id": sku,
            # Rev2 §15.6 default L = 2. RUF's OWN lead time is preserved in
            # sku_attributes.parquet as `lead_time_ruf` for the realism variant.
            "lead_time": 2,
            "unit_cost": float(ip_by.loc[sku, "unit_cost"]),
            "holding_rate": float(ip_by.loc[sku, "holding_rate_monthly"]),
            "adi_init": adi,
            "cv2_init": cv2,
            "sb_cell_init": sb_cell(adi, cv2),
        })
    meta = pd.DataFrame(rows)[need]
    meta["sku_id"] = meta["sku_id"].astype(str)
    meta.to_parquet(args.out / "sku_meta.parquet", index=False)
    print(f"wrote sku_meta.parquet        {len(meta):,} SKUs, burn-in = {args.burn_in}")

    # ---------------- 3. sku_attributes.parquet (evaluation-only) -------------
    # NOT a policy input. Kept out of sku_meta so the §27 contract stays exact.
    attr = pd.DataFrame({
        "sku_id": dm[id_col].astype(str),
        "lead_time_ruf": ip["planned_lead_time_months"].to_numpy(dtype="int32"),
        "protection_period_ruf": ip["protection_period_months"].to_numpy(dtype="int32"),
        "review_period_months": ip["review_period_months"].to_numpy(dtype="int32"),
        "holding_cost_per_unit_month": ip["holding_cost_per_unit_month"].to_numpy(float),
        "service_level_target": ip["service_level_target"].to_numpy(float),
        "safety_stock_units": ip["safety_stock_units"].to_numpy(float),
        "reorder_point_units": ip["reorder_point_units"].to_numpy(float),
        "syntetos_boylan": md["syntetos_boylan"].astype(str).to_numpy(),
        "adi_full": md["adi"].to_numpy(float),
        "cv_squared_full": md["cv_squared"].to_numpy(float),
    })
    attr.to_parquet(args.out / "sku_attributes.parquet", index=False)
    print(f"wrote sku_attributes.parquet  {len(attr):,} SKUs (evaluation-only)")

    # ---------------- 4. panel_profile.json (table T1 + honesty checks) -------
    nz = vals > 0
    n_nonzero = nz.sum(axis=1)
    adi_full = md["adi"].to_numpy(float)
    cv2_full = md["cv_squared"].to_numpy(float)          # 39 NaNs: the "dead" SKUs

    def _cells(pairs):
        s = pd.Series([sb_cell(a, c) for a, c in pairs])
        return {k: int(v) for k, v in s.value_counts().items()}

    cells = _cells(zip(meta["adi_init"], meta["cv2_init"]))
    cells_full = _cells(zip(adi_full, cv2_full))
    n_dead_full = int(np.isnan(cv2_full).sum())
    n_dead_burn = int((meta["sb_cell_init"] == "dead").sum())

    support = {}
    for cause, (horizon, need_nz) in DETECTOR_FLOORS.items():
        horizons = [REV2_W_OCC, REV2_W_OCC_FALLBACK, 36] if horizon is None else [horizon]
        support[cause] = {}
        for h in horizons:
            support[cause][f"W={h}"] = round(
                float(supportable(nz, h, need_nz).mean()), 4)

    profile = {
        "panel": args.panel_name,
        "source": "RUF_synthetic_benchmark_5000_v2.zip (CC-BY-4.0)",
        "source_sha256": sha256(src_zip),
        "generator_seed": 42,
        "n_skus": int(n_sku),
        "n_periods": int(T),
        "period_labels": [period_cols[0], period_cols[-1]],
        "burn_in_periods": args.burn_in,
        "evaluated_periods": int(T - args.burn_in),
        "rev2_assumed_periods": REV2_PANEL_PERIODS,
        "panel_length_shortfall": int(REV2_PANEL_PERIODS - T),
        "total_demand_units": float(vals.sum()),
        "mean_demand_per_sku_period": round(float(vals.mean()), 6),
        "zero_fraction": round(float((vals == 0).mean()), 6),
        "all_values_non_negative": bool((vals >= 0).all()),
        "all_values_integer_valued": bool(np.allclose(vals, np.round(vals))),
        "any_nan_in_demand": bool(np.isnan(vals).any()),
        "nonzero_periods_per_sku": {
            "min": int(n_nonzero.min()), "p10": float(np.quantile(n_nonzero, .10)),
            "p25": float(np.quantile(n_nonzero, .25)), "median": float(np.median(n_nonzero)),
            "p75": float(np.quantile(n_nonzero, .75)), "p90": float(np.quantile(n_nonzero, .90)),
            "max": int(n_nonzero.max()),
        },
        "adi_full_series": {
            "min": float(adi_full.min()), "p25": float(np.quantile(adi_full, .25)),
            "median": float(np.median(adi_full)), "p75": float(np.quantile(adi_full, .75)),
            "max": float(adi_full.max()),
        },
        "cv_squared_full_series": {
            "min": float(np.nanmin(cv2_full)), "p25": float(np.nanquantile(cv2_full, .25)),
            "median": float(np.nanmedian(cv2_full)), "p75": float(np.nanquantile(cv2_full, .75)),
            "max": float(np.nanmax(cv2_full)),
            "n_undefined": n_dead_full,
        },
        "syntetos_boylan_full_series":
            md["syntetos_boylan"].value_counts().to_dict(),
        "dead_skus_full_series": n_dead_full,
        "dead_skus_in_burn_in": n_dead_burn,
        "cells_adi_star_4_cv2_star_0p49_full_series": cells_full,
        "cells_from_burn_in_adi_init_cv2_init": cells,
        "cells_all_four_populated_full":
            len([k for k in cells_full if k != "dead"]) == 4,
        "cells_min_share_full": round(
            min(v for k, v in cells_full.items() if k != "dead") / n_sku, 4),
        "detector_evidence_support_fraction": support,
        "detector_support_note": (
            "Fraction of SKUs for which at least one trailing window of the given "
            "horizon ever contains the required number of NONZERO periods. C2, C6 "
            "and C7 are not sparsity-limited (they count all periods) and are "
            "omitted. C1 needs >=8 nonzero in 24; C3 needs >=10 in W_occ; C4 needs "
            ">=10 in 24. Below these floors the monitor ABSTAINS BY DESIGN "
            "(Rev2 §8, insufficient evidence) — a low number here is a property of "
            "the panel's sparsity, not a defect, and must be reported in T1."
        ),
        "censoring_observed": False,
        "censoring_note": (
            "RUF is a demand panel with no stock-out record. Censoring is present "
            "but UNOBSERVED (Rev2 §16). Consequences: (i) the C-guard has no "
            "stock-out flag to read on this panel; (ii) an unobserved stock-out "
            "cannot be imputed, which forces fit_on = observed on RUF (§14.3 rule 6) "
            "unless the operator's own stock-out record is obtained. Record this as "
            "a stated limitation. UCI Online Retail has the same problem."
        ),
        "ground_truth_present": False,
        "ground_truth_note": (
            "RUF carries no cause labels. Attribution accuracy (RQ1) is measured on "
            "the labelled synthetic panel only (Rev2 §13). This is expected, not a "
            "defect — see Rev2 §13's methodological note."
        ),
        "generator_rerunnable_publicly": False,
        "generator_note": (
            "The v2 archive ships generate_synthetic_replicable_dataset_v2.py, but "
            "it is NOT runnable from the public release: it reads the confidential "
            "real panel (results/material_characterization/demand_matrix.csv and "
            "material_features_full.csv) and a private _real_param_catalogue.csv at "
            "module level and exits with FileNotFoundError. VERIFIED by execution "
            "2026-09-23. The published 5,000 x 60 panel is therefore the ONLY "
            "publicly obtainable RUF artefact, and the panel cannot be regenerated "
            "longer or with a different seed."
        ),
    }
    profile = json.loads(json.dumps(profile, default=str))
    (args.out / "panel_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"wrote panel_profile.json")

    print("\n--- T1 headline numbers ---")
    print(f"  SKUs x periods        : {n_sku:,} x {T}  (Rev2 assumed 84 -> shortfall {REV2_PANEL_PERIODS - T})")
    print(f"  zero fraction         : {profile['zero_fraction']:.4f}")
    print(f"  nonzero/SKU  median   : {profile['nonzero_periods_per_sku']['median']:.0f}")
    print(f"  dead SKUs (full)      : {n_dead_full}   (burn-in: {n_dead_burn})")
    print(f"  ADI          median   : {profile['adi_full_series']['median']:.3f}")
    print(f"  CV^2         median   : {profile['cv_squared_full_series']['median']:.3f}")
    print(f"  ADI*CV^2 cells (full) : {profile['cells_adi_star_4_cv2_star_0p49_full_series']}")
    print(f"  all four populated    : {profile['cells_all_four_populated_full']} "
          f"(min share {profile['cells_min_share_full']:.2%})")
    print("  detector support (fraction of panel the cause can EVER fire on):")
    for c, d in support.items():
        print(f"      {c}: " + "  ".join(f"{k} {v:.1%}" for k, v in d.items()))


if __name__ == "__main__":
    main()
