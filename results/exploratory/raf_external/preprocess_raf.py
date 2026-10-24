"""Preprocess and verify the historical RAF spare-parts demand dataset.

Source 1 (Primary):
    GitHub repo: danieldehaan96/spdf (branch: main)
    URL: https://raw.githubusercontent.com/danieldehaan96/spdf/main/RAF%20data%20-%207%20years%20demand%20%20-%205000%20items.xls
    File: 'RAF data - 7 years demand - 5000 items.xls'
    Size: 3,001,344 bytes
    SHA-256: 1165f2d54759bf7661becc39a2e80265626b59965c61f2ad8ad12a3d7940cf09

Source 2 (Cross-validation):
    GitHub repo: canerturkmen/gluon-ts (branch: intermittent-datasets)
    URL: https://raw.githubusercontent.com/canerturkmen/gluon-ts/intermittent-datasets/datasets/intermittent_raf.tar.gz
    Size: 204,987 bytes
    SHA-256: 66d9674b964b32c0db7243d40cdf50401b3611f0e833a179d01e98d25bfa234c

Generates:
    - results/exploratory/raf_external/data/demand_panel.parquet
    - results/exploratory/raf_external/data/sku_meta.parquet
    - results/exploratory/raf_external/data/sku_attributes.parquet
    - results/exploratory/raf_external/raf_data_profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import xlrd

from cafr.data.schema import (
    DEMAND_PANEL_COLUMNS,
    SKU_META_COLUMNS,
    validate_frame,
)

# Rev2 constants
ADI_STAR = 4.0
CV2_STAR = 0.49
REV2_W_OCC = 60
REV2_W_OCC_FALLBACK = 48

DETECTOR_FLOORS = {
    "C1": (24, 8),
    "C3": (None, 10),
    "C4": (24, 10),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sb_cell(adi: float, cv2: float) -> str:
    if not np.isfinite(cv2):
        return "dead"
    band = "moderate" if adi < ADI_STAR else "high"
    disp = "lowdisp" if cv2 < CV2_STAR else "highdisp"
    return f"{band}_{disp}"


def burn_in_stats(vals: np.ndarray, burn: int) -> tuple[float, float]:
    w = vals[:burn]
    nz = w[w > 0]
    adi = len(w) / len(nz) if len(nz) else float(len(w))
    if len(nz) >= 2 and nz.mean() > 0:
        cv2 = float((nz.std(ddof=1) / nz.mean()) ** 2)
    else:
        cv2 = float("nan")
    return float(adi), cv2


def full_series_stats(vals: np.ndarray) -> tuple[float, float, str]:
    nz = vals[vals > 0]
    adi = len(vals) / len(nz) if len(nz) else float(len(vals))
    if len(nz) >= 2 and nz.mean() > 0:
        cv2 = float((nz.std(ddof=1) / nz.mean()) ** 2)
    else:
        cv2 = float("nan")

    # Syntetos-Boylan standard cutoffs: ADI=1.32, CV2=0.49
    if not np.isfinite(cv2):
        sb = "dead"
    elif adi >= 1.32 and cv2 >= 0.49:
        sb = "lumpy"
    elif adi >= 1.32 and cv2 < 0.49:
        sb = "intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        sb = "erratic"
    else:
        sb = "smooth"
    return float(adi), cv2, sb


def supportable(nz: np.ndarray, horizon: int, need: int) -> np.ndarray:
    n_sku, T = nz.shape
    if horizon > T:
        return np.zeros(n_sku, dtype=bool)
    c = np.cumsum(
        np.concatenate([np.zeros((n_sku, 1), dtype=np.int64), nz.astype(np.int64)], axis=1),
        axis=1,
    )
    counts = c[:, horizon:] - c[:, :-horizon]
    return (counts >= need).any(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    ap.add_argument("--raw-xls", type=Path, default=base_dir / "raw" / "RAF data - 7 years demand - 5000 items.xls")
    ap.add_argument("--raw-tar", type=Path, default=base_dir / "raw" / "intermittent_raf.tar.gz")
    ap.add_argument("--out-data", type=Path, default=base_dir / "data")
    ap.add_argument("--out-profile", type=Path, default=base_dir / "raf_data_profile.json")
    ap.add_argument("--burn-in", type=int, default=24)
    args = ap.parse_args()

    args.out_data.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw XLS from: {args.raw_xls}")
    sha_xls = sha256_file(args.raw_xls)
    sha_tar = sha256_file(args.raw_tar) if args.raw_tar.exists() else "NOT_FOUND"

    wb = xlrd.open_workbook(args.raw_xls)
    sheet = wb.sheet_by_index(0)

    headers = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
    desc_col = headers[1]
    lt_col = headers[2]
    price_col = headers[3]
    demand_col_names = headers[4:]
    T = len(demand_col_names)
    n_sku = sheet.nrows - 1

    print(f"Read {n_sku} SKUs x {T} monthly periods ({demand_col_names[0]} .. {demand_col_names[-1]})")

    # Extract arrays
    item_refs = []
    descriptions = []
    lead_times_raw = []
    prices_raw = []
    demand_matrix = np.zeros((n_sku, T), dtype=np.float64)

    for r in range(1, sheet.nrows):
        row = sheet.row_values(r)
        item_refs.append(int(round(float(row[0]))))
        descriptions.append(str(row[1]).strip())
        lead_times_raw.append(float(row[2]))
        prices_raw.append(float(row[3]))
        demand_matrix[r - 1, :] = [float(v) for v in row[4:]]

    # Cross-verify with tar.gz if present
    if args.raw_tar.exists():
        with tarfile.open(args.raw_tar, "r:gz") as tar:
            f_test = tar.extractfile("intermittent_raf/test/data.json")
            test_json = json.load(f_test)
            mismatches = 0
            for idx in range(n_sku):
                if not np.allclose(demand_matrix[idx], test_json[idx]["target"]):
                    mismatches += 1
            print(f"Cross-verification with GluonTS: {mismatches} mismatches out of {n_sku} series.")
            if mismatches > 0:
                raise ValueError(f"Cross-verification failed with {mismatches} mismatches!")

    # Format SKU IDs as standard strings: RAF_0001 .. RAF_5000
    sku_ids = [f"RAF_{ref:04d}" for ref in item_refs]

    # 1. demand_panel.parquet
    long_df = pd.DataFrame({
        "sku_id": np.repeat(sku_ids, T),
        "period": np.tile(np.arange(T, dtype="int32"), n_sku),
        "demand_observed": demand_matrix.reshape(-1),
        "demand_true": demand_matrix.reshape(-1),
        "stockout_flag": np.zeros(n_sku * T, dtype=bool),
        "panel": np.full(n_sku * T, "raf", dtype=object),
    })
    validate_frame(long_df, DEMAND_PANEL_COLUMNS, name="demand_panel.parquet")
    out_demand_path = args.out_data / "demand_panel.parquet"
    long_df.to_parquet(out_demand_path, index=False)
    print(f"Wrote demand_panel.parquet: {len(long_df):,} rows -> {out_demand_path}")

    # 2. sku_meta.parquet (burn-in only)
    meta_rows = []
    for i in range(n_sku):
        adi_b, cv2_b = burn_in_stats(demand_matrix[i], args.burn_in)
        meta_rows.append({
            "sku_id": sku_ids[i],
            "lead_time": 2,  # Standard CAFR default
            "unit_cost": float(prices_raw[i]),
            "holding_rate": 0.02,  # Standard monthly holding rate
            "adi_init": adi_b,
            "cv2_init": cv2_b,
            "sb_cell_init": sb_cell(adi_b, cv2_b),
        })
    meta_df = pd.DataFrame(meta_rows)
    validate_frame(meta_df, SKU_META_COLUMNS, name="sku_meta.parquet")
    out_meta_path = args.out_data / "sku_meta.parquet"
    meta_df.to_parquet(out_meta_path, index=False)
    print(f"Wrote sku_meta.parquet: {len(meta_df):,} SKUs -> {out_meta_path}")

    # 3. sku_attributes.parquet (full-series & metadata)
    attr_rows = []
    adi_full_list = []
    cv2_full_list = []
    sb_full_list = []

    for i in range(n_sku):
        adi_f, cv2_f, sb_f = full_series_stats(demand_matrix[i])
        adi_full_list.append(adi_f)
        cv2_full_list.append(cv2_f)
        sb_full_list.append(sb_f)
        attr_rows.append({
            "sku_id": sku_ids[i],
            "description": descriptions[i],
            "lead_time_raf": int(round(lead_times_raw[i])),
            "price_gbp": float(prices_raw[i]),
            "total_demand": float(demand_matrix[i].sum()),
            "nonzero_periods": int(np.sum(demand_matrix[i] > 0)),
            "syntetos_boylan": sb_f,
            "adi_full": adi_f,
            "cv_squared_full": cv2_f,
        })
    attr_df = pd.DataFrame(attr_rows)
    out_attr_path = args.out_data / "sku_attributes.parquet"
    attr_df.to_parquet(out_attr_path, index=False)
    print(f"Wrote sku_attributes.parquet: {len(attr_df):,} SKUs -> {out_attr_path}")

    # 4. raf_data_profile.json
    nz = demand_matrix > 0
    adi_full_arr = np.array(adi_full_list)
    cv2_full_arr = np.array(cv2_full_list)

    def _cells(pairs):
        s = pd.Series([sb_cell(a, c) for a, c in pairs])
        return {k: int(v) for k, v in s.value_counts().items()}

    cells_burn = _cells(zip(meta_df["adi_init"], meta_df["cv2_init"]))
    cells_full = _cells(zip(adi_full_arr, cv2_full_arr))

    # Supportable detector horizons
    support = {}
    for cause, (horizon, need_nz) in DETECTOR_FLOORS.items():
        horizons = [REV2_W_OCC, REV2_W_OCC_FALLBACK, 36, 24] if horizon is None else [horizon]
        support[cause] = {}
        for h in horizons:
            support[cause][f"W={h}"] = round(float(supportable(nz, h, need_nz).mean()), 4)

    # Lead times and prices summary
    lt_arr = np.array(lead_times_raw)
    p_arr = np.array(prices_raw)

    profile: dict[str, Any] = {
        "panel": "raf",
        "dataset_name": "Royal Air Force (RAF) 5000 Spare Parts Demand Dataset",
        "sources": [
            {
                "candidate": "Primary (XLS)",
                "repository": "https://github.com/danieldehaan96/spdf",
                "file_url": "https://raw.githubusercontent.com/danieldehaan96/spdf/main/RAF%20data%20-%207%20years%20demand%20%20-%205000%20items.xls",
                "filename": args.raw_xls.name,
                "file_size_bytes": args.raw_xls.stat().st_size,
                "sha256": sha_xls,
                "license": "No stated license (historical academic distribution)",
                "retrieval_date": "2026-10-24",
            },
            {
                "candidate": "Cross-verification (GluonTS TAR)",
                "repository": "https://github.com/canerturkmen/gluon-ts",
                "file_url": "https://raw.githubusercontent.com/canerturkmen/gluon-ts/intermittent-datasets/datasets/intermittent_raf.tar.gz",
                "filename": args.raw_tar.name,
                "file_size_bytes": args.raw_tar.stat().st_size if args.raw_tar.exists() else None,
                "sha256": sha_tar,
                "license": "Apache-2.0 (GluonTS repo)",
                "retrieval_date": "2026-10-24",
            }
        ],
        "n_skus": int(n_sku),
        "n_periods": int(T),
        "period_range": {
            "start": "1996-01",
            "end": "2002-12",
            "frequency": "1M",
            "start_column": demand_col_names[0],
            "end_column": demand_col_names[-1],
        },
        "burn_in_periods": args.burn_in,
        "evaluated_periods": T - args.burn_in,
        "rev2_assumed_periods": 84,
        "panel_length_shortfall": 84 - T,  # 0! Fits Rev2 §14.1 exactly
        "total_demand_units": float(demand_matrix.sum()),
        "mean_demand_per_sku_period": float(demand_matrix.mean()),
        "zero_fraction": float((demand_matrix == 0).sum() / demand_matrix.size),
        "all_values_non_negative": bool((demand_matrix >= 0).all()),
        "all_values_integer_valued": bool(np.all(np.equal(np.mod(demand_matrix, 1), 0))),
        "any_nan_in_demand": bool(np.isnan(demand_matrix).any()),
        "lead_time_months_summary": {
            "min": float(np.min(lt_arr)),
            "p25": float(np.percentile(lt_arr, 25)),
            "median": float(np.median(lt_arr)),
            "mean": float(np.mean(lt_arr)),
            "p75": float(np.percentile(lt_arr, 75)),
            "max": float(np.max(lt_arr)),
        },
        "price_gbp_summary": {
            "min": float(np.min(p_arr)),
            "p25": float(np.percentile(p_arr, 25)),
            "median": float(np.median(p_arr)),
            "mean": float(np.mean(p_arr)),
            "p75": float(np.percentile(p_arr, 75)),
            "max": float(np.max(p_arr)),
        },
        "nonzero_periods_per_sku": {
            "min": int(np.min(nz.sum(axis=1))),
            "p10": float(np.percentile(nz.sum(axis=1), 10)),
            "p25": float(np.percentile(nz.sum(axis=1), 25)),
            "median": float(np.median(nz.sum(axis=1))),
            "p75": float(np.percentile(nz.sum(axis=1), 75)),
            "p90": float(np.percentile(nz.sum(axis=1), 90)),
            "max": int(np.max(nz.sum(axis=1))),
        },
        "adi_full_series": {
            "min": float(np.nanmin(adi_full_arr)),
            "p25": float(np.nanpercentile(adi_full_arr, 25)),
            "median": float(np.nanmedian(adi_full_arr)),
            "p75": float(np.nanpercentile(adi_full_arr, 75)),
            "max": float(np.nanmax(adi_full_arr)),
        },
        "cv_squared_full_series": {
            "min": float(np.nanmin(cv2_full_arr)),
            "p25": float(np.nanpercentile(cv2_full_arr, 25)),
            "median": float(np.nanmedian(cv2_full_arr)),
            "p75": float(np.nanpercentile(cv2_full_arr, 75)),
            "max": float(np.nanmax(cv2_full_arr)),
            "n_undefined": int(np.isnan(cv2_full_arr).sum()),
        },
        "syntetos_boylan_full_series": {
            k: int(v) for k, v in pd.Series(sb_full_list).value_counts().items()
        },
        "rev2_reachable_2x2_burn_in": cells_burn,
        "rev2_reachable_2x2_full_series": cells_full,
        "dead_skus_in_burn_in": int((meta_df["sb_cell_init"] == "dead").sum()),
        "dead_skus_full_series": int(np.isnan(cv2_full_arr).sum()),
        "censoring_observed": False,
        "cause_labels_available": False,
        "inventory_cost_parameters_available": False,
        "detector_evidence_support_fraction": support,
    }

    with args.out_profile.open("w") as fh:
        json.dump(profile, fh, indent=2)
    print(f"Wrote raf_data_profile.json -> {args.out_profile}")


if __name__ == "__main__":
    main()
