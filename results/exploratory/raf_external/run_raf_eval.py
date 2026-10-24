"""RAF Spare-Parts External Evaluation: Pure Forecasting Protocol.

Runs rolling-origin 1-step-ahead forecasting over the historical RAF spare-parts panel
(periods 24..83, 60 evaluated months) comparing classical intermittent-demand methods
and pooled Global LightGBM.

Features:
- Pure out-of-sample evaluation: no data leakage across time or SKUs.
- Disjoint SKU split: GlobalLightGBM is trained only on the training SKUs.
- Predeclared metrics: sMAE (MASE), RMSSE, WAPE, MAE, Mean Bias, PIS.
- 95% bootstrap confidence intervals for aggregate metrics.
- Breakdown by Syntetos-Boylan category (intermittent vs lumpy) and burn-in status.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from cafr.forecasters.croston import Croston
from cafr.forecasters.lightgbm_global import STATIC_COLUMNS, GlobalLightGBM
from cafr.forecasters.sba import SBA
from cafr.forecasters.ses import SESOnSizes
from cafr.forecasters.tsb import TSB


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval of the mean."""
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    n = len(clean)
    for b in range(n_boot):
        sample = rng.choice(clean, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100))
    return lower, upper


def main() -> None:
    ap = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    ap.add_argument("--demand-path", type=Path, default=base_dir / "data" / "demand_panel.parquet")
    ap.add_argument("--meta-path", type=Path, default=base_dir / "data" / "sku_meta.parquet")
    ap.add_argument("--attr-path", type=Path, default=base_dir / "data" / "sku_attributes.parquet")
    ap.add_argument("--n-skus", type=int, default=1000, help="Total SKUs sampled (train + test)")
    ap.add_argument("--train-frac", type=float, default=0.6, help="Fraction of SKUs for training pooled ML")
    ap.add_argument("--burn-in", type=int, default=24, help="Burn-in periods (Rev2 §14.1 = 24)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=base_dir / "raf_forecasting_results.json")
    ap.add_argument("--out-parquet", type=Path, default=base_dir / "raf_series_metrics.parquet")
    args = ap.parse_args()

    print(f"=== Starting RAF External Forecasting Benchmark ===")
    print(f"Dataset: {args.demand_path}")
    print(f"Config: n_skus={args.n_skus}, train_frac={args.train_frac}, burn_in={args.burn_in}, seed={args.seed}")

    rng = np.random.default_rng(args.seed)

    # 1. Load Data
    demand_df = pd.read_parquet(args.demand_path)
    meta_df = pd.read_parquet(args.meta_path)
    attr_df = pd.read_parquet(args.attr_path)

    all_sku_ids = meta_df["sku_id"].to_numpy()
    total_available = len(all_sku_ids)
    n_sample = min(args.n_skus, total_available)

    if n_sample < total_available:
        sampled_skus = rng.choice(all_sku_ids, size=n_sample, replace=False)
    else:
        sampled_skus = all_sku_ids

    n_train = int(round(n_sample * args.train_frac))
    train_skus = set(sampled_skus[:n_train])
    test_skus = list(sampled_skus[n_train:])

    print(f"Total panel SKUs: {total_available} | Sampled: {n_sample} | Train: {len(train_skus)} | Test: {len(test_skus)}")

    # 2. Train Pooled LightGBM on Train SKUs Only
    print("Training pooled GlobalLightGBM on train SKUs...")
    t0 = time.time()
    train_panel = demand_df[demand_df["sku_id"].isin(train_skus)].merge(meta_df, on="sku_id")
    global_lgb = GlobalLightGBM(seed=args.seed).fit_global(train_panel)
    print(f"GlobalLightGBM trained in {time.time() - t0:.2f}s")

    # 3. Prepare Test Data
    meta_by_sku = meta_df.set_index("sku_id")
    attr_by_sku = attr_df.set_index("sku_id")
    test_panel = demand_df[demand_df["sku_id"].isin(set(test_skus))]
    series_dict = {
        sid: grp.sort_values("period")["demand_observed"].to_numpy(dtype=np.float64)
        for sid, grp in test_panel.groupby("sku_id")
    }
    statics_by_sku = {
        sid: {c: meta_by_sku.loc[sid, c] for c in STATIC_COLUMNS} for sid in test_skus
    }

    # 4. Instantiate Forecasters
    forecasters = {
        "zero": None,
        "croston": Croston(alpha=0.1),
        "sba": SBA(alpha=0.1),
        "tsb": TSB(alpha_d=0.2, alpha_p=0.2),
        "ses_sizes": SESOnSizes(alpha=0.1),
        "lightgbm_global": global_lgb,
    }

    methods = list(forecasters.keys())
    print(f"Candidate methods: {methods}")

    # 5. Run Rolling-Origin Forecast Evaluation
    # Evaluation spans period t = burn_in .. T-1 (e.g. 24 .. 83 -> 60 periods)
    T_total = 84
    eval_periods = np.arange(args.burn_in, T_total)
    n_eval = len(eval_periods)
    print(f"Evaluation window: periods {args.burn_in}..{T_total-1} ({n_eval} months)")

    sku_results = []
    t_start_eval = time.time()

    for i_sku, sid in enumerate(test_skus):
        y_full = series_dict[sid]
        y_burn = y_full[:args.burn_in]
        y_eval = y_full[args.burn_in:]
        sum_y_eval = float(np.sum(y_eval))

        # In-sample scale for sMAE/RMSSE (mean demand of burn-in)
        mean_burn = float(np.mean(y_burn))
        if mean_burn > 0:
            d_scale = mean_burn
        else:
            # If all zeros in burn-in, use small epsilon or full series mean
            mean_full = float(np.mean(y_full))
            d_scale = mean_full if mean_full > 0 else 1.0

        sb_class = str(attr_by_sku.loc[sid, "syntetos_boylan"])
        sb_init = str(meta_by_sku.loc[sid, "sb_cell_init"])
        is_dead_burn = (sb_init == "dead")

        sku_statics = statics_by_sku[sid]

        for method in methods:
            preds = np.empty(n_eval, dtype=np.float64)

            if method == "zero":
                preds.fill(0.0)
            elif method == "lightgbm_global":
                for idx, t in enumerate(eval_periods):
                    y_hist = y_full[:t]
                    fstate = global_lgb.fit_series_with_statics(y_hist, sku_statics)
                    preds[idx] = max(float(fstate.mu), 0.0)
            else:
                fmodel = forecasters[method]
                for idx, t in enumerate(eval_periods):
                    y_hist = y_full[:t]
                    fstate = fmodel.fit(y_hist)
                    preds[idx] = max(float(fstate.mu), 0.0)

            errors = y_eval - preds
            abs_errors = np.abs(errors)
            sq_errors = errors ** 2

            mae = float(np.mean(abs_errors))
            smae = float(mae / d_scale)
            rmse = float(np.sqrt(np.mean(sq_errors)))
            rmsse = float(rmse / d_scale)
            wape = float(np.sum(abs_errors) / (sum_y_eval + 1e-8))
            bias = float(np.mean(preds - y_eval))
            # Cumulative tracking error (Period In Stock proxy)
            cum_errors = np.cumsum(preds - y_eval)
            pis = float(np.sum(cum_errors))

            sku_results.append({
                "sku_id": sid,
                "method": method,
                "sb_class": sb_class,
                "sb_init": sb_init,
                "is_dead_burn": is_dead_burn,
                "mae": mae,
                "smae": smae,
                "rmse": rmse,
                "rmsse": rmsse,
                "wape": wape,
                "bias": bias,
                "pis": pis,
                "sum_actual_demand": sum_y_eval,
                "sum_predicted_demand": float(np.sum(preds)),
            })

        if (i_sku + 1) % 100 == 0 or (i_sku + 1) == len(test_skus):
            elapsed = time.time() - t_start_eval
            print(f"Evaluated {i_sku + 1}/{len(test_skus)} test SKUs ({elapsed:.1f}s)")

    # 6. Aggregate Results
    df_res = pd.DataFrame(sku_results)
    df_res.to_parquet(args.out_parquet, index=False)
    print(f"Wrote series-level metrics -> {args.out_parquet}")

    metric_cols = ["smae", "rmsse", "wape", "mae", "bias", "pis"]
    summary_by_method = {}

    for method in methods:
        sub = df_res[df_res["method"] == method]
        method_stats: dict[str, Any] = {"n_series": len(sub)}

        for m in metric_cols:
            arr = sub[m].to_numpy(dtype=float)
            # Remove infinite values if any
            valid_arr = arr[np.isfinite(arr)]
            mean_val = float(np.mean(valid_arr))
            median_val = float(np.median(valid_arr))
            std_val = float(np.std(valid_arr, ddof=1)) if len(valid_arr) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci(valid_arr, n_boot=1000, seed=args.seed)

            method_stats[m] = {
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "std": round(std_val, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            }

        # Breakdown by SB class (intermittent vs lumpy)
        by_sb = {}
        for sb_grp, grp in sub.groupby("sb_class"):
            by_sb[sb_grp] = {
                "n_series": len(grp),
                "smae_mean": round(float(np.mean(grp["smae"])), 4),
                "smae_median": round(float(np.median(grp["smae"])), 4),
                "rmsse_mean": round(float(np.mean(grp["rmsse"])), 4),
                "wape_mean": round(float(np.mean(grp["wape"])), 4),
            }
        method_stats["by_syntetos_boylan"] = by_sb

        # Breakdown by dead in burn-in
        by_dead = {}
        for dead_flag, grp in sub.groupby("is_dead_burn"):
            label = "dead_in_burn_in" if dead_flag else "active_in_burn_in"
            by_dead[label] = {
                "n_series": len(grp),
                "smae_mean": round(float(np.mean(grp["smae"])), 4),
                "smae_median": round(float(np.median(grp["smae"])), 4),
                "rmsse_mean": round(float(np.mean(grp["rmsse"])), 4),
                "wape_mean": round(float(np.mean(grp["wape"])), 4),
            }
        method_stats["by_burn_in_status"] = by_dead

        summary_by_method[method] = method_stats

    # Pairwise statistical comparisons against SBA (the standard intermittent benchmark)
    sba_smae = df_res[df_res["method"] == "sba"].sort_values("sku_id")["smae"].to_numpy()
    pairwise = {}
    for other in [m for m in methods if m != "sba"]:
        other_smae = df_res[df_res["method"] == other].sort_values("sku_id")["smae"].to_numpy()
        diff = other_smae - sba_smae
        # Wilcoxon signed rank test
        try:
            stat, p_val = stats.wilcoxon(other_smae, sba_smae)
            p_val_float = float(p_val)
        except Exception:
            p_val_float = 1.0

        pairwise[f"{other}_vs_sba"] = {
            "mean_smae_diff": round(float(np.mean(diff)), 4),
            "median_smae_diff": round(float(np.median(diff)), 4),
            "p_value": p_val_float,
            "statistically_significant_005": bool(p_val_float < 0.05),
        }

    output_payload = {
        "evaluation_name": "RAF Spare-Parts External Forecasting Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "results/exploratory/raf_external/data/demand_panel.parquet",
        "sample_size": {
            "total_panel_skus": total_available,
            "sampled_skus": n_sample,
            "train_skus": len(train_skus),
            "test_skus": len(test_skus),
        },
        "evaluation_protocol": {
            "burn_in_periods": args.burn_in,
            "evaluation_periods": n_eval,
            "evaluation_range": f"{eval_periods[0]}..{eval_periods[-1]}",
            "step_ahead": 1,
            "seed": args.seed,
        },
        "metrics_summary": summary_by_method,
        "pairwise_tests_vs_sba": pairwise,
    }

    with args.out_json.open("w") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"Wrote summary results -> {args.out_json}")
    print("=== Evaluation Finished Successfully ===")


if __name__ == "__main__":
    main()
