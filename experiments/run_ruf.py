"""Real-RUF experiment driver.

Runs the CAFR arms over the real RUF v2 panel (5000 SKUs x 60 periods) through the
rolling-origin harness in ``cafr.sim.loop`` and writes per-SKU metrics to JSON.

Why this exists
---------------
``rollout_sku`` is the project's one true rolling-origin driver, but until now it had
only ever been exercised on a 60-period synthetic dummy series in the gate tests.
Step 3's deliverable is "runs end-to-end on RUF" -- this script is what produces that.

The pooled LightGBM
-------------------
``GlobalLightGBM`` must be trained ONCE across SKUs (``fit_global``) before any
per-series call. Nothing in the pipeline did that, so arm (c) -- the key comparator --
raised ``NotImplementedError`` the moment the real pool was used. The gate tests missed
it because ``tests/test_baselines.py`` substitutes a ``MockLGB``.

``PooledLightGBM`` below adapts the trained model to the pool's ``fit()`` interface.
Training SKUs and evaluation SKUs are DISJOINT; the model is pooled across SKUs, so the
leak boundary is the SKU, not the period.

Usage
-----
    python experiments/run_ruf.py --arms a,b,c --n-skus 300 --seed 42

Output: ``results/ruf_<arms>_n<N>_seed<S>.json`` with a config hash and per-SKU rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_a import arm_a_factory
from cafr.arms.arm_b import arm_b_factory
from cafr.arms.arm_c import arm_c_factory
from cafr.arms.arm_e import arm_e_factory
from cafr.data.loaders.ruf import load_observed
from cafr.forecasters.base import FittingInput, Forecaster
from cafr.forecasters.lightgbm_global import STATIC_COLUMNS, GlobalLightGBM
from cafr.forecasters.registry import build_pool
from cafr.sim.loop import rollout_sku
from cafr.utils.config import REPO_ROOT, config_hash, load_config


class PooledLightGBM(Forecaster):
    """A globally-trained ``GlobalLightGBM`` exposed through the pool interface.

    The pool contract is ``fit(FittingInput) -> ForecastState``; the global model's
    contract is ``fit_series_with_statics(y, statics)``. This bridges the two, reading
    the SKU id the arms attach to the ``FittingInput``.
    """

    name = "lightgbm_global"

    def __init__(self, model: GlobalLightGBM, statics_by_sku: dict[str, dict]) -> None:
        super().__init__()
        self._model = model
        self._statics = statics_by_sku

    def _fit_series(self, y: np.ndarray):  # pragma: no cover - never used
        raise NotImplementedError("PooledLightGBM fits through fit(), not _fit_series()")

    def fit(self, y, fit_on: str | None = None):  # type: ignore[override]
        if isinstance(y, FittingInput):
            series = y.resolve()
            resolved_on = y.fit_on
        else:
            series = np.asarray(y, dtype="float64")
            resolved_on = fit_on or "observed"

        sku = getattr(y, "sku_id", None)
        statics = self._statics.get(sku, {})
        state = self._model.fit_series_with_statics(series, statics, fit_on=resolved_on)
        state.diag["fit_on"] = resolved_on
        self._state = state
        return state


def _factories(sb_cell: str, margin: float = 0.0):
    """Arm name -> level_fn_factory(pool, cfg, sku_id, alpha, fit_on)."""
    return {
        "a": lambda pool, cfg, sku_id, alpha, fit_on: arm_a_factory(
            pool, cfg, sku_id, alpha, fit_on, sb_cell=sb_cell
        ),
        "b": lambda pool, cfg, sku_id, alpha, fit_on: arm_b_factory(
            pool, cfg, sku_id, alpha, fit_on, sb_cell=sb_cell
        ),
        "c": lambda pool, cfg, sku_id, alpha, fit_on: arm_c_factory(
            pool, cfg, sku_id, alpha, fit_on, margin=margin
        ),
        "e": lambda pool, cfg, sku_id, alpha, fit_on: arm_e_factory(
            pool, cfg, sku_id, alpha, fit_on
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ruf.yaml")
    ap.add_argument("--arms", default="a,b,c")
    ap.add_argument("--n-skus", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ #
    # Load the real panel. load_observed() cannot reach demand_true.
    # ------------------------------------------------------------------ #
    obs = load_observed(cfg)
    meta = obs.groupby("sku_id", as_index=True).first()
    series = {
        sid: grp.sort_values("period")["demand_observed"].to_numpy(dtype="float64")
        for sid, grp in obs.groupby("sku_id", sort=False)
    }
    all_ids = np.array(sorted(series))
    n = min(args.n_skus, len(all_ids))
    sampled = rng.choice(all_ids, size=n, replace=False) if n < len(all_ids) else all_ids

    # Disjoint train / test SKUs. The pooled model is trained on the train SKUs only.
    n_train = int(round(n * args.train_frac))
    train_ids = set(sampled[:n_train])
    test_ids = list(sampled[n_train:])

    print(f"panel: {len(all_ids)} SKUs | sampled {n} | train {len(train_ids)} | test {len(test_ids)}")
    print(f"arms={arm_names} alpha={args.alpha}")

    # ------------------------------------------------------------------ #
    # Train the pooled LightGBM ONCE, on the train SKUs.
    # ------------------------------------------------------------------ #
    t0 = time.time()
    train_panel = obs[obs["sku_id"].isin(train_ids)].copy()
    global_model = GlobalLightGBM(
        params=cfg["forecast"].get("lightgbm", {}), seed=int(cfg["seed_root"])
    ).fit_global(train_panel)
    print(f"trained pooled LightGBM on {len(train_ids)} SKUs in {time.time() - t0:.1f}s")

    statics_by_sku = {
        sid: {c: meta.loc[sid, c] for c in STATIC_COLUMNS} for sid in meta.index
    }

    def pool_for(_sku_id: str) -> dict:
        """Fresh classical forecasters per SKU; the trained model is shared read-only."""
        p = build_pool(cfg)
        p["lightgbm_global"] = PooledLightGBM(global_model, statics_by_sku)
        return p

    # ------------------------------------------------------------------ #
    # Run every arm over the same test SKUs.
    # ------------------------------------------------------------------ #
    rows: list[dict] = []
    t_start = time.time()
    for arm in arm_names:
        t_arm = time.time()
        for k, sid in enumerate(test_ids):
            cell = str(meta.loc[sid, "sb_cell_init"])
            res = rollout_sku(
                sku_id=sid,
                y_true=series[sid],
                censored=None,          # RUF: censoring unobserved (H-3)
                arm=arm,
                level_fn_factory=_factories(cell, args.margin)[arm],
                cfg=cfg,
                fit_on=cfg["panels"]["ruf"]["fit_on"],
                alpha=args.alpha,
                pool=pool_for(sid),
            )
            sim = res.sim
            rows.append(
                {
                    "sku_id": sid,
                    "arm": arm,
                    "sb_cell": cell,
                    "mean_cost": float(res.cost_mean),
                    "cycle_csl": float(sim.cycle_service_level),
                    "fill_rate": float(sim.fill_rate),
                    "n_stockout": int(sim.n_stockout),
                    "mean_S": float(np.nanmean(res.s_level)),
                }
            )
            if (k + 1) % 25 == 0:
                el = time.time() - t_arm
                print(f"  [{arm}] {k + 1}/{len(test_ids)}  {el:.1f}s ({el / (k + 1):.3f}s/sku)", flush=True)
        print(f"  [{arm}] done in {time.time() - t_arm:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = args.out or f"results/ruf_{''.join(arm_names)}_n{n}_seed{args.seed}.json"
    out_path = REPO_ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("arm")[["mean_cost", "cycle_csl", "fill_rate", "n_stockout", "mean_S"]]
        .agg(["mean", "std", "count"])
        .round(6)
    )
    payload = {
        "config": args.config,
        "config_hash": config_hash(cfg),
        "seed": args.seed,
        "alpha": args.alpha,
        "margin": args.margin,
        "n_skus": int(n),
        "n_train_skus": len(train_ids),
        "n_test_skus": len(test_ids),
        "arms": arm_names,
        "runtime_seconds": round(time.time() - t_start, 1),
        "summary": json.loads(summary.to_json(orient="index")),
        "rows": df.to_dict("records"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("\n" + summary.to_string())
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
