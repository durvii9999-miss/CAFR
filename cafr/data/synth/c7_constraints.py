"""C7's injection, and the pre-flight that §15.2 requires before C7 may be claimed.

Rev 2 §15.2, **constraint 6**: the forecast-acceptability gate (§6.3, C7 condition 2)
must pass on **>= 80 %** of post-injection windows. **Constraint 7**: ``|U_i(t)| >= 20``
and >= 2 unmet cycles on **>= 80 %** of C7 windows at ``W_svc = 30``.

C7 is the only cause whose injection touches no demand process: the simulator's safety
factor is set to 60 % of its correct value and the forecaster's fitting input is
``uncensored`` (§14.3 rule 6). Both halves are needed -- without the second, the
stock-outs the injection causes censor the observable series, the forecaster's error
rises, and condition 2 fails by the injection's own consequence (§15.2).

**So the pre-flight runs the REAL pipeline**: arm (c), through ``rollout_sku``, with
``sim.c7_safety_factor_scale = 0.60``. Evaluating the condition set against a stub
forecaster would answer a different question from the one gate 4.9 asks.

**Gate 4.9 is never relaxed.** If this fails, the escalation is §6.3's C7 condition set
-- revised as a team decision, or C7 reported as not separately identifiable in this
simulator. That is a publishable negative finding; a relaxed gate is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...sim.loop import rollout_sku
from ...utils.config import REPO_ROOT

__all__ = [
    "evaluate_c7_constraints",
    "save_c7_report",
    "cycle_service_level",
    "unmet_cycles",
]

#: §6.3's condition set, as configured in cafr/configs/base.yaml.
W_SVC = 30
MIN_USABLE_PERIODS = 20
MIN_UNMET_CYCLES = 2
Z_B_CRIT = 1.96
TAU_COV = 0.10
DELTA_SVC = 0.10
MIN_NONZERO = 8

#: §15.2's C7 row.
C7_SAFETY_SCALE = 0.60

PASS_RATE = 0.80


def unmet_cycles(demand_lost: np.ndarray, cycle_len: int) -> int:
    """How many CYCLES in the window lost demand.

    A cycle is ``cycle_len`` consecutive periods (the protection interval), exactly as
    in :func:`cycle_service_level`. Counting PERIODS with lost demand instead -- the
    first version of this -- inflates the count by up to a factor of ``R`` = 3 and makes
    the exposure condition look satisfied on windows that never completed a failed
    cycle.
    """
    lost = np.asarray(demand_lost, dtype="float64")
    n_cycles = int(lost.size // cycle_len)
    if n_cycles == 0:
        return 0
    blocks = lost[: n_cycles * cycle_len].reshape(n_cycles, cycle_len)
    return int((blocks.sum(axis=1) > 0.0).sum())


def cycle_service_level(demand_lost: np.ndarray, cycle_len: int) -> tuple[float, int]:
    """Per-CYCLE service level over a lost-demand series.

    A cycle is served iff **no** demand was lost anywhere inside it; cycles are
    non-overlapping blocks of ``cycle_len`` (= the protection interval). This is THE
    CSL -- the per-period fraction ``mean(demand_lost <= 0)`` is a different, always
    larger number and must never be compared against ``alpha`` (see
    ``sim/inventory.py``'s docstring, which measures 0.9198 against a true 0.80 at
    ``alpha = 0.80`` on the §15.1 panel).
    """
    lost = np.asarray(demand_lost, dtype="float64")
    n_cycles = int(lost.size // cycle_len)
    if n_cycles == 0:
        return float("nan"), 0
    blocks = lost[: n_cycles * cycle_len].reshape(n_cycles, cycle_len)
    return float((blocks.sum(axis=1) <= 0.0).mean()), n_cycles


def evaluate_c7_constraints(
    series: dict[str, np.ndarray],
    level_fn_factory,
    cfg: dict,
    *,
    sku_ids: list[str],
    tau_inj: int = 120,
    scale: float = C7_SAFETY_SCALE,
    alpha: float | None = None,
    fit_on: str = "uncensored",
    pool: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run arm (c) with the safety factor mis-set, then test conditions 2-4 per window.

    Returns ``(windows, summary)``. The summary carries the two constraint verdicts and
    the combined gate 4.9 verdict, so the report has a manifest rather than a claim.

    ``pool`` must be a *ready* pool. The pooled LightGBM is trained once across SKUs and
    does not satisfy the pool contract until it has been wrapped (see
    ``cafr.forecasters.pooled``); a bare ``build_pool(cfg)`` raises
    ``NotImplementedError`` on first use, which is how this pre-flight first failed.
    """
    if pool is None:
        raise ValueError(
            "evaluate_c7_constraints needs a ready pool: pass "
            "cafr.forecasters.pooled.fit_global_pool(cfg, panel, ...). A bare "
            "build_pool(cfg) carries an untrained GlobalLightGBM and raises on first "
            "use."
        )
    cfg = dict(cfg)
    cfg["sim"] = dict(cfg["sim"], c7_safety_factor_scale=float(scale))
    sim_cfg = cfg["sim"]
    R = int(sim_cfg["R"])
    alpha = float(alpha if alpha is not None else cfg["sim"]["B_over_H"] / (1.0 + cfg["sim"]["B_over_H"]))

    rows: list[dict] = []
    for sku_id in sku_ids:
        y = np.asarray(series[sku_id], dtype="float64")

        # The pre-flight must simulate the WHOLE series. ``rollout_sku`` truncates to
        # ``splits.burn_in + splits.evaluated`` (= 84 on this config), which is far
        # short of ``tau_inj`` = 120: the run would stop before the injection and the
        # window loop below -- which starts at ``tau_inj + W_SVC`` -- would never
        # execute, reporting an empty gate as if it had passed.
        sku_cfg = dict(cfg)
        sku_cfg["splits"] = dict(
            cfg["splits"],
            burn_in=int(cfg["splits"]["burn_in"]),
            evaluated=int(y.size - int(cfg["splits"]["burn_in"])),
        )

        res = rollout_sku(
            sku_id=sku_id,
            y_true=y,
            censored=None,
            arm="c",
            level_fn_factory=level_fn_factory,
            cfg=sku_cfg,
            fit_on=fit_on,
            alpha=alpha,
            pool=pool,
        )
        log = res.sim.log
        true = (log["demand_met"] + log["demand_lost"]).to_numpy(dtype="float64")
        lost = log["demand_lost"].to_numpy(dtype="float64")
        s_level = log["s_level"].to_numpy(dtype="float64")
        alpha_t = log["alpha_t"].to_numpy(dtype="float64")
        periods = log["period"].to_numpy(dtype="int64")

        # The RESIDUALS e_t = y_t - mu_hat_t, aligned to the same periods. Condition 2 is
        # a test on the FORECAST's error, so both of its statistics must be computed on
        # e -- not on the demand level. Computing z_b on `true` instead tests whether
        # mean(demand) = 0, which is ~sqrt(n)/CV ~ 5 for any positive intermittent
        # series and is rejected almost surely; that version reported a 0.51 pass rate
        # that had nothing to do with forecast acceptability.
        #
        # ``res.residuals`` covers the EVALUATED window only (periods 24..199, 176
        # entries) while the inventory log covers all T (200 rows), so the two are
        # aligned by PERIOD. Zipping them positionally would offset every residual by
        # burn_in and silently test the wrong window.
        resid_by_period = np.full(y.size, np.nan)
        resid_by_period[np.asarray(res.periods, dtype="int64")] = res.residuals

        # The SIZE-subprocess forecast, for the size error e^sz = z - mu_nonzero. §6.3
        # line 255 defines ``z_b = b^sz / (sigma_sz / sqrt(n))`` on the SIZE error, not
        # on the per-period residual: on a nonzero period the two differ by
        # ``mu_z(1-p)``, so the per-period version reports a bias of ~mu_z for every
        # intermittent SKU and the gate rejects almost everything.
        mu_nz_by_period = np.full(y.size, np.nan)
        mu_nz_by_period[np.asarray(res.periods, dtype="int64")] = res.extra["mu_nonzero"]

        # Coverage of the POLICY's own quantile: the R-period aggregate against S.
        # `S = Q_alpha(A^agg)` is a predictive quantile, so its empirical coverage is
        # observable from the simulator's own log -- no forecaster interval needed.
        #
        # Indexed by PERIOD, not by position: the simulator's log starts at
        # ``splits.burn_in``, so position 0 is period 24 and the two differ by a
        # constant that would silently mis-align every window.
        agg_by_period = np.full(y.size, np.nan)
        for t in range(R - 1, y.size):
            agg_by_period[t] = y[t - R + 1 : t + 1].sum()

        # RMSSE baseline: the SKU's OWN pre-injection window, of the SAME length as the
        # windows being tested, so the comparison is like-for-like. (§6.3 condition 2:
        # "RMSSE is below its own W_svc-window baseline".) Taking the whole pre-injection
        # span instead would compare a 30-period window against a 96-period one, and the
        # longer window has the smaller sampling variance on both sides.
        base_mask = (periods >= tau_inj - W_SVC) & (periods < tau_inj)
        rmsse_base = (
            float(np.sqrt(np.nanmean(resid_by_period[base_mask] ** 2)))
            if base_mask.any()
            else np.nan
        )

        last_period = int(periods.max()) if periods.size else -1
        for end in range(tau_inj + W_SVC, last_period + 2):
            sl = (periods >= end - W_SVC) & (periods < end)
            if not sl.any():
                continue
            w_lost = lost[sl]
            w_true = true[sl]
            w_resid = resid_by_period[periods[sl]]
            w_mu_nz = mu_nz_by_period[periods[sl]]
            w_agg = agg_by_period[periods[sl]]
            w_s = s_level[sl]
            w_a = alpha_t[sl]

            rmsse = float(np.sqrt(np.nanmean(w_resid**2)))
            # z_b over the NONZERO periods of the window: n = |U_i(t) INTERSECT N_i|
            # (§6.3 line 255). The zeros carry no size error -- the size sub-process is
            # defined only where demand occurred.
            nz_mask = w_true > 0
            e_sz = w_true[nz_mask] - w_mu_nz[nz_mask]
            e_sz = e_sz[np.isfinite(e_sz)]
            z_b = (
                float(e_sz.mean() / (e_sz.std(ddof=1) / np.sqrt(e_sz.size)))
                if e_sz.size >= MIN_NONZERO and e_sz.std(ddof=1) > 1e-9
                else 0.0
            )
            cov_mask = np.isfinite(w_agg) & np.isfinite(w_s)
            coverage = float(np.mean(w_agg[cov_mask] <= w_s[cov_mask])) if cov_mask.any() else np.nan
            alpha_used = float(np.nanmean(w_a)) if np.isfinite(w_a).any() else alpha
            csl, n_cycles = cycle_service_level(w_lost, R)

            rows.append(
                {
                    "sku_id": sku_id,
                    "window_end": int(end),
                    "scale": float(scale),
                    "n_usable": int(len(w_lost)),
                    "n_unmet_cycles": int(unmet_cycles(w_lost, R)),
                    "n_unmet_periods": int((w_lost > 0).sum()),
                    "unmet_units": float(w_lost.sum()),
                    "n_cycles": int(n_cycles),
                    "z_b": z_b,
                    "coverage": coverage,
                    "alpha_used": alpha_used,
                    "achieved_csl": csl,
                    "target_csl": float(sim_cfg["B_over_H"] / (1.0 + sim_cfg["B_over_H"])),
                    "rmsse": rmsse,
                    "rmsse_base": rmsse_base,
                    "rmsse_ok": bool(np.isfinite(rmsse_base) and rmsse <= rmsse_base),
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        return windows, pd.DataFrame()

    windows["exposure_ok"] = (windows["n_usable"] >= MIN_USABLE_PERIODS) & (
        windows["n_unmet_cycles"] >= MIN_UNMET_CYCLES
    )
    windows["acceptability_ok"] = (
        windows["rmsse_ok"]
        & (windows["z_b"].abs() <= Z_B_CRIT)
        & ((windows["coverage"] - windows["alpha_used"]).abs() <= TAU_COV)
    )
    windows["service_gap_ok"] = windows["achieved_csl"] < (
        windows["target_csl"] - DELTA_SVC
    )
    windows["c7_fireable"] = (
        windows["acceptability_ok"] & windows["exposure_ok"] & windows["service_gap_ok"]
    )

    summary = pd.DataFrame(
        [
            {
                "scale": float(scale),
                "fit_on": fit_on,
                "n_windows": int(len(windows)),
                "n_skus": int(windows["sku_id"].nunique()),
                "acceptability_pass_rate": float(windows["acceptability_ok"].mean()),
                "exposure_pass_rate": float(windows["exposure_ok"].mean()),
                "service_gap_rate": float(windows["service_gap_ok"].mean()),
                "c7_fireable_rate": float(windows["c7_fireable"].mean()),
                "mean_achieved_csl": float(windows["achieved_csl"].mean()),
                "mean_coverage": float(windows["coverage"].mean()),
                "constraint_6_pass": bool(
                    windows["acceptability_ok"].mean() >= PASS_RATE
                ),
                "constraint_7_pass": bool(windows["exposure_ok"].mean() >= PASS_RATE),
                "gate_4_9_pass": bool(
                    windows["acceptability_ok"].mean() >= PASS_RATE
                    and windows["exposure_ok"].mean() >= PASS_RATE
                ),
            }
        ]
    )
    return windows, summary


def save_c7_report(
    windows: pd.DataFrame, summary: pd.DataFrame, out_dir: Path | None = None
) -> Path:
    """Write the C7 pre-flight evidence so gates 4.9/4.10 have a manifest."""
    out_dir = out_dir or (REPO_ROOT / "results" / "synth")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "c7_windows.parquet", index=False)
    summary.to_parquet(out_dir / "c7_constraints.parquet", index=False)
    path = out_dir / "c7_constraints.json"
    path.write_text(json.dumps(summary.to_dict("records"), indent=2), encoding="utf-8")
    return path
