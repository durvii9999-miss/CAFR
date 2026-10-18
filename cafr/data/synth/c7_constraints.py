"""C7's injection, and the pre-flight that §15.2 requires before C7 may be claimed.

Rev 2 §15.2, **constraint 6**: the forecast-acceptability gate (§6.3, C7 condition 2)
must pass on **>= 80 %** of post-injection windows. **Constraint 7**: ``|U_i(t)| >= 20``
and >= 2 unmet cycles on **>= 80 %** of C7 windows at ``W_svc = 30``.

C7 is the only cause whose injection touches no demand process: the simulator's safety
factor is set to a fraction of its correct value and the forecaster's fitting input is
``uncensored`` (§14.3 rule 6). Both halves are needed -- without the second, the
stock-outs the injection causes censor the observable series, the forecaster's error
rises, and condition 2 fails by the injection's own consequence (§15.2).

**So the pre-flight runs the REAL pipeline**: arm (c), through ``rollout_sku``, with
``sim.c7_safety_factor_scale`` set from the pre-registered calibration grid. Evaluating
the condition set against a stub forecaster would answer a different question from the
one gate 4.9 asks.

**Gate 4.9 is never relaxed.** If this fails, the escalation is §6.3's C7 condition set
-- revised as a team decision, or C7 reported as not separately identifiable in this
simulator. That is a publishable negative finding; a relaxed gate is not.

Restatements signed off 2026-10-17 (docs/16 §7)
-----------------------------------------------
Two of constraint 6's three clauses were unusable **as written**, not because they
failed, but because the object they named could not support an 80 % bar. They are
restated here, before the run that reports them:

1. **RMSSE** -- "the error did not rise" is tested on the **pooled per-SKU** RMS of the
   rolling residuals, pre- vs post-injection (``[burn_in, tau_inj)`` vs
   ``[tau_inj, T)``), with a pre-registered 10 % tolerance band:
   ``rmsse_post <= (1 + RMSSE_TOL) * rmsse_pre``. A per-window comparison under the null
   has P(pass) = 0.5 by construction (docs/16 §2a), so no implementation could clear it.
   Pooling makes the quantity an estimate of a *ratio*, not a coin flip. The per-window
   ratio is still computed and reported, for transparency, but does not set the verdict.
2. **Coverage** -- measured against the **forecaster's predictive quantile**, i.e. the
   order-up-to level BEFORE C7's safety-factor scaling:
   ``S_forecast = R * mu_hat + (S - R * mu_hat) / scale``. Counting coverage of the
   policy's scaled ``S`` would make C7 indistinguishable from C2 (docs/16 §2b). At a
   scale of exactly 0 that reconstruction is undefined, which is why the pre-registered
   grid bottoms out at 0.05 rather than 0.0.

Constraint 7 is a magnitude problem, not a definitional one: the scale is calibrated
over a pre-declared grid, mildest first, and the first scale whose exposure pass rate
reaches 80 % is used. If no scale does, the gate fails and the strongest scale is
reported -- the gate is not relaxed.
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
    "calibrate_c7_scale",
    "save_c7_report",
    "cycle_service_level",
    "unmet_cycles",
    "unscaled_forecast_level",
    "pooled_rmsse",
    "forward_aggregates",
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

#: Pre-registered restatements of constraint 6 (2026-10-17, docs/16 §7).
#: Pooled RMSSE may exceed its pre-injection baseline by at most this fraction.
RMSSE_TOL = 0.10

#: Pre-registered calibration grid for constraint 7, mildest (closest to 1.0) first.
#: The floor is 0.05, not 0.0, because the unscaled forecast quantile has to be
#: recoverable from the logged level by dividing out the scale (module docstring).
C7_SCALE_GRID: tuple[float, ...] = (0.60, 0.45, 0.30, 0.15, 0.05)


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


def unscaled_forecast_level(
    s_level: np.ndarray, mu_hat: np.ndarray, R: int, scale: float
) -> np.ndarray:
    """Recover the forecaster's UNSCALED predictive quantile from the policy's ``S``.

    The policy computes ``S = R * mu_hat + scale * Q_alpha(A^e)``. Constraint 6 asks
    whether the **forecast's** interval is acceptable, so the object to score is
    ``R * mu_hat + Q_alpha`` -- the interval as reported to the policy *before* C7's
    safety-factor injection (docs/16 §7, decision 2). Scoring ``S`` itself makes C7
    indistinguishable from C2, which is the identifiability failure the constraint
    exists to detect.

    ``scale`` must be strictly positive; at zero the safety term carries no information
    and the reconstruction is undefined.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")
    s = np.asarray(s_level, dtype="float64")
    mu = np.asarray(mu_hat, dtype="float64")
    return R * mu + (s - R * mu) / float(scale)


def pooled_rmsse(
    residuals_by_period: np.ndarray, periods: np.ndarray, tau_inj: int
) -> tuple[float, float]:
    """Pre- and post-injection RMS of the rolling one-step residuals, pooled per SKU.

    The per-window "post <= baseline" comparison is a coin flip under the null
    (docs/16 §2a): no implementation of "the error did not rise" can pass it 80 % of
    the time. Pooling over the whole pre- and post-injection spans gives one ratio
    estimate per SKU, whose sampling variance is small enough for an 80 % bar on a
    10 % tolerance band.
    """
    resid = np.asarray(residuals_by_period, dtype="float64")
    per = np.asarray(periods, dtype="int64")

    def _rms(mask: np.ndarray) -> float:
        vals = resid[per[mask]]
        vals = vals[np.isfinite(vals)]
        return float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan")

    return _rms(per < tau_inj), _rms(per >= tau_inj)


def forward_aggregates(y: np.ndarray, R: int) -> np.ndarray:
    """R-period demand sums starting at ``t+1``, indexed by decision period ``t``.

    ``S_t`` is the inventory-position target set at the end of period ``t``; it covers
    the protection interval of the order placed then, ``[t+1, t+R]``. Coverage of a
    PREDICTIVE quantile must be measured against that forward window. A trailing
    aggregate was correct only while ``S`` was ``Q_alpha(aggregates)`` (pre-docs/17);
    after the residual-bootstrap fix it scores the quantile against the wrong
    realisation (docs/16 §3.7).

    The last ``R`` decision periods have no complete forward window and are NaN. This is
    an evaluation-only quantity; the policy never reads it.
    """
    y = np.asarray(y, dtype="float64")
    out = np.full(y.size, np.nan)
    for t in range(y.size - R):
        out[t] = y[t + 1 : t + R + 1].sum()
    return out


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
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"scale must be > 0, got {scale!r}. Coverage is measured against the "
            "forecaster's UNSCALED quantile, recovered as "
            "S_forecast = R*mu_hat + (S - R*mu_hat)/scale; at scale 0 the reconstruction "
            "is undefined (and the policy carries no information about the forecast)."
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

        # The point forecast, needed to recover the forecaster's UNSCALED predictive
        # quantile from the policy's logged level (docs/16 §7, decision 2).
        mu_by_period = np.full(y.size, np.nan)
        mu_by_period[np.asarray(res.periods, dtype="int64")] = res.mu_hat

        # Coverage of the FORECASTER's quantile, not of the policy's scaled level.
        # `S` is `R*mu_hat + scale * Q_alpha(A^e)`, so `Q_alpha` is recovered by
        # dividing the safety term out. Counting coverage of the scaled `S` would make
        # C7 indistinguishable from C2 -- which is exactly the identifiability problem
        # constraint 6 exists to detect (docs/16 §2b).
        #
        # The aggregate is the R-period demand the level at the END of period t is set
        # to cover: `[t+1, t+R]`, the protection interval of the order placed at t. This
        # is the object `S_t = R*mu_hat_t + safety` predicts. The pre-docs/17 code used
        # a TRAILING aggregate -- correct only while `S` was `Q_alpha(aggregates)` and
        # therefore a statement about the same data; after docs/17's residual bootstrap
        # it compares a predictive quantile against the wrong realisation. Evaluation
        # may read the future here: the policy never does.
        #
        # Indexed by PERIOD, not by position: the simulator's log starts at
        # ``splits.burn_in``, so position 0 is period 24 and the two differ by a
        # constant that would silently mis-align every window.
        agg_by_period = forward_aggregates(y, R)

        # RMSSE: a PER-WINDOW baseline is still recorded for transparency, but the
        # verdict uses the POOLED ratio (docs/16 §7, decision 1). Under the null a
        # per-window "post <= baseline" comparison passes half the time by construction,
        # so no implementation of "the error did not rise" can clear an 80 % bar that
        # way. Pooling over the whole pre- and post-injection spans turns the quantity
        # into one ratio estimate per SKU, and the pre-registered 10 % band says how
        # much sampling noise is permitted.
        base_mask = (periods >= tau_inj - W_SVC) & (periods < tau_inj)
        rmsse_base = (
            float(np.sqrt(np.nanmean(resid_by_period[base_mask] ** 2)))
            if base_mask.any()
            else np.nan
        )

        rmsse_pooled_base, rmsse_pooled_post = pooled_rmsse(
            resid_by_period, periods, tau_inj
        )
        rmsse_pooled_ok = bool(
            np.isfinite(rmsse_pooled_base)
            and np.isfinite(rmsse_pooled_post)
            and rmsse_pooled_post <= (1.0 + RMSSE_TOL) * rmsse_pooled_base
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
            w_mu = mu_by_period[periods[sl]]
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
            # The forecaster's UNSCALED predictive quantile:
            #   S = R*mu_hat + scale * Q_alpha  =>  S_forecast = R*mu_hat + Q_alpha.
            # On a fallback period S == R*mu_hat and the reconstruction returns the same
            # value, so the fallback is handled without a special case. `scale > 0` is
            # enforced at entry.
            s_forecast = unscaled_forecast_level(w_s, w_mu, R, scale)
            cov_mask = np.isfinite(w_agg) & np.isfinite(s_forecast)
            coverage = (
                float(np.mean(w_agg[cov_mask] <= s_forecast[cov_mask]))
                if cov_mask.any()
                else np.nan
            )
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
                    "rmsse_window_ok": bool(
                        np.isfinite(rmsse_base) and rmsse <= rmsse_base
                    ),
                    "rmsse_pooled": rmsse_pooled_post,
                    "rmsse_pooled_base": rmsse_pooled_base,
                    "rmsse_ok": rmsse_pooled_ok,
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
                # Clause-level rates, so a failure can be attributed to a clause rather
                # than to "constraint 6" as a whole (docs/16 §2 was a breakdown).
                "rmsse_pooled_pass_rate": float(windows["rmsse_ok"].mean()),
                "rmsse_window_pass_rate": float(windows["rmsse_window_ok"].mean()),
                "z_b_pass_rate": float(
                    (windows["z_b"].abs() <= Z_B_CRIT).mean()
                ),
                "coverage_pass_rate": float(
                    ((windows["coverage"] - windows["alpha_used"]).abs() <= TAU_COV).mean()
                ),
                "rmsse_tol": float(RMSSE_TOL),
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


def calibrate_c7_scale(
    series: dict[str, np.ndarray],
    level_fn_factory,
    cfg: dict,
    *,
    sku_ids: list[str],
    tau_inj: int = 120,
    grid: tuple[float, ...] = C7_SCALE_GRID,
    alpha: float | None = None,
    fit_on: str = "uncensored",
    pool: dict | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Pick C7's safety-factor scale by the PRE-REGISTERED grid (docs/16 §7).

    The grid is walked **mildest first** (largest scale, closest to the correct 1.0),
    and the first scale whose exposure pass rate reaches ``PASS_RATE`` is used. This is
    §15.2's stop-rule option (i) -- retune the injection table -- done by a rule fixed
    before the run rather than by looking at the result and choosing a scale.

    Selection reads **constraint 7 only**. Constraint 6 is invariant to the scale under
    ``observation_model = "demand"``: the forecaster sees demand, not sales, so neither
    its residuals nor its predictive quantile move when the policy's safety factor
    does. Selecting on constraint 6 would therefore be selecting on noise.

    Returns ``(chosen_scale, windows_at_chosen_scale, ladder)``. If no scale in the grid
    reaches 80 % exposure the strongest scale is returned, marked
    ``grid_exhausted_no_pass``, and gate 4.9 fails honestly -- the gate is not relaxed.
    """
    ordered = tuple(sorted((float(g) for g in grid), reverse=True))
    if not ordered:
        raise ValueError("calibrate_c7_scale needs a non-empty grid")
    if any(not np.isfinite(g) or g <= 0.0 for g in ordered):
        raise ValueError(
            f"every scale must be finite and > 0, got {ordered}. Coverage is measured "
            "against the unscaled forecast quantile, which is recovered by dividing the "
            "safety term by the scale (docs/16 §7)."
        )

    ladder_rows: list[dict] = []
    chosen_scale: float | None = None
    chosen_windows = pd.DataFrame()

    for scale in ordered:
        windows, summary = evaluate_c7_constraints(
            series,
            level_fn_factory,
            cfg,
            sku_ids=sku_ids,
            tau_inj=tau_inj,
            scale=scale,
            alpha=alpha,
            fit_on=fit_on,
            pool=pool,
        )
        row = summary.iloc[0].to_dict() if not summary.empty else {
            "scale": float(scale),
            "n_windows": 0,
            "n_skus": 0,
            "acceptability_pass_rate": float("nan"),
            "exposure_pass_rate": float("nan"),
            "constraint_6_pass": False,
            "constraint_7_pass": False,
            "gate_4_9_pass": False,
        }
        row["scale_position"] = len(ladder_rows) + 1
        row["grid_size"] = len(ordered)
        if chosen_scale is None and bool(row.get("constraint_7_pass", False)):
            chosen_scale = float(scale)
            chosen_windows = windows
            row["selected"] = True
            row["selection_rule"] = "mildest_passing_scale"
            ladder_rows.append(row)
            break
        row["selected"] = False
        row["selection_rule"] = "not_selected"
        ladder_rows.append(row)

    if chosen_scale is None:
        # The whole grid was walked and nothing reached 80 %. Report the strongest
        # scale tried; its own summary still says constraint 7 FAILS.
        chosen_scale = float(ordered[-1])
        chosen_windows = windows
        ladder_rows[-1]["selected"] = True
        ladder_rows[-1]["selection_rule"] = "grid_exhausted_no_pass"

    return chosen_scale, chosen_windows, pd.DataFrame(ladder_rows)


def save_c7_report(
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    ladder: pd.DataFrame | None = None,
) -> Path:
    """Write the C7 pre-flight evidence so gates 4.9/4.10 have a manifest."""
    out_dir = out_dir or (REPO_ROOT / "results" / "synth")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "c7_windows.parquet", index=False)
    summary.to_parquet(out_dir / "c7_constraints.parquet", index=False)
    if ladder is not None:
        ladder.to_parquet(out_dir / "c7_scale_ladder.parquet", index=False)
    path = out_dir / "c7_constraints.json"
    path.write_text(json.dumps(summary.to_dict("records"), indent=2), encoding="utf-8")
    return path
