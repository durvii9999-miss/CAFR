"""Rolling-origin driver (M5). Rev 2 §14.1, handoff §13 Step 3.

This module turns the per-SKU inventory simulator and the forecaster pool into a
single rolling-origin harness. Nothing else in the project loops over periods; every
arm goes through here, so "all arms face identical demand" is true by construction.

Why separate from ``cafr/sim/inventory.py``
-------------------------------------------
``inventory.py`` simulates ONE SKU against ONE level function; it knows nothing about
forecasters, monitors, remedies or arms. That separation lets gate 1.2 test the
simulator with a DGP-aware oracle that is not a forecaster at all.

Three rules enforced structurally
----------------------------------
1. **No look-ahead.** The level function the policy builds receives only
   ``y_observed`` and ``censored`` by the ``inventory.py`` signature; the forecaster
   sees only the history available at decision time ``t``.
2. **Residuals come from the rolling loop.** ``e_t = y_t - mu_hat_t`` where
   ``mu_hat_t`` was made BEFORE ``y_t`` was seen. In-sample residuals understate the
   error and would make C2's coverage test pass on a mis-calibrated interval
   (Rev 2 §17, ``cafr/forecasters/base.py`` module docstring).
3. **One ``fit_on`` per run.** Resolved once from config, logged in every output row.

Done-when (§30.1 Step 3): gates 3.1–3.6 all pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..forecasters.base import FittingInput, ForecastState
from ..forecasters.registry import build_pool
from ..sim.inventory import InventorySimulator, SimConfig, SimResult
from ..utils.config import load_config

__all__ = ["RolloutResult", "rollout_sku", "compute_sku_cost_stats"]


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class RolloutResult:
    """Per-SKU output of the rolling-origin harness."""

    sku_id: str
    arm: str
    fit_on: str

    # Per-period arrays (length = n_evaluated, starting at burn_in)
    periods: np.ndarray               # period indices (int)
    mu_hat: np.ndarray                # one-step-ahead point forecast
    z_hat: np.ndarray                 # occurrence probability
    residuals: np.ndarray             # y_t - mu_hat_t (rolling, out-of-sample)
    active_method: list[str]          # forecaster name used at each period
    s_level: np.ndarray               # order-up-to level S applied at each t
    alpha_t: np.ndarray               # service probability targeted

    # Inventory outcome (comes from SimResult)
    sim: SimResult | None = None

    # Cost stats computed over the evaluated window
    cost_mean: float = float("nan")
    cost_std: float = float("nan")

    # Optional extra columns
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def as_forecast_rows(self) -> pd.DataFrame:
        """Long-format forecast log (§27 FORECASTS_COLUMNS)."""
        n = len(self.periods)
        return pd.DataFrame(
            {
                "sku_id": self.sku_id,
                "period": self.periods,
                "arm": self.arm,
                "method": self.active_method[:n],
                "point": self.mu_hat,
                "q_lo": np.nan,       # filled later by the level function
                "q_hi": np.nan,
                "alpha": self.alpha_t,
                "fit_on": self.fit_on,
            }
        )


# --------------------------------------------------------------------------- #
# Cost statistics (gate 3.6 -- computed before ANY arm is scored)             #
# --------------------------------------------------------------------------- #

def compute_sku_cost_stats(
    sku_id: str,
    y_obs: np.ndarray,
    cfg: dict,
    *,
    burn_in: int | None = None,
    alpha: float = 0.90,
) -> tuple[float, float]:
    """Compute C̄_i and σ_{C,i} for SKU ``sku_id`` on the *observed* series.

    These are estimated during the BURN-IN using a naive (zero-safety-stock)
    level function so the estimator is grounded in the actual cost distribution
    without contaminating the reward from arm selection.

    Returns (cost_mean, cost_std_floored) where the floor is
    ``max(std, sigma_floor_frac * mean)`` (OD-3, Rev 2 §9.3).
    """
    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=cfg["seed_root"])
    burn = burn_in if burn_in is not None else int(cfg["splits"]["burn_in"])

    # Use the burn-in portion only.
    y_burn = y_obs[:burn]
    mu_naive = float(y_burn.mean()) if len(y_burn) else 1.0

    def naive_level(t, y_observed, censored, inventory):
        # Mean-only, no safety stock — deliberately mediocre to spread the cost.
        mu = float(y_observed.mean()) if len(y_observed) > 0 else mu_naive
        return float(cfg["sim"]["R"]) * mu, alpha

    res = sim.run(sku_id, y_burn, naive_level, arm="cost_init")

    costs = res.log["cost_period"].to_numpy()
    cost_mean = float(np.nanmean(costs))
    cost_std = float(np.nanstd(costs, ddof=0))

    # Sigma floor (OD-3).
    floor_frac = float(cfg["forecast"].get("sigma_floor_frac", 0.05))
    cost_std_floored = max(cost_std, floor_frac * max(cost_mean, 1e-9))

    return cost_mean, cost_std_floored


# --------------------------------------------------------------------------- #
# Rolling-origin driver                                                        #
# --------------------------------------------------------------------------- #

def rollout_sku(
    sku_id: str,
    y_true: np.ndarray,
    censored: np.ndarray | None,
    *,
    arm: str,
    level_fn_factory: Callable,
    cfg: dict,
    fit_on: str | None = None,
    alpha: float = 0.90,
) -> RolloutResult:
    """Run the rolling-origin harness for ONE SKU under ONE arm.

    Parameters
    ----------
    sku_id
        SKU identifier (for seeding and logging).
    y_true
        The **true** demand series (length T = burn_in + evaluated).
        The simulator receives this; the policy never does.
    censored
        Boolean mask of censored periods from a previous simulation pass, or
        ``None`` to start with no censoring (e.g., arm (a)/(b)/(c)/(e) on RUF
        where censoring is unobserved — H-3).
    arm
        Arm identifier string (e.g. ``"a"``, ``"c"``).
    level_fn_factory
        Callable that takes the forecast pool dict and the current partial
        history and returns a ``LevelFunction`` (i.e. a callable matching the
        ``inventory.py`` signature ``(t, y_observed, censored, inventory)``.
        For the simple baselines this is a stateful closure.
    cfg
        Full configuration dict (``load_config(...)``).
    fit_on
        Override ``cfg['forecast']['fit_on']``. Normally left as ``None``
        (the config value is used). On RUF ``"observed"`` is forced by H-3.
    alpha
        The target service level.

    Returns
    -------
    RolloutResult
    """
    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #
    burn_in = int(cfg["splits"]["burn_in"])
    evaluated = int(cfg["splits"]["evaluated"])
    T = burn_in + evaluated

    y_true = np.asarray(y_true, dtype="float64")
    if len(y_true) < T:
        # Pad or trim gracefully.
        T = len(y_true)
        evaluated = T - burn_in

    if censored is None:
        censored = np.zeros(T, dtype=bool)
    else:
        censored = np.asarray(censored, dtype=bool)

    _fit_on = fit_on or cfg["forecast"].get("fit_on", "observed")
    pool = build_pool(cfg)

    # Storage for rolling outputs.
    periods_out = np.arange(burn_in, T, dtype=np.int32)
    n_eval = len(periods_out)
    mu_out = np.full(n_eval, np.nan)
    z_out = np.full(n_eval, np.nan)
    resid_out = np.full(n_eval, np.nan)
    s_out = np.full(n_eval, np.nan)
    alpha_out = np.full(n_eval, alpha)
    method_out: list[str] = [""] * n_eval

    # ------------------------------------------------------------------ #
    # The level function wraps the arm's logic                            #
    # ------------------------------------------------------------------ #
    # ``level_fn_factory`` is called once with the pool and cfg; it returns a
    # stateful LevelFunction that updates itself each period.
    level_fn = level_fn_factory(pool, cfg, sku_id, alpha, _fit_on)

    # The inventory simulator owns the inner period loop. We pass it a
    # wrapped level function that also records our rolling forecast.
    _period_mu: list[float] = []
    _period_z: list[float] = []
    _period_method: list[str] = []
    _period_s: list[float] = []
    _period_alpha: list[float] = []

    def _recording_level(t, y_observed, censored_arr, inventory):
        s, a_t = level_fn(t, y_observed, censored_arr, inventory)
        _period_s.append(float(s))
        _period_alpha.append(float(a_t))
        st = level_fn.last_state  # type: ignore[union-attr]
        if st is not None:
            _period_mu.append(float(st.mu))
            _period_z.append(float(st.z))
            _period_method.append(st.method)
        else:
            _period_mu.append(float("nan"))
            _period_z.append(float("nan"))
            _period_method.append("")
        return s, a_t

    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=cfg["seed_root"])
    sim_result = sim.run(sku_id, y_true, _recording_level, arm=arm)

    # ------------------------------------------------------------------ #
    # Slice off the evaluated window                                      #
    # ------------------------------------------------------------------ #
    all_mu = np.array(_period_mu, dtype="float64")
    all_z = np.array(_period_z, dtype="float64")
    all_s = np.array(_period_s, dtype="float64")
    all_alpha = np.array(_period_alpha, dtype="float64")
    all_method = _period_method

    # Rolling residuals: e_t = y_t - mu_hat_t (out-of-sample at each period)
    all_resid = y_true - all_mu  # shape T

    # Slice to the evaluated window only.
    sl = slice(burn_in, T)
    mu_out = all_mu[sl]
    z_out = all_z[sl]
    resid_out = all_resid[sl]
    s_out = all_s[sl]
    alpha_out = all_alpha[sl]
    method_out = all_method[burn_in:T]

    # Cost stats over evaluated window.
    eval_log = sim_result.log.iloc[burn_in:T]
    costs = eval_log["cost_period"].to_numpy()
    cost_mean = float(np.nanmean(costs)) if len(costs) else float("nan")
    cost_std = float(np.nanstd(costs, ddof=0)) if len(costs) else float("nan")

    return RolloutResult(
        sku_id=sku_id,
        arm=arm,
        fit_on=_fit_on,
        periods=periods_out,
        mu_hat=mu_out,
        z_hat=z_out,
        residuals=resid_out,
        active_method=method_out,
        s_level=s_out,
        alpha_t=alpha_out,
        sim=sim_result,
        cost_mean=cost_mean,
        cost_std=cost_std,
    )
