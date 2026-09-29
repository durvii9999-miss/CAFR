"""Arm (c) — Accuracy-based adaptive selection (Rev 2 §12, handoff §8.1).

**What it is:** The key comparator for H1. At each period, evaluate rolling
MASE for each pool member on the trailing window, and switch to the model
with the lowest MASE if it beats the current model by a margin `m`.

**What it is NOT:** It does not use the bandit. It does not constrain churn
(unless explicitly configured to match).

Role: The primary accuracy-based model selection baseline.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ..forecasters.base import FittingInput, ForecastState
from ..sim.inventory import aggregate_order_up_to

__all__ = ["arm_c_factory"]


class _ArmCLevelFn:
    """Stateful level function for arm (c).

    Tracks the out-of-sample performance of ALL pool members.
    """

    def __init__(self, pool, cfg, sku_id, alpha, fit_on, margin=0.0):
        self.pool = pool
        self.cfg = cfg
        self.sku_id = sku_id
        self.alpha = alpha
        self.fit_on = fit_on
        self.margin = margin
        
        self.burn_in = int(cfg["splits"]["burn_in"])
        self.active_method = "croston" # Default start
        self.last_state: ForecastState | None = None
        
        # Track previous point forecasts for all methods: {method: mu_hat_{t-1}}
        self._last_mu = {m: float("nan") for m in pool}
        
        # Track out-of-sample absolute errors: {method: [abs_err_1, ...]}
        self._abs_errors = {m: [] for m in pool}
        
        # Naive MAE from burn-in for MASE denominator
        self._naive_mae = float("nan")

    def __call__(self, t, y_observed, censored, inventory):
        if len(y_observed) == 0:
            self.last_state = None
            return 1.0, self.alpha

        # 1. Update MASE tracking for all methods
        # If we have a new observation, compute the out-of-sample error for the LAST period's forecast
        if len(y_observed) > self.burn_in:
            y_new = float(y_observed[-1])
            for m in self.pool:
                if not np.isnan(self._last_mu[m]):
                    self._abs_errors[m].append(abs(y_new - self._last_mu[m]))
        elif len(y_observed) == self.burn_in:
            # End of burn-in: compute naive MAE
            y_burn = y_observed[:self.burn_in]
            if len(y_burn) > 1:
                diffs = np.abs(y_burn[1:] - y_burn[:-1])
                self._naive_mae = max(float(np.mean(diffs)), 1e-9)
            else:
                self._naive_mae = 1e-9
            
            # Init active method using SB rule as a sensible start (like arm b)
            from .arm_b import SB_MODEL_MAP
            from ..data.loaders.ruf import load_sku_meta
            try:
                # In a real run, we should probably fetch the cell from meta,
                # but to avoid I/O in the loop, we can just use croston or calculate it.
                # Actually, the spec just says "pick the model with lowest MASE". 
                # Before any MASE is available (at t=burn_in), default to croston.
                self.active_method = "croston"
            except Exception:
                pass

        # 2. Select best method based on rolling MASE
        # Evaluate over a trailing window (e.g., 24 periods)
        window = 24
        if len(self._abs_errors["croston"]) > 0:
            current_mae = np.mean(self._abs_errors[self.active_method][-window:])
            best_mae = current_mae
            best_method = self.active_method
            
            for m in self.pool:
                m_mae = np.mean(self._abs_errors[m][-window:])
                # Switch if improvement > margin (margin is absolute MASE improvement, so multiply by naive_mae)
                # m_mase = m_mae / naive_mae
                # m_mae < current_mae - margin * naive_mae
                if m_mae < current_mae - self.margin * self._naive_mae:
                    best_mae = m_mae
                    best_method = m
                    current_mae = best_mae # update to require even better to switch again? No, compared to current
            
            self.active_method = best_method

        # 3. Fit all methods to get the new mu_hat_{t}
        # We MUST fit all methods so we have their mu_hat for the NEXT period's error calc
        fi = FittingInput(y_observed, censored, fit_on=self.fit_on)
        fi.sku_id = self.sku_id
        
        active_state = None
        for m, fc in self.pool.items():
            state = fc.fit(fi)
            self._last_mu[m] = float(state.mu)
            if m == self.active_method:
                active_state = state
                active_state.method = m
                
        self.last_state = active_state

        # 4. Generate Order-Up-To Level using the chosen method's forecast
        R = int(self.cfg["sim"]["R"])
        S, alpha_eff, _, _ = aggregate_order_up_to(
            y_hist=y_observed,
            mu_hat=active_state.mu,
            R=R,
            alpha=self.alpha,
            window=int(self.cfg["sim"].get("quantile_window", 24)),
            min_aggregates=int(self.cfg["sim"].get("min_aggregates", 4)),
            method=self.cfg["sim"].get("quantile_method", "linear"),
        )
        return float(S), float(alpha_eff)


def arm_c_factory(pool, cfg, sku_id, alpha, fit_on, *, margin=0.0):
    """Return a stateful level function for arm (c).
    
    margin: the MASE improvement required to switch models. For gate 3.2, m=0.
    """
    return _ArmCLevelFn(pool, cfg, sku_id, alpha, fit_on, margin=margin)
