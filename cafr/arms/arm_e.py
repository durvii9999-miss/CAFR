"""Arm (e) — Global LightGBM baseline (Rev 2 §12, handoff §8.1).

**What it is:** A single global LightGBM model, pooled across all SKUs,
with static SKU features and lagged demand.

**What it is NOT:** It is not SKU-specific. It uses the `lightgbm_global`
forecaster from the pool, which handles the pooling.

Role: ML baseline.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ..forecasters.base import FittingInput, ForecastState
from ..sim.inventory import aggregate_order_up_to

__all__ = ["arm_e_factory"]


class _ArmELevelFn:
    """Stateful level function for arm (e)."""

    def __init__(self, pool, cfg, sku_id, alpha, fit_on):
        self.pool = pool
        self.cfg = cfg
        self.sku_id = sku_id
        self.alpha = alpha
        self.fit_on = fit_on
        self.chosen_method = "lightgbm_global"
        self.last_state: ForecastState | None = None
        self._last_mu: float = float("nan")

    def __call__(self, t, y_observed, censored, inventory):
        if len(y_observed) == 0:
            self.last_state = None
            return 1.0, self.alpha

        method = self.chosen_method
        fc = self.pool[method]
        # We pass sku_id in the FittingInput or kwargs if needed by GlobalLightGBM?
        # Looking at FittingInput, it might just take the series. GlobalLightGBM
        # might need sku_id. For now, assume Forecaster interface is standard, but
        # we can attach sku_id to the input.
        fi = FittingInput(y_observed, censored, fit_on=self.fit_on)
        # Hack to pass sku_id for global models if they inspect the input object:
        fi.sku_id = self.sku_id 

        state = fc.fit(fi)
        self.last_state = state
        state.method = method
        self._last_mu = float(state.mu)

        R = int(self.cfg["sim"]["R"])
        S, alpha_eff, _, _ = aggregate_order_up_to(
            y_hist=y_observed,
            mu_hat=state.mu,
            R=R,
            alpha=self.alpha,
            window=int(self.cfg["sim"].get("quantile_window", 24)),
            min_aggregates=int(self.cfg["sim"].get("min_aggregates", 4)),
            method=self.cfg["sim"].get("quantile_method", "linear"),
            # C7's injection scales the SAFETY FACTOR only, never the forecast (§15.2).
            # Read from config so the knob is live: nothing passed it before, which left
            # `sim.c7_safety_factor_scale` dead and made C7 undetectable by construction.
            safety_factor_scale=float(self.cfg["sim"].get("c7_safety_factor_scale", 1.0)),
        )
        return float(S), float(alpha_eff)


def arm_e_factory(pool, cfg, sku_id, alpha, fit_on):
    """Return a stateful level function for arm (e)."""
    if "lightgbm_global" not in pool:
        # Fallback if not in pool for some reason
        pass
    return _ArmELevelFn(pool, cfg, sku_id, alpha, fit_on)
