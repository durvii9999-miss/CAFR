"""Arm (a) — fixed best classical method (Rev 2 §12, handoff §8.1).

**What it is:** For each SKU, choose the forecasting model that achieved the
lowest rolling MASE on the VALIDATION split, then hold that choice fixed for
the entire evaluated window. No adaptation, no learning.

**What it is NOT:** It does not switch models, does not use the bandit, and
does not respond to any signal after initialisation.

Role: classical baseline — the simplest thing above random.

Implementation
--------------
Since validation data is not available until Step 10, this module provides two
modes:

1. ``chosen_method`` passed directly: the model was pre-selected externally.
2. No argument: falls back to Syntetos-Boylan rule selection using the SKU's
   burn-in cell (``sb_cell_init``), which is the closest available substitute
   and is what arm (b) does by default. This fallback is NOTED in T7.

The ``level_fn_factory`` signature matches the ``rollout_sku`` contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..forecasters.base import FittingInput, ForecastState
from ..sim.inventory import aggregate_order_up_to

__all__ = ["arm_a_factory"]

# SB-rule map: burn-in cell -> model name (matches arm b's rule).
_SB_MODEL: dict[str, str] = {
    "moderate_lowdisp": "sba",
    "moderate_highdisp": "tsb",
    "high_lowdisp": "croston",
    "high_highdisp": "tsb",
    "dead": "croston",   # sparse SKU default
}


class _ArmALevelFn:
    """Stateful level function for arm (a).

    Fits the chosen method once per period (rolling), but does NOT switch it.
    Exposes ``last_state`` for the driver's residual recording.
    """

    def __init__(self, pool, cfg, sku_id, alpha, fit_on, chosen_method, sb_cell):
        self.pool = pool
        self.cfg = cfg
        self.sku_id = sku_id
        self.alpha = alpha
        self.fit_on = fit_on
        self.chosen_method = chosen_method
        self.sb_cell = sb_cell
        self.last_state: ForecastState | None = None
        self._residuals: list[float] = []
        self._last_mu: float = float("nan")

    def __call__(self, t, y_observed, censored, inventory):
        if len(y_observed) == 0:
            self.last_state = None
            return 1.0, self.alpha

        method = self.chosen_method
        fc = self.pool[method]
        fi = FittingInput(y_observed, censored, fit_on=self.fit_on)
        state = fc.fit(fi)
        self.last_state = state
        state.method = method

        # Residual for interval estimation (out-of-sample: appended at previous t)
        if not np.isnan(self._last_mu):
            self._residuals.append(float(y_observed[-1]) - self._last_mu)
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


def arm_a_factory(pool, cfg, sku_id, alpha, fit_on, *, chosen_method=None, sb_cell="dead"):
    """Return a stateful level function for arm (a).

    Parameters
    ----------
    chosen_method
        Pre-selected model name (from validation). If None, falls back to the
        Syntetos-Boylan rule using ``sb_cell`` (same as arm b).
    sb_cell
        The burn-in SB cell for fallback model selection.
    """
    if chosen_method is None:
        chosen_method = _SB_MODEL.get(sb_cell, "croston")

    return _ArmALevelFn(pool, cfg, sku_id, alpha, fit_on, chosen_method, sb_cell)
