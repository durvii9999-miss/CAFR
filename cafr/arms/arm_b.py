"""Arm (b) — Syntetos-Boylan rule-based model selection (Rev 2 §12, handoff §8.1).

**What it is:** Select the forecasting model using the Syntetos-Boylan
ADI × CV² classification, fixed after burn-in initialisation. The cell is
determined once from ``sb_cell_init`` and never revisited.

**Rule:**
- ``high_lowdisp``    (ADI ≥ ADI*, CV² < CV²*)  → Croston
- ``moderate_lowdisp`` (ADI < ADI*, CV² < CV²*)  → SBA  (SBA corrects Croston's bias)
- ``high_highdisp``   (ADI ≥ ADI*, CV² ≥ CV²*)  → TSB
- ``moderate_highdisp``(ADI < ADI*, CV² ≥ CV²*)  → TSB
- ``dead``            (fewer than 2 nonzero in burn-in) → Croston (default)

ADI* = 4.0 (the reachable boundary on a spare-parts panel).
CV²* = 0.49 (Syntetos & Boylan 2005).

Role: current practice baseline.

Gate 3.3: no single cell may contain > 90 % of RUF SKUs, and the mapping must
be printed into T7.
"""

from __future__ import annotations

import numpy as np

from ..forecasters.base import FittingInput, ForecastState
from ..sim.inventory import aggregate_order_up_to

__all__ = ["arm_b_factory", "SB_MODEL_MAP"]

# Mapping from burn-in SB cell to model name.
# Rev 2 §5, handoff §7.6: ADI* = 4.0, CV²* = 0.49.
SB_MODEL_MAP: dict[str, str] = {
    "high_lowdisp": "croston",      # ADI ≥ 4, CV² < 0.49 → Croston
    "moderate_lowdisp": "sba",      # ADI < 4, CV² < 0.49 → SBA (bias-corrected)
    "high_highdisp": "tsb",         # ADI ≥ 4, CV² ≥ 0.49 → TSB (handles lumpy)
    "moderate_highdisp": "tsb",     # ADI < 4, CV² ≥ 0.49 → TSB
    "dead": "croston",              # ≥ 2 nonzero periods unavailable → Croston
}


class _ArmBLevelFn:
    """Stateful level function for arm (b).

    The model choice is fixed at construction time from the SB cell. Only the
    model parameters are updated each period (rolling fit on the observed history).
    """

    def __init__(self, pool, cfg, sku_id, alpha, fit_on, chosen_method):
        self.pool = pool
        self.cfg = cfg
        self.sku_id = sku_id
        self.alpha = alpha
        self.fit_on = fit_on
        self.chosen_method = chosen_method
        self.last_state: ForecastState | None = None
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


def arm_b_factory(pool, cfg, sku_id, alpha, fit_on, *, sb_cell="dead"):
    """Return a stateful level function for arm (b).

    Parameters
    ----------
    sb_cell
        The burn-in Syntetos-Boylan cell (from ``sku_meta.parquet``).
        Determines the model at construction time; never revisited.
    """
    chosen_method = SB_MODEL_MAP.get(sb_cell, "croston")
    return _ArmBLevelFn(pool, cfg, sku_id, alpha, fit_on, chosen_method)
