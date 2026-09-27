"""Croston (1972) -- separate SES on nonzero sizes and on the inter-demand intervals.

Validated against ``statsforecast.models.CrostonClassic`` to relative error <= 1e-6
(gate 2.1). The definition here matches ``statsforecast.models._croston_classic``
line for line, including the ``_intervals`` convention and the fallback when the
series has no demand at all.

Reference: Croston, J. D. (1972). Forecasting and stock control for intermittent
demands. *Operational Research Quarterly*, 23(3), 289-303.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastState, Forecaster, demand_intervals, residual_sigma, ses_forecast

__all__ = ["Croston"]


class Croston(Forecaster):
    """Classic Croston. ``mu = mu_z / avg_interval``."""

    name = "croston"

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__(alpha=alpha)
        self.alpha = float(alpha)

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        nonzero = y[y > 0]
        if nonzero.size == 0:
            # statsforecast falls back to the naive forecast (repeat last value),
            # which on an all-zero series is 0.
            return ForecastState(
                mu=0.0, z=0.0, sigma=0.0, mu_nonzero=0.0, n_nonzero=0,
                method=self.name, diag={"naive_fallback": True},
            )

        mu_z, fitted_z = ses_forecast(nonzero, self.alpha)
        intervals = demand_intervals(y)
        avg_interval, _ = ses_forecast(intervals, self.alpha)

        if avg_interval != 0.0:
            mu = mu_z / avg_interval
            z = float(np.clip(1.0 / avg_interval, 0.0, 1.0))
        else:
            mu = mu_z
            z = 1.0

        return ForecastState(
            mu=float(mu),
            z=z,
            sigma=residual_sigma(nonzero - fitted_z),
            mu_nonzero=float(mu_z),
            n_nonzero=int(nonzero.size),
            method=self.name,
            diag={"avg_interval": float(avg_interval)},
        )
