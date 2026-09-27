"""SBA -- the Syntetos-Boylan Approximation: Croston with a 0.95 bias correction.

Validated against ``statsforecast.models.CrostonSBA`` to relative error <= 1e-6
(gate 2.1).

The correction is applied to the FINAL forecast only. ``mu_nonzero`` stays at the
raw smoothed size, because R1's damped intercept correction and R4's size refit
operate on the size sub-process, and folding the 0.95 into it would double-count.

Reference: Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent
demand estimates. *International Journal of Forecasting*, 21(2), 303-314.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastState, Forecaster, demand_intervals, residual_sigma, ses_forecast

__all__ = ["SBA"]


class SBA(Forecaster):
    """Croston x 0.95."""

    name = "sba"
    correction = 0.95

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__(alpha=alpha)
        self.alpha = float(alpha)

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        nonzero = y[y > 0]
        if nonzero.size == 0:
            return ForecastState(
                mu=0.0, z=0.0, sigma=0.0, mu_nonzero=0.0, n_nonzero=0,
                method=self.name, diag={"naive_fallback": True},
            )

        mu_z, fitted_z = ses_forecast(nonzero, self.alpha)
        intervals = demand_intervals(y)
        avg_interval, _ = ses_forecast(intervals, self.alpha)

        if avg_interval != 0.0:
            mu = self.correction * mu_z / avg_interval
            z = float(np.clip(1.0 / avg_interval, 0.0, 1.0))
        else:
            mu = self.correction * mu_z
            z = 1.0

        return ForecastState(
            mu=float(mu),
            z=z,
            sigma=residual_sigma(nonzero - fitted_z),
            mu_nonzero=float(mu_z),
            n_nonzero=int(nonzero.size),
            method=self.name,
            diag={"avg_interval": float(avg_interval), "correction": self.correction},
        )
