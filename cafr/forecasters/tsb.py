"""TSB -- Teunter-Syntetos-Babai.

Smooths the OCCURRENCE PROBABILITY directly, so it can decay to zero on a series
that stops demanding. That property is why it is a candidate remedy for C3
(remedy R3) and why gate 2.3 requires the final ``z_hat < 0.01`` after 200 zero
periods.

Validated against ``statsforecast.models.TSB`` to relative error <= 1e-6 (gate 2.1).
Our default smoothing constants are (alpha_d = 0.2, alpha_p = 0.2); statsforecast's
defaults are (0.2, 0.2) as well.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastState, Forecaster, residual_sigma, ses_forecast

__all__ = ["TSB"]


class TSB(Forecaster):
    """``mu = z_hat * mu_z_hat``, both smoothed by SES."""

    name = "tsb"

    def __init__(self, alpha_d: float = 0.2, alpha_p: float = 0.2) -> None:
        super().__init__(alpha_d=alpha_d, alpha_p=alpha_p)
        self.alpha_d = float(alpha_d)
        self.alpha_p = float(alpha_p)

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        if not np.any(y != 0):
            # statsforecast returns zeros here, and does NOT decay: there is
            # nothing to smooth from. Keep that behaviour so the two agree.
            return ForecastState(
                mu=0.0, z=0.0, sigma=0.0, mu_nonzero=0.0, n_nonzero=0,
                method=self.name, diag={"all_zero": True},
            )

        indicator = (y != 0).astype("float64")
        z_hat, _ = ses_forecast(indicator, self.alpha_p)

        nonzero = y[y > 0]
        mu_z, fitted_z = ses_forecast(nonzero, self.alpha_d)

        return ForecastState(
            mu=float(z_hat * mu_z),
            z=float(np.clip(z_hat, 0.0, 1.0)),
            sigma=residual_sigma(nonzero - fitted_z),
            mu_nonzero=float(mu_z),
            n_nonzero=int(nonzero.size),
            method=self.name,
            diag={"z_hat": float(z_hat)},
        )
