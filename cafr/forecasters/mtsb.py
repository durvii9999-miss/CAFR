"""mTSB -- the "modified" TSB variant: occurrence from TSB, size from a damped trend.

**Definition note -- read this before comparing to anything.** As with mSBA,
Rev 2 §15.6 names mTSB but does not pin it, and ``statsforecast`` has no mTSB.
Gate 2.1 therefore has no external referent for this model either.

Adopted definition:

    mTSB = TSB (occurrence smoothed by SES, unchanged) x a DAMPED-TREND size
    sub-model instead of plain SES on the nonzero sizes.

Why this one:

1. It is **exactly reducible**: ``mTSB(phi = 1.0, beta = 0.0)`` reduces to plain SES
   on the nonzero sizes, so mTSB reduces to TSB, and tests/test_forecasters.py
   asserts that reduction.
2. Rev 2 §7 describes remedy **R4** as "refit on nonzero sizes only, with recency
   weighting (**damped trend**, ...)". mTSB is the pool member that already carries
   that estimator, so R4 has something concrete to switch to rather than something
   to be invented at Step 7.

Record the adopted definition in T7.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastState, Forecaster, residual_sigma, ses_forecast

__all__ = ["mTSB"]


class mTSB(Forecaster):
    """TSB occurrence x damped-trend size."""

    name = "mtsb"

    def __init__(
        self,
        alpha_d: float = 0.2,
        alpha_p: float = 0.2,
        beta: float = 0.0,
        phi: float = 1.0,
    ) -> None:
        super().__init__(alpha_d=alpha_d, alpha_p=alpha_p, beta=beta, phi=phi)
        self.alpha_d = float(alpha_d)
        self.alpha_p = float(alpha_p)
        self.beta = float(beta)
        self.phi = float(phi)

    def _size_damped_trend(self, nonzero: np.ndarray) -> tuple[float, np.ndarray]:
        """Damped-trend level on the nonzero sizes.

        ``fitted[i]`` is a prediction of ``nonzero[i]`` from strictly earlier values,
        matching the one-step-ahead convention used elsewhere.
        """
        n = nonzero.size
        fitted = np.full(n, np.nan, dtype="float64")
        if n == 1:
            return float(nonzero[0]), fitted

        level = float(nonzero[0])
        trend = 0.0
        for i in range(1, n):
            fitted[i] = level + self.phi * trend
            prev_level = level
            level = self.alpha_d * nonzero[i] + (1.0 - self.alpha_d) * (level + self.phi * trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * self.phi * trend
        return float(max(level + self.phi * trend, 0.0)), fitted

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        if not np.any(y != 0):
            return ForecastState(
                mu=0.0, z=0.0, sigma=0.0, mu_nonzero=0.0, n_nonzero=0,
                method=self.name, diag={"all_zero": True},
            )

        indicator = (y != 0).astype("float64")
        z_hat, _ = ses_forecast(indicator, self.alpha_p)

        nonzero = y[y > 0]
        mu_z, fitted_z = self._size_damped_trend(nonzero)

        return ForecastState(
            mu=float(z_hat * mu_z),
            z=float(np.clip(z_hat, 0.0, 1.0)),
            sigma=residual_sigma(nonzero - fitted_z),
            mu_nonzero=float(mu_z),
            n_nonzero=int(nonzero.size),
            method=self.name,
            diag={"z_hat": float(z_hat), "phi": self.phi, "beta": self.beta},
        )
