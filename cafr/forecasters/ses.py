"""SES on nonzero sizes, with the occurrence rate held separately.

Rev 2 §15.6's pool row, verbatim: *"SES on nonzero sizes -- Occurrence via a
separate Bernoulli rate."*

So the size sub-model is SES on the nonzero sizes, and the occurrence sub-model is
an **unweighted Bernoulli rate over a trailing window**, not a smoothed indicator.
That is what distinguishes it from TSB, which smooths the indicator exponentially.

Reduction used by the gate tests: with the occurrence rate forced to 1.0
(``occurrence_window=None`` and ``p_fixed=1.0``), this model equals plain SES on the
series, which IS validated against
``statsforecast.models.SimpleExponentialSmoothing``.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastState, Forecaster, residual_sigma, ses_forecast

__all__ = ["SESOnSizes"]


class SESOnSizes(Forecaster):
    """``mu = Bernoulli_rate(window) * SES(nonzero sizes)``."""

    name = "ses_sizes"

    def __init__(self, alpha: float = 0.1, occurrence_window: int | None = 24,
                 p_fixed: float | None = None) -> None:
        super().__init__(alpha=alpha, occurrence_window=occurrence_window, p_fixed=p_fixed)
        self.alpha = float(alpha)
        self.occurrence_window = occurrence_window
        self.p_fixed = p_fixed

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        nonzero = y[y > 0]

        if self.p_fixed is not None:
            p_hat = float(self.p_fixed)
        elif self.occurrence_window is None:
            p_hat = float((y != 0).mean())
        else:
            tail = y[-self.occurrence_window:]
            p_hat = float((tail != 0).mean())

        if nonzero.size == 0:
            return ForecastState(
                mu=0.0, z=p_hat, sigma=0.0, mu_nonzero=0.0, n_nonzero=0,
                method=self.name, diag={"all_zero": True},
            )

        mu_z, fitted_z = ses_forecast(nonzero, self.alpha)
        return ForecastState(
            mu=float(p_hat * mu_z),
            z=float(np.clip(p_hat, 0.0, 1.0)),
            sigma=residual_sigma(nonzero - fitted_z),
            mu_nonzero=float(mu_z),
            n_nonzero=int(nonzero.size),
            method=self.name,
            diag={"p_hat": p_hat},
        )
