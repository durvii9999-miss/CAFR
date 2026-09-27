"""mSBA -- the "modified" SBA variant.

**Definition note -- read this before comparing to anything.** Rev 2 §15.6 names
mSBA and mTSB in the forecaster pool but does not pin their definitions, and
``statsforecast`` implements neither. Gate 2.1 as written therefore has **no external
referent** for these two models. That is escalated as a finding, not papered over.

What is adopted here, and why:

    mSBA = SBA with SEPARATE smoothing constants for the size sub-process and the
    interval sub-process, keeping the 0.95 correction.

Two reasons this is the useful choice rather than an arbitrary one:

1. It is **exactly reducible**: ``mSBA(alpha_z == alpha_p)`` is bit-comparable to
   ``SBA``, so tests/test_forecasters.py can assert the reduction instead of an
   external referent. A model with no referent and no reduction is untestable.
2. Remedies **R3** (switch the occurrence sub-model) and **R4** (re-estimate the
   size model) act on the two sub-processes **separately**. A variant whose two
   smoothing constants can move independently is the natural base for both.

Record the adopted definition in T7.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastState, Forecaster, demand_intervals, residual_sigma, ses_forecast

__all__ = ["mSBA"]


class mSBA(Forecaster):
    """SBA with independent size and interval smoothing constants."""

    name = "msba"
    correction = 0.95

    def __init__(self, alpha_z: float = 0.1, alpha_p: float = 0.1) -> None:
        super().__init__(alpha_z=alpha_z, alpha_p=alpha_p)
        self.alpha_z = float(alpha_z)
        self.alpha_p = float(alpha_p)

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        nonzero = y[y > 0]
        if nonzero.size == 0:
            return ForecastState(
                mu=0.0, z=0.0, sigma=0.0, mu_nonzero=0.0, n_nonzero=0,
                method=self.name, diag={"naive_fallback": True},
            )

        mu_z, fitted_z = ses_forecast(nonzero, self.alpha_z)
        intervals = demand_intervals(y)
        avg_interval, _ = ses_forecast(intervals, self.alpha_p)

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
