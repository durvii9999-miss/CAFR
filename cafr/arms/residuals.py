"""One-step-ahead residuals for the policy's predictive quantile.

``docs/17`` requires the order-up-to level to read the arm's own rolling residuals::

    S = R * mu_hat_t + Q_alpha( sum_{k=0}^{R-1} e_{t-k} )

The residual is **out of sample**: ``e_t = y_t - mu_hat_{t|t-1}``, where
``mu_hat_{t|t-1}`` was produced at the previous call, before ``y_t`` was seen. Using an
in-sample residual here would understate the error and make the safety stock too small.

The tracker is deliberately tiny and stateful. Each arm owns one; it is updated in the
same order every call:

1. ``observe(y_observed[-1])`` -- score the forecast made at the previous call.
2. Fit the model on the history including ``y_t``.
3. ``set_forecast(state.mu)`` -- store the forecast for ``t + 1``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ResidualTracker"]


class ResidualTracker:
    """Collects one-step-ahead residuals in time order."""

    def __init__(self) -> None:
        self._values: list[float] = []
        self._next_forecast: float = float("nan")

    def observe(self, y_now: float) -> None:
        """Append ``y_now - forecast_made_at_previous_call`` when that is available."""
        if np.isfinite(self._next_forecast):
            self._values.append(float(y_now) - float(self._next_forecast))

    def set_forecast(self, mu_hat: float) -> None:
        """Store the current call's forecast for the next period."""
        self._next_forecast = float(mu_hat)

    def to_array(self) -> np.ndarray:
        """Residuals in time order, length = number of scored calls."""
        return np.asarray(self._values, dtype="float64")

    def __len__(self) -> int:
        return len(self._values)
