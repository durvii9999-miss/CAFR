"""The forecaster interface (M0).

Interface per Rev 2 §15.6 / handoff §7.2::

    fit(y, fit_on) -> mu_hat, z_hat, sigma_hat
    predict(h)     -> point, quantiles

Three things this module owns that are easy to get wrong:

1. **``fit_on`` is respected by construction, not by discipline.** Every model fits
   through ``FittingInput.resolve()``, a single shared function. A model could not
   ignore the switch without deliberately bypassing it, and tests/test_fit_on.py
   asserts every model in a run resolves to the same series.

2. **The quantile floor.** ``alpha_max = min(0.999, 1 - 1/(n+1))`` with ``n`` the
   nonzero cycles available. The controller can never demand a quantile the sample
   cannot support (Rev 2 §7.1). The effective level is returned, not swallowed.

3. **Residuals come from the rolling loop, not from inside ``fit``.** The driver
   records the one-step-ahead forecast it actually made at each period, so
   ``e_t = y_t - mu_hat_t`` is a genuine out-of-sample residual. Computing a fitted
   path inside ``fit`` would be an in-sample residual and would understate the error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..data.synth.impute import impute_lost_sales

__all__ = [
    "ForecastState",
    "Forecaster",
    "FittingInput",
    "ses_forecast",
    "demand_intervals",
    "quantile_ceiling",
    "empirical_quantile",
    "residual_sigma",
]


# --------------------------------------------------------------------------- #
# Fitting input -- the single place the fit_on switch is applied               #
# --------------------------------------------------------------------------- #


@dataclass
class FittingInput:
    """What a forecaster is allowed to fit on.

    Constructed by the simulator, which knows the truth. The forecaster reads only
    ``resolve()``. ``y_true`` is used **only** when ``fit_on == "true"``, which Rev 2
    §14.3 rule 6 declares an idealisation to be declared as such in the paper.
    """

    y_observed: np.ndarray
    censored: np.ndarray
    fit_on: str = "uncensored"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    y_true: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.y_observed = np.asarray(self.y_observed, dtype="float64")
        self.censored = np.asarray(self.censored, dtype=bool)
        if self.y_true is not None:
            self.y_true = np.asarray(self.y_true, dtype="float64")
        if self.fit_on not in ("observed", "uncensored", "true"):
            raise ValueError(f"unknown fit_on: {self.fit_on!r}")
        # MEMOISED. The imputation draws from ``rng``, so an uncached second call
        # would return a DIFFERENT series than the first. Two forecasters in one run
        # would then fit on two different imputations of the same SKU -- different by
        # pure chance, invisible, and enough to make a paired arm comparison
        # meaningless. Cache makes ``resolve()`` a pure function of this object.
        self._cache: dict[str, np.ndarray] = {}

    def resolve(self) -> np.ndarray:
        """Return the series this model fits on. THE fit_on switch.

        Idempotent: repeated calls return the identical array object.
        """
        cached = self._cache.get(self.fit_on)
        if cached is not None:
            return cached

        if self.fit_on == "observed":
            series = self.y_observed
        elif self.fit_on == "uncensored":
            series = impute_lost_sales(self.y_observed, self.censored, self.rng).y
        else:
            if self.y_true is None:
                raise ValueError("fit_on='true' requires y_true to be supplied")
            series = self.y_true

        self._cache[self.fit_on] = series
        return series


# --------------------------------------------------------------------------- #
# Shared numerics                                                              #
# --------------------------------------------------------------------------- #


def ses_forecast(x: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Simple exponential smoothing -- one-step-ahead forecast and fitted values.

    Matches ``statsforecast.models._ses_forecast`` **exactly**, verified in
    tests/test_forecasters.py (gate 2.1). The recursion::

        fitted[0] = nan
        fitted[1] = x[0]
        fitted[i] = alpha*x[i-1] + (1-alpha)*fitted[i-1]      i >= 2
        forecast  = alpha*x[n-1] + (1-alpha)*fitted[n-1]      n > 1, else x[0]

    Note ``fitted[i]`` is a prediction OF ``x[i]`` from data strictly before ``i``.
    """
    x = np.asarray(x, dtype="float64")
    n = x.size
    fitted = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return 0.0, fitted
    if n == 1:
        return float(x[0]), fitted
    fitted[1] = x[0]
    for i in range(2, n):
        fitted[i] = alpha * x[i - 1] + (1.0 - alpha) * fitted[i - 1]
    forecast = alpha * x[n - 1] + (1.0 - alpha) * fitted[n - 1]
    return float(forecast), fitted


def demand_intervals(y: np.ndarray) -> np.ndarray:
    """Intervals between nonzero elements, the first measured from index 0.

    Matches ``statsforecast.models._intervals`` (``diff(nonzero+1, prepend=0)``).
    """
    y = np.asarray(y)
    nonzero = np.flatnonzero(y != 0)
    if nonzero.size == 0:
        return np.asarray([], dtype="float64")
    return np.diff(nonzero + 1, prepend=0).astype("float64")


def quantile_ceiling(n: int) -> float:
    """``min(0.999, 1 - 1/(n+1))`` -- the largest estimable quantile level.

    With ``n = 30`` cycles this is ~0.968. Replaces the old ``z_max = Phi^-1(0.999)``,
    which demanded a quantile the data could not support (Rev 2 §7.1).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return min(0.999, 1.0 - 1.0 / (n + 1.0))


def empirical_quantile(
    sample: np.ndarray,
    alpha: float,
    n_cycles: int | None = None,
    method: str = "linear",
) -> tuple[float, float]:
    """Empirical ``alpha``-quantile with the ``alpha_max`` floor applied.

    Returns ``(quantile, alpha_effective)``. The floor is reported, not silent: the
    paper needs to say which decision points were capped.
    """
    sample = np.asarray(sample, dtype="float64")
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan"), float(alpha)
    n = int(n_cycles) if n_cycles is not None else int(sample.size)
    alpha_eff = min(float(alpha), quantile_ceiling(n))
    return float(np.quantile(sample, alpha_eff, method=method)), alpha_eff


def residual_sigma(residuals: np.ndarray) -> float:
    """Robust residual scale, MAD-based, so one spike does not set it."""
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    mad = np.median(np.abs(r - np.median(r)))
    return float(1.4826 * mad)


# --------------------------------------------------------------------------- #
# The interface                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ForecastState:
    """Output of ``fit``: ``mu_hat``, ``z_hat``, ``sigma_hat`` (Rev 2 §15.6)."""

    mu: float                          # one-step-ahead point forecast
    z: float                           # occurrence probability  z_hat
    sigma: float                       # residual scale
    mu_nonzero: float = float("nan")   # conditional nonzero size
    n_nonzero: int = 0
    method: str = ""
    diag: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "mu": self.mu,
            "z": self.z,
            "sigma": self.sigma,
            "mu_nonzero": self.mu_nonzero,
            "n_nonzero": self.n_nonzero,
            **{f"diag_{k}": v for k, v in self.diag.items()},
        }


class Forecaster(ABC):
    """One class per method, one common interface. Every arm composes these."""

    name: str = "abstract"

    def __init__(self, **params) -> None:
        self.params = params
        self._resolved: np.ndarray | None = None
        self._resolved_on: str = "observed"
        self._state: ForecastState | None = None

    # -- required ---------------------------------------------------------- #

    @abstractmethod
    def _fit_series(self, y: np.ndarray) -> ForecastState:
        """Fit on an already-resolved series. Subclasses implement this."""

    # -- provided ---------------------------------------------------------- #

    def fit(self, y, fit_on: str | None = None) -> ForecastState:
        """Fit and return the state.

        ``y`` is normally a ``FittingInput`` (the only path that applies ``fit_on``).
        A raw array is accepted for unit tests of the recursion itself, in which case
        the series is used exactly as given.
        """
        if isinstance(y, FittingInput):
            series = y.resolve()
            resolved_on = y.fit_on
        else:
            series = np.asarray(y, dtype="float64")
            resolved_on = fit_on or "observed"

        if series.size == 0:
            raise ValueError("cannot fit on an empty series")

        self._resolved = series
        self._resolved_on = resolved_on
        state = self._fit_series(series)
        state.diag["fit_on"] = resolved_on
        self._state = state
        return state

    def predict(self, h: int = 1) -> np.ndarray:
        """Point forecasts for horizons 1..h. Croston-family forecasts are flat."""
        state = self.state
        return np.full(h, max(state.mu, 0.0), dtype="float64")

    def predict_quantiles(
        self,
        residuals: np.ndarray,
        alphas: Sequence[float] = (0.80, 0.90, 0.95),
        *,
        method: str = "linear",
    ) -> dict[float, tuple[float, float, float]]:
        """One-period predictive intervals from the **out-of-sample** residuals.

        Returns ``{alpha: (q_lo, q_hi, alpha_effective)}``. Coverage without width is
        meaningless (Rev 2 §17), so both bounds are returned, plus the effective
        level after the ``alpha_max`` floor.

        The residuals must come from the rolling driver, not from ``fit``. An
        in-sample residual understates the error and would make C2's coverage test
        look correct on a mis-calibrated interval.
        """
        residuals = np.asarray(residuals, dtype="float64")
        residuals = residuals[np.isfinite(residuals)]
        state = self.state
        n_cycles = max(state.n_nonzero, 1)

        if residuals.size == 0:
            # No history yet. Fall back to a symmetric scale estimate and say so.
            z = {"0.8": 1.2816, "0.9": 1.6449, "0.95": 1.9600}
            half = state.sigma
            return {
                float(a): (
                    max(state.mu - z[f"{a:.1f}"] * half, 0.0),
                    state.mu + z[f"{a:.1f}"] * half,
                    float(a),
                )
                for a in alphas
            }

        out: dict[float, tuple[float, float, float]] = {}
        for a in alphas:
            lo, a_lo = empirical_quantile(residuals, (1.0 - a) / 2.0, n_cycles, method)
            hi, _ = empirical_quantile(residuals, 1.0 - (1.0 - a) / 2.0, n_cycles, method)
            out[float(a)] = (
                max(state.mu + min(lo, 0.0), 0.0),
                max(state.mu + hi, 0.0),
                a_lo,
            )
        return out

    @property
    def state(self) -> ForecastState:
        if self._state is None:
            raise RuntimeError("fit() must be called first")
        return self._state

    @property
    def resolved_series(self) -> np.ndarray:
        if self._resolved is None:
            raise RuntimeError("fit() must be called first")
        return self._resolved

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.params})"
