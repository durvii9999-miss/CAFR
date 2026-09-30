"""Global (pooled) LightGBM -- the ML baseline, arm (e)'s forecaster.

**Pooled across SKUs.** One model, trained on all training SKUs jointly, with SKU
static features and lagged demand as inputs. **There is no SKU-specific training**:
that would let the model memorise a SKU's own series and would leak across the
train/test split (handoff §8.4).

Determinism (gate 2.4): fixed seed, ``n_jobs = 1``, ``deterministic = True``,
``force_col_wise = True``. Two runs must be byte-identical.

Leakage: every feature is a TRAILING window over the observed series. No centred
window, no full-series statistic, no period index (an absolute time coordinate is
observable, but it hands the model the regime schedule, so it is excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastState, Forecaster

__all__ = ["GlobalLightGBM", "build_design_matrix", "FEATURE_COLUMNS"]

LAGS = (1, 2, 3, 4)
ROLL_WINDOWS = (4, 12, 24)

# Static features the model may use. All are non-leaking (burn-in cell, unit cost,
# lead time, holding rate). adi_full / cv2_full are deliberately absent.
STATIC_COLUMNS = (
    "lead_time",
    "unit_cost",
    "holding_rate",
    "adi_init",
    "cv2_init",
)

FEATURE_COLUMNS = (
    *[f"lag_{k}" for k in LAGS],
    *[f"rollmean_{w}" for w in ROLL_WINDOWS],
    *[f"rollzerofrac_{w}" for w in ROLL_WINDOWS],
    *[f"rollmeannonzero_{w}" for w in ROLL_WINDOWS],
    *STATIC_COLUMNS,
)


def build_design_matrix(
    panel: pd.DataFrame,
    *,
    demand_col: str = "demand_observed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the pooled design matrix from a long panel.

    ``panel`` must have ``sku_id``, ``period``, ``demand_observed`` and the static
    columns. Returns ``(X, y, sku_index, period_index)`` where rows with an
    incomplete trailing window are dropped.

    All rolling statistics are **trailing** (``min_periods=1`` over past values only,
    shifting by one so the current period is never an input to its own prediction).
    """
    required = {"sku_id", "period", demand_col, *STATIC_COLUMNS}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns for the design matrix: {sorted(missing)}")

    # Index-safe: the caller may hand us a filtered panel whose index still carries
    # the original row numbers. Rolling on a Series aligns by LABEL, so without this
    # reset every feature column would silently become all-NaN.
    df = panel.sort_values(["sku_id", "period"]).reset_index(drop=True)
    skus = df["sku_id"]
    grouped = df.groupby("sku_id", sort=False)[demand_col]

    for k in LAGS:
        df[f"lag_{k}"] = grouped.shift(k)

    # The rolling statistics must be computed WITHIN each SKU. A plain
    # ``past.rolling(w)`` walks the concatenated frame and lets the previous SKU's
    # tail leak into the current SKU's features. Group first, then roll.
    past = grouped.shift(1)
    for w in ROLL_WINDOWS:
        df[f"rollmean_{w}"] = (
            past.groupby(skus).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"rollzerofrac_{w}"] = (
            (past == 0)
            .astype("float64")
            .groupby(skus)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        nz = past.where(past > 0)
        df[f"rollmeannonzero_{w}"] = (
            nz.groupby(skus).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    for c in ("adi_init", "cv2_init"):
        df[c] = df[c].fillna(0.0)          # `dead` SKUs: NaN -> explicit 0, cell carries the flag

    complete = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    usable = df[complete]
    if usable.empty:
        raise ValueError("no usable rows: every series is shorter than the largest lag")

    X = usable[list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
    y = usable[demand_col].to_numpy(dtype="float64")
    return X, y, usable["sku_id"].to_numpy(), usable["period"].to_numpy()


class GlobalLightGBM(Forecaster):
    """One pooled LightGBM. ``fit`` on a single series requires a prior ``fit_global``."""

    name = "lightgbm_global"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        super().__init__(params=params or {}, seed=seed)
        self.params = dict(params or {})
        self.seed = int(seed)
        self._model = None

    # -- global training --------------------------------------------------- #

    def fit_global(self, panel: pd.DataFrame) -> "GlobalLightGBM":
        """Train one model on all training SKUs jointly."""
        import lightgbm as lgb

        X, y, _, _ = build_design_matrix(panel)
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "n_estimators": 300,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            **self.params,
        }
        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(X, y)
        return self

    # -- per-series interface ---------------------------------------------- #

    def _fit_series(self, y: np.ndarray) -> ForecastState:
        raise NotImplementedError(
            "GlobalLightGBM is trained once across SKUs. Call fit_global(panel) first, "
            "then use fit_series_with_statics(...) per series."
        )

    def fit_series_with_statics(
        self,
        y: np.ndarray,
        statics: dict,
        *,
        fit_on: str = "observed",
    ) -> ForecastState:
        """Produce a one-step-ahead forecast for one series from the pooled model."""
        if self._model is None:
            raise RuntimeError("call fit_global(panel) before fitting a single series")

        row = _feature_row(np.asarray(y, dtype="float64"), statics)
        mu = float(self._model.predict(row.reshape(1, -1))[0])

        nonzero = y[y > 0]
        z = float((y[-24:] != 0).mean()) if y.size else 0.0
        return ForecastState(
            mu=max(mu, 0.0),
            z=z,
            sigma=float(np.std(nonzero)) if nonzero.size > 1 else 0.0,
            mu_nonzero=float(nonzero.mean()) if nonzero.size else 0.0,
            n_nonzero=int(nonzero.size),
            method=self.name,
            diag={"fit_on": fit_on},
        )


def _feature_row(y: np.ndarray, statics: dict) -> np.ndarray:
    """Build one feature row from a series' trailing history plus its statics."""
    values: list[float] = []
    for k in LAGS:
        values.append(float(y[-k]) if y.size >= k else 0.0)
    for w in ROLL_WINDOWS:
        tail = y[-w:] if y.size else np.asarray([0.0])
        values.append(float(tail.mean()))
        values.append(float((tail == 0).mean()))
        nz = tail[tail > 0]
        values.append(float(nz.mean()) if nz.size else 0.0)
    for c in STATIC_COLUMNS:
        v = statics.get(c, 0.0)
        values.append(0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
    return np.asarray(values, dtype="float64")
