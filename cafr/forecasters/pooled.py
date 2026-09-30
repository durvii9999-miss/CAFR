"""The pooled LightGBM, exposed through the pool interface.

``GlobalLightGBM`` is trained **once** across SKUs and then applied per series, so it
does not satisfy the pool contract (``fit(FittingInput) -> ForecastState``) on its own.
``PooledLightGBM`` bridges the two.

**Why this lives in the package and not in an experiment script.** It was defined inside
``experiments/run_ruf.py``, which meant every other driver that built a pool through
``build_pool()`` got an *untrained* ``GlobalLightGBM`` and died on
``NotImplementedError`` the moment it was used -- which is exactly what happened to the
C7 pre-flight. A pool member that only works in one script is a trap; the fix is to put
it where ``build_pool`` is, so all drivers get the same, working pool.

**Fairness (handoff §8.2).** Every arm receives the same pool dict. The trained model is
shared read-only across SKUs: it is never refitted per arm, so no arm sees a model the
others do not.
"""

from __future__ import annotations

import numpy as np

from .base import FittingInput, Forecaster
from .lightgbm_global import GlobalLightGBM

__all__ = ["PooledLightGBM", "fit_global_pool"]


class PooledLightGBM(Forecaster):
    """A globally-trained ``GlobalLightGBM`` behind the pool interface."""

    name = "lightgbm_global"

    def __init__(self, model: GlobalLightGBM, statics_by_sku: dict[str, dict]) -> None:
        super().__init__()
        self._model = model
        self._statics = statics_by_sku

    def _fit_series(self, y: np.ndarray):  # pragma: no cover - never used
        raise NotImplementedError("PooledLightGBM fits through fit(), not _fit_series()")

    def fit(self, y, fit_on: str | None = None):  # type: ignore[override]
        if isinstance(y, FittingInput):
            series = y.resolve()
            resolved_on = y.fit_on
        else:
            series = np.asarray(y, dtype="float64")
            resolved_on = fit_on or "observed"

        sku = getattr(y, "sku_id", None)
        statics = self._statics.get(sku, {})
        state = self._model.fit_series_with_statics(series, statics, fit_on=resolved_on)
        state.diag["fit_on"] = resolved_on
        self._state = state
        return state


def fit_global_pool(
    cfg: dict,
    panel,
    *,
    statics_by_sku: dict[str, dict] | None = None,
    demand_col: str = "demand_observed",
) -> dict:
    """Train the pooled model once on ``panel`` and return a ready-to-use pool.

    ``panel`` is a long frame with ``sku_id`` / ``period`` / ``demand_col`` plus the
    static columns. Series not present in ``statics_by_sku`` are fitted with empty
    statics, so a caller that only wants the classical members can pass ``None``.
    """
    from .registry import build_pool

    statics_by_sku = statics_by_sku or {}
    model = GlobalLightGBM(
        params=cfg["forecast"].get("lightgbm", {}), seed=int(cfg.get("seed_root", 42))
    ).fit_global(panel)
    pool = build_pool(cfg)
    pool["lightgbm_global"] = PooledLightGBM(model, statics_by_sku)
    return pool
