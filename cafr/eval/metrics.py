"""Evaluation metrics. Rev 2 §17–§19, handoff §7.3, OD-8.

All formula sources are cited in comments. Do NOT use sklearn.metrics defaults for
scaled errors -- the defaults are wrong for intermittent demand (OD-8).

Primary accuracy metric: MASEII (arXiv 2609.13840 definition).
Secondary: MASE with the denominator caveat stated.
Reward: the OD-3 standardised cost reward (frozen).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mase",
    "maseii",
    "standardised_cost_reward",
    "period_service_level",
    "cycle_service_level",
]


# --------------------------------------------------------------------------- #
# Accuracy metrics                                                             #
# --------------------------------------------------------------------------- #

def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray | None = None,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Source: Hyndman & Koehler (2006). The denominator is the mean absolute
    naive-forecast error on the training series (seasonal naive with period
    ``seasonal_period``).

    Caveat (OD-8, Rev 2 §17): on intermittent series the naive-forecast
    denominator approaches zero whenever the series is mostly zeros, making MASE
    unstable. Always report MASEII alongside this.

    Parameters
    ----------
    y_true, y_pred
        Arrays of equal length. For a rolling-origin run these are the evaluated
        window arrays (length = evaluated).
    y_train
        The training / burn-in portion. Used only for the denominator. If None,
        falls back to the first half of y_true.
    seasonal_period
        Naive forecast period m. For monthly intermittent demand use 1 (random
        walk denominator, which is the standard in the intermittent-demand
        literature).
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_train is None:
        y_train = y_true[: max(1, len(y_true) // 2)]

    m = seasonal_period
    if len(y_train) <= m:
        denom = float(np.mean(np.abs(y_train))) + 1e-9
    else:
        diffs = np.abs(y_train[m:] - y_train[:-m])
        denom = float(np.mean(diffs)) + 1e-9

    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae / denom


def maseii(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z_pred: np.ndarray,
    *,
    y_train: np.ndarray | None = None,
) -> float:
    """MASEII — accounts for occurrence misclassification.

    Source: arXiv 2609.13840 (ICDM-26). Standard MASE explodes on intermittent
    series when the naive denominator approaches zero. MASEII separates the
    occurrence and size components.

    z_pred is the predicted occurrence probability (from the forecaster's z_hat).
    The occurrence error is |indicator(y_t > 0) - z_pred_t|.
    The size error, restricted to demand-positive periods, is |y_t - mu_t|.

    Both are scaled by the same MASE denominator.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    z_pred = np.asarray(z_pred, dtype="float64")
    if y_train is None:
        y_train = y_true[: max(1, len(y_true) // 2)]

    denom = float(np.mean(np.abs(y_train))) + 1e-9

    # Occurrence component: all periods
    d = (y_true > 0).astype("float64")
    e_occ = float(np.mean(np.abs(d - np.clip(z_pred, 0.0, 1.0))))

    # Size component: nonzero periods only
    nz = y_true > 0
    if nz.any():
        e_sz = float(np.mean(np.abs(y_true[nz] - y_pred[nz])))
    else:
        e_sz = 0.0

    return (e_occ + e_sz) / denom


# --------------------------------------------------------------------------- #
# OD-3 Reward (FROZEN — do not redesign mid-experiment)                       #
# --------------------------------------------------------------------------- #

def standardised_cost_reward(
    holding_cost: float,
    backorder_cost: float,
    cost_mean: float,
    cost_std_floored: float,
) -> float:
    """OD-3 standardised inventory cost reward (Rev 2 §9.3, handoff §2.8).

    r_{i,t} = -[ (H * I_bar + B * B_bar) - C_bar_i ] / sigma_{C,i}

    The period cost (H * I_bar + B * B_bar) is passed in as
    ``holding_cost + backorder_cost``.

    The floor ``sigma_{C,i} = max(sigma, 0.05 * C_bar_i)`` must be applied
    BEFORE calling this function (see ``compute_sku_cost_stats`` in loop.py).

    Returns a scalar reward (higher = better).
    """
    period_cost = holding_cost + backorder_cost
    return -(period_cost - cost_mean) / max(cost_std_floored, 1e-12)


# --------------------------------------------------------------------------- #
# Service metrics (convenience wrappers)                                       #
# --------------------------------------------------------------------------- #

def period_service_level(demand_met: np.ndarray, demand_true: np.ndarray) -> float:
    """Fraction of periods where all demand was met.

    Note: this is NOT the cycle-service-level (CSL). On intermittent series
    this inflates relative to CSL because zero-demand periods always count as
    served (Rev 2 §15.6, handoff §2.12).
    """
    demand_met = np.asarray(demand_met, dtype="float64")
    demand_true = np.asarray(demand_true, dtype="float64")
    if len(demand_true) == 0:
        return float("nan")
    return float(np.mean(demand_met >= demand_true - 1e-9))


def cycle_service_level(demand_met: np.ndarray, demand_true: np.ndarray) -> float:
    """Fraction of NONZERO-demand periods where all demand was met.

    This is the correct metric to compare against the policy's alpha target
    (Rev 2 §15.6, gate 1.2).
    """
    demand_met = np.asarray(demand_met, dtype="float64")
    demand_true = np.asarray(demand_true, dtype="float64")
    nz = demand_true > 1e-9
    if not nz.any():
        return float("nan")
    return float(np.mean(demand_met[nz] >= demand_true[nz] - 1e-9))
