"""Inventory simulator (M5). Periodic review, order-up-to ``(S, S)``.

Rev 2 §15.6, handoff §2.12 and §7.1.

=============================================================================
THE ORDER-UP-TO LEVEL -- read this before editing anything below
=============================================================================

``S_{i,t} = Q_hat_alpha(i, t)``, the empirical ``alpha``-quantile of the **R-period
AGGREGATE** demand.

    S = R * mu_hat + F_hat^{-1}(alpha)

where ``F_hat`` is the empirical CDF of ``(A^agg - R * mu_hat)`` over the trailing
window, and ``A^agg`` is the R-period AGGREGATE demand.

**NOT** ``S = R*mu_hat + z*sigma_hat``.

Why this is the single most dangerous line in the project (Rev 2 §15.6): Revision 1
estimated ``sigma_hat`` from **per-period** forecast errors but applied it to the
**R-period** protection interval. Under independence the protection-interval sd is
``sigma_R ~ sqrt(R) * sigma_period``, so at ``R = 3`` the safety-stock term was
understated by ``(sqrt(3) - 1)/sqrt(3) = 42.3 %``. The resulting service shortfall has
nothing to do with any hypothesis, and -- this is the point -- **it is silent**: the
numbers still look plausible, just at the wrong service level.

The Gaussian ``z`` is dropped even when ``sigma_hat`` is correct, because
intermittent-demand errors are right-skewed.

``alpha`` is floored at ``min(0.999, 1 - 1/(n+1))`` with ``n`` the nonzero cycles
available, so the controller can never demand a quantile the sample cannot estimate
(Rev 2 §7.1).

=============================================================================
PROTECTION-INTERVAL CONVENTION -- `R = L + 1` is CORRECT. Measured, not assumed.
=============================================================================

Rev 2 §15.6 states ``R = L + 1`` and calls it a definition. An earlier version of this
docstring doubted it: ordering at the end of ``t`` for arrival at the start of ``t + L``
looks like it covers ``y_{t+1} .. y_{t+L}``, i.e. ``L`` periods.

**That doubt was wrong, and the measurement says so.** This is a periodic-review system
with ``review_period = 1``: the protection interval is *review interval + lead time*
= ``1 + L`` = ``L + 1`` = 3. The extra period is the one between the moment the level is
set and the moment the order it generates can possibly arrive.

Measured on 500 SKUs x 200 periods (``results/step1_protection_diagnostic.json``), with
the per-cycle CSL defined below:

    K=2:  0.7737 (alpha=0.80)   0.8719 (alpha=0.90)    -- under-serves
    K=3:  0.7909 (alpha=0.80)   0.8989 (alpha=0.90)    -- within gate 1.2's +/-0.02

``sim.protection_convention`` still exists and still selects between ``"L+1"`` (the
spec's value, and the default) and ``"L"``, but it is now a *recorded comparison*, not
an open question. Do not change the default.

=============================================================================
CSL IS A PER-CYCLE MEASURE. The per-period fraction is a DIFFERENT number.
=============================================================================

``alpha`` is the newsvendor critical ratio ``B/(B+H)`` -- a **service level per
protection interval**: the probability that one review cycle's demand is fully met.

Measuring it as *the fraction of periods with no unmet demand* inflates it, badly, on
intermittent demand: a period with zero demand is trivially "in service". On the §15.1
process with ``p ~ U(0.05, 0.50)``, **71.4 % of periods have zero demand**, so at
``alpha = 0.80`` the per-period fraction reads **0.9198** while the true per-cycle level
is **0.7909**.

That is the same species of error as the 42.3 % bug this module exists to prevent:
plausible-looking numbers, reported at the wrong service level.

So ``SimResult`` carries BOTH, under names that cannot be confused:

    ``cycle_service_level``   **the headline CSL.** Fraction of non-overlapping
                              protection intervals (length ``cfg.protection``) with no
                              unmet demand. This is the number compared against alpha.
    ``period_service_level``  fraction of *evaluated periods* with no unmet demand.
                              Reported for diagnostics; never compared against alpha.

A cycle is served iff no demand was lost in any period of the cycle. Cycles are
non-overlapping and start at ``warmup = L`` -- before period ``L`` no order can have
arrived, so those periods stock out regardless of the policy and would deflate every
metric by a constant. The warm-up periods stay in the log (full audit trail) and are
excluded from all headline metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..utils.seeds import derive_seed

__all__ = [
    "SimConfig",
    "SimResult",
    "InventorySimulator",
    "aggregate_order_up_to",
    "rolling_aggregates",
]


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class SimConfig:
    """The simulator knobs. Built from ``cfg['sim']`` by ``SimConfig.from_config``."""

    L: int = 2
    R: int = 3
    review_period: int = 1
    unmet_model: str = "lost_sales"
    quantile_window: int = 24
    min_aggregates: int = 4
    quantile_method: str = "linear"
    alpha_min: float = 0.50
    alpha_max_cap: float = 0.999
    H: float = 1.0
    B_over_H: float = 19.0
    c7_safety_factor_scale: float = 1.0
    protection_convention: str = "L+1"

    @classmethod
    def from_config(cls, cfg: dict) -> "SimConfig":
        sim = dict(cfg.get("sim", {}))
        sim.setdefault("protection_convention", "L+1")
        known = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        unknown = set(sim) - known
        if unknown:
            raise ValueError(f"unknown sim config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in sim.items() if k in known})

    @property
    def B(self) -> float:
        return self.H * self.B_over_H

    @property
    def protection(self) -> int:
        """The interval the order-up-to level actually covers."""
        return self.R if self.protection_convention == "L+1" else self.L

    @property
    def target_csl(self) -> float:
        """The newsvendor critical ratio B/(B+H). B/H=19 -> 0.95."""
        return self.B / (self.B + self.H)


@dataclass
class SimResult:
    """Per-SKU simulation log plus the headline inventory metrics.

    ``cycle_service_level`` is THE CSL -- see the module docstring. ``period_service_level``
    is a different, always-larger number and must never be compared against ``alpha``.
    """

    sku_id: str
    log: pd.DataFrame
    cycle_service_level: float
    period_service_level: float
    fill_rate: float
    total_cost: float
    mean_cost: float
    pis: float
    nos: float
    censored: np.ndarray
    n_stockout: int
    n_cycles: int
    warmup: int

    def as_row(self) -> dict:
        return {
            "sku_id": self.sku_id,
            "csl": self.cycle_service_level,
            "period_service_level": self.period_service_level,
            "fill_rate": self.fill_rate,
            "total_cost": self.total_cost,
            "mean_cost": self.mean_cost,
            "pis": self.pis,
            "nos": self.nos,
            "n_stockout": self.n_stockout,
            "n_cycles": self.n_cycles,
        }


# --------------------------------------------------------------------------- #
# The corrected order-up-to level                                              #
# --------------------------------------------------------------------------- #


def rolling_aggregates(y: np.ndarray, R: int, window: int) -> np.ndarray:
    """Complete R-period aggregate demands from the trailing window.

    Only aggregates fully contained in ``y`` are returned. No centred window, no
    future period -- this is the no-look-ahead rule in its most concrete form
    (Rev 2 §14.3 rule 3).
    """
    y = np.asarray(y, dtype="float64")
    if y.size < R:
        return np.asarray([], dtype="float64")
    agg = np.convolve(y, np.ones(R, dtype="float64"), mode="valid")
    return agg[-window:] if window and agg.size > window else agg


def aggregate_order_up_to(
    y_hist: np.ndarray,
    mu_hat: float,
    R: int,
    alpha: float,
    *,
    window: int = 24,
    min_aggregates: int = 4,
    method: str = "linear",
    alpha_max_cap: float = 0.999,
    safety_factor_scale: float = 1.0,
) -> tuple[float, float, int, bool]:
    """``S = R * mu_hat + F_hat^{-1}(alpha)`` -- the R-period AGGREGATE quantile.

    Returns ``(S, alpha_effective, n_aggregates, fallback)``.

    ``fallback`` is True when fewer than ``min_aggregates`` complete aggregates were
    available, in which case ``S = R * mu_hat`` (no safety stock) and the decision
    point must be flagged in the log. It is never silently smoothed over.
    """
    y_hist = np.asarray(y_hist, dtype="float64")
    aggregates = rolling_aggregates(y_hist, R, window)

    # n = nonzero cycles available for the quantile (Rev 2 §7.1). Computed on the
    # same trailing window the aggregates came from, so the floor and the sample
    # refer to the same information set.
    tail = y_hist[-window:] if window else y_hist
    n_cycles = int(np.count_nonzero(tail))

    if aggregates.size < min_aggregates:
        return float(R * mu_hat), float(alpha), int(aggregates.size), True

    alpha_eff = min(float(alpha), min(alpha_max_cap, 1.0 - 1.0 / (n_cycles + 1.0)))
    residuals = aggregates - R * mu_hat
    safety = float(np.quantile(residuals, alpha_eff, method=method))
    safety *= float(safety_factor_scale)      # C7's injection scales ONLY this term
    return float(R * mu_hat + safety), float(alpha_eff), int(aggregates.size), False


# --------------------------------------------------------------------------- #
# The simulator                                                                #
# --------------------------------------------------------------------------- #


class LevelFunction(Protocol):
    """Supplies the order-up-to level at the end of period ``t``.

    Receives the OBSERVED history only. It has no access to ``demand_true``, by
    signature, which is the no-look-ahead rule enforced structurally rather than by
    discipline (Rev 2 §14.3 rule 7).
    """

    def __call__(
        self,
        t: int,
        y_observed: np.ndarray,
        censored: np.ndarray,
        inventory: dict,
    ) -> tuple[float, float]: ...


class InventorySimulator:
    """Periodic-review ``(S, S)`` with lead time, censoring, cost and CSL.

    Timing, per Rev 2 §7.1::

        1. receive arrivals scheduled for t
        2. observe demand y_t
        3. fulfil from on_hand; unmet demand is LOST (default) or backordered
        4. censoring: observed = met, true = demand
        5. cost = H * on_hand_end + B * unmet_t
        6. POLICY DECISION -- reads s <= t only
        7. order = max(0, S_{i,t} - inventory_position); arrives at start of t + L

    **Unmet-demand model.** ``lost_sales`` is the default because gate 1.6 requires
    ``demand_observed < demand_true`` on stock-out periods, which only holds when the
    unmet part is genuinely unobservable. In the backorder model the units arrive
    later, so the observed series is not censored and the C-guard has nothing to
    detect. State the model you used in the paper (Rev 2 §7.1 asks for exactly that).
    """

    def __init__(self, cfg: SimConfig, *, root_seed: int = 42) -> None:
        if cfg.unmet_model not in ("lost_sales", "backorder"):
            raise ValueError(f"unknown unmet_model: {cfg.unmet_model!r}")
        if cfg.protection_convention not in ("L+1", "L"):
            raise ValueError(f"unknown protection_convention: {cfg.protection_convention!r}")
        self.cfg = cfg
        self.root_seed = int(root_seed)

    # -- the main loop ----------------------------------------------------- #

    def run(
        self,
        sku_id: str,
        y_true: np.ndarray,
        level_fn: LevelFunction,
        *,
        arm: str = "sim",
    ) -> SimResult:
        """Simulate one SKU against a fixed true-demand sequence.

        ``y_true`` is the DGP's realised demand. The simulator knows it; the policy
        does not. The level function receives only ``y_observed`` and the censoring
        mask.
        """
        return self._run_series(sku_id, np.asarray(y_true, dtype="float64"), level_fn, arm)

    # -- internals --------------------------------------------------------- #

    def _run_series(
        self,
        sku_id: str,
        y_true: np.ndarray,
        level_fn: LevelFunction,
        arm: str,
    ) -> SimResult:
        cfg = self.cfg
        T = int(y_true.size)
        # No order placed at the end of t can arrive before the start of t + L, so
        # periods 0..L-1 have zero stock whatever the policy says. That is a warm-up
        # artefact, not a service failure: excluded from every headline metric.
        warmup = min(int(cfg.L), T)

        on_hand = 0.0
        pipeline: dict[int, float] = {}
        y_observed = np.zeros(T, dtype="float64")
        censored = np.zeros(T, dtype=bool)
        demand_lost = np.zeros(T, dtype="float64")
        demand_met = np.zeros(T, dtype="float64")
        cost = np.zeros(T, dtype="float64")
        s_level = np.full(T, np.nan, dtype="float64")
        alpha_t = np.full(T, np.nan, dtype="float64")
        order_qty = np.zeros(T, dtype="float64")
        on_hand_end = np.zeros(T, dtype="float64")
        backorders = np.zeros(T, dtype="float64")

        for t in range(T):
            # 1. arrivals
            on_hand += pipeline.pop(t, 0.0)
            on_order = float(sum(pipeline.values()))

            # 2 & 3. demand, fulfilment
            y = float(y_true[t])
            met = min(y, on_hand)
            unmet = y - met

            if cfg.unmet_model == "lost_sales":
                lost = unmet
                backorders[t] = 0.0
                on_hand -= met
            else:
                backorders[t] = unmet
                # Backorder model: unmet demand stays owed. It is met from the next
                # arrival, before any new demand is served.
                on_hand -= met
                if on_hand > 0 and backorders[t] > 0:
                    served = min(on_hand, backorders[t])
                    on_hand -= served
                    backorders[t] -= served
                lost = 0.0
                met = y  # observed series is NOT censored in this model

            # 4. censoring -- the C-guard's flag. In the lost-sales model a period is
            #    potentially censored exactly when sales were lost.
            demand_met[t] = met
            demand_lost[t] = lost
            censored[t] = lost > 0
            y_observed[t] = met

            # 5. cost. The penalty term is the unmet quantity, whichever model.
            cost[t] = cfg.H * on_hand + cfg.B * unmet
            on_hand_end[t] = on_hand

            # 6. policy decision -- observed history only
            state = {
                "on_hand": on_hand,
                "on_order": on_order,
                "backorders": backorders[t],
                "cost_to_date": float(cost[: t + 1].sum()),
            }
            S, a_used = level_fn(t, y_observed[: t + 1].copy(), censored[: t + 1].copy(), state)
            s_level[t] = S
            alpha_t[t] = a_used

            # 7. order placement
            inv_position = on_hand + on_order - backorders[t]
            qty = max(0.0, float(S) - inv_position)
            order_qty[t] = qty
            if qty > 0.0:
                arrival = t + cfg.L
                pipeline[arrival] = pipeline.get(arrival, 0.0) + qty

        log = pd.DataFrame(
            {
                "sku_id": sku_id,
                "period": np.arange(T, dtype="int32"),
                "arm": arm,
                "on_hand": on_hand_end,
                "backorders": backorders,
                "order_qty": order_qty,
                "demand_met": demand_met,
                "demand_lost": demand_lost,
                "csl_period": (demand_lost <= 0.0).astype("float64"),
                "fill_rate_period": np.divide(
                    demand_met, y_true, out=np.ones(T), where=y_true > 0
                ),
                "cost_period": cost,
                "s_level": s_level,
                "alpha_t": alpha_t,
                "warmup": np.arange(T, dtype="int32") < warmup,
            }
        )
        log["pis"] = _periods_in_stock(on_hand_end)
        log["nos"] = _number_of_stockouts(demand_lost)

        # --- headline metrics, over the EVALUATED window only --------------------
        # Periods before `warmup` cannot have stock: the first order is placed at the
        # end of period 0 and arrives at the start of period L. Including them adds a
        # constant, policy-independent deflation -- 2/200 on the synthetic panel, but
        # 2/36 on RUF's evaluated split, where it would be 5.5 % of every number.
        lost_eval = demand_lost[warmup:]
        met_eval, true_eval = demand_met[warmup:], y_true[warmup:]

        # A cycle is served iff NO demand was lost anywhere inside it. Cycles are
        # non-overlapping, length = the protection interval, starting after warm-up.
        cycle_len = cfg.protection
        n_cycles = int(lost_eval.size // cycle_len)
        if n_cycles:
            blocks = lost_eval[: n_cycles * cycle_len].reshape(n_cycles, cycle_len)
            cycle_csl = float((blocks.sum(axis=1) <= 0.0).mean())
        else:
            cycle_csl = float("nan")

        total_demand = float(true_eval.sum())
        return SimResult(
            sku_id=sku_id,
            log=log,
            cycle_service_level=cycle_csl,
            period_service_level=float((lost_eval <= 0.0).mean()),
            fill_rate=float(met_eval.sum() / total_demand) if total_demand > 0 else 1.0,
            total_cost=float(cost[warmup:].sum()),
            mean_cost=float(cost[warmup:].mean()),
            pis=float(log["pis"].iloc[-1]),
            nos=float(log["nos"].iloc[-1]),
            censored=censored,
            n_stockout=int((lost_eval > 0).sum()),
            n_cycles=n_cycles,
            warmup=int(warmup),
        )

    # -- synthetic demand draws -------------------------------------------- #

    def draw_demand(self, sku_id: str, t: int, p: float, k: float, mu_z: float) -> float:
        """One Bernoulli(p) x Gamma(k, mu_z/k) draw, seeded from (seed, sku, period).

        A derived seed, not a running stream, so forking the simulator for the oracle
        gives identical future draws across arms (Rev 2 §15.7 condition 2).
        """
        rng = np.random.default_rng(derive_seed(self.root_seed, "demand", sku_id, t))
        if rng.random() >= p:
            return 0.0
        return float(rng.gamma(shape=k, scale=mu_z / k))


# --------------------------------------------------------------------------- #
# Inventory metrics (Rev 2 §18)                                                #
# --------------------------------------------------------------------------- #


def _periods_in_stock(on_hand_end: np.ndarray) -> np.ndarray:
    """PIS -- cumulative fraction of periods with positive on-hand."""
    positive = (np.asarray(on_hand_end) > 0).astype("float64")
    return np.cumsum(positive) / np.arange(1, positive.size + 1)


def _number_of_stockouts(demand_lost: np.ndarray) -> np.ndarray:
    """NOS -- cumulative count of stock-out periods (Clements' number of stockouts)."""
    return np.cumsum((np.asarray(demand_lost) > 0).astype("float64"))
