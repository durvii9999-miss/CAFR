"""Arm (a) -- fixed best classical method, chosen on the VALIDATION window.

**What it is:** For each SKU, score every classical method by rolling-origin
one-step-ahead MASE on the pre-evaluation history, take the argmin, and hold that
choice fixed for the entire evaluated window. No adaptation, no learning, no
revision after the first period.

**What it is NOT:** It does not switch models, does not use the bandit, does not
read the monitor, and does not respond to any signal after initialisation.

Role: the classical baseline -- the strongest *data-driven* member of current
practice, against which the SB rule (arm b) measures the value of a rule and the
attribution arms measure the value of attribution.

Where the validation window comes from (no look-ahead)
------------------------------------------------------
The simulator hands the level function ``y_observed[:t + 1]`` at period ``t``, and
the evaluated window starts at ``splits.burn_in``. So on the FIRST call the arm can
see exactly the burn-in history -- periods ``[0, burn_in)`` -- and nothing from the
evaluated window. The selection runs there, once, and is frozen. The window is
sliced to ``[:burn_in]`` explicitly, so the arm cannot see period ``burn_in`` itself
even though the simulator has already realised it.

Why MASE and not RMSE
---------------------
MASE divides by the in-sample naive error, so the score is comparable across SKUs
and across candidates on the same series. RMSE would favour whichever model
under-predicts least on a series whose scale is set by ``mu_z``, which varies over
two orders of magnitude across the panel.

Determinism
-----------
``forecaster.fit()`` is pure (every model in the pool fits through
``FittingInput.resolve()`` and returns a fresh ``ForecastState``), so scoring a
candidate here cannot contaminate the run that follows.
"""

from __future__ import annotations

import numpy as np

from ..forecasters.base import FittingInput, ForecastState
from ..sim.inventory import aggregate_order_up_to

__all__ = [
    "arm_a_factory",
    "select_by_validation",
    "CLASSICAL_CANDIDATES",
    "SB_FALLBACK",
]

#: The candidates arm (a) chooses among. The pooled LightGBM is deliberately NOT here:
#: it is the arm (c) forecaster, and letting arm (a) pick it would make (a) and (c)
#: differ only in the order-up-to rule rather than in the selection rule.
CLASSICAL_CANDIDATES: tuple[str, ...] = (
    "croston",
    "sba",
    "tsb",
    "msba",
    "mtsb",
    "ses_sizes",
)

#: SB-rule map used ONLY as the fallback when the validation window cannot decide
#: (fewer than ``min_train + 1`` periods, or a degenerate denominator). Same map as
#: arm (b), so a fallback SKU is comparable to arm (b) by construction.
SB_FALLBACK: dict[str, str] = {
    "moderate_lowdisp": "sba",
    "moderate_highdisp": "tsb",
    "high_lowdisp": "croston",
    "high_highdisp": "tsb",
    "dead": "croston",
}

#: Minimum fitting history before a candidate may be scored.
MIN_TRAIN = 8


def _mase_table(
    y_val: np.ndarray,
    censored_val: np.ndarray,
    pool,
    *,
    candidates: tuple[str, ...] = CLASSICAL_CANDIDATES,
    min_train: int = MIN_TRAIN,
    fit_on: str = "observed",
) -> tuple[dict[str, float], float, int]:
    """Rolling-origin one-step-ahead MASE for every candidate on ``y_val``.

    Returns ``(scores, denominator, n_origins)``. A score is ``nan`` when the
    candidate is absent from the pool or raised -- an absent score is never treated
    as a zero error.
    """
    n_val = int(y_val.size)
    origins = range(min_train, n_val)
    n_origins = max(0, n_val - min_train)

    scores: dict[str, float] = {m: float("nan") for m in candidates}
    if n_origins == 0:
        return scores, float("nan"), 0

    # The MASE denominator: the in-sample naive one-step error over the SAME origins,
    # so the scale is like-for-like with the numerator.
    naive = np.array(
        [abs(float(y_val[t]) - float(y_val[t - 1])) for t in origins], dtype="float64"
    )
    denom = float(naive.mean()) if naive.size else float("nan")

    if not np.isfinite(denom) or denom <= 0.0:
        # A constant (or all-zero) validation window: the naive scale is 0, so MASE is
        # undefined. Returning the raw MAE here would silently switch the ranking rule
        # for these SKUs, so the caller is told instead (``denominator`` is returned)
        # and falls back to the SB rule with the reason recorded.
        return scores, denom, n_origins

    for name in candidates:
        fc = pool.get(name)
        if fc is None:
            continue
        errs: list[float] = []
        for t in origins:
            try:
                state = fc.fit(
                    FittingInput(y_val[:t], censored_val[:t], fit_on=fit_on)
                )
            except Exception:  # a candidate that cannot fit is absent, not zero-error
                errs = []
                break
            errs.append(abs(float(y_val[t]) - float(state.mu)))
        if errs:
            scores[name] = float(np.mean(errs)) / denom
    return scores, denom, n_origins


def select_by_validation(
    y_val: np.ndarray,
    censored_val: np.ndarray,
    pool,
    *,
    sb_cell: str = "dead",
    candidates: tuple[str, ...] = CLASSICAL_CANDIDATES,
    min_train: int = MIN_TRAIN,
    fit_on: str = "observed",
) -> tuple[str, dict]:
    """Choose one method by validation MASE, or fall back to the SB rule.

    The returned log always records WHY a method was chosen, so a fallback can never
    be reported as a validation win.
    """
    scores, denom, n_origins = _mase_table(
        y_val, censored_val, pool, candidates=candidates, min_train=min_train, fit_on=fit_on
    )
    finite = {m: s for m, s in scores.items() if np.isfinite(s)}

    if not finite:
        # Most specific cause first: with no origins nothing was ever scored, and the
        # denominator is NaN as a CONSEQUENCE of that, not independently.
        if n_origins == 0:
            reason = "insufficient_validation"
        elif not np.isfinite(denom):
            reason = "degenerate_denominator"
        elif denom <= 0.0:
            reason = "zero_naive_scale"
        else:
            reason = "no_candidate_scored"
        method = SB_FALLBACK.get(sb_cell, "croston")
        return method, {
            "chosen": method,
            "reason": reason,
            "rule": "sb_fallback",
            "scores": scores,
            "denominator": denom,
            "n_origins": int(n_origins),
        }

    # Ties break on the candidate order, so the choice is deterministic.
    best = min(finite, key=lambda m: (finite[m], CLASSICAL_CANDIDATES.index(m)))
    return best, {
        "chosen": best,
        "reason": "validation_mase",
        "rule": "argmin_mase",
        "scores": scores,
        "denominator": denom,
        "n_origins": int(n_origins),
    }


class _ArmALevelFn:
    """Stateful level function for arm (a).

    The method is selected ONCE, on the first evaluated period, from the burn-in
    history; only its parameters are refit each period thereafter.
    """

    def __init__(
        self,
        pool,
        cfg,
        sku_id,
        alpha,
        fit_on,
        chosen_method,
        sb_cell,
        candidates=CLASSICAL_CANDIDATES,
    ):
        self.pool = pool
        self.cfg = cfg
        self.sku_id = sku_id
        self.alpha = alpha
        self.fit_on = fit_on
        self.chosen_method = chosen_method
        self.sb_cell = sb_cell
        self.candidates = candidates
        self.last_state: ForecastState | None = None
        #: The selection log, populated on the first call when no method was supplied.
        self.selection: dict | None = (
            {"chosen": chosen_method, "reason": "pre_selected", "rule": "external"}
            if chosen_method is not None
            else None
        )
        self._last_mu: float = float("nan")

    def _resolve_method(self, y_observed, censored) -> str:
        """Select on the burn-in window. Called exactly once."""
        burn_in = int(self.cfg.get("splits", {}).get("burn_in", y_observed.size))
        # ``[:burn_in]`` and not ``[:t]``: the simulator has already appended period
        # ``t``'s demand to ``y_observed``, and period ``t`` is the first EVALUATED
        # period. Slicing by the burn-in length is what makes the selection strictly
        # pre-evaluation.
        n_val = int(min(burn_in, y_observed.size))
        method, log = select_by_validation(
            np.asarray(y_observed[:n_val], dtype="float64"),
            np.asarray(censored[:n_val], dtype=bool),
            self.pool,
            sb_cell=self.sb_cell,
            candidates=self.candidates,
            fit_on=self.fit_on,
        )
        self.selection = log
        return method

    def __call__(self, t, y_observed, censored, inventory):
        if len(y_observed) == 0:
            self.last_state = None
            return 1.0, self.alpha

        if self.chosen_method is None:
            self.chosen_method = self._resolve_method(y_observed, censored)

        method = self.chosen_method
        fc = self.pool[method]
        fi = FittingInput(y_observed, censored, fit_on=self.fit_on)
        state = fc.fit(fi)
        self.last_state = state
        state.method = method

        self._last_mu = float(state.mu)

        R = int(self.cfg["sim"]["R"])
        S, alpha_eff, _, _ = aggregate_order_up_to(
            y_hist=y_observed,
            mu_hat=state.mu,
            R=R,
            alpha=self.alpha,
            window=int(self.cfg["sim"].get("quantile_window", 24)),
            min_aggregates=int(self.cfg["sim"].get("min_aggregates", 4)),
            method=self.cfg["sim"].get("quantile_method", "linear"),
            # C7's injection scales the SAFETY FACTOR only, never the forecast (§15.2).
            safety_factor_scale=float(self.cfg["sim"].get("c7_safety_factor_scale", 1.0)),
        )
        return float(S), float(alpha_eff)


def arm_a_factory(
    pool,
    cfg,
    sku_id,
    alpha,
    fit_on,
    *,
    chosen_method=None,
    sb_cell="dead",
    candidates=CLASSICAL_CANDIDATES,
):
    """Return a stateful level function for arm (a).

    Parameters
    ----------
    chosen_method
        Pre-selected model name. If None (the normal path), the method is selected
        by validation MASE on the burn-in window at the first call.
    sb_cell
        The burn-in SB cell, used ONLY if the validation window cannot decide.
    """
    return _ArmALevelFn(
        pool, cfg, sku_id, alpha, fit_on, chosen_method, sb_cell, candidates
    )
