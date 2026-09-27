"""Lost-sales imputation for ``fit_on = uncensored``. Rev 2 §14.3 rule 6, handoff §2.11.

**The signature rule, which is the whole point of this module.**
The imputation must be derivable from OBSERVABLES ONLY. It may not receive
``demand_lost`` or ``demand_true``. If it did, ``fit_on = uncensored`` would stop
being the "realistic middle path" and become a leak: the forecaster would be fitting
on the answer.

There is therefore no parameter here that could carry the truth. The ``censored``
mask is the C-guard's flag -- computed from the inventory position reaching zero,
which the policy can see -- not from a recorded lost-sales quantity.

```python
test_fit_on_imputation_signature_refuses_ground_truth()   # tests/test_fit_on.py
```
asserts that no parameter name here contains ``true`` / ``lost`` / ``cause``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["impute_lost_sales", "ImputationResult"]


class ImputationResult:
    """Imputed series plus the audit trail the paper needs."""

    __slots__ = ("y", "n_censored", "n_imputed", "fallback")

    def __init__(self, y: np.ndarray, n_censored: int, n_imputed: int, fallback: bool):
        self.y = y
        self.n_censored = n_censored
        self.n_imputed = n_imputed
        # True when the SKU had no uncensored nonzero history to sample from, so
        # nothing could be imputed. Reportable, not an error.
        self.fallback = fallback

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ImputationResult(n_censored={self.n_censored}, "
            f"n_imputed={self.n_imputed}, fallback={self.fallback})"
        )


def impute_lost_sales(
    y_observed: np.ndarray,
    censored: np.ndarray,
    rng: np.random.Generator,
) -> ImputationResult:
    """Impute lost demand on censored periods from the SKU's OWN observable history.

    Parameters
    ----------
    y_observed
        The series as the policy sees it: met demand only.
    censored
        Boolean mask, True where the inventory position reached zero during the
        period (the C-guard's flag). Derived from observables.
    rng
        Seeded per ``(panel_seed, sku_id, period)``. Never a shared running stream.

    Method
    ------
    Draw the lost quantity for a censored period from that SKU's empirical
    nonzero-size distribution, **estimated on uncensored periods only**, and add it
    to the observed amount. A stock-out period with zero observed demand is the
    severest case and receives a full draw; a partially-met period receives one
    additional draw, which is the standard "at least one further demand event"
    reading.

    The estimate is taken on a trailing basis by the caller (see
    ``sim.loop``), so no future period informs the imputation.
    """
    y_observed = np.asarray(y_observed, dtype="float64")
    censored = np.asarray(censored, dtype=bool)
    if y_observed.shape != censored.shape:
        raise ValueError("y_observed and censored must have the same shape")

    n_censored = int(censored.sum())
    y = y_observed.copy()
    if n_censored == 0:
        return ImputationResult(y, 0, 0, fallback=False)

    # Empirical nonzero-size distribution, uncensored periods only.
    usable = (~censored) & (y_observed > 0)
    sizes = y_observed[usable]

    if sizes.size == 0:
        # No uncensored nonzero history. Nothing can be imputed from observables.
        # Leave the series unchanged and say so; do NOT reach for the truth.
        return ImputationResult(y, n_censored, 0, fallback=True)

    idx = np.flatnonzero(censored)
    draws = rng.choice(sizes, size=idx.size, replace=True)
    y[idx] = y_observed[idx] + draws
    return ImputationResult(y, n_censored, int(idx.size), fallback=False)
