"""The DGP-aware oracle forecaster -- gate 1.2's positive half.

Rev 2 §28.6 Part (i) requires *"a forecaster that supplies the TRUE predictive
distribution of R-period aggregate demand (NOT a zero-error point forecaster)"*.

That distinction is the whole point of the rewritten gate. A **zero-error** forecaster
has ``sigma_hat = 0``, hence safety stock ``= 0``, hence ``S = R * mu_hat``; demand
exceeds the mean in about half of all cycles, so achieved CSL lands near 0.50 against a
target of 0.80-0.95. The original gate would have failed **by construction**, and
because §30 makes it blocking, the project would have stopped at Step 1 for a reason
that was not real. The idealisation was wrong, not the simulator.

The DGP is the §15.1 baseline process::

    occurrence:  delta_t ~ Bernoulli(p)
    size:        z_t     ~ Gamma(shape = k, scale = mu_z / k)
    demand:      y_t     = delta_t * z_t

The R-period aggregate ``A = sum_{j=1..R} delta_j z_j`` has the exact CDF

    P(A <= s) = sum_{n=0}^{R} C(R,n) p^n (1-p)^(R-n) * F_{Gamma(nk, mu_z/k)}(s)

with ``F_{Gamma(0, .)}`` the point mass at 0. ``quantile()`` inverts that numerically,
so the oracle supplies the true distribution rather than an estimate of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = ["BernoulliGammaDGP", "oracle_level_function", "zero_safety_level_function"]


@dataclass(frozen=True)
class BernoulliGammaDGP:
    """The §15.1 baseline process. Ground truth, known exactly."""

    p: float
    k: float
    mu_z: float

    @property
    def mean(self) -> float:
        """``E[y] = p * mu_z``."""
        return self.p * self.mu_z

    @property
    def cv2(self) -> float:
        """``CV^2 = 1/k`` for Gamma(k, theta) -- exactly, not approximately."""
        return 1.0 / self.k

    @property
    def adi(self) -> float:
        """``E[ADI] = 1/p`` under i.i.d. Bernoulli(p)."""
        return 1.0 / self.p

    def aggregate_cdf(self, s: float, R: int) -> float:
        """Exact CDF of the R-period aggregate demand at ``s``."""
        if s < 0:
            return 0.0
        total = 0.0
        for n in range(R + 1):
            weight = math.comb(R, n) * self.p**n * (1.0 - self.p) ** (R - n)
            if weight == 0.0:
                continue
            if n == 0:
                total += weight              # point mass at 0: P(A <= s) = 1 for s >= 0
            else:
                total += weight * stats.gamma.cdf(s, a=n * self.k, scale=self.mu_z / self.k)
        return float(min(max(total, 0.0), 1.0))

    def aggregate_quantile(self, alpha: float, R: int) -> float:
        """Invert the exact aggregate CDF at ``alpha``."""
        alpha = float(min(max(alpha, 0.0), 1.0))

        # A point mass at 0 means the quantile is 0 for every alpha up to
        # P(A = 0) = (1 - p)^R. Handle that exactly rather than by search.
        if alpha <= (1.0 - self.p) ** R:
            return 0.0

        upper = max(self.mu_z, 1.0) * R
        while self.aggregate_cdf(upper, R) < alpha:
            upper *= 2.0
            if upper > 1e12:                  # pragma: no cover - pathological input
                raise RuntimeError("aggregate quantile did not bracket")
        return float(optimize.brentq(lambda s: self.aggregate_cdf(s, R) - alpha, 0.0, upper))


def oracle_level_function(dgp: BernoulliGammaDGP, R: int, alpha: float):
    """A ``LevelFunction`` supplying the TRUE R-period aggregate quantile.

    This is what gate 1.2 Part (i) means by "correctly-set policy".
    """

    level = dgp.aggregate_quantile(alpha, R)

    def fn(t, y_observed, censored, inventory):
        return float(level), float(alpha)

    fn.__doc__ = f"oracle level: exact {alpha:.3f}-quantile of the {R}-period aggregate"
    return fn


def zero_safety_level_function(dgp: BernoulliGammaDGP, R: int, alpha: float):
    """Safety stock set to zero: ``S = R * mu_hat`` with the TRUE mean.

    Gate 1.2 Part (ii). This half is what proves the test CAN fail -- a simulator that
    ignored ``alpha`` entirely could pass a positive-only test at one ``alpha`` by
    accident, but it cannot pass this.
    """
    level = R * dgp.mean

    def fn(t, y_observed, censored, inventory):
        return float(level), float(alpha)

    fn.__doc__ = "negative control: S = R * mu_hat, safety stock removed"
    return fn
