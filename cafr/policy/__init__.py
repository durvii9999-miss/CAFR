"""M4 -- the bandit policy. **Not implemented -- handoff §13 Step 8.**

Thompson sampling over a **linear contextual** model, **ONE global policy** shared
across all SKUs. A per-SKU policy would be 5,000 separate bandits, each with a handful
of updates -- which is not a policy, it is noise.

Planned interface (handoff §7.8, §2.10)
---------------------------------------
::

    @dataclass
    class BanditState:
        mu: np.ndarray      # posterior mean, one row per arm
        precision: np.ndarray

    def select(state, context, eligible_arms, rng) -> int
    def update(state, arm, context, reward) -> BanditState
    def inflate_on_alarm(state, scale) -> BanditState

The reward (OD-3, FROZEN -- do not redesign mid-experiment)
-----------------------------------------------------------
::

    r = -[(H * I_bar + B * B_bar) - C_bar_i] / sigma_{C,i}

with the standardisation floor ``sigma = max(sigma, 0.05 * C_bar_i)``. Without the
floor the reward explodes for SKUs with near-constant cost, and those updates dominate
the posterior for every other SKU.

**The reward is never accuracy.** Accuracy is not the objective; cost is. A policy
rewarded on accuracy will chase point-forecast precision and lose on the frontier,
which is the exact confusion the paper exists to correct.

Forgetting is not optional
--------------------------
C5 and C6 are non-stationarity injections. A plain Thompson sampler's posterior
variance collapses as it accumulates evidence -- precisely when the world changes.

``forgetting: cusum_alarm``: on a C5 CUSUM alarm, inflate the posterior variance. No
extra RNG, no decay schedule to tune, and it uses a signal the monitor already
produces.

**HARD REQUIREMENT (handoff §8.2): the forgetting mechanism must be IDENTICAL on arms
(f), (f') and A4.** If A4 adapts at a different speed, H1' measures adaptation speed
instead of attribution, and the co-primary result is void. This is unit-testable and
must be tested.

Exploration
-----------
``staggered``: for ``{i : hash(i + t) mod 9 == 0}``, force a uniform-random arm.
Forced-exploration decisions are EXCLUDED from every churn count -- otherwise the
policy is penalised for exploring, which is the mechanism that makes it work.

Doing-when: gates at handoff §13 Step 8. Arm (g), the fixed hand-written table, is the
SAFE STOPPING POINT: if everything after Step 8 fails, (f) vs (g) is still a paper.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Step 8"
