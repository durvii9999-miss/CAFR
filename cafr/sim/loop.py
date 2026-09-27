"""The rolling-origin driver. **Not implemented -- handoff §13 Step 3.**

This is the module that turns the pieces below into a run. Nothing else in the
project loops over periods; every arm goes through here, which is what makes "all
arms face identical demand" true by construction rather than by assertion.

Why it is a separate file from ``cafr/sim/inventory.py``
--------------------------------------------------------
``inventory.py`` simulates ONE SKU against ONE level function. It knows nothing about
forecasters, monitors, remedies or arms. That separation is what lets gate 1.2 test the
simulator with a level function that is not a forecaster at all.

Planned interface (do not implement before reading handoff §13 Step 3)
----------------------------------------------------------------------
::

    @dataclass
    class RolloutConfig:
        arm: str                     # a | b | c | d | e | f | f_prime | g | h | A4
        fit_on: str
        burn_in: int
        horizons: tuple[int, ...]    # rolling origins
        context_columns: tuple[str, ...]   # whitelist, or a4_columns() for A4

    def rollout_sku(sku_id, y_true, arm, cfg, *, root_seed) -> RolloutResult:
        '''For each origin t:
             1. build FittingInput from y_observed[:t+1] and censored[:t+1]
             2. fit every pool member              (Step 2 -- DONE)
             3. monitor -> (cause, confidence)     (Step 5)
             4. remedy lookup -> action            (Step 7)
             5. build_policy_input(...)            (Step 0 -- DONE)
             6. arm selects an action              (Step 8/9)
             7. level = S from the remedied forecaster
             8. simulator step                     (Step 1 -- DONE)
        '''

Three rules this module must obey, all of them enforced elsewhere
-----------------------------------------------------------------
1. **No look-ahead.** The policy sees ``build_policy_input(...)`` and nothing else.
   The level function receives ``y_observed``/``censored`` only, by signature
   (``cafr/sim/inventory.py``, ``tests/test_no_lookahead.py``).
2. **Residuals come from the rolling loop.** ``e_t = y_t - mu_hat_t`` where
   ``mu_hat_t`` was made BEFORE ``y_t`` was seen. An in-sample residual understates
   the error and would make C2's coverage test pass on a mis-calibrated interval
   (``cafr/forecasters/base.py``, module docstring).
3. **One ``fit_on`` per run.** Resolved once, from the config, and logged
   (``tests/test_fit_on.py``, gate 2.5).

Doing-when (handoff §13 Step 3): gates 3.1-3.6 pass. Note gate 3.5 cannot pass on RUF
as written -- that is decision **H-1**, and it is escalated, not worked around.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Step 3"
