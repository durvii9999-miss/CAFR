"""M3 -- remedies R0-R7. **Not implemented -- handoff §13 Step 7.**

One remedy per cause, matched one-to-one. A remedy REPAIRS a forecaster; it does not
replace it. That is the whole idea: the pool is fine, the failure is in one
sub-process, and the remedy touches only that sub-process.

Planned interface (handoff §7.3, §2.3)
--------------------------------------
::

    def apply(remedy: str, forecaster, state, history, cfg) -> tuple[Forecaster, dict]:
        '''Return the repaired forecaster and a dict recording exactly what changed.

        The dict is not optional. R7's whole design is "the forecast is untouched",
        and the only way to prove that in the paper is to log the before/after
        mu_hat and show they are identical.'''

The mapping
-----------
===== ============================ ==================================================
cause remedy                       what it must do
===== ============================ ==================================================
C0    R0 do nothing                abstain honestly; log that it abstained
C1    R1 damped intercept          lambda = 1.0 on the occurrence sub-process,
                                   lambda = 0.5 on the size sub-process
C2    R2 conformal recalibration   widen/narrow the interval to restore coverage;
                                   must NOT move the point forecast
C3    R3 switch the occurrence     trailing Bernoulli rate or TSB-style decay
      sub-model
C4    R4 re-estimate the size      refit on nonzero sizes only, recency-weighted
      model                        (damped trend)
C5    R5 discard pre-shift data    reset the state at the CUSUM alarm; requires the
                                   alarm to be logged
C6    R6 re-classify               re-derive the ADI x CV^2 cell; changes which
                                   classical model is used, not its parameters
C7    R7 adjust the safety factor  scales the SAFETY TERM ONLY -- mu_hat is untouched
===== ============================ ==================================================

Two hard rules
--------------
1. **R7 must not touch ``mu_hat``.** It changes ``alpha`` / the safety factor. If a
   future edit lets R7 move the point forecast, C7 becomes undetectable by
   construction (the C7 detector keys on the service gap while the forecast error is
   unchanged) and the C7 result becomes circular.
2. **R5 needs the CUSUM alarm.** It cannot be applied because someone "felt" a shift.

Doing-when: gates at handoff §13 Step 7, including the R7 no-mu-change assertion.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Step 7"
