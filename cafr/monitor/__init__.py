"""M1 -- the outcome monitor. **Not implemented -- handoff §13 Step 5.**

Reads the observed history and emits one statistic per cause, plus an evidence flag.
It NEVER decides anything: the decision is M2's (attribution) and M4's (policy). The
monitor only measures.

Planned interface (handoff §6.1, §6.4)
--------------------------------------
::

    @dataclass(frozen=True)
    class MonitorReading:
        statistics: dict[str, float]   # one entry per detector statistic
        evidence:   dict[str, bool]    # is each detector's evidence floor met?
        window_full: bool              # H-2: was W_occ full at this decision point?
        abstain_reason: str | None     # set when the evidence floor is not met

    def read(sku_features, cfg, t) -> MonitorReading

Every window is a config parameter (``cfg['monitor']['windows']``); none may be
hard-coded. The floors are in ``cfg['monitor']['min_evidence']``.

The seven detectors and their statistic
---------------------------------------
===== ============================== ==========================================
cause statistic                      must NOT move (gate 4.4)
===== ============================== ==========================================
C1    sign-stability of the size     dispersion, occurrence AUC
      forecast error over W_short
C2    empirical coverage of the      mean error (a mis-calibrated interval can
      one-period interval over W_cal have a perfect mean)
C3    occurrence AUC / false-zero    size dispersion
      rate over W_occ
C4    size-dispersion ratio r over   mean level
      W_sz, sustained 2 windows
C5    CUSUM on the size residual     --
      (NEVER merged back with the C6 CUSUM)
C6    ADI x CV^2 cell crossing at    mean level
      ADI* = 4.0, CV^2* = 0.49
C7    service gap |CSL_hat - target| forecast error (R7 must not touch mu_hat)
===== ============================== ==========================================

The priority order lives in Rev 2 §6.4 -- C-guard, C5, C1, C4, C2, C3, C6, C7, C0 --
and gate 5.4 checks the result is not an artifact of that order.

**ADI* = 4.0, not the textbook 1.32.** A spare-parts panel has ADI >= 2 by
construction, so 1.32 is unreachable and a detector keyed to it can never fire.

Doing-when (handoff §13 Step 5): gates 5.1-5.6. Per-cause recall >= 0.60 is BELOW
H2's 0.70 macro-F1 bar on purpose -- Step 5 is a go/no-go on the injections, not the
final result. Do not raise it. If Step 5 fails, revise §15.2 of the spec, NOT
``attribution/rules.py``.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Step 5"
