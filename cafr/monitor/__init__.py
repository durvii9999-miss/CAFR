"""M1 -- the outcome monitor (Rev 2 §6.1-§6.4, handoff §6.1, §6.4).

Reads the observed history and emits one statistic per cause plus an evidence flag.
It NEVER decides anything: the decision is M2's (attribution) and M4's (policy). The
monitor only measures.

Interface
---------
::

    mon = Monitor(cfg, alpha_target=0.95)
    mon.update(t, y_t, censored_t, {"mu": mu_hat, "z": z_hat,
                                    "lo": lo, "hi": hi, "alpha": 0.90},
               stockout_t=stockout)
    reading = mon.read()

Every window is a config parameter (``monitor.windows``) and every floor is in
``monitor.min_evidence``. A missing key is surfaced in
``reading.aux["config_warnings"]``, never silently defaulted into a reported number.

The seven detectors and their statistic
---------------------------------------
===== ==================================== ======================================
cause statistic                            must NOT move (gate 4.4)
===== ==================================== ======================================
C1    pooled bias ``b^sz`` and ``z_b``     dispersion
C2    empirical one-period coverage         mean error
C3    occurrence AUC / false-zero rate      size dispersion
C4    log-scale dispersion ratio, 2 windows mean level
C5    two CUSUMs (size AND occurrence)      --
C6    ADI x CV^2 cell, sustained 3 windows  mean level
C7    CSL gap vs the policy's alpha         forecast error
===== ==================================== ======================================

**ADI* = 4.0, not the textbook 1.32.** A spare-parts panel has ADI >= 2 by
construction, so 1.32 is unreachable and a detector keyed to it can never fire.

Step 5 gates (handoff §13 Step 5): 5.1-5.6. Per-cause recall >= 0.60 is BELOW H2's
0.70 macro-F1 bar on purpose -- Step 5 is a go/no-go on the injections, not the final
result. If Step 5 fails, revise §15.2 of the spec, NOT ``attribution/rules.py``.
"""

from __future__ import annotations

from .censoring import c_guard
from .reading import Monitor, MonitorReading

__all__ = ["Monitor", "MonitorReading", "c_guard"]
