"""M2 -- attribution. **Not implemented -- handoff §13 Step 6.**

Turns the monitor's statistics into ``(cause, confidence)`` by a priority-ordered rule
chain, with **abstention as a first-class outcome**. C0 means "no cause identified",
and it is not a failure: it is the honest answer when the evidence floor is not met.

Planned interface (handoff §8)
------------------------------
::

    @dataclass(frozen=True)
    class Attribution:
        cause: str          # 'C0'..'C7'; C0 = abstain
        confidence: float   # in [0, 1]
        fired: tuple[str, ...]   # which rules fired, in priority order

    def attribute(reading: MonitorReading, cfg) -> Attribution

The priority order (Rev 2 §6.4)
-------------------------------
``C-guard -> C5 -> C1 -> C4 -> C2 -> C3 -> C6 -> C7 -> C0``

First match wins. The order is a modelling choice, not a law, so gate 5.4 re-runs with
a shuffled order and requires the macro-F1 difference to be <= 0.10. If it is larger,
the pipeline is measuring the ordering rather than the detectors, and §6.4 is revised.

Confidence and the gate
-----------------------
``tau_conf = 0.35`` (``cfg['detection']['tau_conf']``). It is TUNED on train to hit the
<= 10 % false-positive design target, REPORTED on validation, and the test panel is
CONFIRMATORY ONLY. State that ordering in the paper -- tuning on test and then
reporting the test number is the failure mode this sentence exists to prevent.

Targets (Rev 2 §29, H2)
-----------------------
macro-F1 >= 0.70 over C1..C7, min per-class F1 >= 0.40, FP rate <= 10 % on C0.
C0 and the ambiguous panel are NEVER folded into the macro-F1 classes.

**Falsification condition.** If macro-F1 over C1..C7 is below 0.50, the mechanisms are
not separable, RQ1's answer is no, and that is a DIFFERENT paper. Say so plainly.

Doing-when: gates at handoff §13 Step 6.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Step 6"
