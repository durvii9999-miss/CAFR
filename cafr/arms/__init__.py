"""The ten arms. **Not implemented -- handoff §13 Steps 8-10.**

Every arm is a thin composition of the SAME four components in the SAME order. An arm
may not build its own feature frame, its own forecaster pool, or its own reward.
That is what makes the fairness table (handoff §8.2) true by construction instead of
by claim.

Planned interface
-----------------
::

    class Arm(Protocol):
        name: str
        context_columns: tuple[str, ...]   # from features.yaml, never local
        def choose(self, context, eligible, state, rng) -> Decision
        def decide(self, reading, attribution, state, cfg) -> int   # remedy index

The arms
--------
== ============================== =====================================================
   arm                            what differs
== ============================== =====================================================
a  fixed best classical           no adaptation at all (the floor)
b  Syntetos-Boylan rule           classical ADI x CV^2 -> model mapping, no learning
c  accuracy-based adaptive        **THE COMPARATOR.** Selects on forecast accuracy
d  EMC drift-triggered switching  switches on drift, not on cause
e  global LightGBM                the ML baseline (shares the pool's model)
f  CAFR full                      attribution in the context; the method
f' CAFR, constrained arms         (f) with a pruned action set
g  fixed hand-written table       the SAFE STOPPING POINT. If all else fails,
                                  (f) vs (g) is still a paper
h  no-R0                          (f) with abstention removed -- tests whether
                                  abstaining is doing the work
A4 attribution features removed   **CARRIES H1', THE CO-PRIMARY CLAIM**
== ============================== =====================================================

A4's four hard constraints (handoff §8.2) -- violating ANY invalidates H1'
----------------------------------------------------------------------------
1. same action space ``{R0..R7}``
2. same reward (the OD-3 standardised cost reward)
3. same action schedule (dwell, confidence gate, forgetting)
4. realised churn reported in T3 so the "matched churn" claim is checkable

A4 differs from (f) in EXACTLY ONE thing: the context block. ``tests/test_no_lookahead.py``
asserts the removed set is exactly ``{conf, cause_C0..cause_C7}`` -- nothing else.

Never handicap arm (c). It is the comparator; a weakened comparator makes H1 vacuous.

Churn (handoff §2.9)
--------------------
``chi_model`` is the STRICT one: it counts only a change in ``base_forecaster_family``.
``chi_action`` counts any remedy change. Both are reported. Forced-exploration
decisions are excluded from every churn count.
"""

from __future__ import annotations

__all__: list[str] = []

NOT_IMPLEMENTED = "handoff §13 Steps 8-10"
