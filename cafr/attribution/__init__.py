"""M2 -- attribution. Turns monitor statistics into ``(cause, confidence)``.

Abstention (C0) is a first-class outcome: it is the honest answer when the evidence
floor is not met, not a failure. The pipeline lives in ``pipeline.py``; the rules and
the priority order are separate so gate 5.4 can re-run with a shuffled order without
touching the detectors.
"""

from __future__ import annotations

from .pipeline import Attribution, CAUSES, attribute
from .priority import PRIORITY_ORDER

__all__ = ["Attribution", "CAUSES", "attribute", "PRIORITY_ORDER"]
