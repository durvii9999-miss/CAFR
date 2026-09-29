"""Attribution priority logic (M2, Rev 2 §6.4)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

__all__ = ["attribute_cause"]

# The mandated evaluation priority (Rev 2 §6.4)
# C4 precedes C2. C7 is last.
PRIORITY_ORDER = [
    "C5",
    "C1",
    "C4",
    "C2",
    "C3",
    "C6",
    "C7"
]

@dataclass
class CauseEvaluation:
    fired: bool
    margin: float

def compute_confidence(evaluations: Dict[str, CauseEvaluation]) -> float:
    """Compute confidence score.
    
    conf = min(1, max_over_fired_causes(|statistic| / threshold) - 0.5*(number of causes that fired))
    """
    fired_causes = {k: v for k, v in evaluations.items() if v.fired}
    n_fired = len(fired_causes)
    
    if n_fired == 0:
        return 1.0 # C0 confidence
        
    max_margin = max([v.margin for v in fired_causes.values()])
    conf = min(1.0, max(0.0, max_margin - 0.5 * n_fired))
    return float(conf)

def attribute_cause(
    evaluations: Dict[str, CauseEvaluation],
    tau_conf: float = 0.35,
    order: list[str] = None
) -> Tuple[str, float]:
    """Determine the final attributed cause using the priority order.
    
    Returns (cause_label, confidence_score)
    """
    if order is None:
        order = PRIORITY_ORDER
        
    conf = compute_confidence(evaluations)
    
    if conf < tau_conf:
        return "C0", conf
        
    for cause in order:
        ev = evaluations.get(cause)
        if ev and ev.fired:
            return cause, conf
            
    return "C0", conf
