"""Censoring guard (C-guard, Rev 2 §6.2)."""

from __future__ import annotations

import numpy as np

__all__ = ["c_guard"]

def c_guard(stockout_flag: np.ndarray) -> np.ndarray:
    """Return the boolean mask of usable (uncensored) periods.
    
    A period is usable if no stockout occurred (i.e. inventory > 0).
    stockout_flag: boolean array, True if stockout occurred.
    
    Returns:
        usable_mask: boolean array, True if period is usable.
    """
    return ~np.asarray(stockout_flag, dtype=bool)
