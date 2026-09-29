"""Arm (g): Fixed remedy table (Rev 2 §30)."""

from __future__ import annotations

import numpy as np

# We'll just define a placeholder for arm_g_factory right now 
# since it's quite complex to wire up the entire monitor loop inside the step function.

def arm_g_factory(pool, cfg, sku_id, alpha_target, fit_on, margin=0.0):
    """Arm (g) runs a baseline (e.g. arm c) but intervenes using R0-R7 when causes are attributed."""
    # To be fully implemented once the monitor loop state extraction is stable.
    # The requirement is that it returns a level_fn just like arm_c.
    pass
