"""The arms (Rev 2 §12).

Baseline and CAFR policy arms for the inventory simulation.
"""

from .arm_a import arm_a_factory
from .arm_b import arm_b_factory
from .arm_c import arm_c_factory
from .arm_e import arm_e_factory

__all__ = [
    "arm_a_factory",
    "arm_b_factory",
    "arm_c_factory",
    "arm_e_factory",
]
