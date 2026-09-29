"""Synthetic baseline generator (Rev 2 §15.1).

Produces the base demand series without injections.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generate_baseline", "get_cell_params"]

# The four reachable ADI x CV2 cells
CELLS = [
    "moderate_lowdisp",
    "moderate_highdisp",
    "high_lowdisp",
    "high_highdisp",
]

def get_cell_params(cell: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample (p, k, mu_z) for the given cell."""
    if "moderate" in cell:
        # ADI < 4 => p > 0.25
        p = rng.uniform(0.2501, 0.50)
    else:
        # ADI >= 4 => p <= 0.25
        p = rng.uniform(0.05, 0.25)

    if "lowdisp" in cell:
        # CV2 < 0.49 => k > 1/0.49 (2.0408)
        k = rng.uniform(2.05, 10.0)
    else:
        # CV2 >= 0.49 => k <= 1/0.49
        k = rng.uniform(0.5, 2.04)

    mu_z = rng.uniform(5.0, 200.0)
    return p, k, mu_z

def generate_baseline(
    T: int,
    p: float,
    k: float,
    mu_z: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate baseline intermittent demand.
    
    Returns (y, delta, z) arrays of length T.
    y = delta * z
    delta ~ Bernoulli(p)
    z ~ Gamma(shape=k, scale=mu_z/k)
    """
    delta = rng.binomial(1, p, size=T).astype(float)
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    y = delta * z
    return y, delta, z
