"""Synthetic baseline generator (Rev 2 §15.1).

Produces the base demand series without injections, plus the ADI x CV^2 helpers.

The cell definitions are the RUF builder's, deliberately: the synthetic and real
panels must be classified by the SAME rule or the two arms of the study would not be
comparable. ``ADI_STAR = 4.0`` and ``CV2_STAR = 0.49`` are the reachable boundaries
(§15.1) -- **not** the textbook 1.32, which a spare-parts panel can never cross.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "generate_baseline",
    "get_cell_params",
    "CELLS",
    "ADI_STAR",
    "CV2_STAR",
    "sb_cell",
    "adi",
    "cv_squared",
    "burn_in_stats",
]

ADI_STAR = 4.0
CV2_STAR = 0.49

# The four reachable ADI x CV2 cells
CELLS = [
    "moderate_lowdisp",
    "moderate_highdisp",
    "high_lowdisp",
    "high_highdisp",
]


def adi(y: np.ndarray) -> float:
    """Average inter-demand interval over the whole array."""
    y = np.asarray(y, dtype="float64")
    nz = int(np.count_nonzero(y > 0))
    return float(len(y) / nz) if nz else float(len(y))


def cv_squared(y: np.ndarray) -> float:
    """Squared coefficient of variation of the NONZERO sizes.

    ``NaN`` when fewer than two nonzero periods: the sample sd of one observation is
    undefined, and writing 0.0 would file the SKU into a ``_lowdisp`` cell it does not
    belong to. Such SKUs are ``dead`` and abstain by construction.
    """
    y = np.asarray(y, dtype="float64")
    nz = y[y > 0]
    if nz.size < 2 or nz.mean() <= 0:
        return float("nan")
    return float((nz.std(ddof=1) / nz.mean()) ** 2)


def burn_in_stats(y: np.ndarray, burn: int) -> tuple[float, float]:
    """``(adi_init, cv2_init)`` on the first ``burn`` periods ONLY (Rev 2 §27)."""
    w = np.asarray(y, dtype="float64")[:burn]
    nz = w[w > 0]
    a = float(len(w) / nz.size) if nz.size else float(len(w))
    c = float("nan")
    if nz.size >= 2 and nz.mean() > 0:
        c = float((nz.std(ddof=1) / nz.mean()) ** 2)
    return a, c


def sb_cell(a: float, c: float) -> str:
    """The reachable ADI-band x CV2-band 2x2 of §15.1."""
    if not np.isfinite(c):
        return "dead"
    band = "moderate" if a < ADI_STAR else "high"
    disp = "lowdisp" if c < CV2_STAR else "highdisp"
    return f"{band}_{disp}"


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


def draw_statics(rng: np.random.Generator) -> dict:
    """Per-SKU static attributes: lead time, unit cost, holding rate.

    RUF carries these in ``sku_attributes.parquet``, and the pooled LightGBM takes them
    as features (``STATIC_COLUMNS``). The synthetic panel needs its own, or the pooled
    model cannot be trained on it at all -- ``build_design_matrix`` raises on the
    missing columns, which is how the C7 pre-flight first failed.

    **Non-leaking, by construction.** These are drawn before the series and never from
    it, so they carry no information about the post-injection regime. ``adi_init`` and
    ``cv2_init`` -- the other two static columns -- come from the burn-in window and are
    computed in ``labels.py``, not here.

    ``lead_time`` is drawn from the config's swept set {1, 2, 4} (E6), so the panel
    covers the lead-time range the robustness experiment varies over.
    """
    return {
        "lead_time": float(rng.choice([1.0, 2.0, 4.0])),
        "unit_cost": float(np.round(rng.lognormal(mean=2.0, sigma=0.5), 2)),
        "holding_rate": float(np.round(rng.uniform(0.05, 0.30), 4)),
    }


def generate_baseline(
    T: int,
    p: float,
    k: float,
    mu_z: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate baseline intermittent demand.

    Returns ``(y, delta, z)`` arrays of length ``T``::

        delta ~ Bernoulli(p)
        z     ~ Gamma(shape=k, scale=mu_z/k)
        y     = delta * z
    """
    delta = rng.binomial(1, p, size=T).astype(float)
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    y = delta * z
    return y, delta, z
