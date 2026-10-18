"""C6's intermittency cell and its bootstrap stability (Rev 2 §6.3, §15.1).

The cell is the **reachable** 2x2 of §15.1::

    ADI band  : adi_lo  (ADI <  ADI*)      adi_hi  (ADI >= ADI*)
    CV^2 band : cv2_lo  (CV^2 < CV^2*)     cv2_hi  (CV^2 >= CV^2*)

with ``ADI* = 4.0`` and ``CV^2* = 0.49`` read from ``detection`` in the config. The
textbook cut-off 1.32 is unreachable on a spare-parts panel (``ADI = 1/p >= 2``), so
a detector keyed to it could never fire -- that is why the reachable one is used.

``dead`` is a real cell, not an error: a window with fewer than two nonzero periods
has no CV^2. H-5 keeps those SKUs in the panel and lets C6 abstain for them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["adi_cv2_cell", "bootstrap_cell_stability"]


def _cells_from_matrix(Y: np.ndarray, adi_star: float, cv2_star: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised cell assignment for a ``(B, n)`` bootstrap matrix."""
    n = Y.shape[1]
    nz = Y > 0
    count = nz.sum(axis=1)
    adi = np.full(Y.shape[0], np.inf, dtype="float64")
    cv2 = np.full(Y.shape[0], np.nan, dtype="float64")

    ok = count >= 2
    if ok.any():
        adi[ok] = n / count[ok]
        # Mean and second moment over nonzero sizes only.
        Yn = np.where(nz, Y, 0.0)
        s1 = Yn.sum(axis=1)
        s2 = (Yn * Yn).sum(axis=1)
        mean = s1 / count
        mean_sq = s2 / count
        cv2[ok] = mean_sq[ok] / (mean[ok] ** 2) - 1.0

    adi_hi = adi >= adi_star
    cv2_hi = cv2 >= cv2_star
    # Encode: 0 dead, then bit1 = adi_hi, bit2 = cv2_hi.
    code = np.where(ok, 1 + adi_hi.astype(int) + 2 * cv2_hi.astype(int), 0)
    return code, adi, cv2


_CELL_NAMES = {
    0: "dead",
    1: "adi_lo_cv2_lo",
    2: "adi_hi_cv2_lo",
    3: "adi_lo_cv2_hi",
    4: "adi_hi_cv2_hi",
}


def adi_cv2_cell(
    y: np.ndarray,
    *,
    adi_star: float = 4.0,
    cv2_star: float = 0.49,
) -> dict:
    """Point estimate of the ADI / CV^2 cell for one window."""
    y = np.asarray(y, dtype="float64").reshape(1, -1)
    code, adi, cv2 = _cells_from_matrix(y, adi_star, cv2_star)
    return {
        "cell": _CELL_NAMES[int(code[0])],
        "adi": float(adi[0]) if np.isfinite(adi[0]) else float("nan"),
        "cv2": float(cv2[0]) if np.isfinite(cv2[0]) else float("nan"),
    }


def bootstrap_cell_stability(
    y: np.ndarray,
    *,
    n_boot: int,
    adi_star: float = 4.0,
    cv2_star: float = 0.49,
    rng: np.random.Generator | None = None,
) -> float:
    """Fraction of period-resamples that reproduce the point-estimate cell.

    Resampling *periods* (not nonzero sizes) is the correct bootstrap for this
    statistic: ADI is a function of where the zeros fall, so resampling only the
    nonzero sizes would condition away the very quantity whose stability is being
    measured.

    Returns ``nan`` when the window is empty or the point estimate is ``dead`` --
    a dead SKU has no cell to be stable about (H-5).
    """
    y = np.asarray(y, dtype="float64")
    if y.size == 0:
        return float("nan")

    point = adi_cv2_cell(y, adi_star=adi_star, cv2_star=cv2_star)
    if point["cell"] == "dead" or n_boot <= 0:
        return float("nan")

    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, y.size, size=(int(n_boot), y.size))
    codes, _, _ = _cells_from_matrix(y[idx], adi_star, cv2_star)
    return float(np.mean(codes == _CELL_NAMES_INV[point["cell"]]))


_CELL_NAMES_INV = {v: k for k, v in _CELL_NAMES.items()}
