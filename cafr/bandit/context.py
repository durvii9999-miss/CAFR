"""Bandit context construction (M4)."""

from __future__ import annotations
import numpy as np

def build_context(
    inv_features: dict[str, float],
    det_features: dict[str, float],
    use_det: bool = True
) -> np.ndarray:
    """Builds the context vector x_{i,t} = [x^inv_{i,t}, x^det_{i,t}].
    
    If use_det is False, returns only x^inv (for arm A4).
    """
    
    # Extract inventory/demand features (x^inv)
    # E.g. gap, fill rate, on_hand, backorders, ADI, CV2, mean_sz, frac_zero
    inv_vec = [
        inv_features.get("csl_gap", 0.0),
        inv_features.get("fill_rate", 0.0),
        inv_features.get("on_hand", 0.0),
        inv_features.get("backorders", 0.0),
        inv_features.get("adi", 1.0),
        inv_features.get("cv2", 1.0),
        inv_features.get("mean_sz", 1.0),
        inv_features.get("frac_zero", 0.0),
    ]
    
    # Extract detection features (x^det)
    # E.g. z_b, r, auc, fz, c_gap, conf
    det_vec = [
        det_features.get("z_b", 0.0),
        det_features.get("r", 1.0),
        det_features.get("auc", 0.5),
        det_features.get("fz", 0.0),
        det_features.get("c_gap", 0.0),
        det_features.get("conf", 0.0),
    ]
    
    # One-hot encoded attributed cause (C0-C7)
    cause = det_features.get("cause", "C0")
    cause_oh = [1.0 if cause == f"C{i}" else 0.0 for i in range(8)]
    det_vec.extend(cause_oh)
    
    if use_det:
        return np.array(inv_vec + det_vec, dtype=float)
    else:
        return np.array(inv_vec, dtype=float)
