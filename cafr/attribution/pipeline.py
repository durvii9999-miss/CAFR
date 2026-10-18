"""M2 -- the attribution pipeline (Rev 2 §6.3-§6.4, handoff §8).

Turns a :class:`~cafr.monitor.MonitorReading` into ``(cause, confidence, fired)``.
The rules live in ``rules.py`` and the ordering/confidence in ``priority.py``; this
module is the seam between them and owns three things they do not:

1. **The margin.** ``rules.py`` returns a boolean. The confidence formula needs a
   normalised strength, so each fired rule gets ``statistic / threshold`` -- monotone
   in the evidence, as §6.3 requires.
2. **The evidence floor.** A rule cannot fire on a statistic whose evidence floor was
   not met, even if the number happens to cross the threshold. An estimate from
   three observations is not evidence because it is large.
3. **Abstention as a first-class outcome.** C0 is not a failure; it is the honest
   answer when nothing fires or the confidence is below ``tau_conf``.

Priority (Rev 2 §6.4): C-guard -> C5 -> C1 -> C4 -> C2 -> C3 -> C6 -> C7 -> C0.
First match wins. Gate 5.4 re-runs with a shuffled order; if macro-F1 moves by more
than 0.10 the pipeline is measuring the ordering, not the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cafr.monitor.reading import MonitorReading

from .priority import CauseEvaluation, PRIORITY_ORDER, attribute_cause, compute_confidence
from .rules import (
    check_c1_bias,
    check_c2_coverage,
    check_c3_occurrence,
    check_c4_size,
    check_c5_cusum,
    check_c6_intermittency,
    check_c7_policy,
)

__all__ = ["Attribution", "attribute", "CAUSES"]


CAUSES: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


@dataclass(frozen=True)
class Attribution:
    cause: str                 # 'C0'..'C7'; 'C0' = abstain
    confidence: float
    fired: tuple[str, ...]
    margins: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "attributed_cause": self.cause,
            "confidence": float(self.confidence),
            "fired": ",".join(self.fired),
        }
        d.update({f"margin_{k}": float(v) for k, v in self.margins.items()})
        return d


def _num(reading: MonitorReading, key: str, default: float = float("nan")) -> float:
    v = reading.statistics.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def attribute(
    reading: MonitorReading,
    cfg: dict,
    *,
    order: list[str] | None = None,
    tau_conf: float | None = None,
) -> Attribution:
    """Attribute one reading. Never raises on a missing statistic; abstains instead."""
    det = cfg.get("detection", {}) if cfg else {}
    z_crit = float(det.get("z_b_crit", 1.96))
    tau_cov = float(det.get("tau_cov", 0.10))
    auc_thr = float(det.get("c3_auc_threshold", 0.65))
    fz_thr = float(det.get("c3_fz_threshold", 0.30))
    tau_occ = float(det.get("tau_occ", 0.10))
    r_thr = float(det.get("c4_r_threshold", 1.5))
    h_cusum = float(det.get("cusum_h", 4.0))
    stab_thr = float(det.get("c6_bootstrap_stability", 0.90))
    tau_svc = float(det.get("delta_svc", 0.10))
    tau = float(det.get("tau_conf", 0.35)) if tau_conf is None else float(tau_conf)

    s = reading.statistics
    ev = reading.evidence

    z_b = _num(reading, "c1_z_b")
    step_bic = _num(reading, "c1_step_bic")
    drift_bic = _num(reading, "c1_drift_bic")
    r_now = _num(reading, "c4_r")
    r_prev = _num(reading, "c4_r_prev")
    auc = _num(reading, "c3_auc")
    fz = _num(reading, "c3_fz")
    occ_gap = _num(reading, "c3_occ_gap")
    occ_p = _num(reading, "c3_occ_p_val")
    c_gap = _num(reading, "c2_c_gap")
    p_val = _num(reading, "c2_p_val")
    max_sz = _num(reading, "c5_sz_max_s", 0.0)
    max_occ = _num(reading, "c5_occ_max_s", 0.0)
    stability = _num(reading, "c6_stability")
    csl_gap = _num(reading, "c7_csl_gap")

    signs = reading.aux.get("signs", [])
    cells = reading.aux.get("cells", [])
    init_cell = reading.aux.get("init_cell", "dead")
    r_history = [r_prev, r_now]

    fired: dict[str, bool] = {}

    fired["C1"] = bool(ev.get("c1", False)) and check_c1_bias(
        z_b, list(signs), int(reading.aux.get("n_usable_sz", 0)),
        step_bic, drift_bic,
        min_evidence=int(cfg.get("monitor", {}).get("min_evidence", {}).get("c1_nonzero", 8)),
    )
    fired["C2"] = bool(ev.get("c2", False)) and check_c2_coverage(
        c_gap, p_val, z_b, r_now, tau_cov=tau_cov
    )
    fired["C3"] = bool(ev.get("c3", False)) and check_c3_occurrence(
        occ_gap, occ_p, r_now, tau_occ=tau_occ
    )
    fired["C4"] = bool(ev.get("c4", False)) and check_c4_size(r_history, auc)
    fired["C5"] = bool(ev.get("c5", False)) and check_c5_cusum(
        bool(s.get("c5_occ_alarm", 0.0)), bool(s.get("c5_sz_alarm", 0.0)),
        step_bic, drift_bic,
        _num(reading, "c5_occ_step_bic"), _num(reading, "c5_occ_drift_bic"),
    )
    fired["C6"] = bool(ev.get("c6", False)) and check_c6_intermittency(
        list(cells), str(init_cell), stability
    )
    fired["C7"] = bool(ev.get("c7", False)) and check_c7_policy(
        csl_gap, z_b, r_now, tau_svc=tau_svc
    )

    # Margins -- statistic / threshold. Only meaningful where a trigger statistic
    # exists; a cause that did not fire still reports its margin so the diagnostic
    # table shows *how close* it came (Step 5.5's statistic distributions).
    def _ratio(num: float, den: float) -> float:
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
            return 0.0
        return float(abs(num) / den)

    r_seen = [
        max(v, 1.0 / v) for v in (r_now, r_prev) if np.isfinite(v) and v > 0
    ]
    margins = {
        "C1": _ratio(z_b, z_crit),
        "C2": _ratio(c_gap, tau_cov),
        "C3": _ratio(occ_gap, tau_occ),
        "C4": _ratio(max(r_seen) if r_seen else float("nan"), r_thr),
        "C5": _ratio(max(max_sz, max_occ), h_cusum),
        "C6": _ratio(stability, stab_thr) if bool(s.get("c6_changed", 0.0)) else 0.0,
        "C7": _ratio(-csl_gap if np.isfinite(csl_gap) else float("nan"), tau_svc),
    }

    evaluations = {
        c: CauseEvaluation(fired=bool(fired[c]), margin=float(margins[c]))
        for c in CAUSES
    }
    cause, conf = attribute_cause(
        evaluations, tau_conf=tau, order=order if order is not None else list(PRIORITY_ORDER)
    )
    fired_order = tuple(
        c for c in (order if order is not None else PRIORITY_ORDER)
        if c in evaluations and evaluations[c].fired
    )
    # compute_confidence returns 1.0 for "nothing fired"; that is C0's confidence and
    # is kept, but it must not be mistaken for evidence of a cause.
    if cause == "C0":
        conf = min(conf, 1.0)
    return Attribution(
        cause=cause, confidence=float(conf), fired=fired_order, margins=margins
    )
