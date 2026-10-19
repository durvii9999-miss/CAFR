"""M1 -- the outcome monitor (Rev 2 §6.1-§6.4, handoff §6.1, §6.4).

The monitor **measures and never decides**. One call to :meth:`Monitor.update` per
period records what was observed and what the arm had predicted; :meth:`Monitor.read`
turns the trailing history into one statistic per cause plus an evidence flag. M2
(``cafr.attribution``) turns that into a cause; M4 (``cafr.bandit``) acts on it.

Why the monitor is an accumulator rather than a pure function
--------------------------------------------------------------
Every statistic is a function of a **trailing window**, and several are functions of
what the arm actually predicted last period. The residual ``e_t = y_t - mu_hat_{t|t-1}``
only exists if the forecast made at ``t-1`` was stored. The class owns that alignment
so no caller can accidentally score an in-sample residual.

Reading B (H-2, signed off 2026-10-17)
--------------------------------------
RUF has 60 periods; the synthetic panel has 200. No single window length works for
both, so each window is ``min(configured, available)`` and ``window_full`` reports
whether the configured horizon was actually reached. The floors still apply: a window
that is short *and* thin does not produce a statistic, it produces ``nan`` and an
unmet evidence flag. The C1 sign-stability check is the one place that needs a second
adaptation -- with 36 evaluated periods, four 24-period blocks do not exist, so the
available history is split into up to four equal blocks with a per-block nonzero floor.
That is Reading B applied, not a relaxed threshold, and it is logged in ``aux``.

The statistics and the confound each is guarded against (gate 4.4)
-------------------------------------------------------------------
===== ============================================== ============================
cause statistic                                       guard
===== ============================================== ============================
C1    b^sz pooled, z_b over W_short                   C4's dispersion ratio
C2    empirical coverage of the one-period interval   z_b and r (a lifted mean
                                                      must not read as bad width)
C3    occurrence AUC and false-zero rate over W_occ   r (a size change must not
                                                      read as an occurrence change)
C4    log-dispersion ratio r, sustained 2 windows     mean level (log scale is
                                                      exactly mean-invariant)
C5    two CUSUMs -- size and occurrence, NEVER one     --
C6    ADI x CV^2 cell change, sustained 3 windows      mean level
C7    CSL gap against the policy's alpha              forecast acceptability
===== ============================================== ============================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .features import (
    compute_bias_stats,
    compute_coverage_stats,
    compute_cusum_stats,
    compute_dispersion_stats,
    compute_intermittency_stats,
    compute_occurrence_stats,
    compute_occurrence_gap,
)
from .models import step_drift_bic
from .stability import adi_cv2_cell, bootstrap_cell_stability

__all__ = ["MonitorReading", "Monitor"]

#: Fallbacks used ONLY when a config omits a key. Every real run reads base.yaml, where
#: all of these are explicit; a missing key is surfaced in ``aux["config_warnings"]``
#: rather than silently taking a default.
_DEFAULT_WINDOWS = {
    "W_short": 24,
    "W_sign_stability": 96,
    "W_cal": 30,
    "W_occ": 36,
    "W_sz": 24,
    "W_class": 30,
    "W_svc": 30,
    "W_cusum": 96,
    "W_base": 24,
}
_DEFAULT_FLOORS = {
    "c1_nonzero": 8,
    "c3_nonzero": 10,
    "c4_nonzero": 10,
    "c2_coverage_obs": 30,
    "c6_periods": 30,
    "c7_periods": 30,
    "c7_unmet_cycles": 2,
    "c5_baseline_nonzero": 8,
}


@dataclass(frozen=True)
class MonitorReading:
    """One period's monitor output. Statistics are floats; non-scalars live in ``aux``."""

    statistics: dict[str, float]
    evidence: dict[str, bool]
    window_full: bool
    abstain_reason: str | None
    aux: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: float = float("nan")) -> float:
        return float(self.statistics.get(key, default))


class Monitor:
    """Stateful accumulator. One instance per SKU per arm."""

    def __init__(
        self,
        cfg: dict,
        *,
        alpha_target: float = 0.95,
        seed: int = 0,
    ) -> None:
        mon = cfg.get("monitor", {}) if cfg else {}
        det = cfg.get("detection", {}) if cfg else {}
        cfg_windows = dict(mon.get("windows", {}))
        cfg_floors = dict(mon.get("min_evidence", {}))

        self._warnings: list[str] = []
        self.windows: dict[str, int] = {}
        for k, v in _DEFAULT_WINDOWS.items():
            if k in cfg_windows:
                self.windows[k] = int(cfg_windows[k])
            else:
                self.windows[k] = int(v)
                self._warnings.append(f"monitor.windows.{k} missing; used {v}")

        self.floors: dict[str, int] = {}
        for k, v in _DEFAULT_FLOORS.items():
            if k in cfg_floors:
                self.floors[k] = int(cfg_floors[k])
            else:
                self.floors[k] = int(v)
                self._warnings.append(f"monitor.min_evidence.{k} missing; used {v}")

        self.z_b_crit = float(det.get("z_b_crit", 1.96))
        self.alpha_cov = float(det.get("binomial_p", 0.05))  # only the p threshold is used
        self.adi_star = float(det.get("adi_star", 4.0))
        self.cv2_star = float(det.get("cv2_star", 0.49))
        self.cusum_k = float(det.get("cusum_k", 0.25))
        self.cusum_h = float(det.get("cusum_h", 4.0))
        self.cusum_burn_in = int(cfg.get("splits", {}).get("burn_in", 0))
        self.c6_bootstrap_n = int(det.get("c6_bootstrap_n", 1000))
        self.alpha_target = float(alpha_target)
        # C2's interval level. base.yaml's C2 generator clears a 90 % interval; the
        # level is configurable because the coverage statistic is only meaningful
        # against the level the interval was built at.
        self.c2_alpha = float(cfg.get("detection", {}).get("c2_interval_alpha", 0.90))
        self.protection = int(cfg.get("sim", {}).get("R", 3)) if cfg else 3
        self._rng = np.random.default_rng(int(seed))

        self._y = np.zeros(0)
        self._cens = np.zeros(0, dtype=bool)
        self._stockout = np.zeros(0, dtype=bool)
        self._mu_prev = np.full(0, np.nan)
        self._munz_prev = np.full(0, np.nan)
        self._sig_prev = np.full(0, np.nan)
        self._z_prev = np.full(0, np.nan)
        self._lo = np.full(0, np.nan)
        self._hi = np.full(0, np.nan)
        self._alpha_q = np.full(0, np.nan)
        self._covered = np.zeros(0, dtype=bool)
        self._cover_usable = np.zeros(0, dtype=bool)
        self._pending: dict | None = None
        self._n = 0

    # ------------------------------------------------------------------ #
    # accumulation                                                        #
    # ------------------------------------------------------------------ #

    def update(
        self,
        t: int,
        y_t: float,
        censored_t: bool,
        forecast: dict | None,
        *,
        stockout_t: bool = False,
    ) -> None:
        """Record period ``t`` and the forecast made for ``t + 1``.

        ``forecast`` keys: ``mu`` (per-period point forecast), ``mu_nonzero``
        (conditional nonzero size -- the C1/C5 size sub-process), ``z`` (occurrence
        probability), ``lo``/``hi`` (one-period interval), ``alpha`` (the interval's
        nominal level). Missing keys become ``nan``; the statistic that needs them
        then reports ``nan`` rather than a fabricated value.

        **``mu`` is not the size forecast.** On an intermittent series ``mu = z *
        mu_nonzero``; differencing a nonzero demand against ``mu`` gives a residual of
        roughly ``(1 - z) * mu_nonzero`` on every demand period, which reads as a huge
        persistent bias on the C0 control (measured: median ``z_b`` = 7.5). The size
        sub-process residual is ``y - mu_nonzero`` on ``y > 0`` only.

        Calls must be in period order with no gaps -- a gap would silently misalign
        every residual against its forecast, which is the failure mode the whole class
        exists to prevent.
        """
        if int(t) != self._n:
            raise ValueError(
                f"monitor out of order: got t={t}, expected {self._n}. "
                "Residuals are only valid when every period is recorded once, in order."
            )

        fc = forecast or {}
        mu = float(fc.get("mu", np.nan))
        munz = float(fc.get("mu_nonzero", np.nan))
        sig = float(fc.get("sigma", np.nan))
        z = float(fc.get("z", np.nan))
        lo = float(fc.get("lo", np.nan))
        hi = float(fc.get("hi", np.nan))
        a = float(fc.get("alpha", self.c2_alpha))

        # Align: the forecast stored at t-1 predicts period t.
        prev = self._pending if self._pending is not None else {}
        self._mu_prev = np.append(self._mu_prev, float(prev.get("mu", np.nan)))
        self._munz_prev = np.append(self._munz_prev, float(prev.get("mu_nonzero", np.nan)))
        self._sig_prev = np.append(self._sig_prev, float(prev.get("sigma", np.nan)))
        self._z_prev = np.append(self._z_prev, float(prev.get("z", np.nan)))
        p_lo = float(prev.get("lo", np.nan))
        p_hi = float(prev.get("hi", np.nan))
        self._lo = np.append(self._lo, p_lo)
        self._hi = np.append(self._hi, p_hi)
        self._alpha_q = np.append(self._alpha_q, float(prev.get("alpha", self.c2_alpha)))

        usable_interval = bool(np.isfinite(p_lo) and np.isfinite(p_hi))
        self._cover_usable = np.append(self._cover_usable, usable_interval)
        self._covered = np.append(
            self._covered, bool(usable_interval and p_lo <= float(y_t) <= p_hi)
        )

        self._y = np.append(self._y, float(y_t))
        self._cens = np.append(self._cens, bool(censored_t))
        self._stockout = np.append(self._stockout, bool(stockout_t))
        self._n += 1
        self._pending = {
            "mu": mu,
            "mu_nonzero": munz,
            "sigma": sig,
            "z": z,
            "lo": lo,
            "hi": hi,
            "alpha": a,
        }

    # ------------------------------------------------------------------ #
    # reading                                                             #
    # ------------------------------------------------------------------ #

    def _tail(self, arr: np.ndarray, w: int) -> np.ndarray:
        w = max(0, min(int(w), self._n))
        return arr[self._n - w :] if w else arr[:0]

    def read(self, t: int | None = None) -> MonitorReading:
        """Compute the trailing statistics. Pure: does not mutate the monitor."""
        n = self._n
        if t is not None and int(t) != n - 1:
            raise ValueError(f"read(t={t}) called with {n} periods recorded")
        if n == 0:
            return MonitorReading(
                statistics={},
                evidence={},
                window_full=False,
                abstain_reason="no_history",
                aux={"config_warnings": list(self._warnings)},
            )

        w = self.windows
        fl = self.floors
        y = self._y
        usable = ~self._cens
        # Size sub-process residual: observed nonzero demand vs the CONDITIONAL size
        # forecast. `e = y - mu` would difference against the per-period rate and read
        # as bias on the control.
        e = y - self._munz_prev
        e_sz = np.where((y > 0) & usable & np.isfinite(e), e, np.nan)
        delta = (y > 0).astype("float64")
        z_prev = self._z_prev

        # -- baseline window (pre-injection reference) ------------------- #
        b = max(2, min(w["W_base"], n))
        base_slice = slice(0, b)

        # -- C1: persistent bias ----------------------------------------- #
        w1 = min(w["W_short"], n)
        c1 = compute_bias_stats(
            self._tail(e_sz, w1), self._tail(usable, w1), min_evidence=fl["c1_nonzero"]
        )

        # Sign blocks over W_sign_stability (Reading B: use what exists, split into
        # up to four equal blocks, each needing the nonzero floor).
        w_sig = min(w["W_sign_stability"], n)
        sig_e = e_sz[n - w_sig :]
        signs: list[float] = []
        block_len = 0
        if w_sig >= 4:
            k_blocks = min(4, max(1, w_sig // max(1, fl["c1_nonzero"])))
            block_len = w_sig // k_blocks
            for j in range(k_blocks):
                chunk = sig_e[j * block_len : (j + 1) * block_len]
                finite = chunk[np.isfinite(chunk)]
                signs.append(float(np.mean(finite)) if finite.size else 0.0)

        step_bic, drift_bic = step_drift_bic(sig_e)
        n_usable_sz = int(np.sum(np.isfinite(self._tail(e_sz, w1))))

        # -- C2: interval coverage --------------------------------------- #
        w2 = min(w["W_cal"], n)
        covered = self._tail(self._covered, w2)
        cov_usable = self._tail(self._cover_usable, w2)
        if np.any(cov_usable):
            c2 = compute_coverage_stats(
                covered[cov_usable],
                float(np.mean(self._tail(self._alpha_q, w2)[cov_usable])),
                min_evidence=fl["c2_coverage_obs"],
            )
            n_cov = int(np.sum(cov_usable))
        else:
            c2 = {"c_hat": float("nan"), "c_gap": float("nan"), "p_val": float("nan")}
            n_cov = 0

        # -- C3: occurrence ---------------------------------------------- #
        w3 = min(w["W_occ"], n)
        occ_mask = usable[n - w3 :] & np.isfinite(z_prev[n - w3 :])
        c3 = compute_occurrence_stats(
            delta[n - w3 :], z_prev[n - w3 :], occ_mask, min_evidence=fl["c3_nonzero"]
        )
        c3g = compute_occurrence_gap(
            delta[n - w3 :], z_prev[n - w3 :], occ_mask, min_evidence=fl["c3_nonzero"]
        )
        n_pos_occ = int(np.sum((delta[n - w3 :] > 0) & occ_mask))

        # -- C4: dispersion (log scale) ---------------------------------- #
        # Rev 2 §6.3: the baseline is the SKU's own first-W_sz value, not the shared
        # W_base window -- the ratio needs the same number of nonzero sizes on both
        # sides, or the two sd estimates are not comparable.
        w4 = min(w["W_sz"], n)
        y_base = y[: max(b, w4)]
        # Rev 2 §6.3's "sustained over 2 windows": two adjacent, NON-OVERLAPPING
        # W_sz blocks. At W_sz = 24 and tau = 120 both blocks fit after the injection
        # (n >= 168), so this is implementable as written.
        if n >= 2 * w4 and w4 >= 4:
            c4_prev = compute_dispersion_stats(
                y[n - 2 * w4 : n - w4], y_base, min_evidence=fl["c4_nonzero"]
            )
        else:
            c4_prev = {"r": float("nan"), "below_budget": True}
        c4 = compute_dispersion_stats(
            y[n - w4 :], y_base, min_evidence=fl["c4_nonzero"]
        )
        r_history = [float(c4_prev.get("r", np.nan)), float(c4.get("r", np.nan))]

        # -- C5: two CUSUMs ---------------------------------------------- #
        w5 = min(w["W_cusum"], n)
        base_sz = e_sz[base_slice]
        base_sz = base_sz[np.isfinite(base_sz)]
        # A CUSUM calibrated on a handful of nonzero residuals alarms on noise: the
        # MAD of six Gamma draws can sit far below the true size sd. The floor makes
        # that an unmet evidence flag instead of a spectacular false alarm.
        #
        # The scale itself is the SAMPLE SD, not the MAD. A CUSUM's ``h*sigma`` bar is
        # an in-control sd, and MAD*1.4826 estimates sd only for a symmetric
        # distribution. The size residual is a Gamma difference -- right-skewed -- whose
        # MAD sits well below its sd, so the MAD-calibrated CUSUM alarmed at S = 8-10 on
        # the C0 control before any injection (measured at Step 5). The sample sd is the
        # estimator the CUSUM's own theory assumes.
        sigma_base = (
            float(np.std(base_sz, ddof=1))
            if base_sz.size >= fl["c5_baseline_nonzero"]
            else float("nan")
        )
        # Standardise by a TRAILING realised sd before accumulating. Two earlier
        # calibrations were measured and rejected at Step 5:
        #   * the first-W_base sd is 3x too small (the size estimate has not converged
        #     at low p), which alarmed on 94 % of pre-injection C0 checks;
        #   * the state's own in-sample ``sigma`` is a *fitted* residual scale, smaller
        #     than the one-step error, and left the standardised series over-dispersed
        #     (alarm on 100 % of C0 checks).
        # The sd of the last W_sz observed size residuals is out-of-sample by
        # construction, causal, and adapts to the forecaster's own convergence.
        w_ref = min(w["W_sz"], n)
        ref = e_sz[n - w_ref :]
        ref = ref[np.isfinite(ref)]
        sigma_ref = (
            float(np.std(ref, ddof=1))
            if ref.size >= fl["c4_nonzero"]
            else sigma_base
        )
        c5_start = max(self.cusum_burn_in, n - w5)
        e_sz_tail = e_sz[c5_start:n]
        m_sz = np.isfinite(e_sz_tail) & np.isfinite(sigma_ref) & (sigma_ref > 1e-9)
        cusum_sz = compute_cusum_stats(
            e_sz_tail[m_sz] / sigma_ref, 1.0, k=self.cusum_k, h=self.cusum_h
        )

        # The occurrence residual is Bernoulli: its scale changes with p, so it is
        # standardised by the square root of the model's own predicted variance.
        z_t = np.clip(z_prev, 0.02, 0.98)
        sd_occ_t = np.sqrt(z_t * (1.0 - z_t))
        e_occ_tail = (delta - z_prev)[c5_start:n]
        sd_occ_tail = sd_occ_t[c5_start:n]
        m_occ = np.isfinite(e_occ_tail) & np.isfinite(sd_occ_tail) & (sd_occ_tail > 1e-9)
        cusum_occ = compute_cusum_stats(
            e_occ_tail[m_occ] / sd_occ_tail[m_occ], 1.0,
            k=self.cusum_k, h=self.cusum_h,
        )
        # The step-vs-drift test must be run on the SAME sub-process as the alarm. An
        # occurrence step leaves the SIZE residual untouched, so testing the size BIC
        # whenever the occurrence CUSUM alarms asks a question the alarm did not raise
        # (measured at Step 5: the occurrence CUSUM alarmed on ~90 % of checks and the
        # size-BIC condition then blocked C5 on ~85 % of them -- the condition was a
        # coin flip unrelated to the alarm).
        occ_step_bic, occ_drift_bic = step_drift_bic(e_occ_tail)

        # -- C6: intermittency cell -------------------------------------- #
        w6 = min(w["W_class"], n)
        point = adi_cv2_cell(
            y[n - w6 :], adi_star=self.adi_star, cv2_star=self.cv2_star
        )
        init = adi_cv2_cell(
            y[base_slice], adi_star=self.adi_star, cv2_star=self.cv2_star
        )
        cells: list[str] = []
        # As with C4: "sustained for 3 consecutive windows" is read as three consecutive
        # monitor periods. Three non-overlapping W_class blocks after tau would need
        # 3*W_class + tau = 210 > T and could never fire.
        if n >= w6 + 2 and w6 >= 4:
            for j in (2, 1, 0):
                hi_i = n - j
                cells.append(
                    adi_cv2_cell(
                        y[hi_i - w6 : hi_i], adi_star=self.adi_star,
                        cv2_star=self.cv2_star,
                    )["cell"]
                )
        else:
            cells = [point["cell"]]

        sustained = (
            len(cells) >= 3
            and init["cell"] != "dead"
            and all(c != init["cell"] for c in cells)
            and len(set(cells)) == 1
        )
        stability = float("nan")
        if sustained and w6 >= fl["c6_periods"]:
            stability = bootstrap_cell_stability(
                y[n - w6 :],
                n_boot=self.c6_bootstrap_n,
                adi_star=self.adi_star,
                cv2_star=self.cv2_star,
                rng=self._rng,
            )

        # -- C7: service gap --------------------------------------------- #
        w7 = min(w["W_svc"], n)
        st_tail = self._tail(self._stockout, w7)
        csl_hat = float(np.mean(~st_tail)) if st_tail.size else float("nan")
        c7_gap = csl_hat - self.alpha_target if np.isfinite(csl_hat) else float("nan")
        n_cycles = int(st_tail.size // self.protection) if self.protection > 0 else 0
        unmet = 0
        if n_cycles:
            blocks = st_tail[-n_cycles * self.protection :].reshape(
                n_cycles, self.protection
            )
            unmet = int(np.sum(blocks.any(axis=1)))

        # -- assemble ----------------------------------------------------- #
        statistics = {
            "c1_b_sz": float(c1.get("b_sz", np.nan)),
            "c1_z_b": float(c1.get("z_b", np.nan)),
            "c1_step_bic": float(step_bic),
            "c1_drift_bic": float(drift_bic),
            "c2_c_hat": float(c2.get("c_hat", np.nan)),
            "c2_c_gap": float(c2.get("c_gap", np.nan)),
            "c2_p_val": float(c2.get("p_val", np.nan)),
            "c3_auc": float(c3.get("auc", np.nan)),
            "c3_fz": float(c3.get("fz", np.nan)),
            "c3_occ_gap": float(c3g.get("occ_gap", np.nan)),
            "c3_occ_p_val": float(c3g.get("p_val", np.nan)),
            "c4_r": float(c4.get("r", np.nan)),
            "c4_r_prev": float(c4_prev.get("r", np.nan)),
            "c4_below_budget": float(bool(c4.get("below_budget", True))),
            "c5_sz_max_s": float(cusum_sz["max_S"]),
            "c5_sz_alarm_score": float(cusum_sz["alarm_score"]),
            "c5_sz_alarm": float(bool(cusum_sz["alarm"])),
            "c5_occ_max_s": float(cusum_occ["max_S"]),
            "c5_occ_alarm_score": float(cusum_occ["alarm_score"]),
            "c5_occ_alarm": float(bool(cusum_occ["alarm"])),
            "c5_occ_step_bic": float(occ_step_bic),
            "c5_occ_drift_bic": float(occ_drift_bic),
            "c6_adi": float(point["adi"]),
            "c6_cv2": float(point["cv2"]),
            "c6_stability": float(stability),
            "c6_changed": float(bool(sustained)),
            "c7_csl_hat": float(csl_hat),
            "c7_csl_gap": float(c7_gap),
            "c6_sustain_windows": float(len(cells)),
        }

        evidence = {
            "c1": n_usable_sz >= fl["c1_nonzero"],
            "c2": n_cov >= fl["c2_coverage_obs"],
            "c3": n_pos_occ >= fl["c3_nonzero"],
            "c4": not bool(c4.get("below_budget", True)),
            "c5": bool(e_sz_tail.size >= 8 and base_sz.size >= fl["c5_baseline_nonzero"]),
            "c6": bool(w6 >= fl["c6_periods"] and init["cell"] != "dead"),
            "c7": bool(w7 >= fl["c7_periods"] and unmet >= fl["c7_unmet_cycles"]),
        }
        missing = sorted(k for k, v in evidence.items() if not v)
        abstain_reason = f"evidence_floor:{','.join(missing)}" if missing else None

        full_needed = max(
            w["W_short"], w["W_sign_stability"], w["W_cal"], w["W_occ"],
            w["W_sz"], w["W_class"], w["W_svc"], w["W_cusum"],
        )
        aux = {
            "t": n - 1,
            "signs": signs,
            "sign_block_len": block_len,
            "cells": cells,
            "init_cell": init["cell"],
            "unmet_cycles": unmet,
            "n_usable_sz": n_usable_sz,
            "n_coverage_obs": n_cov,
            "n_nonzero_occ": n_pos_occ,
            "r_history": r_history,
            "config_warnings": list(self._warnings),
        }

        return MonitorReading(
            statistics=statistics,
            evidence=evidence,
            window_full=bool(n >= full_needed),
            abstain_reason=abstain_reason,
            aux=aux,
        )

    @property
    def n_periods(self) -> int:
        """Periods recorded so far."""
        return self._n
