"""CAFR Diagnostic Experiment: C3-C6 Attribution-Overlap and Deferral Analysis.

Evaluates three predeclared attribution variants offline on identical series:
- V0: Current pipeline (reproduces existing priority C3 -> C6 and latching)
- V1: Simultaneous C3/C6 tie-break (prefers C6 when both rules fire at same checkpoint)
- V2: Bounded C3 deferral (when C3 fires with c6_changed active but stability pending,
      defers for <= 3 consecutive checkpoints; attributes C6 if it fires, else falls back to C3)

Evaluates on the 120 training SKUs only (i % 5 < 3) for:
- 120 C0 controls
- 120 Candidate C6 series (inject_C6_v2)
- 120 True C3 series
Total: 360 series. Quarantines validation and test splits strictly.
Outputs results to results/exploratory/c6_redesign/.
Preserves all production code, configs, and frozen Step 5/Step 6 artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cafr.attribution.pipeline import attribute, PRIORITY_ORDER, CauseEvaluation
from cafr.attribution.priority import attribute_cause, compute_confidence
from cafr.attribution.rules import (
    check_c1_bias,
    check_c2_coverage,
    check_c3_occurrence,
    check_c4_size,
    check_c5_cusum,
    check_c6_intermittency,
    check_c7_policy,
)
from cafr.data.synth.budget import check_c6_budget
from cafr.data.synth.generator import CELLS, draw_statics, get_cell_params
from cafr.data.synth.injections import inject_C6_v2
from cafr.data.synth.labels import _sku_rng
from cafr.monitor.features import conformal_interval
from cafr.arms.arm_c import arm_c_factory
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor.reading import Monitor, MonitorReading
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import load_config
from experiments.step5_monitor_gates import _sku_cfg

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "c6_redesign"
CAUSES_ALL = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")


def _run_recording_sim(
    sku_id: str,
    y: np.ndarray,
    param: pd.Series,
    base_cfg: dict,
    pool,
    seed: int,
    *,
    check_from: int = 24,
    detect_from: int = 120,
    stride: int = 4,
) -> tuple[list[dict], MonitorReading]:
    """Run inventory simulation and record monitor reading at EVERY stride checkpoint."""
    cfg = _sku_cfg(base_cfg, param)
    sim_base = SimConfig.from_config(cfg)
    sim_cfg = SimConfig(
        L=sim_base.L,
        R=sim_base.R,
        review_period=sim_base.review_period,
        unmet_model=sim_base.unmet_model,
        quantile_window=sim_base.quantile_window,
        min_aggregates=sim_base.min_aggregates,
        quantile_method=sim_base.quantile_method,
        alpha_min=sim_base.alpha_min,
        alpha_max_cap=sim_base.alpha_max_cap,
        H=float(param.get("holding_rate", 0.20)),
        B_over_H=sim_base.B_over_H,
        c7_safety_factor_scale=sim_base.c7_safety_factor_scale,
        c2_quantile_scale=sim_base.c2_quantile_scale,
        interval_half_width=sim_base.interval_half_width,
        protection_convention=sim_base.protection_convention,
        observation_model=sim_base.observation_model,
    )
    sim = InventorySimulator(sim_cfg)
    monitor = Monitor(cfg, alpha_target=0.95, seed=seed)
    base = arm_c_factory(pool, cfg, sku_id, 0.95, "observed")
    q_alpha = float(cfg.get("detection", {}).get("c2_interval_alpha", 0.90))
    q_win = int(cfg["sim"].get("quantile_window", 24))
    q_scale = float(cfg["sim"].get("c2_quantile_scale", 1.0))

    checkpoints: list[dict] = []

    def level_fn(t, y_obs, cens, inv):
        S, a = base(t, y_obs, cens, inv)
        st = base.last_state
        res = base._residuals.to_array()
        rr = res[-q_win:] if res.size else res
        half_w = cfg['sim'].get('interval_half_width')
        if half_w is not None and pd.notna(half_w) and np.isfinite(float(half_w)):
            w = float(half_w)
            mu_f = float(st.mu)
            lo = max(0.0, mu_f - w) if np.isfinite(mu_f) else float('nan')
            hi = max(0.0, mu_f + w) if np.isfinite(mu_f) else float('nan')
        else:
            lo, hi = conformal_interval(
                float(st.mu), rr, q_alpha, scale=q_scale, min_evidence=8
            )
        stockout = bool(inv.get("on_hand", 1.0) <= 1e-9 and float(y_obs[-1]) > 0.0)
        monitor.update(
            t,
            float(y_obs[-1]),
            bool(cens[-1]),
            {
                "mu": float(st.mu),
                "mu_nonzero": float(st.mu_nonzero),
                "sigma": float(st.sigma),
                "z": float(st.z),
                "lo": lo,
                "hi": hi,
                "alpha": q_alpha,
            },
            stockout_t=stockout,
        )
        if int(t) >= check_from and (int(t) - check_from) % stride == 0:
            rd = monitor.read()
            # Record reading and evaluated rules
            det = cfg.get("detection", {})
            z_crit = float(det.get("z_b_crit", 1.96))
            tau_cov = float(det.get("tau_cov", 0.10))
            tau_occ = float(det.get("tau_occ", 0.10))
            r_thr = float(det.get("c4_r_threshold", 1.5))
            h_cusum = float(det.get("cusum_h", 4.0))
            stab_thr = float(det.get("c6_bootstrap_stability", 0.90))
            tau_svc = float(det.get("delta_svc", 0.10))

            s = rd.statistics
            ev = rd.evidence

            def _num(key: str, default=float("nan")):
                v = s.get(key, default)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            z_b = _num("c1_z_b")
            step_bic = _num("c1_step_bic")
            drift_bic = _num("c1_drift_bic")
            r_now = _num("c4_r")
            r_prev = _num("c4_r_prev")
            auc = _num("c3_auc")
            occ_gap = _num("c3_occ_gap")
            occ_p = _num("c3_occ_p_val")
            c_gap = _num("c2_c_gap")
            p_val = _num("c2_p_val")
            max_sz = _num("c5_sz_max_s", 0.0)
            max_occ = _num("c5_occ_max_s", 0.0)
            stability = _num("c6_stability")
            csl_gap = _num("c7_csl_gap")

            signs = rd.aux.get("signs", [])
            cells = rd.aux.get("cells", [])
            init_cell = rd.aux.get("init_cell", "dead")
            r_history = [r_prev, r_now]

            fired = {
                "C1": bool(ev.get("c1", False)) and check_c1_bias(
                    z_b, list(signs), int(rd.aux.get("n_usable_sz", 0)),
                    step_bic, drift_bic,
                    min_evidence=int(cfg.get("monitor", {}).get("min_evidence", {}).get("c1_nonzero", 8)),
                ),
                "C2": bool(ev.get("c2", False)) and check_c2_coverage(
                    c_gap, p_val, z_b, r_now, tau_cov=tau_cov
                ),
                "C3": bool(ev.get("c3", False)) and check_c3_occurrence(
                    occ_gap, occ_p, r_now, tau_occ=tau_occ
                ),
                "C4": bool(ev.get("c4", False)) and check_c4_size(r_history, auc),
                "C5": bool(ev.get("c5", False)) and check_c5_cusum(
                    bool(s.get("c5_occ_alarm", 0.0)), bool(s.get("c5_sz_alarm", 0.0)),
                    step_bic, drift_bic,
                    _num("c5_occ_step_bic"), _num("c5_occ_drift_bic"),
                ),
                "C6": bool(ev.get("c6", False)) and check_c6_intermittency(
                    list(cells), str(init_cell), stability
                ),
                "C7": bool(ev.get("c7", False)) and check_c7_policy(
                    csl_gap, z_b, r_now, tau_svc=tau_svc
                ),
            }

            def _ratio(num: float, den: float) -> float:
                if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
                    return 0.0
                return float(abs(num) / den)

            r_seen = [max(v, 1.0 / v) for v in (r_now, r_prev) if np.isfinite(v) and v > 0]
            margins = {
                "C1": _ratio(z_b, z_crit),
                "C2": _ratio(c_gap, tau_cov),
                "C3": _ratio(occ_gap, tau_occ),
                "C4": _ratio(max(r_seen) if r_seen else float("nan"), r_thr),
                "C5": _ratio(max(max_sz, max_occ), h_cusum),
                "C6": _ratio(stability, stab_thr) if bool(s.get("c6_changed", 0.0)) else 0.0,
                "C7": _ratio(-csl_gap if np.isfinite(csl_gap) else float("nan"), tau_svc),
            }

            checkpoints.append({
                "t": int(t),
                "reading": rd,
                "fired": fired,
                "margins": margins,
                "c6_changed": float(s.get("c6_changed", 0.0)),
                "c6_stability": float(stability),
                "c3_occ_gap": float(occ_gap),
            })
        return S, a

    sim.run(sku_id, y, level_fn, arm="arm_c")
    final_reading = monitor.read()
    return checkpoints, final_reading


def _eval_v0(checkpoints: list[dict], final_rd: MonitorReading, cfg: dict, detect_from: int = 120) -> dict:
    """Evaluate V0 (current production priority and latching)."""
    tau_conf = float(cfg.get("detection", {}).get("tau_conf", 0.35))
    for cp in checkpoints:
        t = cp["t"]
        if t < detect_from:
            continue
        evals = {c: CauseEvaluation(fired=cp["fired"][c], margin=cp["margins"][c]) for c in PRIORITY_ORDER}
        cause, conf = attribute_cause(evals, tau_conf=tau_conf, order=list(PRIORITY_ORDER))
        if cause != "C0":
            return {
                "cause": cause,
                "t": t,
                "conf": conf,
                "fired": tuple(c for c in PRIORITY_ORDER if cp["fired"][c]),
                "c6_changed": cp["c6_changed"],
                "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
            }
    # Fallback to final
    att_final = attribute(final_rd, cfg, order=list(PRIORITY_ORDER))
    return {
        "cause": att_final.cause,
        "t": 199 if att_final.cause != "C0" else None,
        "conf": att_final.confidence,
        "fired": att_final.fired,
        "c6_changed": float(final_rd.statistics.get("c6_changed", 0.0)),
        "c6_stability": float(final_rd.statistics.get("c6_stability", float("nan"))),
        "c3_occ_gap": float(final_rd.statistics.get("c3_occ_gap", float("nan"))),
    }


def _eval_v1(checkpoints: list[dict], final_rd: MonitorReading, cfg: dict, detect_from: int = 120) -> dict:
    """Evaluate V1 (simultaneous C3/C6 tie-break: prefers C6 if both fire)."""
    tau_conf = float(cfg.get("detection", {}).get("tau_conf", 0.35))
    v1_order = ["C5", "C1", "C4", "C2", "C6", "C3", "C7"]
    for cp in checkpoints:
        t = cp["t"]
        if t < detect_from:
            continue
        fired = cp["fired"]
        evals = {c: CauseEvaluation(fired=fired[c], margin=cp["margins"][c]) for c in PRIORITY_ORDER}
        # If both C3 and C6 fire, use v1_order (C6 before C3); otherwise standard priority
        order = v1_order if (fired.get("C3") and fired.get("C6")) else list(PRIORITY_ORDER)
        cause, conf = attribute_cause(evals, tau_conf=tau_conf, order=order)
        if cause != "C0":
            return {
                "cause": cause,
                "t": t,
                "conf": conf,
                "fired": tuple(c for c in order if fired[c]),
                "c6_changed": cp["c6_changed"],
                "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
            }
    # Fallback to final
    fired_final = attribute(final_rd, cfg).fired
    order_final = v1_order if ("C3" in fired_final and "C6" in fired_final) else list(PRIORITY_ORDER)
    att_final = attribute(final_rd, cfg, order=order_final)
    return {
        "cause": att_final.cause,
        "t": 199 if att_final.cause != "C0" else None,
        "conf": att_final.confidence,
        "fired": att_final.fired,
        "c6_changed": float(final_rd.statistics.get("c6_changed", 0.0)),
        "c6_stability": float(final_rd.statistics.get("c6_stability", float("nan"))),
        "c3_occ_gap": float(final_rd.statistics.get("c3_occ_gap", float("nan"))),
    }


def _eval_v2(checkpoints: list[dict], final_rd: MonitorReading, cfg: dict, detect_from: int = 120) -> dict:
    """Evaluate V2 (bounded C3 deferral when c6_changed==1.0 and stability pending).

    Rules:
    - If higher-priority causes (C5, C1, C4, C2) fire with conf >= tau_conf: latch immediately.
    - If C6 fires with conf >= tau_conf: latch C6 immediately (resolves any deferral).
    - If C3 would be attributed:
        - If c6_changed == 1.0 and stability < 0.90 (or nan):
            - Defer C3 for <= 3 consecutive checkpoints (deferral_count <= 3).
            - During deferral, monitor does not latch.
            - If deferral_count reaches 3 and C6 does not fire by the 4th checkpoint:
              fall back and attribute C3.
        - Else (no cell change active): latch C3 immediately.
    - If deferral was active but at subsequent checkpoint neither C3 nor C6 fires:
        - deferral_count increments; if reaches 3 without C6 firing, fall back to C3.
    """
    tau_conf = float(cfg.get("detection", {}).get("tau_conf", 0.35))
    v1_order = ["C5", "C1", "C4", "C2", "C6", "C3", "C7"]

    deferral_active = False
    deferral_count = 0
    deferred_res = None

    for cp in checkpoints:
        t = cp["t"]
        if t < detect_from:
            continue
        fired = cp["fired"]
        evals = {c: CauseEvaluation(fired=fired[c], margin=cp["margins"][c]) for c in PRIORITY_ORDER}

        # 1. Higher priority causes (C5, C1, C4, C2)
        conf_raw = compute_confidence(evals)
        for higher_cause in ("C5", "C1", "C4", "C2"):
            if fired.get(higher_cause) and conf_raw >= tau_conf:
                # Immediate higher-priority latch
                return {
                    "cause": higher_cause,
                    "t": t,
                    "conf": conf_raw,
                    "fired": tuple(c for c in PRIORITY_ORDER if fired[c]),
                    "c6_changed": cp["c6_changed"],
                    "c6_stability": cp["c6_stability"],
                    "c3_occ_gap": cp["c3_occ_gap"],
                    "deferral_events": deferral_count,
                }

        # 2. Does C6 fire?
        if fired.get("C6") and conf_raw >= tau_conf:
            # C6 fires! Resolves deferral in favor of C6
            return {
                "cause": "C6",
                "t": t,
                "conf": conf_raw,
                "fired": tuple(c for c in v1_order if fired[c]),
                "c6_changed": cp["c6_changed"],
                "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
                "deferral_events": deferral_count,
            }

        # 3. Does C3 fire?
        if fired.get("C3") and conf_raw >= tau_conf:
            # Check if C6 cell change is present but stability pending
            cell_changed = (cp["c6_changed"] == 1.0)
            stability_pending = (np.isnan(cp["c6_stability"]) or cp["c6_stability"] < 0.90)

            if cell_changed and stability_pending:
                if deferral_count < 3:
                    # Defer C3!
                    if not deferral_active:
                        deferral_active = True
                        deferral_count = 1
                        deferred_res = {
                            "cause": "C3",
                            "t": t,
                            "conf": conf_raw,
                            "fired": tuple(c for c in PRIORITY_ORDER if fired[c]),
                            "c6_changed": cp["c6_changed"],
                            "c6_stability": cp["c6_stability"],
                            "c3_occ_gap": cp["c3_occ_gap"],
                        }
                    else:
                        deferral_count += 1
                    # Deferral continues
                    continue
                else:
                    # Deferral limit reached (deferred for 3 consecutive checks)! Fall back to C3
                    return {
                        "cause": "C3",
                        "t": t,
                        "conf": conf_raw,
                        "fired": tuple(c for c in PRIORITY_ORDER if fired[c]),
                        "c6_changed": cp["c6_changed"],
                        "c6_stability": cp["c6_stability"],
                        "c3_occ_gap": cp["c3_occ_gap"],
                        "deferral_events": deferral_count,
                    }
            else:
                # No cell change active -> latch C3 immediately
                return {
                    "cause": "C3",
                    "t": t,
                    "conf": conf_raw,
                    "fired": tuple(c for c in PRIORITY_ORDER if fired[c]),
                    "c6_changed": cp["c6_changed"],
                    "c6_stability": cp["c6_stability"],
                    "c3_occ_gap": cp["c3_occ_gap"],
                    "deferral_events": deferral_count,
                }

        # 4. Neither higher priority, nor C6, nor C3 fired
        if deferral_active:
            deferral_count += 1
            if deferral_count >= 3:
                # Deferral window expired without C6 firing -> fall back to deferred C3
                deferred_res["deferral_events"] = deferral_count
                return deferred_res
            continue

        # 5. Check C7
        if fired.get("C7") and conf_raw >= tau_conf:
            return {
                "cause": "C7",
                "t": t,
                "conf": conf_raw,
                "fired": tuple(c for c in PRIORITY_ORDER if fired[c]),
                "c6_changed": cp["c6_changed"],
                "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
                "deferral_events": 0,
            }

    # End of checkpoints loop
    if deferral_active and deferred_res is not None:
        deferred_res["deferral_events"] = deferral_count
        return deferred_res

    # Terminal fallback at t=199
    att_final = attribute(final_rd, cfg, order=v1_order)
    return {
        "cause": att_final.cause,
        "t": 199 if att_final.cause != "C0" else None,
        "conf": att_final.confidence,
        "fired": att_final.fired,
        "c6_changed": float(final_rd.statistics.get("c6_changed", 0.0)),
        "c6_stability": float(final_rd.statistics.get("c6_stability", float("nan"))),
        "c3_occ_gap": float(final_rd.statistics.get("c3_occ_gap", float("nan"))),
        "deferral_events": deferral_count,
    }


def _paired_ci_diff(d1: np.ndarray, d0: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Compute bootstrap 95% CI for paired proportion difference mean(d1) - mean(d0)."""
    assert len(d1) == len(d0)
    diff = d1.astype(float) - d0.astype(float)
    observed_diff = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = [float(np.mean(rng.choice(diff, size=n, replace=True))) for _ in range(n_boot)]
    ci_low = float(np.percentile(boots, 2.5))
    ci_high = float(np.percentile(boots, 97.5))
    return {
        "observed_diff": observed_diff,
        "ci_95": [ci_low, ci_high],
        "std_err": float(np.std(boots)),
    }


def run_overlap_diagnostic():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config("synthetic.yaml")
    data_dir = REPO_ROOT / "data" / "synth"
    demand = pd.read_parquet(data_dir / "demand_panel.parquet")
    params = pd.read_parquet(data_dir / "generator_params.parquet").set_index("sku_id")

    burn = int(cfg["splits"]["burn_in"])
    T = int(cfg["synthetic"]["T"])
    tau_inj = int(cfg["synthetic"]["tau_inj"])

    # 1. Select 120 training SKUs for C0, C6, C3 (i % 5 < 3)
    c0_all = sorted(params.index[params["true_cause"] == "C0"])
    c6_all = sorted(params.index[params["true_cause"] == "C6"])
    c3_all = sorted(params.index[params["true_cause"] == "C3"])

    c0_train = [c0_all[i] for i in range(200) if i % 5 < 3]
    c6_train = [c6_all[i] for i in range(200) if i % 5 < 3]
    c3_train = [c3_all[i] for i in range(200) if i % 5 < 3]

    print("=" * 70)
    print("CAFR C3-C6 OVERLAP & DEFERRAL DIAGNOSTIC (TRAINING SPLIT ONLY)")
    print(f"C0 Training Controls: {len(c0_train)}")
    print(f"C6 Training Candidate SKUs: {len(c6_train)}")
    print(f"C3 Training SKUs: {len(c3_train)}")
    print("=" * 70)

    # 2. Build global pooled forecaster on burn-in (periods 0..24)
    all_panel_skus = sorted(params.index)
    statics_by_sku = {
        sid: {
            "lead_time": float(params.loc[sid, "lead_time"]),
            "unit_cost": float(params.loc[sid, "unit_cost"]),
            "holding_rate": float(params.loc[sid, "holding_rate"]),
            "adi_init": float(params.loc[sid, "adi_init"]),
            "cv2_init": float(params.loc[sid, "cv2_init"]),
        }
        for sid in all_panel_skus
    }
    series_map = {
        sid: grp.sort_values("period")["demand_observed"].to_numpy(dtype="float64")
        for sid, grp in demand[demand["sku_id"].isin(all_panel_skus)].groupby("sku_id", sort=False)
    }

    train_panel = pd.DataFrame(
        [
            {"sku_id": sid, "period": t, "demand_observed": float(series_map[sid][t]),
             **statics_by_sku[sid]}
            for sid in all_panel_skus
            for t in range(burn)
        ]
    )
    pool = fit_global_pool(cfg, train_panel, statics_by_sku=statics_by_sku)

    # Calibrated CUSUM threshold
    with open(REPO_ROOT / "results" / "step5" / "step5_summary.json") as f:
        s5_summary = json.load(f)
    calibrated_h = float(s5_summary["cusum_calibration"]["h"])
    cfg.setdefault("detection", {})["cusum_h"] = calibrated_h

    # 3. Simulate and Record Every Checkpoint for All 360 SKUs
    records = []
    checkpoint_rows = []

    # Cohort 1: 120 C0 Controls
    print("\n[1/3] Simulating 120 C0 Training Controls...")
    for sid in c0_train:
        y = series_map[sid]
        param_row = params.loc[sid]
        cps, final_rd = _run_recording_sim(sid, y, param_row, cfg, pool, 42)
        v0 = _eval_v0(cps, final_rd, cfg)
        v1 = _eval_v1(cps, final_rd, cfg)
        v2 = _eval_v2(cps, final_rd, cfg)
        records.append({
            "sku_id": sid, "cohort": "C0", "true_cause": "C0",
            "v0_cause": v0["cause"], "v0_t": v0["t"], "v0_conf": v0["conf"],
            "v1_cause": v1["cause"], "v1_t": v1["t"], "v1_conf": v1["conf"],
            "v2_cause": v2["cause"], "v2_t": v2["t"], "v2_conf": v2["conf"],
            "v2_deferral_events": v2.get("deferral_events", 0),
        })
        for cp in cps:
            checkpoint_rows.append({
                "sku_id": sid, "cohort": "C0", "t": cp["t"],
                "c3_fired": cp["fired"]["C3"], "c6_fired": cp["fired"]["C6"],
                "c6_changed": cp["c6_changed"], "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
            })

    # Cohort 2: 120 Candidate C6 (inject_C6_v2)
    print("\n[2/3] Simulating 120 Candidate C6 Training SKUs (inject_C6_v2)...")
    for sid in c6_train:
        idx = int(sid.split(":")[-1])
        rng = _sku_rng(42, "single", idx)
        cell = str(rng.choice(CELLS))
        statics = draw_statics(rng.spawn(1)[0])
        p_draw, k_draw, mu_z_draw = get_cell_params(cell, rng)
        y_cand, delta_cand, z_cand, cand_info = inject_C6_v2(
            T, tau_inj, p_draw, k_draw, mu_z_draw, rng,
            w_ramp=24, target_up=0.45, target_down=0.10,
        )
        param_row = params.loc[sid].copy()
        param_row["p"] = cand_info["p_init"]
        param_row["p_new"] = cand_info["p_target"]
        cps, final_rd = _run_recording_sim(sid, y_cand, param_row, cfg, pool, 42)
        v0 = _eval_v0(cps, final_rd, cfg)
        v1 = _eval_v1(cps, final_rd, cfg)
        v2 = _eval_v2(cps, final_rd, cfg)
        records.append({
            "sku_id": sid, "cohort": "C6_cand", "true_cause": "C6",
            "v0_cause": v0["cause"], "v0_t": v0["t"], "v0_conf": v0["conf"],
            "v1_cause": v1["cause"], "v1_t": v1["t"], "v1_conf": v1["conf"],
            "v2_cause": v2["cause"], "v2_t": v2["t"], "v2_conf": v2["conf"],
            "v2_deferral_events": v2.get("deferral_events", 0),
        })
        for cp in cps:
            checkpoint_rows.append({
                "sku_id": sid, "cohort": "C6_cand", "t": cp["t"],
                "c3_fired": cp["fired"]["C3"], "c6_fired": cp["fired"]["C6"],
                "c6_changed": cp["c6_changed"], "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
            })

    # Cohort 3: 120 True C3 SKUs
    print("\n[3/3] Simulating 120 True C3 Training SKUs...")
    for sid in c3_train:
        y = series_map[sid]
        param_row = params.loc[sid]
        cps, final_rd = _run_recording_sim(sid, y, param_row, cfg, pool, 42)
        v0 = _eval_v0(cps, final_rd, cfg)
        v1 = _eval_v1(cps, final_rd, cfg)
        v2 = _eval_v2(cps, final_rd, cfg)
        records.append({
            "sku_id": sid, "cohort": "C3", "true_cause": "C3",
            "v0_cause": v0["cause"], "v0_t": v0["t"], "v0_conf": v0["conf"],
            "v1_cause": v1["cause"], "v1_t": v1["t"], "v1_conf": v1["conf"],
            "v2_cause": v2["cause"], "v2_t": v2["t"], "v2_conf": v2["conf"],
            "v2_deferral_events": v2.get("deferral_events", 0),
        })
        for cp in cps:
            checkpoint_rows.append({
                "sku_id": sid, "cohort": "C3", "t": cp["t"],
                "c3_fired": cp["fired"]["C3"], "c6_fired": cp["fired"]["C6"],
                "c6_changed": cp["c6_changed"], "c6_stability": cp["c6_stability"],
                "c3_occ_gap": cp["c3_occ_gap"],
            })

    df_res = pd.DataFrame(records)
    df_cps = pd.DataFrame(checkpoint_rows)

    # 4. Save Checkpoint Trajectories and Paired Attributions
    traj_path = OUT_DIR / "c3_c6_checkpoint_trajectories.parquet"
    attrib_path = OUT_DIR / "c3_c6_paired_attributions.parquet"
    df_cps.to_parquet(traj_path, index=False)
    df_res.to_parquet(attrib_path, index=False)
    print(f"\nSaved checkpoint trajectories to: {traj_path}")
    print(f"Saved paired attributions to:     {attrib_path}")

    # 5. Compute Metrics and Paired Comparisons
    df_c0 = df_res[df_res["cohort"] == "C0"]
    df_c6 = df_res[df_res["cohort"] == "C6_cand"]
    df_c3 = df_res[df_res["cohort"] == "C3"]

    def _cohort_metrics(df_sub, true_label):
        return {
            v: {
                "attributed_counts": df_sub[f"{v}_cause"].value_counts().to_dict(),
                "recall": float((df_sub[f"{v}_cause"] == true_label).mean()),
                "recall_count": int((df_sub[f"{v}_cause"] == true_label).sum()),
                "c6_count": int((df_sub[f"{v}_cause"] == "C6").sum()),
                "c3_count": int((df_sub[f"{v}_cause"] == "C3").sum()),
                "c0_count": int((df_sub[f"{v}_cause"] == "C0").sum()),
            }
            for v in ("v0", "v1", "v2")
        }

    c0_m = _cohort_metrics(df_c0, "C0")
    c6_m = _cohort_metrics(df_c6, "C6")
    c3_m = _cohort_metrics(df_c3, "C3")

    # 3-Class Macro-F1 across C0, C3, C6
    classes = ["C0", "C3", "C6"]
    macro_f1 = {}
    f1_per_class = {}
    for v in ("v0", "v1", "v2"):
        y_true = df_res["true_cause"].to_numpy()
        y_pred = df_res[f"{v}_cause"].to_numpy()
        f1_m = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
        prec, rec, f1s, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
        macro_f1[v] = f1_m
        f1_per_class[v] = {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1s[i])}
                           for i, c in enumerate(classes)}

    # Paired Statistical Differences
    # C6 recall: V1 vs V0, V2 vs V0
    c6_v0_hit = (df_c6["v0_cause"] == "C6").to_numpy()
    c6_v1_hit = (df_c6["v1_cause"] == "C6").to_numpy()
    c6_v2_hit = (df_c6["v2_cause"] == "C6").to_numpy()
    c6_diff_v1_v0 = _paired_ci_diff(c6_v1_hit, c6_v0_hit)
    c6_diff_v2_v0 = _paired_ci_diff(c6_v2_hit, c6_v0_hit)

    # C0 false positive to C6: V1 vs V0, V2 vs V0
    c0_v0_fp_c6 = (df_c0["v0_cause"] == "C6").to_numpy()
    c0_v1_fp_c6 = (df_c0["v1_cause"] == "C6").to_numpy()
    c0_v2_fp_c6 = (df_c0["v2_cause"] == "C6").to_numpy()
    c0_diff_v1_v0 = _paired_ci_diff(c0_v1_fp_c6, c0_v0_fp_c6)
    c0_diff_v2_v0 = _paired_ci_diff(c0_v2_fp_c6, c0_v0_fp_c6)

    # C3 recall: V1 vs V0, V2 vs V0
    c3_v0_hit = (df_c3["v0_cause"] == "C3").to_numpy()
    c3_v1_hit = (df_c3["v1_cause"] == "C3").to_numpy()
    c3_v2_hit = (df_c3["v2_cause"] == "C3").to_numpy()
    c3_diff_v1_v0 = _paired_ci_diff(c3_v1_hit, c3_v0_hit)
    c3_diff_v2_v0 = _paired_ci_diff(c3_v2_hit, c3_v0_hit)

    # C3 misattributed to C6
    c3_v0_to_c6 = (df_c3["v0_cause"] == "C6").to_numpy()
    c3_v1_to_c6 = (df_c3["v1_cause"] == "C6").to_numpy()
    c3_v2_to_c6 = (df_c3["v2_cause"] == "C6").to_numpy()
    c3_c6_diff_v1_v0 = _paired_ci_diff(c3_v1_to_c6, c3_v0_to_c6)
    c3_c6_diff_v2_v0 = _paired_ci_diff(c3_v2_to_c6, c3_v0_to_c6)

    # Per-SKU changes
    c6_v1_changes = df_c6[df_c6["v0_cause"] != df_c6["v1_cause"]][["sku_id", "v0_cause", "v1_cause", "v0_t", "v1_t"]].to_dict(orient="records")
    c6_v2_changes = df_c6[df_c6["v0_cause"] != df_c6["v2_cause"]][["sku_id", "v0_cause", "v2_cause", "v0_t", "v2_t", "v2_deferral_events"]].to_dict(orient="records")
    c3_v1_changes = df_c3[df_c3["v0_cause"] != df_c3["v1_cause"]][["sku_id", "v0_cause", "v1_cause"]].to_dict(orient="records")
    c3_v2_changes = df_c3[df_c3["v0_cause"] != df_c3["v2_cause"]][["sku_id", "v0_cause", "v2_cause", "v2_deferral_events"]].to_dict(orient="records")
    c0_v1_changes = df_c0[df_c0["v0_cause"] != df_c0["v1_cause"]][["sku_id", "v0_cause", "v1_cause"]].to_dict(orient="records")
    c0_v2_changes = df_c0[df_c0["v0_cause"] != df_c0["v2_cause"]][["sku_id", "v0_cause", "v2_cause", "v2_deferral_events"]].to_dict(orient="records")

    report = {
        "metadata": {
            "title": "CAFR C3-C6 Attribution Overlap and Bounded Deferral Diagnostic",
            "protocol": "Training split only (i % 5 < 3), paired comparison across V0, V1, V2",
            "quarantined_splits": "Validation (i % 5 == 3) and Test (i % 5 == 4) untouched",
            "n_c0": len(c0_train),
            "n_c6_candidate": len(c6_train),
            "n_c3": len(c3_train),
            "total_series": len(df_res),
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "v0_verification": {
            "c0_to_c6_fp": int(c0_m["v0"]["c6_count"]),
            "c0_to_c6_fp_rate": float(c0_m["v0"]["c6_count"] / 120),
            "c6_candidate_recall": int(c6_m["v0"]["recall_count"]),
            "c6_candidate_recall_rate": float(c6_m["v0"]["recall"]),
            "matches_saved_diagnostic": bool(
                c0_m["v0"]["c6_count"] == 14 and c6_m["v0"]["recall_count"] == 31
            ),
        },
        "summary_table": {
            "c6_recall": {
                "V0": f"{c6_m['v0']['recall_count']}/120 ({c6_m['v0']['recall']:.1%})",
                "V1": f"{c6_m['v1']['recall_count']}/120 ({c6_m['v1']['recall']:.1%})",
                "V2": f"{c6_m['v2']['recall_count']}/120 ({c6_m['v2']['recall']:.1%})",
            },
            "c0_fp_to_c6": {
                "V0": f"{c0_m['v0']['c6_count']}/120 ({c0_m['v0']['c6_count']/120:.1%})",
                "V1": f"{c0_m['v1']['c6_count']}/120 ({c0_m['v1']['c6_count']/120:.1%})",
                "V2": f"{c0_m['v2']['c6_count']}/120 ({c0_m['v2']['c6_count']/120:.1%})",
            },
            "c0_overall_fp_rate": {
                "V0": f"{(120 - c0_m['v0']['c0_count'])}/120 ({(120 - c0_m['v0']['c0_count'])/120:.1%})",
                "V1": f"{(120 - c0_m['v1']['c0_count'])}/120 ({(120 - c0_m['v1']['c0_count'])/120:.1%})",
                "V2": f"{(120 - c0_m['v2']['c0_count'])}/120 ({(120 - c0_m['v2']['c0_count'])/120:.1%})",
            },
            "c3_recall": {
                "V0": f"{c3_m['v0']['recall_count']}/120 ({c3_m['v0']['recall']:.1%})",
                "V1": f"{c3_m['v1']['recall_count']}/120 ({c3_m['v1']['recall']:.1%})",
                "V2": f"{c3_m['v2']['recall_count']}/120 ({c3_m['v2']['recall']:.1%})",
            },
            "c3_to_c6_misattribution": {
                "V0": f"{c3_m['v0']['c6_count']}/120 ({c3_m['v0']['c6_count']/120:.1%})",
                "V1": f"{c3_m['v1']['c6_count']}/120 ({c3_m['v1']['c6_count']/120:.1%})",
                "V2": f"{c3_m['v2']['c6_count']}/120 ({c3_m['v2']['c6_count']/120:.1%})",
            },
            "three_class_macro_f1": {
                "V0": round(macro_f1["v0"], 4),
                "V1": round(macro_f1["v1"], 4),
                "V2": round(macro_f1["v2"], 4),
            },
        },
        "paired_differences": {
            "c6_recall_v1_minus_v0": c6_diff_v1_v0,
            "c6_recall_v2_minus_v0": c6_diff_v2_v0,
            "c0_c6_fp_v1_minus_v0": c0_diff_v1_v0,
            "c0_c6_fp_v2_minus_v0": c0_diff_v2_v0,
            "c3_recall_v1_minus_v0": c3_diff_v1_v0,
            "c3_recall_v2_minus_v0": c3_diff_v2_v0,
            "c3_to_c6_v1_minus_v0": c3_c6_diff_v1_v0,
            "c3_to_c6_v2_minus_v0": c3_c6_diff_v2_v0,
        },
        "per_class_f1": f1_per_class,
        "cohort_breakdowns": {
            "C0": c0_m,
            "C6_candidate": c6_m,
            "C3": c3_m,
        },
        "per_sku_changes": {
            "c6_candidate_v1_changes": c6_v1_changes,
            "c6_candidate_v2_changes": c6_v2_changes,
            "c3_v1_changes": c3_v1_changes,
            "c3_v2_changes": c3_v2_changes,
            "c0_v1_changes": c0_v1_changes,
            "c0_v2_changes": c0_v2_changes,
        },
    }

    report_path = OUT_DIR / "c3_c6_overlap_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full diagnostic report to: {report_path}")

    # CSV Summary Table
    rows = [
        {"Metric": "C6 Recall (candidate)", "V0 (Current)": report["summary_table"]["c6_recall"]["V0"],
         "V1 (Tie-Break)": report["summary_table"]["c6_recall"]["V1"],
         "V2 (Bounded Deferral)": report["summary_table"]["c6_recall"]["V2"]},
        {"Metric": "C0 False Positive to C6", "V0 (Current)": report["summary_table"]["c0_fp_to_c6"]["V0"],
         "V1 (Tie-Break)": report["summary_table"]["c0_fp_to_c6"]["V1"],
         "V2 (Bounded Deferral)": report["summary_table"]["c0_fp_to_c6"]["V2"]},
        {"Metric": "C0 Overall False Positive Rate", "V0 (Current)": report["summary_table"]["c0_overall_fp_rate"]["V0"],
         "V1 (Tie-Break)": report["summary_table"]["c0_overall_fp_rate"]["V1"],
         "V2 (Bounded Deferral)": report["summary_table"]["c0_overall_fp_rate"]["V2"]},
        {"Metric": "C3 Recall (true C3)", "V0 (Current)": report["summary_table"]["c3_recall"]["V0"],
         "V1 (Tie-Break)": report["summary_table"]["c3_recall"]["V1"],
         "V2 (Bounded Deferral)": report["summary_table"]["c3_recall"]["V2"]},
        {"Metric": "C3 misattributed to C6", "V0 (Current)": report["summary_table"]["c3_to_c6_misattribution"]["V0"],
         "V1 (Tie-Break)": report["summary_table"]["c3_to_c6_misattribution"]["V1"],
         "V2 (Bounded Deferral)": report["summary_table"]["c3_to_c6_misattribution"]["V2"]},
        {"Metric": "3-Class Macro-F1 (C0, C3, C6)", "V0 (Current)": f"{report['summary_table']['three_class_macro_f1']['V0']:.4f}",
         "V1 (Tie-Break)": f"{report['summary_table']['three_class_macro_f1']['V1']:.4f}",
         "V2 (Bounded Deferral)": f"{report['summary_table']['three_class_macro_f1']['V2']:.4f}"},
    ]
    df_summary = pd.DataFrame(rows)
    csv_path = OUT_DIR / "c3_c6_overlap_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved CSV summary table to:     {csv_path}")

    print("\n" + "=" * 70)
    print("PAIRED DIAGNOSTIC SUMMARY")
    print(df_summary.to_string(index=False))
    print("=" * 70)
    print(f"V0 Verification against saved diagnostic: {report['v0_verification']['matches_saved_diagnostic']}")
    print(f"  C0 -> C6 FP: {report['v0_verification']['c0_to_c6_fp']}/120 (expected 14)")
    print(f"  C6 Recall:   {report['v0_verification']['c6_candidate_recall']}/120 (expected 31)")
    print(f"Total elapsed: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    run_overlap_diagnostic()
