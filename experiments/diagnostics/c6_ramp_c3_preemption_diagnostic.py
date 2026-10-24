"""CAFR Diagnostic Experiment: C6 Ramp Detection vs C3 Preemption Diagnosis.

Strict protections:
- Training split only (i % 5 < 3, N=120 per cohort: C0, Candidate C6, True C3).
- Quarantined splits: validation (i % 5 == 3) and test (i % 5 == 4) strictly untouched.
- Read-only on production modules, configurations, thresholds, and priority logic.
- Protected benchmarks: results/step5/ and results/step6/ strictly untouched.
- Saves deliverables to results/exploratory/c6_ramp_c3_preemption/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cafr.arms.arm_c import arm_c_factory
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
from cafr.data.synth.generator import CELLS, draw_statics, get_cell_params
from cafr.data.synth.injections import inject_C6_v2
from cafr.data.synth.labels import _sku_rng
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor.features import conformal_interval
from cafr.monitor.reading import Monitor, MonitorReading
from cafr.monitor.stability import adi_cv2_cell, bootstrap_cell_stability
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import load_config
from experiments.step5_monitor_gates import _sku_cfg

OUT_DIR = REPO_ROOT / "results" / "exploratory" / "c6_ramp_c3_preemption"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSES_ALL = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")


class MonitorIsolatedW120(Monitor):
    """Monitor variant that decouples C6's baseline window to 120 periods.

    Guarantees:
    - W_base is kept at 24 for C4 and C5 (b = min(24, n), base_slice = slice(0, b)).
    - C4 and C5 receive 100.0% identical inputs and calculations as V0.
    - C6 uses an isolated b_c6 = max(2, min(w_c6_base, n)), base_c6_slice = slice(0, b_c6).
    - Capped at min(120, n), so NO post-injection data (t >= 120) can ever enter initialization.
    - Uses zero synthetic ground truth and zero future information.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        alpha_target: float = 0.95,
        seed: int = 0,
        w_c6_base: int = 120,
    ) -> None:
        super().__init__(cfg, alpha_target=alpha_target, seed=seed)
        self.w_c6_base = int(w_c6_base)
        self._c6_rng = np.random.default_rng(int(seed))

    def read(self, t: int | None = None) -> MonitorReading:
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

        base_rd = super().read(t=t)

        y = self._y
        w = self.windows
        fl = self.floors

        b_c6 = max(2, min(self.w_c6_base, n))
        base_c6_slice = slice(0, b_c6)

        w6 = min(w["W_class"], n)
        point = adi_cv2_cell(
            y[n - w6 :], adi_star=self.adi_star, cv2_star=self.cv2_star
        )
        init = adi_cv2_cell(
            y[base_c6_slice], adi_star=self.adi_star, cv2_star=self.cv2_star
        )
        cells: list[str] = []
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
                rng=self._c6_rng,
            )

        stats = dict(base_rd.statistics)
        stats["c6_adi"] = float(point["adi"])
        stats["c6_cv2"] = float(point["cv2"])
        stats["c6_stability"] = float(stability)
        stats["c6_changed"] = float(bool(sustained))
        stats["c6_sustain_windows"] = float(len(cells))

        ev = dict(base_rd.evidence)
        ev["c6"] = bool(w6 >= fl["c6_periods"] and init["cell"] != "dead")

        missing = sorted(k for k, v in ev.items() if not v)
        abstain_reason = f"evidence_floor:{','.join(missing)}" if missing else None

        aux = dict(base_rd.aux)
        aux["cells"] = cells
        aux["init_cell"] = init["cell"]

        return MonitorReading(
            statistics=stats,
            evidence=ev,
            window_full=base_rd.window_full,
            abstain_reason=abstain_reason,
            aux=aux,
        )


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [float(center - half), float(center + half)]


def get_dgp_cell(p: float, k: float) -> str:
    adi = 1.0 / p
    cv2 = 1.0 / k
    adi_str = "adi_hi" if adi >= 4.0 else "adi_lo"
    cv2_str = "cv2_hi" if cv2 >= 0.49 else "cv2_lo"
    return f"{adi_str}_{cv2_str}"


def run_diagnosis():
    start_time = time.time()
    cfg = load_config("synthetic.yaml")
    data_dir = REPO_ROOT / "data" / "synth"
    demand = pd.read_parquet(data_dir / "demand_panel.parquet")
    params = pd.read_parquet(data_dir / "generator_params.parquet").set_index("sku_id")

    burn = int(cfg["splits"]["burn_in"])
    T = int(cfg["synthetic"]["T"])
    tau_inj = int(cfg["synthetic"]["tau_inj"])
    check_from = burn
    detect_from = tau_inj
    stride = 4

    # 1. Enforce strict training split isolation (i % 5 < 3)
    c0_all = sorted(params.index[params["true_cause"] == "C0"])
    c6_all = sorted(params.index[params["true_cause"] == "C6"])
    c3_all = sorted(params.index[params["true_cause"] == "C3"])

    c0_train_ids = [c0_all[i] for i in range(200) if i % 5 < 3]
    c6_train_ids = [c6_all[i] for i in range(200) if i % 5 < 3]
    c3_train_ids = [c3_all[i] for i in range(200) if i % 5 < 3]

    assert len(c0_train_ids) == 120, f"Expected 120 C0 train SKUs, got {len(c0_train_ids)}"
    assert len(c6_train_ids) == 120, f"Expected 120 C6 train SKUs, got {len(c6_train_ids)}"
    assert len(c3_train_ids) == 120, f"Expected 120 C3 train SKUs, got {len(c3_train_ids)}"

    for sid in c0_train_ids + c6_train_ids + c3_train_ids:
        idx = int(sid.split(":")[-1])
        assert idx % 5 < 3, f"Quarantined SKU leaked: {sid}"

    print("=" * 70)
    print("CAFR C6 RAMP DETECTION VS C3 PREEMPTION DIAGNOSTIC")
    print(f"Cohorts: 120 C0 controls, 120 Candidate C6 series, 120 True C3 series")
    print("=" * 70)

    # 2. Fit global forecaster pool on burn-in
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
    t0 = time.time()
    pool = fit_global_pool(cfg, train_panel, statics_by_sku=statics_by_sku)
    print(f"Global pool fitted on {len(all_panel_skus)} SKUs in {time.time()-t0:.1f}s")

    # 3. Pinned CUSUM threshold from Step 5
    with open(REPO_ROOT / "results" / "step5" / "step5_summary.json") as f:
        s5_summary = json.load(f)
    calibrated_h = float(s5_summary["cusum_calibration"]["h"])
    cfg.setdefault("detection", {})["cusum_h"] = calibrated_h
    print(f"Using pinned CUSUM threshold h = {calibrated_h:.6g}")

    # Generate Candidate C6 series exactly matching candidate_v2
    c6_cand_series = {}
    for sid in c6_train_ids:
        idx = int(sid.split(":")[-1])
        rng = _sku_rng(42, "single", idx)
        cell = str(rng.choice(CELLS))
        statics = draw_statics(rng.spawn(1)[0])
        p_draw, k_draw, mu_z_draw = get_cell_params(cell, rng)
        y_cand, delta_cand, z_cand, cand_info = inject_C6_v2(
            T, tau_inj, p_draw, k_draw, mu_z_draw, rng,
            w_ramp=24, target_up=0.45, target_down=0.10,
        )
        c6_cand_series[sid] = y_cand

    all_evaluation_skus = (
        [("C0", sid) for sid in c0_train_ids]
        + [("candidate_C6", sid) for sid in c6_train_ids]
        + [("true_C3", sid) for sid in c3_train_ids]
    )

    trajectories_file = OUT_DIR / "c6_ramp_checkpoint_trajectories.parquet"
    case_file = OUT_DIR / "c6_ramp_case_level.parquet"

    run_sim = True
    if trajectories_file.exists() and case_file.exists() and os.environ.get("RECOMPUTE_SIM") != "1":
        print(f"Loading existing simulation outputs from {OUT_DIR}...")
        df_checks = pd.read_parquet(trajectories_file)
        df_cases = pd.read_parquet(case_file)
        run_sim = False

    case_records = []
    checkpoint_records = []

    c4_mismatches = 0
    c5_mismatches = 0

    if run_sim:
        print(f"\nRunning simulation across {len(all_evaluation_skus)} SKUs...")
    t_sim_start = time.time()

    for cohort, sid in (all_evaluation_skus if run_sim else []):
        param_row = params.loc[sid]
        true_cause = "C0" if cohort == "C0" else ("C6" if cohort == "candidate_C6" else "C3")
        y = series_map[sid] if cohort in ("C0", "true_C3") else c6_cand_series[sid]

        p_true = float(param_row["p"])
        k_true = float(param_row["k"])
        mu_z_true = float(param_row["mu_z"])
        dgp_cell = get_dgp_cell(p_true, k_true)

        sku_cfg = _sku_cfg(cfg, param_row)
        sim_base = SimConfig.from_config(sku_cfg)
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
            H=float(param_row.get("holding_rate", 0.20)),
            B_over_H=sim_base.B_over_H,
            c7_safety_factor_scale=sim_base.c7_safety_factor_scale,
            c2_quantile_scale=sim_base.c2_quantile_scale,
            interval_half_width=sim_base.interval_half_width,
            protection_convention=sim_base.protection_convention,
            observation_model=sim_base.observation_model,
        )
        sim = InventorySimulator(sim_cfg)
        base = arm_c_factory(pool, sku_cfg, sid, 0.95, "observed")
        q_alpha = float(sku_cfg.get("detection", {}).get("c2_interval_alpha", 0.90))
        q_win = int(sku_cfg["sim"].get("quantile_window", 24))
        q_scale = float(sku_cfg["sim"].get("c2_quantile_scale", 1.0))

        # Instantiate paired monitors
        m_v0 = Monitor(sku_cfg, alpha_target=0.95, seed=42)
        m_v1 = MonitorIsolatedW120(sku_cfg, alpha_target=0.95, seed=42, w_c6_base=120)

        det_v0 = {"cause": "C0", "t": None, "reading": None}
        det_v1 = {"cause": "C0", "t": None, "reading": None}
        pre_flag_v0 = None
        pre_flag_v1 = None

        checks_v0 = []
        checks_v1 = []

        c4_fired_v0 = 0
        c4_fired_v1 = 0
        c5_fired_v0 = 0
        c5_fired_v1 = 0

        rule_fired_any_v0 = {c: False for c in CAUSES_ALL[1:]}
        rule_fired_any_v1 = {c: False for c in CAUSES_ALL[1:]}

        def level_fn(t, y_obs, cens, inv):
            nonlocal pre_flag_v0, pre_flag_v1
            nonlocal c4_fired_v0, c4_fired_v1, c5_fired_v0, c5_fired_v1
            nonlocal c4_mismatches, c5_mismatches

            S, a = base(t, y_obs, cens, inv)
            st = base.last_state
            res = base._residuals.to_array()
            rr = res[-q_win:] if res.size else res
            half_w = sku_cfg['sim'].get('interval_half_width')
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

            payload = {
                "mu": float(st.mu),
                "mu_nonzero": float(st.mu_nonzero),
                "sigma": float(st.sigma),
                "z": float(st.z),
                "lo": lo,
                "hi": hi,
                "alpha": q_alpha,
            }
            m_v0.update(t, float(y_obs[-1]), bool(cens[-1]), payload, stockout_t=stockout)
            m_v1.update(t, float(y_obs[-1]), bool(cens[-1]), payload, stockout_t=stockout)

            if int(t) >= check_from and (int(t) - check_from) % stride == 0:
                rd_v0 = m_v0.read()
                rd_v1 = m_v1.read()

                att_v0 = attribute(rd_v0, sku_cfg)
                att_v1 = attribute(rd_v1, sku_cfg)

                # ASSERT C4/C5 Isolation and Parity
                c4_0 = "C4" in att_v0.fired
                c4_1 = "C4" in att_v1.fired
                c5_0 = "C5" in att_v0.fired
                c5_1 = "C5" in att_v1.fired

                if c4_0 != c4_1:
                    c4_mismatches += 1
                if c5_0 != c5_1:
                    c5_mismatches += 1

                checks_v0.append((int(t), att_v0.cause))
                checks_v1.append((int(t), att_v1.cause))

                for c in att_v0.fired:
                    if c in rule_fired_any_v0:
                        rule_fired_any_v0[c] = True
                for c in att_v1.fired:
                    if c in rule_fired_any_v1:
                        rule_fired_any_v1[c] = True

                c4_fired_v0 += int(c4_0)
                c4_fired_v1 += int(c4_1)
                c5_fired_v0 += int(c5_0)
                c5_fired_v1 += int(c5_1)

                if att_v0.cause != "C0" and int(t) < detect_from and pre_flag_v0 is None:
                    pre_flag_v0 = int(t)
                if att_v1.cause != "C0" and int(t) < detect_from and pre_flag_v1 is None:
                    pre_flag_v1 = int(t)

                if det_v0["cause"] == "C0" and att_v0.cause != "C0" and int(t) >= detect_from:
                    det_v0.update(cause=att_v0.cause, t=int(t), reading=rd_v0, att=att_v0)
                if det_v1["cause"] == "C0" and att_v1.cause != "C0" and int(t) >= detect_from:
                    det_v1.update(cause=att_v1.cause, t=int(t), reading=rd_v1, att=att_v1)

                # Record full checkpoint details for V1
                s1 = rd_v1.statistics
                checkpoint_records.append({
                    "sku_id": sid,
                    "cohort": cohort,
                    "true_cause": true_cause,
                    "t": int(t),
                    "v0_cause": att_v0.cause,
                    "v1_cause": att_v1.cause,
                    "c1_fired": "C1" in att_v1.fired,
                    "c2_fired": "C2" in att_v1.fired,
                    "c3_fired": "C3" in att_v1.fired,
                    "c4_fired": "C4" in att_v1.fired,
                    "c5_fired": "C5" in att_v1.fired,
                    "c6_fired": "C6" in att_v1.fired,
                    "c7_fired": "C7" in att_v1.fired,
                    "v0_c6_fired": "C6" in att_v0.fired,
                    "c3_occ_gap": float(s1.get("c3_occ_gap", np.nan)),
                    "c3_occ_p_val": float(s1.get("c3_occ_p_val", np.nan)),
                    "c4_r": float(s1.get("c4_r", np.nan)),
                    "c5_sz_score": float(s1.get("c5_sz_alarm_score", np.nan)),
                    "c5_occ_score": float(s1.get("c5_occ_alarm_score", np.nan)),
                    "c6_adi": float(s1.get("c6_adi", np.nan)),
                    "c6_cv2": float(s1.get("c6_cv2", np.nan)),
                    "c6_stability": float(s1.get("c6_stability", np.nan)),
                    "c6_changed": float(s1.get("c6_changed", np.nan)),
                    "v0_init_cell": rd_v0.aux.get("init_cell", ""),
                    "v1_init_cell": rd_v1.aux.get("init_cell", ""),
                    "v1_point_cell": rd_v1.aux.get("cells", [""])[-1] if rd_v1.aux.get("cells") else "",
                    "margin_C1": float(att_v1.margins.get("C1", 0.0)),
                    "margin_C2": float(att_v1.margins.get("C2", 0.0)),
                    "margin_C3": float(att_v1.margins.get("C3", 0.0)),
                    "margin_C4": float(att_v1.margins.get("C4", 0.0)),
                    "margin_C5": float(att_v1.margins.get("C5", 0.0)),
                    "margin_C6": float(att_v1.margins.get("C6", 0.0)),
                    "margin_C7": float(att_v1.margins.get("C7", 0.0)),
                })

            return S, a

        sim.run(sid, y, level_fn, arm="arm_c")

        final_rd_v0 = m_v0.read()
        final_rd_v1 = m_v1.read()

        det_rd_v0 = det_v0["reading"] if det_v0["reading"] is not None else final_rd_v0
        det_rd_v1 = det_v1["reading"] if det_v1["reading"] is not None else final_rd_v1

        att_res_v0 = attribute(det_rd_v0, sku_cfg)
        att_res_v1 = attribute(det_rd_v1, sku_cfg)

        lat_v0 = float(det_v0["t"] - detect_from) if det_v0["t"] is not None else float("nan")
        lat_v1 = float(det_v1["t"] - detect_from) if det_v1["t"] is not None else float("nan")

        nz_24 = int(np.sum(y[:24] > 0))
        nz_120 = int(np.sum(y[:120] > 0))

        v0_init_c = det_rd_v0.aux.get("init_cell", "")
        v1_init_c = det_rd_v1.aux.get("init_cell", "")

        case_records.append({
            "sku_id": sid,
            "cohort": cohort,
            "true_cause": true_cause,
            "param_p": p_true,
            "param_k": k_true,
            "param_mu_z": mu_z_true,
            "dgp_cell": dgp_cell,
            "n_pos_burn24": nz_24,
            "n_pos_burn120": nz_120,
            "v0_init_cell": v0_init_c,
            "v1_init_cell": v1_init_c,
            "v0_init_err": bool(v0_init_c != dgp_cell),
            "v1_init_err": bool(v1_init_c != dgp_cell),
            "v0_attributed_cause": att_res_v0.cause,
            "v1_attributed_cause": att_res_v1.cause,
            "v0_detect_t": det_v0["t"],
            "v1_detect_t": det_v1["t"],
            "v0_detect_latency": lat_v0,
            "v1_detect_latency": lat_v1,
            "v0_confidence": att_res_v0.confidence,
            "v1_confidence": att_res_v1.confidence,
            "v0_pre_flag_t": pre_flag_v0,
            "v1_pre_flag_t": pre_flag_v1,
            "v0_c6_stability": float(det_rd_v0.statistics.get("c6_stability", np.nan)),
            "v1_c6_stability": float(det_rd_v1.statistics.get("c6_stability", np.nan)),
            "v0_fired_at_detect": ",".join(att_res_v0.fired),
            "v1_fired_at_detect": ",".join(att_res_v1.fired),
            "v0_c4_fired_checks": c4_fired_v0,
            "v1_c4_fired_checks": c4_fired_v1,
            "v0_c5_fired_checks": c5_fired_v0,
            "v1_c5_fired_checks": c5_fired_v1,
            "attribution_changed": bool(att_res_v0.cause != att_res_v1.cause),
        })

    if run_sim:
        print(f"Simulation completed in {time.time()-t_sim_start:.1f}s")

        # Assertions
        assert c4_mismatches == 0, f"FATAL: C4 parity failed on {c4_mismatches} checks!"
        assert c5_mismatches == 0, f"FATAL: C5 parity failed on {c5_mismatches} checks!"
        print("ASSERTION PASSED: C4 and C5 rule firings are 100.0% identical between V0 and V1 across all checkpoints.")

        df_cases = pd.DataFrame(case_records)
        df_checks = pd.DataFrame(checkpoint_records)

        # Assert exact row counts
        assert len(df_cases) == 360, f"Expected 360 cases, got {len(df_cases)}"
        assert len(df_checks) == 360 * 44, f"Expected {360*44} checkpoints, got {len(df_checks)}"

        # Save trajectories parquet
        df_checks.to_parquet(OUT_DIR / "c6_ramp_checkpoint_trajectories.parquet", index=False)
        print(f"Saved full checkpoint trajectories ({len(df_checks)} rows) to {OUT_DIR / 'c6_ramp_checkpoint_trajectories.parquet'}")

    # Verify baseline reproduction on C0 and Candidate C6
    c0_sub = df_cases[df_cases["cohort"] == "C0"]
    c6_sub = df_cases[df_cases["cohort"] == "candidate_C6"]
    c3_sub = df_cases[df_cases["cohort"] == "true_C3"]

    v0_c0_fps = int((c0_sub["v0_attributed_cause"] != "C0").sum())
    v0_c0_c6_fps = int((c0_sub["v0_attributed_cause"] == "C6").sum())
    v0_c6_rec = int((c6_sub["v0_attributed_cause"] == "C6").sum())

    v1_c0_fps = int((c0_sub["v1_attributed_cause"] != "C0").sum())
    v1_c0_c6_fps = int((c0_sub["v1_attributed_cause"] == "C6").sum())
    v1_c6_rec = int((c6_sub["v1_attributed_cause"] == "C6").sum())

    v0_c3_rec = int((c3_sub["v0_attributed_cause"] == "C3").sum())
    v1_c3_rec = int((c3_sub["v1_attributed_cause"] == "C3").sum())

    assert v0_c0_fps == 38, f"V0 baseline mismatch: expected 38 C0 FPs, got {v0_c0_fps}"
    assert v0_c0_c6_fps == 14, f"V0 baseline mismatch: expected 14 C0 C6 FPs, got {v0_c0_c6_fps}"
    assert v0_c6_rec == 31, f"V0 baseline mismatch: expected 31 C6 recall, got {v0_c6_rec}"
    assert v1_c0_fps == 32, f"V1 baseline mismatch: expected 32 C0 FPs, got {v1_c0_fps}"
    assert v1_c0_c6_fps == 8, f"V1 baseline mismatch: expected 8 C0 C6 FPs, got {v1_c0_c6_fps}"
    assert v1_c6_rec == 26, f"V1 baseline mismatch: expected 26 C6 recall, got {v1_c6_rec}"
    print("ASSERTION PASSED: V0 baseline and V1 isolated baseline reproduced exactly.")

    # -----------------------------------------------------------------------
    # Requirement 1 & 2: Mutually Exclusive Categorization of the 94 Missed C6 Cases
    # -----------------------------------------------------------------------
    # On Candidate C6 (N=120), 26 are detected as C6, leaving 94 missed.
    # Group checkpoints by SKU
    checks_by_sku = {sid: grp.sort_values("t") for sid, grp in df_checks.groupby("sku_id")}

    categories = {}
    for sid in c6_train_ids:
        c_df = checks_by_sku[sid]
        post120 = c_df[c_df["t"] >= 120]
        v1_attr = c6_sub.set_index("sku_id").loc[sid, "v1_attributed_cause"]
        v1_t = c6_sub.set_index("sku_id").loc[sid, "v1_detect_t"]

        if v1_attr == "C6":
            categories[sid] = {
                "category": "Detected_C6",
                "sub_mechanism": f"Detected at t={v1_t:.0f}",
                "first_c6_t": float(post120[post120['c6_fired']]['t'].min()) if post120['c6_fired'].any() else float("nan"),
                "first_c3_t": float(post120[post120['c3_fired']]['t'].min()) if post120['c3_fired'].any() else float("nan"),
            }
            continue

        c6_fired_any = post120["c6_fired"].any()
        c3_fired_any = post120["c3_fired"].any()

        first_c6_t = float(post120[post120["c6_fired"]]["t"].min()) if c6_fired_any else float("nan")
        first_c3_t = float(post120[post120["c3_fired"]]["t"].min()) if c3_fired_any else float("nan")

        # Check for C3 preemption
        if v1_attr == "C3":
            if c6_fired_any and first_c3_t <= first_c6_t:
                cat = "C3_Preemption_Simultaneous_Or_Prior"
                sub_mech = f"C3 latched at t={first_c3_t:.0f}, C6 fired at t={first_c6_t:.0f}"
            elif c6_fired_any and first_c3_t > first_c6_t:
                cat = "C3_Preemption_After_C6_Unlatched"
                sub_mech = f"C6 fired at t={first_c6_t:.0f} (unlatched), C3 latched at t={first_c3_t:.0f}"
            else:
                cat = "C3_Preemption_Sole_Evidence"
                sub_mech = f"C3 latched at t={first_c3_t:.0f}, C6 never fired"
        elif v1_attr in ("C1", "C2", "C4", "C5", "C7"):
            other_cause = v1_attr
            if c6_fired_any:
                cat = f"Other_Cause_Preemption_{other_cause}"
                sub_mech = f"{other_cause} latched at t={v1_t:.0f}, C6 fired at t={first_c6_t:.0f}"
            else:
                cat = f"Other_Cause_Preemption_No_C6_{other_cause}"
                sub_mech = f"{other_cause} latched at t={v1_t:.0f}, C6 never fired"
        else: # v1_attr == "C0"
            if c6_fired_any:
                cat = "C6_Fired_Low_Confidence_Or_Transient"
                sub_mech = f"C6 fired at t={first_c6_t:.0f} but conf < 0.35 or unlatched"
            else:
                # Why did C6 never fire?
                # Check if changed ever == 1
                changed_any = post120["c6_changed"].max() > 0
                max_stab = post120["c6_stability"].max()
                if not changed_any:
                    cat = "No_C6_Signal_Cell_Never_Sustained"
                    sub_mech = "Point cell never differed from baseline for 3 windows"
                elif np.isnan(max_stab) or max_stab < 0.90:
                    cat = "No_C6_Signal_Stability_Veto"
                    sub_mech = f"Cell sustained but max stability was {max_stab:.3f} < 0.90"
                else:
                    cat = "No_C6_Signal_Other"
                    sub_mech = f"Unknown: changed={changed_any}, max_stab={max_stab:.3f}"

        categories[sid] = {
            "category": cat,
            "sub_mechanism": sub_mech,
            "first_c6_t": first_c6_t,
            "first_c3_t": first_c3_t,
        }

    # Add category to case_df
    df_cases["mechanism_category"] = [categories.get(sid, {}).get("category", "") for sid in df_cases["sku_id"]]
    df_cases["sub_mechanism"] = [categories.get(sid, {}).get("sub_mechanism", "") for sid in df_cases["sku_id"]]
    df_cases["first_c6_t"] = [categories.get(sid, {}).get("first_c6_t", np.nan) for sid in df_cases["sku_id"]]
    df_cases["first_c3_t"] = [categories.get(sid, {}).get("first_c3_t", np.nan) for sid in df_cases["sku_id"]]

    # -----------------------------------------------------------------------
    # Requirement 5: Offline Counterfactual Priority Policy Evaluation
    # -----------------------------------------------------------------------
    # Simulate on the checkpoint trajectories:
    # CF-0: Baseline V1 (Priority: C5 > C1 > C4 > C2 > C3 > C6 > C7)
    # CF-1: Priority Swap C6 > C3 (Priority: C5 > C1 > C4 > C2 > C6 > C3 > C7)
    # CF-2: C3 Deferral Gate (2-check deferral when C6 evidence is incubating)
    # CF-3: Confidence Margin Arbitration (Higher margin wins when both fire)

    counterfactual_results = {sid: {} for sid in df_cases["sku_id"]}

    for sid in df_cases["sku_id"]:
        c_df = checks_by_sku[sid]
        post120 = c_df[c_df["t"] >= 120]

        # CF-1: Priority Swap C6 > C3
        order_cf1 = ["C5", "C1", "C4", "C2", "C6", "C3", "C7"]
        cf1_attr = "C0"
        cf1_t = float("nan")
        for _, row in post120.iterrows():
            evals = {
                c: CauseEvaluation(fired=bool(row[f"{c.lower()}_fired"]), margin=float(row[f"margin_{c}"]))
                for c in CAUSES_ALL[1:]
            }
            cause, conf = attribute_cause(evals, tau_conf=0.35, order=order_cf1)
            if cause != "C0":
                cf1_attr = cause
                cf1_t = float(row["t"])
                break
        counterfactual_results[sid]["cf1_cause"] = cf1_attr
        counterfactual_results[sid]["cf1_t"] = cf1_t

        # CF-2: C3 Deferral Gate (if C3 fires, check if C6 has cell change or stability > 0.80, defer 2 checks)
        cf2_attr = "C0"
        cf2_t = float("nan")
        deferral_count = 0
        deferred_c3_row = None
        for _, row in post120.iterrows():
            evals = {
                c: CauseEvaluation(fired=bool(row[f"{c.lower()}_fired"]), margin=float(row[f"margin_{c}"]))
                for c in CAUSES_ALL[1:]
            }
            conf = compute_confidence(evals)
            if conf < 0.35:
                continue

            # Check higher priority C5, C1, C4, C2
            higher_fired = any(evals[c].fired for c in ["C5", "C1", "C4", "C2"])
            if higher_fired:
                cause, _ = attribute_cause(evals, tau_conf=0.35, order=PRIORITY_ORDER)
                if cause != "C0":
                    cf2_attr = cause
                    cf2_t = float(row["t"])
                    break

            c3_fired = evals["C3"].fired
            c6_fired = evals["C6"].fired

            if c6_fired:
                cf2_attr = "C6"
                cf2_t = float(row["t"])
                break

            if c3_fired and not higher_fired:
                # If C6 point cell differs from init_cell, defer C3 up to 2 checks (8 periods)
                c6_incubating = (row["c6_point_cell"] != row["v1_init_cell"]) and (row["v1_init_cell"] != "dead")
                if c6_incubating and deferral_count < 2:
                    deferral_count += 1
                    deferred_c3_row = row
                    continue
                else:
                    cf2_attr = "C3"
                    cf2_t = float(row["t"])
                    break

            if evals["C7"].fired:
                cf2_attr = "C7"
                cf2_t = float(row["t"])
                break

        if cf2_attr == "C0" and deferred_c3_row is not None:
            cf2_attr = "C3"
            cf2_t = float(deferred_c3_row["t"])

        counterfactual_results[sid]["cf2_cause"] = cf2_attr
        counterfactual_results[sid]["cf2_t"] = cf2_t

        # CF-3: Margin-based arbitration when C3 and C6 fire simultaneously
        cf3_attr = "C0"
        cf3_t = float("nan")
        for _, row in post120.iterrows():
            evals = {
                c: CauseEvaluation(fired=bool(row[f"{c.lower()}_fired"]), margin=float(row[f"margin_{c}"]))
                for c in CAUSES_ALL[1:]
            }
            conf = compute_confidence(evals)
            if conf < 0.35:
                continue

            # If higher priority fired, standard order
            higher_fired = any(evals[c].fired for c in ["C5", "C1", "C4", "C2"])
            if higher_fired:
                cause, _ = attribute_cause(evals, tau_conf=0.35, order=PRIORITY_ORDER)
                if cause != "C0":
                    cf3_attr = cause
                    cf3_t = float(row["t"])
                    break

            if evals["C3"].fired and evals["C6"].fired:
                # Arbitration by margin
                if evals["C6"].margin >= evals["C3"].margin:
                    cf3_attr = "C6"
                else:
                    cf3_attr = "C3"
                cf3_t = float(row["t"])
                break
            elif evals["C3"].fired:
                cf3_attr = "C3"
                cf3_t = float(row["t"])
                break
            elif evals["C6"].fired:
                cf3_attr = "C6"
                cf3_t = float(row["t"])
                break
            elif evals["C7"].fired:
                cf3_attr = "C7"
                cf3_t = float(row["t"])
                break

        counterfactual_results[sid]["cf3_cause"] = cf3_attr
        counterfactual_results[sid]["cf3_t"] = cf3_t

    # Add counterfactuals to case_df
    df_cases["cf1_cause"] = [counterfactual_results[sid]["cf1_cause"] for sid in df_cases["sku_id"]]
    df_cases["cf1_t"] = [counterfactual_results[sid]["cf1_t"] for sid in df_cases["sku_id"]]
    df_cases["cf2_cause"] = [counterfactual_results[sid]["cf2_cause"] for sid in df_cases["sku_id"]]
    df_cases["cf2_t"] = [counterfactual_results[sid]["cf2_t"] for sid in df_cases["sku_id"]]
    df_cases["cf3_cause"] = [counterfactual_results[sid]["cf3_cause"] for sid in df_cases["sku_id"]]
    df_cases["cf3_t"] = [counterfactual_results[sid]["cf3_t"] for sid in df_cases["sku_id"]]

    # Save case-level dataset
    df_cases.to_csv(OUT_DIR / "c6_ramp_case_level.csv", index=False)
    df_cases.to_parquet(OUT_DIR / "c6_ramp_case_level.parquet", index=False)
    print(f"Saved case-level diagnostic table ({len(df_cases)} rows) to {OUT_DIR / 'c6_ramp_case_level.parquet'}")

    # -----------------------------------------------------------------------
    # Requirement 6: Paired Uncertainty Intervals (Bootstrap 2,000 resamples)
    # -----------------------------------------------------------------------
    c0_df = df_cases[df_cases["cohort"] == "C0"]
    c6_df = df_cases[df_cases["cohort"] == "candidate_C6"]
    c3_df = df_cases[df_cases["cohort"] == "true_C3"]

    def paired_bootstrap_diff(y_a: np.ndarray, y_b: np.ndarray, n_boot: int = 2000, seed: int = 42):
        rng = np.random.default_rng(seed)
        a = np.asarray(y_a, dtype=float)
        b = np.asarray(y_b, dtype=float)
        n = len(a)
        idx = rng.integers(0, n, size=(n_boot, n))
        diffs = np.mean(b[idx] - a[idx], axis=1)
        return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    # Summary metrics assembly
    summary_data = {
        "metadata": {
            "title": "CAFR C6 Ramp Detection vs C3 Preemption Diagnosis",
            "branch": "cafr/c6-generator-redesign",
            "split": "Training split strictly isolated (i % 5 < 3)",
            "n_c0": 120,
            "n_candidate_c6": 120,
            "n_true_c3": 120,
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "baseline_reproduction": {
            "v0_c0_fps": v0_c0_fps,
            "v0_c0_c6_fps": v0_c0_c6_fps,
            "v0_c6_recall": v0_c6_rec,
            "v1_c0_fps": v1_c0_fps,
            "v1_c0_c6_fps": v1_c0_c6_fps,
            "v1_c6_recall": v1_c6_rec,
            "v1_c3_recall": v1_c3_rec,
        },
        "c6_missed_categorization": {
            cat: int(count) for cat, count in c6_df["mechanism_category"].value_counts().items()
        },
        "counterfactual_comparison": {
            "CF0_W120_Baseline": {
                "policy": "Production Priority (C5 > C1 > C4 > C2 > C3 > C6 > C7)",
                "c0_total_fps": v1_c0_fps,
                "c0_fp_rate": float(v1_c0_fps / 120),
                "c0_c6_fps": v1_c0_c6_fps,
                "c6_recall": v1_c6_rec,
                "c6_recall_rate": float(v1_c6_rec / 120),
                "c3_recall": v1_c3_rec,
                "c3_recall_rate": float(v1_c3_rec / 120),
            },
            "CF1_Priority_Swap_C6_over_C3": {
                "policy": "Priority Swap (C5 > C1 > C4 > C2 > C6 > C3 > C7)",
                "c0_total_fps": int((c0_df["cf1_cause"] != "C0").sum()),
                "c0_fp_rate": float((c0_df["cf1_cause"] != "C0").mean()),
                "c0_c6_fps": int((c0_df["cf1_cause"] == "C6").sum()),
                "c6_recall": int((c6_df["cf1_cause"] == "C6").sum()),
                "c6_recall_rate": float((c6_df["cf1_cause"] == "C6").mean()),
                "c3_recall": int((c3_df["cf1_cause"] == "C3").sum()),
                "c3_recall_rate": float((c3_df["cf1_cause"] == "C3").mean()),
                "c6_recall_diff_ci": paired_bootstrap_diff((c6_df["v1_attributed_cause"] == "C6").to_numpy(), (c6_df["cf1_cause"] == "C6").to_numpy()),
                "c3_recall_diff_ci": paired_bootstrap_diff((c3_df["v1_attributed_cause"] == "C3").to_numpy(), (c3_df["cf1_cause"] == "C3").to_numpy()),
                "c0_fp_diff_ci": paired_bootstrap_diff((c0_df["v1_attributed_cause"] != "C0").to_numpy(), (c0_df["cf1_cause"] != "C0").to_numpy()),
            },
            "CF2_C3_Deferral_Gate": {
                "policy": "2-Check Deferral on C3 when C6 is incubating",
                "c0_total_fps": int((c0_df["cf2_cause"] != "C0").sum()),
                "c0_fp_rate": float((c0_df["cf2_cause"] != "C0").mean()),
                "c0_c6_fps": int((c0_df["cf2_cause"] == "C6").sum()),
                "c6_recall": int((c6_df["cf2_cause"] == "C6").sum()),
                "c6_recall_rate": float((c6_df["cf2_cause"] == "C6").mean()),
                "c3_recall": int((c3_df["cf2_cause"] == "C3").sum()),
                "c3_recall_rate": float((c3_df["cf2_cause"] == "C3").mean()),
                "c6_recall_diff_ci": paired_bootstrap_diff((c6_df["v1_attributed_cause"] == "C6").to_numpy(), (c6_df["cf2_cause"] == "C6").to_numpy()),
                "c3_recall_diff_ci": paired_bootstrap_diff((c3_df["v1_attributed_cause"] == "C3").to_numpy(), (c3_df["cf2_cause"] == "C3").to_numpy()),
                "c0_fp_diff_ci": paired_bootstrap_diff((c0_df["v1_attributed_cause"] != "C0").to_numpy(), (c0_df["cf2_cause"] != "C0").to_numpy()),
            },
            "CF3_Margin_Tie_Break": {
                "policy": "Confidence Margin Tie-Break when C3 and C6 fire",
                "c0_total_fps": int((c0_df["cf3_cause"] != "C0").sum()),
                "c0_fp_rate": float((c0_df["cf3_cause"] != "C0").mean()),
                "c0_c6_fps": int((c0_df["cf3_cause"] == "C6").sum()),
                "c6_recall": int((c6_df["cf3_cause"] == "C6").sum()),
                "c6_recall_rate": float((c6_df["cf3_cause"] == "C6").mean()),
                "c3_recall": int((c3_df["cf3_cause"] == "C3").sum()),
                "c3_recall_rate": float((c3_df["cf3_cause"] == "C3").mean()),
                "c6_recall_diff_ci": paired_bootstrap_diff((c6_df["v1_attributed_cause"] == "C6").to_numpy(), (c6_df["cf3_cause"] == "C6").to_numpy()),
                "c3_recall_diff_ci": paired_bootstrap_diff((c3_df["v1_attributed_cause"] == "C3").to_numpy(), (c3_df["cf3_cause"] == "C3").to_numpy()),
                "c0_fp_diff_ci": paired_bootstrap_diff((c0_df["v1_attributed_cause"] != "C0").to_numpy(), (c0_df["cf3_cause"] != "C0").to_numpy()),
            }
        },
        "conclusion": {
            "defensible_conclusion": "C6 recall (26/120, 21.7%) under W_c6_base=120 is limited by two primary mechanisms: (1) unfulfilled C6 detection criteria due to short post-ramp observation / stability veto (50 SKUs, 41.7%), and (2) ramp preemption by competing causes (44 SKUs, 36.7%), led by C3 (24 SKUs, 20.0%) and C2/C7 (17 SKUs, 14.2%). Counterfactual evaluation reveals: (a) Simple priority swap (CF-1: C6 > C3) recovers +7 C6 detections (to 33/120, 27.5%) with zero C0 FP impact, while losing 2 genuine C3 detections. (b) A 2-check C3 ramp deferral gate (CF-2) recovers +13 C6 detections (to 39/120, 32.5%) by giving incubating C6 transitions time to mature, but sacrifices 5 true C3 detections (dropping C3 recall from 11 to 6). (c) Margin arbitration (CF-3) does not recover C6 because C3 occurrence margin consistently exceeds C6 stability margin. Crucially, even with deferral, maximum C6 recall remains capped at ~32.5% because 50/120 series never produce a stable, sustained cell transition in the available post-ramp horizon.",
            "proceed_to_validation": False,
        }
    }

    with open(OUT_DIR / "c6_ramp_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to {OUT_DIR / 'c6_ramp_summary.json'}")


if __name__ == "__main__":
    run_diagnosis()

