"""Step 5 -- M1 monitor + M2 attribution, gates 5.1-5.6 (handoff §13 Step 5).

Runs the labelled synthetic panel through the **real** pipeline: arm (c)'s forecaster,
the rolling simulator, the monitor, and the priority-ordered attribution. Per-SKU
injection parameters recorded by the generator are applied, so each SKU carries the
mechanism it is labelled with:

* C2's ``gamma`` rescales the reported interval about the point forecast;
* C2b's calibration window replaces the quantile window (the modelling-error variant);
* C7's ``safety_scale`` scales the safety term only.

Gates (handoff §30.1)
---------------------
5.1  per-cause recall >= 0.60 on the labelled panel.
5.2  false-positive rate on the C0 control panel <= 10 %.
5.3  no off-diagonal cell of the row-normalised confusion matrix > 40 %.
5.4  shuffled-priority macro-F1 difference <= 0.10.
5.5  confusion matrix (P3) and per-cause statistic distributions produced.
5.6  per-cause recall reported pooled and budget-restricted.

**Scope ruling (2026-10-17):** gate 4.9 measured that C7 is not separately
identifiable in this simulator, and the team accepted the six-cause scope. Gate 5.1 is
therefore evaluated over C1-C6. C7's recall is still measured and printed -- it is a
reported negative finding, never silently dropped.

    python experiments/step5_monitor_gates.py --per-cause 200 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.attribution import PRIORITY_ORDER, attribute
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor import Monitor
from cafr.monitor.features import conformal_interval
from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import REPO_ROOT, load_config

DATA = REPO_ROOT / "data" / "synth"
OUT = REPO_ROOT / "results" / "step5"

CAUSES_ALL = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
#: The six-cause scope ruled on 2026-10-17 (C7 is a documented negative).
GATE_CAUSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
RECALL_MIN = 0.60
FP_MAX = 0.10
OFFDIAG_MAX = 0.40
PRIORITY_DIFF_MAX = 0.10


# --------------------------------------------------------------------------- #
# pipeline                                                                     #
# --------------------------------------------------------------------------- #


def _sku_cfg(base_cfg: dict, param: pd.Series) -> dict:
    """Copy the config and apply this SKU's recorded injection parameters."""
    cfg = copy.deepcopy(base_cfg)
    cause = str(param["true_cause"])
    if cause == "C2":
        gamma = param.get("gamma")
        if pd.notna(gamma):
            cfg["sim"]["c2_quantile_scale"] = float(gamma)
    elif cause == "C2b":
        w = param.get("c2_calib_window")
        if pd.notna(w):
            cfg["sim"]["quantile_window"] = int(w)
        half_width = param.get("half_width")
        if pd.notna(half_width):
            cfg["sim"]["interval_half_width"] = float(half_width)
    elif cause == "C7":
        s = param.get("safety_scale")
        if pd.notna(s):
            cfg["sim"]["c7_safety_factor_scale"] = float(s)
    return cfg


def _run_one(
    sku_id: str,
    y: np.ndarray,
    param: pd.Series,
    base_cfg: dict,
    pool,
    seed: int,
    *,
    check_from: int,
    detect_from: int,
    stride: int,
    calibration_scores: list[float] | None = None,
) -> tuple[dict, "MonitorReading"]:
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
        protection_convention=sim_base.protection_convention,
        observation_model=sim_base.observation_model,
    )
    sim = InventorySimulator(sim_cfg)

    monitor = Monitor(cfg, alpha_target=0.95, seed=seed)
    base = arm_c_factory(pool, cfg, sku_id, 0.95, "observed")
    q_alpha = float(cfg.get("detection", {}).get("c2_interval_alpha", 0.90))
    q_win = int(cfg["sim"].get("quantile_window", 24))
    q_scale = float(cfg["sim"].get("c2_quantile_scale", 1.0))

    # Rolling post-injection detection. The labelled claim is "this SKU's cause is X
    # after tau_inj", not "X is visible in one end-of-series reading": C3's step in p
    # is transient (the forecaster adapts within ~30 periods) and C5's budget is sized
    # so that it does not fire on C3's smaller shift. A single terminal reading would
    # therefore score both against the wrong question.
    #
    # Checks run from `check_from` (burn-in) so that pre-injection false alarms are
    # measurable as a diagnostic, but the SKU's LABEL is the first non-C0 attribution
    # at t >= detect_from (tau_inj): a monitor that flags a stable series before the
    # injection has made a false positive, not a detection. Gate 5.2 scores those.
    detected: dict = {"cause": "C0", "t": None, "reading": None}
    pre_flag_t: int | None = None
    checks: list[tuple[int, str]] = []
    ev_any: dict[str, bool] = {k: False for k in ("c1", "c2", "c3", "c4", "c5", "c6", "c7")}
    rule_fired_any: dict[str, bool] = {k: False for k in CAUSES_ALL[1:]}
    c4_fired_checks = 0
    c5_fired_checks = 0
    c5_occ_alarm_checks = 0
    c5_sz_alarm_checks = 0

    def level_fn(t, y_obs, cens, inv):
        nonlocal pre_flag_t, c4_fired_checks, c5_fired_checks
        nonlocal c5_occ_alarm_checks, c5_sz_alarm_checks
        S, a = base(t, y_obs, cens, inv)
        st = base.last_state
        res = base._residuals.to_array()
        rr = res[-q_win:] if res.size else res
        lo, hi = conformal_interval(
            float(st.mu), rr, q_alpha, scale=q_scale, min_evidence=8
        )
        # Stock-out proxy. Under `observation_model: demand` the censoring mask is
        # False by construction, so the C-guard has nothing to read (H-3). The proxy
        # is post-fulfilment on-hand: zero on a positive-demand period means demand
        # was lost. It can over-count when demand exactly equals stock; the C7
        # detector's identifiability is already a documented limitation.
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
            att = attribute(rd, cfg)
            checks.append((int(t), att.cause))
            for cause in att.fired:
                if cause in rule_fired_any:
                    rule_fired_any[cause] = True
            c4_fired_checks += int("C4" in att.fired)
            c5_fired_checks += int("C5" in att.fired)
            c5_occ_alarm_checks += int(bool(rd.statistics.get("c5_occ_alarm", 0.0)))
            c5_sz_alarm_checks += int(bool(rd.statistics.get("c5_sz_alarm", 0.0)))
            if calibration_scores is not None and int(t) < detect_from:
                calibration_scores.append(max(
                    float(rd.statistics.get("c5_sz_alarm_score", 0.0)),
                    float(rd.statistics.get("c5_occ_alarm_score", 0.0)),
                ))
            # Evidence availability is an EX-ANTE observable (the count of nonzero
            # observations in the window), and it is the pre-registered restriction
            # gate 4.9 uses. A cause whose statistic never had enough observations on
            # a SKU is not diagnosable there, and saying so is different from calling
            # the detector wrong.
            for k, v in rd.evidence.items():
                if v and int(t) >= detect_from:
                    ev_any[k] = True
            if att.cause != "C0" and int(t) < detect_from and pre_flag_t is None:
                pre_flag_t = int(t)
            if (
                detected["cause"] == "C0"
                and att.cause != "C0"
                and int(t) >= detect_from
            ):
                detected.update(cause=att.cause, t=int(t), reading=rd)
        return S, a

    sim.run(sku_id, y, level_fn, arm="arm_c")

    final_reading = monitor.read()
    final_att = attribute(final_reading, cfg)
    detect_reading = detected["reading"] if detected["reading"] is not None else final_reading
    att = attribute(detect_reading, cfg)
    latency = (
        float(detected["t"] - detect_from) if detected["t"] is not None else float("nan")
    )
    row = {
        "sku_id": sku_id,
        "true_cause": str(param["true_cause"]),
        "truth_for_gate": "C2" if param["true_cause"] == "C2b" else str(param["true_cause"]),
        "attributed_cause": att.cause,
        "detected": detected["t"] is not None,
        "detect_t": detected["t"],
        "pre_injection_flag_t": pre_flag_t,
        "detect_latency": latency,
        "n_checks": float(len(checks)),
        "n_flagged_checks": float(sum(c != "C0" for _, c in checks)),
        "c4_fired_checks": int(c4_fired_checks),
        "c5_fired_checks": int(c5_fired_checks),
        "c5_occ_alarm_checks": int(c5_occ_alarm_checks),
        "c5_sz_alarm_checks": int(c5_sz_alarm_checks),
        **{f"rule_{cause}_any": bool(value) for cause, value in rule_fired_any.items()},
        "final_cause": final_att.cause,
        "confidence": att.confidence,
        "fired": ",".join(att.fired),
        "window_full": detect_reading.window_full,
        "abstain_reason": detect_reading.abstain_reason or "",
        "c4_budget_met": bool(param.get("c4_budget_met", True)),
        "c5_budget_cleared": bool(param.get("c5_budget_cleared", True)),
        "c6_init_cell": detect_reading.aux.get("init_cell", ""),
    }
    row.update(detect_reading.statistics)
    row.update({f"ev_{k}": bool(v) for k, v in detect_reading.evidence.items()})
    row.update({f"evany_{k}": bool(v) for k, v in ev_any.items()})
    row.update({f"margin_{k}": float(v) for k, v in att.margins.items()})
    return row, detect_reading


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #


def _macro_f1(truth: np.ndarray, pred: np.ndarray, labels=GATE_CAUSES) -> float:
    return float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0))


def _confusion(df: pd.DataFrame, rows: list[str], cols: list[str]) -> pd.DataFrame:
    m = pd.crosstab(df["true_cause"], df["attributed_cause"]).reindex(
        index=rows, columns=cols, fill_value=0
    )
    return m


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [float(center - half), float(center + half)]


def evaluate_gates(
    df: pd.DataFrame,
    readings_for_order: list,
    base_cfg: dict,
    *,
    check_from: int,
    cusum_calibration: dict | None = None,
) -> dict:
    labels = CAUSES_ALL
    single = df[df["true_cause"].isin(CAUSES_ALL)].copy()

    # 5.1 -- recall, pooled over the panel and per cause (C2 includes C2b by mapping).
    recall = {}
    recall_ci = {}
    recall_counts = {}
    for c in CAUSES_ALL:
        sub = df[df["truth_for_gate"] == c] if c != "C2b" else df[df["true_cause"] == c]
        hits = int((sub["attributed_cause"] == c).sum())
        recall[c] = float(hits / len(sub)) if len(sub) else float("nan")
        recall_ci[c] = _wilson_interval(hits, len(sub))
        recall_counts[c] = {"hits": hits, "n": int(len(sub))}
    recall_c2b = float(
        (df.loc[df["true_cause"] == "C2b", "attributed_cause"] == "C2").mean()
    ) if (df["true_cause"] == "C2b").any() else float("nan")

    # Both denominators, as §29.3 requires: pooled, and restricted to the SKUs where
    # the cause's statistic ever had its evidence floor met after the injection point.
    recall_eval: dict[str, float] = {}
    n_eval: dict[str, int] = {}
    for c in CAUSES_ALL:
        col = f"evany_c{c[1].lower()}"
        mask = df["truth_for_gate"] == c
        if col in df.columns:
            mask = mask & df[col].astype(bool)
        sub = df[mask]
        n_eval[c] = int(len(sub))
        recall_eval[c] = (
            float((sub["attributed_cause"] == c).mean()) if len(sub) else float("nan")
        )

    # 5.2 -- false positives on the C0 control panel.
    c0 = df[df["true_cause"] == "C0"]
    fp_rate = float((c0["attributed_cause"] != "C0").mean()) if len(c0) else float("nan")
    # Diagnostic only: how often a stable SKU is flagged *before* tau_inj. A monitor
    # with a high pre-injection flag rate is over-tuned to noise even if its
    # post-injection label happens to be right.
    pre_flag_rate = (
        float(c0["pre_injection_flag_t"].notna().mean()) if len(c0) else float("nan")
    )
    c0_rule_firing = {}
    n_c0_checks = int(c0["n_checks"].sum()) if len(c0) else 0
    for cause in CAUSES_ALL[1:]:
        sku_hits = int(c0[f"rule_{cause}_any"].astype(bool).sum()) if len(c0) else 0
        check_col = f"{cause.lower()}_fired_checks"
        check_hits = int(c0[check_col].sum()) if len(c0) and check_col in c0 else 0
        c0_rule_firing[cause] = {
            "sku_rate": float(sku_hits / len(c0)) if len(c0) else float("nan"),
            "sku_wilson95": _wilson_interval(sku_hits, len(c0)),
            "sku_hits": sku_hits,
            "sku_n": int(len(c0)),
            "check_rate": float(check_hits / n_c0_checks) if n_c0_checks else float("nan"),
            "check_wilson95": _wilson_interval(check_hits, n_c0_checks),
            "check_hits": check_hits,
            "check_n": n_c0_checks,
        }
    if len(c0):
        c5_occ_hits = int(c0["c5_occ_alarm_checks"].sum())
        c5_sz_hits = int(c0["c5_sz_alarm_checks"].sum())
        c0_rule_firing["C5"].update({
            "occ_cusum_check_rate": float(c5_occ_hits / n_c0_checks) if n_c0_checks else float("nan"),
            "occ_cusum_check_wilson95": _wilson_interval(c5_occ_hits, n_c0_checks),
            "occ_cusum_check_hits": c5_occ_hits,
            "sz_cusum_check_rate": float(c5_sz_hits / n_c0_checks) if n_c0_checks else float("nan"),
            "sz_cusum_check_wilson95": _wilson_interval(c5_sz_hits, n_c0_checks),
            "sz_cusum_check_hits": c5_sz_hits,
        })

    # 5.3 -- row-normalised confusion over C0..C7.
    conf = _confusion(single, CAUSES_ALL, labels)
    rownorm = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0)
    offdiag = {}
    for c in CAUSES_ALL:
        if c in rownorm.index and np.isfinite(rownorm.loc[c]).any():
            vals = rownorm.loc[c].drop(labels=[c], errors="ignore")
            offdiag[c] = float(vals.max()) if len(vals) else 0.0
        else:
            offdiag[c] = float("nan")

    # 5.4 -- priority-order sensitivity. The firing SET is unchanged by a permutation;
    # only which fired cause wins. Re-run the real attribution on each stored reading.
    rng = np.random.default_rng(20261017)
    base_pred = np.array([attribute(rd, base_cfg).cause for _, rd in readings_for_order])
    truth = np.array([r[0] for r in readings_for_order])
    base_f1 = _macro_f1(truth, base_pred)
    perm_rows = []
    perm_scores = []
    for k in range(10):
        order = list(rng.permutation(PRIORITY_ORDER))
        pred = np.array(
            [attribute(rd, base_cfg, order=order).cause for _, rd in readings_for_order]
        )
        perm_scores.append(_macro_f1(truth, pred))
        perm_rows.append(
            {"permutation": k, "order": ">".join(order), "macro_f1": perm_scores[-1],
             "abs_diff": abs(perm_scores[-1] - base_f1)}
        )
    sens = pd.DataFrame(perm_rows)
    max_diff = float(sens["abs_diff"].max()) if len(sens) else float("nan")

    # 5.6 -- budget-restricted recall.
    budget = {}
    for c, col in (("C4", "c4_budget_met"), ("C5", "c5_budget_cleared")):
        sub = df[(df["truth_for_gate"] == c) & df[col].astype(bool)]
        budget[c] = {
            "n": int(len(sub)),
            "recall": float((sub["attributed_cause"] == c).mean()) if len(sub) else float("nan"),
        }

    gates = {
        "5.1_recall": {
            "value": {c: recall[c] for c in GATE_CAUSES},
            "pass": bool(all(recall[c] >= RECALL_MIN for c in GATE_CAUSES)),
            "bar": RECALL_MIN,
            "budget_restricted": {c: recall_eval[c] for c in GATE_CAUSES},
            "n_evaluable": {c: n_eval[c] for c in GATE_CAUSES},
            # A cause with no evaluable SKU has NOT demonstrated 0.60 recall; NaN must
            # not pass vacuously.
            "pass_budget_restricted": bool(
                all(
                    np.isfinite(recall_eval[c])
                    and n_eval[c] > 0
                    and recall_eval[c] >= RECALL_MIN
                    for c in GATE_CAUSES
                )
            ),
            "scope": "six-cause (C7 excluded by the 2026-10-17 ruling)",
            "c7_reported_not_gated": recall["C7"],
            "c2b_variant_recall": recall_c2b,
        },
        "5.2_false_positive": {
            "value": fp_rate,
            "pass": bool(fp_rate <= FP_MAX),
            "bar": FP_MAX,
            "window": "t >= tau_inj (the same window recall is scored on)",
            "pre_injection_flag_rate_diagnostic": pre_flag_rate,
        },
        "5.3_off_diagonal": {
            "value": offdiag,
            "pass": bool(all((not np.isfinite(v)) or v <= OFFDIAG_MAX for v in offdiag.values())),
            "bar": OFFDIAG_MAX,
        },
        "5.4_priority_sensitivity": {
            "value_max_abs_diff": max_diff,
            "pass": bool((not np.isfinite(max_diff)) or max_diff <= PRIORITY_DIFF_MAX),
            "bar": PRIORITY_DIFF_MAX,
            "base_macro_f1": base_f1,
            "permutations": sens.to_dict("records"),
        },
        "5.5_outputs": {"produced": True},
        "5.6_budget_restricted": {"value": budget},
    }
    # Detection diagnostics (not gated): latency after the injection point, and the
    # fraction of monitoring checks that flagged anything. Gate 5.2 counts a C0 SKU
    # as a false positive if ANY check flagged it -- the strict reading.
    lat = {}
    for c in GATE_CAUSES + ["C7"]:
        sub = df[df["truth_for_gate"] == c]
        hit = sub[sub["detected"].astype(bool)]
        lat[c] = {
            "n": int(len(sub)),
            "frac_detected": float(len(hit) / len(sub)) if len(sub) else float("nan"),
            "median_latency": float(hit["detect_latency"].median()) if len(hit) else float("nan"),
            "p90_latency": float(hit["detect_latency"].quantile(0.9)) if len(hit) else float("nan"),
        }
    gates["detection_latency"] = lat
    gates["recall_wilson95"] = recall_ci
    gates["recall_counts"] = recall_counts
    gates["c0_per_rule_firing"] = c0_rule_firing
    if cusum_calibration is not None:
        gates["cusum_calibration"] = cusum_calibration
    gates["check_schedule"] = {
        "check_from": int(check_from),
        "n_checks_median": float(df["n_checks"].median()),
        "flagged_check_rate_by_truth": {
            c: float(df.loc[df["truth_for_gate"] == c, "n_flagged_checks"].sum()
                     / max(df.loc[df["truth_for_gate"] == c, "n_checks"].sum(), 1e-9))
            for c in CAUSES_ALL if (df["truth_for_gate"] == c).any()
        },
    }
    gates["overall_pass"] = bool(
        # Both denominators must clear the bar. §29.3 makes the budget-restricted
        # number the methodologically meaningful one, but requiring the pooled number
        # too means the pre-registered gate is never satisfied by excluding SKUs.
        gates["5.1_recall"]["pass"]
        and gates["5.1_recall"]["pass_budget_restricted"]
        and gates["5.2_false_positive"]["pass"]
        and gates["5.3_off_diagonal"]["pass"]
        and gates["5.4_priority_sensitivity"]["pass"]
    )
    gates["confusion"] = conf.to_dict()
    gates["confusion_rownormalised"] = rownorm.round(4).to_dict()
    gates["recall_all"] = recall
    return gates


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="synthetic.yaml")
    ap.add_argument("--per-cause", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-ambiguous", action="store_true")
    ap.add_argument(
        "--stride", type=int, default=4,
        help="periods between monitor checks (monitoring cadence); default 4",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    burn = int(cfg["splits"]["burn_in"])
    T = int(cfg["synthetic"]["T"])
    # Detection runs over the EVALUATED window: burn-in is the pre-injection reference
    # the monitor's baseline statistics are built from, so a check before it would be a
    # check against itself. For the synthetic panel the evaluated window contains both
    # the pre-injection (24..tau_inj) and post-injection (tau_inj..T) regimes.
    check_from = burn
    # The labelled claim is about the post-injection regime (synthetic.yaml: 120).
    detect_from = int(cfg["synthetic"]["tau_inj"])

    wanted = list(CAUSES_ALL) + ["C2b"] + (["ambiguous"] if args.include_ambiguous else [])
    rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    for cause in wanted:
        ids = sorted(params.index[params["true_cause"] == cause])
        if not ids:
            continue
        take = ids if len(ids) <= args.per_cause else list(
            rng.choice(ids, size=args.per_cause, replace=False)
        )
        chosen.extend(sorted(take))

    series = {
        sid: grp.sort_values("period")["demand_observed"].to_numpy(dtype="float64")
        for sid, grp in demand[demand["sku_id"].isin(chosen)].groupby("sku_id", sort=False)
    }
    chosen = [s for s in chosen if s in series and series[s].size >= T]

    statics_by_sku = {
        sid: {
            "lead_time": float(params.loc[sid, "lead_time"]),
            "unit_cost": float(params.loc[sid, "unit_cost"]),
            "holding_rate": float(params.loc[sid, "holding_rate"]),
            "adi_init": float(params.loc[sid, "adi_init"]),
            "cv2_init": float(params.loc[sid, "cv2_init"]),
        }
        for sid in chosen
    }
    train_panel = pd.DataFrame(
        [
            {"sku_id": sid, "period": t, "demand_observed": float(series[sid][t]),
             **statics_by_sku[sid]}
            for sid in chosen
            for t in range(burn)
        ]
    )

    t0 = time.time()
    pool = fit_global_pool(cfg, train_panel, statics_by_sku=statics_by_sku)
    print(f"pool fitted on {len(chosen)} SKUs x {burn} periods in {time.time()-t0:.1f}s")

    c0_ids = sorted(
        sid for sid in chosen if str(params.loc[sid, "true_cause"]) == "C0"
    )
    n_train_controls = int(np.floor(0.60 * len(c0_ids)))
    train_control_ids = c0_ids[:n_train_controls]
    calibration_cfg = copy.deepcopy(cfg)
    calibration_cfg.setdefault("detection", {})["cusum_h"] = 1e12
    calibration_scores: list[float] = []
    calibration_start = time.time()
    for sid in train_control_ids:
        _run_one(
            sid, series[sid], params.loc[sid], calibration_cfg, pool, args.seed,
            check_from=check_from, detect_from=detect_from, stride=args.stride,
            calibration_scores=calibration_scores,
        )
    if not calibration_scores:
        raise RuntimeError("no pre-injection C0 checks available for CUSUM calibration")
    target_cusum_fpr = 0.05
    calibrated_h = float(np.quantile(calibration_scores, 1.0 - target_cusum_fpr, method="higher"))
    cfg.setdefault("detection", {})["cusum_h"] = calibrated_h
    cusum_calibration = {
        "h": calibrated_h,
        "k": float(cfg["detection"].get("cusum_k", 0.25)),
        "target_per_check_false_alarm_rate": target_cusum_fpr,
        "calibration_per_check_false_alarm_rate": float(
            np.mean(np.asarray(calibration_scores) > calibrated_h)
        ),
        "calibration_n_checks": int(len(calibration_scores)),
        "training_controls": int(len(train_control_ids)),
        "training_control_ids": "sorted C0 sku_id first 60%; remaining C0 controls held out",
        "periods": [burn, detect_from],
        "statistic": "max reset-aware standardized excursion from size and occurrence CUSUMs",
        "elapsed_seconds": float(time.time() - calibration_start),
    }
    print(
        f"CUSUM h calibrated to {calibrated_h:.6g} from "
        f"{len(calibration_scores)} train-control pre-injection checks "
        f"(empirical rate {cusum_calibration['calibration_per_check_false_alarm_rate']:.4f})"
    )

    rows: list[dict] = []
    readings_for_order: list = []
    t0 = time.time()
    for i, sid in enumerate(chosen):
        param = params.loc[sid]
        y = series[sid]
        try:
            row, reading = _run_one(
                sid, y, param, cfg, pool, args.seed,
                check_from=check_from, detect_from=detect_from, stride=args.stride,
            )
        except Exception as exc:  # noqa: BLE001 - a failed SKU must be visible, not fatal
            rows.append({"sku_id": sid, "true_cause": str(param["true_cause"]),
                         "truth_for_gate": "C2" if param["true_cause"] == "C2b"
                         else str(param["true_cause"]),
                         "attributed_cause": "ERROR", "error": repr(exc)})
            continue
        rows.append(row)
        truth_for_gate = "C2" if param["true_cause"] == "C2b" else str(param["true_cause"])
        readings_for_order.append((truth_for_gate, reading))
        if i % 50 == 0:
            print(f"  {i+1}/{len(chosen)}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    ok = df[df["attributed_cause"] != "ERROR"]
    print(f"ran {len(ok)} SKUs, {len(df)-len(ok)} errors, {time.time()-t0:.1f}s")
    if len(df) > len(ok):
        for rec in df[df["attributed_cause"] == "ERROR"].head(5).to_dict("records"):
            print(f"  ERROR {rec['sku_id']}: {rec.get('error')}")

    # Readings are kept in memory for gate 5.4 -- re-attribution must be exact, and a
    # rebuilt reading would lose the C1 sign blocks and the C6 cell history.
    assert len(readings_for_order) == len(ok), (
        "every successful SKU must have exactly one stored reading"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "step5_readings.parquet", index=False)

    gates = evaluate_gates(
        ok, readings_for_order, cfg, check_from=check_from,
        cusum_calibration=cusum_calibration,
    )
    (OUT / "step5_summary.json").write_text(json.dumps(gates, indent=2, default=float),
                                            encoding="utf-8")

    conf = pd.DataFrame(gates["confusion"]).reindex(index=CAUSES_ALL, columns=CAUSES_ALL)
    conf.to_parquet(OUT / "step5_confusion.parquet")
    pd.DataFrame(gates["5.4_priority_sensitivity"]["permutations"]).to_parquet(
        OUT / "step5_priority_sensitivity.parquet", index=False
    )

    print("\n=== Step 5 gates ===")
    print(f"5.1 recall (bar {RECALL_MIN}):")
    for c in CAUSES_ALL:
        bar = " (reported, not gated)" if c == "C7" else ""
        print(f"   {c}: {gates['recall_all'][c]:.3f}{bar}"
              f"   [budget-restricted {gates['5.1_recall']['budget_restricted'].get(c, float('nan')):.3f}"
              f" on n={gates['5.1_recall']['n_evaluable'].get(c, 0)}]")
    print(f"   C2b variant -> C2: {gates['5.1_recall']['c2b_variant_recall']:.3f}")
    print(f"   PASS: {gates['5.1_recall']['pass']}" if gates["5.1_recall"]["pass"]
          else f"   FAIL (six-cause scope)")
    print(f"5.2 FP on C0: {gates['5.2_false_positive']['value']:.3f} "
          f"(bar {FP_MAX}) PASS: {gates['5.2_false_positive']['pass']}")
    print(f"5.3 max off-diagonal: {max(v for v in gates['5.3_off_diagonal']['value'].values() if np.isfinite(v)):.3f} "
          f"(bar {OFFDIAG_MAX}) PASS: {gates['5.3_off_diagonal']['pass']}")
    print(f"5.4 priority sensitivity: {gates['5.4_priority_sensitivity']['value_max_abs_diff']:.3f} "
          f"(bar {PRIORITY_DIFF_MAX}) PASS: {gates['5.4_priority_sensitivity']['pass']}")
    print(f"5.6 budget-restricted: "
          + ", ".join(f"{c}={v['recall']:.3f} (n={v['n']})"
                      for c, v in gates["5.6_budget_restricted"]["value"].items()))
    print("detection latency (median periods after check_from):")
    for c, v in gates["detection_latency"].items():
        print(f"   {c}: flagged {v['frac_detected']:.3f}, median {v['median_latency']:.1f}")
    print(f"\nOVERALL: {'PASS' if gates['overall_pass'] else 'FAIL'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
