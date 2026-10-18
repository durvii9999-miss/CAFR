"""C2-only reliability-gated CAFR -- the Step 6 screening experiment.

Protocol (fixed before the comparison run)
------------------------------------------
* Panel: the committed seed-42 synthetic panel, T = 200, ``tau_inj`` = 120.
* Split: the deterministic per-cause 60/20/20 split used by the reliability
  screening (sorted ``sku_id``, cycle ``train, train, train, val, test``).
* Gate training rows: ``detected == True`` and ``attributed_cause == "C2"``.
  These are the first post-tau checkpoint readings, i.e. the exact event at which
  the policy makes its one decision. Undetected final-reading rows are excluded.
  C2b is excluded everywhere.
* Label: ``true_cause == "C2"``.
* Threshold: the lowest validation threshold with precision >= 0.90 and at least
  20 admitted true-C2 validation SKUs. If none exists, the rule-only variant is
  used and reported as such. The test split is never used for selection.
* Policy: at the first non-C0 attribution at ``t >= tau_inj`` (checked every
  ``stride`` periods), decide once. Act only if the attribution is C2, the C2
  evidence floor is met, and the calibrated score clears the threshold. The
  remedy resets C2's safety-stock scale to 1.0 (nominal conformal interval);
  the point forecast is not touched. Otherwise R0 and no re-query.
* Outcome window: ``[tau_inj + 12, T)`` = [132, 200), identical for both arms.
  Chosen from the training split only: 95.8 % of train C2 SKUs have
  ``detect_t + 1 + lead_time <= 132``; the late detectors are reported.

This file does not modify ``results/step5`` or any frozen configuration.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafr.arms.arm_c import arm_c_factory
from cafr.attribution import attribute
from cafr.forecasters.pooled import fit_global_pool
from cafr.monitor import Monitor
from cafr.monitor.features import conformal_interval
from cafr.sim.inventory import InventorySimulator, SimConfig, aggregate_order_up_to
from cafr.utils.config import REPO_ROOT, load_config

DATA = REPO_ROOT / "data" / "synth"
READINGS = REPO_ROOT / "results" / "step5" / "step5_readings.parquet"
OUT = REPO_ROOT / "results" / "step6"

SEED = 42
STRIDE = 4
ALPHA = 0.95
TAU_INJ = 120
EVAL_START = TAU_INJ + 12
SPLIT_CYCLE = np.array(["train", "train", "train", "val", "test"])

GATE_FEATURES = [
    "margin_C2",
    "n_fired",
    "ev_c2",
    "window_full",
    "abs_gap",
    "neglog10_p",
    "abs_z_b",
    "r_strength",
    "confidence",
]


# --------------------------------------------------------------------------- #
# panel, split, gate features                                                  #
# --------------------------------------------------------------------------- #


def load_context():
    cfg = load_config("synthetic.yaml")
    demand = pd.read_parquet(DATA / "demand_panel.parquet")
    params = pd.read_parquet(DATA / "generator_params.parquet").set_index("sku_id")
    readings = pd.read_parquet(READINGS)
    assert readings["sku_id"].is_unique, "step5_readings.parquet must be one row per SKU"
    readings["n_fired"] = (
        readings["fired"].fillna("").astype(str)
        .apply(lambda s: 0 if not s.strip() else len(s.split(",")))
    )
    readings["split"] = assign_split(readings)
    # Keep ``sku_id`` as a column as well: the split and pool code read it, while
    # the per-SKU lookups use the index.
    readings = readings.set_index("sku_id", drop=False)
    series = {
        sid: g.sort_values("period")["demand_observed"].to_numpy(dtype="float64")
        for sid, g in demand.groupby("sku_id", sort=False)
    }
    return cfg, readings, params, series


def assign_split(readings: pd.DataFrame) -> pd.Series:
    """The exact split recovered from the seed-42 screening run.

    Per true cause, sorted ``sku_id`` -> cycle (train, train, train, val, test).
    """
    out = pd.Series("", index=readings.index, dtype=object)
    for _, g in readings.groupby("true_cause"):
        g = g.sort_values("sku_id")
        out.loc[g.index] = SPLIT_CYCLE[np.arange(len(g)) % len(SPLIT_CYCLE)]
    return out


def _r_strength(value) -> float:
    try:
        r = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(r) or r <= 0.0:
        return float("nan")
    return float(max(r, 1.0 / r))


def _neglog10_p(value) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float(-np.log10(np.clip(p, 1e-12, 1.0)))


def features_from_row(row) -> dict:
    return {
        "margin_C2": float(row.get("margin_C2", np.nan)),
        "n_fired": float(row.get("n_fired", 0.0)),
        "ev_c2": float(bool(row.get("ev_c2", False))),
        "window_full": float(bool(row.get("window_full", False))),
        "abs_gap": abs(float(row.get("c2_c_gap", np.nan))),
        "neglog10_p": _neglog10_p(row.get("c2_p_val", np.nan)),
        "abs_z_b": abs(float(row.get("c1_z_b", np.nan))),
        "r_strength": _r_strength(row.get("c4_r", np.nan)),
        "confidence": float(row.get("confidence", np.nan)),
    }


def features_from_reading(reading, att) -> dict:
    s = reading.statistics
    return {
        "margin_C2": float(att.margins.get("C2", np.nan)),
        "n_fired": float(len(att.fired)),
        "ev_c2": float(bool(reading.evidence.get("c2", False))),
        "window_full": float(bool(reading.window_full)),
        "abs_gap": abs(float(s.get("c2_c_gap", np.nan))),
        "neglog10_p": _neglog10_p(s.get("c2_p_val", np.nan)),
        "abs_z_b": abs(float(s.get("c1_z_b", np.nan))),
        "r_strength": _r_strength(s.get("c4_r", np.nan)),
        "confidence": float(att.confidence),
    }


def _frame(rows_feats: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows_feats, columns=GATE_FEATURES)


def fit_gate(train: pd.DataFrame):
    X = _frame([features_from_row(r) for _, r in train.iterrows()])
    y = (train["true_cause"] == "C2").astype(int).to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0),
    )
    model.fit(X, y)
    return model


def score_rows(model, rows: pd.DataFrame) -> np.ndarray:
    if len(rows) == 0:
        return np.asarray([], dtype=float)
    return model.predict_proba(
        _frame([features_from_row(r) for _, r in rows.iterrows()])
    )[:, 1]


def choose_tau(model, val: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    """Lowest validation threshold with precision >= 0.90 and >= 20 true-C2."""
    p = score_rows(model, val)
    y = (val["true_cause"] == "C2").astype(int).to_numpy()
    table = []
    for t in np.unique(p):
        admitted = p >= t
        n = int(admitted.sum())
        k = int(y[admitted].sum())
        table.append(
            {
                "tau": float(t),
                "admitted": n,
                "true_c2": k,
                "precision": float(k / n) if n else float("nan"),
            }
        )
    table = pd.DataFrame(table).sort_values("tau").reset_index(drop=True)
    ok = table[(table["precision"] >= 0.90) & (table["true_c2"] >= 20)]
    tau = float(ok["tau"].iloc[0]) if len(ok) else None
    return tau, table


def decide(att, reading, tau: float | None, model) -> dict:
    """The one-shot gate decision. Pure; no state, no look-ahead."""
    if att.cause != "C2":
        return {"action": "R0", "reason": "not_C2", "score": float("nan")}
    if not bool(reading.evidence.get("c2", False)):
        return {"action": "R0", "reason": "insufficient_evidence", "score": float("nan")}
    if tau is None:
        return {"action": "R2", "reason": "rule_only", "score": float("nan")}
    score = float(
        model.predict_proba(_frame([features_from_reading(reading, att)]))[0, 1]
    )
    if score >= tau:
        return {"action": "R2", "reason": "passed", "score": score}
    return {"action": "R0", "reason": "low_reliability", "score": score}


def fit_and_select(readings: pd.DataFrame):
    prim = readings[(readings["true_cause"] != "C2b") & readings["detected"]]
    prim = prim[prim["attributed_cause"] == "C2"]
    train = prim[prim["split"] == "train"]
    val = prim[prim["split"] == "val"]
    model = fit_gate(train)
    tau, table = choose_tau(model, val)
    return model, tau, table, train, val


# --------------------------------------------------------------------------- #
# the gated policy                                                             #
# --------------------------------------------------------------------------- #


class GatedC2Arm:
    """One decision per SKU, at the first post-tau non-C0 attribution.

    The base arm is ``arm_c``. Its C2 scale is driven by ``provider``; the wrapper
    changes that scale only when the single decision is R2. The monitor sees the
    same scale the policy uses, so after the remedy its coverage returns to
    nominal.
    """

    def __init__(self, pool, cfg, sku_id, *, gate_model, tau,
                 detect_from=TAU_INJ, check_from=24, stride=STRIDE,
                 alpha=ALPHA, seed=SEED):
        self.cfg = cfg
        self.tau = tau
        self.gate_model = gate_model
        self.detect_from = int(detect_from)
        self.check_from = int(check_from)
        self.stride = int(stride)
        self.state = {
            "scale": float(cfg["sim"].get("c2_quantile_scale", 1.0)),
            "decided": False,
            "decision": None,
            "n_decisions": 0,
            "n_checks": 0,
            "action_t": None,
        }
        # Per-period diagnostics: point forecast, S, and the scale the policy used.
        self.trace: list[dict] = []
        self.base = arm_c_factory(
            pool, cfg, sku_id, alpha, "observed",
            c2_scale_provider=self.provider,
        )
        self.monitor = Monitor(cfg, alpha_target=alpha, seed=seed)
        self.q_alpha = float(cfg.get("detection", {}).get("c2_interval_alpha", 0.90))
        self.q_win = int(cfg["sim"].get("quantile_window", 24))

    def provider(self, t: int) -> float:
        return float(self.state["scale"])

    def __call__(self, t, y_observed, censored, inventory):
        scale_used = float(self.state["scale"])
        S, a = self.base(t, y_observed, censored, inventory)
        st = self.base.last_state
        self.trace.append(
            {"t": int(t), "mu": float(st.mu), "S": float(S), "scale": scale_used}
        )
        res = self.base._residuals.to_array()
        rr = res[-self.q_win:] if res.size else res
        lo, hi = conformal_interval(
            float(st.mu), rr, self.q_alpha, scale=scale_used, min_evidence=8
        )
        stockout = bool(
            inventory.get("on_hand", 1.0) <= 1e-9
            and float(y_observed[-1]) > 0.0
        )
        self.monitor.update(
            t,
            float(y_observed[-1]),
            bool(censored[-1]),
            {
                "mu": float(st.mu),
                "mu_nonzero": float(st.mu_nonzero),
                "sigma": float(st.sigma),
                "z": float(st.z),
                "lo": lo,
                "hi": hi,
                "alpha": self.q_alpha,
            },
            stockout_t=stockout,
        )

        checkpoint = (
            (not self.state["decided"])
            and t >= self.check_from
            and t >= self.detect_from
            and (t - self.check_from) % self.stride == 0
        )
        if checkpoint:
            self.state["n_checks"] += 1
            rd = self.monitor.read()
            att = attribute(rd, self.cfg)
            if att.cause != "C0":
                decision = decide(att, rd, self.tau, self.gate_model)
                decision.update(
                    {
                        "t": int(t),
                        "attributed": att.cause,
                        "confidence": float(att.confidence),
                        "c2_c_gap": float(rd.statistics.get("c2_c_gap", np.nan)),
                        "c2_p_val": float(rd.statistics.get("c2_p_val", np.nan)),
                    }
                )
                self.state["decided"] = True
                self.state["n_decisions"] += 1
                self.state["decision"] = decision
                if decision["action"] == "R2":
                    self.state["scale"] = 1.0
                    self.state["action_t"] = int(t) + 1
        return S, a


def run_sku(sku_id, y, param, cfg, pool, *, mode, gate_model=None, tau=None,
            trace=None):
    """Run one SKU under ``baseline`` or ``gated``. Paired by construction."""
    cfg_s = copy.deepcopy(cfg)
    cause = str(param["true_cause"])
    gamma = 1.0
    if cause == "C2":
        if pd.notna(param.get("gamma")):
            gamma = float(param["gamma"])
        cfg_s["sim"]["c2_quantile_scale"] = gamma
    elif cause == "C2b":
        raise ValueError("C2b is excluded from the C2-only experiment")

    sim_base = SimConfig.from_config(cfg_s)
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
        c2_quantile_scale=gamma,
        protection_convention=sim_base.protection_convention,
        observation_model=sim_base.observation_model,
    )
    sim = InventorySimulator(sim_cfg)

    if mode == "baseline":
        level_fn = arm_c_factory(
            pool, cfg_s, sku_id, ALPHA, "observed",
            c2_scale_provider=lambda t: gamma,
        )
        state = {"decision": None, "n_decisions": 0, "action_t": None,
                 "n_checks": 0, "scale": gamma}
    elif mode == "gated":
        arm = GatedC2Arm(
            pool, cfg_s, sku_id, gate_model=gate_model, tau=tau,
            detect_from=int(cfg_s["synthetic"]["tau_inj"]),
            check_from=int(cfg_s["splits"]["burn_in"]),
            stride=STRIDE,
        )
        level_fn = arm
        state = arm.state
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if trace is not None:
        _inner = level_fn

        def level_fn(t, y_obs, cens, inv):
            S, a = _inner(t, y_obs, cens, inv)
            obj = getattr(_inner, "base", _inner)
            st = obj.last_state
            trace.append(
                {
                    "t": int(t),
                    "mu": float(st.mu) if st is not None else float("nan"),
                    "S": float(S),
                }
            )
            return S, a

    res = sim.run(sku_id, y, level_fn, arm=mode)
    return res, state


# --------------------------------------------------------------------------- #
# outcomes                                                                     #
# --------------------------------------------------------------------------- #


def evaluate(res, start: int, R: int) -> dict:
    log = res.log
    cost = float(log["cost_period"].to_numpy()[start:].sum())
    lost = log["demand_lost"].to_numpy()[start:]
    n_cycles = int(lost.size // R)
    csl = (
        float((lost[: n_cycles * R].reshape(n_cycles, R).sum(axis=1) <= 0).mean())
        if n_cycles else float("nan")
    )
    period_sl = float((lost <= 0).mean()) if lost.size else float("nan")
    n_periods = int(lost.size)
    return {
        "cost": cost,
        "mean_cost_per_period": cost / n_periods if n_periods else float("nan"),
        "csl": csl,
        "period_sl": period_sl,
        "n_periods": n_periods,
        "n_cycles": n_cycles,
    }


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = SEED):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def build_pool(cfg, series, params, ids=None):
    """Fit the pooled forecaster on burn-in windows only.

    Uses the same SKU order and static columns as the Step 5 run so the live
    monitor readings reproduce the stored training readings. The global model
    sees only the first ``burn_in`` periods; the gate model itself is trained on
    train-split rows only.
    """
    burn = int(cfg["splits"]["burn_in"])
    if ids is None:
        ids = []
        for c in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C2b"]:
            ids.extend(sorted(params.index[params["true_cause"] == c]))
    rows = []
    statics = {}
    for sid in ids:
        p = params.loc[sid]
        statics[sid] = {
            "lead_time": float(p["lead_time"]),
            "unit_cost": float(p["unit_cost"]),
            "holding_rate": float(p["holding_rate"]),
            "adi_init": float(p["adi_init"]),
            "cv2_init": float(p["cv2_init"]),
        }
        y = series[sid]
        for t in range(min(burn, y.size)):
            rows.append(
                {
                    "sku_id": sid,
                    "period": t,
                    "demand_observed": float(y[t]),
                    **statics[sid],
                }
            )
    panel = pd.DataFrame(rows)
    return fit_global_pool(cfg, panel, statics_by_sku=statics)


# --------------------------------------------------------------------------- #
# gate diagnostics                                                             #
# --------------------------------------------------------------------------- #


def gate_table(model, tau: float | None, readings: pd.DataFrame) -> dict:
    prim = readings[readings["true_cause"] != "C2b"].copy()
    cand = prim[prim["detected"] & (prim["attributed_cause"] == "C2")].copy()
    scores = score_rows(model, cand) if len(cand) else np.asarray([])
    acted = scores >= tau if tau is not None else np.ones(len(cand), dtype=bool)

    out = {
        "tau": tau,
        "mode": "calibrated" if tau is not None else "rule_only",
        "n_test_rows": int(len(prim[prim["split"] == "test"])),
        "n_candidates": int((cand["split"] == "test").sum()),
        "by_split": {},
    }
    for split in ("train", "val", "test"):
        q = cand[cand["split"] == split]
        if len(q) == 0:
            continue
        a = acted[(cand["split"] == split).to_numpy()]
        y = (q["true_cause"] == "C2").to_numpy()
        all_rows = prim[prim["split"] == split]
        out["by_split"][split] = {
            "candidates": int(len(q)),
            "acted": int(a.sum()),
            "true_c2_acted": int((y & a).sum()),
            "precision": float(y[a].mean()) if a.sum() else float("nan"),
            "recall_all_c2_split": float(
                (y & a).sum()
                / max(int((all_rows["true_cause"] == "C2").sum()), 1)
            ),
            "recall_among_c2_candidates": float(
                (y & a).sum() / max(int(y.sum()), 1)
            ),
            "coverage_all_rows": float(a.sum() / max(len(all_rows), 1)),
            "coverage_candidates": float(a.sum() / max(len(q), 1)),
            "true_c0_action_rate": float(
                (
                    (q["true_cause"] == "C0").to_numpy() & a
                ).sum()
                / max(int((all_rows["true_cause"] == "C0").sum()), 1)
            ),
            "auroc": float(
                roc_auc_score(y, scores[(cand["split"] == split).to_numpy()])
            ) if 0 < int(y.sum()) < len(y) else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
# stages                                                                       #
# --------------------------------------------------------------------------- #


def _prepare(quiet: bool = False):
    cfg, readings, params, series = load_context()
    model, tau, table, train, val = fit_and_select(readings)
    if not quiet:
        print(f"gate train rows={len(train)} val rows={len(val)}")
        print(f"validation-selected tau={tau}")
        print(table.tail(12).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return cfg, readings, params, series, model, tau, table


def _eval_window_check(readings, params):
    tr = readings[
        (readings["split"] == "train")
        & (readings["true_cause"] == "C2")
        & readings["detected"]
    ]
    lead = params.loc[tr["sku_id"], "lead_time"].astype(float).to_numpy()
    first_affect = tr["detect_t"].astype(float).to_numpy() + 1.0 + lead
    frac = float((first_affect <= EVAL_START).mean()) if len(tr) else float("nan")
    return {
        "eval_start": EVAL_START,
        "eval_periods": int(200 - EVAL_START),
        "train_c2_detected": int(len(tr)),
        "frac_affected_by_eval_start": frac,
        "late_detectors": int((first_affect > EVAL_START).sum()),
        "median_detect_t": float(tr["detect_t"].median()) if len(tr) else float("nan"),
        "p95_detect_t": float(tr["detect_t"].quantile(0.95)) if len(tr) else float("nan"),
    }


def stage_gate():
    cfg, readings, params, series, model, tau, table = _prepare()
    diag = gate_table(model, tau, readings)
    window = _eval_window_check(readings, params)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c2_gate_summary.json").write_text(
        json.dumps({"gate": diag, "window": window}, indent=2, default=str)
    )
    print("\nGate diagnostics")
    print(json.dumps(diag, indent=2, default=str))
    print("\nOutcome window")
    print(json.dumps(window, indent=2, default=str))


def _compare(ids, readings, params, series, cfg, pool, model, tau, label):
    rows = []
    paired = []
    t0 = time.time()
    for i, sid in enumerate(ids):
        p = params.loc[sid]
        y = series[sid]
        res_b, st_b = run_sku(sid, y, p, cfg, pool, mode="baseline")
        res_g, st_g = run_sku(sid, y, p, cfg, pool, mode="gated",
                              gate_model=model, tau=tau)
        R = int(cfg["sim"]["R"])
        mb = evaluate(res_b, EVAL_START, R)
        mg = evaluate(res_g, EVAL_START, R)
        dec = st_g.get("decision") or {}
        stored = readings.loc[sid]
        stored_cand = bool(stored["detected"]) and stored["attributed_cause"] == "C2"
        stored_score = (
            float(score_rows(model, readings.loc[[sid]])[0]) if stored_cand else np.nan
        )
        stored_act = bool(
            stored_cand and (tau is None or stored_score >= tau)
        )
        live_act = dec.get("action") == "R2"
        base = {
            "sku_id": sid,
            "true_cause": str(p["true_cause"]),
            "gamma": float(p["gamma"]) if pd.notna(p.get("gamma")) else 1.0,
        }
        rows.append({**base, "arm": "baseline", **mb})
        rows.append(
            {
                **base,
                "arm": "gated",
                **mg,
                "decision_t": dec.get("t"),
                "attributed": dec.get("attributed"),
                "confidence": dec.get("confidence"),
                "gate_score": dec.get("score"),
                "action": dec.get("action"),
                "reason": dec.get("reason"),
                "action_t": st_g.get("action_t"),
                "n_decisions": st_g.get("n_decisions"),
                "n_checks": st_g.get("n_checks"),
                "stored_detect_t": stored["detect_t"],
                "stored_attributed": stored["attributed_cause"],
                "stored_score": stored_score,
                "stored_act": stored_act,
                "live_act": live_act,
                "decision_match": bool(stored_act == live_act),
            }
        )
        paired.append(
            {
                "sku_id": sid,
                "true_cause": str(p["true_cause"]),
                "gamma": base["gamma"],
                "d_cost": mg["cost"] - mb["cost"],
                "d_csl": mg["csl"] - mb["csl"],
                "d_period_sl": mg["period_sl"] - mb["period_sl"],
                "action": dec.get("action"),
                "reason": dec.get("reason"),
                "decision_t": dec.get("t"),
                "action_t": st_g.get("action_t"),
                "decision_match": bool(stored_act == live_act),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(ids)} SKUs, {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    pdf = pd.DataFrame(paired)
    summary = {"label": label, "n_skus": len(ids), "elapsed_s": time.time() - t0}
    for arm in ("baseline", "gated"):
        q = df[df["arm"] == arm]
        summary[arm] = {
            "mean_cost": float(q["cost"].mean()),
            "mean_cost_per_period": float(q["mean_cost_per_period"].mean()),
            "mean_csl": float(q["csl"].mean()),
            "mean_period_sl": float(q["period_sl"].mean()),
        }
    for name, mask in (
        ("all", pdf["true_cause"].isin(["C2", "C0"])),
        ("C2", pdf["true_cause"] == "C2"),
        ("C0", pdf["true_cause"] == "C0"),
    ):
        q = pdf[mask]
        mc, ci_c = paired_bootstrap(q["d_cost"].to_numpy())
        ms, ci_s = paired_bootstrap(q["d_csl"].to_numpy())
        summary[f"paired_{name}"] = {
            "n": int(len(q)),
            "d_cost_mean": mc,
            "d_cost_ci95": list(ci_c),
            "d_csl_mean": ms,
            "d_csl_ci95": list(ci_s),
        }
    summary["acted"] = int((df[df["arm"] == "gated"]["action"] == "R2").sum())
    summary["decision_match_rate"] = float(pdf["decision_match"].mean())
    summary["live_actions_by_cause"] = (
        df[df["arm"] == "gated"]["attributed"].value_counts().to_dict()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"c2_{label}_arms.csv", index=False)
    pdf.to_csv(OUT / f"c2_{label}_paired.csv", index=False)
    (OUT / f"c2_{label}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary, df, pdf


def _select_test_ids(readings, params, n_c2, n_c0):
    test = readings[readings["split"] == "test"]
    c2 = sorted(test.loc[test["true_cause"] == "C2", "sku_id"])[:n_c2]
    c0 = sorted(test.loc[test["true_cause"] == "C0", "sku_id"])[:n_c0]
    return c2 + c0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "smoke", "timing", "compare"],
                    default="gate")
    args = ap.parse_args()

    if args.stage == "gate":
        stage_gate()
        return

    cfg, readings, params, series, model, tau, table = _prepare()
    window = _eval_window_check(readings, params)
    print("\nOutcome window:", json.dumps(window, default=str))
    if not np.isfinite(window["frac_affected_by_eval_start"]) or \
            window["frac_affected_by_eval_start"] < 0.90:
        raise SystemExit("STOP: the evaluated window does not let the remedy act")

    t0 = time.time()
    pool = build_pool(cfg, series, params)
    print(f"pool fitted in {time.time() - t0:.1f}s")

    n_c2 = {"smoke": 2, "timing": 10, "compare": 40}[args.stage]
    n_c0 = {"smoke": 2, "timing": 10, "compare": 40}[args.stage]
    ids = _select_test_ids(readings, params, n_c2, n_c0)
    print(f"{args.stage}: {len(ids)} SKUs ({n_c2} C2, {n_c0} C0)")
    _compare(ids, readings, params, series, cfg, pool, model, tau, args.stage)


if __name__ == "__main__":
    main()
