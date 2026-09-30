"""Gates 4.1 - 4.12: does the injected panel actually carry the signatures the monitor
is supposed to read? (Rev 2 §15.2, §30.)

This is the project's own STOP gate. If it fails, the injection table is wrong and
Step 5 must not start -- and the fix is to revise §15.2's injection design, **never** to
tune a detector until it passes.

What changed at Step 4, and why
-------------------------------
The panel's C4 statistic used to be a ratio of ABSOLUTE size dispersions. That cannot
work: for a Gamma size distribution the sd is proportional to the mean, so C4's own
statistic was moved by C1's mean ramp (AUC 0.615) and, worse, by C5's size step
(AUC 0.805) -- a confound that would have shown up in the confusion matrix as C4 being
"attributed" to every level change. C4's statistic is now the ratio of LOG-scale
dispersions, ``r = sd(log z)_now / sd(log z)_base``, which is exactly invariant to the
mean because ``log z = log theta + log Gamma(k, 1)``.

The test also stops inventing a residual: it reads the OBSERVED nonzero demand, which
is what the monitor actually has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cafr.data.synth.budget import check_c4_budget
from cafr.data.synth.generator import CELLS, get_cell_params
from cafr.data.synth.injections import (
    INJECTORS,
    inject_C0,
    inject_C1,
    inject_C3,
    inject_C4,
    inject_C5,
    inject_C6,
)
from cafr.monitor.features import (
    compute_bias_stats,
    compute_cusum_stats,
    compute_dispersion_stats,
    compute_intermittency_stats,
    robust_sd,
)

T = 200
TAU = 120
N_SERIES = 200

#: Straight from ``monitor.windows`` / ``monitor.min_evidence`` in cafr/configs/base.yaml.
W_SHORT = 24
W_SZ = 24
W_CLASS = 30
C4_MIN_EVIDENCE = 8

#: The pre-injection baseline window for every "now vs base" statistic.
BASE = slice(0, 60)

#: AUC bars (§15.2 constraints, gate 4.3 / 4.4).
AUC_MIN = 0.80
CONFOUND_MAX = 0.65


def _draw(cause: str, T_: int, tau: int, p: float, k: float, mu_z: float, rng):
    if cause == "C0":
        return inject_C0(T_, tau, p, k, mu_z, rng)
    if cause == "C5_size":
        return inject_C5(T_, tau, p, k, mu_z, rng, target="size")
    if cause == "C5_occ":
        return inject_C5(T_, tau, p, k, mu_z, rng, target="occurrence")
    return INJECTORS[cause](T_, tau, p, k, mu_z, rng)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """One panel of idealised series: the injections with NO forecasting error added.

    Isolating the injection from the forecaster is the point -- gate 4.3 asks whether
    the *signature* is present, not whether a particular forecaster finds it. That is
    what Step 5 tests.
    """
    rng = np.random.default_rng(42)
    rows = []
    for cause in ["C0", "C1", "C3", "C4", "C5_size", "C5_occ", "C6"]:
        for _ in range(N_SERIES):
            cell = rng.choice(CELLS)
            p, k, mu_z = get_cell_params(cell, rng)
            y, delta, z, params = _draw(cause, T, TAU, p, k, mu_z, rng)

            # The idealised forecast error: the baseline parameters ARE the forecast.
            e_sz = z - mu_z
            e_sz[delta == 0] = np.nan
            e_occ = delta - p

            post_sz = e_sz[-W_SZ:]
            post_occ = e_occ[-W_SZ:]
            post_delta = delta[-W_SZ:]

            c1 = compute_bias_stats(
                e_sz[-W_SHORT:], np.ones(W_SHORT, dtype=bool), min_evidence=8
            )["z_b"]

            disp = compute_dispersion_stats(
                y[-W_SZ:], y[BASE], min_evidence=C4_MIN_EVIDENCE
            )

            sz_sigma = robust_sd(e_sz[BASE][~np.isnan(e_sz[BASE])])
            rows.append(
                {
                    "cause": cause,
                    "p": p,
                    "k": k,
                    "c1_stat": abs(c1) if np.isfinite(c1) else np.nan,
                    "c4_stat": (
                        abs(np.log(disp["r"]))
                        if np.isfinite(disp["r"]) and disp["r"] > 0
                        else np.nan
                    ),
                    "c4_below_budget": disp["below_budget"],
                    "c4_n_now": disp["n_now"],
                    "c5_sz_stat": compute_cusum_stats(
                        post_sz[~np.isnan(post_sz)], sz_sigma
                    )["max_S"],
                    "c5_occ_stat": compute_cusum_stats(
                        post_occ, np.sqrt(p * (1 - p))
                    )["max_S"],
                    "c6_stat": abs(
                        compute_intermittency_stats(
                            delta[-W_CLASS:], np.ones(W_CLASS, dtype=bool)
                        )["adi"]
                        - 1.0 / p
                    ),
                    "c3_stat": abs(float(np.mean(delta[-60:])) - p),
                }
            )
    return pd.DataFrame(rows)


def _auc(df: pd.DataFrame, target: str, col: str, *, drop_nan: bool = True) -> float:
    """AUC separating ``target`` from C0 on ``col``, oriented so higher = more target."""
    d = df[df["cause"].isin([target, "C0"])]
    if drop_nan:
        d = d[d[col].notna()]
    y_true = (d["cause"] == target).astype(int).to_numpy()
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return 0.5
    return float(roc_auc_score(y_true, d[col].to_numpy()))


# --------------------------------------------------------------------------- #
# Gate 4.3 -- targeted separation
# --------------------------------------------------------------------------- #


def test_gate_4_3_targeted_separation(panel: pd.DataFrame) -> None:
    """Gate 4.3: each cause's own statistic separates it from C0 at AUC >= 0.80.

    C4 is evaluated on the subset where its evidence floor is met (constraint 9). The
    floor is an ex-ante observable -- the count of nonzero observations in the window --
    so this is a pre-registered restriction, not a selected one. The below-budget share
    is asserted separately in ``test_gate_4_9_c4_detection_budget``.
    """
    got = {
        "C1": _auc(panel, "C1", "c1_stat"),
        "C3": _auc(panel, "C3", "c3_stat"),
        "C4": _auc(panel[~panel["c4_below_budget"]], "C4", "c4_stat"),
        "C5_size": _auc(panel, "C5_size", "c5_sz_stat"),
        "C5_occ": _auc(panel, "C5_occ", "c5_occ_stat"),
        "C6": _auc(panel, "C6", "c6_stat"),
    }
    print("\ngate 4.3 -- targeted separation")
    for cause, auc in got.items():
        print(f"  {cause:<8} AUC {auc:.3f}   {'ok' if auc >= AUC_MIN else 'FAIL'}")

    failures = {c: a for c, a in got.items() if a < AUC_MIN}
    assert not failures, (
        f"gate 4.3: {failures} below {AUC_MIN}. Revise §15.2's injection table "
        "(magnitudes, not detectors) -- do NOT tune the monitor to pass this."
    )


# --------------------------------------------------------------------------- #
# Gate 4.4 -- the confound matrix
# --------------------------------------------------------------------------- #


def test_gate_4_4_confounds(panel: pd.DataFrame) -> None:
    """Gate 4.4: a cause must not move another cause's statistic (AUC <= 0.65).

    ``C5_size`` on C4 is included deliberately: on the pre-revision absolute dispersion
    ratio it scored **0.805**, which is why the statistic was moved to the log scale.
    """
    checks = {
        "C1->C4": ("C1", "c4_stat"),
        "C4->C1": ("C4", "c1_stat"),
        "C3->C4": ("C3", "c4_stat"),
        "C5_size->C4": ("C5_size", "c4_stat"),
        "C6->C1": ("C6", "c1_stat"),
    }
    print("\ngate 4.4 -- confounds (must stay at or below 0.65)")
    got = {}
    for name, (cause, col) in checks.items():
        sub = panel[~panel["c4_below_budget"]] if col == "c4_stat" else panel
        got[name] = _auc(sub, cause, col)
        print(f"  {name:<14} AUC {got[name]:.3f}")

    bad = {k: v for k, v in got.items() if v > CONFOUND_MAX}
    assert not bad, (
        f"gate 4.4: {bad} above {CONFOUND_MAX}. The injection moves a statistic it "
        "claims to leave alone -- revise §15.2, do not widen the bar."
    )


# --------------------------------------------------------------------------- #
# Constraint 9 -- C4's detectability budget
# --------------------------------------------------------------------------- #


def test_gate_4_9_c4_detection_budget(panel: pd.DataFrame) -> None:
    """Constraint 9: below-budget SKUs are FLAGGED, and the budget is not a fiction.

    The statistic returns ``below_budget`` rather than a silent NaN, so the SKU is
    excluded from the budget-restricted recall (§29.3) and reported as not evaluable.
    Both facts are asserted: the flag exists, and the floor is the configured one.
    """
    c4 = panel[panel["cause"] == "C4"]
    n_below = int(c4["c4_below_budget"].sum())
    n_ok = int((~c4["c4_below_budget"]).sum())
    print(
        f"\nconstraint 9 -- C4 evidence budget at W_sz={W_SZ}, "
        f"floor={C4_MIN_EVIDENCE}: {n_ok} evaluable, {n_below} flagged below budget"
    )

    assert n_ok >= 50, "too few C4 series clear the evidence floor; the panel is unusable"
    assert set(c4["c4_below_budget"].unique()) <= {True, False}

    # The flag is exactly the configured floor applied to the observable nonzero count.
    for _, row in c4.iterrows():
        assert row["c4_below_budget"] == (not check_c4_budget(row["c4_n_now"], C4_MIN_EVIDENCE))

    # C0 must be subject to the same floor, or the AUC above compares unlike groups.
    c0 = panel[panel["cause"] == "C0"]
    assert c0["c4_below_budget"].any() and (~c0["c4_below_budget"]).any()
