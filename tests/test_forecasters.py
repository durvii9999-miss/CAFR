"""Forecaster pool gates 2.1-2.5. Rev 2 §30.1 Step 2, handoff §10.2.

Gate 2.1 -- "Croston / SBA / TSB / mSBA / mTSB / SES-on-sizes vs ``statsforecast``,
fixed 200-period series, fixed seed, relative error <= 1e-6."

**The gate as written cannot be satisfied for mSBA and mTSB.** ``statsforecast``
implements ``CrostonClassic``, ``CrostonSBA``, ``TSB`` and ``SimpleExponentialSmoothing``
-- and nothing named mSBA or mTSB. Rev 2 §15.6 names them in the pool but never pins
their definitions. That is a specification gap, and it is escalated as a finding
rather than closed by inventing a referent.

What is done instead, so the two models are still **tested rather than trusted**:

* ``msba`` is defined as SBA with independent size/interval smoothing constants, so
  ``mSBA(alpha_z == alpha_p)`` must reduce to ``SBA`` **exactly**.
* ``mtsb`` is defined as TSB with a damped-trend size sub-model, so
  ``mTSB(phi=1, beta=0)`` must reduce to ``TSB`` **exactly**.

A model with no external referent and no exact reduction would be untestable. These
have an exact reduction. Record the adopted definitions in T7.
"""

from __future__ import annotations

import numpy as np
import pytest

from cafr.forecasters.base import quantile_ceiling, ses_forecast
from cafr.forecasters.croston import Croston
from cafr.forecasters.msba import mSBA
from cafr.forecasters.mtsb import mTSB
from cafr.forecasters.sba import SBA
from cafr.forecasters.ses import SESOnSizes
from cafr.forecasters.tsb import TSB

GATE_TOL = 1e-6
SERIES_LEN = 200


def rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


@pytest.fixture(scope="module")
def intermittent_series() -> np.ndarray:
    """A fixed 200-period intermittent series, fixed seed. Gate 2.1's input."""
    rng = np.random.default_rng(20260923)
    n = SERIES_LEN
    delta = rng.random(n) < 0.3
    sizes = rng.gamma(shape=2.0, scale=8.0, size=n)
    y = np.where(delta, sizes, 0.0)
    y[-30:] = 0.0                      # a tail of zeros, so TSB's decay is exercised
    return np.round(y, 6)


# --------------------------------------------------------------------------- #
# Gate 2.1 -- the external referent                                            #
# --------------------------------------------------------------------------- #


def test_ses_matches_statsforecast(intermittent_series):
    """``ses_forecast`` is the shared primitive -- if it is wrong, everything is."""
    sf = pytest.importorskip("statsforecast.models")
    ours, _ = ses_forecast(intermittent_series, 0.1)
    theirs, _ = sf._ses_forecast(intermittent_series.astype("float64"), 0.1)
    assert rel_err(ours, theirs) <= GATE_TOL


def test_croston_matches_statsforecast(intermittent_series):
    sf = pytest.importorskip("statsforecast.models")
    ours = Croston(alpha=0.1).fit(intermittent_series).mu
    theirs = float(sf._croston_classic(intermittent_series.astype("float64"), h=1, fitted=False)["mean"][0])
    assert rel_err(ours, theirs) <= GATE_TOL, f"croston {ours} vs statsforecast {theirs}"


def test_sba_matches_statsforecast(intermittent_series):
    sf = pytest.importorskip("statsforecast.models")
    ours = SBA(alpha=0.1).fit(intermittent_series).mu
    theirs = float(sf._croston_sba(intermittent_series.astype("float64"), h=1, fitted=False)["mean"][0])
    assert rel_err(ours, theirs) <= GATE_TOL, f"sba {ours} vs statsforecast {theirs}"


def test_tsb_matches_statsforecast(intermittent_series):
    sf = pytest.importorskip("statsforecast.models")
    ours = TSB(alpha_d=0.2, alpha_p=0.2).fit(intermittent_series).mu
    theirs = float(
        sf._tsb(intermittent_series.astype("float64"), h=1, fitted=False,
                alpha_d=0.2, alpha_p=0.2)["mean"][0]
    )
    assert rel_err(ours, theirs) <= GATE_TOL, f"tsb {ours} vs statsforecast {theirs}"


def test_ses_on_sizes_size_component_matches_statsforecast(intermittent_series):
    """SES-on-sizes has no statsforecast counterpart, but its SIZE sub-model does.

    The size sub-model is SES on the **nonzero sizes**, so the exact reduction needs a
    series with no zeros: there, ``nonzero == y`` and occurrence is 1.0, and the model
    is plain SES. That is then checked against
    ``statsforecast.models.SimpleExponentialSmoothing``.
    """
    sf = pytest.importorskip("statsforecast.models")
    sizes = intermittent_series[intermittent_series > 0]
    assert (sizes > 0).all()

    ours = SESOnSizes(alpha=0.1, p_fixed=1.0).fit(sizes).mu
    assert rel_err(ours, ses_forecast(sizes, 0.1)[0]) <= GATE_TOL

    model = sf.SimpleExponentialSmoothing(alpha=0.1)
    model.fit(sizes.astype("float64"))
    predicted = float(np.asarray(model.predict(h=1)["mean"])[0])
    assert rel_err(ours, predicted) <= GATE_TOL


def test_ses_on_sizes_occurrence_rate_is_a_plain_bernoulli_rate(intermittent_series):
    """The occurrence sub-model is a trailing UNWEIGHTED rate -- not TSB's smoothing.

    That is the whole difference between this pool member and TSB (Rev 2 §15.6's own
    wording: "Occurrence via a separate Bernoulli rate").
    """
    window = 24
    state = SESOnSizes(alpha=0.1, occurrence_window=window).fit(intermittent_series)
    tail = intermittent_series[-window:]
    assert state.z == pytest.approx(float((tail != 0).mean()))
    assert state.z != pytest.approx(TSB(alpha_d=0.1, alpha_p=0.1).fit(intermittent_series).z)


# --------------------------------------------------------------------------- #
# The replacements for the two missing referents                               #
# --------------------------------------------------------------------------- #


def test_msba_reduces_exactly_to_sba(intermittent_series):
    """mSBA's definition is only defensible if this holds exactly."""
    assert rel_err(mSBA(alpha_z=0.1, alpha_p=0.1).fit(intermittent_series).mu,
                   SBA(alpha=0.1).fit(intermittent_series).mu) <= 1e-12


def test_mtsb_reduces_exactly_to_tsb(intermittent_series):
    """mTSB's definition is only defensible if this holds exactly."""
    assert rel_err(mTSB(alpha_d=0.2, alpha_p=0.2, beta=0.0, phi=1.0).fit(intermittent_series).mu,
                   TSB(alpha_d=0.2, alpha_p=0.2).fit(intermittent_series).mu) <= 1e-12


# --------------------------------------------------------------------------- #
# Gate 2.2 -- non-negative points, monotone quantiles                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["croston", "sba", "tsb", "msba", "mtsb", "ses_sizes"])
def test_gate_2_2_non_negative_and_monotone(name, intermittent_series):
    from cafr.forecasters.registry import POOL_NAMES, make_forecaster

    assert name in POOL_NAMES
    model = make_forecaster(name, {})
    model.fit(intermittent_series)

    residuals = intermittent_series - np.mean(intermittent_series)
    q = model.predict_quantiles(residuals, alphas=(0.5, 0.8, 0.9, 0.95))
    los = [q[a][0] for a in (0.5, 0.8, 0.9, 0.95)]
    his = [q[a][1] for a in (0.5, 0.8, 0.9, 0.95)]

    assert all(bool(np.isfinite(v)) for v in los + his)
    assert min(los) >= 0.0 and min(his) >= 0.0, "point/interval forecasts must be non-negative"
    assert los == sorted(los), f"lower bounds not monotone in alpha: {los}"
    assert his == sorted(his), f"upper bounds not monotone in alpha: {his}"


def test_quantile_ceiling_is_the_statistical_limit():
    """n = 30 cycles caps the estimable quantile at ~0.968, not 0.999."""
    assert quantile_ceiling(30) == pytest.approx(1.0 - 1.0 / 31.0)
    assert quantile_ceiling(30) < 0.999
    assert quantile_ceiling(10**6) == pytest.approx(0.999)
    assert quantile_ceiling(0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Gate 2.3 -- TSB decays to zero                                              #
# --------------------------------------------------------------------------- #


def test_gate_2_3_tsb_decays_on_an_all_zero_series():
    """Final z_hat < 0.01 after 200 zero periods.

    TSB is the only pool member that can decay to zero, which is why it is a
    candidate remedy for C3 (remedy R3).
    """
    y = np.concatenate([np.full(5, 10.0), np.zeros(200)])
    state = TSB(alpha_d=0.2, alpha_p=0.2).fit(y)
    assert state.z < 0.01, f"TSB z_hat did not decay: {state.z}"


def test_tsb_all_zero_series_is_zero():
    state = TSB().fit(np.zeros(200))
    assert state.mu == 0.0 and state.z == 0.0


# --------------------------------------------------------------------------- #
# Gate 2.4 -- LightGBM determinism                                             #
# --------------------------------------------------------------------------- #


def test_gate_2_4_lightgbm_is_deterministic():
    """Fixed seed + n_jobs=1. Two runs must be byte-identical."""
    import pandas as pd

    from cafr.forecasters.lightgbm_global import GlobalLightGBM

    rng = np.random.default_rng(7)
    skus, rows = [], []
    for i in range(30):
        p = rng.uniform(0.1, 0.5)
        k = rng.uniform(1.0, 5.0)
        mu_z = rng.uniform(10, 100)
        delta = rng.random(60) < p
        y = np.where(delta, rng.gamma(k, mu_z / k, 60), 0.0)
        rows.append(pd.DataFrame({
            "sku_id": f"S{i:03d}", "period": np.arange(60),
            "demand_observed": y, "lead_time": 2, "unit_cost": 10.0,
            "holding_rate": 0.02, "adi_init": 1.0 / p, "cv2_init": 1.0 / k,
        }))
    panel = pd.concat(rows, ignore_index=True)

    preds = []
    for _ in range(2):
        model = GlobalLightGBM(params={"n_estimators": 40}, seed=42).fit_global(panel)
        preds.append(model._model.predict(
            np.zeros((1, len(model._model.feature_name_)), dtype="float64")
        ))
    assert np.array_equal(preds[0], preds[1]), "LightGBM is not deterministic"


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["croston", "sba", "tsb", "msba", "mtsb", "ses_sizes"])
def test_all_zero_series_produces_zero_forecast(name):
    from cafr.forecasters.registry import make_forecaster

    state = make_forecaster(name, {}).fit(np.zeros(50))
    assert state.mu == 0.0, f"{name} forecast {state.mu} on an all-zero series"


@pytest.mark.parametrize("name", ["croston", "sba", "tsb", "msba", "mtsb", "ses_sizes"])
def test_single_observation(name):
    from cafr.forecasters.registry import make_forecaster

    state = make_forecaster(name, {}).fit(np.asarray([7.0]))
    assert np.isfinite(state.mu) and state.mu >= 0.0
