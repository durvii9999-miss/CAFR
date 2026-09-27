"""``fit_on`` -- gate 1.7 and gate 2.5. Rev 2 §14.3 rule 6.

``fit_on`` is the project's most tempting leak. ``uncensored`` is supposed to be the
honest middle path: impute the lost sales from **observables** and fit on that. If the
imputation could see ``demand_lost`` or ``demand_true``, it would stop being an
imputation and become the answer, and every C7 number in the paper would be measured
against a forecaster that was handed the stock-outs.

So the separation is enforced by **signature and by a shared switch**, not by
discipline:

* ``impute_lost_sales`` has no parameter that could carry the truth.
* every model fits through ``FittingInput.resolve()`` -- one function, one switch.

Both are asserted structurally below.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from cafr.data.synth.impute import impute_lost_sales
from cafr.forecasters.base import FittingInput
from cafr.forecasters.registry import POOL_NAMES, make_forecaster

FORBIDDEN_IN_SIGNATURE = ("true", "lost", "cause", "ground", "injection", "label")


# --------------------------------------------------------------------------- #
# Gate 1.7 -- the config row, and the imputation's signature                    #
# --------------------------------------------------------------------------- #


def test_gate_1_7_config_row_is_present_and_valid(cfg):
    """The switch exists, is logged in the config, and has a closed value set."""
    f = cfg["forecast"]
    assert "fit_on" in f, "the config row gate 1.7 asks for is absent"
    assert f["fit_on"] in f["fit_on_valid"]
    assert set(f["fit_on_valid"]) == {"observed", "uncensored", "true"}


def test_gate_1_7_ruf_forces_observed(cfg):
    """H-3: RUF has no stock-out record, so ``uncensored`` is not available there.

    Not a preference -- an imputation needs a censoring mask, and RUF's is all-False.
    Forcing ``observed`` is what keeps the RUF numbers honest.
    """
    assert cfg["panels"]["ruf"]["fit_on"] == "observed"


def test_gate_1_7_imputation_signature_refuses_ground_truth():
    """No parameter of the imputation may be able to carry ``demand_lost``/``true``.

    Asserted on the signature, because a runtime check could be bypassed by the next
    person who adds a convenience argument. A signature cannot.
    """
    names = list(inspect.signature(impute_lost_sales).parameters)
    assert names == ["y_observed", "censored", "rng"], (
        f"impute_lost_sales takes {names}; it must take observables only"
    )
    for name in names:
        assert not any(bad in name for bad in FORBIDDEN_IN_SIGNATURE), name


def test_gate_1_7_fitting_input_exposes_the_same_three_modes():
    params = inspect.signature(FittingInput).parameters
    assert "fit_on" in params
    assert "y_true" in params, (
        "the idealisation must be an explicit, nameable field so it can be declared"
    )


def test_imputation_uses_only_uncensored_periods():
    """A censored period must never contribute to the size distribution it draws from."""
    y = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    censored = np.array([False, True, True, True, True])
    out = impute_lost_sales(y, censored, np.random.default_rng(0))
    # Only period 0 is uncensored and nonzero -> every draw must be exactly 10.
    assert np.allclose(out.y[1:], 10.0)
    assert out.y[0] == 10.0
    assert out.n_censored == 4 and out.n_imputed == 4 and not out.fallback


def test_imputation_falls_back_rather_than_reaching_for_the_truth():
    """No uncensored nonzero history -> leave the series alone and SAY SO."""
    y = np.array([0.0, 0.0, 5.0])
    censored = np.array([False, False, True])
    out = impute_lost_sales(y, censored, np.random.default_rng(0))
    assert out.fallback and out.n_imputed == 0
    assert np.array_equal(out.y, y), "the fallback changed the series"


def test_imputation_is_inert_without_censoring():
    y = np.array([3.0, 0.0, 7.0])
    out = impute_lost_sales(y, np.zeros(3, dtype=bool), np.random.default_rng(0))
    assert np.array_equal(out.y, y) and out.n_censored == 0


# --------------------------------------------------------------------------- #
# FittingInput.resolve -- THE switch                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fitting_input():
    y_observed = np.array([5.0, 0.0, 0.0, 8.0, 0.0])
    censored = np.array([False, True, True, False, False])
    y_true = np.array([5.0, 4.0, 4.0, 8.0, 0.0])
    return lambda fit_on: FittingInput(
        y_observed=y_observed, censored=censored, fit_on=fit_on, rng=np.random.default_rng(1),
        y_true=y_true,
    )


def test_resolve_observed_returns_the_series_as_seen(fitting_input):
    assert np.array_equal(fitting_input("observed").resolve(), [5.0, 0.0, 0.0, 8.0, 0.0])


def test_resolve_true_returns_the_truth(fitting_input):
    assert np.array_equal(fitting_input("true").resolve(), [5.0, 4.0, 4.0, 8.0, 0.0])


def test_resolve_uncensored_imputes_upward_only_on_censored_periods(fitting_input):
    r = fitting_input("uncensored").resolve()
    assert r[0] == 5.0 and r[3] == 8.0 and r[4] == 0.0, "uncensored periods must not move"
    assert r[1] > 0.0 and r[2] > 0.0, "censored zero periods must receive a draw"
    assert not np.array_equal(r, fitting_input("observed").resolve())


def test_resolve_uncensored_is_not_the_truth(fitting_input):
    """The middle path must be a middle path -- not the truth wearing its name."""
    assert not np.array_equal(
        fitting_input("uncensored").resolve(), fitting_input("true").resolve()
    )


def test_resolve_true_without_truth_raises(fitting_input):
    fi = FittingInput(
        y_observed=np.array([1.0]), censored=np.array([False]), fit_on="true"
    )
    with pytest.raises(ValueError, match="requires y_true"):
        fi.resolve()


def test_unknown_fit_on_is_rejected():
    with pytest.raises(ValueError, match="unknown fit_on"):
        FittingInput(y_observed=np.array([1.0]), censored=np.array([False]), fit_on="anything")


# --------------------------------------------------------------------------- #
# Gate 2.5 -- every model respects fit_on                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pooled_model(cfg):
    """A trained pooled LightGBM, so the fit_on path can be exercised rather than skipped.

    Four SKUs is enough to fit; the point is the plumbing, not the accuracy. Training
    it once at module scope keeps the parametrised test cheap.
    """
    import pandas as pd

    from cafr.forecasters.lightgbm_global import GlobalLightGBM

    rng = np.random.default_rng(5)
    rows = []
    for i in range(4):
        y = np.where(rng.random(40) < 0.3, rng.gamma(2.0, 5.0, 40), 0.0)
        rows.append(pd.DataFrame({
            "sku_id": f"P{i}", "period": np.arange(40), "demand_observed": y,
            "lead_time": 2, "unit_cost": 10.0, "holding_rate": 0.02,
            "adi_init": 3.0, "cv2_init": 0.5,
        }))
    model = GlobalLightGBM(params={"n_estimators": 20}, seed=cfg["seed_root"])
    return model.fit_global(pd.concat(rows, ignore_index=True))


@pytest.mark.parametrize("name", POOL_NAMES)
@pytest.mark.parametrize("mode", ["observed", "uncensored", "true"])
def test_gate_2_5_every_model_logs_the_mode_it_used(name, mode, cfg, pooled_model):
    fi = FittingInput(
        y_observed=np.array([5.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 6.0]),
        censored=np.array([False, True, True, False, False, False, False, False]),
        fit_on=mode,
        rng=np.random.default_rng(3),
        y_true=np.array([5.0, 4.0, 4.0, 8.0, 0.0, 3.0, 0.0, 6.0]),
    )

    if name == "lightgbm_global":
        # Trained once, pooled, then applied per series. It carries no fitting series
        # of its own, so the switch is RECORDED rather than consumed -- assert that,
        # rather than skipping the model and leaving a hole in the gate.
        state = pooled_model.fit_series_with_statics(fi.resolve(), {}, fit_on=mode)
        assert state.diag["fit_on"] == mode
        return

    model = make_forecaster(name, cfg)
    state = model.fit(fi)
    assert state.diag["fit_on"] == mode
    assert np.array_equal(model.resolved_series, fi.resolve()), (
        f"{name} did not fit on the series fit_on={mode} resolves to"
    )


def test_fitting_input_resolve_is_idempotent():
    """Two calls must return the SAME series -- they draw from the same rng.

    Without memoisation the second call returns a different imputation, so two
    forecasters in one run would fit on two different versions of the same SKU.
    Invisible, and enough to void a paired comparison.
    """
    fi = FittingInput(
        y_observed=np.array([5.0, 0.0, 0.0, 8.0]),
        censored=np.array([False, True, True, False]),
        fit_on="uncensored", rng=np.random.default_rng(7),
    )
    assert fi.resolve() is fi.resolve()


@pytest.mark.parametrize("name", POOL_NAMES)
def test_gate_2_5_all_arms_would_read_the_same_value(name, cfg, pooled_model):
    """A run is one ``fit_on``. Two modes in one run is a bug, and it is silent."""
    fi = FittingInput(
        y_observed=np.array([5.0, 0.0, 0.0, 8.0]), censored=np.array([False, True, True, False]),
        fit_on="uncensored", rng=np.random.default_rng(3),
    )
    if name == "lightgbm_global":
        assert pooled_model.fit_series_with_statics(fi.resolve(), {}, fit_on="uncensored").diag[
            "fit_on"
        ] == "uncensored"
        return
    assert make_forecaster(name, cfg).fit(fi).diag["fit_on"] == "uncensored"


def test_fit_on_actually_changes_the_forecast(cfg):
    """If the switch were a no-op the gates above would pass while nothing happened."""
    y_observed = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0])
    censored = np.array([False, True, True, True, False, True, True, True, False])
    y_true = np.array([10.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 10.0])

    def fit(mode):
        fi = FittingInput(
            y_observed=y_observed, censored=censored, fit_on=mode,
            rng=np.random.default_rng(0), y_true=y_true,
        )
        return make_forecaster("sba", cfg).fit(fi).mu

    observed, uncensored, true = fit("observed"), fit("uncensored"), fit("true")
    assert observed < uncensored, "imputing lost sales must raise the level"
    assert observed < true, "fitting on the truth must raise the level"
    assert uncensored != true, (
        "uncensored and true agree exactly here, which would make the middle path "
        "indistinguishable from the idealisation"
    )
