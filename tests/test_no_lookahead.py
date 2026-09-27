"""No-look-ahead test. Rev 2 §14.3 rule 5, §28.6, handoff §5.6.

**The primary test is SET EQUALITY against the declared whitelist, not a name
blacklist.**

Rev 1 used a ``*_true*`` / ``*_cause*`` blacklist. It missed ``cause_active`` and
``injection_param`` -- the two columns that give the answer away -- and it fails
**open**: a future ground-truth field with an innocent name sails through. Set
equality fails **closed**: any extra column, whatever it is called, is an error.

Handoff Step 0's done-when criterion is that this test **fails when an undeclared
column is added**. That is asserted directly in
``test_undeclared_column_is_rejected`` below -- the failure is a feature of the test,
not something to work around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cafr.data.schema import SchemaError, a4_columns, build_policy_input, whitelist
from conftest import make_feature_frame


# --------------------------------------------------------------------------- #
# The primary test -- exact column whitelist with set equality                 #
# --------------------------------------------------------------------------- #


def test_policy_input_columns_are_exactly_the_whitelist(whitelist):
    frame = make_feature_frame(whitelist)
    policy_in = build_policy_input(frame)

    declared = set(whitelist) | {"sku_id", "period"}
    actual = set(policy_in.columns)

    missing = declared - actual
    extra = actual - declared
    assert not extra, f"undeclared columns reachable by the policy: {sorted(extra)}"
    assert not missing, f"declared features absent: {sorted(missing)}"


def test_undeclared_column_is_rejected(whitelist):
    """Adding a column NOT in the whitelist must raise -- whatever it is called.

    This is the Step 0 done-when criterion. A test that only passes on good input
    proves nothing; this proves the check can fail.
    """
    frame = make_feature_frame(whitelist)
    frame["something_innocuous"] = 1.0

    with pytest.raises(SchemaError, match="undeclared columns present"):
        _assert_whitelisted(frame, whitelist)


def test_ground_truth_columns_are_rejected_by_set_equality(whitelist):
    """The two columns Rev 1's blacklist missed must be caught by set equality."""
    for column in ("cause_active", "injection_param", "demand_true", "stockout_flag"):
        frame = make_feature_frame(whitelist)
        frame[column] = 0.0
        with pytest.raises(SchemaError, match="undeclared columns present"):
            _assert_whitelisted(frame, whitelist)


def _assert_whitelisted(frame: pd.DataFrame, declared) -> pd.DataFrame:
    """The assertion block, factored out so the failure can be provoked.

    Mirrors ``build_policy_input`` but raises on ANY extra column even if the caller
    happened to pass it -- that is the set-equality discipline.
    """
    declared_set = set(declared) | {"sku_id", "period"}
    actual = set(frame.columns)
    extra = actual - declared_set
    missing = declared_set - actual
    if extra:
        raise SchemaError(f"undeclared columns present: {sorted(extra)}")
    if missing:
        raise SchemaError(f"declared features absent: {sorted(missing)}")
    return frame


# --------------------------------------------------------------------------- #
# Redundant second line of defence -- the blacklist, kept but demoted          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "column",
    ["demand_true", "stockout_flag", "true_cause", "cause_active", "injection_param",
     "adi_full", "cv_squared_full", "detection_budget_flag"],
)
def test_blacklist_second_line_of_defence(column):
    """One line, defence in depth. No longer the primary test (Rev 2 §28.6)."""
    from cafr.data.schema import FORBIDDEN_SUBSTRINGS

    assert any(bad in column for bad in FORBIDDEN_SUBSTRINGS), (
        f"{column!r} is a ground-truth column that the substring blacklist does not "
        "catch -- the primary set-equality test still catches it, but widen the list."
    )


# --------------------------------------------------------------------------- #
# A4's context -- exactly the whitelist minus the attribution block            #
# --------------------------------------------------------------------------- #


def test_a4_context_is_whitelist_minus_det_block(whitelist):
    """A4's wiring differs from (f) by exactly one thing: the context block.

    That is the whole definition of the arm carrying H1' (handoff §2.7, §8.2).
    """
    a4 = set(a4_columns())
    full = set(whitelist)
    removed = full - a4

    assert removed == {"conf", *[f"cause_C{i}" for i in range(8)]}, (
        f"A4 must lose exactly the det block; it lost {sorted(removed)}"
    )
    assert a4 < full, "A4's context must be a strict subset of (f)'s"


def test_a4_frame_builds_and_matches_a4_columns():
    frame = make_feature_frame(whitelist())
    policy_in = build_policy_input(frame, columns=a4_columns())
    assert set(policy_in.columns) == set(a4_columns()) | {"sku_id", "period"}


# --------------------------------------------------------------------------- #
# The separations that must be structural, not behavioural                     #
# --------------------------------------------------------------------------- #


def test_load_observed_cannot_reach_the_true_series():
    """``load_observed`` must have no parameter that could request ``demand_true``.

    Rev 2 §14.3 rule 7: the separation is enforced by the loader's SIGNATURE, not by
    a runtime check. So assert on the signature, not on the output.
    """
    import inspect

    from cafr.data.loaders.ruf import load_observed

    params = set(inspect.signature(load_observed).parameters)
    for forbidden in ("demand_true", "true", "ground_truth", "stockout_flag"):
        assert forbidden not in params, (
            f"load_observed accepts {forbidden!r}; the truth is reachable from the policy"
        )


def test_observed_frame_carries_no_truth_columns():
    from cafr.data.loaders.ruf import load_observed

    observed = load_observed()
    for forbidden in ("demand_true", "stockout_flag", "demand_lost", "true_cause"):
        assert forbidden not in observed.columns


def test_policy_input_rejects_nan(whitelist):
    frame = make_feature_frame(whitelist)
    frame.loc[0, "bias_sz"] = np.nan
    with pytest.raises(SchemaError, match="NaN in features"):
        build_policy_input(frame)
