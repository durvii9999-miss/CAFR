"""§27 data contracts, and the burn-in-only guarantee on ``adi_init`` / ``cv2_init``.

Two jobs:

1. Every committed parquet must validate against its §27 contract **with set
   equality** -- a missing column and an unexpected column are both errors.
2. ``sku_meta.adi_init`` / ``cv2_init`` must be **burn-in only**, and
   ``sku_attributes``' ``adi_full`` / ``cv_squared_full`` must be a *different*
   number. If they agreed, the split would be cosmetic and the leak the split exists
   to prevent would be live. That is asserted here, not assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cafr.data.schema import (
    DEMAND_PANEL_COLUMNS,
    SKU_ATTRIBUTES_COLUMNS,
    SKU_META_COLUMNS,
    SchemaError,
    validate_frame,
)
from cafr.utils.io import read_parquet
from conftest import repo_path

BURN_IN = 24


@pytest.fixture(scope="module")
def demand_panel():
    return read_parquet(repo_path("data/ruf/demand_panel.parquet"))


@pytest.fixture(scope="module")
def sku_meta():
    return read_parquet(repo_path("data/ruf/sku_meta.parquet"))


@pytest.fixture(scope="module")
def sku_attributes():
    return read_parquet(repo_path("data/ruf/sku_attributes.parquet"))


# --------------------------------------------------------------------------- #
# The contracts                                                                #
# --------------------------------------------------------------------------- #


def test_demand_panel_matches_contract(demand_panel):
    validate_frame(demand_panel, DEMAND_PANEL_COLUMNS, name="demand_panel")


def test_sku_meta_matches_contract(sku_meta):
    validate_frame(sku_meta, SKU_META_COLUMNS, name="sku_meta")


def test_sku_attributes_matches_contract(sku_attributes):
    validate_frame(sku_attributes, SKU_ATTRIBUTES_COLUMNS, name="sku_attributes")


_KIND_TO_DTYPE = {"str": object, "int": "int64", "float": "float64", "bool": bool}


def _conforming(contract, n: int = 4) -> pd.DataFrame:
    """A frame carrying exactly ``contract``'s columns at the declared dtypes.

    Built from the contract rather than sliced out of a real file, so it works for
    every contract regardless of which file happens to hold those columns.
    """
    return pd.DataFrame({
        col: np.zeros(n, dtype=_KIND_TO_DTYPE[kind]) if kind != "str" else ["x"] * n
        for col, kind in contract.items()
    })


@pytest.mark.parametrize(
    "contract, name",
    [
        (DEMAND_PANEL_COLUMNS, "demand_panel"),
        (SKU_META_COLUMNS, "sku_meta"),
        (SKU_ATTRIBUTES_COLUMNS, "sku_attributes"),
    ],
)
def test_contracts_reject_missing_and_extra(contract, name, demand_panel):
    """Set equality: both directions must fail. A one-way check is a blacklist."""
    with pytest.raises(SchemaError, match="missing declared columns"):
        validate_frame(demand_panel.drop(columns=["sku_id"]), contract, name=name)

    conforming = _conforming(contract)
    validate_frame(conforming, contract, name=name)     # the control: this one passes
    with pytest.raises(SchemaError, match="undeclared columns present"):
        validate_frame(conforming.assign(surprise=1.0), contract, name=name)


def test_contracts_check_dtypes(demand_panel):
    bad = demand_panel.copy()
    bad["demand_observed"] = bad["demand_observed"].astype("int64")
    with pytest.raises(SchemaError, match="expected float"):
        validate_frame(bad, DEMAND_PANEL_COLUMNS, name="demand_panel")


# --------------------------------------------------------------------------- #
# The no-invented-stock-outs rule (build_ruf_panel.py header, handoff §6.1)     #
# --------------------------------------------------------------------------- #


def test_ruf_does_not_invent_stockouts(demand_panel):
    """RUF has no stock-out record. ``False`` everywhere is the honest encoding.

    The simulator generates censoring downstream. Anything else here would be a
    fabricated ground truth (handoff §14).
    """
    assert not demand_panel["stockout_flag"].any()
    assert np.array_equal(
        demand_panel["demand_observed"].to_numpy(),
        demand_panel["demand_true"].to_numpy(),
    )


def test_panel_length_and_geometry(demand_panel):
    """60 periods, 5,000 SKUs -- decision H-1, the number the split depends on."""
    T = demand_panel["period"].nunique()
    assert T == 60, f"expected 60 periods (H-1), got {T}"
    assert demand_panel["sku_id"].nunique() == 5000
    assert demand_panel.groupby("sku_id", sort=False).size().eq(T).all(), (
        "ragged panel: H-1's split arithmetic assumes a balanced panel"
    )
    assert BURN_IN + 36 == T, "burn-in 24 + evaluated 36 must exactly consume the panel"


# --------------------------------------------------------------------------- #
# adi_init / cv2_init are burn-in only -- recomputed here, not trusted          #
# --------------------------------------------------------------------------- #


def _burn_in_stats(vals: np.ndarray, burn: int) -> tuple[float, float]:
    """Re-implementation of build_ruf_panel.burn_in_stats, written from the spec.

    Deliberately NOT imported: importing the producer would make this test assert
    that the function equals itself. Written independently so a change in the
    builder that broke the burn-in-only rule would be caught here.
    """
    w = vals[:burn]
    nz = w[w > 0]
    adi = float(len(w)) / len(nz) if len(nz) else float(len(w))
    if len(nz) >= 2 and nz.mean() > 0:
        cv2 = float((np.std(nz, ddof=1) / nz.mean()) ** 2)
    else:
        cv2 = float("nan")
    return adi, cv2


def test_adi_init_and_cv2_init_are_burn_in_only(demand_panel, sku_meta):
    """Recompute ADI/CV^2 on the first 24 periods and require an exact match.

    This is the recomputation assertion §27 asks for. If adi_init had been computed
    on the full series, the monitor would be reading a post-regime statistic at
    t = 0 -- a direct ground-truth leak, and the worse kind, because it is a leak of
    the *answer* rather than merely of the future.
    """
    wide = (
        demand_panel.sort_values(["sku_id", "period"])
        .pivot(index="sku_id", columns="period", values="demand_observed")
    )
    vals = wide.to_numpy(dtype="float64")
    recomputed = pd.DataFrame(
        [_burn_in_stats(vals[i], BURN_IN) for i in range(vals.shape[0])],
        columns=["adi", "cv2"],
        index=wide.index,
    )
    merged = sku_meta.set_index("sku_id").join(recomputed)

    assert np.allclose(merged["adi_init"], merged["adi"], rtol=0, atol=1e-12), (
        "adi_init is not the burn-in ADI"
    )
    both = merged["cv2_init"].notna() & merged["cv2"].notna()
    assert np.allclose(
        merged.loc[both, "cv2_init"], merged.loc[both, "cv2"], rtol=0, atol=1e-12
    ), "cv2_init is not the burn-in CV^2"
    assert merged.loc[~both, "cv2_init"].isna().all(), (
        "cv2_init must be NaN exactly where the burn-in has < 2 nonzero periods"
    )


def test_burn_in_and_full_series_statistics_differ(sku_meta, sku_attributes):
    """The guard is only meaningful if the two numbers are actually different."""
    merged = sku_meta.merge(
        sku_attributes[["sku_id", "adi_full", "cv_squared_full"]], on="sku_id"
    )
    adi_differs = ~np.isclose(merged["adi_init"], merged["adi_full"], rtol=1e-9, atol=1e-9)
    assert adi_differs.mean() > 0.5, (
        "adi_init equals adi_full for most SKUs -- the burn-in split is not doing anything"
    )
    assert (merged.loc[merged["cv2_init"].notna(), "cv2_init"]
            != merged.loc[merged["cv2_init"].notna(), "cv_squared_full"]).mean() > 0.5


def test_full_series_columns_live_only_in_the_evaluation_file(
    sku_meta, sku_attributes
):
    """``adi_full`` / ``cv_squared_full`` must be unreachable from the policy's file.

    Two separate FILES, not two columns. A column in ``sku_meta`` would ride along on
    every merge; a separate file has to be opened deliberately.
    """
    assert "adi_full" not in sku_meta.columns
    assert "cv_squared_full" not in sku_meta.columns
    assert "adi_full" in sku_attributes.columns
    assert "cv_squared_full" in sku_attributes.columns


def test_load_observed_never_carries_an_evaluation_column(sku_meta):
    """``load_observed``'s output columns must be demand + burn-in metadata only.

    Asserted as a whitelist of the two source contracts, not as a blacklist of the
    two known leak columns -- a future full-series statistic added to the attributes
    file would sail past a blacklist.
    """
    from cafr.data.loaders.ruf import load_observed

    observed = load_observed()
    allowed = (
        {"sku_id", "period", "demand_observed"} | set(sku_meta.columns)
    )
    extra = set(observed.columns) - allowed
    assert not extra, f"load_observed leaked columns it has no business returning: {extra}"


# --------------------------------------------------------------------------- #
# Panel profile -- T1's numbers, and the honesty notes attached to them         #
# --------------------------------------------------------------------------- #


def test_panel_profile_records_the_known_limitations():
    """The three facts the paper must not lose: no censoring record, no ground
    truth, and a generator that cannot be re-run publicly."""
    from cafr.utils.io import read_json

    profile = read_json(repo_path("data/ruf/panel_profile.json"))

    assert profile["n_periods"] == 60
    assert profile["panel_length_shortfall"] == 24, "H-1: 84 assumed, 60 shipped"
    assert profile["censoring_observed"] is False
    assert profile["ground_truth_present"] is False
    assert profile["generator_rerunnable_publicly"] is False
    assert int(profile["dead_skus_in_burn_in"]) >= 0
