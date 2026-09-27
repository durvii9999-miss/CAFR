"""I/O contracts and the no-look-ahead enforcement point. Rev 2 §27, handoff §5.6.

Two jobs:

1. **Validate every frame** against its §27 contract on read and write. A run that
   produces a frame violating the contract must FAIL LOUDLY, not silently produce a
   wrong number (Rev 2 §27, "Schema enforcement").

2. **Build the policy's input frame in exactly one place.** ``build_policy_input()``
   is the only function that may hand data to the policy. The no-look-ahead test
   asserts SET EQUALITY between its output columns and the declared whitelist --
   no extra columns, none missing. Any new column, whatever it is named, fails.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..utils.config import load_features

__all__ = [
    "DEMAND_PANEL_COLUMNS",
    "SKU_META_COLUMNS",
    "SKU_ATTRIBUTES_COLUMNS",
    "GROUND_TRUTH_COLUMNS",
    "INVENTORY_LOG_COLUMNS",
    "ACTIONS_COLUMNS",
    "FORECASTS_COLUMNS",
    "SB_CELLS",
    "CAUSES",
    "REMEDIES",
    "SchemaError",
    "validate_frame",
    "build_policy_input",
    "whitelist",
    "a4_columns",
]


class SchemaError(ValueError):
    """A frame does not conform to its §27 contract."""


# --------------------------------------------------------------------------- #
# §27 contracts                                                                #
# --------------------------------------------------------------------------- #

DEMAND_PANEL_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "period": "int",
    "demand_observed": "float",
    "demand_true": "float",
    "stockout_flag": "bool",
    "panel": "str",
}

SKU_META_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "lead_time": "int",
    "unit_cost": "float",
    "holding_rate": "float",
    # Burn-in ONLY (Rev 2 §27). Never full-series.
    "adi_init": "float",
    "cv2_init": "float",
    "sb_cell_init": "str",
}

# EVALUATION ONLY. adi_full / cv_squared_full are computed over the FULL series and
# would leak the post-regime answer into the feature vector. Kept in a separate file
# precisely so sku_meta's contract stays exact. Never load this in the policy or the
# monitor (handoff §6.1).
SKU_ATTRIBUTES_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "lead_time_ruf": "int",
    "protection_period_ruf": "int",
    "review_period_months": "int",
    "holding_cost_per_unit_month": "float",
    "service_level_target": "float",
    "safety_stock_units": "float",
    "reorder_point_units": "float",
    "syntetos_boylan": "str",
    "adi_full": "float",
    "cv_squared_full": "float",
}

# Synthetic panels only. EVERY column here is ground truth and is never reachable
# by the policy. `cause_active` and `injection_param` are exactly the two columns
# that Rev 1's name blacklist missed.
GROUND_TRUTH_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "period": "int",
    "true_cause": "str",
    "cause_active": "bool",
    "injection_param": "str",       # JSON
    "detection_budget_flag": "bool",
    "c2_variant": "str",            # direct | C2b | -
    "panel_gap": "int",
}

INVENTORY_LOG_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "period": "int",
    "arm": "str",
    "on_hand": "float",
    "backorders": "float",
    "order_qty": "float",
    "demand_met": "float",
    "demand_lost": "float",
    "csl_period": "float",
    "fill_rate_period": "float",
    "cost_period": "float",
    "pis": "float",
    "nos": "float",
    "s_level": "float",             # the order-up-to level actually used
    "alpha_t": "float",             # the service probability actually targeted
}

ACTIONS_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "period": "int",
    "arm": "str",
    "attributed_cause": "str",
    "confidence": "float",
    "remedy": "str",
    "chosen_by": "str",             # rule | bandit | forced
    "exploratory": "bool",
    "switched": "bool",
    "dwell_override": "bool",
    "context_block": "str",         # full | inv_only
    # H-2: log whether the W_occ window was full or partial at each decision point.
    "window_full": "bool",
}

FORECASTS_COLUMNS: dict[str, str] = {
    "sku_id": "str",
    "period": "int",
    "arm": "str",
    "method": "str",
    "point": "float",
    "q_lo": "float",
    "q_hi": "float",
    "alpha": "float",
    "fit_on": "str",
}

SB_CELLS: tuple[str, ...] = (
    "moderate_lowdisp",
    "moderate_highdisp",
    "high_lowdisp",
    "high_highdisp",
    "dead",
)

CAUSES: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")
REMEDIES: tuple[str, ...] = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")

_STR = ("object", "string", "str")
_INT = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
_FLOAT = ("float16", "float32", "float64")


def validate_frame(
    df: pd.DataFrame,
    contract: Mapping[str, str],
    *,
    name: str = "<frame>",
    allow_extra: bool = False,
    require_all: bool = True,
) -> pd.DataFrame:
    """Validate ``df`` against a §27 contract. Raise ``SchemaError`` on any violation.

    Set equality by default: a missing column and an unexpected column are both errors.
    """
    declared = set(contract)
    actual = set(df.columns)

    if require_all:
        missing = declared - actual
        if missing:
            raise SchemaError(f"{name}: missing declared columns: {sorted(missing)}")
    if not allow_extra:
        extra = actual - declared
        if extra:
            raise SchemaError(
                f"{name}: undeclared columns present: {sorted(extra)}. "
                "Extend the contract in cafr/data/schema.py if this is intended."
            )

    for col, kind in contract.items():
        if col not in df.columns:
            continue
        dtype = str(df[col].dtype)
        if kind == "str" and dtype not in _STR:
            raise SchemaError(f"{name}.{col}: expected str, got {dtype}")
        if kind == "int" and dtype not in _INT:
            raise SchemaError(f"{name}.{col}: expected int, got {dtype}")
        if kind == "float" and dtype not in _FLOAT:
            raise SchemaError(f"{name}.{col}: expected float, got {dtype}")
        if kind == "bool" and dtype != "bool":
            raise SchemaError(f"{name}.{col}: expected bool, got {dtype}")
    return df


# --------------------------------------------------------------------------- #
# The policy's input frame -- the single enforcement point                     #
# --------------------------------------------------------------------------- #

_FEATURES = load_features()
WHITELIST: tuple[str, ...] = tuple(_FEATURES["policy_features"])
POLICY_KEYS: tuple[str, ...] = tuple(_FEATURES["policy_input_keys"])
A4_EXCLUDED: frozenset[str] = frozenset(
    f for block in _FEATURES["a4_excluded_blocks"]
    for f in _FEATURES["context_blocks"][block]
)
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = tuple(_FEATURES["forbidden_substrings"])


def whitelist() -> tuple[str, ...]:
    """The declared policy feature list (Rev 2 §27)."""
    return WHITELIST


def a4_columns() -> tuple[str, ...]:
    """A4's context: the whitelist minus the attribution block. See features.yaml."""
    return tuple(f for f in WHITELIST if f not in A4_EXCLUDED)


def build_policy_input(
    features: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the ONLY frame the policy is allowed to see.

    The policy is handed this frame and never the full dataframe (Rev 2 §14.3 rule 2:
    enforce by passing a narrowed view, not by discipline).

    Parameters
    ----------
    features
        Long frame keyed by ``(sku_id, period)`` carrying at least the whitelist
        columns plus the identifiers.
    columns
        Pass ``a4_columns()`` for arm A4. Defaults to the full whitelist.

    Raises
    ------
    SchemaError
        If a declared feature is absent, or a forbidden substring appears.
    """
    declared = tuple(columns) if columns is not None else WHITELIST
    keys = list(POLICY_KEYS)

    missing = [c for c in declared if c not in features.columns]
    if missing:
        raise SchemaError(f"policy input: declared features absent: {missing}")
    if not all(k in features.columns for k in keys):
        raise SchemaError(f"policy input: identifier columns absent: {keys}")

    out = features.loc[:, keys + list(declared)].copy()

    # Redundant second line of defence. The primary test is set equality against the
    # whitelist; this catches a *reintroduced* leak whose name gives it away.
    for col in out.columns:
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in str(col):
                raise SchemaError(
                    f"policy input: column {col!r} matches forbidden substring {bad!r}"
                )

    if out.isna().any().any():
        # NaN is legitimate ONLY for cv2_init on `dead` SKUs, which the context
        # builder replaces with an explicit indicator before this point.
        bad = sorted(out.columns[out.isna().any()].tolist())
        raise SchemaError(f"policy input: NaN in features {bad}")
    return out
