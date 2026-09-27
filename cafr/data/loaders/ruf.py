"""RUF v2 loader.

The separation between ``load_observed`` and ``load_ground_truth`` is enforced by the
loader's SIGNATURE, not by a runtime check (Rev 2 §14.3 rule 7). ``load_observed``
has no code path that can reach ``demand_true`` or ``stockout_flag``; it returns a
frame with those columns dropped, plus the whitelisted static features.

RUF is a DEMAND panel, not an inventory transaction log. Censoring is present but
UNOBSERVED -- ``stockout_flag`` is False everywhere, which is the honest encoding
(handoff §5.4). Do not invent stock-outs in it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.config import REPO_ROOT, load_config
from ...utils.io import read_parquet
from ..schema import (
    DEMAND_PANEL_COLUMNS,
    SKU_META_COLUMNS,
    SB_CELLS,
    validate_frame,
)

__all__ = ["load_observed", "load_ground_truth", "load_sku_meta", "load_attributes", "ruf_paths"]


def ruf_paths(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or load_config("base.yaml")
    entry = cfg["panels"]["ruf"]
    return {
        "demand": REPO_ROOT / entry["path"],
        "meta": REPO_ROOT / entry["meta"],
        "attributes": REPO_ROOT / entry["attributes"],
        "profile": REPO_ROOT / entry["profile"],
    }


def load_sku_meta(cfg: dict | None = None) -> pd.DataFrame:
    """Static per-SKU attributes. Non-leaking: burn-in only.

    ``adi_init`` / ``cv2_init`` / ``sb_cell_init`` are computed on the first
    ``burn_in`` periods ONLY. Asserted in tests/test_schema.py.
    """
    paths = ruf_paths(cfg)
    meta = read_parquet(paths["meta"])
    validate_frame(meta, SKU_META_COLUMNS, name="sku_meta.parquet")
    bad = set(meta["sb_cell_init"].unique()) - set(SB_CELLS)
    if bad:
        raise ValueError(f"sku_meta.sb_cell_init has unknown cells: {sorted(bad)}")
    # cv2_init is NaN for `dead` SKUs (fewer than 2 nonzero periods in burn-in).
    # That is correct, not a bug: writing 0.0 would file the SKU into a _lowdisp
    # cell it does not belong to (handoff §4.3).
    return meta


def load_observed(
    cfg: dict | None = None,
    *,
    panel: str = "ruf",
    skus: list[str] | None = None,
) -> pd.DataFrame:
    """Return the OBSERVED demand frame plus whitelisted static features -- and nothing else.

    Deliberately has no parameter that could request ``demand_true``. The policy and
    the monitor call this function; the evaluator calls ``load_ground_truth``.
    """
    paths = ruf_paths(cfg)
    df = read_parquet(paths["demand"])
    validate_frame(df, DEMAND_PANEL_COLUMNS, name="demand_panel.parquet")

    if panel and "panel" in df.columns:
        df = df[df["panel"] == panel]

    # Narrow the view HERE. After this line the true series is unreachable.
    observed = df.loc[:, ["sku_id", "period", "demand_observed"]].copy()
    observed["demand_observed"] = observed["demand_observed"].astype("float64")

    meta = load_sku_meta(cfg)
    observed = observed.merge(meta, on="sku_id", how="left", validate="many_to_one")
    if skus is not None:
        observed = observed[observed["sku_id"].isin(skus)]
    return observed.sort_values(["sku_id", "period"], ignore_index=True)


def load_ground_truth(cfg: dict | None = None, *, panel: str = "ruf") -> pd.DataFrame:
    """The evaluator's view: observed AND true series plus the stock-out flag.

    Opened ONLY by ``cafr.eval``. Never reachable from the policy or the monitor.
    For RUF, ``true_cause`` does not exist -- RUF is unlabelled -- so this returns the
    demand side only. The synthetic panel's labelled variant lives in ``data/synth``.
    """
    paths = ruf_paths(cfg)
    df = read_parquet(paths["demand"])
    validate_frame(df, DEMAND_PANEL_COLUMNS, name="demand_panel.parquet")
    if panel and "panel" in df.columns:
        df = df[df["panel"] == panel]
    return df.sort_values(["sku_id", "period"], ignore_index=True)


def load_attributes(cfg: dict | None = None) -> pd.DataFrame:
    """RUF's real planning parameters.

    EVALUATION AND REPORTING ONLY. ``adi_full`` / ``cv_squared_full`` are computed over
    all 60 periods; if they reach the feature vector the detector is handed the
    post-regime answer. That is worse than a look-ahead -- it is a direct
    ground-truth leak (handoff §6.1).
    """
    paths = ruf_paths(cfg)
    df = read_parquet(paths["attributes"])
    validate_frame(df, SKU_ATTRIBUTES_COLUMNS, name="sku_attributes.parquet")
    return df


def panel_length(cfg: dict | None = None) -> int:
    df = read_parquet(ruf_paths(cfg)["demand"], columns=["period"])
    return int(df["period"].max()) + 1
