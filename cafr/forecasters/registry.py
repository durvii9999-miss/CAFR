"""The forecaster pool. One lookup table, so no arm builds its own pool.

``build_pool(cfg)`` returns ``{name: Forecaster}`` for the names listed in
``cfg['forecast']['pool']``. Every arm receives the same dict -- that is what makes
"same forecaster pool" in the fairness table (handoff §8.2) true by construction
rather than by assertion.
"""

from __future__ import annotations

from typing import Any

from .base import Forecaster
from .croston import Croston
from .lightgbm_global import GlobalLightGBM
from .msba import mSBA
from .mtsb import mTSB
from .sba import SBA
from .ses import SESOnSizes
from .tsb import TSB

__all__ = ["build_pool", "POOL_NAMES", "make_forecaster"]

POOL_NAMES = ("croston", "sba", "tsb", "msba", "mtsb", "ses_sizes", "lightgbm_global")


def make_forecaster(name: str, cfg: dict[str, Any] | None = None) -> Forecaster:
    """Construct one forecaster by name, using the smoothing constants from config."""
    f = (cfg or {}).get("forecast", {})
    lgb_params = f.get("lightgbm", {})
    seed = int((cfg or {}).get("seed_root", 42))

    builders = {
        "croston": lambda: Croston(alpha=f.get("croston_alpha", 0.1)),
        "sba": lambda: SBA(alpha=f.get("croston_alpha", 0.1)),
        "tsb": lambda: TSB(
            alpha_d=f.get("tsb_alpha_d", 0.2), alpha_p=f.get("tsb_alpha_p", 0.2)
        ),
        "msba": lambda: mSBA(
            alpha_z=f.get("msba_alpha", 0.1), alpha_p=f.get("msba_alpha", 0.1)
        ),
        "mtsb": lambda: mTSB(
            alpha_d=f.get("mtsb_alpha_d", 0.2), alpha_p=f.get("mtsb_alpha_p", 0.2)
        ),
        "ses_sizes": lambda: SESOnSizes(alpha=f.get("ses_alpha", 0.1)),
        "lightgbm_global": lambda: GlobalLightGBM(params=lgb_params, seed=seed),
    }
    if name not in builders:
        raise KeyError(f"unknown forecaster {name!r}; known: {sorted(builders)}")
    return builders[name]()


def build_pool(cfg: dict[str, Any]) -> dict[str, Forecaster]:
    """Fresh instances for every pool member named in the config."""
    names = cfg.get("forecast", {}).get("pool", list(POOL_NAMES))
    return {name: make_forecaster(name, cfg) for name in names}
