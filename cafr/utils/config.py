"""Config loading. Every threshold in this project lives in a YAML file (handoff §12
design rule 4: no magic numbers in code), so this module is the only place that reads them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "load_features", "config_hash", "CONFIG_DIR", "REPO_ROOT"]

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
REPO_ROOT = CONFIG_DIR.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """Load ``configs/base.yaml`` and merge the named experiment config on top.

    ``path`` may be an absolute path, or a name relative to ``cafr/configs``.
    A small config file therefore only states what it changes from the base.
    """
    path = Path(path)
    if not path.exists():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = cfg.pop("inherits", None)
    if base_name:
        cfg = _deep_merge(load_config(base_name), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    _check_sim_geometry(cfg)
    return cfg


def _check_sim_geometry(cfg: dict) -> None:
    """``R = L + 1`` is a definition, not a tuning knob (Rev 2 §15.6). Assert it."""
    sim = cfg.get("sim")
    if not sim:
        return
    lead, protection = sim.get("L"), sim.get("R")
    if lead is not None and protection is not None and protection != lead + 1:
        raise ValueError(
            f"sim.R must equal sim.L + 1 (Rev 2 §15.6). Got L={lead}, R={protection}."
        )


def load_features(path: str | Path | None = None) -> dict[str, Any]:
    """Load the ``policy_features`` whitelist (Rev 2 §27). Never hard-code it elsewhere."""
    path = Path(path) if path else CONFIG_DIR / "features.yaml"
    if not path.exists():
        path = CONFIG_DIR / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_hash(cfg: dict) -> str:
    """Stable hash of a resolved config, for T8 (reproducibility/runtime)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
