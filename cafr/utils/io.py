"""Small I/O helpers. Parquet only for data; JSON for reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["read_parquet", "write_parquet", "write_json", "read_json", "ensure_dir"]


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, indent=indent, default=_json_default, sort_keys=False),
        encoding="utf-8",
    )
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if value == value else None       # NaN -> null, not invalid JSON
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
