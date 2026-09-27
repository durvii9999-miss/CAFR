"""Shared fixtures.

The data path resolves to ``<repo>/data`` and the RUF panels are committed, so the
suite runs on a fresh clone with no download step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cafr.utils.config import load_config, load_features  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config("base.yaml")


@pytest.fixture(scope="session")
def features_cfg() -> dict:
    return load_features()


@pytest.fixture(scope="session")
def whitelist() -> tuple[str, ...]:
    from cafr.data.schema import whitelist as _w

    return _w()


def repo_path(*parts: str) -> Path:
    """Absolute path under the repo root.

    Tests must not depend on the working directory: ``pytest`` is run from the repo
    root, but an IDE or ``pytest tests/test_x.py`` need not be.
    """
    return REPO_ROOT.joinpath(*parts)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return repo_path("data")


def make_feature_frame(columns, n: int = 12, seed: int = 0):
    """Build a frame carrying ``sku_id``, ``period`` and exactly ``columns``.

    Used by the whitelist test. The producer is a fixture until the monitor lands at
    Step 5; the CONTRACT it tests (``cafr/configs/features.yaml``) is already the real
    one, and Step 5 swaps the fixture for the monitor's output without touching the
    assertions.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    data = {
        "sku_id": [f"SYN5Y_{i:04d}" for i in range(n)],
        "period": np.arange(n, dtype="int32"),
    }
    for c in columns:
        if c.startswith(("cell_", "cause_", "active_remedy_")):
            data[c] = rng.integers(0, 2, size=n).astype("float64")
        else:
            data[c] = rng.random(n)
    return pd.DataFrame(data)
