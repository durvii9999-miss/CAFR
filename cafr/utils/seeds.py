"""Derived seeding.

Rev 2 §15.6 / handoff §7.1: every stochastic element is seeded from
``(panel_seed, sku_id, period)`` -- a **derived** seed, not a running RNG stream.

Why this matters (handoff §7.9 condition 2): the oracle replays the simulator by
forking it from a state. A running RNG stream would give different future draws in
each fork, so the arms would not face identical demand and the regret number would be
meaningless. A derived seed makes the draw at ``(sku, t)`` a pure function of its
coordinates, so every fork sees the same future.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["derive_seed", "rng_for", "SeedSequenceError"]

_UINT32 = 2**32


class SeedSequenceError(ValueError):
    """Raised when a seed cannot be derived from the given coordinates."""


def derive_seed(root_seed: int, *parts: object) -> int:
    """Return a stable uint32 seed derived from ``root_seed`` and ``parts``.

    Stable across processes and platforms: SHA-256 over the string form, not
    Python's ``hash()`` (which is salted per process by default).
    """
    if root_seed is None:
        raise SeedSequenceError("root_seed must not be None")
    key = "|".join([str(int(root_seed)), *(str(p) for p in parts)])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _UINT32


def rng_for(root_seed: int, *parts: object) -> np.random.Generator:
    """Return a fresh ``Generator`` for the coordinate ``(root_seed, *parts)``.

    Never returns the same generator twice, so callers cannot accidentally share a
    running stream between coordinates.
    """
    return np.random.default_rng(derive_seed(root_seed, *parts))
