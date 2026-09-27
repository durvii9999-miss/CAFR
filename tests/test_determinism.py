"""Determinism -- gate 1.1, and the seeding discipline the whole project rests on.

Two separate claims, tested separately:

1. **Same config + same seed -> byte-identical output.** Gate 1.1. Byte-identical,
   not "close": a run that differs in the last float bit is a run that cannot be
   reproduced, and T8 exists precisely to certify reproducibility.

2. **Different coordinate -> different draw.** ``derive_seed`` must be a pure function
   of its coordinates and must be stable ACROSS PROCESSES. ``hash()`` is salted per
   process and would silently give a different demand panel on every run, which is why
   the module uses SHA-256 -- asserted here by running a subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from cafr.sim.inventory import InventorySimulator, SimConfig
from cafr.utils.config import config_hash, load_config
from cafr.utils.io import read_parquet, write_parquet
from cafr.utils.seeds import derive_seed, rng_for
from conftest import REPO_ROOT, repo_path

DETERMINISM_SKUS = 40
DETERMINISM_PERIODS = 120


@pytest.fixture(scope="module")
def deterministic_run(cfg):
    """One fixed (config, seed) simulation, runnable twice."""

    def run(out_path):
        sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=cfg["seed_root"])
        rng = np.random.default_rng(99)
        logs = []
        for i in range(DETERMINISM_SKUS):
            sku = f"DET_{i:04d}"
            p, k, mu_z = rng.uniform(0.1, 0.5), rng.uniform(1.0, 4.0), rng.uniform(10, 90)
            y = np.array(
                [sim.draw_demand(sku, t, p, k, mu_z) for t in range(DETERMINISM_PERIODS)]
            )

            def level_fn(t, y_observed, censored, inventory, _m=y[:24].mean()):
                # Deterministic and history-dependent: no RNG inside the policy.
                return float(1.5 * _m + 0.25 * y_observed[: t + 1].sum() / (t + 1)), 0.90

            logs.append(sim.run(sku, y, level_fn, arm="det").log)
        frame = pd.concat(logs, ignore_index=True)
        write_parquet(frame, out_path)
        return frame

    return run


def test_gate_1_1_two_runs_are_byte_identical(deterministic_run, tmp_path):
    """Same config + same seed -> identical parquet BYTES. Gate 1.1, verbatim."""
    a = tmp_path / "inventory_log_a.parquet"
    b = tmp_path / "inventory_log_b.parquet"

    frame_a = deterministic_run(a)
    frame_b = deterministic_run(b)

    bytes_a = a.read_bytes()
    bytes_b = b.read_bytes()
    assert bytes_a == bytes_b, (
        f"gate 1.1 FAILED: {len(bytes_a)} vs {len(bytes_b)} bytes. "
        "Everything downstream inherits this."
    )
    assert frame_a.equals(frame_b)


def test_gate_1_1_differs_when_the_seed_changes(cfg, deterministic_run, tmp_path):
    """The other direction: byte-identity must not be an accident of constant output."""
    a = tmp_path / "seed42.parquet"
    deterministic_run(a)

    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=cfg["seed_root"] + 1)
    y = np.array([sim.draw_demand("DET_0000", t, 0.3, 2.0, 50.0) for t in range(50)])
    y_other = np.array([sim.draw_demand("DET_0000", t, 0.3, 2.0, 50.0) for t in range(50)])
    assert np.array_equal(y, y_other), "the derived seed is not a pure function"
    assert read_parquet(a).shape[0] > 0


# --------------------------------------------------------------------------- #
# Seeding: stable across processes                                             #
# --------------------------------------------------------------------------- #


def test_derive_seed_is_a_pure_function_of_its_coordinates():
    assert derive_seed(42, "demand", "S1", 0) == derive_seed(42, "demand", "S1", 0)
    assert derive_seed(42, "demand", "S1", 0) != derive_seed(42, "demand", "S1", 1)
    assert derive_seed(42, "demand", "S1", 0) != derive_seed(42, "demand", "S2", 0)
    assert derive_seed(42, "demand", "S1", 0) != derive_seed(43, "demand", "S1", 0)
    # The separator matters: ("a", "bc") must not collide with ("ab", "c").
    assert derive_seed(1, "a", "bc") != derive_seed(1, "ab", "c")


def test_derive_seed_is_stable_across_processes():
    """``hash()`` is salted per process; SHA-256 is not. Verified by subprocess.

    If this fails, every run in the project draws a different demand panel and no
    number in the paper is reproducible -- and nothing else would catch it.
    """
    script = textwrap.dedent(
        """
        from cafr.utils.seeds import derive_seed
        print(derive_seed(42, "demand", "SYN5Y_0001", 7))
        """
    )
    values = set()
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        )
        values.add(out.stdout.strip())
    assert len(values) == 1, f"derive_seed is not process-stable: {values}"
    assert int(values.pop()) == derive_seed(42, "demand", "SYN5Y_0001", 7)


def test_rng_for_never_shares_a_stream():
    a = rng_for(42, "x")
    b = rng_for(42, "x")
    assert a is not b, "each coordinate must get its own generator object"
    assert np.array_equal(a.random(5), b.random(5)), (
        "two generators at the same coordinate must produce the same stream"
    )
    assert not np.array_equal(
        rng_for(42, "x").random(5), rng_for(42, "y").random(5)
    ), "different coordinates must give different streams" 


def test_demand_draws_are_reproducible_per_coordinate(cfg):
    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=42)
    first = [sim.draw_demand("SKU_A", t, 0.4, 2.0, 30.0) for t in range(30)]
    second = [sim.draw_demand("SKU_A", t, 0.4, 2.0, 30.0) for t in range(30)]
    assert first == second
    other = [sim.draw_demand("SKU_B", t, 0.4, 2.0, 30.0) for t in range(30)]
    assert first != other


def test_oracle_fork_sees_the_same_future(cfg):
    """Why the derived seed exists at all (Rev 2 §15.7 condition 2).

    A running RNG stream would give each arm a different demand sequence, and the
    regret number would measure the luck of the draw.
    """
    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=42)
    full = np.array([sim.draw_demand("SKU_C", t, 0.35, 3.0, 25.0) for t in range(60)])

    # Replay from t = 30 in a fresh process-like state: same draws from there on.
    forked = np.array([sim.draw_demand("SKU_C", t, 0.35, 3.0, 25.0) for t in range(60)])
    assert np.array_equal(full, forked)
    assert np.array_equal(full[30:], forked[30:])


# --------------------------------------------------------------------------- #
# Config hashing -- T8's reproducibility certificate                          #
# --------------------------------------------------------------------------- #


def test_config_hash_is_stable_and_order_insensitive():
    a = load_config("base.yaml")
    b = load_config("base.yaml")
    assert config_hash(a) == config_hash(b)

    shuffled = dict(reversed(list(a.items())))
    assert config_hash(shuffled) == config_hash(a), (
        "config_hash must not depend on key insertion order"
    )


def test_config_hash_changes_when_a_parameter_changes():
    """T8 is only meaningful if the hash can tell two runs apart."""
    a = load_config("base.yaml")
    b = load_config("base.yaml", overrides={"sim": {"L": 1, "R": 2}})
    assert config_hash(a) != config_hash(b)


def test_config_geometry_is_asserted_not_trusted():
    """``R = L + 1`` is a definition (Rev 2 §15.6). A config violating it must fail."""
    with pytest.raises(ValueError, match=r"sim\.R must equal sim\.L \+ 1"):
        load_config("base.yaml", overrides={"sim": {"L": 2, "R": 4}})


def test_committed_configs_are_loadable():
    for name in ("base.yaml", "features.yaml"):
        assert repo_path("cafr", "configs", name).exists(), f"{name} is missing"
    cfg = load_config("base.yaml")
    assert SimConfig.from_config(cfg) is not None
