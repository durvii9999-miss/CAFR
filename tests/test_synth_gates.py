"""Gates 4.1 and 4.2 -- the panel's own properties (Rev 2 §15.4, §30).

**Gate 4.2 changed at Step 4, and the reason matters.** It used to assert that exactly
four cells appear, and it passed -- because the labeller wrote ``sb_cell_init`` as the
cell name it had SAMPLED, not the cell computed from the burn-in window. A ``dead`` SKU
(a SKU with fewer than two nonzero periods in the burn-in, whose CV^2 is undefined)
could therefore never appear in the synthetic panel, while 39 of RUF's 5,000 SKUs are
``dead``. The test was passing on a defect.

``sb_cell_init`` is now computed from the burn-in window, exactly as
``data/build_ruf_panel.py`` computes it for RUF, so the two panels are classified by the
same rule. Three consequences, all asserted below:

* a ``dead`` cell appears, and its share is bounded -- a panel that is mostly dead SKUs
  cannot measure attribution at all, because a dead SKU never reaches any detector's
  evidence floor and abstains by construction;
* the four reachable cells are populated at or above the floor **among the SKUs that
  have a cell at all**;
* the written cell is genuinely computed, not echoed from the sampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cafr.data.synth.generator import sb_cell
from cafr.data.synth.labels import generate_labelled_panel, generate_panel

#: The four reachable ADI-band x CV2-band cells of §15.1. ``dead`` is not one of them:
#: it is the absence of a cell.
REACHABLE = ("moderate_lowdisp", "moderate_highdisp", "high_lowdisp", "high_highdisp")

CELL_FLOOR = 0.15

#: A declared cap on the ``dead`` share, not a tuned one. RUF's own panel is 0.78 %.
#: The synthetic panel is higher because its generator reaches p = 0.05 (ADI = 20, which
#: §15.1 wants) while the burn-in is only 24 periods, so a p = 0.05 SKU has a ~66 %
#: chance of fewer than two nonzero periods. That is a property of the panel to REPORT,
#: not to hide -- and at 12 % it still leaves the four reachable cells dominant.
DEAD_CAP = 0.12


def test_gate_4_1_deterministic():
    """Gate 4.1: regenerating from the same seed reproduces the panel exactly."""
    df1 = generate_panel(42, n_series=180)
    df2 = generate_panel(42, n_series=180)
    pd.testing.assert_frame_equal(df1, df2)


def test_gate_4_1b_series_are_order_independent():
    """§15.6: a series must not depend on how many series were generated before it.

    Each series draws from ``SeedSequence(seed, spawn_key=(panel_id, index))``, so
    adding the C2b panel must not move the single-cause panel's numbers.
    """
    a_demand, _, _ = generate_labelled_panel(42, n_per_cause=5, n_control=5, n_mixture=5, n_c2b=0)
    b_demand, _, _ = generate_labelled_panel(42, n_per_cause=5, n_control=5, n_mixture=5, n_c2b=7)
    shared = a_demand["sku_id"].unique()
    a = a_demand.set_index(["sku_id", "period"])["demand_true"]
    b = b_demand.set_index(["sku_id", "period"])["demand_true"]
    pd.testing.assert_series_equal(a, b.loc[a.index])
    assert len(shared) == 45


def test_gate_4_2_class_counts():
    """Gate 4.2: the panel spans the reachable scheme, and the cell is COMPUTED."""
    df = generate_panel(42, n_series=1800)
    first = df[df["period"] == 0]
    cells = first["sb_cell_init"].value_counts(normalize=True)

    print("\ngate 4.2 -- sb_cell_init shares")
    for cell, frac in cells.items():
        print(f"  {cell:<20} {frac:.4f}")

    # (i) every reachable cell is present, at or above the floor, among SKUs WITH a cell
    live = first[first["sb_cell_init"] != "dead"]
    assert len(live) > 0
    live_shares = live["sb_cell_init"].value_counts(normalize=True)
    for cell in REACHABLE:
        assert cell in live_shares.index, f"reachable cell {cell} never populated"
        assert live_shares[cell] >= CELL_FLOOR, (
            f"cell {cell} holds only {live_shares[cell]:.3f} of the SKUs with a cell"
        )

    # (ii) the dead share is bounded and reported
    dead_share = float(cells.get("dead", 0.0))
    print(f"  {'dead':<20} {dead_share:.4f}   (cap {DEAD_CAP})")
    assert dead_share <= DEAD_CAP, (
        f"{dead_share:.3f} of the panel is dead; attribution is unmeasurable there"
    )
    assert set(cells.index) <= set(REACHABLE) | {"dead"}, (
        "an unknown cell appeared; SB_CELLS and the labeller have diverged"
    )


def test_gate_4_2b_cell_is_computed_not_echoed():
    """``sb_cell_init`` must be derived from the burn-in window, not the sampler's pick.

    This is the assertion that would have caught the original defect: the labeller used
    to write the sampled cell straight into ``sb_cell_init``, which is why no ``dead``
    SKU could appear and why gate 4.2 passed.
    """
    _, _, params = generate_labelled_panel(
        42, n_per_cause=40, n_control=40, n_mixture=0, n_c2b=0
    )
    mismatches = int((params["sb_cell_init"] != params["sb_cell_sampled"]).sum())

    print(f"\ngate 4.2b -- computed cell differs from sampled cell for "
          f"{mismatches}/{len(params)} SKUs")
    assert mismatches > 0, (
        "sb_cell_init equals the sampled cell for every SKU: the cell is being echoed "
        "rather than computed from the burn-in window"
    )

    # And it agrees with an independent recomputation from the recorded burn-in stats.
    recomputed = [
        sb_cell(a, c) for a, c in zip(params["adi_init"], params["cv2_init"], strict=True)
    ]
    assert list(params["sb_cell_init"]) == recomputed


def test_gate_4_2c_no_dead_sku_has_a_cell():
    """A ``dead`` SKU has an undefined CV^2; it must never be filed into a dispersion band."""
    _, _, params = generate_labelled_panel(
        42, n_per_cause=20, n_control=20, n_mixture=0, n_c2b=0
    )
    dead = params[params["sb_cell_init"] == "dead"]
    assert np.isnan(dead["cv2_init"]).all(), (
        "a dead SKU carries a numeric cv2_init; writing 0.0 there files it into a "
        "_lowdisp cell it does not belong to (handoff §4.3)"
    )
