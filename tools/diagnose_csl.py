"""One-off diagnostic: WHY does the gate 1.2 oracle over-serve?

Throwaway measurement, run once. The answer gets encoded as a test in
tests/test_simulator_sanity.py and this file is then deleted.

Question: the oracle sets S to the EXACT alpha-quantile of the R-period aggregate,
so P(A_R > S) = 1 - alpha by construction. Yet the measured per-period "achieved
CSL" runs well above alpha. Either

  (a) the measure is a per-PERIOD fraction while alpha is a per-CYCLE target, and
      intermittent demand makes those differ; or
  (b) the simulator is handing out service the level does not justify.

Distinguish by computing both, plus the fraction of periods with zero demand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cafr.sim.inventory import InventorySimulator, SimConfig  # noqa: E402
from cafr.sim.oracle_dgp import BernoulliGammaDGP, oracle_level_function  # noqa: E402
from cafr.utils.config import load_config  # noqa: E402

cfg = load_config("base.yaml")
sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=int(cfg["seed_root"]))

rng = np.random.default_rng(20260923)
lo, hi = cfg["synthetic"]["p_range"]
N, T = 60, 400
p = rng.uniform(lo, hi, N)
k = rng.uniform(0.5, 4.0, N)
mu_z = np.exp(rng.uniform(np.log(5.0), np.log(200.0), N))

L = int(cfg["sim"]["L"])
print(f"L={L}  R={cfg['sim']['R']}  T={T}  n={N}")

for K in (2, 3):
    for alpha in (0.80, 0.90):
        per_period, per_cycle, zero_frac, cycles_hit = [], [], [], []
        for i in range(N):
            sku = f"DIAG_{i:04d}"
            y = np.array([sim.draw_demand(sku, t, p[i], k[i], mu_z[i]) for t in range(T)])
            dgp = BernoulliGammaDGP(p=float(p[i]), k=float(k[i]), mu_z=float(mu_z[i]))
            res = sim.run(sku, y, oracle_level_function(dgp, K, alpha), arm="diag")
            log = res.log
            # per-period, as the current gate measures it
            per_period.append(res.achieved_csl)
            zero_frac.append(float((y == 0).mean()))
            # per-cycle: non-overlapping blocks of K, starting after the warm-up
            # window (periods < L have no stock by construction -- the first order
            # cannot have arrived yet).
            start = L
            nblocks = (T - start) // K
            hits = 0
            for b in range(nblocks):
                a, bnd = start + b * K, start + (b + 1) * K
                if log["demand_lost"].to_numpy()[a:bnd].sum() <= 0.0:
                    hits += 1
            per_cycle.append(hits / nblocks if nblocks else np.nan)
            cycles_hit.append(nblocks)

        print(
            f"K={K} alpha={alpha:.2f}  "
            f"per-period CSL {np.mean(per_period):.4f} (se "
            f"{np.std(per_period, ddof=1)/np.sqrt(N):.4f})  "
            f"per-cycle CSL {np.nanmean(per_cycle):.4f}  "
            f"zero-demand periods {np.mean(zero_frac):.4f}"
        )
