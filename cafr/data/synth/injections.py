"""Demand generation and injection (Rev 2 §15.2).

One function per cause, all with the same contract::

    y, delta, z, params = inject_Cx(T, tau, p, k, mu_z, rng)

``y = delta * z`` is the demand series; ``params`` carries the injection's
parameters so they can be written into ``ground_truth.parquet`` and audited. Every
function draws the baseline occurrence and size processes from the SAME parameters
and only then perturbs the one object its cause owns.

The three-cause orthogonality problem of Revision 1
---------------------------------------------------
C2, C4 and C6 all changed the Gamma shape ``k`` at fixed ``mu_z``, so they were one
intervention under three names. Revision 2 gives each cause its own object, and this
module implements exactly that:

===== ====================================== ==========================
cause object manipulated                    demand DGP touched?
===== ====================================== ==========================
C1    the size MEAN (linear ramp in mu_z)   yes
C2    the FORECASTER's reported interval    **no** -- see note below
C3    the occurrence PROBABILITY (step)     yes
C4    the size DISPERSION (shape k)         yes
C5    an abrupt step in mu_z or p           yes
C6    a slow drift in p                     yes
C7    the policy's safety factor            **no**
===== ====================================== ==========================

C2 and C7 are **policy/forecaster** injections, not demand injections. ``inject_C2``
and ``inject_C7`` therefore return the untouched baseline series and put the action in
``params``: for C2 a per-SKU interval-rescale factor ``gamma``; for C7 the safety
factor and the ``fit_on`` override. This is what makes the labels unambiguous (§15.5)
and is why C7 can fire at all -- if the injection also perturbed demand, the
forecast-acceptability gate would fail and C7 could never be attributed (§15.2).

Constraint traceability (Rev 2 §15.2). Each constraint is enforced here or flagged,
never relaxed:

1. C4 band invariance      -- ``_same_cv2_band``; ``k`` is RESAMPLED when the
                              sampled ``k`` cannot satisfy it.
2. C4 fixed mean           -- ``mu_z`` is carried through untouched.
3. C5 detectability budget -- ``check_c5_budget``, with the step scaled up to clear.
4. C6 learnability guard   -- ``check_c6_learnability``, checked in ``labels.py``.
5. C2 coverage gap         -- ``gamma`` is solved per SKU so the induced coverage
                              gap clears ``tau_cov`` with margin.
6/7. C7 acceptability+exposure -- simulator constraints; see ``c7_constraints.py``.
8. C1 occurrence untouched -- ``p`` is carried through untouched.
9. C4 detectability budget -- nonzero-size count; flagged in ``labels.py``.
"""

from __future__ import annotations

import numpy as np

from .budget import check_c5_budget

__all__ = [
    "inject_C0",
    "inject_C1",
    "inject_C2",
    "inject_C2b",
    "inject_C3",
    "inject_C4",
    "inject_C5",
    "inject_C6",
    "inject_C7",
    "inject_mixture",
    "INJECTORS",
    "CV2_STAR",
]

CV2_STAR = 0.49
"""The SB dispersion cut-off (§15.1). ``CV^2 = 1/k`` for a Gamma size distribution."""

#: Interval-rescale factors are solved, not drawn, so the bounds only need to be wide
#: enough to bracket the target coverage.
_GAMMA_BOUNDS = (0.02, 8.0)

_MC_AGG_SAMPLES = 20000
"""Monte-Carlo size for the R-period aggregate quantile used to solve C2's ``gamma``."""

#: C2b's calibration-window ladder, longest first. The longest window whose induced
#: coverage clears ``tau_cov + margin`` is chosen, so the modelling error stays as small
#: as constraint 5 allows.
_C2B_WINDOW_LADDER = (30, 24, 18, 12, 8, 6, 4)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _baseline_draw(
    T: int, p: float, k: float, mu_z: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """The baseline occurrence and size processes, before any injection."""
    delta = rng.binomial(1, p, size=T).astype("float64")
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    return delta, z


def _cv2(k: float) -> float:
    return 1.0 / k


def _same_cv2_band(k_a: float, k_b: float) -> bool:
    """Constraint 1: both shapes must lie on the same side of ``CV2_STAR``."""
    return (_cv2(k_a) - CV2_STAR) * (_cv2(k_b) - CV2_STAR) > 0


def _band_of(k: float) -> tuple[float, float]:
    """The ``k`` range of the CV2 band ``k`` sits in, as sampled by ``get_cell_params``."""
    if _cv2(k) < CV2_STAR:
        return (2.05, 10.0)          # lowdisp: CV2 < 0.49
    return (0.5, 2.04)               # highdisp: CV2 >= 0.49


def _feasible_k_range(k: float, factor: float) -> tuple[float, float] | None:
    """The sub-range of ``k``'s band for which ``k * factor`` stays in the SAME band.

    Writing the bands as ``k > 1/CV2*`` (lowdisp) and ``k <= 1/CV2*`` (highdisp), the
    scaled shape stays in band exactly when ``k * factor`` lies on the same side of
    ``1/CV2*`` as ``k`` does. For a factor that scales UP in the highdisp band this
    bounds ``k`` from above at ``1/(CV2* * factor)``; scaling DOWN in the lowdisp band
    bounds it from below at the same value::

        factor > 1, highdisp :  k <= 1/(0.49 * factor)   e.g. 0.5102 at factor 4.0
        factor < 1, lowdisp  :  k >  1/(0.49 * factor)   e.g. 8.1633 at factor 0.25

    Both are the same line, and the complementary (band, factor) pairings are feasible
    for every ``k`` in the band.
    """
    lo, hi = _band_of(k)
    boundary = 1.0 / (CV2_STAR * factor)
    lowdisp = _cv2(k) < CV2_STAR
    if factor > 1.0 and not lowdisp:
        hi = min(hi, boundary)
    elif factor < 1.0 and lowdisp:
        lo = max(lo, boundary)
    return (lo, hi) if hi > lo else None


def _resample_k_same_band(k: float, factor: float, rng: np.random.Generator) -> float:
    """Draw a new ``k`` from the SAME band that can absorb ``factor`` (constraint 1)."""
    rng_range = _feasible_k_range(k, factor)
    if rng_range is None:  # pragma: no cover - unreachable for {0.25, 4.0}
        raise RuntimeError(f"no k in band {_band_of(k)} absorbs factor {factor}")
    lo, hi = rng_range
    return float(rng.uniform(lo, hi))


def _aggregate_samples(
    p: float, k: float, mu_z: float, R: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Monte-Carlo draws of the R-period aggregate demand ``A^{agg}`` (§6.1)."""
    delta = rng.binomial(1, p, size=(n, R)).astype("float64")
    z = rng.gamma(shape=k, scale=mu_z / k, size=(n, R))
    return (delta * z).sum(axis=1)


def _induced_coverage(a: np.ndarray, alpha: float, gamma: float) -> float:
    """Coverage of the interval ``mu_hat + gamma * (Q_alpha - mu_hat)``.

    ``gamma = 1`` reproduces the nominal coverage ``alpha``. ``gamma < 1`` shrinks the
    interval (under-coverage), ``gamma > 1`` inflates it (over-coverage).
    """
    mu = float(a.mean())
    q = float(np.quantile(a, alpha, method="linear"))
    return float(np.mean(a <= mu + gamma * (q - mu)))


def _solve_gamma(
    a: np.ndarray,
    alpha: float,
    target_gap: float,
) -> tuple[float, float, str]:
    """Solve C2's ``gamma`` so the induced coverage gap clears ``target_gap``.

    Coverage is monotone increasing in ``gamma``, so a bisection on the exact
    Monte-Carlo coverage surface is exact rather than approximate. Over-coverage may
    be infeasible (coverage is capped at 1, and ``alpha + target_gap > 1`` for
    ``alpha = 0.95`` and ``target_gap = 0.12``); in that case the under-coverage
    branch is used. Returns ``(gamma, achieved_coverage, direction)``.
    """
    lo, hi = _GAMMA_BOUNDS
    over_target = min(alpha + target_gap, 1.0)
    over_feasible = alpha + target_gap <= 1.0 and _induced_coverage(a, alpha, hi) >= over_target
    under_target = alpha - target_gap
    under_feasible = _induced_coverage(a, alpha, lo) <= under_target

    if over_feasible:
        target, direction = over_target, "over"
    elif under_feasible:
        target, direction = under_target, "under"
    else:
        # Neither extreme reaches the target: return the extreme that moves furthest.
        c_lo, c_hi = _induced_coverage(a, alpha, lo), _induced_coverage(a, alpha, hi)
        if abs(c_lo - alpha) >= abs(c_hi - alpha):
            return lo, c_lo, "under"
        return hi, c_hi, "over"

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _induced_coverage(a, alpha, mid) < target:
            lo = mid
        else:
            hi = mid

    # Return the endpoint on the SATISFYING side, and which side that is depends on the
    # branch -- the bisection maintains ``coverage(lo) < target <= coverage(hi)``, so
    # ``hi`` is the side for OVER-coverage (we want coverage >= alpha + gap) and ``lo``
    # is the side for UNDER-coverage (we want coverage <= alpha - gap).
    #
    # Returning a fixed endpoint, or the midpoint, puts the achieved gap on the wrong
    # side of the threshold for whichever branch it does not suit: with the midpoint,
    # C2's gap landed at exactly -0.120 for 200 of 200 SKUs and ``>= 0.12`` was False
    # for every one of them.
    if direction == "over":
        return hi, _induced_coverage(a, alpha, hi), direction
    return lo, _induced_coverage(a, alpha, lo), direction


# --------------------------------------------------------------------------- #
# C0 -- the control
# --------------------------------------------------------------------------- #


def inject_C0(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C0: no injection. The control, and the source of every 'must not move' bar."""
    delta, z = _baseline_draw(T, p, k, mu_z, rng)
    return delta * z, delta, z, {"demand_touched": False}


# --------------------------------------------------------------------------- #
# C1 -- persistent bias
# --------------------------------------------------------------------------- #


def inject_C1(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C1: persistent size bias. Linear ramp in ``mu_z`` over >= 20 periods.

    ``p`` is untouched (constraint 8): C1's statistic is computed on nonzero periods
    only, so moving ``p`` would put C1's series onto C6's statistic. The shape ``k`` is
    also untouched, so the LOG-scale dispersion (C4's statistic) is exactly invariant.
    """
    phi = float(rng.choice([-0.4, -0.2, 0.2, 0.4]))

    delta = rng.binomial(1, p, size=T).astype("float64")
    t_idx = np.arange(T)
    mu_z_t = np.full(T, float(mu_z))
    ramp = (t_idx[tau:] - tau) / 40.0
    mu_z_t[tau:] = mu_z * (1.0 + phi * ramp)
    mu_z_t = np.maximum(mu_z_t, 0.1)

    z = rng.gamma(shape=k, scale=mu_z_t / k)
    return delta * z, delta, z, {
        "phi": phi,
        "mu_z_new": float(mu_z_t[-1]),
        "direction": "up" if phi > 0 else "down",
        # Constraint 8, recorded explicitly rather than left to inference: C1 moves the
        # SIZE distribution only. A nonzero p_shift here would put C1's series onto
        # C6's occurrence statistic; a nonzero k_shift would move C4's log dispersion.
        "p_shift": 0.0,
        "k_shift": 0.0,
        "demand_touched": True,
    }


# --------------------------------------------------------------------------- #
# C2 -- mis-calibration (the demand DGP is NOT touched)
# --------------------------------------------------------------------------- #


def _c2(
    T: int,
    p: float,
    k: float,
    mu_z: float,
    rng: np.random.Generator,
    *,
    alpha: float,
    R: int,
    tau_cov: float,
    margin: float,
) -> dict:
    """Shared C2 bookkeeping: solve the interval rule, keep the demand untouched."""
    # A private stream: solving gamma must not consume the series' own draw.
    mc = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    a = _aggregate_samples(p, k, mu_z, R, _MC_AGG_SAMPLES, mc)

    gamma, cov, direction = _solve_gamma(a, alpha, tau_cov + margin)
    gap = cov - alpha
    return {
        "c2_variant": "interval_rescale",
        "gamma": round(float(gamma), 6),
        "alpha_nominal": float(alpha),
        "coverage_achieved": round(float(cov), 6),
        "coverage_gap": round(float(gap), 6),
        "gap_direction": direction,
        "gap_clears_tau_cov": bool(abs(gap) >= tau_cov),
        "gap_clears_with_margin": bool(abs(gap) >= tau_cov + margin),
        "demand_touched": False,
    }


def inject_C2(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C2: mis-calibration by rescaling the FORECASTER's reported interval.

    ``Q~_alpha = mu_hat + gamma * (Q_alpha - mu_hat)`` with ``gamma`` solved per SKU so
    the induced coverage gap clears ``tau_cov`` with margin (constraint 5). The demand
    DGP is **not** touched -- ``y``, ``delta`` and ``z`` are the baseline draw.
    """
    delta, z = _baseline_draw(T, p, k, mu_z, rng)
    params = _c2(
        T, p, k, mu_z, rng,
        alpha=0.95, R=3, tau_cov=0.10, margin=0.02,
    )
    return delta * z, delta, z, params


def inject_C2b(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C2b: mis-calibration from a genuine modelling error, data untouched.

    §15.2 constraint 5 requires the C2 detector to be run on **two** panels: C2 itself
    (a direct interval injection) and C2b, which answers the objection that *"C2 is
    defined as the thing the C2 detector measures"* by mis-calibrating the interval
    through a modelling error instead of a dial.

    Two errors are combined, both named in §15.2's C2b row:

    1. **per-period sigma applied to the R-period aggregate.** Under independence the
       correct sd is ``sqrt(R) * sigma_period``, so the interval is too short by a known
       factor -- the same 42.3 % understatement the order-up-to rule avoids.
    2. **a half-length calibration window.** ``sigma_period`` is estimated from a short
       trailing window of the SKU's own demand history. For a zero-inflated series a
       short window under-estimates the sd, widening the error.

    **The window length is chosen per SKU**, as the LONGEST window on the ladder whose
    induced coverage clears ``tau_cov + margin``. This is required, not optional:
    constraint 5 applies to both panels, and the per-period-sigma error alone clears
    ``tau_cov`` only for SKUs whose ``p`` is large enough for the aggregate to be
    non-degenerate -- for small ``p`` the aggregate is zero most of the time, a Gaussian
    half-width covers those zeros, and coverage lands within ``tau_cov`` of alpha with no
    detectable gap at all. Taking the longest clearing window keeps the modelling error
    as small as the constraint allows. SKUs whose gap cannot be made to clear even at
    the shortest window are FLAGGED, not silently accepted.
    """
    delta, z = _baseline_draw(T, p, k, mu_z, rng)

    mc = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    a = _aggregate_samples(p, k, mu_z, 3, _MC_AGG_SAMPLES, mc)
    alpha = 0.95
    target = 0.10 + 0.02          # tau_cov + margin, per §15.2 constraint 5

    mu_hat = float(a.mean())
    y_period = delta * z

    best: dict | None = None
    largest: dict | None = None
    for w in _C2B_WINDOW_LADDER:
        if w > tau or w < 2:
            continue
        # The smallest sd among ALL contiguous w-length windows of the pre-injection
        # history. Nested prefixes (``y[:w]``) give the search no freedom -- every window
        # contains the first few periods, so if period 0 happens to be nonzero every
        # candidate inherits it and the ladder returns one interval, not seven. Scanning
        # placements lets the window land on a quiet stretch, which is the realistic
        # form of this modelling error: a forecaster that calibrated on a run of zero
        # demand reports a narrow interval and under-covers afterwards.
        sds = [
            float(y_period[s : s + w].std(ddof=1))
            for s in range(0, min(tau, y_period.size) - w + 1)
        ]
        if not sds:
            continue
        sigma_hat = min(sds)
        # ... applied to the R-period aggregate WITHOUT the sqrt(R) correction --
        # error (1).
        half = 1.6449 * sigma_hat
        cov = float(np.mean(a <= mu_hat + half))
        gap = abs(cov - alpha)
        entry = {
            "c2_variant": "calibration_window",
            "calib_window": int(w),
            "sigma_period_hat": round(sigma_hat, 6),
            "half_width": round(half, 6),
            "gamma": None,
            "alpha_nominal": alpha,
            "coverage_achieved": round(cov, 6),
            "coverage_gap": round(cov - alpha, 6),
            "gap_direction": "under" if cov < alpha else "over",
            "demand_touched": False,
        }
        if largest is None or gap > abs(largest["coverage_gap"]):
            largest = entry
        if gap >= target:
            best = entry           # the LONGEST window that clears wins
            break

    # No window on the ladder clears: fall back to the one with the LARGEST induced gap.
    # Coverage is monotone in the half-width, so the largest gap is the closest this
    # mechanism gets to the threshold -- it is not a relaxation of the target, and the
    # SKU is still FLAGGED below when it misses.
    if best is None:
        best = largest

    assert best is not None
    gap = abs(best["coverage_gap"])
    best["gap_clears_tau_cov"] = bool(gap >= 0.10)
    best["gap_clears_with_margin"] = bool(gap >= target)
    return delta * z, delta, z, best


# --------------------------------------------------------------------------- #
# C3 -- occurrence failure
# --------------------------------------------------------------------------- #


def inject_C3(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C3: occurrence failure. A step in ``p`` only; sizes untouched."""
    shift = float(rng.choice([-0.15, 0.15]))
    p_new = float(np.clip(p + shift, 0.03, 0.60))

    delta = rng.binomial(1, p, size=T).astype("float64")
    delta[tau:] = rng.binomial(1, p_new, size=T - tau)
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    return delta * z, delta, z, {
        "p_new": p_new,
        "shift": p_new - p,
        "direction": "up" if p_new > p else "down",
        "demand_touched": True,
    }


# --------------------------------------------------------------------------- #
# C4 -- size failure
# --------------------------------------------------------------------------- #


def inject_C4(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C4: size failure. Dispersion change at FIXED mean.

    ``k -> k * factor`` with ``mu_z`` untouched (constraint 2). The factor is drawn
    first and ``k`` is then RESAMPLED until both ``k`` and ``k * factor`` lie in the
    same ADI x CV2 cell (constraint 1) -- the constraint is never relaxed, and the
    resampling is bounded so a failure is loud rather than silent.
    """
    factor = float(rng.choice([0.25, 4.0]))
    k_start = float(k)
    resampled = False
    if not _same_cv2_band(k, k * factor):
        k = _resample_k_same_band(k, factor, rng)
        resampled = True
        if not _same_cv2_band(k, k * factor):  # pragma: no cover - defensive
            raise RuntimeError(
                f"constraint 1 still violated after resampling: k={k}, factor={factor}"
            )

    k_new = k * factor
    delta = rng.binomial(1, p, size=T).astype("float64")
    k_t = np.full(T, k)
    k_t[tau:] = k_new
    z = rng.gamma(shape=k_t, scale=mu_z / k_t)
    return delta * z, delta, z, {
        "k_new": k_new,
        "factor": factor,
        "k_sampled": k_start,
        "k_used": float(k),
        "k_resampled_for_constraint_1": bool(resampled),
        "cv2_before": _cv2(k),
        "cv2_after": _cv2(k_new),
        "cv2_star": CV2_STAR,
        "sd_log_before": float(np.sqrt(_trigamma(k))),
        "sd_log_after": float(np.sqrt(_trigamma(k_new))),
        "demand_touched": True,
    }


def _trigamma(k: float) -> float:
    from scipy.special import polygamma

    return float(polygamma(1, k))


# --------------------------------------------------------------------------- #
# C5 -- level shift
# --------------------------------------------------------------------------- #


def _c5_required_sigma(p: float) -> float:
    """The step, in sd units, that constraint 3 demands at this ``p``."""
    return 0.25 + 4.0 / (p * 30.0)


def _c5_occurrence_feasible(p: float) -> bool:
    """Can an occurrence step ever clear constraint 3 at this ``p``?

    No, when ``p`` is small. The required step is ``k + h/(p * W_detect)``, which grows
    as ``1/p``, while the largest step ``p`` can absorb is bounded by its own range
    ``[0.03, 0.60]``: the achievable step in sd units is at most
    ``max(p - 0.03, 0.60 - p) / sqrt(p(1-p))``, which **falls** as ``p`` falls. The two
    cross at roughly ``p = 0.058``; below it the occurrence branch is unsatisfiable by
    arithmetic, not by tuning.

    The budget is never relaxed to fix this. ``inject_C5`` instead takes the SIZE
    branch, which is always satisfiable -- a size step can be scaled up without bound,
    whereas a probability cannot leave ``[0.03, 0.60]``. C5 is defined as a step in
    ``mu_z`` **or** ``p`` (§15.2), so this selects a satisfiable mechanism for the same
    cause rather than weakening the test.
    """
    sigma_occ = np.sqrt(p * (1.0 - p))
    best = max(p - 0.03, 0.60 - p) / sigma_occ
    return bool(best >= _c5_required_sigma(p))


def inject_C5(
    T: int,
    tau: int,
    p: float,
    k: float,
    mu_z: float,
    rng: np.random.Generator,
    *,
    target: str | None = None,
):
    """C5: an abrupt step in ``mu_z`` or in ``p``, in one period.

    The step is scaled up until it clears the detectability budget (constraint 3) --
    Revision 1 injected ``0.50 sigma`` against a required ``1.33 sigma``, undetectable
    by a factor of 2.7. ``target`` forces the branch, which the per-cause gate needs;
    when it is not forced and the occurrence branch is infeasible, the size branch is
    taken (see ``_c5_occurrence_feasible``).
    """
    if target is None:
        target = "occurrence" if _c5_occurrence_feasible(p) else "size"

    p_t = np.full(T, float(p))
    mu_z_t = np.full(T, float(mu_z))
    param: dict = {"target": target, "demand_touched": True}

    if target == "size":
        factor = float(rng.choice([0.5, 2.0]))
        sigma_z = mu_z / np.sqrt(k)
        req = _c5_required_sigma(p)
        direction = 1.0 if factor > 1 else -1.0
        mu_z_new = mu_z * factor
        # Scaling DOWN can hit the positivity floor at large req / small k. The UP
        # direction is unbounded, so flip rather than fail -- constraint 3 is about the
        # step being large enough, not about its sign.
        if direction < 0 and mu_z - req * sigma_z * 1.05 <= 0.1:
            direction = 1.0
            mu_z_new = mu_z * 2.0  # the flip's own candidate, before the budget check
            param["direction_flipped"] = True
        # If the candidate step does not clear the budget, take the smallest step that
        # does. The candidate is kept when it already clears, so a factor of 0.5 stays a
        # 0.5 rather than being truncated to the minimum -- truncating it would make the
        # injection weaker than the table specifies.
        if abs(mu_z_new - mu_z) / sigma_z < req:
            mu_z_new = mu_z + direction * req * sigma_z * 1.05
        mu_z_new = max(mu_z_new, 0.1)
        mu_z_t[tau:] = mu_z_new
        param.update(
            mu_z_new=float(mu_z_new),
            factor=factor,
            delta_sigma=float(abs(mu_z_new - mu_z) / sigma_z),
            budget_cleared=bool(check_c5_budget(abs(mu_z_new - mu_z) / sigma_z, p)),
        )
    else:
        shift = float(rng.choice([-0.20, 0.20]))
        sigma_occ = float(np.sqrt(p * (1 - p)))
        req = _c5_required_sigma(p)
        direction = 1.0 if shift > 0 else -1.0
        # Prefer a direction with headroom; with none available, take the largest step
        # the range allows and FLAG it (constraint 3's own escape clause).
        room_up, room_down = 0.60 - p, p - 0.03
        if direction > 0 and room_up < req * sigma_occ <= room_down:
            direction = -1.0
            param["direction_flipped"] = True
        elif direction < 0 and room_down < req * sigma_occ <= room_up:
            direction = 1.0
            param["direction_flipped"] = True
        p_new = float(np.clip(p + direction * req * sigma_occ * 1.05, 0.03, 0.60))
        p_t[tau:] = p_new
        param.update(
            p_new=p_new,
            shift=float(p_new - p),
            delta_sigma=float(abs(p_new - p) / sigma_occ),
            budget_cleared=bool(check_c5_budget(abs(p_new - p) / sigma_occ, p)),
            occurrence_feasible=bool(_c5_occurrence_feasible(p)),
        )

    delta = rng.binomial(1, p_t).astype("float64")
    z = rng.gamma(shape=k, scale=mu_z_t / k)
    return delta * z, delta, z, param


# --------------------------------------------------------------------------- #
# C6 -- intermittency regime
# --------------------------------------------------------------------------- #


def inject_C6(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C6: a slow drift in ``p``, sustained, so ADI crosses ``ADI* = 4.0``.

    ``p`` crosses ``0.25`` (the ``ADI = 4`` line) by the end of the panel. ``mu_z`` is
    untouched, so the size statistic is moved only through the level -- which is why
    C4's statistic is measured on the log scale. Per §15.2 the mean-demand change this
    causes is a DECLARED consequence, not a denial: ``E[y] = p * mu_z``, so it does
    move the mean, and it is invisible to C1's and C5's size statistics because both
    are restricted to nonzero periods.
    """
    direction = 1.0 if p < 0.25 else -1.0
    target_p = 0.35 if direction > 0 else 0.15
    psi = (target_p / p) - 1.0

    t_idx = np.arange(T)
    drift = np.zeros(T)
    drift[tau:] = (t_idx[tau:] - tau) / 60.0
    p_t = np.clip(p * (1.0 + psi * drift), 0.03, 0.60)

    delta = rng.binomial(1, p_t).astype("float64")
    z = rng.gamma(shape=k, scale=mu_z / k, size=T)
    return delta * z, delta, z, {
        "psi": float(psi),
        "p_end": float(p_t[-1]),
        "adi_end": float(1.0 / p_t[-1]),
        "adi_crosses_star": bool(1.0 / p_t[-1] >= 4.0 or 1.0 / p >= 4.0),
        "demand_touched": True,
    }


# --------------------------------------------------------------------------- #
# C7 -- policy mis-set (no change to the demand DGP at all)
# --------------------------------------------------------------------------- #


def inject_C7(T: int, tau: int, p: float, k: float, mu_z: float, rng: np.random.Generator):
    """C7: the policy's safety factor is set to 60 % of the correct value.

    **No change to the demand DGP**, and the forecaster's fitting input is
    ``uncensored`` (§14.3 rule 6) so the injection's own consequence -- stock-outs
    censoring the observed series -- does not corrupt the forecast. Without that second
    half, C7 stayed undetectable even after the window was fixed (§15.2).
    """
    delta, z = _baseline_draw(T, p, k, mu_z, rng)
    return delta * z, delta, z, {
        "safety_scale": 0.60,
        "fit_on": "uncensored",
        "demand_touched": False,
    }


# --------------------------------------------------------------------------- #
# the mixture panel (Rev 2 §15.3, resolution (a))
# --------------------------------------------------------------------------- #


def inject_mixture(
    T: int,
    tau: int,
    p: float,
    k: float,
    mu_z: float,
    rng: np.random.Generator,
    *,
    pair: tuple[str, str] = ("C3", "C2"),
):
    """Two mechanisms active SIMULTANEOUSLY, at the same ``tau``.

    Rev 2 §15.3 offers two honest resolutions and **recommends (a)**: restate the panel
    as a *mixture-detection* panel rather than calling a 6-period gap "ambiguous" when
    every monitor horizon is longer than the gap. This is resolution (a): ``gap = 0``,
    the panel is named a mixture panel, and it tests that confidence drops and
    abstention rises when two mechanisms are active at once. It is evaluated separately
    and never folded into the macro-F1.
    """
    a, b = pair
    if a == "C3" and b == "C2":
        y, delta, z, pa = inject_C3(T, tau, p, k, mu_z, rng)
        _, _, _, pb = inject_C2(T, tau, p, k, mu_z, rng)
    else:  # pragma: no cover - only the C3 x C2 pair is registered
        raise ValueError(f"unregistered mixture pair: {pair!r}")

    params = {
        "pair": [a, b],
        "gap": 0,
        "members": {"C3": pa, "C2": pb},
        "demand_touched": True,
    }
    return y, delta, z, params


INJECTORS = {
    "C0": inject_C0,
    "C1": inject_C1,
    "C2": inject_C2,
    "C2b": inject_C2b,
    "C3": inject_C3,
    "C4": inject_C4,
    "C5": inject_C5,
    "C6": inject_C6,
    "C7": inject_C7,
    "ambiguous": inject_mixture,
}
