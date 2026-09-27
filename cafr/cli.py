"""CAFR command line. **The only source of any number that goes in the paper.**

Rev 2 §30.2 / handoff §12, design rule 5: *"no notebook-sourced numbers."* Every
reported figure comes from a run of this module with a config and a seed, and the run
writes a manifest carrying the config hash (T8) so the number can be traced back.

Usage::

    python -m cafr.cli doctor                       # environment + data sanity
    python -m cafr.cli panels                       # dataset summary (T1 inputs)
    python -m cafr.cli run    --config base.yaml --seed 42 --steps 1,2
    python -m cafr.cli verify --config base.yaml --seed 42
    python -m cafr.cli gates                        # run the pytest gate suite

Every subcommand takes ``--config`` and ``--seed``; nothing reads a hard-coded path.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .utils.config import REPO_ROOT, config_hash, load_config

# --------------------------------------------------------------------------- #
# Stage registry. A stage that is not implemented yet must SAY SO and name the     #
# handoff step, rather than silently producing a partial run whose results look    #
# complete. Step numbers are handoff §13.                                          #
# --------------------------------------------------------------------------- #

STAGES: dict[str, tuple[str, str]] = {
    "0": ("no-look-ahead whitelist", "DONE"),
    "1": ("inventory simulator (M5)", "DONE"),
    "2": ("forecaster pool (M0)", "DONE"),
    "3": ("baselines (a),(b),(c),(e) on RUF", "handoff §13 Step 3"),
    "4": ("labelled synthetic panel", "handoff §13 Step 4"),
    "5": ("M1 detectors + policy features", "handoff §13 Step 5"),
    "6": ("M2 attribution classifier", "handoff §13 Step 6"),
    "7": ("M3 remedies R0-R7", "handoff §13 Step 7"),
    "8": ("M4 bandit + arm (g) fixed table", "handoff §13 Step 8"),
    "9": ("full CAFR arm (f)", "handoff §13 Step 9"),
    "10": ("all arms + churn matching", "handoff §13 Step 10"),
    "11": ("statistics + bootstrap CIs", "handoff §13 Step 11"),
    "12": ("figures P1-P11, tables T1-T8", "handoff §13 Step 12"),
    "13": ("diagnostics + ablations", "handoff §13 Step 13"),
    "14": ("reproducibility pass", "handoff §13 Step 14"),
}
IMPLEMENTED = ("0", "1", "2")


def _banner(text: str) -> None:
    print(f"\n=== {text} " + "=" * max(0, 68 - len(text)))


def _manifest(cfg: dict, seed: int, steps: list[str]) -> dict:
    return {
        "config_hash": config_hash(cfg),
        "seed_root": seed,
        "steps": steps,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "produced_by": "python -m cafr.cli",
        "note": (
            "Every number reported in the paper must be traceable to one of these "
            "manifests. A number with no manifest is a number with no provenance."
        ),
    }


# --------------------------------------------------------------------------- #
# doctor                                                                        #
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment and the committed data before anything else runs."""
    _banner("doctor")
    cfg = load_config(args.config)
    ok = True

    print(f"repo root      : {REPO_ROOT}")
    print(f"python         : {sys.version.split()[0]}  ({platform.platform()})")
    print(f"config         : {args.config}  hash {config_hash(cfg)}")

    for mod in ("numpy", "pandas", "scipy", "pyarrow", "yaml", "statsforecast", "lightgbm"):
        try:
            m = __import__(mod)
            print(f"  {mod:<14}: {getattr(m, '__version__', '?'):<12} OK")
        except Exception as exc:                                   # pragma: no cover
            print(f"  {mod:<14}: MISSING -- {exc}")
            ok = False

    try:
        import torch  # noqa: F401
        print("  note           : torch present; CAFR is CPU-only and does not use it")
    except Exception:
        pass

    panels = cfg["panels"]["ruf"]
    for name, rel in (("demand_panel", panels["path"]), ("sku_meta", panels["meta"]),
                      ("attributes", panels["attributes"]), ("profile", panels["profile"])):
        p = REPO_ROOT / rel
        print(f"  {name:<14}: {'OK ' if p.exists() else 'MISSING '} {rel}")
        ok &= p.exists()

    for step, (label, status) in STAGES.items():
        if status != "DONE":
            print(f"  step {step:<3} : not implemented ({label}) -- {status}")

    _banner("doctor: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# panels                                                                        #
# --------------------------------------------------------------------------- #


def cmd_panels(args: argparse.Namespace) -> int:
    """Print the T1 dataset numbers straight from the committed panel."""
    _banner("panels")
    from .data.loaders.ruf import load_observed, load_sku_meta
    from .utils.io import read_json

    cfg = load_config(args.config)
    observed = load_observed(cfg)
    meta = load_sku_meta(cfg)
    profile = read_json(REPO_ROOT / cfg["panels"]["ruf"]["profile"])

    print(f"SKUs x periods        : {observed['sku_id'].nunique():,} x "
          f"{observed['period'].nunique()}")
    print(f"burn-in / evaluated   : {cfg['splits']['burn_in']} / {cfg['splits']['evaluated']}"
          f"   (H-1: Rev 2 assumed {profile['rev2_assumed_periods']} periods)")
    print(f"zero fraction         : {profile['zero_fraction']:.4f}")
    print(f"ADI median (full)     : {profile['adi_full_series']['median']:.3f}")
    print(f"CV^2 median (full)    : {profile['cv_squared_full_series']['median']:.3f}")
    print(f"dead SKUs             : full {profile['dead_skus_full_series']}, "
          f"burn-in {profile['dead_skus_in_burn_in']}")
    print(f"cells (burn-in)       : {profile['cells_from_burn_in_adi_init_cv2_init']}")
    print(f"censoring observed    : {profile['censoring_observed']}  "
          "-> fit_on is FORCED to 'observed' (H-3)")
    print(f"ground truth present  : {profile['ground_truth_present']}  "
          "-> RQ1 is measured on the synthetic panel only")
    print("\ndetector evidence support (fraction of SKUs the cause can EVER fire on):")
    for cause, row in profile["detector_evidence_support_fraction"].items():
        print(f"  {cause}: " + "  ".join(f"{k} {v:.1%}" for k, v in row.items()))
    return 0


# --------------------------------------------------------------------------- #
# run                                                                           #
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    """Run the implemented stages and write a manifest. Later stages raise."""
    _banner(f"run: steps {','.join(args.steps)} seed {args.seed}")
    cfg = load_config(args.config, overrides={"seed_root": args.seed})

    unknown = [s for s in args.steps if s not in STAGES]
    if unknown:
        print(f"unknown steps: {unknown}; known: {sorted(STAGES, key=int)}")
        return 2

    blocked = [s for s in args.steps if s not in IMPLEMENTED]
    if blocked:
        for s in blocked:
            label, where = STAGES[s]
            print(f"step {s} ({label}) is NOT IMPLEMENTED -- see {where}")
        print(
            "\nThis is deliberate: a partial run that writes plausible results is "
            "worse than no run. Implement the stage, then add it to IMPLEMENTED."
        )
        return 3

    started = time.time()
    from .sim.inventory import InventorySimulator, SimConfig

    sim = InventorySimulator(SimConfig.from_config(cfg), root_seed=args.seed)
    print(f"simulator      : L={sim.cfg.L} R={sim.cfg.R} "
          f"protection={sim.cfg.protection} ({sim.cfg.protection_convention}) "
          f"target CSL {sim.cfg.target_csl:.3f}")
    print(f"fit_on         : {cfg['forecast']['fit_on']}")

    if "2" in args.steps:
        from .forecasters.registry import build_pool

        pool = build_pool(cfg)
        print(f"forecaster pool: {', '.join(pool)}")

    out_dir = REPO_ROOT / "results" / f"run_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(cfg, args.seed, args.steps)
    manifest["runtime_seconds"] = round(time.time() - started, 3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest       : {out_dir / 'manifest.json'}")
    print(f"config hash    : {manifest['config_hash']}   (T8)")
    return 0


# --------------------------------------------------------------------------- #
# verify / gates                                                                #
# --------------------------------------------------------------------------- #


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the gate suite through pytest and report. Never edits a threshold."""
    _banner("verify")
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    if args.marker:
        cmd += ["-m", args.marker]
    if args.only:
        cmd += ["-k", args.only]
    print(" ".join(cmd), "\n")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def cmd_gates(args: argparse.Namespace) -> int:
    """Print the gate table with its implementation status. No side effects."""
    _banner("gates")
    from .utils.io import read_json

    print(f"{'step':<5} {'stage':<42} {'status'}")
    print("-" * 68)
    for step, (label, status) in STAGES.items():
        mark = "DONE" if status == "DONE" else "not started"
        print(f"{step:<5} {label:<42} {mark}")

    for extra in ("step1_protection_diagnostic.json", "step1_gate_1_3.json"):
        path = REPO_ROOT / "results" / extra
        if path.exists():
            payload = read_json(path)
            print(f"\n{extra}:")
            print(json.dumps(payload, indent=2)[:2000])
        else:
            print(f"\n{extra}: not produced yet -- run `cafr verify`")
    return 0


# --------------------------------------------------------------------------- #
# entry point                                                                   #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cafr", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="base.yaml", help="config name or path")
        p.add_argument("--seed", type=int, default=42, help="root seed")

    p = sub.add_parser("doctor", help="environment + data sanity check")
    p.add_argument("--config", default="base.yaml")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("panels", help="dataset summary (T1 inputs)")
    p.add_argument("--config", default="base.yaml")
    p.set_defaults(func=cmd_panels)

    p = sub.add_parser("run", help="run the implemented stages")
    common(p)
    p.add_argument("--steps", default="0,1,2", help="comma-separated step numbers")
    p.set_defaults(func=lambda a: (setattr(a, "steps", a.steps.split(",")), cmd_run(a))[1])

    p = sub.add_parser("verify", help="run the gate suite")
    common(p)
    p.add_argument("--marker", default=None, help="pytest -m expression")
    p.add_argument("--only", default=None, help="pytest -k expression")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("gates", help="gate table + last measured diagnostics")
    p.add_argument("--config", default="base.yaml")
    p.set_defaults(func=cmd_gates)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
