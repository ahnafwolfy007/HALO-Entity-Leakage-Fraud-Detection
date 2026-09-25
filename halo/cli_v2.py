"""v2 command line. Every run is resumable; re-issuing a command continues it.

    python -m halo.cli_v2 smoke                       # end-to-end, no downloads
    python -m halo.cli_v2 datasets                    # what is present on disk
    python -m halo.cli_v2 status                      # ledger progress per stage
    python -m halo.cli_v2 run --stage ladder --datasets ieee_cis --max-hours 11.5
    python -m halo.cli_v2 run --stage all --datasets auto --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import CFG, CKPT_DIR, RESULTS_DIR
from .datasets import ALL_DATASETS, describe_all
from .resume import Ledger, sweep_stale_tmp


def _parse_datasets(arg: str) -> tuple[str, ...]:
    """Resolve the --datasets argument to datasets that are actually loadable.

    Named datasets are checked, not taken on trust. Data tends to arrive one
    dataset at a time, so a run naming four of them when three are present should
    produce results for those three rather than a traceback -- losing a whole
    invocation because the last name in a list has not been downloaded yet is an
    expensive way to learn that.
    """
    from .experiments_v2 import available_datasets

    if arg == "auto":
        found = available_datasets()
        if not found:
            print("no datasets found on disk; falling back to 'synthetic'")
            return ("synthetic",)
        return tuple(found)

    asked = tuple(x.strip() for x in arg.split(",") if x.strip())
    real = tuple(d for d in asked if d != "synthetic")
    if not real:
        return asked

    usable = set(available_datasets(real)) | {"synthetic"}
    keep = tuple(d for d in asked if d in usable)
    missing = [d for d in asked if d not in usable]
    if missing:
        print(f"  [skip] not loadable, dropped from this run: {', '.join(missing)}")
    if not keep:
        print("none of the requested datasets are loadable; nothing to do")
    return keep


def cmd_datasets(args) -> int:
    from .experiments_v2 import available_datasets
    print(describe_all())
    print("\nPresent on disk:")
    for name in available_datasets():
        print(f"  ok  {name}")
    return 0


def cmd_status(args) -> int:
    from .experiments_v2 import feature_code_version
    # Must match `_ledger`: a status that counts entries the runner would redo
    # is worse than no status at all, because it reports work as banked that is
    # about to be recomputed.
    chash = f"{CFG.config_hash()}|code{feature_code_version()}"
    rows = []
    for path in sorted(Path(RESULTS_DIR).glob("ledger_*.jsonl")):
        led = Ledger(path, config_hash=chash)
        frame = led.to_frame()
        rows.append({
            "stage": path.stem.replace("ledger_", ""),
            "completed": len(led),
            "datasets": (sorted(frame["u_dataset"].unique().tolist())
                         if "u_dataset" in frame.columns else []),
            "hours": round(frame["seconds"].sum() / 3600, 2) if "seconds" in frame else 0,
        })
    if not rows:
        print("no ledgers yet")
        return 0
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_smoke(args) -> int:
    """Exercise every v2 component on generated data. No downloads, ~1 minute."""
    from .experiments_v2 import (FeatureConfig, build_features, fit_one, get_bundle,
                                 stage_ablation, stage_ladder)
    from .protocol import rolling_origin_folds

    print("1/5  synthetic bundle")
    b = get_bundle("synthetic")
    print(f"     {len(b.df):,} rows, {b.df['y'].mean():.3%} fraud, "
          f"{len(set(b.entity)):,} entities")

    print("2/5  folds")
    folds = rolling_origin_folds(b.df, b.entity, latency_delta=30, entity_disjoint=True)
    assert folds, "no folds produced"
    print(f"     {len(folds)} folds, first: train={folds[0].n_train:,} "
          f"test={folds[0].n_test:,} dropped_entity_rows={folds[0].dropped_entity_rows:,}")

    print("3/5  features (Block B v2: counters + 2-hop + variance)")
    X = build_features(b, folds[0], FeatureConfig(lam=30, hops=2), cache=False)
    hop2 = [c for c in X.columns if "hop2" in c]
    var = [c for c in X.columns if c.startswith("riskvar_")]
    print(f"     {X.shape[1]} features; {len(hop2)} hop-2, {len(var)} variance")
    assert hop2 and var, "C2/C3 features missing"

    print("4/5  fit + score (C4 dollar cost, C5 region monotonicity)")
    res = fit_one({"dataset": "synthetic", "model": "halo", "seed": 0,
                   "fold": 0, "lam": 30})
    print(f"     AUPRC={res.get('auprc'):.4f}  AUROC={res.get('auroc'):.4f}  "
          f"monotone={res.get('n_monotone')}")

    print("5/5  resumable stage (2 seeds x 1 fold)")
    df = stage_ladder(("synthetic",), seeds=(0, 1), folds=(0,))
    print(f"     ladder rows: {len(df)}")
    print("\nsmoke PASSED")
    return 0


def cmd_verify(args) -> int:
    """Property tests for the claims the paper asserts."""
    from .validate_v2 import run_all
    print("verifying v2 properties (synthetic data)\n")
    df = run_all()
    n_fail = int((~df["passed"].astype(bool)).sum())
    print(f"\n{len(df) - n_fail}/{len(df)} properties hold")
    return 1 if n_fail else 0


def cmd_selftest(args) -> int:
    """Crash-safety tests. Run these before trusting a long unattended run."""
    from .test_resume import run_all
    print("crash-safety self-tests (simulated power loss)\n")
    df = run_all()
    n_fail = int((~df["passed"].astype(bool)).sum())
    print(f"\n{len(df) - n_fail}/{len(df)} crash-safety tests pass")
    return 1 if n_fail else 0


def cmd_run(args) -> int:
    from .experiments_v2 import STAGES

    datasets = _parse_datasets(args.datasets)
    if not datasets:
        return 1
    seeds = tuple(range(args.seeds))
    folds = tuple(range(args.folds))
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    ckpt_dir = Path(args.ckpt) if args.ckpt else CKPT_DIR
    swept = sweep_stale_tmp(RESULTS_DIR) + sweep_stale_tmp(ckpt_dir)
    if swept:
        print(f"swept {swept} stale temp files from a previous interruption")

    print(f"datasets: {datasets}\nstages:   {stages}\n"
          f"seeds:    {seeds}\nfolds:    {folds}\n")

    for stage in stages:
        fn = STAGES[stage]
        kw = dict(datasets=datasets, seeds=seeds, max_hours=args.max_hours)
        if stage != "censoring":
            kw["folds"] = folds
        print(f"\n=== {stage} ===")
        try:
            fn(**kw)
        except KeyboardInterrupt:
            print(f"\ninterrupted during {stage}; progress is on disk. "
                  f"Re-run the same command to continue.")
            return 130
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="halo.cli_v2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    sub.add_parser("datasets").set_defaults(fn=cmd_datasets)
    sub.add_parser("status").set_defaults(fn=cmd_status)

    r = sub.add_parser("run")
    r.add_argument("--stage", default="all",
                   choices=["all", "ladder", "benchmark", "latency", "ablation", "censoring"])
    r.add_argument("--datasets", default="auto",
                   help=f"comma-separated, or 'auto'. known: {','.join(ALL_DATASETS)}")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--folds", type=int, default=4)
    r.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly before this budget (e.g. 11.5 on Kaggle)")
    r.add_argument("--ckpt", default=None)
    r.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
