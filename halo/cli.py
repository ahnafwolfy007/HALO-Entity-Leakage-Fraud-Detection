"""Command-line entry point. Every Kaggle notebook stage is one subcommand.

    python -m halo.cli smoke            # synthetic end-to-end verification, ~2 min
    python -m halo.cli run-entities     # NB-1: Block A + T2  (the go/no-go)
    python -m halo.cli run-ladder       # NB-2: T1
    python -m halo.cli run-size-control # NB-2: T1b (the confound control)
    python -m halo.cli run-main         # NB-4: T3
    python -m halo.cli run-latency      # NB-4: T5
    python -m halo.cli run-cost         # NB-4: T7
    python -m halo.cli run-ablation     # NB-5: T4
    python -m halo.cli run-drift        # NB-5: F3
    python -m halo.cli run-shap         # NB-6: F1
    python -m halo.cli run-faithfulness # NB-6: T6
    python -m halo.cli run-paysim       # NB-7: L5
    python -m halo.cli report           # NB-8: report + ZIP

``--synthetic`` runs any stage against generated data instead of the real CSVs, which is
how the code is verified before a Kaggle session is spent on it.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, IEEE_DIR, WORK_DIR
from .data import add_base_features, load_ieee
from .entities import add_d1n, resolve_entities
from .io import Timer, environment_manifest, load_obj, save_obj
from .protocol import index_event_mask


# --------------------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------------------

def prepare(args) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Load, join, derive base features, resolve entities, mark index events.

    Cached to disk, because every stage needs it and entity resolution is not free.
    """
    cache = f"prepared_{'synth' if args.synthetic else 'ieee'}_{args.rows or 'all'}"
    cached = load_obj(cache)
    if cached is not None and not args.no_cache:
        print(f"  (using cached {cache})")
        return cached

    truth = None
    if args.synthetic:
        from .synth import make_synthetic
        with Timer("generate synthetic data"):
            txn, ident, truth = make_synthetic(n_entities=args.entities, seed=0)
        df = txn.merge(ident, on="TransactionID", how="left")
    else:
        if not IEEE_DIR.exists():
            raise SystemExit(
                f"IEEE-CIS data not found at {IEEE_DIR}.\n"
                "On Kaggle: Add Data -> ieee-fraud-detection (accept the competition rules).\n"
                "Locally: place train_transaction.csv and train_identity.csv there,\n"
                "or pass --synthetic to run against generated data.")
        df = load_ieee(nrows=args.rows)

    with Timer("base features"):
        df = add_base_features(add_d1n(df))
    with Timer("Block A: entity resolution"):
        entity = resolve_entities(df, variant=args.uid).to_numpy()
    with Timer("index events"):
        is_index = index_event_mask(df, entity)

    print(f"  rows={len(df):,}  features={df.shape[1]}  entities={len(np.unique(entity)):,}"
          f"  fraud_rate={df['isFraud'].mean():.4f}  index_events={is_index.sum():,}")
    out = (df, entity, is_index, truth)
    save_obj(out, cache)
    return out


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------

def stage_entities(args):
    from .experiments import run_entity_stats
    df, entity, is_index, truth = prepare(args)
    t2 = run_entity_stats(df, truth)
    cols = [c for c in ("uid_variant", "n_entities", "entity_label_purity",
                        "propagated_positive_share", "c_monotonicity_violation_rate",
                        "er_pair_f1", "entity_size_median", "left_truncated_entities")
            if c in t2.columns]
    print(t2[cols].to_string(index=False))
    purity = float(t2.loc[t2["uid_variant"] == args.uid, "entity_label_purity"].iloc[0])
    print(f"\nGO/NO-GO: entity label purity = {purity:.4f}")
    print("  High purity means entity identity nearly determines the label, which is the"
          "\n  evidence L3 rests on. Low purity would mean the thesis needs rethinking."
          "\n  Either way this is the number to report; do not tune anything to move it.")
    return t2


def stage_ladder(args):
    from .experiments import run_ladder
    df, entity, is_index, _ = prepare(args)
    out = run_ladder(df, entity, is_index, seeds=tuple(args.seeds), model_name=args.model)
    print(out.to_string(index=False))
    return out


def stage_size_control(args):
    from .experiments import run_size_control
    df, entity, is_index, _ = prepare(args)
    out = run_size_control(df, entity, is_index, seeds=tuple(args.seeds))
    print(out.to_string(index=False))
    from .io import load_table
    attr = load_table("T1b_attribution")
    if attr is not None:
        print("\nDecomposition of the entity-disjointness drop:")
        print(attr.T.to_string())
        print("\n  A high leakage_share means the collapse really is entity leakage."
              "\n  A low one means much of it was simply training on less data, and the"
              "\n  paper must say so. Report whichever you get.")
    return out


def stage_main(args):
    from .experiments import run_main
    df, entity, is_index, _ = prepare(args)
    out = run_main(df, entity, is_index, seeds=tuple(args.seeds),
                   delta=args.delta, tuning_budget=args.tuning_budget)
    print(out.to_string(index=False))
    return out


def stage_ablation(args):
    from .experiments import run_ablation
    df, entity, is_index, _ = prepare(args)
    out = run_ablation(df, entity, is_index, seeds=tuple(args.seeds), delta=args.delta)
    print(out.to_string(index=False))
    return out


def stage_latency(args):
    from .experiments import run_latency_sweep
    df, entity, is_index, _ = prepare(args)
    out = run_latency_sweep(df, entity, is_index, seeds=tuple(args.seeds))
    print(out.to_string(index=False))
    return out


def stage_cost(args):
    from .experiments import run_cost_and_throughput
    df, entity, is_index, _ = prepare(args)
    out = run_cost_and_throughput(df, entity, is_index)
    print(out.to_string(index=False))
    return out


def stage_drift(args):
    from .experiments import run_drift
    df, entity, is_index, _ = prepare(args)
    out = run_drift(df, entity, is_index)
    print(out.head(20).to_string(index=False))
    return out


def stage_shap(args):
    from .explain import run_shap_migration
    df, entity, is_index, _ = prepare(args)
    out = run_shap_migration(df, entity, is_index)
    print(out.to_string(index=False))
    _plot_migration(out)
    return out


def stage_faithfulness(args):
    from .explain import run_faithfulness
    df, entity, is_index, _ = prepare(args)
    out = run_faithfulness(df, entity, is_index)
    print(out.T.to_string())
    return out


def stage_paysim(args):
    from .paysim import load_paysim, run_paysim_audit, paysim_shortcut_evidence
    df = load_paysim(nrows=args.rows)
    ev = paysim_shortcut_evidence(df)
    print(ev.to_string(index=False))
    out = run_paysim_audit(df, seeds=tuple(args.seeds[:3]))
    print(out.to_string(index=False))
    return out


def stage_report(args):
    from .report import build_report
    from .package import build_zip, write_docs
    write_docs()
    path = build_report()
    print(f"  report -> {path}")
    z = build_zip()
    print(f"  bundle -> {z}")
    print("\nOn Kaggle: the file appears under the notebook's Output tab (right-hand"
          "\npanel, 'Data' -> /kaggle/working). Click the download icon beside"
          f"\n'{z.name}'. Or run the notebook and use 'Download All' on the Output tab.")
    return path


def stage_smoke(args):
    """End-to-end verification on synthetic data with known ground truth."""
    args.synthetic = True
    args.entities = args.entities or 3000
    args.seeds = args.seeds[:2] or [0, 1]
    print("=" * 78)
    print("SMOKE TEST -- synthetic data, known ground truth")
    print("=" * 78)
    ok, failed = [], []
    for name, fn in (("entities", stage_entities), ("ladder", stage_ladder),
                     ("main", stage_main), ("ablation", stage_ablation),
                     ("latency", stage_latency), ("cost", stage_cost),
                     ("size-control", stage_size_control),
                     ("drift", stage_drift), ("shap", stage_shap),
                     ("faithfulness", stage_faithfulness)):
        print(f"\n--- {name} " + "-" * (70 - len(name)))
        try:
            fn(args)
            ok.append(name)
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(limit=3)
    print("\n" + "=" * 78)
    print(f"PASSED : {', '.join(ok) if ok else '(none)'}")
    for n, e in failed:
        print(f"FAILED : {n} -- {e}")
    print("=" * 78)
    if failed:
        sys.exit(1)


def _plot_migration(piv: pd.DataFrame) -> None:
    """F1 as a chart. Diverging bars: which families gain mass under the honest protocol."""
    if piv is None or len(piv) == 0 or "migration" not in piv.columns:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .config import FIG_DIR

    d = piv.sort_values("migration")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.6), dpi=150)
    colors = ["#8C6310" if v < 0 else "#1B6A61" for v in d["migration"]]
    ax.barh(d["family"], d["migration"], color=colors)
    ax.axvline(0, color="#6E7A7C", lw=0.8)
    ax.set_xlabel("Change in SHAP mass share  (CEP  −  leaky protocol)")
    ax.set_title("F1 — attribution mass migrates from identity to behaviour", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F1_shap_mass_migration.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------

STAGES = {
    "smoke": stage_smoke, "run-entities": stage_entities, "run-ladder": stage_ladder,
    "run-main": stage_main, "run-ablation": stage_ablation, "run-latency": stage_latency,
    "run-cost": stage_cost, "run-size-control": stage_size_control, "run-drift": stage_drift, "run-shap": stage_shap,
    "run-faithfulness": stage_faithfulness, "run-paysim": stage_paysim,
    "report": stage_report,
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="halo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("--synthetic", action="store_true",
                   help="run against generated data with known ground truth")
    p.add_argument("--entities", type=int, default=6000,
                   help="synthetic entity count")
    p.add_argument("--rows", type=int, default=None, help="limit rows loaded")
    p.add_argument("--seeds", type=int, nargs="+", default=list(CFG.seeds))
    p.add_argument("--delta", type=int, default=None,
                   help="latency gate in days (default: config headline_delta)")
    p.add_argument("--uid", default=CFG.uid_variant,
                   choices=["strict", "medium", "loose"])
    p.add_argument("--model", default="lightgbm", help="model held constant for T1")
    p.add_argument("--tuning-budget", type=int, default=0,
                   help="inner random-search iterations, applied equally to all models")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    env = environment_manifest()
    print(f"halo | config={CFG.config_hash()} | python={env['python']} | "
          f"cpu={env['cpu_count']} | ram={env.get('ram_total_gb')}GB | "
          f"work={WORK_DIR}")
    return STAGES[args.stage](args)


if __name__ == "__main__":
    main()
