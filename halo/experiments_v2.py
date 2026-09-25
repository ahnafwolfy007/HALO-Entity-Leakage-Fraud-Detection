"""v2 experiment drivers: multi-dataset, resumable, crash-safe.

Every stage is a list of independent work units executed through
``resume.run_resumable``, so an interruption -- a power cut or a Kaggle session
cap -- costs one model fit rather than one stage.

Stages
------
``ladder``     T1 leakage ladder, per dataset
``benchmark``  T3 model comparison under CEP, per dataset
``latency``    T5 maturity sweep, lambda in {0, 7, 14, 30, 60, 120}
``censoring``  C1: drop vs zero vs soft under simulated reporting delay
``ablation``   C1-C5 on/off, plus the H1 constraint-value-under-noise test
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as B
from .config import CFG, CKPT_DIR, RESULTS_DIR
from .censoring import (CensoredLabelTrainer, CensoringConfig, DelayModel,
                        simulate_report_days, visible_labels)
from .counters import fit_beta_prior  # noqa: F401  (re-exported for notebooks)
from .datasets import ALL_DATASETS, DatasetNotAvailable, load_dataset
from .io import Timer, save_table
from .metrics import evaluate
from .model_v2 import RegionConfig, RegionMonotoneGBDT, dollar_weights
from .protocol import cold_entity_mask, rolling_origin_folds
from .resume import (Ledger, atomic_save_df, expand_grid, prune_cache,
                     run_resumable, safe_load_df)
from .risk_v2 import AssociationRiskV2

LAMBDA_GRID = (0, 7, 14, 30, 60, 120)
BASELINES = ("lightgbm", "xgboost", "catboost", "mlp", "logreg")

CACHE_BUDGET_GB = float(os.environ.get("HALO_CACHE_GB", "12"))
"""Disk ceiling for cached feature tables, evicted least-recently-used.

A full sweep builds roughly 80 feature tables per dataset. At IEEE-CIS scale
that is a few gigabytes; at TabFormer scale it is tens. Filling the disk makes
the next atomic write fail, which loses the run that checkpointing exists to
protect -- so the cache is capped rather than left to grow. Set HALO_CACHE_GB=4
on Kaggle (20 GB working limit), higher on a roomy local disk."""


# --------------------------------------------------------------------------------------
# Code fingerprint for the feature cache
# --------------------------------------------------------------------------------------

_FEATURE_MODULES = ("counters.py", "propagation.py", "risk_v2.py")


@lru_cache(maxsize=1)
def feature_code_version() -> str:
    """Short hash of the code that computes features, ignoring prose.

    A cached feature table is only reusable if the code that produced it has not
    changed. Keying the cache on parameters alone means that fixing a bug in the
    risk path leaves every stale parquet file on disk still winning the lookup --
    the run then resumes onto silently wrong features, which is the same class of
    failure the leave-one-entity-out bug was.

    Docstrings and comments are stripped before hashing, so editing an
    explanation does not invalidate a week of computation on a machine that can
    only run between power cuts. Editing anything executable does.
    """
    import ast
    import hashlib

    h = hashlib.blake2b(digest_size=6)
    for name in _FEATURE_MODULES:
        src = (Path(__file__).parent / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        h.update(ast.dump(tree).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Dataset bundles
# --------------------------------------------------------------------------------------

@dataclass
class Bundle:
    name: str
    df: pd.DataFrame
    spec: object
    entity: np.ndarray
    edges: np.ndarray | None = None

    @property
    def y(self) -> np.ndarray:
        return self.df["y"].to_numpy()


def make_synthetic(n: int = 40_000, seed: int = 0, n_entities: int = 4_000
                   ) -> tuple[pd.DataFrame, object]:
    """Self-contained synthetic dataset so the pipeline is testable with no downloads.

    The generative story matters, because a careless one produces a dataset on
    which *no* method can score above chance under CEP, which would make the
    smoke test meaningless.

    Fraud here is driven by **shared infrastructure**, not by entity identity:
    a small set of devices and email domains is compromised, entities that
    transact through them are far more likely to be fraudulent, and entities
    are otherwise exchangeable. That is exactly the signal CEP is designed to
    leave intact -- an unseen entity using a known-bad device is still
    detectable -- so Block B should beat the base rate here while a model that
    relies on entity memorisation should not.
    """
    from .datasets import DatasetSpec
    rng = np.random.default_rng(seed)

    n_dev, n_email, n_bin = 600, 300, 120
    bad_dev = set(rng.choice(n_dev, size=n_dev // 12, replace=False))
    bad_email = set(rng.choice(n_email, size=n_email // 12, replace=False))

    ent = rng.integers(0, n_entities, size=n)
    # each entity prefers a small set of devices/emails, so infrastructure is
    # shared across entities rather than being an entity alias
    ent_dev = rng.integers(0, n_dev, size=n_entities)
    ent_email = rng.integers(0, n_email, size=n_entities)
    dev = np.where(rng.random(n) < 0.8, ent_dev[ent], rng.integers(0, n_dev, n))
    email = np.where(rng.random(n) < 0.8, ent_email[ent], rng.integers(0, n_email, n))
    bins = rng.integers(0, n_bin, size=n)

    day = np.sort(rng.integers(0, 182, size=n)).astype(np.float64)
    amount = np.exp(rng.normal(4.0, 1.2, n))

    is_bad_dev = np.array([d in bad_dev for d in dev])
    is_bad_email = np.array([e in bad_email for e in email])
    logit = (-4.2 + 2.3 * is_bad_dev + 1.7 * is_bad_email
             + 0.45 * (np.log1p(amount) - np.log1p(amount).mean()))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int8)

    df = pd.DataFrame({
        "ts": day * 86_400.0 + rng.random(n) * 86_400.0,
        "y": y,
        "amount": amount.astype(np.float32),
        "entity": ent.astype(np.int64),
        "device": [f"d{v}" for v in dev],
        "email": [f"e{v}" for v in email],
        "bin": [f"b{v}" for v in bins],
    })
    df = df.sort_values("ts").reset_index(drop=True)
    df["day"] = np.floor(df["ts"] / 86_400.0).astype(np.int32)
    df["isFraud"] = df["y"]
    spec = DatasetSpec(name="synthetic", risk_attributes=("device", "email", "bin"),
                       entity_from="native", has_amount=True, horizon_days=120.0,
                       notes="generated in-process; for smoke tests only")
    return df, spec


@lru_cache(maxsize=2)
def get_bundle(name: str, nrows: int | None = None) -> Bundle:
    """Load a dataset and attach entity ids. Cached -- these are expensive."""
    if name == "synthetic":
        df, spec = make_synthetic()
        return Bundle(name, df, spec, df["entity"].to_numpy())

    out = load_dataset(name, nrows=nrows)
    edges = None
    if len(out) == 3:
        df, spec, edges = out
    else:
        df, spec = out

    if spec.entity_from == "resolve":
        # IEEE-CIS: entity must be inferred label-free (Block A) before the
        # protocol can group by it. This is what makes CEP more than GroupKFold.
        from .entities import resolve_entities
        tmp = df.copy()
        tmp["TransactionDT"] = tmp["ts"]
        ent = resolve_entities(tmp, variant=CFG.uid_variant)
        entity = pd.factorize(ent)[0].astype(np.int64)
    else:
        entity = df["entity"].to_numpy(np.int64)

    df = df.copy()
    df["entity"] = entity
    df["isFraud"] = df["y"]          # alias for v1 modules (protocol, metrics)
    if "amount" in df.columns and not spec.has_amount:
        df["amount"] = np.nan
    return Bundle(name, df, spec, entity, edges)


# --------------------------------------------------------------------------------------
# Feature construction (cached per dataset x fold x config)
# --------------------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    lam: float = 30.0            # label maturity (delta), days
    hops: int = 2                # C2
    gamma: float = 0.4
    tau_days: float = 30.0
    tau_scales: tuple = ()       # C3 multi-scale
    variance: bool = True        # C3
    use_risk: bool = True
    use_memory: bool = True
    past_only: bool = True

    def key(self) -> str:
        return (f"lam{self.lam}_h{self.hops}_g{self.gamma}_t{self.tau_days}"
                f"_s{'-'.join(str(s) for s in self.tau_scales) or 'none'}"
                f"_v{int(self.variance)}_r{int(self.use_risk)}_m{int(self.use_memory)}")


def build_features(bundle: Bundle, fold, cfg: FeatureConfig,
                   cache: bool = True) -> pd.DataFrame:
    tag = f"feat_{bundle.name}_f{fold.fold_id}_{cfg.key()}_c{feature_code_version()}"
    path = CKPT_DIR / f"{tag}.parquet"
    if cache:
        hit = safe_load_df(path)
        if hit is not None:
            return hit

    df = bundle.df
    attrs = tuple(a for a in bundle.spec.risk_attributes if a in df.columns)

    ingest = np.zeros(len(df), dtype=bool)
    ingest[fold.train_idx] = True          # only training-window labels may be admitted

    X_parts = []
    if cfg.use_risk and attrs:
        ar = AssociationRiskV2(attrs, delta_days=cfg.lam, tau_days=cfg.tau_days,
                               tau_scales=cfg.tau_scales, gamma=cfg.gamma,
                               hops=cfg.hops, emit_variance=cfg.variance)
        now_ts = float(df["ts"].to_numpy()[fold.test_idx].min())
        ar.fit_prior(df.iloc[fold.train_idx], df["y"].to_numpy()[fold.train_idx],
                     now_ts=now_ts)
        X_parts.append(ar.transform(df, bundle.entity, ingest))

    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ("y", "isFraud", "ts", "day", "entity") if c in df.columns],
        errors="ignore")
    X_parts.append(num)

    X = pd.concat(X_parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    if not cfg.use_memory:
        X = X.drop(columns=[c for c in X.columns if c.startswith("mem_")])
    X = X.astype(np.float32)

    if cache:
        atomic_save_df(X, path)
        prune_cache(CKPT_DIR, CACHE_BUDGET_GB)
    return X


# --------------------------------------------------------------------------------------
# One fit
# --------------------------------------------------------------------------------------

def _model_kwargs(unit: dict, X: pd.DataFrame) -> dict:
    mono = AssociationRiskV2.monotone_feature_names(X.columns) if unit.get("monotone", True) else []
    return {
        "monotone_features": mono,
        "region": RegionConfig(alert_quantile=CFG.review_budget_k,
                               enforce=unit.get("monotone", True),
                               margin=unit.get("region_margin", 3.0)),
    }


def fit_one(unit: dict) -> dict:
    """Execute a single work unit. Pure with respect to the ledger."""
    bundle = get_bundle(unit["dataset"])
    spec = bundle.spec
    df = bundle.df
    lam = float(unit.get("lam", 30))

    folds = rolling_origin_folds(df, bundle.entity, latency_delta=int(lam),
                                 entity_disjoint=unit.get("entity_disjoint", True))
    if not folds:
        return {"status": "no_folds"}
    fold = folds[min(unit.get("fold", 0), len(folds) - 1)]

    fcfg = FeatureConfig(lam=lam, hops=unit.get("hops", 2), gamma=unit.get("gamma", 0.4),
                         tau_days=unit.get("tau_days", 30.0),
                         tau_scales=tuple(unit.get("tau_scales", ())),
                         variance=unit.get("variance", True),
                         use_risk=unit.get("use_risk", True),
                         use_memory=unit.get("use_memory", True))
    X = build_features(bundle, fold, fcfg)

    y = df["y"].to_numpy()
    tr, te = fold.train_idx, fold.test_idx
    amount = df["amount"].to_numpy() if spec.has_amount else None
    seed = int(unit.get("seed", 0))

    w_tr = (dollar_weights(y[tr], None if amount is None else amount[tr],
                           kappa=CFG.fn_cost_ratio)
            if unit.get("dollar_cost", True)
            else np.where(y[tr] == 1, CFG.fn_cost_ratio, 1.0))

    model_name = unit.get("model", "halo")
    if model_name == "halo":
        m = RegionMonotoneGBDT(seed=seed, **_model_kwargs(unit, X))
        m.fit(X.iloc[tr], y[tr], sample_weight=w_tr)
        s = m.decision_score(X.iloc[te])
        extra = {"n_monotone": len(m.monotone_features),
                 "region_fallback": getattr(m, "region_fallback_", None)}
    else:
        s, info = B.fit_predict(model_name, X.iloc[tr], y[tr], X.iloc[te], seed=seed,
                                resample=unit.get("resample_in_fold", False),
                                tuning_budget=unit.get("tuning_budget", 0))
        extra = {k: v for k, v in info.items() if not k.startswith("_")}

    res = evaluate(y[te], s, amount=None if amount is None else amount[te],
                   entity=bundle.entity[te],
                   cold_mask=cold_entity_mask(bundle.entity, fold), seed=seed)
    res.update(fold.meta())
    res.update(extra)
    res["n_features"] = int(X.shape[1])
    del X
    gc.collect()
    return res


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------

def _ledger(stage: str, config_hash: str | None = None) -> Ledger:
    """Ledger for one stage, keyed on both configuration and feature code.

    Folding in ``feature_code_version`` means a fix to the risk path retires the
    results it invalidates instead of letting a resumed run inherit them. The
    cost of being wrong here is a paper built on stale numbers; the cost of being
    conservative is recomputation, so this errs toward recomputation.
    """
    base = config_hash or CFG.config_hash()
    return Ledger(RESULTS_DIR / f"ledger_{stage}.jsonl",
                  config_hash=f"{base}|code{feature_code_version()}")


def stage_ladder(datasets: tuple[str, ...], seeds=(0, 1, 2, 3, 4), folds=(0, 1, 2, 3),
                 max_hours: float | None = None) -> pd.DataFrame:
    """T1: close one leakage channel at a time, model held constant."""
    rungs = [
        dict(rung="r0", entity_disjoint=False, lam=0, resample_in_fold=False, use_risk=False),
        dict(rung="r1", entity_disjoint=False, lam=0, resample_in_fold=True, use_risk=False),
        dict(rung="r2", entity_disjoint=False, lam=0, use_risk=True),
        dict(rung="r3", entity_disjoint=True, lam=0, use_risk=True),
        dict(rung="r4", entity_disjoint=True, lam=30, use_risk=True),
    ]
    units = [{**r, "dataset": d, "seed": s, "fold": f, "model": "lightgbm",
              "stage": "ladder"}
             for d in datasets for r in rungs for s in seeds for f in folds]
    df = run_resumable(units, fit_one, _ledger("ladder"), max_hours=max_hours,
                       label="T1-ladder")
    save_table(df, "T1_v2", "Leakage ladder, all datasets")
    return df


def stage_benchmark(datasets: tuple[str, ...], seeds=(0, 1, 2), folds=(0, 1, 2, 3),
                    models=("halo",) + BASELINES, lam: float = 30,
                    max_hours: float | None = None) -> pd.DataFrame:
    units = [{"dataset": d, "model": m, "seed": s, "fold": f, "lam": lam,
              "stage": "benchmark"}
             for d in datasets for m in models for s in seeds for f in folds]
    df = run_resumable(units, fit_one, _ledger("benchmark"), max_hours=max_hours,
                       label="T3-benchmark")
    save_table(df, "T3_v2", "Main benchmark under CEP, all datasets")
    return df


def stage_latency(datasets: tuple[str, ...], seeds=(0, 1, 2), folds=(0, 1, 2, 3),
                  models=("halo", "lightgbm"), lambdas=LAMBDA_GRID,
                  max_hours: float | None = None) -> pd.DataFrame:
    """T5: the maturity sweep. W2's flattening claim is read off this table."""
    units = [{"dataset": d, "model": m, "seed": s, "fold": f, "lam": lam,
              "stage": "latency"}
             for d in datasets for m in models for lam in lambdas
             for s in seeds for f in folds]
    df = run_resumable(units, fit_one, _ledger("latency"), max_hours=max_hours,
                       label="T5-latency")
    save_table(df, "T5_v2", "Label-maturity sweep, all datasets")
    return df


def stage_ablation(datasets: tuple[str, ...], seeds=(0, 1, 2), folds=(0, 1, 2, 3),
                   max_hours: float | None = None) -> pd.DataFrame:
    """C2-C5 on/off, plus the H1 test (constraint value as a function of lambda).

    C1 is deliberately absent: censored-label learning needs an imposed reporting
    delay to ablate against, so it has its own stage (``stage_censoring``) rather
    than a row here.
    """
    variants = [
        dict(variant="full"),
        dict(variant="no_hop2", hops=1),                       # C2 off
        dict(variant="no_variance", variance=False),           # C3 off
        dict(variant="no_dollar_cost", dollar_cost=False),     # C4 off
        dict(variant="no_monotone", monotone=False),           # C5 off
        dict(variant="global_monotone", region_margin=1e9),    # C5 -> v1 behaviour
        # `no_risk` switches off the whole Block B pass, so it removes the
        # association-risk features *and* the entity-memory and velocity features
        # that the same pass emits. It is therefore "Block B entirely off", not
        # "risk columns off"; `no_memory` is the narrower ablation that keeps the
        # pass and drops only the mem_ columns.
        dict(variant="no_risk", use_risk=False),
        dict(variant="no_memory", use_memory=False),
    ]
    units = [{**v, "dataset": d, "seed": s, "fold": f, "model": "halo",
              "lam": 30, "stage": "ablation"}
             for d in datasets for v in variants for s in seeds for f in folds]
    # H1: does the monotone constraint earn more as labels get noisier?
    units += [{"variant": f"h1_mono{int(mono)}", "monotone": mono, "lam": lam,
               "dataset": d, "seed": s, "fold": f, "model": "halo", "stage": "ablation"}
              for d in datasets for lam in (0, 7, 30) for mono in (True, False)
              for s in seeds for f in folds]

    df = run_resumable(units, fit_one, _ledger("ablation"), max_hours=max_hours,
                       label="T4-ablation")
    save_table(df, "T4_v2", "Ablations and the H1 constraint-vs-noise test")
    try:
        summary = summarise_ablation(df)
        if not summary.empty:
            save_table(summary, "T4_v2_paired", "Paired ablation deltas against `full`")
            print("\n" + summary.to_string(index=False))
    except Exception as e:                      # analysis must never lose the run
        print(f"  [warn] ablation summary failed: {type(e).__name__}: {e}")
    return df


def summarise_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of each ablation variant against ``full``.

    The design is paired: every variant is fitted on the identical
    (dataset, seed, fold) cells, so the seed-to-seed spread that dominates the
    raw means is shared and cancels. Comparing unpaired means instead throws that
    away and makes every effect look like noise -- which is how an ablation table
    ends up reporting a ranking that is really sampling variation, the exact
    failure this project already hit once in v1 (see ``config.lgbm_params``).

    Reported per variant: the mean paired delta, the standard deviation *of the
    deltas*, how many of the paired cells moved in the same direction, and the
    region-fallback rate. ``wins`` is a sign count, not a p-value; with a handful
    of seeds nothing here is significant and the table should not pretend it is.
    """
    need = {"u_variant", "auprc"}
    if df is None or df.empty or not need.issubset(df.columns):
        return pd.DataFrame()

    d = df[df["u_variant"].notna()].copy()
    d = d[~d["u_variant"].astype(str).str.startswith("h1_")]
    if "full" not in set(d["u_variant"]):
        return pd.DataFrame()

    cells = [c for c in ("u_dataset", "u_seed", "u_fold") if c in d.columns]
    metrics = [m for m in ("auprc", "dollar_recall_at_k", "alert_precision_at_k")
               if m in d.columns]

    base = d[d["u_variant"] == "full"].set_index(cells)[metrics]
    rows = []
    for variant, grp in d.groupby("u_variant"):
        if variant == "full":
            continue
        g = grp.set_index(cells)[metrics]
        common = base.index.intersection(g.index)
        if len(common) == 0:
            continue
        delta = g.loc[common] - base.loc[common]
        rec = {"variant": variant, "n_pairs": int(len(common))}
        for m in metrics:
            dm = delta[m].dropna()
            rec[f"d_{m}"] = round(float(dm.mean()), 5) if len(dm) else float("nan")
            rec[f"sd_{m}"] = round(float(dm.std(ddof=1)), 5) if len(dm) > 1 else float("nan")
        rec["wins_auprc"] = int((delta["auprc"] > 0).sum())
        if "region_fallback" in grp.columns:
            # NaN means the region machinery never ran (enforce=False, or no
            # monotone features survived the ablation) -- which is not the same
            # as "ran and did not fall back". Collapsing the two with fillna
            # would report a variant that never built a region as if it had
            # built a healthy one.
            fbv = grp["region_fallback"].dropna()
            rec["fallback_rate"] = (round(float(fbv.astype(bool).mean()), 2)
                                    if len(fbv) else "n/a")
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("d_auprc")

    # A degenerate alert region means the specialist was trained on everything, so
    # the C5 variants are not measuring what their names say. Better to say so
    # here than to let the number reach a table.
    fb = d[d["u_variant"] == "full"]
    if "region_fallback" in fb.columns:
        rate = float(fb["region_fallback"].fillna(False).astype(bool).mean())
        if rate > 0:
            print(f"\n  [warn] the `full` variant fell back to global training in "
                  f"{rate:.0%} of cells: the alert region was degenerate "
                  f"(<200 rows or <5 positives).\n"
                  f"         C5 rows (`no_monotone`, `global_monotone`) do not mean "
                  f"what their labels say for those cells.\n"
                  f"         This is a fold-size problem, not a code fault -- use a "
                  f"dataset or fold large enough to populate the top "
                  f"{CFG.review_budget_k:.0%} of scores.")
    return out


def stage_censoring(datasets: tuple[str, ...], seeds=(0, 1, 2),
                    delay_scales=(3.0, 7.0, 21.0), max_hours: float | None = None
                    ) -> pd.DataFrame:
    """C1: drop vs zero vs soft, under an imposed reporting-delay distribution.

    The benchmark ships final labels, so the delay is simulated and reported
    alongside the result -- see ``censoring`` module docstring.
    """
    units = [{"dataset": d, "seed": s, "delay_scale": ds, "clustered": c,
              "mode": mode, "stage": "censoring", "fold": 0, "lam": 30}
             for d in datasets for ds in delay_scales for c in (True, False)
             for mode in ("drop", "zero", "soft") for s in seeds]

    def run_unit(u: dict) -> dict:
        bundle = get_bundle(u["dataset"])
        df = bundle.df
        folds = rolling_origin_folds(df, bundle.entity, latency_delta=0,
                                     entity_disjoint=True)
        if not folds:
            return {"status": "no_folds"}
        fold = folds[0]
        X = build_features(bundle, fold, FeatureConfig(lam=0))
        y = df["y"].to_numpy()

        model = DelayModel(scale_days=u["delay_scale"],
                           horizon_days=bundle.spec.horizon_days or 120.0,
                           cluster_by_entity=u["clustered"])
        obs_day = float(df["day"].to_numpy()[fold.train_idx].max())
        rep = simulate_report_days(df, model, seed=u["seed"])
        y_vis, reported, age = visible_labels(df, rep, obs_day)

        tr, te = fold.train_idx, fold.test_idx
        amount = df["amount"].to_numpy() if bundle.spec.has_amount else None

        def fit_fn(Xi, yi, wi):
            m = RegionMonotoneGBDT(
                monotone_features=AssociationRiskV2.monotone_feature_names(Xi.columns),
                seed=u["seed"])
            m.fit(Xi, yi, sample_weight=wi)
            return m

        trainer = CensoredLabelTrainer(
            fit_fn, lambda m, Xi: m.decision_score(Xi), model,
            CensoringConfig(mode=u["mode"]))
        fitted = trainer.fit(X.iloc[tr], y_vis[tr], reported[tr], age[tr],
                             base_weight=dollar_weights(
                                 y_vis[tr], None if amount is None else amount[tr],
                                 kappa=CFG.fn_cost_ratio))

        res = evaluate(y[te], fitted.decision_score(X.iloc[te]),
                       amount=None if amount is None else amount[te],
                       entity=bundle.entity[te],
                       cold_mask=cold_entity_mask(bundle.entity, fold), seed=u["seed"])
        res.update({
            "delay": model.describe(),
            "censored_positive_mass": int(y[tr].sum() - y_vis[tr].sum()),
            "observation_day": obs_day,
            **({"em_last": trainer.history_[-1]} if trainer.history_ else {}),
        })
        return res

    df = run_resumable(units, run_unit, _ledger("censoring"), max_hours=max_hours,
                       label="C1-censoring")
    save_table(df, "C1_v2", "Censored-label strategies under simulated delay")
    return df


STAGES = {
    "ladder": stage_ladder,
    "benchmark": stage_benchmark,
    "latency": stage_latency,
    "ablation": stage_ablation,
    "censoring": stage_censoring,
}


def available_datasets(candidates: tuple[str, ...] = ALL_DATASETS) -> list[str]:
    """Which datasets are actually present on disk. Missing ones are skipped, loudly."""
    ok = []
    for name in candidates:
        try:
            load_dataset(name, nrows=5)
            ok.append(name)
        except DatasetNotAvailable as e:
            print(f"  [skip] {e}")
        except Exception as e:                       # malformed file, wrong layout
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
    return ok
