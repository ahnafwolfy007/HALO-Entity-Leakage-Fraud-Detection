"""Experiment orchestration: T1-T7, F2, F3.

The one rule everything here obeys: **Blocks B and C are estimators, fitted per fold on
training rows only.** Building association risk or provenance regimes once over the whole
frame and reusing them across folds would leak, and it would leak in exactly the way the
paper accuses the field of leaking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, LADDER, Rung
from .data import add_base_features, feature_columns, prepare_matrix
from .entities import (assert_uid_is_label_free, entity_stats, resolve_entities,
                       add_d1n)
from .io import Timer, save_table
from .metrics import aggregate_seeds, evaluate, time_to_detection
from .model import HaloModel
from .protocol import (Fold, cold_entity_mask, folds_for_rung, index_event_mask,
                       left_truncated_entities, rolling_origin_folds)
from .regimes import RegimeMiner
from .risk import AssociationRisk
from . import baselines as B


# --------------------------------------------------------------------------------------
# Feature assembly, fold-aware
# --------------------------------------------------------------------------------------

def build_fold_features(df: pd.DataFrame, entity: np.ndarray, fold: Fold,
                        latency_delta: int = 0, past_only: bool = True,
                        use_risk: bool = True, use_regimes: bool = True,
                        ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return ``(X, regime, info)`` for the whole frame, fitted on this fold's train rows.

    ``past_only=False`` reproduces L2: encodings and risk counters are fitted over the
    entire period, test rows included, exactly as a careless pipeline would.
    """
    info: dict = {}
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[fold.train_idx] = True

    parts = [df]
    regime = np.zeros(len(df), dtype=np.int16)

    if use_regimes:
        with Timer("Block C: regime mining"):
            miner = RegimeMiner()
            miner.fit(df.iloc[fold.train_idx] if past_only else df)
            mf = miner.missingness_features(df)
            regime = mf["regime"].to_numpy()
            parts.append(mf.drop(columns=["regime"]))
            info["regimes"] = miner.summary()
            info["_miner"] = miner

    if use_risk:
        with Timer("Block B: association risk"):
            order = np.argsort(df["TransactionDT"].to_numpy(), kind="stable")
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            dfs = df.iloc[order]
            ent_s = entity[order]
            ingest = train_mask[order] if past_only else np.ones(len(df), dtype=bool)

            ar = AssociationRisk(delta_days=latency_delta)
            ar.fit_prior(df.iloc[fold.train_idx], df["isFraud"].to_numpy()[fold.train_idx])
            rf = ar.transform(dfs, ent_s, ingest_mask=ingest)
            rf = rf.iloc[inv].reset_index(drop=True)
            rf.index = df.index
            parts.append(rf)
            info["risk_attributes"] = list(ar.priors.keys())

    wide = pd.concat(parts, axis=1)
    X, names = prepare_matrix(wide)
    assert_uid_is_label_free(names)
    info["n_features"] = X.shape[1]
    return X, regime, info


def _monotone_names(X: pd.DataFrame) -> list[str]:
    return AssociationRisk.risk_feature_names(X.columns)


# --------------------------------------------------------------------------------------
# One fold, one model
# --------------------------------------------------------------------------------------

def run_fold(df: pd.DataFrame, entity: np.ndarray, fold: Fold, model_name: str,
             *, latency_delta: int = 0, past_only: bool = True,
             resample_before_split: bool = False, seed: int = 0,
             use_risk: bool = True, use_regimes: bool = True,
             use_memory: bool = True, enforce_monotone: bool = True,
             is_index_all: np.ndarray | None = None,
             tuning_budget: int = 0) -> dict:
    """Fit one model on one fold and return every metric plus provenance."""
    X, regime, finfo = build_fold_features(
        df, entity, fold, latency_delta=latency_delta, past_only=past_only,
        use_risk=use_risk, use_regimes=use_regimes)

    if not use_memory:
        drop = [c for c in X.columns if c.startswith("mem_")]
        X = X.drop(columns=drop)

    y = df["isFraud"].to_numpy()
    tr, te = fold.train_idx, fold.test_idx

    if resample_before_split:
        # L1 open: oversample the whole frame *then* honour the split, so synthetic
        # points interpolated from test-fold minority rows land in training.
        Xr, yr = B.naive_smote(X, y, seed=seed)
        n0 = len(X)
        extra = np.arange(n0, len(Xr))
        X_tr = pd.concat([X.iloc[tr], Xr.iloc[extra]], ignore_index=True)
        y_tr = np.concatenate([y[tr], yr[extra]])
        resample_in_fold = False
    else:
        X_tr, y_tr = X.iloc[tr], y[tr]
        resample_in_fold = False

    X_te = X.iloc[te]

    if model_name == "halo":
        mono = _monotone_names(X_tr) if enforce_monotone else []
        m = HaloModel(monotone_features=mono, seed=seed,
                      enforce_monotone=enforce_monotone)
        m.fit(X_tr, y_tr, regime=regime[tr] if len(regime) == len(df) else None)
        s = m.decision_score(X_te)
        p = m.predict_proba(X_te, regime=regime[te] if len(regime) == len(df) else None)
        info = {"model": "halo", "n_monotone": len(mono)}
        model_obj = m
    else:
        s, info = B.fit_predict(model_name, X_tr, y_tr, X_te, seed=seed,
                                resample=resample_in_fold, tuning_budget=tuning_budget)
        p = s
        model_obj = None

    idx_mask = (is_index_all[te] if is_index_all is not None else None)
    cold = cold_entity_mask(entity, fold)

    res = evaluate(y[te], s, amount=df["TransactionAmt"].to_numpy()[te],
                   entity=entity[te], is_index=idx_mask, cold_mask=cold,
                   proba=p, seed=seed)
    if idx_mask is not None:
        ttd = time_to_detection(df.iloc[te], entity[te], s, idx_mask)
        ttd.pop("_ttd_frame", None)
        res.update(ttd)

    res.update(fold.meta())
    res.update({k: v for k, v in info.items() if not k.startswith("_")})
    res["n_features"] = finfo["n_features"]
    res["seed"] = seed
    res["_model"] = model_obj
    res["_X_test"] = X_te if model_name == "halo" else None
    return res


# --------------------------------------------------------------------------------------
# T1 -- the leakage ladder
# --------------------------------------------------------------------------------------

def run_ladder(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
               seeds: tuple[int, ...] | None = None,
               model_name: str = "lightgbm") -> pd.DataFrame:
    """Close one leakage channel per rung, holding the model constant."""
    seeds = seeds or CFG.seeds
    rows = []
    for rung in LADDER:
        for seed in seeds:
            folds = folds_for_rung(df, entity, rung, seed=seed)
            if not folds:
                continue
            for fold in folds:
                r = run_fold(df, entity, fold, model_name,
                             latency_delta=rung.latency_delta,
                             past_only=rung.past_only_encoding,
                             resample_before_split=rung.resample_before_split,
                             seed=seed, is_index_all=is_index_all)
                r = {k: v for k, v in r.items() if not k.startswith("_")}
                r.update({"rung": rung.key, "rung_label": rung.label})
                rows.append(r)
    raw = pd.DataFrame(rows)
    save_table(raw, "T1_raw", "Leakage ladder, per fold and seed")
    agg = aggregate_seeds(rows, keys=["rung", "rung_label"])
    keep = [c for c in agg.columns
            if c in {"rung", "rung_label"} or c.startswith(
                ("auprc", "auroc", "index_auprc", "cold_entity_auprc",
                 "dollar_recall", "base_rate"))]
    agg = agg[keep]
    save_table(agg, "T1", "Leakage ladder (mean +/- std across seeds and folds)")
    return agg


# --------------------------------------------------------------------------------------
# T2 -- entity statistics, the go/no-go
# --------------------------------------------------------------------------------------

def run_entity_stats(df: pd.DataFrame, truth: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    rows = []
    for variant in ("loose", "medium", "strict"):
        ent = resolve_entities(df, variant=variant)
        st = entity_stats(df, ent, truth)
        st["uid_variant"] = variant
        st["conflict_rejects"] = ent.attrs.get("conflict_rejects", 0)
        st["score_rejects"] = ent.attrs.get("score_rejects", 0)
        st["left_truncated_entities"] = left_truncated_entities(df, ent.to_numpy())
        rows.append(st)
    out = pd.DataFrame(rows)
    save_table(out, "T2", "Entity statistics and label purity (UID sensitivity)")
    return out


# --------------------------------------------------------------------------------------
# T3 -- main results under CEP
# --------------------------------------------------------------------------------------

def run_main(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
             models: list[str] | None = None, seeds: tuple[int, ...] | None = None,
             delta: int | None = None, tuning_budget: int = 0) -> pd.DataFrame:
    models = models or (B.AVAILABLE_BASELINES + ["halo"])
    seeds = seeds or CFG.seeds
    delta = CFG.headline_delta if delta is None else delta
    folds = rolling_origin_folds(df, entity, latency_delta=delta, entity_disjoint=True)
    rows = []
    for name in models:
        for seed in seeds:
            for fold in folds:
                try:
                    r = run_fold(df, entity, fold, name, latency_delta=delta,
                                 past_only=True, seed=seed,
                                 is_index_all=is_index_all,
                                 tuning_budget=tuning_budget)
                except Exception as exc:                      # keep the grid alive
                    rows.append({"model": name, "seed": seed, "fold_id": fold.fold_id,
                                 "error": str(exc)[:200]})
                    continue
                rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    raw = pd.DataFrame(rows)
    save_table(raw, "T3_raw", "Main CEP results, per fold and seed")
    agg = aggregate_seeds([r for r in rows if "error" not in r], keys=["model"])
    save_table(agg, "T3", "Main results under the Cold-Entity Protocol")
    return agg


# --------------------------------------------------------------------------------------
# T4 -- HALO ablation
# --------------------------------------------------------------------------------------

ABLATIONS = [
    ("full",            dict(use_risk=True,  use_regimes=True,  use_memory=True,  enforce_monotone=True)),
    ("no_riskB",        dict(use_risk=False, use_regimes=True,  use_memory=True,  enforce_monotone=True)),
    ("no_regimesC",     dict(use_risk=True,  use_regimes=False, use_memory=True,  enforce_monotone=True)),
    ("no_entity_memory", dict(use_risk=True, use_regimes=True,  use_memory=False, enforce_monotone=True)),
    ("no_monotoneD",    dict(use_risk=True,  use_regimes=True,  use_memory=True,  enforce_monotone=False)),
    ("behaviour_only",  dict(use_risk=False, use_regimes=False, use_memory=False, enforce_monotone=False)),
]


def run_ablation(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
                 seeds: tuple[int, ...] | None = None,
                 delta: int | None = None) -> pd.DataFrame:
    seeds = seeds or CFG.seeds
    delta = CFG.headline_delta if delta is None else delta
    folds = rolling_origin_folds(df, entity, latency_delta=delta, entity_disjoint=True)
    rows = []
    for name, kw in ABLATIONS:
        for seed in seeds:
            for fold in folds:
                r = run_fold(df, entity, fold, "halo", latency_delta=delta,
                             past_only=True, seed=seed, is_index_all=is_index_all, **kw)
                r = {k: v for k, v in r.items() if not k.startswith("_")}
                r["ablation"] = name
                rows.append(r)
    save_table(pd.DataFrame(rows), "T4_raw", "HALO ablation, per fold and seed")
    agg = aggregate_seeds(rows, keys=["ablation"])
    save_table(agg, "T4", "HALO block ablation")
    return agg


# --------------------------------------------------------------------------------------
# T5 -- latency sensitivity
# --------------------------------------------------------------------------------------

def run_latency_sweep(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
                      models: list[str] | None = None,
                      seeds: tuple[int, ...] | None = None,
                      deltas: tuple[int, ...] | None = None) -> pd.DataFrame:
    models = models or ["lightgbm", "halo"]
    seeds = seeds or CFG.seeds
    deltas = deltas or CFG.latency_deltas
    rows = []
    for delta in deltas:
        folds = rolling_origin_folds(df, entity, latency_delta=delta,
                                     entity_disjoint=True)
        if not folds:
            rows.append({"delta_days": delta, "note": "no admissible folds at this delta"})
            continue
        for name in models:
            for seed in seeds:
                for fold in folds:
                    r = run_fold(df, entity, fold, name, latency_delta=delta,
                                 past_only=True, seed=seed, is_index_all=is_index_all)
                    r = {k: v for k, v in r.items() if not k.startswith("_")}
                    r.update({"delta_days": delta, "model": name})
                    rows.append(r)
    save_table(pd.DataFrame(rows), "T5_raw", "Latency sweep, per fold and seed")
    agg = aggregate_seeds([r for r in rows if "note" not in r],
                          keys=["delta_days", "model"])
    save_table(agg, "T5", "Latency sensitivity")
    return agg


# --------------------------------------------------------------------------------------
# T7 -- operating cost and throughput
# --------------------------------------------------------------------------------------

def run_cost_and_throughput(df: pd.DataFrame, entity: np.ndarray,
                            is_index_all: np.ndarray, seed: int = 0) -> pd.DataFrame:
    import time
    folds = rolling_origin_folds(df, entity, latency_delta=CFG.headline_delta,
                                 entity_disjoint=True)
    rows = []
    for name in (B.AVAILABLE_BASELINES + ["halo"]):
        for fold in folds[:1]:
            t0 = time.perf_counter()
            try:
                r = run_fold(df, entity, fold, name, latency_delta=CFG.headline_delta,
                             past_only=True, seed=seed, is_index_all=is_index_all)
            except Exception as exc:
                rows.append({"model": name, "error": str(exc)[:200]})
                continue
            dt = time.perf_counter() - t0
            rows.append({
                "model": name,
                "dollar_recall_at_k": r.get("dollar_recall_at_k"),
                "alert_precision_at_k": r.get("alert_precision_at_k"),
                "count_recall_at_k": r.get("count_recall_at_k"),
                "auprc": r.get("auprc"),
                "fit_score_seconds": round(dt, 2),
                "microseconds_per_txn": round(dt / max(fold.n_train + fold.n_test, 1) * 1e6, 1),
            })
    out = pd.DataFrame(rows)
    save_table(out, "T7", "Operating cost and throughput")
    return out


# --------------------------------------------------------------------------------------
# F3 -- label-free drift
# --------------------------------------------------------------------------------------

def run_drift(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
              seed: int = 0) -> pd.DataFrame:
    """Regime-distribution drift against per-fold AUPRC decay. No labels in the signal."""
    folds = rolling_origin_folds(df, entity, latency_delta=CFG.headline_delta,
                                 entity_disjoint=True)
    if not folds:
        return pd.DataFrame()
    miner = RegimeMiner().fit(df.iloc[folds[0].train_idx])
    drift = miner.drift_series(df)

    perf = []
    for fold in folds:
        r = run_fold(df, entity, fold, "lightgbm",
                     latency_delta=CFG.headline_delta, past_only=True, seed=seed,
                     is_index_all=is_index_all)
        perf.append({"day_start": fold.test_day_start, "auprc": r["auprc"],
                     "auprc_lift": r["auprc_lift"], "base_rate": r["test_base_rate"]})
    perf = pd.DataFrame(perf)

    drift["nearest_fold_day"] = drift["day_start"].apply(
        lambda d: perf["day_start"].iloc[(perf["day_start"] - d).abs().argmin()]
        if len(perf) else np.nan)
    out = drift.merge(perf, left_on="nearest_fold_day", right_on="day_start",
                      how="left", suffixes=("", "_fold"))
    save_table(out, "F3", "Label-free regime drift vs AUPRC decay")
    return out


# --------------------------------------------------------------------------------------
# T1b -- the training-size control (adversarial review item 13)
# --------------------------------------------------------------------------------------

def run_size_control(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
                     seeds: tuple[int, ...] | None = None,
                     model_name: str = "lightgbm") -> pd.DataFrame:
    """Disentangle 'removed memorisation' from 'trained on less data'.

    Entity-disjoint splitting drops every training row belonging to a test entity, which
    shrinks the training set as well as removing memorisation. The rung 2 -> rung 3 drop
    therefore confounds two effects, and a reviewer will say so.

    The control: re-run rung 2 (entity leakage still OPEN) with the training set randomly
    subsampled to exactly the size rung 3 was left with, on the same folds.

        rung2_full     entity leakage open, full training set
        rung2_matched  entity leakage open, training set cut to rung 3's size
        rung3          entity leakage closed, full remaining training set

    Read it as: rung2_full -> rung2_matched is the price of less data.
                rung2_matched -> rung3 is what entity leakage was actually worth.

    If rung2_matched sits close to rung2_full, the collapse is leakage and the headline
    claim stands. If it falls most of the way to rung3, a large part of the drop was
    sample size and the paper must say so.
    """
    seeds = seeds or CFG.seeds
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        open_folds = rolling_origin_folds(df, entity, latency_delta=0,
                                          entity_disjoint=False)
        closed_folds = rolling_origin_folds(df, entity, latency_delta=0,
                                            entity_disjoint=True)
        by_id = {f.fold_id: f for f in closed_folds}

        for fold in open_folds:
            closed = by_id.get(fold.fold_id)
            if closed is None:
                continue

            # rung 2: entity leakage open, full training set
            r_full = run_fold(df, entity, fold, model_name, latency_delta=0,
                              past_only=True, seed=seed, is_index_all=is_index_all)

            # rung 2 matched: same condition, training set cut to rung 3's size
            n_target = min(len(closed.train_idx), len(fold.train_idx))
            sub = np.sort(rng.choice(fold.train_idx, size=n_target, replace=False))
            matched = Fold(train_idx=sub, test_idx=fold.test_idx,
                           fold_id=fold.fold_id,
                           test_day_start=fold.test_day_start,
                           test_day_end=fold.test_day_end,
                           n_train=len(sub), n_test=fold.n_test,
                           train_base_rate=float(df["isFraud"].to_numpy()[sub].mean()),
                           test_base_rate=fold.test_base_rate)
            r_match = run_fold(df, entity, matched, model_name, latency_delta=0,
                               past_only=True, seed=seed, is_index_all=is_index_all)

            # rung 3: entity leakage closed
            r_closed = run_fold(df, entity, closed, model_name, latency_delta=0,
                                past_only=True, seed=seed, is_index_all=is_index_all)

            for label, r in (("rung2_full", r_full), ("rung2_size_matched", r_match),
                             ("rung3_entity_disjoint", r_closed)):
                rec = {k: v for k, v in r.items() if not k.startswith("_")}
                rec.update({"condition": label, "seed": seed})
                rows.append(rec)

    raw = pd.DataFrame(rows)
    save_table(raw, "T1b_raw", "Training-size control, per fold and seed")
    agg = aggregate_seeds(rows, keys=["condition"])
    keep = [c for c in agg.columns
            if c == "condition" or c.startswith(("auprc", "n_train", "base_rate"))]
    agg = agg[keep]

    # Attribute the drop, so the report does not have to do arithmetic by hand.
    try:
        m = agg.set_index("condition")["auprc_mean"]
        size_cost = m["rung2_full"] - m["rung2_size_matched"]
        leak_value = m["rung2_size_matched"] - m["rung3_entity_disjoint"]
        total = m["rung2_full"] - m["rung3_entity_disjoint"]
        attribution = pd.DataFrame([{
            "total_drop_auprc": total,
            "attributable_to_training_size": size_cost,
            "attributable_to_entity_leakage": leak_value,
            "leakage_share_of_drop": leak_value / total if total else float("nan"),
        }])
        save_table(attribution, "T1b_attribution",
                   "Decomposition of the rung 2 -> rung 3 drop")
    except (KeyError, ZeroDivisionError):
        pass

    save_table(agg, "T1b", "Training-size control for the entity-leakage step")
    return agg
