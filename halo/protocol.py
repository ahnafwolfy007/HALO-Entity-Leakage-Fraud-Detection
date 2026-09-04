"""The Cold-Entity Protocol (CEP), and the leaky protocols it is contrasted against.

A fold is defined by four independent switches, which is what lets T1 close one leakage
channel at a time while holding everything else fixed:

    random_split      True  -> L2 open  (shuffle a time-ordered stream)
    past_only_encoding False -> L2 open  (encodings fitted over the whole period)
    entity_disjoint   False -> L3 open  (same card in train and test)
    latency_delta     0     -> L4 open  (labels used before they could exist)

L1 (resampling leakage) is handled at fit time in ``baselines.py``, because it is a
property of where SMOTE sits relative to the split, not of the split itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG, Rung


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: int
    test_day_start: int
    test_day_end: int
    n_train: int
    n_test: int
    train_base_rate: float
    test_base_rate: float
    dropped_entity_rows: int = 0
    label_maturity_dropped: int = 0

    def meta(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "test_day_start": self.test_day_start,
            "test_day_end": self.test_day_end,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_base_rate": self.train_base_rate,
            "test_base_rate": self.test_base_rate,
            "dropped_entity_rows": self.dropped_entity_rows,
            "label_maturity_dropped": self.label_maturity_dropped,
        }


def rolling_origin_folds(df: pd.DataFrame, entity: np.ndarray,
                         latency_delta: int = 0, entity_disjoint: bool = True,
                         fold_days: int | None = None,
                         min_train_days: int | None = None) -> list[Fold]:
    """CEP folds: forward in time, entity-disjoint, latency-gated.

    For a test fold starting on day S, a training row on day d is admissible only if its
    label had matured before the fold opened -- that is, ``d + delta < S``.
    """
    fold_days = fold_days or CFG.fold_days
    min_train_days = min_train_days or CFG.min_train_days
    day = df["day"].to_numpy()
    y = df["isFraud"].to_numpy()
    d0, d1 = int(day.min()), int(day.max())

    folds: list[Fold] = []
    fid = 0
    start = d0 + min_train_days
    while start <= d1:
        end = min(start + fold_days, d1 + 1)
        test_mask = (day >= start) & (day < end)
        if test_mask.sum() < 50:
            start = end
            continue

        # Label maturity: d + delta < S
        raw_train = day < start
        mature = day < (start - latency_delta)
        train_mask = raw_train & mature
        maturity_dropped = int((raw_train & ~mature).sum())

        dropped_entity_rows = 0
        if entity_disjoint:
            test_ents = np.unique(entity[test_mask])
            overlap = np.isin(entity, test_ents) & train_mask
            dropped_entity_rows = int(overlap.sum())
            train_mask = train_mask & ~overlap

        if train_mask.sum() < 200 or y[train_mask].sum() < 5:
            start = end
            continue

        folds.append(Fold(
            train_idx=np.flatnonzero(train_mask),
            test_idx=np.flatnonzero(test_mask),
            fold_id=fid,
            test_day_start=int(start), test_day_end=int(end),
            n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
            train_base_rate=float(y[train_mask].mean()),
            test_base_rate=float(y[test_mask].mean()),
            dropped_entity_rows=dropped_entity_rows,
            label_maturity_dropped=maturity_dropped,
        ))
        fid += 1
        start = end
    return folds


def random_folds(df: pd.DataFrame, entity: np.ndarray, n_splits: int = 3,
                 test_frac: float = 0.25, seed: int = 0) -> list[Fold]:
    """The leaky baseline: shuffle a time-ordered stream. Rungs 0 and 1 only."""
    rng = np.random.default_rng(seed)
    n = len(df)
    y = df["isFraud"].to_numpy()
    day = df["day"].to_numpy()
    folds = []
    for k in range(n_splits):
        perm = rng.permutation(n)
        cut = int(n * (1 - test_frac))
        tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])
        folds.append(Fold(
            train_idx=tr, test_idx=te, fold_id=k,
            test_day_start=int(day[te].min()), test_day_end=int(day[te].max()),
            n_train=len(tr), n_test=len(te),
            train_base_rate=float(y[tr].mean()), test_base_rate=float(y[te].mean()),
        ))
    return folds


def folds_for_rung(df: pd.DataFrame, entity: np.ndarray, rung: Rung,
                   seed: int = 0) -> list[Fold]:
    if rung.random_split:
        return random_folds(df, entity, seed=seed)
    return rolling_origin_folds(df, entity,
                                latency_delta=rung.latency_delta,
                                entity_disjoint=rung.entity_disjoint)


def cold_entity_mask(entity: np.ndarray, fold: Fold) -> np.ndarray:
    """Boolean mask over the *test* rows: entities never seen in this fold's training set."""
    train_ents = np.unique(entity[fold.train_idx])
    return ~np.isin(entity[fold.test_idx], train_ents)


def index_event_mask(df: pd.DataFrame, entity: np.ndarray) -> np.ndarray:
    """Mark each entity's first positive in time as its index event.

    Inferred from the data alone, exactly as a researcher without ground truth would.
    Entities whose history is truncated by the window edge are handled by the caller:
    an entity already positive on its first observed transaction is still counted, but
    ``entity_stats`` reports how many such cases exist so the report can qualify them.
    """
    y = df["isFraud"].to_numpy()
    order = np.lexsort((df["TransactionDT"].to_numpy(), entity))
    out = np.zeros(len(df), dtype=bool)
    seen: set = set()
    for i in order:
        if y[i] == 1:
            e = entity[i]
            if e not in seen:
                seen.add(e)
                out[i] = True
    return out


def left_truncated_entities(df: pd.DataFrame, entity: np.ndarray) -> int:
    """Entities whose first observed transaction is already positive.

    Their true index event may lie before the observation window, so their TTD is a
    lower bound. Reported as a caveat rather than silently dropped.
    """
    y = df["isFraud"].to_numpy()
    order = np.lexsort((df["TransactionDT"].to_numpy(), entity))
    first_seen: dict = {}
    for i in order:
        e = entity[i]
        if e not in first_seen:
            first_seen[e] = y[i]
    return int(sum(v == 1 for v in first_seen.values()))
