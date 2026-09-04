"""Block C -- missingness-regime mining.

The team's original "missingness as signal" idea, upgraded from indicator flags into
actual pattern mining. The V-columns arrive in blocks that go missing together; that
block structure is a fingerprint of which upstream enrichment service fired for a given
transaction. Recovering it gives a latent *data-provenance* variable.

Three uses, and the third is the one that matters:
  1. a categorical feature
  2. a stratum for per-regime probability calibration
  3. a LABEL-FREE drift monitor -- the regime distribution is observable the moment a
     transaction lands, while its label takes up to 120 days. Under verification latency
     you cannot monitor drift with labels, so this is not a nicety, it is the only option.

Leakage note (adversarial review item 2)
----------------------------------------
Mining patterns over the full dataset, test folds included, is transductive leakage.
This module is therefore a fit/transform estimator: ``fit`` sees the training window
only, and ``transform`` assigns unseen patterns to the nearest known regime by Hamming
distance. ``fit_transform`` on the full frame is deliberately NOT provided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG


class RegimeMiner:
    """Mines closed co-missingness patterns and assigns provenance regimes.

    Each row's missingness pattern over the mined columns is an itemset (the set of
    columns that are missing). Grouping identical patterns yields *closed* itemsets
    directly -- a pattern's support cannot be increased without changing the itemset --
    so retaining the patterns above ``min_support`` is exactly closed frequent itemset
    mining for this transaction database, without the exponential candidate search.
    """

    def __init__(self, min_support: float | None = None, max_regimes: int | None = None):
        self.min_support = min_support if min_support is not None else CFG.regime_min_support
        self.max_regimes = max_regimes or CFG.regime_max_count
        self.columns_: list[str] = []
        self.patterns_: np.ndarray | None = None   # (n_regimes, n_cols) uint8
        self.support_: np.ndarray | None = None
        self.column_blocks_: dict[str, list[str]] = {}
        self.fitted_ = False

    # ---- fitting -------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "RegimeMiner":
        cols = columns or self._minable_columns(df)
        self.columns_ = cols
        M = df[cols].isna().to_numpy(dtype=np.uint8)

        # Column blocks: columns whose missingness vectors are identical are one
        # upstream service. This recovers the V-block structure as a by-product.
        self.column_blocks_ = self._detect_column_blocks(M, cols)

        # Deduplicate columns to one representative per block before mining patterns:
        # otherwise a 300-column block dominates the Hamming metric 300 : 1.
        reps = [members[0] for members in self.column_blocks_.values()]
        self.rep_idx_ = np.array([cols.index(r) for r in reps])
        Mr = M[:, self.rep_idx_]

        keys, inverse, counts = np.unique(Mr, axis=0, return_inverse=True,
                                          return_counts=True)
        support = counts / len(df)
        keep = support >= self.min_support
        if keep.sum() > self.max_regimes:
            top = np.argsort(-support)[:self.max_regimes]
            keep = np.zeros_like(keep)
            keep[top] = True
        if not keep.any():          # degenerate data: fall back to a single regime
            keep[np.argmax(support)] = True

        self.patterns_ = keys[keep]
        self.support_ = support[keep]
        self.n_raw_patterns_ = int(len(keys))
        self.covered_support_ = float(support[keep].sum())
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Assign each row to its regime; unseen patterns go to the nearest by Hamming."""
        if not self.fitted_:
            raise RuntimeError("RegimeMiner.transform called before fit")
        M = df[self.columns_].isna().to_numpy(dtype=np.uint8)[:, self.rep_idx_]
        # (n_rows, n_regimes) Hamming distances. Regime count is capped, so this is small.
        dist = (M[:, None, :] != self.patterns_[None, :, :]).sum(axis=2)
        return pd.Series(dist.argmin(axis=1).astype(np.int16),
                         index=df.index, name="regime")

    # ---- features ------------------------------------------------------------------
    def missingness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One indicator per column block, plus the regime id and a density summary."""
        out = pd.DataFrame(index=df.index)
        for name, members in self.column_blocks_.items():
            out[f"miss_blk_{name}"] = df[members[0]].isna().astype(np.int8)
        out["miss_density"] = df[self.columns_].isna().mean(axis=1).astype(np.float32)
        out["regime"] = self.transform(df).to_numpy()
        return out

    # ---- drift (F3) ------------------------------------------------------------------
    def regime_distribution(self, df: pd.DataFrame) -> np.ndarray:
        r = self.transform(df).to_numpy()
        counts = np.bincount(r, minlength=len(self.patterns_)).astype(float)
        return counts / max(counts.sum(), 1)

    def drift_series(self, df: pd.DataFrame, day_col: str = "day",
                     window: int = 7) -> pd.DataFrame:
        """Label-free drift signal: JS divergence of each window against the reference.

        The reference is the first window, standing in for "the distribution the model
        was trained on". No label is read anywhere in this function.
        """
        r = self.transform(df).to_numpy()
        days = df[day_col].to_numpy()
        bins = (days - days.min()) // window
        n_reg = len(self.patterns_)
        rows = []
        ref = None
        for b in np.unique(bins):
            m = bins == b
            p = np.bincount(r[m], minlength=n_reg).astype(float)
            p = p / max(p.sum(), 1)
            if ref is None:
                ref = p
            rows.append({
                "window": int(b),
                "day_start": int(days[m].min()),
                "n": int(m.sum()),
                "js_divergence": float(_js(p, ref)),
                "top_regime": int(np.argmax(p)),
            })
        return pd.DataFrame(rows)

    # ---- internals -------------------------------------------------------------------
    @staticmethod
    def _minable_columns(df: pd.DataFrame) -> list[str]:
        skip = {"TransactionID", "isFraud", "TransactionDT", "day", "hour", "dow",
                "entity", "uid", "D1n", "D15n", "true_entity", "is_index",
                "true_regime", "within", "regime"}
        cols = [c for c in df.columns if c not in skip]
        na = df[cols].isna().mean()
        # A column that is never missing, or always missing, carries no pattern.
        return [c for c in cols if 0.0 < na[c] < 1.0]

    @staticmethod
    def _detect_column_blocks(M: np.ndarray, cols: list[str]) -> dict[str, list[str]]:
        """Group columns with byte-identical missingness vectors."""
        blocks: dict[bytes, list[str]] = {}
        for j, c in enumerate(cols):
            blocks.setdefault(M[:, j].tobytes(), []).append(c)
        return {members[0]: members for members in blocks.values()}

    def summary(self) -> dict:
        return {
            "n_columns_mined": len(self.columns_),
            "n_column_blocks": len(self.column_blocks_),
            "largest_block_size": max((len(v) for v in self.column_blocks_.values()),
                                      default=0),
            "n_raw_patterns": getattr(self, "n_raw_patterns_", 0),
            "n_regimes": int(len(self.patterns_)) if self.patterns_ is not None else 0,
            "support_covered_by_regimes": getattr(self, "covered_support_", float("nan")),
            "min_support": self.min_support,
        }


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, with a floor to avoid log(0)."""
    eps = 1e-12
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
