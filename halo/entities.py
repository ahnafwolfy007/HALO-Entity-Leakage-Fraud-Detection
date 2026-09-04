"""Block A -- entity resolution by constraint mining.

This is the load-bearing block: L3 cannot be measured without it. Everything here is
unsupervised. The label is never read, which is what makes the C-monotonicity check a
legitimate precision proxy rather than a circular one.

The key identity
----------------
``D1`` is days since the card began transacting, so ``floor(TransactionDT/86400) - D1``
is the card's first-seen day and is *constant for the life of that card identity*.
Combining it with ``card1`` and ``addr1`` yields the UID that the Kaggle community
found by trial and error in 2019 and never published.

Leakage note (adversarial review item 1)
----------------------------------------
``D1n`` is derived from ``TransactionDT``, so it is fair to ask whether the UID smuggles
temporal information into the split. It does not: ``D1n`` is a *constant per entity*, it
is computed without reference to any label, and it is used only to group rows, never as
a model feature. ``assert_uid_is_label_free`` enforces the second half of that claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG

C_COLS = [f"C{i}" for i in range(1, 15)]

UID_RECIPES: dict[str, list[str]] = {
    # Loose: coarse grouping, high recall, risks merging distinct cards.
    "loose": ["card1", "D1n"],
    # Medium: the community's canonical key. Default.
    "medium": ["card1", "addr1", "D1n"],
    # Strict: adds the issuer fields, high precision, risks splitting one card in two.
    "strict": ["card1", "addr1", "card2", "card5", "D1n"],
}


# --------------------------------------------------------------------------------------
# Union-find for transitive closure
# --------------------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)
        self.rejected = 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


# --------------------------------------------------------------------------------------
# UID derivation
# --------------------------------------------------------------------------------------

def add_d1n(df: pd.DataFrame) -> pd.DataFrame:
    """Add the card's first-seen day. Constant per entity by construction."""
    df = df.copy()
    day = np.floor(df["TransactionDT"] / 86_400)
    df["D1n"] = (day - df["D1"]).astype("Float64")
    if "D15" in df.columns:
        df["D15n"] = (day - df["D15"]).round().astype("Float64")
    return df


def base_uid(df: pd.DataFrame, variant: str = "medium") -> pd.Series:
    """Blocking key. Rows with a missing component fall back to a per-row singleton."""
    cols = UID_RECIPES[variant]
    parts = []
    for c in cols:
        parts.append(df[c].astype("Float64").astype(str) if df[c].dtype.kind in "fiu"
                     else df[c].astype(str))
    uid = parts[0]
    for p in parts[1:]:
        uid = uid + "|" + p
    # A row missing any blocking component cannot be safely grouped; isolate it rather
    # than letting every such row collapse into one giant fake entity.
    bad = df[cols].isna().any(axis=1)
    uid = uid.where(~bad, other=pd.Series([f"__solo_{i}" for i in df.index], index=df.index))
    return uid.astype(str)


def fs_agreement(df: pd.DataFrame, i_rows: np.ndarray, j_rows: np.ndarray) -> np.ndarray:
    """Fellegi-Sunter style weighted agreement between candidate block representatives.

    Returns a score in [0, 1]: the weighted share of comparable attributes that agree.
    Attributes missing on either side are excluded from both numerator and denominator,
    so an absent field neither helps nor hurts.
    """
    num = np.zeros(len(i_rows))
    den = np.zeros(len(i_rows))
    for col, w in CFG.fs_weights.items():
        if col not in df.columns:
            continue
        a = df[col].to_numpy()[i_rows]
        b = df[col].to_numpy()[j_rows]
        comparable = pd.notna(a) & pd.notna(b)
        agree = comparable & (a == b)
        num += w * agree
        den += w * comparable
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def resolve_entities(df: pd.DataFrame, variant: str | None = None,
                     refine: bool = True) -> pd.Series:
    """Full Block A: blocking -> Fellegi-Sunter refinement -> guarded transitive closure.

    Returns an integer entity id per row.
    """
    variant = variant or CFG.uid_variant
    df = add_d1n(df) if "D1n" not in df.columns else df
    uid = base_uid(df, variant)
    codes, _ = pd.factorize(uid)
    n_blocks = codes.max() + 1
    if not refine or n_blocks < 2:
        return pd.Series(codes, index=df.index, name="entity")

    # One representative row per block, for candidate comparison.
    rep = pd.DataFrame({"block": codes, "row": np.arange(len(df))}) \
        .groupby("block", sort=True)["row"].first().to_numpy()

    # Candidate pairs: blocks sharing a coarser key. Cheap, and it is the only place a
    # merge can originate, so the conflict guard below sees every proposed merge.
    coarse = base_uid(df, "loose").to_numpy()[rep]
    uf = UnionFind(n_blocks)
    order = np.argsort(coarse, kind="stable")
    coarse_sorted = coarse[order]
    starts = np.flatnonzero(np.r_[True, coarse_sorted[1:] != coarse_sorted[:-1]])
    groups = np.split(order, starts[1:])

    conflict_rejects = 0
    score_rejects = 0
    for g in groups:
        if len(g) < 2 or len(g) > 200:  # runaway groups are noise, not entities
            continue
        anchor = g[0]
        others = g[1:]
        scores = fs_agreement(df, np.repeat(rep[anchor], len(others)), rep[others])
        for k, b in enumerate(others):
            if scores[k] < CFG.fs_agreement_threshold:
                score_rejects += 1
                continue
            if _conflicts(df, rep[anchor], rep[b]):
                conflict_rejects += 1
                continue
            uf.union(int(anchor), int(b))

    block_label = uf.labels()
    entity = block_label[codes]
    entity = pd.factorize(entity)[0]
    out = pd.Series(entity, index=df.index, name="entity")
    out.attrs["conflict_rejects"] = conflict_rejects
    out.attrs["score_rejects"] = score_rejects
    out.attrs["variant"] = variant
    return out


def _conflicts(df: pd.DataFrame, ra: int, rb: int) -> bool:
    """Conflict guard: reject merges implying two card brands or inconsistent D arithmetic."""
    for col in ("card4", "card6", "card3"):
        if col in df.columns:
            a, b = df[col].iat[ra], df[col].iat[rb]
            if pd.notna(a) and pd.notna(b) and a != b:
                return True
    if "D15n" in df.columns:
        a, b = df["D15n"].iat[ra], df["D15n"].iat[rb]
        if pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) > 3:
            return True
    return False


# --------------------------------------------------------------------------------------
# Label-free validation: C-column monotonicity
# --------------------------------------------------------------------------------------

def c_monotonicity_violation_rate(df: pd.DataFrame, entity: pd.Series) -> dict:
    """Unsupervised precision proxy for the entity resolution.

    C1..C14 are cumulative per-card counts, so within a true entity they must be
    non-decreasing in time. A merge that glues two different cards together will
    generally break that. NaNs are skipped, never treated as zero -- treating a missing
    count as 0 would manufacture violations that are artefacts of the imputation.
    """
    present = [c for c in C_COLS if c in df.columns]
    if not present:
        return {"c_monotonicity_violation_rate": float("nan"), "c_columns_checked": 0}

    work = df[["TransactionDT"] + present].copy()
    work["entity"] = entity.to_numpy()
    work = work.sort_values(["entity", "TransactionDT"], kind="stable")

    total_steps = 0
    violations = 0
    for col in present:
        v = work[col].to_numpy()
        e = work["entity"].to_numpy()
        same = e[1:] == e[:-1]
        prev, cur = v[:-1], v[1:]
        comparable = same & pd.notna(prev) & pd.notna(cur)
        total_steps += int(comparable.sum())
        violations += int((comparable & (cur < prev)).sum())
    rate = violations / total_steps if total_steps else float("nan")
    return {
        "c_monotonicity_violation_rate": rate,
        "c_columns_checked": len(present),
        "c_steps_compared": total_steps,
    }


def assert_uid_is_label_free(feature_names: list[str]) -> None:
    """Guard against the entity id or its components leaking into the model matrix."""
    banned = {"entity", "uid", "D1n", "D15n", "true_entity"}
    found = banned & set(feature_names)
    if found:
        raise AssertionError(
            f"Entity identifiers must never be model features; found {sorted(found)}. "
            "The audit's whole point is that identity is not behaviour.")


# --------------------------------------------------------------------------------------
# T2 -- entity statistics, including the go/no-go number
# --------------------------------------------------------------------------------------

def entity_stats(df: pd.DataFrame, entity: pd.Series,
                 truth: pd.DataFrame | None = None) -> dict:
    """Descriptives plus the two numbers C1 rests on: label purity and propagated share."""
    y = df["isFraud"].to_numpy()
    e = entity.to_numpy()
    tab = pd.DataFrame({"e": e, "y": y, "dt": df["TransactionDT"].to_numpy()})

    sizes = tab.groupby("e").size()
    nuniq = tab.groupby("e")["y"].nunique()
    purity = float((nuniq == 1).mean())

    # Index events, inferred without ground truth: an entity's first positive in time.
    frauds = tab[tab["y"] == 1].sort_values("dt", kind="stable")
    n_pos = int(y.sum())
    n_index_inferred = int(frauds.groupby("e").size().shape[0]) if n_pos else 0
    propagated_share = 1 - (n_index_inferred / n_pos) if n_pos else float("nan")

    out = {
        "n_rows": int(len(df)),
        "n_entities": int(tab["e"].nunique()),
        "entity_size_mean": float(sizes.mean()),
        "entity_size_median": float(sizes.median()),
        "entity_size_p95": float(sizes.quantile(0.95)),
        "entity_size_max": int(sizes.max()),
        "singleton_entity_share": float((sizes == 1).mean()),
        "fraud_rate": float(y.mean()),
        "entity_label_purity": purity,
        "n_positives": n_pos,
        "n_index_events_inferred": n_index_inferred,
        "propagated_positive_share": float(propagated_share),
        "fraud_entity_share": float((tab.groupby("e")["y"].max() == 1).mean()),
    }
    out.update(c_monotonicity_violation_rate(df, entity))

    if truth is not None and "true_entity" in truth.columns:
        out.update(entity_resolution_quality(entity.to_numpy(),
                                             truth["true_entity"].to_numpy()))
        if "is_index" in truth.columns:
            out["n_index_events_true"] = int(truth["is_index"].sum())
    return out


def entity_resolution_quality(pred: np.ndarray, true: np.ndarray) -> dict:
    """Pairwise precision/recall/F1 of the clustering. Synthetic-data check only.

    Computed via per-cluster combinatorics rather than materialising the O(n^2) pair set.
    """
    def pairs(counts: np.ndarray) -> float:
        return float((counts * (counts - 1) / 2).sum())

    df = pd.DataFrame({"p": pred, "t": true})
    same_pred = pairs(df.groupby("p").size().to_numpy())
    same_true = pairs(df.groupby("t").size().to_numpy())
    both = pairs(df.groupby(["p", "t"]).size().to_numpy())

    precision = both / same_pred if same_pred else 0.0
    recall = both / same_true if same_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"er_pair_precision": precision, "er_pair_recall": recall, "er_pair_f1": f1}
