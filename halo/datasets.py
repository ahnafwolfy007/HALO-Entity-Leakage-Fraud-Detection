"""Unified dataset adapters for the five-dataset replication study.

Every adapter returns the same canonical frame so the rest of the pipeline is
dataset-agnostic:

    ts       float64  seconds on a monotonic clock
    day      int32    floor(ts / 86400), the clock the protocol splits on
    y        int8     fraud label
    amount   float32  transaction value (NaN where the dataset has none)
    entity   int64    resolved or native entity id
    + the dataset's own columns, including its risk attributes

Each adapter declares a ``DatasetSpec`` recording what the dataset can and
cannot support, so downstream code degrades honestly rather than silently
computing a metric the data does not license. In particular:

* ``has_amount=False``  -> dollar-recall is skipped, not faked.
* ``entity_from``       -> whether entity ids are native, must be resolved
                           (Block A), derived from graph structure, or are a
                           documented *proxy* (BAF).
* ``horizon_days``      -> the known censoring horizon, if the labelling rule
                           defines one. Required by the C1 delay model.

Datasets are not redistributed with this repository. Each adapter raises a
``DatasetNotAvailable`` naming the expected path and the source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, INPUT_DIR, PAYSIM_CSV
from .io import downcast


class DatasetNotAvailable(FileNotFoundError):
    """Raised with the expected path and where to obtain the data."""


# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    risk_attributes: tuple[str, ...]
    entity_from: str                      # native | resolve | graph | proxy
    entity_cols: tuple[str, ...] = ()
    has_amount: bool = True
    has_graph: bool = False
    horizon_days: float | None = None     # known censoring horizon, if defined
    time_unit: str = "seconds"
    notes: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        lines = [f"{self.name}: entity={self.entity_from}, "
                 f"{len(self.risk_attributes)} risk attrs, "
                 f"amount={'yes' if self.has_amount else 'NO'}, "
                 f"horizon={self.horizon_days}"]
        lines += [f"  caveat: {c}" for c in self.caveats]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _hash_ids(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Stable int64 id from a column combination (used for proxy entities)."""
    joined = frame[cols].astype(str).agg("\x1f".join, axis=1)
    return joined.map(
        lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                                 "little", signed=True)
    ).to_numpy(dtype=np.int64)


def _finalise(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Enforce the canonical schema and sort by time."""
    missing = {"ts", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{spec.name}: adapter did not produce {missing}")
    if "amount" not in df.columns:
        df["amount"] = np.nan
    df["ts"] = df["ts"].astype(np.float64)
    df["day"] = np.floor(df["ts"] / 86_400.0).astype(np.int32)
    df["y"] = df["y"].astype(np.int8)
    df["amount"] = df["amount"].astype(np.float32)
    if "entity" in df.columns:
        df["entity"] = df["entity"].astype(np.int64)
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return downcast(df)


def _require(path: Path, name: str, source: str) -> Path:
    if not path.exists():
        raise DatasetNotAvailable(
            f"{name}: expected data at {path}\n"
            f"  obtain from: {source}\n"
            f"  (datasets are not redistributed with this repository)"
        )
    return path


# --------------------------------------------------------------------------------------
# IEEE-CIS -- the headline dataset
# --------------------------------------------------------------------------------------

IEEE_SPEC = DatasetSpec(
    name="ieee_cis",
    risk_attributes=CFG.risk_attributes,
    entity_from="resolve",
    entity_cols=("card1", "addr1", "D1"),
    has_amount=True,
    horizon_days=120.0,          # Vesta rule: unreported at 120d => legitimate
    notes="590,540 transactions. Entity must be inferred (Block A); no native key.",
    caveats=(
        "Labelling rule documented only in a Kaggle forum reply; verified "
        "empirically via entity label purity (T2), not taken on trust.",
    ),
)


def load_ieee_cis(data_dir: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    from .data import load_ieee
    df = load_ieee(data_dir, nrows=nrows)
    df = df.rename(columns={"TransactionDT": "ts", "isFraud": "y",
                            "TransactionAmt": "amount"})
    # entity is resolved downstream by Block A; leave a placeholder column out.
    return _finalise(df, IEEE_SPEC)


# --------------------------------------------------------------------------------------
# PaySim -- synthetic negative control
# --------------------------------------------------------------------------------------

PAYSIM_SPEC = DatasetSpec(
    name="paysim",
    risk_attributes=("nameDest", "type", "dest_prefix"),
    entity_from="native",
    entity_cols=("nameOrig",),
    has_amount=True,
    horizon_days=None,
    time_unit="hours",
    notes="6.36M transactions. Kept as the generative-shortcut control (L5).",
    caveats=(
        "Raw balance fields encode the generator's fraud rule (amount == prior "
        "balance in 97.8% of frauds). Dropped by default; enable only to "
        "reproduce the L5 collapse.",
    ),
)


def load_paysim(data_dir: Path | None = None, nrows: int | None = None,
                keep_raw_balances: bool = False) -> pd.DataFrame:
    root = Path(data_dir or INPUT_DIR)
    cand = [Path(PAYSIM_CSV), root / "paysim" / "paysim.csv",
            root / "paysim" / "PS_20174392719_1491204439457_log.csv"]
    path = next((c for c in cand if c.exists()), None)
    if path is None:
        path = _require(cand[0], "paysim",
                        "https://www.kaggle.com/datasets/ealaxi/paysim1")
    df = pd.read_csv(path, nrows=nrows)
    df = df.rename(columns={"isFraud": "y"})
    df["ts"] = df["step"].astype(np.float64) * 3600.0          # step is an hour
    df["entity"] = _hash_ids(df, ["nameOrig"])
    df["dest_prefix"] = df["nameDest"].astype(str).str[0]      # C=customer, M=merchant
    if not keep_raw_balances:
        df = df.drop(columns=[c for c in ("oldbalanceOrg", "newbalanceOrig",
                                          "oldbalanceDest", "newbalanceDest")
                              if c in df.columns])
    df = df.drop(columns=[c for c in ("isFlaggedFraud",) if c in df.columns])
    return _finalise(df, PAYSIM_SPEC)


# --------------------------------------------------------------------------------------
# BAF -- Bank Account Fraud (NeurIPS 2022 D&B)
# --------------------------------------------------------------------------------------

BAF_SPEC = DatasetSpec(
    name="baf",
    risk_attributes=("device_os", "payment_type", "employment_status",
                     "housing_status", "source", "proposed_credit_limit"),
    entity_from="proxy",
    entity_cols=("device_os", "employment_status", "housing_status",
                 "payment_type", "zip_count_4w"),
    has_amount=False,            # application fraud: no transaction value
    horizon_days=None,
    time_unit="months",
    notes="1M applications x 6 variants. Real-derived, built for evaluation research.",
    caveats=(
        "No native entity key: this is application fraud, not transaction "
        "fraud. Entity is a documented PROXY over categorical combinations, so "
        "the L3 rung on BAF measures proxy-entity leakage and must be reported "
        "as such, not as a like-for-like replication of IEEE-CIS.",
        "No transaction amount: dollar-recall is not computed for BAF.",
        "Time resolution is monthly (0-7), so the latency sweep is coarse.",
    ),
)


def load_baf(data_dir: Path | None = None, nrows: int | None = None,
             variant: str = "Base") -> pd.DataFrame:
    root = Path(data_dir or INPUT_DIR)
    path = _require(root / "baf" / f"{variant}.csv", "baf",
                    "https://github.com/feedzai/bank-account-fraud")
    df = pd.read_csv(path, nrows=nrows)
    df = df.rename(columns={"fraud_bool": "y"})
    # month is 0..7; spread rows evenly within a month so intra-month order is
    # arbitrary but deterministic, and the protocol still has a usable clock.
    rank_in_month = df.groupby("month").cumcount()
    per_month = df.groupby("month")["y"].transform("size").to_numpy()
    frac = rank_in_month.to_numpy() / np.maximum(per_month, 1)
    df["ts"] = (df["month"].to_numpy() + frac) * 30.0 * 86_400.0
    df["entity"] = _hash_ids(df, list(BAF_SPEC.entity_cols))
    return _finalise(df, BAF_SPEC)


# --------------------------------------------------------------------------------------
# Elliptic -- real graph-native dataset
# --------------------------------------------------------------------------------------

ELLIPTIC_SPEC = DatasetSpec(
    name="elliptic",
    risk_attributes=("cc_id",),
    entity_from="graph",
    entity_cols=(),
    has_amount=False,
    has_graph=True,
    horizon_days=None,
    time_unit="steps",
    notes="203,769 nodes / 234,355 edges over 49 time steps. Native edge list.",
    caveats=(
        "Features are anonymised; no transaction amount, so dollar-recall is "
        "not computed.",
        "Entity is the weakly-connected component of the transaction graph, a "
        "structural proxy for a controlling actor.",
        "Only ~23% of nodes are labelled; unlabelled nodes are dropped for "
        "supervised evaluation and retained for propagation.",
    ),
)


def load_elliptic(data_dir: Path | None = None, nrows: int | None = None
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (canonical frame, edge array of shape (n_edges, 2) in row index space)."""
    root = Path(data_dir or INPUT_DIR) / "elliptic"
    f_feat = _require(root / "elliptic_txs_features.csv", "elliptic",
                      "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
    f_class = _require(root / "elliptic_txs_classes.csv", "elliptic", "see above")
    f_edge = _require(root / "elliptic_txs_edgelist.csv", "elliptic", "see above")

    feats = pd.read_csv(f_feat, header=None, nrows=nrows)
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    classes = pd.read_csv(f_class)
    df = feats.merge(classes, on="txId", how="left")

    # class: '1' illicit, '2' licit, 'unknown'
    df["y"] = np.where(df["class"].astype(str) == "1", 1,
                       np.where(df["class"].astype(str) == "2", 0, -1)).astype(np.int8)
    df["ts"] = df["time_step"].astype(np.float64) * 86_400.0
    df = df.drop(columns=["class"])

    edges = pd.read_csv(f_edge)
    idx = pd.Series(np.arange(len(df)), index=df["txId"].to_numpy())
    e1 = edges["txId1"].map(idx)
    e2 = edges["txId2"].map(idx)
    keep = e1.notna() & e2.notna()
    edge_arr = np.stack([e1[keep].to_numpy(np.int64), e2[keep].to_numpy(np.int64)], axis=1)

    df["entity"] = _connected_components(len(df), edge_arr)
    df["cc_id"] = df["entity"].astype(str)
    df = _finalise(df, ELLIPTIC_SPEC)
    return df, edge_arr


def _connected_components(n: int, edges: np.ndarray) -> np.ndarray:
    """Union-find over the edge list; component id becomes the entity proxy."""
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


# --------------------------------------------------------------------------------------
# IBM TabFormer -- scale replication
# --------------------------------------------------------------------------------------

TABFORMER_SPEC = DatasetSpec(
    name="tabformer",
    risk_attributes=("Merchant Name", "MCC", "Merchant City", "Zip", "Use Chip"),
    entity_from="native",
    entity_cols=("User", "Card"),
    has_amount=True,
    horizon_days=None,
    notes="24.4M transactions with explicit User/Card keys -- ideal for entity work.",
    caveats=(
        "Synthetic (IBM generator). Included for scale replication, not as "
        "independent real-world evidence.",
        "Subsampled by default; full 24.4M exceeds a 30GB session.",
    ),
)


def load_tabformer(data_dir: Path | None = None, nrows: int | None = None,
                   subsample_rows: int | None = 5_000_000,
                   seed: int = 0) -> pd.DataFrame:
    root = Path(data_dir or INPUT_DIR)
    path = _require(root / "tabformer" / "card_transaction.v1.csv", "tabformer",
                    "https://github.com/IBM/TabFormer (credit card transactions)")
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Is Fraud?": "y_raw", "Amount": "amount_raw"})

    df["y"] = (df["y_raw"].astype(str).str.strip().str.lower() == "yes").astype(np.int8)
    df["amount"] = (df["amount_raw"].astype(str)
                    .str.replace("$", "", regex=False).astype(np.float32))

    hh = df["Time"].astype(str).str.split(":", expand=True)
    df["ts"] = (
        pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]),
                       errors="coerce").astype("int64") / 1e9
        + hh[0].astype(np.float64) * 3600.0 + hh[1].astype(np.float64) * 60.0
    )
    df = df[df["ts"].notna()]
    df["entity"] = _hash_ids(df, ["User", "Card"])
    df = df.drop(columns=["y_raw", "amount_raw"])

    if subsample_rows is not None and len(df) > subsample_rows:
        # Contiguous tail rather than a random sample: preserves the temporal
        # structure the protocol depends on. Random sampling would destroy the
        # velocity and association signals.
        df = df.sort_values("ts", kind="mergesort").iloc[-subsample_rows:]
        print(f"  [tabformer] subsampled to the most recent {subsample_rows:,} rows")

    return _finalise(df, TABFORMER_SPEC)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

REGISTRY = {
    "ieee_cis":   (IEEE_SPEC, load_ieee_cis),
    "paysim":     (PAYSIM_SPEC, load_paysim),
    "baf":        (BAF_SPEC, load_baf),
    "elliptic":   (ELLIPTIC_SPEC, load_elliptic),
    "tabformer":  (TABFORMER_SPEC, load_tabformer),
}

ALL_DATASETS = tuple(REGISTRY)


def load_dataset(name: str, data_dir: Path | None = None, nrows: int | None = None,
                 **kwargs):
    """Load a dataset by name. Returns (df, spec) -- or (df, spec, edges) for Elliptic."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {ALL_DATASETS}")
    spec, loader = REGISTRY[name]
    out = loader(data_dir=data_dir, nrows=nrows, **kwargs)
    if isinstance(out, tuple):                    # elliptic returns edges too
        df, edges = out
        return df, spec, edges
    return out, spec


def describe_all() -> str:
    return "\n\n".join(spec.describe() for spec, _ in REGISTRY.values())
