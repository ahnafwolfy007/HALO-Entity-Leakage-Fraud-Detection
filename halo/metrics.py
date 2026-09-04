"""Evaluation. Four new metrics plus the secondaries, and the guards that keep them honest.

Accuracy is deliberately absent. At a 3.5% base rate it is noise dressed as a result.

Base-rate comparability (adversarial review item 9)
---------------------------------------------------
AUPRC is not comparable across folds or protocols with different positive rates: its
no-skill floor *is* the base rate. Every AUPRC returned here is therefore accompanied by
the base rate and by ``auprc_lift`` = AUPRC / base_rate, which is comparable. The report
uses lift whenever it crosses protocols.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from .config import CFG


# --------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------

def auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


# --------------------------------------------------------------------------------------
# The four protocol metrics
# --------------------------------------------------------------------------------------

def index_auprc(y: np.ndarray, s: np.ndarray, is_index: np.ndarray) -> dict:
    """AUPRC where the positive class is index events only.

    Propagated positives are *masked out of the evaluation entirely*, not relabelled
    negative. Calling them negatives would punish a model for flagging a transaction on
    a card that genuinely is compromised; dropping them asks the only question that
    matters -- can you find the first one?
    """
    y = np.asarray(y).astype(bool)
    is_index = np.asarray(is_index).astype(bool)
    propagated = y & ~is_index
    keep = ~propagated
    if keep.sum() == 0 or is_index[keep].sum() == 0:
        return {"index_auprc": float("nan"), "n_index_positives": int(is_index.sum()),
                "n_masked_propagated": int(propagated.sum()), "index_base_rate": float("nan")}
    yy = is_index[keep].astype(int)
    ss = np.asarray(s)[keep]
    br = float(yy.mean())
    val = auprc(yy, ss)
    return {
        "index_auprc": val,
        "index_base_rate": br,
        "index_auprc_lift": val / br if br > 0 else float("nan"),
        "n_index_positives": int(yy.sum()),
        "n_masked_propagated": int(propagated.sum()),
    }


def cold_entity_auprc(y: np.ndarray, s: np.ndarray, cold_mask: np.ndarray) -> dict:
    """AUPRC restricted to entities never seen in training. Measures memorisation directly."""
    cold = np.asarray(cold_mask).astype(bool)
    if cold.sum() == 0:
        return {"cold_entity_auprc": float("nan"), "n_cold_rows": 0,
                "cold_base_rate": float("nan"), "cold_positive_count": 0}
    yy, ss = np.asarray(y)[cold], np.asarray(s)[cold]
    br = float(yy.mean()) if len(yy) else float("nan")
    val = auprc(yy, ss)
    return {
        "cold_entity_auprc": val,
        "cold_base_rate": br,
        "cold_entity_auprc_lift": val / br if br and br > 0 else float("nan"),
        "n_cold_rows": int(cold.sum()),
        "cold_positive_count": int(yy.sum()),
        "cold_row_share": float(cold.mean()),
    }


def time_to_detection(df_test: pd.DataFrame, entity: np.ndarray, s: np.ndarray,
                      is_index: np.ndarray, budget_k: float | None = None) -> dict:
    """Lead time between the model's first alert on an entity and that entity's index event.

    Negative TTD means the alert fired *before* the compromise -- genuine early detection.
    The alert threshold is the score at the review budget, so TTD is measured at a
    realistic operating point rather than at an arbitrary 0.5.
    """
    budget_k = budget_k if budget_k is not None else CFG.review_budget_k
    s = np.asarray(s, dtype=float)
    if len(s) == 0:
        return {"ttd_median_txn": float("nan"), "ttd_n_entities": 0}
    thr = float(np.quantile(s, 1 - budget_k))
    alert = s >= thr

    t = df_test["TransactionDT"].to_numpy(dtype=float)
    tab = pd.DataFrame({"e": entity, "t": t, "alert": alert,
                        "idx": np.asarray(is_index).astype(bool),
                        "rank": np.arange(len(s))})
    tab = tab.sort_values(["e", "t"], kind="stable")
    tab["seq"] = tab.groupby("e").cumcount()

    rows = []
    for e, g in tab.groupby("e", sort=False):
        if not g["idx"].any():
            continue
        gi = g[g["idx"]].iloc[0]
        ga = g[g["alert"]]
        if ga.empty:
            rows.append({"entity": e, "detected": False,
                         "ttd_txn": np.nan, "ttd_hours": np.nan})
            continue
        first = ga.iloc[0]
        rows.append({
            "entity": e, "detected": True,
            "ttd_txn": float(first["seq"] - gi["seq"]),
            "ttd_hours": float((first["t"] - gi["t"]) / 3600.0),
        })
    if not rows:
        return {"ttd_median_txn": float("nan"), "ttd_n_entities": 0,
                "ttd_detected_share": float("nan")}
    r = pd.DataFrame(rows)
    det = r[r["detected"]]
    return {
        "ttd_n_entities": int(len(r)),
        "ttd_detected_share": float(r["detected"].mean()),
        "ttd_median_txn": float(det["ttd_txn"].median()) if len(det) else float("nan"),
        "ttd_median_hours": float(det["ttd_hours"].median()) if len(det) else float("nan"),
        "ttd_early_share": float((det["ttd_txn"] < 0).mean()) if len(det) else float("nan"),
        "_ttd_frame": r,
    }


def dollar_recall_at_k(y: np.ndarray, s: np.ndarray, amount: np.ndarray,
                       k: float | None = None, seed: int = 0) -> dict:
    """Share of fraudulent value recovered inside a k% manual-review budget.

    Ties (adversarial review item 14): equal scores are broken *randomly* with a fixed
    seed, never by amount. Breaking ties by amount would quietly optimise the metric the
    model is being judged on and inflate the result.
    """
    k = k if k is not None else CFG.review_budget_k
    y = np.asarray(y).astype(float)
    s = np.asarray(s, dtype=float)
    amount = np.asarray(amount, dtype=float)
    n = len(s)
    if n == 0 or y.sum() == 0:
        return {"dollar_recall_at_k": float("nan"), "k": k}

    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -s))
    take = max(1, int(round(n * k)))
    sel = order[:take]

    total_fraud_value = float((y * amount).sum())
    caught = float((y[sel] * amount[sel]).sum())
    return {
        "dollar_recall_at_k": caught / total_fraud_value if total_fraud_value > 0 else float("nan"),
        "count_recall_at_k": float(y[sel].sum() / y.sum()),
        "alert_precision_at_k": float(y[sel].mean()),
        "k": k,
        "n_reviewed": int(take),
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def evaluate(y, s, *, amount=None, entity=None, is_index=None, cold_mask=None,
             proba=None, seed: int = 0) -> dict:
    """Everything at once. Missing inputs simply omit their metrics rather than crashing."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    br = float(y.mean()) if len(y) else float("nan")
    out = {
        "n": int(len(y)),
        "base_rate": br,
        "auprc": auprc(y, s),
        "auroc": auroc(y, s),
    }
    out["auprc_lift"] = out["auprc"] / br if br and br > 0 else float("nan")

    if proba is not None:
        p = np.clip(np.asarray(proba, dtype=float), 0, 1)
        try:
            out["brier"] = float(brier_score_loss(y, p))
        except ValueError:
            out["brier"] = float("nan")
        out["ece"] = expected_calibration_error(y, p)

    if is_index is not None:
        out.update(index_auprc(y, s, is_index))
    if cold_mask is not None:
        out.update(cold_entity_auprc(y, s, cold_mask))
    if amount is not None:
        out.update(dollar_recall_at_k(y, s, amount, seed=seed))
    return out


def aggregate_seeds(rows: list[dict], keys: list[str] | None = None) -> pd.DataFrame:
    """Mean +/- std across seeds. Ladder deltas inside one std are not results."""
    df = pd.DataFrame(rows)
    group_cols = [c for c in (keys or []) if c in df.columns]
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if c not in group_cols and not c.startswith("_")]
    if not group_cols:
        agg = df[num].agg(["mean", "std"]).T.reset_index()
        agg.columns = ["metric", "mean", "std"]
        return agg
    g = df.groupby(group_cols)[num].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()
