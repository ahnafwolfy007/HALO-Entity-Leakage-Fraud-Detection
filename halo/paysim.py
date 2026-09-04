"""L5 -- the PaySim case study.

PaySim is *not* used here to validate generalisation. It is trivially solvable, and
reporting a second inflated number beside the one the paper is trying to expose would
undercut the paper's own thesis. It is used instead as a second worked example of the
audit, which upgrades the contribution from "we audited one dataset's quirk" to "we
built a general benchmark-auditing method, and it finds a *different* failure mode in a
structurally unrelated dataset."

The L5 mechanism
----------------
PaySim is an agent-based simulation. Transactions the simulator flags as fraudulent are
**cancelled**, so for ``isFraud=1`` rows the balance fields frequently fail to update the
way a genuine transaction's would. A model that learns

    amount != (oldbalanceOrg - newbalanceOrig)  =>  fraud

scores near-perfectly by detecting an artefact of how the simulator writes rows, not by
detecting fraud. ``isFlaggedFraud`` is worse still: it is set by a hard rule inside the
simulator (a single transfer above 200,000), so it is not a feature at all -- it is the
label's cousin, baked in by an ``if`` statement.

The audit is therefore a feature-ablation ladder rather than an entity ladder, and the
gap between the rungs *is* the measurement.

Honest note for the report: Block A does not transfer. ``nameOrig``/``nameDest`` are
largely single-use identifiers with little repeat structure, so L3 and L4 are not
PaySim's story. That is the point -- different data-generating process, different
failure mode, same audit lens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, PAYSIM_CSV
from .io import Timer, save_table
from .metrics import auprc, auroc, dollar_recall_at_k

BALANCE_ARTIFACT_COLS = [
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "balance_err_orig", "balance_err_dest",
]
RULE_LEAK_COLS = ["isFlaggedFraud"]


def load_paysim(path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    p = Path(path or PAYSIM_CSV)
    if not p.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {p}. On Kaggle, attach the dataset 'ealaxi/paysim1'.")
    with Timer("load PaySim"):
        df = pd.read_csv(p, nrows=nrows)
    return df


def add_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the balance-consistency residuals that make the shortcut explicit.

    These are exactly the features a competent practitioner would engineer, which is why
    the shortcut is so easy to fall into: it looks like good feature engineering.
    """
    df = df.copy()
    df["balance_err_orig"] = (df["oldbalanceOrg"] - df["amount"]
                              - df["newbalanceOrig"]).astype(np.float32)
    df["balance_err_dest"] = (df["oldbalanceDest"] + df["amount"]
                              - df["newbalanceDest"]).astype(np.float32)
    df["orig_emptied"] = ((df["newbalanceOrig"] == 0) &
                          (df["oldbalanceOrg"] > 0)).astype(np.int8)
    df["amount_eq_oldbalance"] = (np.isclose(df["amount"], df["oldbalanceOrg"])
                                  ).astype(np.int8)
    df["day"] = (df["step"] // 24).astype(np.int32)
    df["hour_of_day"] = (df["step"] % 24).astype(np.int8)
    df["log_amount"] = np.log1p(df["amount"]).astype(np.float32)
    return df


def _fit_eval(df: pd.DataFrame, drop_cols: list[str], seed: int = 0,
              chronological: bool = True) -> dict:
    import lightgbm as lgb
    y = df["isFraud"].to_numpy()
    drop = set(drop_cols) | {"isFraud", "nameOrig", "nameDest", "step"}
    X = df[[c for c in df.columns if c not in drop]].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0].astype(np.int32)
        elif isinstance(X[c].dtype, pd.api.extensions.ExtensionDtype):
            X[c] = X[c].astype(np.float32)

    if chronological:
        cut = int(len(df) * 0.7)
        tr = np.arange(cut)
        te = np.arange(cut, len(df))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(df))
        cut = int(len(df) * 0.7)
        tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])

    if y[tr].sum() < 5 or y[te].sum() < 5:
        return {"auprc": float("nan"), "note": "insufficient positives in a split"}

    params = dict(CFG.lgbm_params); params["random_state"] = seed
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr], y[tr])
    s = m.predict_proba(X.iloc[te])[:, 1]

    br = float(y[te].mean())
    out = {
        "auprc": auprc(y[te], s), "auroc": auroc(y[te], s),
        "base_rate": br, "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
    }
    out["auprc_lift"] = out["auprc"] / br if br > 0 else float("nan")
    out.update(dollar_recall_at_k(y[te], s, df["amount"].to_numpy()[te], seed=seed))
    return out


PAYSIM_RUNGS = [
    ("p0_all_features", [], "Everything, including isFlaggedFraud and raw balances"),
    ("p1_drop_rule_leak", RULE_LEAK_COLS,
     "Drop isFlaggedFraud (a simulator if-statement, not a feature)"),
    ("p2_drop_balance_residuals", RULE_LEAK_COLS + ["balance_err_orig", "balance_err_dest",
                                                    "amount_eq_oldbalance"],
     "Also drop the balance-consistency residuals"),
    ("p3_drop_all_balance", RULE_LEAK_COLS + BALANCE_ARTIFACT_COLS +
     ["amount_eq_oldbalance", "orig_emptied"],
     "Also drop every raw balance field -- the cancellation artefact is gone"),
]


def run_paysim_audit(df: pd.DataFrame | None = None, nrows: int | None = None,
                     seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """The L5 ladder. Each rung removes one more simulator artefact and re-measures."""
    if df is None:
        df = load_paysim(nrows=nrows)
    df = add_paysim_features(df)

    rows = []
    for key, drop, label in PAYSIM_RUNGS:
        for seed in seeds:
            r = _fit_eval(df, drop, seed=seed, chronological=True)
            r.update({"rung": key, "rung_label": label, "seed": seed,
                      "dropped": ",".join(drop) or "(none)"})
            rows.append(r)
    raw = pd.DataFrame(rows)
    save_table(raw, "L5_raw", "PaySim generative-determinism audit, per seed")

    num = raw.select_dtypes(include=[np.number]).columns.tolist()
    agg = raw.groupby(["rung", "rung_label"])[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    save_table(agg, "L5", "PaySim: AUPRC collapse as simulator artefacts are removed")
    return agg


def paysim_shortcut_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Direct evidence for the mechanism, before any model is fitted.

    Shows the balance-residual behaviour conditional on the label. If the cancellation
    story is right, fraudulent rows show a residual pattern legitimate rows do not.
    """
    df = add_paysim_features(df) if "balance_err_orig" not in df.columns else df
    rows = []
    for label, g in df.groupby("isFraud"):
        rows.append({
            "isFraud": int(label),
            "n": int(len(g)),
            "share_amount_eq_oldbalance": float(g["amount_eq_oldbalance"].mean()),
            "share_orig_emptied": float(g["orig_emptied"].mean()),
            "median_balance_err_orig": float(g["balance_err_orig"].median()),
            "share_dest_balance_unchanged": float(
                (g["oldbalanceDest"] == g["newbalanceDest"]).mean()),
            "share_flagged": float(g["isFlaggedFraud"].mean())
            if "isFlaggedFraud" in g else float("nan"),
        })
    out = pd.DataFrame(rows)
    save_table(out, "L5_evidence", "PaySim shortcut evidence, conditional on the label")
    return out
