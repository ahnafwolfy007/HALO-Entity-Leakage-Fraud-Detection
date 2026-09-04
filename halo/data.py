"""Loading, joining and base feature derivation.

Works unchanged against the real IEEE-CIS CSVs and against `synth.py` output, so the
smoke path exercises exactly the code the Kaggle run uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, IEEE_DIR
from .io import Timer, downcast

C_COLS = [f"C{i}" for i in range(1, 15)]


def load_ieee(data_dir: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load and join the transaction and identity tables.

    The identity table covers only a minority of transactions; the join is a LEFT JOIN
    and the resulting all-NaN identity block is itself signal (Block C mines it).
    """
    d = Path(data_dir or IEEE_DIR)
    with Timer("load transaction table"):
        txn = pd.read_csv(d / "train_transaction.csv", nrows=nrows)
    ident_path = d / "train_identity.csv"
    if ident_path.exists():
        with Timer("load identity table"):
            ident = pd.read_csv(ident_path)
        # The real competition files use `id-01` in test and `id_01` in train; normalise.
        ident.columns = [c.replace("-", "_") for c in ident.columns]
        with Timer("join transaction <- identity"):
            txn = txn.merge(ident, on="TransactionID", how="left")
    txn = downcast(txn, verbose=True)
    return txn


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time decomposition and cheap transaction-intrinsic features (information Level 0).

    Nothing here touches a label, so it is safe under every protocol.
    """
    df = df.copy()
    df["day"] = np.floor(df["TransactionDT"] / 86_400).astype(np.int32)
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["dow"] = (df["day"] % 7).astype(np.int8)
    df["log_amt"] = np.log1p(df["TransactionAmt"]).astype(np.float32)
    # Fractional cents are a classic card-testing tell and cost nothing to compute.
    df["amt_cents"] = ((df["TransactionAmt"] * 100) % 100).astype(np.float32)
    df["amt_is_round"] = (df["amt_cents"] == 0).astype(np.int8)
    df["odd_hour"] = (((df["hour"] < 6) | (df["hour"] >= 22))).astype(np.int8)

    if "card4" in df.columns and "ProductCD" in df.columns:
        df["ProductCD_card4"] = (df["ProductCD"].astype(str) + "_"
                                 + df["card4"].astype(str))
    # A stand-in for the issuing-bank BIN, which the anonymised data does not expose.
    if {"card1", "card3", "card5"} <= set(df.columns):
        df["bin_proxy"] = (df["card1"].astype("Int64").astype(str) + "_"
                           + df["card3"].astype("Int64").astype(str) + "_"
                           + df["card5"].astype("Int64").astype(str))
    df["nan_count"] = df.isna().sum(axis=1).astype(np.int16)
    return df


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group columns into families. Used by ablations (T4) and by F1's SHAP grouping."""
    cols = set(df.columns)
    exclude = {"TransactionID", "isFraud", "TransactionDT", "uid", "entity",
               "true_entity", "is_index", "true_regime", "within", "day"}

    fam: dict[str, list[str]] = {
        "identity_proxy": [c for c in df.columns
                           if c.startswith(("card", "addr")) or c in {"bin_proxy",
                                                                      "ProductCD_card4"}],
        "D_columns": [c for c in df.columns if c.startswith("D") and c[1:].isdigit()],
        "C_columns": [c for c in df.columns if c in C_COLS],
        "V_columns": [c for c in df.columns if c.startswith("V") and c[1:].isdigit()],
        "M_columns": [c for c in df.columns if c.startswith("M") and c[1:].isdigit()],
        "id_columns": [c for c in df.columns if c.startswith("id_")],
        "behaviour": [c for c in ("hour", "dow", "log_amt", "amt_cents", "amt_is_round",
                                  "odd_hour", "TransactionAmt") if c in cols],
        "missingness": [c for c in df.columns
                        if c.startswith("miss_") or c in {"nan_count", "regime"}],
        "association_risk": [c for c in df.columns if c.startswith("risk_")],
        "entity_memory": [c for c in df.columns if c.startswith("mem_")],
        "velocity": [c for c in df.columns if c.startswith("vel_")],
    }
    for k in fam:
        fam[k] = [c for c in fam[k] if c not in exclude]
    return fam


def prepare_matrix(df: pd.DataFrame, drop: list[str] | None = None
                   ) -> tuple[pd.DataFrame, list[str]]:
    """Encode categoricals and return a numeric matrix ready for a tree model.

    Object columns become ordinal codes fitted on the frame handed in. Callers running
    a past-only protocol must therefore pass train and test through together *only*
    after the split has been decided -- codes carry no label information, so this is
    safe, but frequency and target encodings are deliberately NOT done here.
    """
    # D1n / D15n are the entity *keys* derived in entities.py. They group rows; they are
    # never model features. Letting them through would hand the model the very identity
    # the Cold-Entity Protocol exists to withhold.
    drop = set(drop or []) | {"TransactionID", "isFraud", "TransactionDT",
                              "uid", "entity", "D1n", "D15n",
                              "true_entity", "is_index", "true_regime", "within"}
    use = [c for c in df.columns if c not in drop]
    X = df[use].copy()
    for col in X.columns:
        dt = X[col].dtype
        if pd.api.types.is_bool_dtype(dt):
            X[col] = X[col].astype(np.int8)
        elif pd.api.types.is_numeric_dtype(dt):
            # Nullable extension ints (Int64) upset LightGBM; force a plain float.
            if isinstance(dt, pd.api.extensions.ExtensionDtype):
                X[col] = X[col].astype(np.float32)
        else:
            # Everything else -- object, the pandas 3.0 `str` dtype, categorical,
            # datetime -- becomes an ordinal code. Codes carry no label information.
            X[col] = pd.factorize(X[col])[0].astype(np.int32)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, list(X.columns)
