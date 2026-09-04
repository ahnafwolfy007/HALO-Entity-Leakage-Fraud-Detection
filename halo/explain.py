"""T6 and F1 -- explainability, and the figure the paper is built around.

F1 (SHAP mass migration) is the argument, compressed into one chart. Run SHAP on the
same model family under the leaky protocol and under CEP, then group attribution mass by
feature family. Under the leaky protocol the mass sits on identity proxies -- card
columns, D-columns, V-block components. Under CEP it moves to behavioural and
association features.

The reading is the compliance argument: under the leaky protocol the field's fraud
explanations are effectively saying *"this resembles a card we have seen before."* That
is not an adverse-action reason a regulator can accept, and an investigator cannot act
on it.

T6 contrasts two ways of getting trustworthy explanations:
  * post-hoc audit  -- SHAP, then ROAR to check whether SHAP was telling the truth
  * by construction -- monotone constraints, exact counterfactual thresholds, and a
    measured price in AUPRC
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, LADDER
from .data import feature_columns
from .io import Timer, save_table
from .metrics import auprc
from .model import HaloModel
from .protocol import folds_for_rung, rolling_origin_folds
from . import experiments as EX

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


# --------------------------------------------------------------------------------------
# F1 -- SHAP mass migration
# --------------------------------------------------------------------------------------

def _family_of(col: str) -> str:
    if col.startswith("risk_") or col.startswith("riskn_"):
        return "association_risk"
    if col.startswith("mem_"):
        return "entity_memory"
    if col.startswith("vel_"):
        return "velocity"
    if col.startswith("miss_") or col in {"nan_count", "regime", "miss_density"}:
        return "missingness"
    if col.startswith(("card", "addr")) or col in {"bin_proxy", "ProductCD_card4"}:
        return "identity_proxy"
    if col.startswith("D") and col[1:].isdigit():
        return "D_columns"
    if col.startswith("C") and col[1:].isdigit():
        return "C_columns"
    if col.startswith("V") and col[1:].isdigit():
        return "V_columns"
    if col.startswith("M") and col[1:].isdigit():
        return "M_columns"
    if col.startswith("id_") or col in {"DeviceType", "DeviceInfo"}:
        return "id_columns"
    return "behaviour"


def shap_mass_by_family(model, X: pd.DataFrame, max_rows: int = 3_000,
                        seed: int = 0) -> pd.Series:
    """Mean |SHAP| per feature, summed by family and normalised to a share of total mass."""
    if shap is None:
        raise ImportError("shap is required for F1")
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        X = X.iloc[rng.choice(len(X), max_rows, replace=False)]
    booster = getattr(model, "model_", model)
    explainer = shap.TreeExplainer(booster)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):          # older shap returns [neg, pos]
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # (n, features, classes)
        vals = vals[:, :, -1]
    mass = pd.Series(np.abs(vals).mean(axis=0), index=X.columns)
    fam = mass.groupby([_family_of(c) for c in mass.index]).sum()
    total = fam.sum()
    return (fam / total) if total > 0 else fam


def run_shap_migration(df: pd.DataFrame, entity: np.ndarray,
                       is_index_all: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """F1: attribution mass by family, leaky protocol vs CEP, same model family."""
    rows = []
    leaky, honest = LADDER[0], LADDER[-1]
    for rung in (leaky, honest):
        folds = folds_for_rung(df, entity, rung, seed=seed)
        if not folds:
            continue
        fold = folds[len(folds) // 2]
        X, regime, _ = EX.build_fold_features(
            df, entity, fold, latency_delta=rung.latency_delta,
            past_only=rung.past_only_encoding)
        y = df["isFraud"].to_numpy()
        m = HaloModel(monotone_features=[], seed=seed, enforce_monotone=False)
        m.fit(X.iloc[fold.train_idx], y[fold.train_idx])
        share = shap_mass_by_family(m, X.iloc[fold.test_idx], seed=seed)
        for fam, v in share.items():
            rows.append({"protocol": "leaky" if rung is leaky else "CEP",
                         "rung": rung.key, "family": fam, "mass_share": float(v)})
    out = pd.DataFrame(rows)
    if len(out):
        piv = out.pivot_table(index="family", columns="protocol",
                              values="mass_share", aggfunc="mean").fillna(0.0)
        if {"leaky", "CEP"} <= set(piv.columns):
            piv["migration"] = piv["CEP"] - piv["leaky"]
        piv = piv.sort_values("migration" if "migration" in piv else piv.columns[0])
        save_table(piv.reset_index(), "F1", "SHAP attribution mass by family, leaky vs CEP")
        return piv.reset_index()
    return out


# --------------------------------------------------------------------------------------
# T6 -- faithfulness: post-hoc audit vs guarantee by construction
# --------------------------------------------------------------------------------------

def roar_curve(df: pd.DataFrame, entity: np.ndarray, fold, seed: int = 0,
               fractions: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)) -> pd.DataFrame:
    """RemOve-And-Retrain: drop the top-k SHAP features, retrain, watch AUPRC fall.

    A faithful attribution should cause a steeper fall than removing random features.
    The gap between the two curves is what 'faithfulness by audit' actually buys, and it
    is the baseline HALO's construction-time guarantee is compared against.
    """
    X, regime, _ = EX.build_fold_features(df, entity, fold,
                                          latency_delta=CFG.headline_delta,
                                          past_only=True)
    y = df["isFraud"].to_numpy()
    tr, te = fold.train_idx, fold.test_idx

    base = HaloModel(monotone_features=[], seed=seed, enforce_monotone=False)
    base.fit(X.iloc[tr], y[tr])
    ranked = shap_mass_by_feature(base, X.iloc[te], seed=seed)
    order = list(ranked.sort_values(ascending=False).index)

    rng = np.random.default_rng(seed)
    rand_order = list(rng.permutation(list(X.columns)))

    rows = []
    for frac in fractions:
        k = int(len(order) * frac)
        for label, cols in (("shap", order[:k]), ("random", rand_order[:k])):
            keep = [c for c in X.columns if c not in set(cols)]
            if len(keep) < 3:
                continue
            m = HaloModel(monotone_features=[], seed=seed, enforce_monotone=False)
            m.fit(X.iloc[tr][keep], y[tr])
            s = m.decision_score(X.iloc[te][keep])
            rows.append({"removal": label, "fraction_removed": frac,
                         "n_removed": k, "auprc": auprc(y[te], s)})
    return pd.DataFrame(rows)


def shap_mass_by_feature(model, X: pd.DataFrame, max_rows: int = 2_000,
                         seed: int = 0) -> pd.Series:
    if shap is None:
        raise ImportError("shap is required")
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        X = X.iloc[rng.choice(len(X), max_rows, replace=False)]
    booster = getattr(model, "model_", model)
    vals = shap.TreeExplainer(booster).shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return pd.Series(np.abs(vals).mean(axis=0), index=X.columns)


def run_faithfulness(df: pd.DataFrame, entity: np.ndarray, is_index_all: np.ndarray,
                     seed: int = 0) -> pd.DataFrame:
    """T6: coverage, the price of monotonicity, monotone-direction verification, ROAR."""
    folds = rolling_origin_folds(df, entity, latency_delta=CFG.headline_delta,
                                 entity_disjoint=True)
    if not folds:
        return pd.DataFrame()
    fold = folds[len(folds) // 2]
    y = df["isFraud"].to_numpy()
    tr, te = fold.train_idx, fold.test_idx

    X, regime, _ = EX.build_fold_features(df, entity, fold,
                                          latency_delta=CFG.headline_delta,
                                          past_only=True)
    from .risk import AssociationRisk
    mono = AssociationRisk.risk_feature_names(X.columns)

    rows = []
    models = {}
    for label, enforce in (("monotone", True), ("unconstrained", False)):
        m = HaloModel(monotone_features=mono if enforce else [], seed=seed,
                      enforce_monotone=enforce)
        m.fit(X.iloc[tr], y[tr], regime=regime[tr])
        s = m.decision_score(X.iloc[te])
        rows.append({"variant": label, "auprc": auprc(y[te], s),
                     "n_monotone_features": len(mono) if enforce else 0})
        models[label] = (m, s)

    price = rows[1]["auprc"] - rows[0]["auprc"]

    m_mono, s_mono = models["monotone"]
    thr = float(np.quantile(s_mono, 1 - CFG.review_budget_k))
    cov = m_mono.reason_code_coverage(X.iloc[te], thr, max_rows=400)
    ver = m_mono.verify_monotonicity(X.iloc[te], n_probe=120, seed=seed)

    with Timer("T6: ROAR curve"):
        roar = roar_curve(df, entity, fold, seed=seed)

    summary = pd.DataFrame([{
        "auprc_monotone": rows[0]["auprc"],
        "auprc_unconstrained": rows[1]["auprc"],
        "monotonicity_price_auprc": price,
        "monotonicity_price_relative": (price / rows[1]["auprc"]
                                        if rows[1]["auprc"] else float("nan")),
        "n_monotone_features": len(mono),
        "monotone_raw_violations": int(ver["monotone_violations"].sum()) if len(ver) else 0,
        "monotone_max_violation": float(ver["max_violation"].max()) if len(ver) else 0.0,
        "monotone_features_within_tolerance": int(ver["direction_ok"].sum()) if len(ver) else 0,
        "monotone_features_checked": int(len(ver)),
        **cov,
    }])
    save_table(summary, "T6", "Faithfulness: coverage, price, and direction verification")
    save_table(ver, "T6_monotone_verification", "Per-feature monotone direction check")
    save_table(roar, "T6_roar", "ROAR: SHAP-guided vs random feature removal")
    return summary
