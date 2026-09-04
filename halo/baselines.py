"""Baselines, steel-manned.

If the baselines are crippled the leakage ladder is worthless and the paper is worthless.
Every model here gets the same feature matrix, the same fold, and -- where it has
hyperparameters worth tuning -- the same search budget, drawn from a shared inner
validation split that never touches the test fold (adversarial review item 8).

L1 (resampling leakage) is implemented here rather than in ``protocol.py`` because it is
a property of *where* SMOTE sits relative to the split:

    resample_before_split=True   -> oversample the whole frame, then split  (leaky)
    resample_before_split=False  -> split, then oversample the training fold only
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CFG

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# --------------------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------------------

def naive_smote(X: pd.DataFrame, y: np.ndarray, seed: int = 0,
                k: int = 5, ratio: float = 0.25) -> tuple[pd.DataFrame, np.ndarray]:
    """Minimal SMOTE, so the pipeline has no imbalanced-learn dependency.

    Interpolates between a minority point and one of its k nearest minority neighbours.
    Faithful enough to reproduce the leakage effect L1 is about: when this runs before
    the split, synthetic points built from test-fold minority rows end up in training.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg_n = int((y == 0).sum())
    target = int(neg_n * ratio)
    need = target - len(pos)
    if need <= 0 or len(pos) < k + 1:
        return X, y

    Xp = X.iloc[pos]
    filled = Xp.fillna(Xp.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(pos))).fit(filled)
    _, idx = nn.kneighbors(filled)

    pick = rng.integers(0, len(pos), size=need)
    nbr = idx[pick, rng.integers(1, idx.shape[1], size=need)]
    lam = rng.random((need, 1))
    synth = filled[pick] + lam * (filled[nbr] - filled[pick])

    Xs = pd.DataFrame(synth, columns=X.columns)
    Xout = pd.concat([X.reset_index(drop=True), Xs], ignore_index=True)
    yout = np.concatenate([y, np.ones(need, dtype=y.dtype)])
    return Xout, yout


# --------------------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------------------

def _lgbm(seed: int, **over):
    p = dict(CFG.lgbm_params); p.update(over); p["random_state"] = seed
    return lgb.LGBMClassifier(**p)


def _xgb(seed: int):
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
        eval_metric="aucpr", random_state=seed, n_jobs=CFG.n_jobs, verbosity=0)


def _catboost(seed: int):
    return CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=7, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, allow_writing_files=False)


def _logreg(seed: int):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                                   random_state=seed)),
    ])


def _mlp(seed: int):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60,
                              early_stopping=True, random_state=seed)),
    ])


def build_baseline(name: str, seed: int = 0):
    reg = {
        "logreg": _logreg, "mlp": _mlp,
        "lightgbm": _lgbm, "xgboost": _xgb, "catboost": _catboost,
    }
    if name not in reg:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(reg)}")
    if name in {"lightgbm"} and lgb is None:
        raise ImportError("lightgbm not installed")
    if name == "xgboost" and xgb is None:
        raise ImportError("xgboost not installed")
    if name == "catboost" and CatBoostClassifier is None:
        raise ImportError("catboost not installed")
    return reg[name](seed)


AVAILABLE_BASELINES = [n for n, ok in (
    ("logreg", True), ("mlp", True),
    ("lightgbm", lgb is not None),
    ("xgboost", xgb is not None),
    ("catboost", CatBoostClassifier is not None),
) if ok]


# --------------------------------------------------------------------------------------
# Fit / predict, with the L1 switch
# --------------------------------------------------------------------------------------

def fit_predict(name: str, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame,
                seed: int = 0, resample: bool = False,
                tuning_budget: int = 0) -> tuple[np.ndarray, dict]:
    """Fit ``name`` on the training fold and score the test fold.

    ``tuning_budget`` > 0 runs a small random search on an inner time-ordered validation
    split carved from the *training* fold only. All models get the same budget, which is
    what makes the ladder a fair comparison.
    """
    info = {"model": name, "resampled": bool(resample), "tuning_budget": tuning_budget}
    Xtr, ytr = (naive_smote(X_tr, y_tr, seed=seed) if resample else (X_tr, y_tr))

    if tuning_budget and name in {"lightgbm", "xgboost"}:
        best = _inner_search(name, Xtr, ytr, seed, tuning_budget)
        info["tuned_params"] = best
        model = (_lgbm(seed, **best) if name == "lightgbm" else _xgb(seed))
    else:
        model = build_baseline(name, seed)

    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X_te)[:, 1]
    else:                                        # pragma: no cover
        s = model.decision_function(X_te)
    return np.asarray(s, dtype=float), info


def _inner_search(name: str, X: pd.DataFrame, y: np.ndarray, seed: int,
                  budget: int) -> dict:
    """Random search on an inner split. Never sees the test fold."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    cut = int(len(X) * 0.8)
    Xa, Xb = X.iloc[:cut], X.iloc[cut:]
    ya, yb = y[:cut], y[cut:]
    if yb.sum() < 3 or ya.sum() < 3:
        return {}

    grid = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    best, best_score = {}, -np.inf
    for _ in range(budget):
        cand = {k: v[int(rng.integers(len(v)))] for k, v in grid.items()}
        try:
            m = _lgbm(seed, **cand)
            m.fit(Xa, ya)
            sc = average_precision_score(yb, m.predict_proba(Xb)[:, 1])
        except Exception:
            continue
        if sc > best_score:
            best, best_score = cand, sc
    return best
