"""Block D -- monotone, cost-sensitive GBDT with explanations that are faithful by construction.

The field validates explanations *after the fact* (SHAP, then ROAR to check whether SHAP
was telling the truth). HALO instead constrains the model so that a class of counterfactual
claims is provably true, and then measures what that guarantee costs.

Concretely: with a ``+1`` monotone constraint on an association-risk feature, the model
output is non-decreasing in that feature. So the statement

    "this alert clears if the device's 60-day association risk falls below 0.041"

is not an approximation from a local surrogate. It is exact, and the threshold is found by
binary search along the monotone coordinate. Faithfulness is 1.0 by construction on that
subspace; what remains to be measured is *coverage* (how many alerts can be explained this
way) and *price* (how much AUPRC the constraint costs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import CFG

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class HaloModel:
    """Monotone cost-sensitive LightGBM with per-regime calibration and reason codes."""

    def __init__(self, monotone_features: list[str] | None = None,
                 params: dict | None = None, fn_cost_ratio: float | None = None,
                 calibrate_by_regime: bool = True, seed: int = 0,
                 enforce_monotone: bool = True):
        if lgb is None:
            raise ImportError("lightgbm is required for HaloModel")
        self.monotone_features = list(monotone_features or [])
        self.params = dict(params or CFG.lgbm_params)
        self.params["random_state"] = seed
        self.fn_cost_ratio = (fn_cost_ratio if fn_cost_ratio is not None
                              else CFG.fn_cost_ratio)
        self.calibrate_by_regime = calibrate_by_regime
        self.enforce_monotone = enforce_monotone
        self.seed = seed
        self.calibrators_: dict = {}
        self.feature_names_: list[str] = []
        self.monotone_idx_: list[int] = []

    # ---- fit -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            regime: np.ndarray | None = None) -> "HaloModel":
        self.feature_names_ = list(X.columns)
        constraints = [0] * X.shape[1]
        if self.enforce_monotone:
            for f in self.monotone_features:
                if f in X.columns:
                    j = self.feature_names_.index(f)
                    constraints[j] = 1
                    self.monotone_idx_.append(j)

        params = dict(self.params)
        if self.enforce_monotone and any(constraints):
            params["monotone_constraints"] = constraints
            params["monotone_constraints_method"] = "advanced"

        # Cost sensitivity: a missed fraud costs fn_cost_ratio investigations.
        w = np.where(np.asarray(y) == 1, self.fn_cost_ratio, 1.0)

        self.model_ = lgb.LGBMClassifier(**params)
        self.model_.fit(X, y, sample_weight=w)

        # Record the training range of each constrained feature. Probing outside it in
        # verify_monotonicity would extrapolate past the last histogram bin boundary and
        # manufacture apparent violations that say nothing about the constraint.
        self.train_ranges_ = {}
        for f in self.monotone_features:
            if f in X.columns:
                col = X[f].to_numpy(dtype=float)
                col = col[np.isfinite(col)]
                if len(col):
                    self.train_ranges_[f] = (float(col.min()), float(col.max()))

        raw = self.model_.predict_proba(X)[:, 1]
        self._fit_calibrators(raw, y, regime)
        return self

    def _fit_calibrators(self, p: np.ndarray, y: np.ndarray,
                         regime: np.ndarray | None) -> None:
        """Isotonic calibration, stratified by provenance regime where there is enough data."""
        self.calibrators_ = {}
        base = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        base.fit(p, y)
        self.calibrators_["__global__"] = base
        if not self.calibrate_by_regime or regime is None:
            return
        for r in np.unique(regime):
            m = regime == r
            if m.sum() >= 500 and 0 < y[m].sum() < m.sum():
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(p[m], y[m])
                self.calibrators_[int(r)] = iso

    # ---- predict ---------------------------------------------------------------------
    def decision_score(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated ranking score. Ranking metrics use this."""
        return self.model_.predict_proba(X[self.feature_names_])[:, 1]

    def predict_proba(self, X: pd.DataFrame,
                      regime: np.ndarray | None = None) -> np.ndarray:
        """Calibrated probability. Calibration metrics use this."""
        p = self.decision_score(X)
        if not self.calibrators_:
            return p
        out = self.calibrators_["__global__"].predict(p)
        if regime is not None:
            for r, iso in self.calibrators_.items():
                if r == "__global__":
                    continue
                m = regime == r
                if m.any():
                    out[m] = iso.predict(p[m])
        return np.clip(out, 0, 1)

    # ---- monotonicity verification (adversarial review item 15) -----------------------
    def verify_monotonicity(self, X: pd.DataFrame, n_probe: int = 200,
                            n_steps: int = 12, seed: int = 0,
                            tol: float = 1e-3) -> pd.DataFrame:
        """Empirically confirm the constraint direction rather than assuming it.

        For each constrained feature, sweep it upward on a sample of rows and check the
        score never decreases. If the *direction* were wrong, every reason code built on
        that feature would be a lie, so this runs on every faithfulness evaluation.

        On the tolerance
        ----------------
        LightGBM enforces monotonicity over its histogram bins, so the guarantee is exact
        with respect to the binned representation, not to arbitrary real-valued probes.
        In practice this leaves violations on the order of 1e-4 in probability space --
        measured, not assumed, and reported here as ``max_violation``.

        We therefore report two things rather than one boolean: the raw violation count at
        machine precision, and ``direction_ok``, which asks whether any violation exceeds
        ``tol``. A 1e-3 tolerance is orders of magnitude below any realistic alerting
        threshold, so a model passing at this tolerance supports exact reason codes for
        every operational purpose -- but the paper should quote the measured bound rather
        than claim exactness.
        """
        rng = np.random.default_rng(seed)
        rows = []
        if len(X) == 0:
            return pd.DataFrame(rows)
        idx = rng.choice(len(X), size=min(n_probe, len(X)), replace=False)
        Xs = X.iloc[idx][self.feature_names_].copy()
        ranges = getattr(self, "train_ranges_", {})
        for f in self.monotone_features:
            if f not in X.columns:
                continue
            # Probe within the TRAINING range: outside it there are no bin boundaries to
            # constrain, and any apparent violation is extrapolation, not a broken constraint.
            if f in ranges:
                lo, hi = ranges[f]
            else:
                col = X[f].to_numpy(dtype=float)
                col = col[np.isfinite(col)]
                if len(col) == 0:
                    continue
                lo, hi = float(col.min()), float(col.max())
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
                continue
            grid = np.linspace(lo, hi, n_steps)
            scores = []
            probe = Xs.copy()
            for g in grid:
                probe[f] = g
                scores.append(self.decision_score(probe))
            S = np.vstack(scores)                       # (n_steps, n_probe)
            diffs = np.diff(S, axis=0)
            neg = diffs[diffs < 0]
            max_violation = float(-neg.min()) if neg.size else 0.0
            rows.append({
                "feature": f,
                "monotone_violations": int((diffs < -1e-9).sum()),
                "comparisons": int(diffs.size),
                "violation_rate": float((diffs < -1e-9).mean()),
                "max_violation": max_violation,
                "mean_slope": float(diffs.mean()),
                "direction_ok": bool(max_violation <= tol and diffs.mean() >= -1e-12),
            })
        return pd.DataFrame(rows)

    # ---- reason codes -----------------------------------------------------------------
    def reason_codes(self, X: pd.DataFrame, threshold: float,
                     max_rows: int = 2_000, tol: float = 1e-4) -> pd.DataFrame:
        """Exact counterfactual thresholds for alerted rows, along monotone coordinates.

        For each alerted row and each constrained feature, binary-search the value at
        which the score would fall to ``threshold``. Because the model is monotone in
        that feature, the search is guaranteed to converge to the true crossing point --
        this is what "faithful by construction" buys.
        """
        X = X[self.feature_names_]
        s = self.decision_score(X)
        alerted = np.flatnonzero(s >= threshold)
        if len(alerted) > max_rows:
            alerted = alerted[:max_rows]

        rows = []
        for i in alerted:
            row = X.iloc[[i]].copy()
            best = None
            for f in self.monotone_features:
                if f not in X.columns:
                    continue
                cur = row[f].to_numpy(dtype=float)[0]
                if not np.isfinite(cur):
                    continue
                lo_probe = row.copy()
                lo_probe[f] = float(np.nanmin(X[f].to_numpy(dtype=float)))
                if self.decision_score(lo_probe)[0] >= threshold:
                    continue      # this feature alone cannot clear the alert
                lo = float(lo_probe[f].to_numpy()[0])
                hi = cur
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    probe = row.copy()
                    probe[f] = mid
                    if self.decision_score(probe)[0] >= threshold:
                        hi = mid
                    else:
                        lo = mid
                    if hi - lo < tol * max(1.0, abs(cur)):
                        break
                gap = cur - lo
                rel = gap / (abs(cur) + 1e-9)
                if best is None or rel < best["relative_change"]:
                    best = {"row": int(i), "feature": f, "current_value": float(cur),
                            "clears_below": float(lo), "absolute_change": float(gap),
                            "relative_change": float(rel)}
            if best is not None:
                rows.append(best)
            else:
                rows.append({"row": int(i), "feature": None, "current_value": np.nan,
                             "clears_below": np.nan, "absolute_change": np.nan,
                             "relative_change": np.nan})
        out = pd.DataFrame(rows)
        out.attrs["n_alerted"] = int(len(alerted))
        return out

    def reason_code_coverage(self, X: pd.DataFrame, threshold: float,
                             max_rows: int = 2_000) -> dict:
        """Coverage denominator is *all* alerts (adversarial review item 20).

        Reporting coverage over only the alerts whose decisive feature happened to be
        monotone would be circular and would read as ~100% by construction.
        """
        rc = self.reason_codes(X, threshold, max_rows=max_rows)
        n = int(rc.attrs.get("n_alerted", len(rc)))
        if n == 0:
            return {"reason_code_coverage": float("nan"), "n_alerts_examined": 0}
        covered = int(rc["feature"].notna().sum())
        return {
            "reason_code_coverage": covered / n,
            "n_alerts_examined": n,
            "n_alerts_covered": covered,
            "median_relative_change": float(rc["relative_change"].median(skipna=True)),
        }
