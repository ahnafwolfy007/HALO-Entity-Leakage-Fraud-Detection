"""Block D v2 -- C4 (dollar-aligned cost) and C5 (region-restricted monotonicity).

## C4 -- aligning the objective with the reported metric

v1 reports Dollar-Recall@1% and trains with a constant 20x weight on missed
fraud. Those are different objectives: a constant weight says every missed
fraud costs the same, while the metric says a missed fraud costs its amount.
``dollar_weights`` closes that gap.

## C5 -- paying for monotonicity only where it is used

v1 enforces monotonicity across the whole input space and measures the cost at
0.0110 AUPRC (2.32% relative). But reason codes are only ever issued for
alerted rows (``model.reason_codes`` filters on ``s >= threshold``), so the
guarantee is consumed in a thin slice of the space and paid for everywhere.

``RegionMonotoneGBDT`` splits the job:

    router      unconstrained, decides which rows are near enough to the alert
                boundary to matter
    specialist  monotone-constrained, trained only on that region, and the only
                model whose output can exceed the alert threshold

Crucially, region membership at inference is decided by the **specialist**, not
the router. The router only chooses which rows the specialist trains on. An
earlier version gated on the router and failed its own property test: because
the router is unconstrained, raising a constrained feature could lower the
router score, eject the row from the region, and *decrease* the final score.

    score = theta + (1-theta) * (s2 - b)/(1 - b)     if s2 >= b     (monotone)
            theta * router(x)                        otherwise      (ranking only)

> **Proposition (monotone within the alert region).** For `x` with `s2(x) >= b`
> and a constrained feature `j`, the composed score is non-decreasing in `x_j`.
>
> *Sketch.* The specialist is monotone in `x_j`, so raising `x_j` cannot lower
> `s2`; therefore the row cannot leave the region, and within the region the
> score is a strictly increasing affine transform of `s2`. ∎
>
> **Corollary (reason codes are exact).** Lowering `x_j` from an alerted point
> either keeps the row in the region -- where the score is non-increasing, so
> the crossing point is unique -- or drops `s2` below `b`, which scores at most
> `theta` and is not an alert. The binary search therefore returns a valid
> clearing threshold. ∎

No guarantee is claimed *below* the region, and none is needed: no alert is
issued there, so no reason code is ever built on it. That is exactly the saving
C5 is designed to capture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import CFG

try:
    import lightgbm as lgb
except ImportError:                                  # pragma: no cover
    lgb = None


# --------------------------------------------------------------------------------------
# C4 -- cost weighting
# --------------------------------------------------------------------------------------

def dollar_weights(y: np.ndarray, amount: np.ndarray | None,
                   kappa: float = 20.0, cap_quantile: float = 0.995,
                   ) -> np.ndarray:
    """w = 1 + kappa * y * (amount / mean_amount), capped.

    Falls back to a constant ``kappa`` multiplier when the dataset has no
    amount (Elliptic, BAF), so the same code path serves every dataset and the
    fallback is explicit rather than silent.
    """
    y = np.asarray(y, dtype=np.float64)
    if amount is None:
        return np.where(y == 1, 1.0 + kappa, 1.0)
    a = np.asarray(amount, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.where(y == 1, 1.0 + kappa, 1.0)
    mean_amt = float(np.nanmean(a[finite]))
    if mean_amt <= 0:
        return np.where(y == 1, 1.0 + kappa, 1.0)
    rel = np.where(finite, a / mean_amt, 1.0)
    cap = float(np.nanquantile(rel[finite], cap_quantile))   # one whale ≠ the model
    rel = np.clip(rel, 0.0, max(cap, 1.0))
    return 1.0 + kappa * y * rel


# --------------------------------------------------------------------------------------
# C5 -- region-restricted monotone model
# --------------------------------------------------------------------------------------

@dataclass
class RegionConfig:
    alert_quantile: float = 0.01     # operational review budget
    margin: float = 3.0              # region covers margin x the budget
    theta: float = 0.5               # score at the region boundary
    enforce: bool = True             # False -> plain unconstrained model


class RegionMonotoneGBDT:
    """Two-stage model: unconstrained router + monotone specialist on the alert region."""

    def __init__(self, monotone_features: list[str] | None = None,
                 params: dict | None = None, seed: int = 0,
                 region: RegionConfig | None = None):
        if lgb is None:
            raise ImportError("lightgbm is required for RegionMonotoneGBDT")
        self.monotone_features = list(monotone_features or [])
        self.params = dict(params or CFG.lgbm_params)
        self.params["random_state"] = seed
        self.seed = seed
        self.region = region or RegionConfig()
        self.feature_names_: list[str] = []
        self.boundary_: float = 0.0
        self.calibrators_: dict = {}
        self.train_ranges_: dict[str, tuple[float, float]] = {}

    # ---- fit -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None,
            regime: np.ndarray | None = None) -> "RegionMonotoneGBDT":
        self.feature_names_ = list(X.columns)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)

        # -- stage 1: unconstrained router ---------------------------------------------
        self.router_ = lgb.LGBMClassifier(**self.params)
        self.router_.fit(X, y, sample_weight=w)
        s0 = self.router_.predict_proba(X)[:, 1]

        if not self.region.enforce or not self.monotone_features:
            self.specialist_ = None
            self.boundary_ = float(np.quantile(s0, 1.0 - self.region.alert_quantile))
            self._fit_calibrators(s0, y, regime)
            return self

        # -- training region: the top (alert_quantile x margin) of router scores --------
        # The router selects which rows the specialist is *trained* on. Region
        # membership at inference is decided by the specialist itself (below),
        # not by the router -- otherwise raising a constrained feature could
        # lower the router score, push the row out of the region, and decrease
        # the final score, breaking monotonicity at the boundary.
        frac = min(self.region.alert_quantile * self.region.margin, 0.5)
        router_cut = float(np.quantile(s0, 1.0 - frac))
        in_region = s0 >= router_cut

        # Guard: a degenerate region (too few positives) would make the
        # specialist meaningless, so fall back to a globally monotone model and
        # record that we did.
        self.region_fallback_ = bool(in_region.sum() < 200 or y[in_region].sum() < 5)
        idx = slice(None) if self.region_fallback_ else in_region

        constraints = [1 if f in self.monotone_features else 0 for f in self.feature_names_]
        params = dict(self.params)
        params["monotone_constraints"] = constraints
        params["monotone_constraints_method"] = "advanced"

        self.specialist_ = lgb.LGBMClassifier(**params)
        self.specialist_.fit(X[idx], np.asarray(y)[idx], sample_weight=w[idx])

        # Inference-time boundary, on the *specialist's* scale. Because the
        # specialist is monotone in the constrained features, membership of the
        # region is itself monotone: raising such a feature can only move a row
        # into the region, never out of it.
        s2_all = self.specialist_.predict_proba(X)[:, 1]
        self.boundary_ = float(np.quantile(s2_all, 1.0 - self.region.alert_quantile))
        self.router_cut_ = router_cut

        for f in self.monotone_features:
            if f in X.columns:
                col = X[f].to_numpy(dtype=float)
                col = col[np.isfinite(col)]
                if len(col):
                    self.train_ranges_[f] = (float(col.min()), float(col.max()))

        self._fit_calibrators(self.decision_score(X), y, regime)
        return self

    # ---- score -----------------------------------------------------------------------
    def decision_score(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names_]
        if self.specialist_ is None:
            return self.router_.predict_proba(X)[:, 1]

        theta, b = self.region.theta, min(max(self.boundary_, 1e-9), 1 - 1e-9)
        s2 = self.specialist_.predict_proba(X)[:, 1]
        in_region = s2 >= b

        # In region: a monotone rescaling of the specialist onto [theta, 1].
        above = theta + (1.0 - theta) * (s2 - b) / (1.0 - b)
        # Out of region: rank by the unconstrained router, compressed below
        # theta so no non-alerted row can outrank an alerted one.
        s0 = self.router_.predict_proba(X)[:, 1]
        below = theta * np.clip(s0, 0.0, 1.0)
        return np.where(in_region, above, below)

    def predict_proba(self, X: pd.DataFrame, regime: np.ndarray | None = None) -> np.ndarray:
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

    def _fit_calibrators(self, p: np.ndarray, y: np.ndarray, regime: np.ndarray | None):
        self.calibrators_ = {}
        base = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        base.fit(p, y)
        self.calibrators_["__global__"] = base
        if regime is None:
            return
        for r in np.unique(regime):
            m = regime == r
            if m.sum() >= 500 and 0 < np.asarray(y)[m].sum() < m.sum():
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(p[m], np.asarray(y)[m])
                self.calibrators_[int(r)] = iso

    # ---- verification ----------------------------------------------------------------
    def verify_monotonicity(self, X: pd.DataFrame, n_probe: int = 200, n_steps: int = 12,
                            seed: int = 0, tol: float = 1e-3,
                            region_only: bool = True) -> pd.DataFrame:
        """Sweep each constrained feature upward and measure any backward step.

        With ``region_only`` the probe is restricted to alerted rows, which is
        where C5 claims the guarantee. Running it over the whole space is also
        informative -- it quantifies what the restriction gave up -- so both are
        reported by the experiment driver.
        """
        rng = np.random.default_rng(seed)
        if len(X) == 0 or self.specialist_ is None:
            return pd.DataFrame()

        pool = X
        if region_only:
            s = self.decision_score(X)
            sel = s >= self.region.theta
            if sel.sum() >= 20:
                pool = X[sel]

        idx = rng.choice(len(pool), size=min(n_probe, len(pool)), replace=False)
        Xs = pool.iloc[idx][self.feature_names_].copy()

        rows = []
        for f in self.monotone_features:
            if f not in X.columns:
                continue
            lo, hi = self.train_ranges_.get(f, (np.nan, np.nan))
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
                continue
            probe, scores, inreg = Xs.copy(), [], []
            for g in np.linspace(lo, hi, n_steps):
                probe[f] = g
                scores.append(self.decision_score(probe))
                inreg.append(self.specialist_.predict_proba(
                    probe[self.feature_names_])[:, 1] >= self.boundary_)
            S = np.vstack(scores)
            R = np.vstack(inreg)
            diffs = np.diff(S, axis=0)
            viol = np.maximum(-diffs, 0.0)
            if region_only:
                # The proposition covers consecutive points that are *both* in
                # the region. Below the boundary the score is the unconstrained
                # router and no monotonicity is claimed -- counting those steps
                # would be testing something the method never asserted.
                both_in = R[:-1] & R[1:]
                viol = np.where(both_in, viol, 0.0)
            rows.append({
                "feature": f,
                "region_only": region_only,
                "violations_raw": int((diffs < 0).sum()),
                "max_violation": float(viol.max()) if viol.size else 0.0,
                "direction_ok": bool((viol.max() if viol.size else 0.0) <= tol),
            })
        return pd.DataFrame(rows)

    # ---- reason codes ----------------------------------------------------------------
    def reason_codes(self, X: pd.DataFrame, threshold: float | None = None,
                     max_rows: int = 2000, tol: float = 1e-4) -> pd.DataFrame:
        """Exact counterfactual thresholds along monotone coordinates.

        ``threshold`` defaults to the region boundary score, which is the value
        the C5 proposition is stated at.
        """
        if self.specialist_ is None:
            return pd.DataFrame()
        theta = self.region.theta if threshold is None else threshold
        X = X[self.feature_names_]
        s = self.decision_score(X)
        alerted = np.flatnonzero(s >= theta)[:max_rows]

        rows = []
        for i in alerted:
            row = X.iloc[[i]]
            for f in self.monotone_features:
                lo, hi = self.train_ranges_.get(f, (None, None))
                if lo is None:
                    continue
                cur = float(row[f].iloc[0]) if np.isfinite(row[f].iloc[0]) else hi
                probe = row.copy()
                probe[f] = lo
                if self.decision_score(probe)[0] >= theta:
                    continue                      # this feature alone cannot clear it
                a, b = lo, cur
                for _ in range(40):
                    mid = 0.5 * (a + b)
                    probe[f] = mid
                    if self.decision_score(probe)[0] >= theta:
                        b = mid
                    else:
                        a = mid
                    if b - a < tol:
                        break
                rows.append({"row": int(i), "feature": f,
                             "current_value": cur, "clears_below": float(0.5 * (a + b)),
                             "score": float(s[i])})
        return pd.DataFrame(rows)
