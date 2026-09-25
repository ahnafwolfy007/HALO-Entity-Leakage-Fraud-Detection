"""C1 -- latency-aware learning from right-censored fraud labels.

## The problem

A transaction labelled 0 today is not a negative. It is *right-censored*: "no
chargeback has arrived yet". v1 handles this by discarding every label inside
the maturity window -- at lambda=30 that is 20-33% of the training window, and
specifically its most recent, most distribution-relevant part.

Three ways to treat a transaction whose label has not matured:

    (a) drop it               -- v1. No label noise, but throws away data.
    (b) call it a negative    -- what most published pipelines do implicitly by
                                 using a static label column with a random
                                 split. No data loss, but injects noise whose
                                 rate grows as the window widens.
    (c) impute a soft label   -- this module.

For a transaction of age `a` not yet reported, with unconditional fraud
probability `p(x)` and report-delay CDF `F`:

    P(fraud | not reported by a, x) = p(x)(1 - F(a)) / (1 - p(x) F(a))

which is (b) when F(a)=1 and uninformative when F(a)=0.

## What is and is not observable

IEEE-CIS ships *final* labels with no report timestamp, so `F` cannot be read
off the data. This is a property of the benchmark, not an oversight, and it has
a consequence worth stating plainly in the paper:

> **Reporting delay must be simulated to be studied.**

So the experimental design is: take a window old enough that every label is
final, impose a delay model to generate the labels that *would have been
visible* at an observation time, and compare (a), (b) and (c) against the known
final labels. ``artificial_censoring_experiment`` implements exactly that.

Where a dataset does expose report times, ``DelayModel.fit_from_report_times``
estimates `F` directly and no simulation is needed.

## Identifiability

The Vesta rule supplies something most delayed-feedback settings lack: a known
hard horizon -- unreported at 120 days implies legitimate. `F` therefore has
compact known support [0, H], which pins down the tail that is otherwise the
least identifiable part of a delay distribution.

Prior art: Chapelle (KDD'14) on delayed feedback in display advertising; Ktena
et al. (2019); Yasui et al. (2020); Elkan & Noto (2008) on PU learning. All
assume censoring is *independent* across samples. Under entity-disjoint
evaluation it is not -- one compromised card produces many correlated labels
reported together -- which is why ``DelayModel`` supports entity-clustered
draws (``cluster_by_entity=True``) and why the naive estimator is biased here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Delay model
# --------------------------------------------------------------------------------------

@dataclass
class DelayModel:
    """CDF of the delay between a fraudulent transaction and its report.

    Parameters
    ----------
    kind
        ``exponential`` (memoryless, one parameter) or ``weibull`` (allows a
        front-loaded or heavy tail, two parameters).
    scale_days, shape
        Distribution parameters, in days.
    horizon_days
        Hard support boundary. Mass beyond it is renormalised into [0, H],
        encoding "unreported by H implies legitimate".
    cluster_by_entity
        Draw one delay per entity rather than per transaction, reflecting that
        a single chargeback resolves an entity's whole timeline at once. This
        is the dependence structure that breaks the standard IID estimators.
    """

    kind: str = "weibull"
    scale_days: float = 7.0
    shape: float = 1.2
    horizon_days: float | None = 120.0
    cluster_by_entity: bool = True

    # ---- cdf ------------------------------------------------------------------------
    def cdf(self, age_days: np.ndarray | float) -> np.ndarray:
        a = np.maximum(np.asarray(age_days, dtype=np.float64), 0.0)
        if self.kind == "exponential":
            f = 1.0 - np.exp(-a / self.scale_days)
        elif self.kind == "weibull":
            f = 1.0 - np.exp(-((a / self.scale_days) ** self.shape))
        else:
            raise ValueError(f"unknown delay kind {self.kind!r}")
        if self.horizon_days is not None:
            fh = self.cdf_raw(self.horizon_days)
            f = np.minimum(f / max(fh, 1e-12), 1.0)
            f = np.where(a >= self.horizon_days, 1.0, f)
        return f

    def cdf_raw(self, age_days: float) -> float:
        if self.kind == "exponential":
            return 1.0 - np.exp(-age_days / self.scale_days)
        return 1.0 - np.exp(-((age_days / self.scale_days) ** self.shape))

    # ---- sampling --------------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(n)
        if self.kind == "exponential":
            d = -self.scale_days * np.log1p(-u)
        else:
            d = self.scale_days * (-np.log1p(-u)) ** (1.0 / self.shape)
        if self.horizon_days is not None:
            d = np.minimum(d, self.horizon_days)
        return d

    # ---- fitting ---------------------------------------------------------------------
    @classmethod
    def fit_from_report_times(cls, delays_days: np.ndarray, horizon_days: float | None,
                              kind: str = "weibull") -> "DelayModel":
        """Direct fit where report times exist. Method of moments."""
        d = np.asarray(delays_days, dtype=np.float64)
        d = d[np.isfinite(d) & (d >= 0)]
        if len(d) < 10:
            return cls(kind=kind, horizon_days=horizon_days)
        m, s = float(d.mean()), float(d.std(ddof=1))
        if kind == "exponential" or s <= 0:
            return cls(kind="exponential", scale_days=max(m, 0.5),
                       horizon_days=horizon_days)
        cv = s / max(m, 1e-9)
        shape = float(np.clip(cv ** -1.086, 0.3, 6.0))     # Weibull CV approximation
        from math import gamma as gammafn
        scale = float(max(m / gammafn(1.0 + 1.0 / shape), 0.5))
        return cls(kind="weibull", scale_days=scale, shape=shape,
                   horizon_days=horizon_days)

    def describe(self) -> str:
        return (f"DelayModel({self.kind}, scale={self.scale_days:.2f}d, "
                f"shape={self.shape:.2f}, horizon={self.horizon_days}, "
                f"clustered={self.cluster_by_entity})")


# --------------------------------------------------------------------------------------
# Simulating what would have been visible
# --------------------------------------------------------------------------------------

def simulate_report_days(df: pd.DataFrame, model: DelayModel, seed: int = 0,
                         day_col: str = "day", y_col: str = "y",
                         entity_col: str = "entity") -> np.ndarray:
    """Report day per row: transaction day + delay for frauds, +inf for legitimate.

    With ``cluster_by_entity`` the delay is drawn once per entity, so an
    entity's fraudulent transactions all resolve together -- the correlated
    censoring that distinguishes this setting from display advertising.
    """
    rng = np.random.default_rng(seed)
    day = df[day_col].to_numpy(np.float64)
    y = df[y_col].to_numpy()
    out = np.full(len(df), np.inf, dtype=np.float64)
    fraud = np.flatnonzero(y == 1)
    if len(fraud) == 0:
        return out

    if model.cluster_by_entity and entity_col in df.columns:
        ent = df[entity_col].to_numpy()[fraud]
        uniq, inv = np.unique(ent, return_inverse=True)
        per_entity = model.sample(len(uniq), rng)
        delays = per_entity[inv]
    else:
        delays = model.sample(len(fraud), rng)

    out[fraud] = day[fraud] + delays
    return out


def visible_labels(df: pd.DataFrame, report_day: np.ndarray, observation_day: float,
                   day_col: str = "day") -> tuple[np.ndarray, np.ndarray]:
    """Labels observable at ``observation_day``.

    Returns
    -------
    y_visible : int8, 1 if already reported, else 0
    matured   : bool, True where the label is final (reported, or past horizon)
    """
    reported = report_day <= observation_day
    y_visible = reported.astype(np.int8)
    age = observation_day - df[day_col].to_numpy(np.float64)
    return y_visible, reported, age


# --------------------------------------------------------------------------------------
# The correction
# --------------------------------------------------------------------------------------

def soft_labels(p_hat: np.ndarray, age_days: np.ndarray, model: DelayModel,
                already_reported: np.ndarray) -> np.ndarray:
    """P(fraud | not yet reported, age) for unreported rows; 1.0 for reported ones."""
    p = np.clip(np.asarray(p_hat, dtype=np.float64), 1e-6, 1 - 1e-6)
    F = model.cdf(age_days)
    q = p * (1.0 - F) / np.maximum(1.0 - p * F, 1e-9)
    q = np.clip(q, 0.0, 1.0)
    return np.where(already_reported, 1.0, q)


def expand_soft(X: pd.DataFrame, q: np.ndarray, base_weight: np.ndarray | None = None,
                min_weight: float = 1e-3) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Turn fractional targets into a weighted hard-label set.

    A row with target q becomes two rows -- (y=1, w=q) and (y=0, w=1-q) --
    which is exactly equivalent to fractional-label cross-entropy and works
    with any binary learner, including LightGBM's native objective. Rows whose
    weight falls below ``min_weight`` are dropped to bound the expansion.
    """
    w0 = np.ones(len(X)) if base_weight is None else np.asarray(base_weight, float)
    q = np.asarray(q, dtype=np.float64)

    keep_pos = q > min_weight
    keep_neg = (1.0 - q) > min_weight

    X_out = pd.concat([X[keep_pos], X[keep_neg]], axis=0, ignore_index=True)
    y_out = np.concatenate([np.ones(keep_pos.sum(), dtype=np.int8),
                            np.zeros(keep_neg.sum(), dtype=np.int8)])
    w_out = np.concatenate([q[keep_pos] * w0[keep_pos],
                            (1.0 - q[keep_neg]) * w0[keep_neg]])
    return X_out, y_out, w_out


@dataclass
class CensoringConfig:
    mode: str = "soft"          # drop | zero | soft
    em_rounds: int = 2
    min_weight: float = 1e-3
    notes: str = ""


class CensoredLabelTrainer:
    """EM over censored labels: fit, impute, refit.

    ``mode``:
      ``drop``  reproduce v1 -- train only on matured rows.
      ``zero``  reproduce the common naive pipeline -- unmatured rows as 0.
      ``soft``  C1 -- unmatured rows enter with an imputed probability.
    """

    def __init__(self, fit_fn, predict_fn, model: DelayModel,
                 cfg: CensoringConfig | None = None):
        self.fit_fn = fit_fn            # (X, y, w) -> model
        self.predict_fn = predict_fn    # (model, X) -> p
        self.delay = model
        self.cfg = cfg or CensoringConfig()
        self.history_: list[dict] = []

    def fit(self, X: pd.DataFrame, y_visible: np.ndarray, reported: np.ndarray,
            age_days: np.ndarray, base_weight: np.ndarray | None = None):
        mode = self.cfg.mode
        if mode == "drop":
            m = reported | (age_days >= (self.delay.horizon_days or np.inf))
            w = None if base_weight is None else base_weight[m]
            return self.fit_fn(X[m], y_visible[m], w)

        if mode == "zero":
            return self.fit_fn(X, y_visible, base_weight)

        if mode != "soft":
            raise ValueError(f"unknown censoring mode {mode!r}")

        # Round 0: a conservative model from matured rows only.
        matured = reported | (age_days >= (self.delay.horizon_days or np.inf))
        w0 = None if base_weight is None else base_weight[matured]
        model = self.fit_fn(X[matured], y_visible[matured], w0)

        for rnd in range(self.cfg.em_rounds):
            p_hat = np.asarray(self.predict_fn(model, X), dtype=np.float64)
            q = soft_labels(p_hat, age_days, self.delay, reported)
            Xe, ye, we = expand_soft(X, q, base_weight, self.cfg.min_weight)
            model = self.fit_fn(Xe, ye, we)
            self.history_.append({
                "round": rnd, "n_expanded": int(len(Xe)),
                "mean_q_unreported": float(q[~reported].mean()) if (~reported).any() else 0.0,
                "imputed_positive_mass": float(q[~reported].sum()),
            })
        return model


# --------------------------------------------------------------------------------------
# Validation harness
# --------------------------------------------------------------------------------------

def artificial_censoring_experiment(df: pd.DataFrame, X: pd.DataFrame,
                                    fit_fn, predict_fn, score_fn,
                                    observation_day: float,
                                    model: DelayModel,
                                    modes: tuple[str, ...] = ("drop", "zero", "soft"),
                                    seed: int = 0) -> pd.DataFrame:
    """Compare censoring strategies against known final labels.

    Because the benchmark ships final labels, reporting delay is *imposed*
    rather than observed: we generate the labels that would have been visible
    at ``observation_day`` under ``model``, train each strategy on those, and
    score against the true labels. This is the only honest way to measure a
    delay correction on a dataset without report timestamps, and the simulated
    delay distribution must be reported alongside the result.
    """
    report_day = simulate_report_days(df, model, seed=seed)
    y_vis, reported, age = visible_labels(df, report_day, observation_day)
    y_true = df["y"].to_numpy()

    rows = []
    for mode in modes:
        trainer = CensoredLabelTrainer(fit_fn, predict_fn, model,
                                       CensoringConfig(mode=mode))
        fitted = trainer.fit(X, y_vis, reported, age)
        rows.append({
            "mode": mode,
            "observation_day": observation_day,
            "delay": model.describe(),
            "n_visible_positive": int(y_vis.sum()),
            "n_true_positive": int(y_true.sum()),
            "censored_positive_mass": int(y_true.sum() - y_vis.sum()),
            **score_fn(fitted, X, y_true),
            **({"em": trainer.history_[-1]} if trainer.history_ else {}),
        })
    return pd.DataFrame(rows)
