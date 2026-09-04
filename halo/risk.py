"""Block B -- latency-gated empirical-Bayes association risk.

Guilt by association, computed honestly. For each attribute value we maintain a
time-decayed Beta posterior over its fraud rate, updated only from labels that have
actually matured.

    r(a, v, t) = (alpha0 + sum_i w(t - t_i) y_i) / (alpha0 + beta0 + sum_i w(t - t_i))
      over i : t_i < t - delta  and  a_i = v
      w(D)  = exp(-D / tau)

Three design decisions carry the method:

1. **The Beta prior is what makes this not target encoding.** Shrinkage toward the
   population rate kills the small-count blow-up that sinks naive mean encoding: a value
   seen twice, once fraudulent, scores near the prior rather than at 0.5.

2. **Leave-one-entity-out.** Every risk feature subtracts the *querying entity's own*
   contribution to the counter. Without this an entity's own past labels leak in through
   the back door and the "association" feature silently becomes an entity-memory feature
   -- which would destroy the very separation C1 needs to make. This is the single most
   important line in the module.

3. **Strictly causal ingestion.** A separate pointer admits a transaction's label only
   once ``t_i < t - delta``, with a strict inequality, so same-timestamp rows can never
   see each other (adversarial review item 4).

Entity-memory features live here too but in their own ``mem_`` namespace, so T4 can
ablate memory and behaviour apart.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CFG


# --------------------------------------------------------------------------------------
# Empirical Bayes prior
# --------------------------------------------------------------------------------------

def fit_eb_prior(values: pd.Series, y: np.ndarray,
                 min_count: int | None = None) -> tuple[float, float]:
    """Moment-match a Beta(alpha0, beta0) to the per-value fraud rates.

    MUST be called on training rows only (adversarial review item 3).
    """
    min_count = min_count or CFG.eb_min_count
    tab = pd.DataFrame({"v": values.astype(str).to_numpy(), "y": y})
    g = tab.groupby("v")["y"].agg(["sum", "count"])
    g = g[g["count"] >= min_count]
    if len(g) < 5:
        m = float(y.mean()) if len(y) else 0.01
        m = min(max(m, 1e-4), 1 - 1e-4)
        return m * 20.0, (1 - m) * 20.0            # weak prior at the population rate

    p = (g["sum"] / g["count"]).to_numpy()
    m, v = float(p.mean()), float(p.var(ddof=1))
    m = min(max(m, 1e-4), 1 - 1e-4)
    if v <= 1e-12 or v >= m * (1 - m):
        return m * 20.0, (1 - m) * 20.0
    strength = m * (1 - m) / v - 1
    strength = float(np.clip(strength, 1.0, 5_000.0))
    return m * strength, (1 - m) * strength


# --------------------------------------------------------------------------------------
# Streaming risk accumulator
# --------------------------------------------------------------------------------------

class _Accum:
    """Time-decayed (positive, total) counters with lazy decay."""

    __slots__ = ("pos", "tot", "t")

    def __init__(self):
        self.pos = 0.0
        self.tot = 0.0
        self.t = 0.0

    def decay_to(self, t: float, tau_s: float) -> None:
        if self.tot == 0.0 and self.pos == 0.0:
            self.t = t
            return
        dt = t - self.t
        if dt > 0:
            f = math.exp(-dt / tau_s)
            self.pos *= f
            self.tot *= f
            self.t = t

    def add(self, y: float, t: float, tau_s: float) -> None:
        self.decay_to(t, tau_s)
        self.pos += y
        self.tot += 1.0


class AssociationRisk:
    """Single-pass, O(1)-per-transaction association risk over a heterogeneous attribute set.

    Usage::

        ar = AssociationRisk(attributes, delta_days=30)
        ar.fit_prior(train_df, train_y)          # training rows only
        feats = ar.transform(full_df_sorted_by_time, y_for_ingestion, train_mask)

    ``y_for_ingestion`` supplies labels for the counter updates. Rows outside the
    training window contribute nothing, because in deployment their labels would not be
    known -- that is enforced by ``ingest_mask``, not by trust.
    """

    def __init__(self, attributes: tuple[str, ...] | list[str] | None = None,
                 delta_days: int = 0, tau_days: float | None = None,
                 gamma: float | None = None):
        self.attributes = list(attributes or CFG.risk_attributes)
        self.delta_s = float(delta_days) * 86_400.0
        self.tau_s = float(tau_days if tau_days is not None else CFG.tau_days) * 86_400.0
        self.gamma = gamma if gamma is not None else CFG.gamma_damping
        self.priors: dict[str, tuple[float, float]] = {}

    # ---- prior -----------------------------------------------------------------------
    def fit_prior(self, df: pd.DataFrame, y: np.ndarray) -> "AssociationRisk":
        for a in self.attributes:
            if a in df.columns:
                self.priors[a] = fit_eb_prior(df[a], y)
        return self

    # ---- main pass -------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, entity: np.ndarray,
                  ingest_mask: np.ndarray | None = None) -> pd.DataFrame:
        """Compute risk, memory and velocity features for every row of ``df``.

        ``df`` must be sorted by ``TransactionDT`` ascending. ``ingest_mask`` marks rows
        whose labels may ever be used (i.e. rows inside the training window).
        """
        if not df["TransactionDT"].is_monotonic_increasing:
            raise ValueError("AssociationRisk.transform requires time-sorted input")

        n = len(df)
        t = df["TransactionDT"].to_numpy(dtype=np.float64)
        y = df["isFraud"].to_numpy(dtype=np.float64)
        ingest = (np.ones(n, dtype=bool) if ingest_mask is None
                  else np.asarray(ingest_mask, dtype=bool))
        attrs = [a for a in self.attributes if a in df.columns]
        avals = {a: df[a].astype(str).to_numpy() for a in attrs}

        # counters --------------------------------------------------------------------
        glob: dict[str, _Accum] = {a: _Accum() for a in attrs}
        val: dict[str, dict] = {a: {} for a in attrs}
        val_ent: dict[str, dict] = {a: {} for a in attrs}   # leave-one-entity-out
        ent_mem: dict[int, dict] = {}

        out_risk = {a: np.full(n, np.nan, dtype=np.float32) for a in attrs}
        out_cnt = {a: np.zeros(n, dtype=np.float32) for a in attrs}
        risk_max = np.zeros(n, dtype=np.float32)
        risk_mean = np.zeros(n, dtype=np.float32)
        risk_prop = np.zeros(n, dtype=np.float32)

        mem_known = np.zeros(n, dtype=np.int8)
        mem_prior_fraud = np.zeros(n, dtype=np.float32)
        mem_days_since = np.full(n, np.nan, dtype=np.float32)
        mem_prior_txn = np.zeros(n, dtype=np.float32)

        vel_1d = np.zeros(n, dtype=np.float32)
        vel_7d = np.zeros(n, dtype=np.float32)
        vel_amt7 = np.zeros(n, dtype=np.float32)
        recency = np.full(n, np.nan, dtype=np.float32)

        amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
        ent_hist: dict[int, list] = {}

        ingest_ptr = 0
        for i in range(n):
            ti = t[i]
            cutoff = ti - self.delta_s

            # --- admit every label that has matured, strictly before the cutoff -------
            while ingest_ptr < n and t[ingest_ptr] < cutoff:
                j = ingest_ptr
                if ingest[j]:
                    yj, tj, ej = y[j], t[j], entity[j]
                    for a in attrs:
                        v = avals[a][j]
                        glob[a].add(yj, tj, self.tau_s)
                        val[a].setdefault(v, _Accum()).add(yj, tj, self.tau_s)
                        val_ent[a].setdefault((v, ej), _Accum()).add(yj, tj, self.tau_s)
                    m = ent_mem.setdefault(ej, {"pos": 0.0, "n": 0.0, "first_fraud": None})
                    m["pos"] += yj
                    m["n"] += 1.0
                    if yj > 0 and m["first_fraud"] is None:
                        m["first_fraud"] = tj
                ingest_ptr += 1

            # --- score the current row -----------------------------------------------
            ei = entity[i]
            rs = []
            for a in attrs:
                a0, b0 = self.priors.get(a, (1.0, 99.0))
                v = avals[a][i]
                acc = val[a].get(v)
                own = val_ent[a].get((v, ei))
                if acc is None:
                    r = a0 / (a0 + b0)
                    c = 0.0
                else:
                    acc.decay_to(ti, self.tau_s)
                    pos, tot = acc.pos, acc.tot
                    if own is not None:                     # leave-one-entity-out
                        own.decay_to(ti, self.tau_s)
                        pos -= own.pos
                        tot -= own.tot
                    pos = max(pos, 0.0); tot = max(tot, 0.0)
                    r = (a0 + pos) / (a0 + b0 + tot)
                    c = tot
                out_risk[a][i] = r
                out_cnt[a][i] = c
                rs.append(r)

            if rs:
                risk_max[i] = max(rs)
                risk_mean[i] = float(np.mean(rs))
                # One-hop damped propagation: the entity's neighbourhood risk, seen
                # through the attributes it shares, discounted by gamma.
                risk_prop[i] = self.gamma * risk_max[i] + (1 - self.gamma) * risk_mean[i]

            m = ent_mem.get(ei)
            if m is not None:
                mem_prior_fraud[i] = m["pos"]
                mem_prior_txn[i] = m["n"]
                mem_known[i] = 1 if m["pos"] > 0 else 0
                if m["first_fraud"] is not None:
                    mem_days_since[i] = (ti - m["first_fraud"]) / 86_400.0

            # --- velocity: the entity's own past transactions, no labels involved -----
            h = ent_hist.setdefault(ei, [])
            if h:
                recency[i] = (ti - h[-1][0]) / 86_400.0
                lo1, lo7 = ti - 86_400.0, ti - 7 * 86_400.0
                c1 = c7 = 0
                s7 = 0.0
                for (tt, aa) in reversed(h):
                    if tt < lo7:
                        break
                    c7 += 1
                    s7 += aa
                    if tt >= lo1:
                        c1 += 1
                vel_1d[i], vel_7d[i], vel_amt7[i] = c1, c7, s7
            h.append((ti, amt[i]))
            if len(h) > 64:                                  # bound memory on hot entities
                del h[0]

        feats = {f"risk_{a}": out_risk[a] for a in attrs}
        feats.update({f"riskn_{a}": out_cnt[a] for a in attrs})
        feats.update({
            "risk_max": risk_max, "risk_mean": risk_mean, "risk_prop": risk_prop,
            "mem_known_compromised": mem_known,
            "mem_prior_fraud_count": mem_prior_fraud,
            "mem_prior_txn_count": mem_prior_txn,
            "mem_days_since_first_fraud": mem_days_since,
            "vel_txn_1d": vel_1d, "vel_txn_7d": vel_7d, "vel_amt_7d": vel_amt7,
            "vel_recency_days": recency,
        })
        return pd.DataFrame(feats, index=df.index)

    # ---- names, for monotone constraints and ablations --------------------------------
    @staticmethod
    def risk_feature_names(cols) -> list[str]:
        """Features whose direction is known a priori: more neighbourhood fraud is worse."""
        return [c for c in cols
                if c.startswith("risk_") or c.startswith("mem_")
                or c in {"vel_txn_1d", "vel_txn_7d"}]

    @staticmethod
    def memory_feature_names(cols) -> list[str]:
        return [c for c in cols if c.startswith("mem_")]
