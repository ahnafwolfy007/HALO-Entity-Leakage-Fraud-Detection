"""Block B v2 -- association risk with real propagation, bounded memory, uncertainty.

Differences from v1 (`risk.py`), each independently ablatable:

| change | what it fixes |
|---|---|
| ``MultiHopPropagator`` | V1: v1's "propagation" never left the row |
| ``AttributeCounters``  | B1: unbounded dict store, 26 GB at TabFormer scale |
| posterior variance     | C3: v1 emitted the mean and discarded the uncertainty |
| per-attribute / multi-scale tau | V3: one global 30-day half-life for every attribute |
| maturity gate kept exact | the strict ``t_i < t - delta`` inequality from v1 |

The streaming contract is unchanged and is what keeps the whole thing
leakage-free: rows are processed in time order, a label is admitted to the
counters only once it has matured, and the row being scored never contributes
to its own features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG
from .counters import fit_beta_prior
from .propagation import MultiHopPropagator

DAY = 86_400.0


class AssociationRiskV2:
    """Single-pass, O(1)-per-row association risk over a heterogeneous attribute set.

    Parameters
    ----------
    attributes
        Columns participating in the bipartite graph.
    delta_days
        Label maturity gate. A label is admitted only once
        ``t_label < t_now - delta``, strictly.
    tau_days
        Primary decay half-life.
    tau_scales
        Optional extra decay scales. Each adds one propagator (and so one set
        of counters), giving the model a choice of memory horizon instead of a
        single hand-set constant. Off by default: it triples counter memory,
        which matters at TabFormer scale.
    hops
        1 or 2. 2 enables the real second hop (C2).
    """

    def __init__(self, attributes: tuple[str, ...] | list[str] | None = None,
                 delta_days: float = 0.0, tau_days: float | None = None,
                 tau_scales: tuple[float, ...] = (),
                 gamma: float = 0.4, hops: int = 2,
                 min_count_for_loo: int = 5, max_entries: int = 4_000_000,
                 emit_variance: bool = True):
        self.attributes = tuple(attributes or CFG.risk_attributes)
        self.delta_s = float(delta_days) * DAY
        self.tau_days = float(tau_days if tau_days is not None else CFG.tau_days)
        self.tau_scales = tuple(tau_scales)
        self.gamma = gamma
        self.hops = hops
        self.min_count_for_loo = min_count_for_loo
        self.max_entries = max_entries
        self.emit_variance = emit_variance
        self.priors: dict[str, tuple[float, float]] = {}

    # ---- prior -----------------------------------------------------------------------
    def fit_prior(self, df: pd.DataFrame, y: np.ndarray,
                  now_ts: float | None = None) -> "AssociationRiskV2":
        """Moment-match the Beta prior per attribute. Training rows only.

        ``now_ts`` applies the maturity gate to the prior as well. Without it
        the prior is fit on labels that had not matured at scoring time, which
        leaks a small amount of future supervision into every row's posterior
        -- second-order compared with the counters, but it is leakage and the
        protocol claims there is none.
        """
        frame, yy = df, np.asarray(y)
        if now_ts is not None and "ts" in df.columns:
            mature = df["ts"].to_numpy() < (now_ts - self.delta_s)
            if mature.sum() >= 50:
                frame, yy = df[mature], yy[mature]
        for a in self.attributes:
            if a in frame.columns:
                self.priors[a] = fit_beta_prior(frame[a].astype(str).to_numpy(), yy,
                                                min_count=CFG.eb_min_count)
        return self

    # ---- main pass -------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, entity: np.ndarray,
                  ingest_mask: np.ndarray | None = None) -> pd.DataFrame:
        if not df["ts"].is_monotonic_increasing:
            raise ValueError("AssociationRiskV2.transform requires time-sorted input")

        n = len(df)
        t = df["ts"].to_numpy(np.float64)
        y = df["y"].to_numpy(np.float64)
        amt = (df["amount"].to_numpy(np.float64) if "amount" in df.columns
               else np.zeros(n))
        ingest = (np.ones(n, dtype=bool) if ingest_mask is None
                  else np.asarray(ingest_mask, dtype=bool))
        attrs = [a for a in self.attributes if a in df.columns]
        avals = {a: df[a].astype(str).to_numpy() for a in attrs}

        scales = (self.tau_days,) + tuple(s for s in self.tau_scales
                                          if s != self.tau_days)
        props = {
            s: MultiHopPropagator(tuple(attrs), s * DAY, gamma=self.gamma,
                                  hops=self.hops,
                                  min_count_for_loo=self.min_count_for_loo,
                                  max_entries=self.max_entries)
            for s in scales
        }
        primary = props[self.tau_days]

        # ---- output buffers ----------------------------------------------------------
        base_names = primary.feature_names()
        if not self.emit_variance:
            base_names = [c for c in base_names if not c.startswith("riskvar_")]
        out: dict[str, np.ndarray] = {c: np.zeros(n, np.float32) for c in base_names}
        for s in scales[1:]:
            for c in ("risk_mean", "risk_max", "risk_hop2", "risk_prop"):
                out[f"{c}_tau{int(s)}"] = np.zeros(n, np.float32)

        mem_known = np.zeros(n, np.int8)
        mem_prior_fraud = np.zeros(n, np.float32)
        mem_prior_txn = np.zeros(n, np.float32)
        mem_days_since = np.full(n, np.nan, np.float32)
        vel_1d = np.zeros(n, np.float32)
        vel_7d = np.zeros(n, np.float32)
        vel_amt7 = np.zeros(n, np.float32)
        recency = np.full(n, np.nan, np.float32)

        ent_mem: dict[int, dict] = {}
        ent_hist: dict[int, list] = {}
        ingest_ptr = 0

        for i in range(n):
            ti = t[i]
            cutoff = ti - self.delta_s

            # -- admit every matured label, strictly before the cutoff -----------------
            while ingest_ptr < n and t[ingest_ptr] < cutoff:
                j = ingest_ptr
                if ingest[j]:
                    vals_j = {a: avals[a][j] for a in attrs}
                    ej, yj, tj = int(entity[j]), y[j], t[j]
                    for p in props.values():
                        p.ingest(vals_j, ej, yj, tj)
                    m = ent_mem.setdefault(ej, {"pos": 0.0, "n": 0.0, "first": None})
                    m["pos"] += yj
                    m["n"] += 1.0
                    if yj > 0 and m["first"] is None:
                        m["first"] = tj
                ingest_ptr += 1

            # -- score the current row -------------------------------------------------
            ei = int(entity[i])
            vals_i = {a: avals[a][i] for a in attrs}

            feats = primary.score(vals_i, ei, ti, self.priors)
            for c in base_names:
                out[c][i] = feats.get(c, 0.0)
            for s in scales[1:]:
                f2 = props[s].score(vals_i, ei, ti, self.priors)
                for c in ("risk_mean", "risk_max", "risk_hop2", "risk_prop"):
                    out[f"{c}_tau{int(s)}"][i] = f2[c]

            m = ent_mem.get(ei)
            if m is not None:
                mem_prior_fraud[i] = m["pos"]
                mem_prior_txn[i] = m["n"]
                mem_known[i] = 1 if m["pos"] > 0 else 0
                if m["first"] is not None:
                    mem_days_since[i] = (ti - m["first"]) / DAY

            # -- velocity: the entity's own past, no labels involved -------------------
            h = ent_hist.setdefault(ei, [])
            if h:
                recency[i] = (ti - h[-1][0]) / DAY
                lo1, lo7 = ti - DAY, ti - 7 * DAY
                c1 = c7 = 0
                s7 = 0.0
                for tt, aa in reversed(h):
                    if tt < lo7:
                        break
                    c7 += 1
                    s7 += aa
                    if tt >= lo1:
                        c1 += 1
                vel_1d[i], vel_7d[i], vel_amt7[i] = c1, c7, s7
            h.append((ti, amt[i]))
            if len(h) > 64:
                del h[0]

        out.update({
            "mem_known_compromised": mem_known,
            "mem_prior_fraud_count": mem_prior_fraud,
            "mem_prior_txn_count": mem_prior_txn,
            "mem_days_since_first_fraud": mem_days_since,
            "vel_txn_1d": vel_1d, "vel_txn_7d": vel_7d,
            "vel_amt_7d": vel_amt7, "vel_recency_days": recency,
        })
        self.last_stats_ = {f"tau{int(s)}": p.stats() for s, p in props.items()}
        return pd.DataFrame(out, index=df.index)

    # ---- names -----------------------------------------------------------------------
    @staticmethod
    def monotone_feature_names(cols) -> list[str]:
        """Features whose direction is known a priori: more neighbourhood fraud is worse.

        Posterior *variance* is deliberately excluded -- higher uncertainty is
        not directionally "more fraud", so constraining it would assert
        something we do not believe. Evidence counts (``riskn_``) are excluded
        for the same reason.
        """
        keep = []
        for c in cols:
            if c.startswith("riskvar_") or c.startswith("riskn_"):
                continue
            if (c.startswith("risk_") or c.startswith("mem_")
                    or c in {"vel_txn_1d", "vel_txn_7d"}):
                keep.append(c)
        return keep

    @staticmethod
    def memory_feature_names(cols) -> list[str]:
        return [c for c in cols if c.startswith("mem_")]

    @staticmethod
    def risk_feature_names(cols) -> list[str]:
        return [c for c in cols if c.startswith(("risk_", "riskn_", "riskvar_"))]
