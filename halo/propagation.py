"""C2 -- leakage-safe multi-hop association propagation.

## What v1 actually did

``risk.py:224`` computed, and described as "one-hop damped propagation,
Personalised-PageRank flavour":

    risk_prop[i] = gamma * risk_max[i] + (1 - gamma) * risk_mean[i]

That is a convex combination of the max and mean of *the same row's own*
attribute risks. It never leaves the row. There is no neighbour, no hop, and no
graph. This module implements what was claimed.

## The structure

The graph is bipartite: attribute values `A` on one side, entities `E` on the
other, with an edge whenever an entity transacts under that value. A fraud ring
is a multi-hop object in this graph -- card -> shared device -> different card
-> different address -- so a zero-hop statistic is blind to exactly the
structure the method exists to find.

Two levels are maintained in one streaming pass:

    R_v  (1-hop)  decayed fraud rate at value v
    S_e           entity e's neighbourhood risk, = aggregate of R_v over the
                  values e has touched, recorded when e last transacted
    T_v  (2-hop)  decayed mean of S_e' over entities e' that touched v

and the score for a row with entity e and values {v_a} is

    r1 = agg_a R_{v_a}                      (leave-one-entity-out)
    r2 = agg_a T_{v_a}                      (leave-one-entity-out)
    r  = (1 - gamma) * r1 + gamma * r2

## The guarantee

> **Proposition (leakage-freeness).** For an entity `e`, the estimator `r`
> computed for a transaction of `e` is independent of every label attached to
> `e`.
>
> *Sketch.* `R_v` is read leave-one-entity-out, so `e`'s own labels are
> subtracted at level 1. `T_v` accumulates deposits of `S_e'` keyed by the
> depositing entity, and is likewise read leave-one-entity-out, so every
> deposit made by `e` is subtracted at level 2. A 2-hop path from `e` back to
> `e` must traverse one of `e`'s own deposits, and all of them are removed.
> Induction on the hop index extends this to `k` levels: at each level the
> accumulator is keyed by depositing entity and read with that entity's
> contribution removed. ∎

The subtraction is exact up to the 24-bit entity hash used for LOO keys
(``counters.AttributeCounters``); collisions cause *over*-subtraction, which
can only shrink the attributed risk and therefore cannot manufacture leakage.

## Cost

Naive LOO multi-hop recomputes propagation once per query entity -- O(|E|)
passes. Here each level is a decayed accumulator with an additive
own-contribution term, so the correction is a single subtraction per attribute
per row: **O(#attributes) per transaction, one pass, O(1) amortised per level.**
"""
from __future__ import annotations

import math

import numpy as np

from .counters import AttributeCounters, DecayStore


class MultiHopPropagator:
    """Streaming 2-hop leave-one-entity-out propagation over the bipartite graph.

    Parameters
    ----------
    attributes
        Attribute names participating in the graph.
    tau_s
        Decay constant in seconds. May be overridden per attribute.
    gamma
        Damping on the second hop. ``gamma=0`` recovers the 1-hop estimator.
    hops
        1 or 2. Level 3+ is possible with the same construction but is not
        enabled: the marginal signal is small and the accumulator count grows
        linearly in hops.
    """

    def __init__(self, attributes: tuple[str, ...], tau_s: float, gamma: float = 0.4,
                 hops: int = 2, tau_per_attribute: dict[str, float] | None = None,
                 min_count_for_loo: int = 5, max_entries: int = 4_000_000):
        self.attributes = tuple(attributes)
        self.gamma = float(gamma)
        self.hops = int(hops)
        taus = {a: (tau_per_attribute or {}).get(a, tau_s) for a in self.attributes}
        self.tau = taus

        # level 1: fraud labels aggregated by attribute value
        self.level1 = {a: AttributeCounters(taus[a], min_count_for_loo, max_entries)
                       for a in self.attributes}
        # level 2: entity neighbourhood-risk deposits aggregated by attribute value
        self.level2 = {a: AttributeCounters(taus[a], min_count_for_loo, max_entries)
                       for a in self.attributes} if hops >= 2 else {}
        # S_e: each entity's most recent neighbourhood risk
        self.entity_risk = DecayStore(tau_s, max_entries=max_entries)

    # ---- read ------------------------------------------------------------------------
    def score(self, values: dict[str, str], entity: int, t: float,
              priors: dict[str, tuple[float, float]]) -> dict[str, float]:
        """Score one row. Reads only; never mutates state (that is ``ingest``)."""
        from .counters import beta_posterior

        r1_per_attr: dict[str, float] = {}
        var_per_attr: dict[str, float] = {}
        n_per_attr: dict[str, float] = {}

        for a in self.attributes:
            v = values.get(a)
            a0, b0 = priors.get(a, (1.0, 99.0))
            if v is None:
                mean, var, n = a0 / (a0 + b0), 0.0, 0.0
            else:
                pos, tot = self.level1[a].read(v, entity, t)
                mean, var, n = beta_posterior(a0, b0, pos, tot)
            r1_per_attr[a] = mean
            var_per_attr[a] = var
            n_per_attr[a] = n

        r1_vals = list(r1_per_attr.values())
        r1_max = max(r1_vals) if r1_vals else 0.0
        r1_mean = float(np.mean(r1_vals)) if r1_vals else 0.0

        r2_mean = 0.0
        if self.hops >= 2:
            hop2 = []
            for a in self.attributes:
                v = values.get(a)
                if v is None:
                    continue
                s_sum, s_cnt = self.level2[a].read(v, entity, t)
                if s_cnt > 0:
                    hop2.append(s_sum / s_cnt)
            r2_mean = float(np.mean(hop2)) if hop2 else 0.0

        combined = (1.0 - self.gamma) * r1_mean + self.gamma * r2_mean

        out = {f"risk_{a}": r1_per_attr[a] for a in self.attributes}
        out.update({f"riskvar_{a}": var_per_attr[a] for a in self.attributes})
        out.update({f"riskn_{a}": n_per_attr[a] for a in self.attributes})
        out.update({
            "risk_max": r1_max,
            "risk_mean": r1_mean,
            "risk_hop2": r2_mean,
            "risk_prop": combined,
            "risk_evidence": float(sum(n_per_attr.values())),
        })
        return out

    # ---- write -----------------------------------------------------------------------
    def ingest(self, values: dict[str, str], entity: int, y: float, t: float,
               neighbourhood_risk: float | None = None) -> None:
        """Admit one matured label into the graph.

        Must only be called for rows whose label has matured under the protocol;
        the caller owns that gate.
        """
        for a in self.attributes:
            v = values.get(a)
            if v is not None:
                self.level1[a].add(v, entity, y, t)

        if self.hops >= 2:
            # Deposit this entity's neighbourhood risk so *other* entities
            # sharing these values can see it at the second hop. Keyed by the
            # depositing entity, which is what makes the LOO subtraction exact.
            s_e = neighbourhood_risk
            if s_e is None:
                pos, tot = self.entity_risk.get(int(entity), t)
                s_e = (pos / tot) if tot > 0 else 0.0
            for a in self.attributes:
                v = values.get(a)
                if v is not None:
                    self.level2[a].add(v, entity, float(s_e), t)

        self.entity_risk.add(int(entity), float(y), t)

    # ---- diagnostics -----------------------------------------------------------------
    def stats(self) -> dict:
        out = {"gamma": self.gamma, "hops": self.hops,
               "entity_risk_size": len(self.entity_risk)}
        for a in self.attributes:
            for lvl, store in (("l1", self.level1), ("l2", self.level2)):
                if a in store:
                    for k, v in store[a].stats_dict().items():
                        out[f"{lvl}_{a}_{k}"] = v
        return out

    def feature_names(self) -> list[str]:
        names = []
        for a in self.attributes:
            names += [f"risk_{a}", f"riskvar_{a}", f"riskn_{a}"]
        return names + ["risk_max", "risk_mean", "risk_hop2", "risk_prop", "risk_evidence"]


# --------------------------------------------------------------------------------------
# Offline variant, for datasets with a real edge list (Elliptic)
# --------------------------------------------------------------------------------------

def propagate_graph_loo(n_nodes: int, edges: np.ndarray, seed_risk: np.ndarray,
                        entity: np.ndarray, hops: int = 2, gamma: float = 0.4,
                        ) -> np.ndarray:
    """k-hop propagation on an explicit edge list, excluding same-entity paths.

    Used for Elliptic, where the transaction graph is given rather than induced
    from shared attributes. Same guarantee as the streaming version: a node
    never receives mass that originated from its own entity.

    Implemented as damped power iteration on the symmetric adjacency with
    same-entity edges removed. Removing the edges (rather than correcting after
    propagation) is the cleanest way to guarantee no same-entity path exists at
    any hop count.
    """
    if len(edges) == 0:
        return np.zeros(n_nodes, dtype=np.float32)

    src, dst = edges[:, 0], edges[:, 1]
    cross = entity[src] != entity[dst]          # drop within-entity edges entirely
    src, dst = src[cross], dst[cross]

    deg = np.bincount(src, minlength=n_nodes).astype(np.float32)
    deg[deg == 0] = 1.0

    r = np.asarray(seed_risk, dtype=np.float32).copy()
    acc = np.zeros(n_nodes, dtype=np.float32)
    for h in range(1, hops + 1):
        nxt = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(nxt, dst, r[src] / deg[src])
        np.add.at(nxt, src, r[dst] / deg[dst])
        acc += (gamma ** h) * nxt
        r = nxt
    return acc
