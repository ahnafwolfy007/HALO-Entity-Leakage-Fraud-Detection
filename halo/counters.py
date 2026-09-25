"""Bounded-memory time-decayed counters for Block B.

Replaces the v1 store, which was ``dict[(value:str, entity:int)] -> _Accum``.
That design allocates one Python object plus one tuple key per observed
(value, entity) pair, roughly 216 bytes, and never releases anything:

    IEEE-CIS   590K rows x 10 attrs =   5.9M pairs ~  1.3 GB   (survivable)
    PaySim     6.4M rows x  3 attrs =  19.1M pairs ~  4.1 GB   (thrashes at 16GB)
    TabFormer 24.4M rows x  5 attrs = 122.0M pairs ~ 26.4 GB   (impossible)

Two changes bound it:

1. **Pruning.** Counters decay as ``exp(-dt/tau)``. An entry untouched for
   several tau has decayed to numerical noise and cannot influence any future
   score. Evicting those makes memory a function of the *active* key set rather
   than of the row count -- so a 24M-row pass costs no more than a 600K-row one.

2. **A frequency floor on leave-one-entity-out slots.** The LOO correction
   subtracts an entity's own contribution from an attribute value's counter.
   For a value observed once or twice the posterior is dominated by the prior
   anyway, so tracking per-entity contributions there buys almost nothing while
   costing the majority of all pairs (value frequency is heavy-tailed).

Both are approximations and both are declared: ``stats()`` reports how much was
pruned and how many LOO slots were declined, so the paper can state it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EMPTY = -1


@dataclass
class StoreStats:
    adds: int = 0
    lookups: int = 0
    prunes: int = 0
    pruned_entries: int = 0
    loo_declined: int = 0
    peak_size: int = 0

    def as_dict(self) -> dict:
        return {f"ctr_{k}": v for k, v in self.__dict__.items()}


class DecayStore:
    """Lazily-decayed (positive, total) counters keyed by int64, with eviction.

    Decay is applied on read/write rather than on a schedule, so cost is O(1)
    per access regardless of how long a key has been idle.
    """

    __slots__ = ("tau_s", "max_entries", "eps", "_pos", "_tot", "_t", "stats")

    def __init__(self, tau_s: float, max_entries: int = 4_000_000, eps: float = 1e-4):
        self.tau_s = float(tau_s)
        self.max_entries = int(max_entries)
        self.eps = float(eps)
        self._pos: dict[int, float] = {}
        self._tot: dict[int, float] = {}
        self._t: dict[int, float] = {}
        self.stats = StoreStats()

    def __len__(self) -> int:
        return len(self._tot)

    # ---- core ------------------------------------------------------------------------
    def _decayed(self, key: int, t: float) -> tuple[float, float]:
        tot = self._tot.get(key)
        if tot is None:
            return 0.0, 0.0
        dt = t - self._t[key]
        if dt <= 0:
            return self._pos[key], tot
        f = math.exp(-dt / self.tau_s)
        return self._pos[key] * f, tot * f

    def add(self, key: int, y: float, t: float) -> None:
        pos, tot = self._decayed(key, t)
        self._pos[key] = pos + y
        self._tot[key] = tot + 1.0
        self._t[key] = t
        self.stats.adds += 1
        if len(self._tot) > self.max_entries:
            self.prune(t)

    def get(self, key: int, t: float) -> tuple[float, float]:
        self.stats.lookups += 1
        return self._decayed(key, t)

    # ---- eviction --------------------------------------------------------------------
    def prune(self, t: float) -> int:
        """Drop entries whose decayed total has fallen below ``eps``.

        An entry at ``tot < eps`` contributes less than one part in 1/eps to any
        posterior it participates in, so its removal is below the numerical
        resolution of the estimate it feeds.
        """
        eps, tau = self.eps, self.tau_s
        dead = [k for k, tt in self._t.items()
                if self._tot[k] * math.exp(-(t - tt) / tau) < eps]
        for k in dead:
            del self._pos[k], self._tot[k], self._t[k]
        self.stats.prunes += 1
        self.stats.pruned_entries += len(dead)
        self.stats.peak_size = max(self.stats.peak_size, len(self._tot) + len(dead))

        if len(self._tot) > 0.9 * self.max_entries:
            # Still crowded after dropping the dead: evict the oldest decile so
            # the pass cannot grow without bound on a pathological key space.
            cut = int(0.1 * len(self._t))
            if cut:
                oldest = sorted(self._t, key=self._t.get)[:cut]
                for k in oldest:
                    del self._pos[k], self._tot[k], self._t[k]
                self.stats.pruned_entries += cut
        return len(dead)


class AttributeCounters:
    """Counters for one attribute: global-by-value, plus per-(value, entity) for LOO.

    Keys are packed into a single int64 rather than tuples:

        value key  =  value_id
        LOO key    = (value_id << 24) ^ (entity_id & 0xFFFFFF)

    Collisions in the low 24 bits of the entity id are possible and benign: a
    collision slightly over-subtracts another entity's contribution, which is
    conservative (it can only *reduce* the risk attributed to the current row,
    never inflate it, so it cannot manufacture leakage).
    """

    __slots__ = ("tau_s", "by_value", "by_value_entity", "_value_ids", "_value_count",
                 "_pending", "min_count_for_loo", "residual_eps", "stats")

    def __init__(self, tau_s: float, min_count_for_loo: int = 5,
                 max_entries: int = 4_000_000, residual_eps: float = 1e-9):
        self.tau_s = tau_s
        self.min_count_for_loo = int(min_count_for_loo)
        self.residual_eps = float(residual_eps)
        self.by_value = DecayStore(tau_s, max_entries=max_entries)
        self.by_value_entity = DecayStore(tau_s, max_entries=max_entries)
        self._value_ids: dict[str, int] = {}
        self._value_count: dict[int, int] = {}
        self._pending: dict[int, list] = {}
        self.stats = StoreStats()

    def value_id(self, value: str) -> int:
        vid = self._value_ids.get(value)
        if vid is None:
            vid = len(self._value_ids)
            self._value_ids[value] = vid
        return vid

    @staticmethod
    def _loo_key(vid: int, entity: int) -> int:
        return (vid << 24) ^ (int(entity) & 0xFFFFFF)

    def add(self, value: str, entity: int, y: float, t: float) -> None:
        """Admit one observation, keeping the two stores exactly in step.

        The subtraction in ``read`` is only exact if every contribution present
        in ``by_value`` has a counterpart in ``by_value_entity``. Writing to the
        first unconditionally while opening the per-entity slot only above the
        frequency floor breaks that: the first ``k-1`` observations of a value
        stay in the global counter with nothing to subtract them against, and
        once the value crosses the floor they leak back into the score of
        whichever entity produced them.

        So the pre-floor observations are held in a small buffer instead and
        flushed into *both* stores at the crossing. The buffer holds at most
        ``k-1`` triples per value that has not yet crossed, i.e. O(#values), not
        O(#pairs) -- the memory argument for the floor is untouched.
        """
        vid = self.value_id(value)
        n = self._value_count.get(vid, 0) + 1
        self._value_count[vid] = n
        k = self.min_count_for_loo

        if n < k:                                   # below the floor: buffer only
            self._pending.setdefault(vid, []).append((int(entity), float(y), float(t)))
            self.stats.loo_declined += 1
            return
        if n == k:                                  # crossing: replay in time order
            for e0, y0, t0 in self._pending.pop(vid, ()):
                self.by_value.add(vid, y0, t0)
                self.by_value_entity.add(self._loo_key(vid, e0), y0, t0)
        self.by_value.add(vid, y, t)
        self.by_value_entity.add(self._loo_key(vid, entity), y, t)

    def read(self, value: str, entity: int, t: float) -> tuple[float, float]:
        """Decayed (positive, total) for ``value``, excluding ``entity``'s own history.

        If the value has been seen fewer than ``min_count_for_loo`` times we did
        not allocate a per-entity slot for it, so its own contribution cannot be
        subtracted. Returning the raw statistics here would leak the querying
        entity's own labels straight back into its score -- which is precisely
        the failure the protocol exists to prevent.

        Instead we return no evidence, so the posterior falls back to the prior.
        That is the same value such a low-count statistic would have shrunk to
        anyway, which is why the frequency cap was safe to introduce; what was
        *not* safe was using the statistic without the correction.

        The residual is clamped *jointly*, not term by term. Callers divide the
        two returned quantities (level 2 reads a sum of neighbourhood risks over
        a count of depositors), and clamping numerator and denominator
        independently breaks the identity precisely when the querying entity is
        the dominant contributor -- the case that matters most. If no other
        entity's evidence survives the subtraction, the honest answer is that
        there is none.
        """
        vid = self._value_ids.get(value)
        if vid is None:
            return 0.0, 0.0
        if self._value_count.get(vid, 0) < self.min_count_for_loo:
            return 0.0, 0.0                       # unsubtractable -> unusable
        pos, tot = self.by_value.get(vid, t)
        opos, otot = self.by_value_entity.get(self._loo_key(vid, entity), t)
        res_tot = tot - otot
        if res_tot <= self.residual_eps:
            return 0.0, 0.0                       # all evidence was the entity's own
        return min(max(pos - opos, 0.0), res_tot), res_tot

    def n_values(self) -> int:
        return len(self._value_ids)

    def stats_dict(self) -> dict:
        return {
            "n_values": self.n_values(),
            "size_value": len(self.by_value),
            "size_value_entity": len(self.by_value_entity),
            "loo_declined": self.stats.loo_declined,
            "pending_values": len(self._pending),
            "pruned": (self.by_value.stats.pruned_entries
                       + self.by_value_entity.stats.pruned_entries),
        }


# --------------------------------------------------------------------------------------
# Empirical-Bayes prior (moment matched), unchanged in substance from v1
# --------------------------------------------------------------------------------------

def fit_beta_prior(values: np.ndarray, y: np.ndarray, min_count: int = 30,
                   strength_clip: tuple[float, float] = (1.0, 5000.0)
                   ) -> tuple[float, float]:
    """Moment-match Beta(a0, b0) to per-value fraud rates. Training rows only."""
    import pandas as pd
    tab = pd.DataFrame({"v": values.astype(str), "y": np.asarray(y, dtype=float)})
    g = tab.groupby("v")["y"].agg(["sum", "count"])
    g = g[g["count"] >= min_count]
    base = float(np.mean(y)) if len(y) else 0.01
    base = min(max(base, 1e-4), 1 - 1e-4)
    if len(g) < 5:
        return base * 20.0, (1 - base) * 20.0
    p = (g["sum"] / g["count"]).to_numpy()
    m, v = float(p.mean()), float(p.var(ddof=1))
    m = min(max(m, 1e-4), 1 - 1e-4)
    if v <= 1e-12 or v >= m * (1 - m):
        return m * 20.0, (1 - m) * 20.0
    s = float(np.clip(m * (1 - m) / v - 1, *strength_clip))
    return m * s, (1 - m) * s


def beta_posterior(a0: float, b0: float, pos: float, tot: float
                   ) -> tuple[float, float, float]:
    """Return (mean, variance, evidence_mass) of Beta(a0+pos, b0+tot-pos).

    v1 emitted only the mean, discarding the posterior's own statement about how
    much it should be trusted. The variance separates 'risk 0.05 from two
    observations' from 'risk 0.05 from five hundred' -- a distinction a monotone
    tree cannot reconstruct from mean and count as separate features.
    """
    a = a0 + pos
    b = b0 + max(tot - pos, 0.0)
    s = a + b
    mean = a / s
    var = (a * b) / (s * s * (s + 1.0))
    return mean, var, tot
