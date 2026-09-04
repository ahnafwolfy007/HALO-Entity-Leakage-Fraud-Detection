"""Synthetic IEEE-CIS-shaped generator with known ground truth.

Why this exists
---------------
The pipeline claims to measure entity leakage. Before spending Kaggle hours on the
real data we need to know the code measures what it says it measures. This generator
reproduces the structural properties the audit depends on, and -- unlike the real
data -- it hands us the ground truth:

  * entities whose ``day - D1`` is constant (the UID identity that Block A must recover)
  * Vesta-style label propagation: an entity is compromised at an *index* transaction
    and every subsequent transaction inherits the fraud label
  * C1..C14 as cumulative per-entity counts (monotone in time -- Block A's label-free
    validation signal)
  * V-columns arriving in blocks that go missing together (Block C's provenance regimes)
  * an identity table covering only a minority of transactions
  * genuine behavioural signal at the index event, plus shared-infrastructure risk,
    so that honest detection is possible but memorisation is *more* rewarding

If the pipeline is correct then on this data it must find: near-total entity label
purity, a large drop when entities are made disjoint, and a recoverable-but-much-lower
honest ceiling. If it does not, the pipeline is broken -- not the benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

V_BLOCKS = [
    ("V1", 11), ("V12", 23), ("V35", 18), ("V53", 22), ("V75", 20), ("V95", 16),
]
"""(prefix, width) -- mirrors the real dataset's block structure at reduced scale."""

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = ["D1", "D2", "D3", "D4", "D10", "D15"]


def make_synthetic(
    n_entities: int = 6_000,
    span_days: int = 182,
    compromise_rate: float = 0.055,
    identity_coverage: float = 0.24,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(transactions, identity, ground_truth)``.

    ``ground_truth`` carries columns the real data does not have: ``true_entity``,
    ``is_index`` (this row is the entity's first fraud) and ``true_regime``.
    """
    rng = np.random.default_rng(seed)

    # ---- entity-level attributes -----------------------------------------------------
    n_devices = max(60, n_entities // 40)
    n_domains = 24
    n_addr = max(40, n_entities // 60)

    ent_first_day = rng.integers(0, span_days - 3, size=n_entities)
    ent_card1 = rng.integers(1000, 19000, size=n_entities)
    ent_addr1 = rng.integers(100, 100 + n_addr, size=n_entities)
    ent_card2 = rng.integers(100, 600, size=n_entities)
    ent_card3 = rng.choice([150, 185], size=n_entities, p=[0.9, 0.1])
    ent_card4 = rng.choice(["visa", "mastercard", "amex", "discover"],
                           size=n_entities, p=[0.6, 0.32, 0.05, 0.03])
    ent_card5 = rng.integers(100, 240, size=n_entities)
    ent_card6 = rng.choice(["debit", "credit"], size=n_entities, p=[0.72, 0.28])
    ent_domain = rng.integers(0, n_domains, size=n_entities)
    ent_device = rng.integers(0, n_devices, size=n_entities)
    ent_dist1 = rng.integers(0, 400, size=n_entities).astype(float)

    # Shared infrastructure carries real risk: a minority of devices and domains are bad.
    device_risk = rng.beta(0.7, 9.0, size=n_devices)
    domain_risk = rng.beta(0.8, 8.0, size=n_domains)

    # ---- transaction counts per entity -----------------------------------------------
    n_txn = 1 + rng.poisson(4.5, size=n_entities)
    n_txn = np.clip(n_txn, 1, 60)
    total = int(n_txn.sum())

    ent_idx = np.repeat(np.arange(n_entities), n_txn)
    within = np.concatenate([np.arange(k) for k in n_txn])

    # Inter-event gaps -> day of each transaction, never past the window edge.
    gaps = rng.exponential(6.0, size=total)
    day_offset = np.zeros(total)
    start = 0
    for e in range(n_entities):
        k = n_txn[e]
        day_offset[start:start + k] = np.cumsum(gaps[start:start + k]) - gaps[start]
        start += k
    day = ent_first_day[ent_idx] + day_offset
    keep = day < span_days
    ent_idx, within, day = ent_idx[keep], within[keep], day[keep]
    total = len(ent_idx)

    order = np.argsort(day, kind="stable")
    ent_idx, within, day = ent_idx[order], within[order], day[order]

    # floor(day) first: adding intra-day seconds on top of a fractional day would push
    # rows into the next calendar day and break the `day - D1 == first_day` invariant.
    day_int = np.floor(day)
    seconds = (day_int * 86_400).astype(np.int64) + rng.integers(0, 86_400, size=total)

    # Re-sort by the actual timestamp and recompute within-entity rank. Without this,
    # two same-day transactions of one entity can be generated in `day` order but stored
    # in a different `TransactionDT` order, which would make the cumulative C-columns
    # look non-monotone and corrupt the ground truth the validation check is graded on.
    order2 = np.argsort(seconds, kind="stable")
    ent_idx, day, day_int, seconds = (ent_idx[order2], day[order2],
                                      day_int[order2], seconds[order2])
    within = pd.Series(ent_idx).groupby(ent_idx).cumcount().to_numpy()
    hour = (seconds // 3600) % 24

    # ---- amount ----------------------------------------------------------------------
    base_amt = np.exp(rng.normal(3.9, 0.85, size=n_entities))
    amt = base_amt[ent_idx] * np.exp(rng.normal(0, 0.45, size=total))

    # ---- compromise: which entities, and when ----------------------------------------
    ent_hazard = (0.35 * device_risk[ent_device] + 0.35 * domain_risk[ent_domain]
                  + 0.30 * rng.random(n_entities))
    ent_hazard = ent_hazard / ent_hazard.mean() * compromise_rate
    compromised = rng.random(n_entities) < np.clip(ent_hazard, 0, 0.9)

    # The index event is a specific transaction, biased toward behaviourally odd ones:
    # unusual amount for the entity, and odd-hour activity. This is the honest signal.
    z_amt = np.log(amt) - np.log(base_amt[ent_idx])
    odd_hour = ((hour < 6) | (hour >= 22)).astype(float)
    # Early-life bias: in the real data a compromised UID is nearly all-fraud, because the
    # chargeback tends to land soon after the identity starts transacting.
    early_bias = np.exp(-0.9 * within)
    index_score = (1.4 * np.abs(z_amt) + 0.9 * odd_hour
                   + 2.2 * early_bias + rng.normal(0, 0.8, size=total))

    is_index = np.zeros(total, dtype=bool)
    df_pick = pd.DataFrame({"e": ent_idx, "s": index_score,
                            "pos": np.arange(total), "w": within})
    # Entities cannot be compromised on their very first transaction more often than
    # elsewhere; require at least one prior transaction where possible.
    elig = df_pick[df_pick["e"].isin(np.flatnonzero(compromised))]
    if len(elig):
        chosen = elig.sort_values("s").groupby("e", sort=False)["pos"].last()
        is_index[chosen.to_numpy()] = True

    # ---- Vesta label propagation ------------------------------------------------------
    # Once an entity's index event fires, every *subsequent* transaction is labelled fraud.
    is_fraud = np.zeros(total, dtype=bool)
    idx_within = {}
    for e, w in zip(ent_idx[is_index], within[is_index]):
        idx_within[e] = w
    if idx_within:
        thresh = np.full(n_entities, np.inf)
        for e, w in idx_within.items():
            thresh[e] = w
        is_fraud = within >= thresh[ent_idx]

    # ---- D columns: D1 = days since the card began transacting ------------------------
    # Therefore day - D1 == first_day, constant per entity. This is the UID key.
    D1 = day_int - ent_first_day[ent_idx]
    D2 = np.where(D1 > 0, D1, np.nan)
    prev_day = np.full(total, np.nan)
    last_seen: dict[int, float] = {}
    for i in range(total):
        e = ent_idx[i]
        if e in last_seen:
            prev_day[i] = day[i] - last_seen[e]
        last_seen[e] = day[i]
    D3 = prev_day
    D4 = D1 + rng.normal(0, 1.5, size=total)
    D10 = np.where(rng.random(total) < 0.15, np.nan, D1 * 0.8)
    D15 = D1 + rng.normal(0, 0.5, size=total)

    # ---- C columns: cumulative per-entity counts (monotone within entity) -------------
    c_data = {}
    counter = np.zeros((n_entities, 14))
    C = np.zeros((total, 14), dtype=np.float32)
    incr = rng.random((total, 14)) < 0.35
    for i in range(total):
        e = ent_idx[i]
        counter[e] += incr[i]
        C[i] = counter[e]
    for j, col in enumerate(C_COLS):
        c_data[col] = C[:, j]

    # ---- provenance regimes drive V-block missingness ---------------------------------
    n_regimes = 7
    regime_p = rng.dirichlet(np.ones(n_regimes) * 1.4)
    true_regime = rng.choice(n_regimes, size=total, p=regime_p)
    block_present = rng.random((n_regimes, len(V_BLOCKS))) < 0.6
    block_present[0, :] = True  # one regime always has full enrichment

    v_data = {}
    fraud_f = is_fraud.astype(float)
    for b, (prefix, width) in enumerate(V_BLOCKS):
        present = block_present[true_regime, b]
        start_no = int(prefix[1:])
        for k in range(width):
            col = f"V{start_no + k}"
            vals = rng.normal(0, 1, size=total)
            if k < 3:  # a few V-columns carry weak genuine signal
                vals += 0.35 * fraud_f
            vals[~present] = np.nan
            v_data[col] = vals.astype(np.float32)

    # ---- assemble the transaction table ----------------------------------------------
    txn = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + total, dtype=np.int64),
        "isFraud": is_fraud.astype(np.int8),
        "TransactionDT": seconds,
        "TransactionAmt": amt.astype(np.float32),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=total,
                                p=[0.74, 0.12, 0.06, 0.05, 0.03]),
        "card1": ent_card1[ent_idx],
        "card2": ent_card2[ent_idx].astype(np.float32),
        "card3": ent_card3[ent_idx].astype(np.float32),
        "card4": ent_card4[ent_idx],
        "card5": ent_card5[ent_idx].astype(np.float32),
        "card6": ent_card6[ent_idx],
        "addr1": ent_addr1[ent_idx].astype(np.float32),
        "addr2": np.full(total, 87.0, dtype=np.float32),
        "dist1": ent_dist1[ent_idx].astype(np.float32),
        "P_emaildomain": [f"dom{d:02d}.com" for d in ent_domain[ent_idx]],
        "R_emaildomain": np.where(rng.random(total) < 0.7, None,
                                  [f"dom{d:02d}.com" for d in ent_domain[ent_idx]]),
        "D1": D1.astype(np.float32), "D2": D2.astype(np.float32),
        "D3": D3.astype(np.float32), "D4": D4.astype(np.float32),
        "D10": D10.astype(np.float32), "D15": D15.astype(np.float32),
    })
    extra = {col: vals.astype(np.float32) for col, vals in c_data.items()}
    for m in range(1, 10):
        extra[f"M{m}"] = np.where(rng.random(total) < 0.25, None,
                                  rng.choice(["T", "F"], size=total))
    extra.update(v_data)
    txn = pd.concat([txn, pd.DataFrame(extra, index=txn.index)], axis=1)

    # ---- identity table: minority coverage -------------------------------------------
    has_id = rng.random(total) < identity_coverage
    id_rows = int(has_id.sum())
    ident = pd.DataFrame({
        "TransactionID": txn.loc[has_id, "TransactionID"].to_numpy(),
        "id_01": rng.normal(-5, 5, id_rows).astype(np.float32),
        "id_02": rng.normal(100_000, 50_000, id_rows).astype(np.float32),
        "id_30": rng.choice(["Windows 10", "iOS 11.1.2", "Android 7.0", "Mac OS X"],
                            size=id_rows),
        "id_31": rng.choice(["chrome 63.0", "mobile safari 11.0", "ie 11.0", "firefox 57.0"],
                            size=id_rows),
        "id_33": rng.choice(["1920x1080", "1366x768", "2208x1242", "1334x750"],
                            size=id_rows),
        "DeviceType": rng.choice(["desktop", "mobile"], size=id_rows),
        "DeviceInfo": [f"Device_{d:03d}" for d in ent_device[ent_idx[has_id]]],
    })

    truth = pd.DataFrame({
        "TransactionID": txn["TransactionID"].to_numpy(),
        "true_entity": ent_idx,
        "is_index": is_index,
        "true_regime": true_regime,
        "within": within,
    })
    return txn, ident, truth


def write_synthetic(out_dir, **kwargs) -> dict:
    """Write synthetic CSVs in the IEEE-CIS layout so `data.py` can load them unchanged."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txn, ident, truth = make_synthetic(**kwargs)
    txn.to_csv(out / "train_transaction.csv", index=False)
    ident.to_csv(out / "train_identity.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)
    return {
        "rows": len(txn), "entities": int(truth["true_entity"].nunique()),
        "fraud_rate": float(txn["isFraud"].mean()),
        "index_events": int(truth["is_index"].sum()),
        "identity_coverage": float(len(ident) / len(txn)),
        "columns": txn.shape[1],
    }
