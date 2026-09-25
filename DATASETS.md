# Datasets

None of these are redistributed with this repository. Download each one and place
it at the exact path given below, then run `python -m halo.cli_v2 datasets` to
confirm the code can see it.

## Summary

| Name | Rows | Entity key | Amount | Real or synthetic |
|---|---|---|---|---|
| `ieee_cis` | 590,540 | resolved (Block A) | yes | real |
| `paysim` | 6,362,620 | native | yes | synthetic |
| `baf` | 1,000,000 per variant | proxy | no | synthetic, real-derived |
| `elliptic` | 203,769 nodes | graph component | no | real |
| `tabformer` | 24,386,900 | native | yes | synthetic |

`synthetic` needs no download. It is generated in memory and is what `smoke`,
`verify`, and `selftest` run against.

---

## 1. IEEE-CIS Fraud Detection

The primary dataset. Every headline result in the paper comes from this one.

- **Link:** https://www.kaggle.com/c/ieee-fraud-detection/data
- **Files needed:** `train_transaction.csv`, `train_identity.csv`
- **Place at:**
  ```
  data/ieee-fraud-detection/train_transaction.csv
  data/ieee-fraud-detection/train_identity.csv
  ```
- **Access:** Kaggle competition data. Accept the competition rules once from the
  web interface before the API will serve it.

```bash
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/ieee-fraud-detection
kaggle competitions download -c ieee-fraud-detection -f train_identity.csv    -p data/ieee-fraud-detection
```

**Caveat carried into the paper.** The labelling rule is documented only in a
Kaggle forum reply from the data provider, not in the dataset documentation. A
reported chargeback marks every later transaction on the linked card, email, or
billing address, and anything unreported after 120 days is marked legitimate.
The rule is checked empirically through entity label purity rather than taken on
trust.

---

## 2. PaySim

Used as a negative control and for the L5 generative-determinism audit.

- **Link:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **File needed:** the single CSV from that dataset
- **Place at either:**
  ```
  data/paysim/paysim.csv
  data/paysim/PS_20174392719_1491204439457_log.csv
  ```
  The loader accepts either filename, so renaming is optional.

```bash
kaggle datasets download -d ealaxi/paysim1 -p data/paysim --unzip
```

**Caveat.** Raw balance fields encode the generator's own fraud rule: the
transfer amount equals the prior balance in 97.8% of frauds. Those fields are
dropped by default. Enable them only to reproduce the L5 collapse
(0.9997 to 0.2580 AUPRC), never for a headline number.

---

## 3. BAF (Bank Account Fraud)

- **Link:** https://github.com/feedzai/bank-account-fraud
- **Mirror:** https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- **File needed:** `Base.csv` (the default variant)
- **Place at:**
  ```
  data/baf/Base.csv
  ```
- **Other variants:** the suite also ships `Variant I` through `Variant V`, each
  1M rows with a different bias profile. Load one with
  `load_baf(variant="Variant I")`. Only `Base` is used by default.

**Caveats.** BAF is application fraud, not transaction fraud, and it has no
native entity key. The entity here is a documented proxy built from categorical
combinations, so the L3 rung on BAF measures proxy-entity leakage and must be
reported as such. It carries no transaction amount, so dollar-recall is not
computed. Time resolution is monthly (months 0 to 7), so the latency sweep is
coarse on this dataset.

---

## 4. Elliptic

- **Link:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Files needed:** all three
- **Place at:**
  ```
  data/elliptic/elliptic_txs_features.csv
  data/elliptic/elliptic_txs_classes.csv
  data/elliptic/elliptic_txs_edgelist.csv
  ```

```bash
kaggle datasets download -d ellipticco/elliptic-data-set -p data/elliptic --unzip
```

**Caveats.** Features are anonymised and there is no transaction amount, so
dollar-recall is not computed. The entity is the weakly-connected component of
the transaction graph, which is a structural proxy for a controlling actor.
Only about 23% of nodes carry a label; unlabelled nodes are dropped for
supervised scoring and kept for propagation. This is the only dataset with a
real edge list, so it uses `propagate_graph_loo` rather than the streaming
propagator.

---

## 5. TabFormer

- **Link:** https://github.com/IBM/TabFormer
- **File needed:** `card_transaction.v1.csv` (from the credit-card transaction
  release; the repository links the download)
- **Place at:**
  ```
  data/tabformer/card_transaction.v1.csv
  ```

**Caveats.** TabFormer is synthetic, produced by an IBM generator. It is
included for scale replication, not as independent real-world evidence. The
loader subsamples to the most recent contiguous 5,000,000 rows by default,
because the full 24.4M will not fit a 30 GB session. Temporal order is preserved
by taking a contiguous block rather than a random sample.

---

## Where the files go

```
HALO — Entity-Leakage Fraud Detection/
  data/
    ieee-fraud-detection/
      train_transaction.csv
      train_identity.csv
    paysim/
      paysim.csv
    baf/
      Base.csv
    elliptic/
      elliptic_txs_features.csv
      elliptic_txs_classes.csv
      elliptic_txs_edgelist.csv
    tabformer/
      card_transaction.v1.csv
```

`data/` is gitignored, so nothing downloaded here is ever committed.

On Kaggle the paths resolve automatically: `ON_KAGGLE` detects `/kaggle/input`
and switches the input and working directories with no configuration. Attach the
datasets to the notebook and the loaders will find them.

## Verifying

```bash
python -m halo.cli_v2 datasets
```

Every dataset present prints `ok <name>`. Every missing one prints the exact path
it looked for and the URL to obtain it, then is skipped rather than failing the
run. Naming a dataset you have not downloaded yet is safe: it is reported and
dropped, and the run continues with the rest.

## What to run where

Guidance from `RUN_GUIDE_V2.md` section 7, repeated here because it depends on
which datasets you have:

| Dataset | 16 GB local machine | Kaggle |
|---|---|---|
| `synthetic` | yes, seconds | not needed |
| `elliptic` | yes, smallest real dataset | fine |
| `ieee_cis` | yes, about 165 s per feature build | fine |
| `baf` | yes, watch RAM on the wide frame | fine |
| `paysim` | tight, use `--folds 3` and cap the cache | better here |
| `tabformer` | no, subsample only | run it here |

See [RUN_GUIDE_V2.md](RUN_GUIDE_V2.md) for the full run instructions, including
resuming after a power cut and the cache size budget.
