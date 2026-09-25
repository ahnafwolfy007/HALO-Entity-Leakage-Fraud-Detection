# HALO v2 — Run Guide

How to run the v2 research programme end to end, on a machine that loses power
without warning.

Everything here assumes the working directory is the repository root
(`HALO — Entity-Leakage Fraud Detection/`).

---

## 0. The one thing to understand first

**Every run is resumable, and resuming is the normal case, not the recovery case.**

Work is split into independent *units* (one model fit on one dataset, fold, seed
and configuration). Each completed unit is appended to a `.jsonl` ledger with an
`fsync` before the call returns. Re-issuing the identical command skips every
unit already banked and continues from the first one that is not.

So the correct response to a power cut is: **run the same command again.** Not a
flag, not a repair step — the same command.

```bash
python -m halo.cli_v2 run --stage all --datasets ieee_cis --seeds 3
```

Kill it, pull the plug, close the laptop. Run the same line again. It picks up.

---

## 1. Install

Python 3.11+ (3.14 is what this was developed against).

```bash
pip install numpy pandas scikit-learn scipy lightgbm xgboost catboost pyarrow
```

`pyarrow` is **not** currently installed in this environment. Without it the
feature cache silently falls back to pickle: correct, but larger on disk and
slower to read. Install it before a long run.

Nothing here uses the GPU. LightGBM, XGBoost and CatBoost all run on CPU in this
configuration, and a 1050 Ti (4 GB) is too small to help with GBDTs on tabular
data anyway. Do not spend time on CUDA — spend it on cores and RAM.

Verify the install without downloading anything:

```bash
python -m halo.cli_v2 smoke
```

~1 minute. It builds synthetic data, runs Block B v2 (counters, 2-hop, variance),
fits the region-monotone model, and executes a resumable stage. It must end with
`smoke PASSED`.

---

## 2. Check correctness before you check results

Two suites. Both must pass before any number from this codebase is worth
reporting.

```bash
python -m halo.cli_v2 verify      # 5 protocol properties, ~2 min
python -m halo.cli_v2 selftest    # 8 crash-safety tests, ~5 sec
```

### `verify` — the claims the paper asserts

| | property | what a failure means |
|---|---|---|
| P1 | leave-one-entity-out | an entity's own labels move its own risk features — the protocol is void |
| P2 | maturity gate | a label inside the λ window moves a score — you are reading the future |
| P3 | region monotonicity | a backward step inside the alert region — C5's auditability claim is false |
| P4 | reason codes | a stated threshold does not actually clear the alert — the explanation lies |
| P5 | censoring correction | soft imputation loses to dropping — C1 buys nothing |

Current state: **5/5 pass.** P3 additionally reports
`max_violation_anywhere ≈ 0.21`, which is not a failure — it is the *measured
price* of restricting monotonicity to the alert region, and it belongs in the
paper as a number rather than a caveat.

P5 reports `soft_minus_zero` as a diagnostic, not an assertion. C1's claim is
against v1's behaviour (`drop`), and it wins there. Whether it also beats naive
zero-filling depends on how well calibrated `p(x)` is, because the imputation is
only as good as the model doing the imputing.

### `selftest` — crash safety

Simulates the specific ways a process dies mid-write: torn final ledger line,
NUL-padded file tail, corrupt checkpoint, abandoned temp files. **8/8 pass.**

Run `selftest` on any new machine before starting a multi-day job. It takes five
seconds and it is the difference between losing one unit and losing a week.

---

## 3. Get the data

Nothing is redistributed with this repository. Place files exactly here:

```
data/
  ieee-fraud-detection/train_transaction.csv      (+ train_identity.csv)
  paysim/paysim.csv
  baf/Base.csv
  elliptic/elliptic_txs_features.csv
  elliptic/elliptic_txs_classes.csv
  elliptic/elliptic_txs_edgelist.csv
  tabformer/card_transaction.v1.csv
```

| dataset | source | rows |
|---|---|---|
| ieee_cis | kaggle.com/c/ieee-fraud-detection | 590,540 |
| paysim | kaggle.com/datasets/ealaxi/paysim1 | 6,362,620 |
| baf | github.com/feedzai/bank-account-fraud | 1,000,000 |
| elliptic | kaggle.com/datasets/ellipticco/elliptic-data-set | 203,769 |
| tabformer | github.com/IBM/TabFormer | 24,386,900 |

[DATASETS.md](DATASETS.md) has the full entry for each one: exact filenames,
`kaggle` download commands, dataset variants, and the caveats that constrain
what may be claimed from each.

Check what the code can actually see:

```bash
python -m halo.cli_v2 datasets
```

This prints each dataset's entity source and — importantly — its **caveats**.
Read them. They are not boilerplate; they constrain what you may claim:

- **BAF has no entity key.** It is application fraud. The entity is a documented
  *proxy* over categorical combinations, so BAF's L3 rung measures proxy-entity
  leakage. It is not a like-for-like replication of IEEE-CIS and must not be
  presented as one.
- **Elliptic**: entity is the weakly-connected component of the transaction
  graph. Only ~23% of nodes are labelled; unlabelled nodes are dropped for
  supervised scoring but retained for propagation. No amounts, so no
  dollar-recall.
- **BAF** has no amount either, and monthly time resolution, so its latency
  sweep is coarse.
- **TabFormer is synthetic** (IBM generator). It is scale replication, not
  independent real-world evidence. Subsampled to the most recent contiguous 5M
  rows by default; the full 24.4M will not fit a 30 GB session.

---

## 4. Run

```
python -m halo.cli_v2 run --stage STAGE --datasets LIST [options]
```

| option | default | notes |
|---|---|---|
| `--stage` | `all` | `ladder`, `benchmark`, `latency`, `ablation`, `censoring`, `all` |
| `--datasets` | `auto` | comma-separated, or `auto` to use whatever is on disk |
| `--seeds` | 3 | 5 for final numbers |
| `--folds` | 4 | rolling-origin test folds |
| `--max-hours` | none | **stop cleanly before this budget** |
| `--ckpt` | `out/checkpoints` | checkpoint directory |

### The stages

| stage | answers | units per dataset (3 seeds × 4 folds) |
|---|---|---|
| `ladder` | T1: how much does each leakage channel inflate results? | 60 |
| `benchmark` | T3: HALO vs 5 baselines under CEP | 72 |
| `latency` | T5/W2: the λ sweep, 0→120 days | 144 |
| `ablation` | C1–C5 on/off, plus the H1 constraint-value test | 168 |
| `censoring` | C1: drop vs zero vs soft under simulated delay | 54 |
| | **total** | **498** |

Five real datasets at defaults is ~2,500 units.

### Order to run them in

Do **not** start with `--stage all`. Run in this order, checking output between:

```bash
# 1. cheapest, and it is the paper's headline claim
python -m halo.cli_v2 run --stage ladder --datasets ieee_cis --seeds 3

# 2. the comparison everything else is judged against
python -m halo.cli_v2 run --stage benchmark --datasets ieee_cis --seeds 3

# 3. the λ curve — this is where W2 lives
python -m halo.cli_v2 run --stage latency --datasets ieee_cis --seeds 3

# 4. then widen to the other datasets
python -m halo.cli_v2 run --stage ladder    --datasets elliptic,baf --seeds 3
python -m halo.cli_v2 run --stage benchmark --datasets elliptic,baf --seeds 3
```

`--datasets` takes a comma-separated list; `--stage` takes exactly one stage (or
`all`).

Naming a dataset you have not downloaded yet is safe. It is checked, reported and
dropped, and the run proceeds with the rest — so you can keep using the same
command line as data arrives one dataset at a time.

The reason is not politeness about compute. `ablation` is the largest stage and
the least informative if the benchmark underneath it turns out to be wrong.

### Progress

```bash
python -m halo.cli_v2 status
```

Completed units per stage, which datasets, and cumulative compute hours.

---

## 5. Running under load-shedding

This is the configuration the checkpointing was built for.

**Always pass `--max-hours`.** It stops cleanly at a unit boundary rather than
being killed mid-write:

```bash
python -m halo.cli_v2 run --stage benchmark --datasets ieee_cis --max-hours 2
```

Set it slightly below however long your power typically holds. A clean stop
leaves zero debris; an unclean one costs at most the single unit in flight.

**What survives a power cut, precisely:**

- Every unit written to the ledger before the cut. Guaranteed by `fsync`.
- The unit in flight is lost and recomputed. This is the entire exposure.
- A torn final ledger line is truncated automatically on the next start, with a
  `[resume] repaired ledger, dropped N trailing bytes` message. That message is
  normal and expected — it is the system working.
- A checkpoint that cannot be read is renamed to `.corrupt` and recomputed rather
  than trusted.
- Abandoned `.tmp` files older than an hour are swept at the start of each run.

**A UPS that gives you five minutes is worth more than any code change here** —
it converts every cut into a clean stop. Failing that, keep `--max-hours` small
and run more often.

### Disk

Cached feature tables are capped, LRU-evicted, at 12 GB by default:

```bash
HALO_CACHE_GB=4 python -m halo.cli_v2 run --stage all --datasets tabformer
```

Set this. A full sweep builds ~80 feature tables per dataset, and at TabFormer
scale each is several hundred megabytes — uncapped, that is tens of gigabytes.
Filling the disk makes the next atomic write fail, which loses the run that
checkpointing exists to protect. Evicting a cache entry only costs
recomputation, so the cap is always safe.

Use 4 GB on Kaggle (20 GB working limit), 12–30 GB on a local NVMe.

---

## 6. If you change the code

The feature cache and the ledger are both keyed on a fingerprint of the code
that computes features (`counters.py`, `propagation.py`, `risk_v2.py`), with
docstrings and comments stripped.

- Edit a **comment or docstring** → nothing is invalidated. A week of computation
  survives a typo fix.
- Edit anything **executable** in those three modules → cached features and
  completed units are retired automatically and recomputed.

This is deliberately conservative. A stale cache winning a lookup after a bug fix
means the paper gets built on numbers that no longer correspond to any code —
the same failure class as the leave-one-entity-out bug itself.

To see the current fingerprint:

```bash
python -c "import sys;sys.path.insert(0,'.');from halo.experiments_v2 import feature_code_version;print(feature_code_version())"
```

---

## 7. Hardware

### Your machine (16 GB RAM, i5-8400, NVMe gen3)

| dataset | feasible locally | notes |
|---|---|---|
| synthetic | yes | seconds |
| elliptic | yes | 204k rows, smallest real dataset |
| ieee_cis | yes | 590k rows; ~165 s per feature build |
| baf | yes | 1M rows, but watch RAM on the wide frame |
| paysim | tight | 6.4M rows; use `--folds 3`, cap the cache |
| tabformer | **no** | subsample only; full 24.4M needs ~30 GB |

Block B is a single-pass Python loop, and throughput is what it is:
**~12,000 rows/s at 3 attributes, ~3,600 rows/s at 10.** IEEE-CIS uses 10
attributes, so budget ~165 s per feature build and ~80 builds for a full sweep —
roughly 4 hours of feature construction alone, before any model fitting. The
cache means you pay that once, which is exactly why the cache budget must not be
set so low that entries are evicted before they are reused.

Six cores is the real constraint, not the GPU. Leave `n_jobs` alone; it is
already `cpu_count - 1`.

### Kaggle

Paths resolve automatically — `ON_KAGGLE` detects `/kaggle/input` and switches
to `/kaggle/input` and `/kaggle/working` with no configuration.

Sessions cap at 12 hours, so:

```bash
HALO_CACHE_GB=4 python -m halo.cli_v2 run --stage latency \
    --datasets tabformer --max-hours 11.5
```

**Copy `out/results/*.jsonl` out of `/kaggle/working` before the session ends,
and copy it back at the start of the next one.** Kaggle does not persist the
working directory between sessions. The ledger is the only thing that must
survive; feature caches can always be rebuilt. This is the single most common way
to lose progress on Kaggle, and it has nothing to do with power.

Run TabFormer and PaySim there. Keep IEEE-CIS, Elliptic and BAF local.

---

## 8. Reading the output

Results land in `out/results/`:

- `ledger_<stage>.jsonl` — append-only, one line per completed unit. **This is
  the durable artefact.** Back it up.
- Summary tables written per stage on completion.

The ledger flattens each unit's parameters with a `u_` prefix, so to aggregate:

```python
import json, pandas as pd
rows = [json.loads(l) for l in open('out/results/ledger_benchmark.jsonl', encoding='utf-8') if l.strip()]
d = pd.DataFrame([r for r in rows if 'auprc' in r])
print(d.groupby('u_model')[['auprc', 'dollar_recall_at_k']].agg(['mean', 'std']))
```

### What to expect, and what not to

On synthetic data (3 seeds × 3 folds), post-leakage-fix:

| model | AUPRC | ±sd | Dollar-Recall@1% |
|---|---|---|---|
| halo | 0.0509 | 0.0080 | **0.1208** |
| lightgbm | 0.0501 | 0.0115 | 0.0283 |
| catboost | 0.0447 | 0.0092 | 0.0620 |

Read this honestly: **on raw AUPRC, HALO and LightGBM are tied.** 0.0509 vs
0.0501 is well inside one standard deviation. The separation is in
Dollar-Recall@1%, where C4's dollar-aligned loss gives ~4×.

This is expected and was predicted before the code was written. C1, C2, C3, C4
and C6 are all portable to LightGBM — anyone can adopt them. "Beat LightGBM on
AUPRC" is therefore the wrong target, and a paper built on it will be refereed
into the ground. The defensible claims are:

- **W1** — win at *equal auditability*, i.e. against a monotone-constrained
  LightGBM, not an unconstrained one.
- **W2** — flatten the λ curve, so detection at 7–30 days approaches what
  previously required 120.
- **W3** — make λ = 0 operationally viable at all.

Those live in `latency` and `ablation`, not in `benchmark`.

### Reading the ablation

`stage_ablation` prints a **paired** summary and also writes it as `T4_v2_paired`.
Paired is the only correct reading: every variant is fitted on the identical
(dataset, seed, fold) cells, so the seed-to-seed spread cancels. Comparing
unpaired means throws that away — on synthetic it inflates the spread ~5×
(sd 0.0047 unpaired vs 0.0008 paired) and makes every real effect vanish into
noise. v1 already made this mistake once and published a ranking that was
sampling variation.

Read the columns as: `d_auprc` is the mean paired delta against `full`, `sd_auprc`
is the standard deviation *of those deltas*, and `wins_auprc` is a sign count out
of `n_pairs`. It is a sign count, not a p-value; with 3–5 seeds nothing here is
significant and the table must not be written up as if it were.

Two traps this table is built to expose:

- **`fallback_rate`.** If the `full` variant shows a non-zero rate, the alert
  region was degenerate (<200 rows or <5 positives) and the specialist trained on
  everything — so the C5 rows (`no_monotone`, `global_monotone`) are not
  measuring what their labels say. A loud warning is printed when this happens.
  It is a fold-size problem: the region is the top `alert_quantile × margin` =
  3% of scores, so a 1,200-row training fold gives ~36 rows and cannot populate
  it. Real datasets at 590k rows will not hit this; synthetic always does.
  `n/a` in that column means the region machinery never ran at all
  (`enforce=False`, or no monotone features survived the ablation), which is a
  different thing from running and not falling back.
- **`no_risk` is Block B entirely off**, not "risk columns off". The same pass
  emits the association-risk, entity-memory and velocity features, so switching
  it off removes all of them. `no_memory` is the narrower ablation. On synthetic
  this leaves a **one-feature model**, which is why `no_risk` appears to *win* —
  an artefact of the generator, not a finding.

C1 has no row here. Censored-label learning needs an imposed reporting delay to
ablate against, so it lives in `stage_censoring`.

---

## 9. Troubleshooting

| symptom | cause | action |
|---|---|---|
| `[resume] repaired ledger, dropped N trailing bytes` | power cut mid-append | none — this is the system working |
| `[resume] unreadable checkpoint quarantined` | torn checkpoint | none — it will be recomputed |
| `[cache] evicted N cached feature tables` | cache hit its budget | raise `HALO_CACHE_GB` if the disk allows |
| everything recomputes after a pull | feature code changed | expected; the fingerprint retired stale work |
| `no datasets found on disk; falling back to 'synthetic'` | wrong paths | check §3 against `cli_v2 datasets` |
| `[skip] not loadable, dropped from this run` | named a dataset you have not downloaded yet | expected; the run continues with the rest |
| `ImportError: pyarrow` on reading results | no parquet engine | `pip install pyarrow`, or read the `.jsonl` ledger |
| a `verify` property fails | a real leak | **stop.** Do not run experiments. Fix it first. |

That last row is not a formality. A benchmark number produced by a leaky
estimator is worse than no number, because it looks like evidence.

---

## 10. Reproducing a result from scratch

```bash
pip install numpy pandas scikit-learn scipy lightgbm xgboost catboost pyarrow
python -m halo.cli_v2 smoke
python -m halo.cli_v2 selftest
python -m halo.cli_v2 verify
# place data as in §3
python -m halo.cli_v2 datasets
python -m halo.cli_v2 run --stage ladder    --datasets auto --seeds 5 --max-hours 2
python -m halo.cli_v2 run --stage benchmark --datasets auto --seeds 5 --max-hours 2
python -m halo.cli_v2 run --stage latency   --datasets auto --seeds 5 --max-hours 2
python -m halo.cli_v2 run --stage ablation  --datasets auto --seeds 5 --max-hours 2
python -m halo.cli_v2 run --stage censoring --datasets auto --seeds 5 --max-hours 2
python -m halo.cli_v2 status
```

Re-run any line as many times as it takes. That is the design.
