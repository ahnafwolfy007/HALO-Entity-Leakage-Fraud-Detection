# HALO — Entity-Leakage Fraud Detection

Research pipeline for *"Guilt by Association: Entity Leakage and Latency-Honest
Evaluation in Transaction Fraud Detection."*

**Team Transparent** · CSE 4891 Data Mining (E)
Md. Wali Ullah Khan · Ahnaf Atique · Abir Reza · Abir Hossain · Nadia Akter Labonno

---

## The claim in one paragraph

The IEEE-CIS labelling rule — stated by Vesta in a Kaggle forum reply, and absent from the
dataset documentation and from every academic paper on the dataset — propagates a fraud
label from a reported chargeback to **every subsequent transaction** on the linked card,
email or billing address, and marks anything unreported after 120 days as legitimate.
Three things follow: the label is an entity state rather than a transaction property, so
high AUC partly rewards entity re-identification; a chronological split does not fix it,
because the same card sits on both sides carrying its label; and labels are censored for
up to 120 days, so every published result trains on supervision that would not have
existed at scoring time. This repository audits all of that, and proposes a method that
survives an honest protocol.

## What is here

```
halo/            the pipeline (importable package)
  config.py      every hyperparameter, seed and path — the single source of truth
  synth.py       synthetic IEEE-CIS generator WITH ground truth (verification harness)
  data.py        loading, joining, base features, matrix preparation
  entities.py    Block A — entity resolution + label-free C-monotonicity validation
  risk.py        Block B — latency-gated empirical-Bayes association risk
  regimes.py     Block C — missingness-regime mining + label-free drift monitor
  model.py       Block D — monotone cost-sensitive GBDT + exact reason codes
  protocol.py    the Cold-Entity Protocol and the leaky protocols it is contrasted with
  metrics.py     Index-AUPRC, Cold-Entity AUPRC, TTD, Dollar-Recall@k
  baselines.py   LR, MLP, LightGBM, XGBoost, CatBoost + the SMOTE-placement switch
  experiments.py T1–T7, F3 orchestration
  explain.py     T6 (faithfulness) and F1 (SHAP mass migration)
  paysim.py      L5 — the PaySim generative-determinism audit
  report.py      HTML report assembly
  package.py     ZIP bundling
  cli.py         one subcommand per Kaggle stage

notebooks/       nine Kaggle stage notebooks (run make_notebooks.py to regenerate)
RUN_GUIDE.md     step-by-step Kaggle instructions, written for someone with no context
ADVERSARIAL_REVIEW.md   red-team pass; 12 resolved, 6 mitigated, 5 open
WHY_IT_WORKS.md  the justification, written against measured results
HANDOFF_PROMPT.md  self-contained context for a fresh session
```

## Quick start

```bash
python -m halo.cli smoke --entities 3000 --seeds 0 1
```

Runs every stage against generated data with known ground truth in a few minutes. This is
the verification harness, and it is the thing to run before spending a Kaggle session.

Against real data (place the CSVs in `data/ieee-fraud-detection/`, or run on Kaggle):

```bash
python -m halo.cli run-entities     # T2 — the go/no-go
python -m halo.cli run-ladder       # T1 — the headline table
python -m halo.cli run-main         # T3
python -m halo.cli report           # report + ZIP
```

See `RUN_GUIDE.md` for the full Kaggle chain.

## The five leakage channels

| ID | Channel | Status |
|---|---|---|
| L1 | Resampling leakage (SMOTE before the split) | cite, don't claim |
| L2 | Temporal leakage (random k-fold on a stream) | cite, don't claim |
| **L3** | **Entity leakage** (labels propagate across an entity's timeline) | **new** |
| **L4** | **Latency leakage** (labels used before the 120-day window closed) | **new here** |
| **L5** | **Generative-determinism leakage** (PaySim's simulator artefacts) | **new** |

## Verified behaviour

Measured on synthetic data with known ground truth (`cli smoke`):

- entity-resolution pair-F1 **1.000** (medium/strict), 0.9994 (loose)
- C-monotonicity violation rate **0.00000** when resolution is correct, **0.00022** when
  over-merged — the label-free check detects over-merging without reading a label
- entity label purity **0.983–0.986**; propagated share of positives **0.79**
- leakage ladder AUPRC **0.812 → 0.808 → 0.449 → 0.079 → 0.079**; the L3 step alone
  removes 82% of measured performance

See `WHY_IT_WORKS.md` for the full reading, including where the planning document's
predictions were wrong.

## Integrity rules this code enforces

- `assert_uid_is_label_free` raises if an entity key ever reaches the feature matrix.
  It has already caught one real leak introduced by refactoring.
- `RegimeMiner` has no `fit_transform`; the leaky path must be requested explicitly.
- `AssociationRisk.transform` raises rather than silently mis-computing if input is not
  time-sorted, and uses a strict inequality at the latency boundary.
- Every risk feature is **leave-one-entity-out**, so association signal cannot become
  entity memory by the back door.
- The report renders un-run stages as `NOT RUN`, never as an estimate.

## Open items before submission

From `ADVERSARIAL_REVIEW.md`:

1. **Training-size control (item 13).** Entity-disjoint splitting also shrinks training
   data. Until rung 2 is re-run with a size-matched training set, the rung 2 → 3 drop is
   not cleanly attributable to leakage.
2. **GNN baseline (item 22).** Not implemented. Narrow the claim, or run one.
