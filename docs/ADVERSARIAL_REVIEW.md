# Adversarial Review — HALO pipeline

Red-team pass over the implementation. Each item is marked **RESOLVED** (fixed and
verified in code), **MITIGATED** (guarded, but the guard has limits), or **OPEN** (a real
risk that survives; must be reported in the paper's limitations).

The paper's thesis is that other people's pipelines leak. A leak in *this* pipeline
would be fatal to the argument, so items 1–8 got the most attention.

Two vulnerabilities were found and fixed *during* development, not after. They are
written up as items 1 and 2 because they are the kind of thing that would otherwise have
survived to publication.

---

## A. Leakage vulnerabilities

### 1. `D1n` is derived from `TransactionDT` — does the UID smuggle time into the split?
**RESOLVED.** The concern is real: `D1n = floor(TransactionDT/86400) − D1`. Three defences:

- `D1n` is *constant per entity* by construction, so it carries no within-entity temporal
  ordering.
- It is computed without reading any label.
- It is used only to **group** rows, never as a model feature.

The third point was not true when first written. `prepare_matrix` passed `D1n` and `D15n`
straight through into the feature matrix, and the model would have been handed the exact
identity key the Cold-Entity Protocol exists to withhold. This was caught by
`assert_uid_is_label_free`, which is called on every fold's feature list and raises rather
than warns. Both columns are now in the drop set, and the assertion remains as a
permanent tripwire.

> Lesson worth stating in the paper: the leak was introduced by ordinary refactoring and
> was invisible in the metrics — AUPRC simply looked good. Only an explicit assertion
> caught it.

### 2. Are missingness regimes mined on the full dataset? (transductive leakage)
**RESOLVED.** `RegimeMiner` is a fit/transform estimator. `fit` sees training rows only;
`transform` assigns unseen patterns to the nearest known regime by Hamming distance.
There is deliberately **no `fit_transform` on the full frame** — the convenient method is
absent so it cannot be reached for by accident. `build_fold_features` passes
`df.iloc[fold.train_idx]` under `past_only=True`.

The leaky path is still reachable, but only by explicitly setting `past_only=False`,
which is what rungs 0–2 of the ladder *are*: the leak is a first-class experimental
condition, not an accident.

### 3. Does the empirical-Bayes prior see test data?
**RESOLVED.** `AssociationRisk.fit_prior` is called with `df.iloc[fold.train_idx]` and
the corresponding training labels only. The prior is moment-matched on per-value fraud
rates among values with at least `eb_min_count` observations.

### 4. Off-by-one at the latency boundary
**RESOLVED.** Label ingestion uses a separate pointer with a **strict** inequality:
`while t[ingest_ptr] < t[i] − delta`. Same-timestamp rows therefore cannot see each
other, and a row can never contribute to its own risk features. Verified by construction
in `AssociationRisk.transform`, which also raises if the input is not time-sorted rather
than silently producing wrong numbers.

### 5. Encodings fitted across the split boundary
**MITIGATED.** `prepare_matrix` uses ordinal factorisation only, which carries no label
information, so fitting it across the split is safe. Frequency and target encodings —
the genuinely dangerous kind — are deliberately **not implemented**; their honest
equivalent is Block B, which is latency-gated and shrunk.

Residual risk: ordinal codes are assigned by order of appearance, so the *code values*
differ between folds. This is harmless for tree models but would matter for a linear
model. Logistic regression and MLP are in the baseline set, so their numbers are
slightly pessimistic on high-cardinality categoricals. Disclosed rather than hidden.

### 6. Scaler / imputer fitted inside folds only
**RESOLVED.** `SimpleImputer` and `StandardScaler` live inside sklearn `Pipeline`
objects, fitted in `fit_predict` on the training fold only.

### 7. SMOTE placement
**RESOLVED, and it is the experiment.** `resample_before_split=True` oversamples the
whole frame *then* honours the split, so synthetic points interpolated from test-fold
minority rows land in training — that is L1, reproduced faithfully. With `False`,
resampling happens inside the training fold. The switch is per-rung config, not a code
path anyone has to remember to set.

### 8. Hyperparameters chosen on test-fold performance
**RESOLVED.** `_inner_search` carves an 80/20 **time-ordered** split from the training
fold and never touches the test fold. Budget is a shared parameter applied equally to
every tunable model.

**OPEN:** `tau_days`, `gamma_damping`, and `fn_cost_ratio` are currently fixed constants
in `config.py`, not tuned. That is honest (no test data was used) but means HALO may be
running below its own ceiling. If the paper tunes them, it must be through
`_inner_search` on the inner validation split, and the paper must say so.

---

## B. Statistical validity

### 9. AUPRC is not comparable across folds with different base rates
**RESOLVED.** Every AUPRC is accompanied by `base_rate` and `auprc_lift = AUPRC /
base_rate`. Fold base rates are recorded in `Fold.meta()` and appear in every raw table.
The report uses lift whenever it compares across protocols.

This matters more than it sounds: entity-disjoint splitting *changes the test base rate*,
so a naive AUPRC comparison between rung 2 and rung 3 would partly measure base-rate
drift rather than leakage.

### 10. Seed variance — are ladder deltas larger than noise?
**MITIGATED.** `aggregate_seeds` reports mean and std over ≥5 seeds by default, and every
raw per-fold row is persisted (`T1_raw`, `T3_raw`, …) so variance can be re-examined.

**OPEN:** no formal significance test is applied. Given ~200 fits, the paper should
report deltas with std and avoid claiming small differences. A delta inside one std is
not a result and must not be described as one.

### 11. Multiple comparisons
**OPEN.** The grid runs ~200 fits. The *headline* claims (L3 collapse, HALO vs baseline)
are pre-registered in the sense that they were specified before running, which limits the
garden of forking paths — but no correction is applied. Report the number of comparisons
and treat any secondary finding as exploratory.

### 12. Index-AUPRC degeneracy
**MITIGATED.** `index_auprc` returns `n_index_positives` alongside the metric, and
returns NaN rather than a misleading number when the positive class is empty. On the
synthetic run, index events are ~1% of rows, so the metric is noisier than plain AUPRC.

**OPEN:** on real IEEE-CIS, per-fold index-event counts may be small enough that
Index-AUPRC is unstable. The paper must report the count beside every Index-AUPRC and
should pool folds if counts are low.

### 13. Entity-disjoint splitting shrinks and shifts the training set
**RESOLVED, AND ANSWERED on synthetic data (87.8% leakage, 5-seed).** This was the most
important open item. Implemented as `run_size_control` (`halo/cli.py run-size-control`,
tables `T1b` and `T1b_attribution`).

Removing test entities from training reduces training size (measured per fold as
`dropped_entity_rows`). So the rung 2 -> rung 3 drop conflates *removing memorisation*
with *training on less data*. The control runs three conditions on the same folds:

| Condition | Entity leakage | Training set |
|---|---|---|
| `rung2_full` | open | full |
| `rung2_size_matched` | open | randomly cut to rung 3's size |
| `rung3_entity_disjoint` | closed | full remaining |

`T1b_attribution` then decomposes the total drop:

    rung2_full -> rung2_size_matched   = the price of less data
    rung2_size_matched -> rung3        = what entity leakage was actually worth

and reports `leakage_share_of_drop`. **The paper must quote that share rather than
attributing the whole rung 2 -> 3 drop to leakage.**

Measured on synthetic data, 5 seeds (a 2-seed first pass reported 98.6% — rerun at full
seed count before trusting a number this load-bearing):

    total drop                      0.5004
    attributable to training size   0.0609
    attributable to entity leakage  0.4395
    leakage share of drop           87.8%

Cutting the training set to rung 3's size while leaving entity leakage open costs 0.061
AUPRC; closing entity leakage at that same size costs 0.440 — about seven times more. The
size confound is real, larger than the 2-seed estimate suggested, but still a minority of
the effect. A third, independent confirmation comes free from the ladder itself:
cold-entity AUPRC measured *inside* rung 2 on the full training set is 0.066, against
rung 3's 0.078 — no training rows removed at all.

**Still required:** re-run on IEEE-CIS and quote that share, not this one.

### 14. Dollar-Recall@k tie-breaking
**RESOLVED.** Ties are broken **randomly** with a fixed seed via
`np.lexsort((rng.random(n), -s))`. Breaking ties by amount would quietly optimise the
very metric being reported.

---

## C. Method correctness

### 15. Monotone constraint sign — and is the guarantee actually exact?
**RESOLVED, with a correction to the claim.** `HaloModel.verify_monotonicity` sweeps each
constrained feature upward on sampled rows and counts violations, rather than trusting
LightGBM's `+1` to mean what we assume.

Running it produced a finding that changes what the paper may claim. The first version
reported **16 violations across 14 features, only 10 verified clean** — which looked like
a broken sign convention. It was not. Two things were wrong, and the second is a real
limitation:

1. **A bug in the probe.** The sweep grid was built from the *full* frame's feature range,
   including test rows outside the training range. Probing beyond the last histogram bin
   boundary extrapolates, and the resulting non-monotonicity says nothing about the
   constraint. Fixed: `fit` now records `train_ranges_` and the probe stays inside it.
2. **The guarantee is exact over LightGBM's binned representation, not over the reals.**
   After the probe fix, violations persist at a measured magnitude of
   **max 5.35e-4 in probability space** (7 raw violations at machine precision, 18 of 18
   features within a 1e-3 tolerance).

So the honest formulation is **not** "faithfulness = 1.0 by construction". It is:

> Monotonicity holds to within a measured bound of ~5e-4 in probability space — orders of
> magnitude below any realistic alerting threshold, and therefore exact for every
> operational purpose, but not exact in the mathematical sense.

`T6` now reports `monotone_max_violation` alongside the counts so the bound is stated
rather than assumed. **The paper must quote the measured bound.** Claiming exactness
would be the same category of overclaim the paper accuses the field of.

### 16. C-monotonicity with NaNs
**RESOLVED.** Missing `C*` values are skipped in both numerator and denominator. Treating
a missing count as zero would manufacture violations that are artefacts of imputation.
Verified on synthetic data: violation rate is exactly 0.00000 for correct resolution.

### 17. UID sensitivity
**RESOLVED.** T2 runs all three variants. On synthetic data, `medium` and `strict`
recover ground truth exactly (pair-F1 = 1.000); `loose` over-merges slightly
(pair-precision 0.999) — and the C-monotonicity violation rate rises from 0.00000 to
0.00022 in step, which **demonstrates the label-free validation actually detects
over-merging**. That is a publishable micro-result in its own right.

**OPEN:** on real data there is no pair-F1 to check against. The C-monotonicity rate is
the only available signal, and its *absolute* scale is uncalibrated there.

### 18. Conflict-guard coverage
**MITIGATED.** `conflict_rejects` and `score_rejects` are recorded in T2. False merges
may still survive when `card4`/`card6`/`D15n` are all missing on one side.

### 19. Index-event identification at window boundaries
**MITIGATED.** `left_truncated_entities` counts entities whose first *observed*
transaction is already positive — their true index event may precede the window, so
their TTD is a lower bound. Reported in T2 as a caveat rather than silently dropped.

### 20. Reason-code coverage denominator
**RESOLVED.** Coverage is computed over **all** alerts, not only those whose decisive
feature happened to be monotone. The latter would read ~100% by construction and would
be circular.

---

## D. Fairness to baselines

### 21. Steel-manning
**MITIGATED.** All models get identical features, folds, and tuning budget; the budget is
a single shared parameter. LightGBM/XGBoost share one random-search grid.

**OPEN:** CatBoost, logistic regression and the MLP are not tuned at all (budget applies
only to the two GBDTs). Either extend `_inner_search` to cover them or state plainly in
the paper that untuned baselines are a limitation. **Do not report an untuned baseline
losing to a tuned HALO as evidence for HALO.**

### 22. GNN baseline
**OPEN — not implemented.** T3 currently covers LR, MLP, LightGBM, XGBoost, CatBoost and
HALO. The planned GraphSAGE / lightweight ATM-GAD baseline is absent. The paper cannot
claim to beat graph methods until it is run. Options: implement it, or narrow the claim
to tabular baselines and say so explicitly.

---

## E. Reproducibility

### 23. Determinism
**MITIGATED.** Seeds are set for every model and for tie-breaking; `config_hash()` is
written into the report and the ZIP manifest; the environment manifest is captured at
runtime rather than assumed.

**OPEN:** LightGBM with `n_jobs=-1` is not bit-reproducible across thread counts. Numbers
may shift in the last decimal between machines. Bounded by the reported seed variance,
but the paper should not quote four significant figures.

---

## Summary

| Status | Count | Items |
|---|---|---|
| **RESOLVED** | 13 | 1, 2, 3, 4, 6, 7, 9, 13, 14, 15, 16, 17, 20 |
| **MITIGATED** | 6 | 5, 10, 12, 18, 19, 21, 23 (partial) |
| **OPEN** | 4 | 8 (untuned HALO constants), 11 (multiple comparisons), 21 (untuned baselines), 22 (**no GNN baseline**) |

### What must still be closed before submission

1. **Item 22 — the GNN baseline.** Not implemented. Either run one, or narrow the claim
   to tabular baselines and say so explicitly. This is now the only structural gap.
2. **Item 13 is resolved in code but must actually be run** on the real data, and its
   `leakage_share_of_drop` quoted in the paper. Implementing the control is not the same
   as having its answer.
3. **Item 21 — untuned baselines.** Either extend the inner search to CatBoost / LR / MLP,
   or state the limitation. Never present an untuned baseline losing to a tuned HALO as
   evidence for HALO.

Everything else can be handled in a limitations section written honestly.

---

## Vulnerabilities found and fixed during development

Worth listing separately, because they are evidence the review was real rather than
decorative:

0. **The monotonicity guarantee was weaker than claimed.** See item 15: the constraint
   holds to ~5e-4, not exactly, and the first verification run was itself buggy. Both the
   bug and the residual bound were found by the check rather than by inspection.
1. **`D1n` / `D15n` reached the feature matrix.** The entity key itself was being handed
   to the model. Caught by `assert_uid_is_label_free`, which raises rather than warns.
   Metrics looked *better* with the leak in place, so nothing except the assertion would
   have caught it.
2. **Synthetic ground truth was itself corrupted.** Intra-day random seconds were added on
   top of a fractional day, which pushed rows into the next calendar day and broke the
   `day - D1 == first_day` invariant — only 14.8% of entities had a constant key. Fixed by
   flooring the day first; now 100%. A verification harness that is silently wrong is
   worse than none.
3. **C-monotonicity was reporting 1.87% violations on clean data.** The cumulative counters
   were built in `day` order while rows were stored in `TransactionDT` order. Fixed by
   re-sorting and recomputing within-entity rank before building the counters. The rate is
   now exactly 0.00000 when resolution is correct and 0.00022 when over-merged — which is
   what makes it usable as a precision proxy.
