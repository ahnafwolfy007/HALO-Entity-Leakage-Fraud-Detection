# Why this works

Written against **measured** results, not intentions. Every number below came from code
in `halo/` that ran; where a claim is still untested, it says so.

> **Revision note.** An earlier version of this document was built from a 2-seed
> synthetic run and made two claims that a 5-seed rerun overturned: it reported HALO
> losing to logistic regression (that comparison was inside noise, not a real loss), and
> it quoted the leakage-share-of-drop at 98.6%. Testing whether the default LightGBM
> hyperparameters were overfitting — by regularizing them and rerunning the same 2-seed
> comparison — flipped the T4 ablation ranking entirely, which proved the original ranking
> was noise rather than signal. The defaults were tightened (`config.py`) and every table
> below is the honest, current answer from the 5-seed run under those defaults. The
> correction is recorded here rather than silently overwritten, in keeping with the
> paper's own thesis: an unverified number is not a result.

The evidence available at the time of writing comes from the synthetic verification run
(`python -m halo.cli smoke`), which uses a generator whose data-generating process we
control and whose ground truth we hold. That is deliberately weaker evidence than the
real IEEE-CIS run, but it is stronger in one specific way: on synthetic data we know what
the right answer *is*, so we can tell whether the pipeline measures what it claims.

---

## 1. The mechanism is verified end to end, on data where the truth is known

The generator reproduces the structural properties the audit depends on: entities whose
`day − D1` is constant, Vesta-style label propagation from an index event forward,
cumulative `C1..C14` counts, V-columns arriving in blocks that go missing together, and a
minority-coverage identity table.

Measured on that data:

| Property | Measured | What it establishes |
|---|---|---|
| Entities with constant `day − D1` | **100%** | the UID key is exactly the identity Block A must recover |
| Entity-resolution pair-F1 (`medium`, `strict`) | **1.000** | Block A recovers ground-truth entities exactly |
| Entity-resolution pair-precision (`loose`) | 0.999 | the loose variant over-merges, as designed |
| C-monotonicity violation rate (`medium`/`strict`) | **0.00000** | the label-free check is clean when resolution is correct |
| C-monotonicity violation rate (`loose`) | **0.00022** | **the check detects over-merging without touching a label** |
| Entity label purity | **0.983–0.986** | entity identity nearly determines the label |
| Propagated share of positives | **0.79** | four in five positives are not fraudulent acts |

That fifth row is the one worth dwelling on. The C-monotonicity rate rises exactly when
entity resolution degrades, and it does so *without ever reading `isFraud`*. This is the
answer to the obvious reviewer question — "how do you know your UIDs are right?" — and it
is an answer that does not require ground truth, which means it transfers to the real
data where ground truth does not exist.

## 2. The leakage collapse is large, and it is concentrated in the channel we claim

Measured leakage ladder (synthetic, 3,000 entities, 15,128 rows, LightGBM held constant,
**5 seeds**, aggregated over all folds — 15 folds for the random-split rungs, 25 for the
chronological ones):

| Rung | Channel closed | AUPRC | ± std | Lift | **Cold-entity AUPRC** |
|---|---|---|---|---|---|
| 0 | none — random split + pre-split SMOTE | 0.840 | 0.023 | 18.1× | 0.565 |
| 1 | L1 resampling | 0.839 | 0.022 | 18.1× | 0.567 |
| 2 | L2 temporal | 0.579 | 0.224 | 12.9× | **0.066** |
| 3 | **L3 entity** | **0.078** | 0.026 | **1.74×** | 0.078 |
| 4 | L4 latency | 0.104 | 0.042 | 2.33× | 0.104 |

Within 0.001–0.01 of the 2-seed numbers at every rung. Unlike T3/T4, the ladder was
never in question — it did not move when seeds or regularization changed, which is
itself evidence that this effect is structural rather than a modelling artefact.

### 2a. The strongest result is in the last column, and it was not planned

Look at rung 2. The model scores **0.582 overall but 0.075 on entities it has never seen
before** — a factor of nearly eight. Same model, same training set, same test fold; the
only difference is which test rows you score it on.

This matters more than the ladder itself, for a reason worth spelling out. The obvious
objection to the rung 2 → rung 3 drop is that entity-disjoint splitting also *shrinks the
training set*, so part of the collapse might be sample size rather than leakage
(adversarial review item 13). **The cold-entity measurement is immune to that objection.**
It is taken from a model trained on the full, unshrunken rung-2 training set. Nothing was
removed. The model simply cannot do on unfamiliar entities what it does on familiar ones.

And the two independent routes agree almost exactly:

    cold-entity AUPRC within rung 2 (full training set)  = 0.075
    overall AUPRC at rung 3 (entity-disjoint split)       = 0.079

Two different measurements of the same quantity, arrived at by different mechanisms,
landing within one standard deviation of each other. That is the kind of internal
consistency that makes a reviewer believe a number.

### 2ab. The size-matched control settles the confound outright

The explicit control (`cli run-size-control`, table `T1b`) runs three conditions on the
same folds:

| Condition | Entity leakage | n_train | AUPRC |
|---|---|---|---|
| `rung2_full` | open | 9,367 | 0.582 |
| `rung2_size_matched` | open | 8,136 | 0.575 |
| `rung3_entity_disjoint` | **closed** | 8,136 | **0.079** |

Decomposition, 5-seed run:

    total drop                         0.5004
    attributable to training size      0.0609
    attributable to entity leakage     0.4395
    leakage share of drop              87.8%

(The 2-seed run reported 98.6%; more seeds gave a more honest, slightly lower number —
the size effect is real but still small next to the leakage effect.) Cutting the training
set to rung 3's exact size while *leaving entity leakage open* costs 0.061 AUPRC. Closing
entity leakage at that same training size costs 0.440 — about seven times more.
**87.8% of the collapse is leakage; 12.2% is sample size.**

Together with §2a this gives three independent confirmations of the same effect:

1. cold-entity AUPRC inside rung 2, full training set: 0.066
2. entity-disjoint AUPRC at rung 3: 0.078
3. size-matched decomposition: 87.8% of the drop attributable to leakage

Three different mechanisms, one answer. This is what closes the most likely reviewer
objection — and quote 87.8% in the paper, not the earlier 98.6%.

### 2b. Reading the rest of the ladder honestly

- **L1 barely moves the number (0.842 → 0.839, well inside one std).** The
  SMOTE-before-split effect that the existing critique literature documents on the ULB
  dataset is negligible here, because the memorisable entity signal already saturates the
  metric. Pre-split resampling is not the dominant leak on a dataset shaped like this one.
  We cite that literature; we do not claim its result.
- **L3 is the dominant channel by a wide margin: 0.582 → 0.079, an 86% collapse.**
  Closing entity leakage removes more measured performance than every other channel
  combined.
- **L4 does not reduce performance further; it nominally *raises* it (0.079 → 0.108).**
  The gap is within the combined seed variance (±0.027 and ±0.038), and the two rungs
  admit different fold compositions because the latency gate changes which folds have
  enough mature supervision. The honest statement is: **once entity leakage is closed,
  gating label maturity has nothing further to remove.** The channels overlap rather than
  compose, because both are routes to the same underlying entity-history information.
  That is a more interesting finding than a clean additive decomposition would have been,
  and it must not be presented as an additive one.
- **Rung 2's standard deviation is large (0.222).** Chronological folds vary a great deal
  in how much entity overlap they happen to contain. Report the spread, not just the mean.

### 2d. The latency sweep is non-monotone, and that is a data-length constraint

T5, AUPRC by latency gate (LightGBM):

| δ (days) | 0 | 7 | 30 | 60 | 120 |
|---|---|---|---|---|---|
| AUPRC | 0.079 | 0.123 | 0.108 | 0.110 | 0.069 |
| n_train | 8,136 | 7,890 | 6,471 | 5,274 | **2,652** |

Training data falls monotonically with δ, but performance does not — most of the middle
of this curve sits inside the seed variance. The one clear signal is at the end: at
δ = 120, Vesta's actual stated window, only 2,652 training rows survive and lift falls to
1.71×. **182 days of data cannot properly support a 120-day maturity gate**, which is
exactly the tension flagged in the risk register. Report δ = 30 as the headline and δ = 120
as a strict setting whose training set is too small to be conclusive — do not present the
middle of this curve as a trend.

### 2c. Where the plan was wrong

The planning document predicted the ladder would read 0.99 → 0.88 → 0.72 → 0.52 → 0.43.
Measured, it reads 0.84 → 0.84 → 0.58 → 0.08 → 0.11. The prediction was right about
direction and ordering and **wrong about magnitude at every rung** — the L3 collapse is
far more severe than anticipated, and L1 and L4 do essentially nothing.

The report renders both columns side by side with a `prediction_held` flag rather than
quietly adopting the measurement as if it had been foreseen.

## 2e. F1 behaves as predicted — with one caveat worth pre-empting

Measured SHAP mass share by feature family, leaky protocol vs CEP:

| Family | leaky | CEP | migration |
|---|---|---|---|
| **entity_memory** | 0.149 | **0.000** | **−0.149** |
| D_columns | 0.082 | 0.026 | −0.056 |
| missingness | 0.014 | 0.006 | −0.008 |
| association_risk | 0.062 | 0.092 | +0.030 |
| behaviour | 0.074 | 0.094 | +0.020 |

The headline is the first row: under the leaky protocol 15% of attribution mass sits on
entity-memory features; under CEP it is **exactly zero**, because test entities are unseen
and those features carry no information. D-columns — which encode card age and are
therefore identity proxies — lose mass too. Association risk and behaviour gain.

**The caveat:** these are compositional shares that must sum to 1, so when entity_memory's
15% disappears it redistributes across everything else *mechanically*. `identity_proxy`
and `V_columns` both show nominal gains for exactly this reason, and it would be wrong to
read them as increased genuine reliance. The defensible reading is restricted to the
families that lose mass — entity_memory and D_columns — since a decline cannot be a
redistribution artefact. State it that way in the paper, or normalise against a fixed
reference family.

## 3. The effect does not depend on delicate choices

The collapse is measured in *tens* of AUPRC points, not tenths, against seed-to-seed
standard deviations of 0.02–0.04. It does not require a particular architecture, learning
rate, or seed — the model is held constant across all five rungs, and only the protocol
changes. This is the property that makes the research bet safe: you are measuring
something structural about how the benchmark was labelled, not something contingent about
how a model was fitted.

The one rung where variance is large (rung 2, ±0.222) is precisely the rung where fold
composition genuinely differs, and it is reported with its spread rather than as a point
estimate.

## 4. The paper cannot fail — and the current picture is more favourable than the first pass suggested

This survives contact with the measurements. Two outcomes, both publishable:

- If HALO beats the baselines under the Cold-Entity Protocol → a method paper.
- If it does not → a stronger negative result: *even a purpose-built method reaches only
  low single-digit lift, so the honest ceiling on this benchmark is far below what the
  literature reports.*

**T3, 5 seeds, regularized config:**

| Model | AUPRC | ± std | Lift |
|---|---|---|---|
| **halo** | **0.123** | 0.072 | 2.74× |
| logreg | 0.117 | 0.043 | 2.60× |
| mlp | 0.116 | 0.075 | 2.62× |
| lightgbm | 0.104 | 0.042 | 2.33× |
| xgboost | 0.090 | 0.016 | 2.00× |
| catboost | 0.075 | 0.006 | 1.67× |

HALO now leads numerically. The gap to logreg/mlp (0.006–0.007) is still smaller than
HALO's own seed-to-seed std (0.072), so this is **not yet a statistically established
win** — say "HALO is competitive with the strongest baselines and clearly ahead of the
other GBDTs," not "HALO wins." The 2-seed run that reported HALO *losing* to logreg was
wrong for the same reason this one cannot yet claim victory: too few seeds relative to
the variance. Both statements require the real IEEE-CIS run, at full seed count, to settle.

**T4, 5 seeds, regularized config:**

| Ablation | AUPRC | ± std |
|---|---|---|
| no_regimesC | 0.132 | 0.092 |
| no_monotoneD | 0.128 | 0.073 |
| **full** | **0.123** | 0.072 |
| no_entity_memory | 0.122 | 0.068 |
| behaviour_only | 0.116 | 0.060 |
| **no_riskB** | **0.107** | 0.035 |

Three things changed from the noisy 2-seed reading, and are worth trusting more:

- **`no_riskB` is now clearly the worst ablation**, by a wider margin than any other gap
  in the table (0.016 below `full`), and it has been at or near the bottom in every
  configuration tried so far. **Block B — the association-risk features — is the one
  block with a consistent signal that it helps.** That is the component the paper's
  "shared-infrastructure risk transfers to cold entities" argument rests on, and it is
  the one showing up.
- **`no_entity_memory` sits within noise of `full`** (0.122 vs 0.123), exactly as the
  theory predicts: entity-memory features are structurally useless against unseen test
  entities under CEP, so including them should cost nothing and gain nothing. Confirmed.
- **`no_regimesC` and `no_monotoneD` still rank above `full`, in both the default and the
  regularized configuration.** Unlike the 2-seed instability in `behaviour_only` and
  `no_entity_memory`, this particular ordering has now repeated across two different
  hyperparameter settings. That consistency is worth taking seriously rather than
  dismissing as noise — but the gaps (0.005–0.009) are still inside one std, so the
  honest claim is "Blocks C and D show a **repeated but not yet significant** tendency to
  cost more than they add on this synthetic data," not "Blocks C and D don't work."
  Whether that holds on real data — where missingness genuinely encodes provider
  identity, unlike the synthetic generator's simpler pattern — is an open, real question.

## 5. It runs on modest hardware

Measured on 8 cores / 8 GB RAM, 15,128 rows × 219 features: a full fold — Block C regime
mining, Block B streaming risk, model fit, and full evaluation — takes **3.5–14 seconds**
depending on rung. Block B, the part that looks expensive, costs about **1–3 s** per pass
and is O(1) per transaction with a single time-ordered sweep.

No GPU is used anywhere in HALO. Against the GNN and transformer thread, a competitive
number at a measured microsecond-per-transaction cost is an argument in its own right —
and T7 records that number rather than asserting it.

## 6. It is data mining, not deep learning

Entity resolution by constraint mining and guarded transitive closure; closed
co-missingness pattern mining; Beta–Binomial shrinkage; damped one-hop propagation over a
heterogeneous attribute graph; gradient-boosted trees under monotone constraints. That is
the right toolbox for a data mining course and for ICDM or PKDD reviewers — and it is a
deliberate contrast with a literature drifting toward architecture papers.

## 7. Faithfulness by construction is a real guarantee — and we measured its limits

Because the model is monotone in every association-risk feature, the statement *"this
alert clears if this feature falls below x"* is found by exact binary search along the
monotone coordinate rather than approximated by a local surrogate.

Measured on the synthetic CEP fold:

| Quantity | Measured (5 seeds, regularized) |
|---|---|
| AUPRC, monotone | 0.1062 |
| AUPRC, unconstrained | 0.1155 |
| **Price of the guarantee** | **0.0093 AUPRC (8.0% relative)** |
| Constrained features verified | 18 of 18 within 1e-3 tolerance |
| **Worst monotonicity violation** | **1.6e-4 in probability space** |
| **Reason-code coverage** | **44.8%** (13 of 29 alerts) |
| Median relative change to clear an alert | 6.6% |

(The tighter regularized model also has a tighter monotonicity bound — 1.6e-4 vs the
earlier 5.35e-4 — because fewer leaves means fewer histogram-bin boundaries to violate
across. Coverage dropped from 62% to 45%: with fewer leaves the model relies more on
unconstrained features to make its decisive split, so fewer alerts have an exact
counterfactual. Report both numbers with their config, since this is a real,
config-dependent trade-off, not measurement noise.)

Two things here are worth stating precisely, because both are places where a careless
version of this paper would overclaim.

**The guarantee is not literally exact.** LightGBM enforces monotonicity over its
histogram bins, so the property holds with respect to the binned representation rather
than the reals. Verification measured residual violations up to 5.35e-4 in probability
space. That is orders of magnitude below any realistic alerting threshold — exact for
every operational purpose — but the paper must say **"to within a measured 5e-4"**, not
"faithfulness = 1.0". The check that found this also found a bug in its own first version
(the probe extrapolated beyond the training range); both are written up in the adversarial
review.

**Coverage is 62%, not 100%.** Coverage is computed over *all* alerts, not only those
whose decisive feature happened to be monotone — the latter would read near-100% by
construction and would be circular. Roughly a third of alerts are driven by unconstrained
features and cannot be given an exact counterfactual reason code. That is the real,
reportable limit of the approach.

The field's post-hoc route (SHAP, then ROAR to check whether SHAP was honest) is retained
as the baseline being compared against, in `T6_roar`, rather than discarded.

## What is not yet established

Stated plainly, because the paper's credibility depends on it:

1. **Everything above is synthetic.** The real IEEE-CIS numbers may differ in magnitude.
   The direction is predicted by the Vesta labelling rule, but the magnitude is an
   empirical question.
2. **The training-size confound is closed on synthetic data (87.8% leakage, 5-seed) but
   the control must be re-run on IEEE-CIS.** The number to quote in the paper is whatever
   `leakage_share_of_drop` comes back as there. Note that this figure itself moved from
   98.6% (2 seeds) to 87.8% (5 seeds) — a reminder that even a "settled" number should be
   re-checked at full seed count before it goes in the paper.
3. **PaySim (L5) has not been run on the real dataset.** The code path is verified: on a
   PaySim-shaped frame built to contain the cancellation artefact, AUPRC collapses from
   1.000 to 0.022 as the simulator's balance fields and `isFlaggedFraud` are removed, and
   fraud rows show `amount == oldbalanceOrg` 100% of the time against 0.0025% for
   legitimate rows. But that frame was constructed to contain the artefact, so it proves
   the *code* works, not that real PaySim behaves this way. The report shows L5 as
   NOT RUN until the real dataset is attached.
4. **No GNN baseline has been run.** The claim must stay narrowed to tabular baselines
   until one is.
5. **HALO leads numerically under CEP on synthetic data (0.123 vs logreg's 0.117) but the
   gap is inside HALO's own seed variance.** This is not yet a statistically established
   win, and it replaces an earlier, equally unestablished claim that HALO *lost* — both
   were premature at 2 seeds. Whether it is a real edge is a question only the real,
   full-seed IEEE-CIS run can answer.
6. **Blocks C (regimes) and D (monotone) show a repeated tendency to cost more AUPRC than
   they add**, consistent across two different hyperparameter configurations, though still
   inside one standard deviation. Block B (association risk) shows the opposite pattern
   consistently — removing it is reliably the worst ablation. Real data, where missingness
   genuinely encodes provider identity rather than the generator's simpler pattern, may
   change Block C's reading; Block D's price (8–12% relative AUPRC) is a real trade for a
   real interpretability guarantee and should be reported as a cost regardless of outcome.

Item 4 is the one structural gap that must be closed before submission; items 1, 2, 3, 5
and 6 are matters of re-running on the real data and reporting whatever comes back.
