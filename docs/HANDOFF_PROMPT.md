# HANDOFF PROMPT — HALO / Entity-Leakage Fraud Detection Experiment

> **How to use this file:** copy everything below the line into a fresh Claude Code session.
> It is fully self-contained. The new session needs no other context.

---

# MISSION

You are building the complete, runnable, end-to-end experimental pipeline for a data-mining
research paper. The pipeline runs on **Kaggle Notebooks** against the **IEEE-CIS Fraud
Detection** dataset (with a secondary **PaySim** case study). You will produce working code,
a step-by-step run guide, real measured results, an auto-generated report, a downloadable ZIP
of all artefacts, an adversarial self-review, and a written justification of why the method works.

**Integrity requirement, absolute:** every number in every table and figure must come from code
that actually ran. Never fabricate, estimate, extrapolate, or "illustrate" a result. If something
did not run, say so explicitly and leave the cell empty with a stated reason. The entire paper is
an argument about research integrity — fabricated numbers would be self-refuting.

---

# PART 1 — BACKGROUND YOU MUST INTERNALISE

## 1.1 The project

- **Course:** CSE 4891 Data Mining (E). **Team:** "Team Transparent" — Md. Wali Ullah Khan,
  Ahnaf Atique, Abir Reza, Abir Hossain, Nadia Akter Labonno.
- **Target venues:** ECML-PKDD Applied Data Science, DSAA, IEEE Big Data, CIKM short paper
  (realistic); KDD ADS / ICDM (stretch).
- **Timeline:** 12 weeks. The go/no-go evidence must land by week 4.

## 1.2 The original proposal, and why it was rejected

The team's first draft was titled *"Interpretable Multimodal Mining for Early Financial Fraud
Detection from Transaction and Identity Data."* It claimed three novelties, **all of which are
established technique and would be dismissed by a reviewer in one sentence each**:

1. *"Missingness as first-class signal"* → this is the **missing-indicator method**
   (Little & Rubin). Every public IEEE-CIS Kaggle kernel since 2019 adds `nan_count` and
   per-column NaN flags.
2. *"Lightweight multimodal fusion of identity + transaction"* → this is a `LEFT JOIN` on
   `TransactionID`. The two tables are one table by design.
3. *"Quantitative faithfulness evaluation of SHAP"* → this is **ROAR** (Hooker et al.,
   NeurIPS 2019), plus deletion/insertion curves and comprehensiveness/sufficiency (ERASER 2020).

The pipeline itself (EDA → preprocessing → feature engineering → GBDT → explainability) was sound.
Only the *claim* changed. Everything below reuses that pipeline, pointed at an open question.

## 1.3 THE HINGE FACT — the single most important thing in this document

The IEEE-CIS data was contributed by **Vesta**. In Kaggle discussion thread **101203**, a Vesta
representative explained how the `isFraud` label was actually assigned. **This statement is not in
the dataset documentation, not in the competition overview, and appears in none of the 19 academic
papers the team collected.** Paraphrased faithfully:

> The labelling logic defines a reported chargeback on the card as a fraud transaction, and
> transactions **posterior** to it that are linked by user account, email address, or billing
> address as fraud **too**. If none of the above is reported within **120 days**, the transaction
> is defined as legitimate.

Three consequences follow, and each breaks something the field assumes:

**(a) The label is an entity state, not a transaction property.**
Once a card is reported, every *later* transaction on that card is labelled fraud whether or not it
was itself fraudulent. A large share of the positive class are not fraudulent acts — they are
downstream transactions on an already-known-compromised entity. Therefore a model maximising AUC on
this benchmark is rewarded for **recognising the entity**, not for recognising fraudulent behaviour.
This is precisely why the famous Kaggle "UID magic" worked: the winning move was **entity
resolution** (reconstructing client identity from `card1`, `addr1`, `D1`), not better fraud
modelling. This has been Kaggle folklore for six years and has never been written up as what it is:
evidence that the benchmark is partly a re-identification task.

**(b) A chronological split does not fix it.** Time-ordered splitting removes temporal leakage but
the *same card* still sits on both sides of the split, carrying its label. Entity leakage passes
straight through untouched.

**(c) Labels are censored and delayed by up to 120 days.** At real scoring time a transaction's
label does not exist yet. Every paper trains on supervision that would not have been available.

## 1.4 The paper's three contributions

- **C1 — A leakage taxonomy and audit** for transactional fraud benchmarks (five channels, below),
  with a "leakage ladder" quantifying each channel's contribution to reported performance.
- **C2 — Task reformulation:** the Cold-Entity Protocol (CEP) plus four new metrics.
- **C3 — HALO**, a streaming method designed to survive CEP, with explanations that are faithful
  by construction rather than by post-hoc audit.

**The one claim to actually make:** *entity-propagated labels make IEEE-CIS partly an entity
re-identification benchmark; under an entity-disjoint, latency-honest protocol the reported state of
the art collapses; and a streaming association-risk method recovers most of the remaining signal
with explanations that are faithful by construction.*

## 1.5 Honesty ledger — what may be claimed vs. cited

Do not let the report overclaim. Prior work exists for several components:

| Component | Closest prior work | Permitted framing |
|---|---|---|
| L1 resampling leakage | MDPI *Mathematics* 13(16):2563 (2025); arXiv 2412.07437 | **Cite only.** Do not claim novelty. |
| Leakage as reproducibility crisis | Kapoor & Narayanan, *Patterns* (2023) | Domain instantiation **plus a repair**. |
| Verification latency | Dal Pozzolo, Boracchi, Caelen, Alippi, Bontempi — IJCNN 2015 | Prior work used a private stream, no entity structure, not reproducible. Ours is the first public-benchmark instantiation and shows latency × entity leakage **interact**. |
| UID entity resolution | Kaggle folklore (Deotte et al., 2019) | Never published, never validated label-free, never used as a leakage argument. |
| Cold-entity evaluation | Exists for *review* fraud (DFraud³, SparseFraudNet) | First for card-not-present transactions with propagated labels. |
| Missingness pattern submodels | Fletcher Mercaldo & Blume, *Biostatistics* (2020) | Ours mines at 394-column scale, interprets as provenance, uses for label-free drift. |
| Monotone GBDT | Standard in credit scoring | Reframed as a faithfulness *guarantee* with measured coverage and price. |

---

# PART 2 — THE FIVE LEAKAGE CHANNELS

| ID | Channel | Mechanism | Status |
|---|---|---|---|
| **L1** | Resampling leakage | SMOTE/ADASYN or scaling fitted **before** the train/test split; synthetic minority points interpolate across the split. | Cite, don't claim |
| **L2** | Temporal leakage | Random k-fold over a time-ordered stream; target/frequency encodings fitted over the full period. | Cite, don't claim |
| **L3** | **Entity leakage** | Labels propagate across an entity's posterior timeline, so the same card sits in train and test with its outcome attached. Model memorises identity, is scored as if it predicted behaviour. | **NEW — never audited on this dataset** |
| **L4** | **Latency leakage** | Supervision used before the 120-day chargeback window closed. Those labels do not exist at scoring time. | **NEW for this benchmark** |
| **L5** | **Generative-determinism leakage** | (PaySim only) The label is a near-deterministic function of fields the simulator derives it from, rather than of behavioural signal. | **NEW — see §6** |

---

# PART 3 — THE COLD-ENTITY PROTOCOL (CEP)

Implement exactly this. Deviations must be flagged and justified in the report.

- **Rolling-origin folds.** The IEEE-CIS training period spans ~182 days. Split it into monthly
  test folds. For test month *m*, train **only** on transactions whose labels had matured before
  *m* began.
- **Entity-disjoint.** Entities appearing in a test fold are removed from that fold's training set.
  **Report both the entity-disjoint condition and the all-entities condition — the gap between them
  IS the measurement of L3.**
- **Latency gate δ.** A transaction at time *t* may use labels only from transactions before
  *t − δ*. Sweep δ ∈ {0, 7, 30, 60, 120} days. **Headline setting: δ = 30** (realistic
  card-not-present chargeback lag). **Strict setting: δ = 120** on a reduced window.
- **Scope.** All work happens inside `train_transaction.csv` / `train_identity.csv`, because the
  Kaggle test labels are not public. State this in the **protocol** section, not the limitations
  section — it is what makes the study reproducible.
- **Known tension to report honestly:** with only 182 days of data, δ = 120 leaves little mature
  supervision. Do not hide this. Naming a real constraint reads as rigour; hiding it reads as an error.

## 3.1 Metrics — implement all four, plus secondaries

| Metric | Definition |
|---|---|
| **Index-AUPRC** | AUPRC computed over *index events* only — each entity's **first** fraudulent transaction. Propagated positives are **excluded from the positive class** (decide and document: excluded entirely, or masked from evaluation). This is the number that matters and nobody reports it. |
| **Cold-Entity AUPRC** | AUPRC restricted to entities never seen during training. Directly quantifies how much performance was memorisation. |
| **Time-to-Detection (TTD)** | Transactions **and** hours between the model's first alert on an entity and that entity's index event. Negative = early detection. Report the full distribution, not just a mean. |
| **Dollar-Recall@k** | Fraction of fraudulent `TransactionAmt` recovered within a *k%* manual-review budget. k = 1% is the headline. Define tie-breaking explicitly. |

Secondary: AUPRC, AUROC, calibration (reliability curve + Brier / ECE), alert precision at review
capacity, throughput (µs/transaction).

**Drop accuracy entirely.** At a 3.5% base rate it is noise dressed as a result.

**Base-rate caveat you must handle:** AUPRC is not comparable across folds with different base
rates. Report the positive base rate for every fold, and additionally report **lift over base rate**
or normalised AUPRC so cross-fold and cross-protocol comparisons are valid.

---

# PART 4 — HALO: THE MODEL

**H**onest **A**ssociation under **L**atency and **O**rdering. Four blocks. All classical data
mining, all single-pass streaming, no GPU required. The design question it answers: *when you are
forbidden from memorising the entity, what signal is left?* Answer: risk that flows over **shared
infrastructure** — which is what real fraud teams use, and what entity leakage has been masking.

## Block A — Entity resolution by constraint mining (unsupervised, label-free)

This is the load-bearing block. L3 cannot be measured without it.

```
day = floor(TransactionDT / 86400)
D1n = day - D1          # the card's first-seen day (D1 = days since card began transacting)
uid = card1 ‖ addr1 ‖ D1n
```

- Block on `uid`, then refine with a Fellegi–Sunter style weighted agreement score over
  `card2`–`card6`, `P_emaildomain`, `dist1`, `DeviceInfo`, `id_30`/`id_31`/`id_33`.
- Transitive closure with a **conflict guard**: reject merges implying two different card brands
  (`card4`), or inconsistent `D`-column arithmetic.
- Emit **three UID variants — strict / medium / loose** — and re-run every downstream result on all
  three. Sensitivity to the UID definition must be reported.

**The validation trick (do not skip this — it pre-empts the obvious reviewer objection):**
`C1`–`C14` are cumulative per-card counts, so **within a true entity they must be non-decreasing in
time**. The C-monotonicity violation rate is therefore an **unsupervised precision proxy** for the
entity resolution — you validate identity reconstruction without ever touching the label. Handle
NaNs explicitly when computing it.

## Block B — Latency-gated empirical-Bayes association risk

For each attribute *a* ∈ {card1, card2, addr1, P_emaildomain, R_emaildomain, DeviceInfo, id_31,
id_33, ProductCD×card4, BIN-proxy} and each value *v*, maintain at scoring time *t*:

```
                α₀ + Σ w(t − tᵢ) · yᵢ
r(a, v, t) = ─────────────────────────────
              α₀ + β₀ + Σ w(t − tᵢ)

  over  i : tᵢ < t − δ  and  aᵢ = v
  w(Δ)  = exp(−Δ / τ),   τ ≈ 30 days
  (α₀, β₀) fitted by empirical Bayes (moment-matching on population fraud rate + attribute overdispersion)
```

- The Beta prior is what makes this **not** target encoding: shrinkage toward the population rate
  kills the small-count leakage that sinks naive mean-encoding.
- **One-hop damped propagation**: entity → shared device → other entities, damping γ ≈ 0.4.
  Personalised-PageRank flavour, computed with incremental counters. **No GNN.**
- **Entity-memory features** (is this entity known-compromised as of *t − δ*? days since? count of
  prior confirmed frauds?) live in a **separate, separately-ablatable feature group**, so memory and
  behaviour contributions can be reported apart. This separation is essential to C1.
- O(1) per transaction, single pass, streaming. Measure and report µs/transaction — against a GNN
  baseline this is a deployability claim with numbers behind it.

## Block C — Missingness-regime mining

The team's original "missingness as signal" idea, upgraded from flags into actual mining. The
`V`-columns arrive in blocks with identical NaN counts — that block structure is a fingerprint of
*which upstream enrichment service fired*.

- Build the 590,540 × 394 binary missing-indicator matrix **M**.
- Mine **closed frequent itemsets** over M (LCM or CHARM; `mlxtend` or `pyfim`; min support 0.5%).
- Interpret each maximal pattern as a **data-provenance regime**; expect ~20–60. Assign every
  transaction a regime label.
- Use it three ways: (i) categorical feature; (ii) **stratum for per-regime calibration**;
  (iii) **label-free drift monitor** — the regime distribution is observable immediately while
  labels take 120 days. Under verification latency you *cannot* monitor drift with labels, so a
  label-free trigger is not a nicety, it is the only option. This closes the loop back to L4.

**⚠ Leakage hazard — you must handle this:** mining itemsets on the *full* dataset (including test
folds) is transductive leakage. **Mine regimes on the training window only**, then assign test
transactions to the nearest existing regime. Document the assignment rule for unseen patterns.

## Block D — Monotone, cost-sensitive GBDT (faithful by construction)

- LightGBM with **monotonic constraints (+1)** on every Block-B risk feature and every velocity
  feature whose direction is known a priori. **Verify the sign convention empirically** — do not
  assume `+1` means what you think it means for your feature orientation.
- Monotonicity makes a class of counterfactual claims **provably** true. The model emits
  **verifiable reason codes** with exact thresholds found by binary search along the monotone
  coordinate: *"this alert clears if the device's 60-day association risk falls below 0.041."*
- Faithfulness = 1.0 by construction on that subspace. Then measure the two things that matter:
  **coverage** (share of alerts explainable this way) and **the price of the guarantee**
  (ΔAUPRC, constrained vs. unconstrained).
- Cost-sensitive objective with an explicit false-negative : investigation-cost ratio, tuned on
  Dollar-Recall@1%.

**Framing:** the field does *faithfulness by post-hoc audit*; HALO does **faithfulness by
construction, with a measured price**. The team's original SHAP-vs-ablation protocol survives — as
the baseline being beaten.

---

# PART 5 — THE EXPERIMENT GRID (T1–T7, F1–F3)

Every baseline runs under **both** protocols — leaky and CEP. **The comparison between protocols is
the result; the comparison between models is secondary.**

| ID | Artefact | Content |
|---|---|---|
| **T1** | **Leakage ladder** (headline table) | AUPRC at each rung, **LightGBM held constant**. Rung 0: random split + SMOTE pre-split. Rung 1: close L1. Rung 2: close L2. Rung 3: close L3. Rung 4: close L4. Rung 4+HALO. |
| **T2** | Entity statistics (**the go/no-go**) | Entity count, size distribution, C-monotonicity violation rate, **label purity**, and share of positives that are propagated rather than index. |
| **T3** | Main results under CEP | Logistic regression · XGBoost · LightGBM · CatBoost · MLP · SMOTE variants · a GNN baseline (GraphSAGE or lightweight ATM-GAD) · HALO. |
| **T4** | HALO ablation | Blocks A/B/C/D on and off; entity-memory features separated from behaviour features. |
| **T5** | Latency sensitivity | δ ∈ {0, 7, 30, 60, 120} days × all models. |
| **T6** | Faithfulness | Reason-code coverage, ΔAUPRC price of monotonicity, vs. SHAP+ROAR baseline. |
| **T7** | Operating cost | Dollar-Recall@1%, alert precision at review capacity, throughput µs/transaction. |
| **F1** | **SHAP mass migration** (the figure people will screenshot) | Attribution mass by feature family, leaky protocol vs. CEP. Expect mass on identity proxies (`card1` freq-encodings, `D`-columns, `V`-block components) under leaky, migrating to behavioural/association features under CEP. |
| **F2** | Time-to-detection | Distribution of lead time relative to each entity's index event. |
| **F3** | Label-free drift | Missingness-regime distribution shift plotted against AUPRC decay over the 182 days. |

*(Note: the request mentioned "T1–T70"; the grid is T1–T7 plus F1–F3. If additional sub-tables are
warranted, number them T3a, T3b, etc.)*

**Hypothesised leakage ladder** (from the planning document — these are the *predictions to test*,
NOT results. Your job is to measure the real values; report them whatever they are):

| Rung | Condition | Hypothesised AUPRC |
|---|---|---|
| 0 | All leaks open | ~0.99 |
| 1 | Close L1 | ~0.88 |
| 2 | Close L2 | ~0.72 |
| 3 | Close L3 | ~0.52 |
| 4 | Close L4 | ~0.43 |
| 4+ | HALO under CEP | ~0.58 |

**Critical fairness requirement:** the leaky-protocol baselines must be **steel-manned**, tuned as
carefully as HALO. If you strawman the baseline, the entire ladder is invalid and the paper is
worthless. Give the baselines the same hyperparameter search budget.

**Variance requirement:** run ≥5 seeds and report mean ± std for every headline number. Ladder
deltas that are within noise are not results.

---

# PART 6 — PAYSIM SECONDARY CASE STUDY (L5)

PaySim is a synthetic agent-based simulation (Lopez-Rojas et al.), Kaggle dataset `ealaxi/paysim1`
(~6.3M rows, 744 hourly steps ≈ 30 days, ~0.13% fraud).

**Do NOT use PaySim to "validate generalization."** It is trivially solvable, and reporting a second
inflated number would undercut the paper's own thesis. **Use it as a second case study proving the
audit methodology generalises across data-generating processes.** This upgrades the contribution
from "we audited one dataset's quirk" to "we built a general benchmark-auditing method that finds a
*different* failure mode in a structurally unrelated dataset."

**The L5 mechanism to demonstrate:**
- In the simulator, transactions flagged as fraudulent are **cancelled**, so for `isFraud=1` rows
  the balance fields (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
  frequently do not update the way a genuine transaction's would. A model learning
  "amount ≠ (oldbalance − newbalance) ⇒ fraud" achieves near-perfect scores by detecting an artifact
  of how the simulator writes rows.
- `isFlaggedFraud` is set by a hard rule inside the simulator (single transfer > 200,000). It is not
  a feature; it is the label's cousin baked in by an `if` statement.

**The L5 audit:** report AUPRC with vs. without the balance-delta features and `isFlaggedFraud`.
The gap **is** the measurement. Expect collapse from ~1.0 to something far more modest.

**Note honestly:** Block A (entity resolution) does not transfer — `nameOrig`/`nameDest` are largely
single-use identifiers with little repeat structure, so L3/L4 are not PaySim's story. That is the
point: different data-generating process, different failure mode, same audit lens.

---

# PART 7 — WHAT TO BUILD

## 7.1 Code structure

Write modular, importable Python (not one monolithic notebook). Suggested layout:

```
halo/
  config.py            # all hyperparameters, seeds, paths, δ sweep values — single source of truth
  data.py              # loading, dtype downcasting, transaction↔identity join
  entities.py          # Block A: UID derivation, Fellegi–Sunter refinement, conflict guard,
                       #          transitive closure, C-monotonicity validation
  risk.py              # Block B: latency-gated Beta posteriors, time decay, damped propagation
  regimes.py           # Block C: missing-indicator matrix, closed itemset mining, regime assignment
  model.py             # Block D: monotone cost-sensitive LightGBM, reason-code generation
  protocol.py          # CEP: rolling-origin folds, entity-disjoint splits, latency gating
  metrics.py           # Index-AUPRC, Cold-Entity AUPRC, TTD, Dollar-Recall@k, calibration, lift
  baselines.py         # LR, XGB, LGBM, CatBoost, MLP, SMOTE variants, GNN baseline
  ladder.py            # T1 rung orchestration
  explain.py           # SHAP, ROAR, reason-code coverage, F1 mass-migration
  paysim.py            # L5 case study
  report.py            # results aggregation → HTML/Markdown report
  package.py           # ZIP bundling
```

**Every module must be runnable and tested on a small subsample first.** Build a `--smoke` mode
that runs the entire pipeline on ~50k rows in under 5 minutes, so correctness is verified before
committing to the full grid.

## 7.2 Kaggle execution plan

The full grid (~200 model fits, 10–14 hours) **exceeds a single Kaggle session's runtime limit**.
Design for this from the start:

- **Stage the work across multiple notebooks.** Each notebook's output persists in
  `/kaggle/working/` and can be published as a **Kaggle Dataset**, which the next notebook attaches
  as input. This is the correct Kaggle chaining pattern.
- Suggested staging:
  - **NB-0 — Setup & smoke test.** Attach data, verify environment, run `--smoke`, confirm shapes.
  - **NB-1 — Block A + T2 (the go/no-go).** Entity resolution, C-monotonicity, label purity.
    *Publish output as a Dataset: `halo-entities`.*
  - **NB-2 — T1 leakage ladder.** Rungs 0–4. *Publish: `halo-ladder`.*
  - **NB-3 — Blocks B & C.** Association risk + regime features. *Publish: `halo-features`.*
  - **NB-4 — Block D + baselines (T3, T5, T7).** May need splitting further by δ.
  - **NB-5 — Ablations & drift (T4, F2, F3).**
  - **NB-6 — Explainability (T6, F1).**
  - **NB-7 — PaySim L5 case study.**
  - **NB-8 — Report generation + ZIP packaging.**
- **Checkpoint aggressively.** Persist intermediate results as Parquet/Feather after every stage so
  a timeout never costs more than one stage.
- **Verify resource limits yourself at runtime** rather than trusting remembered numbers — print
  `psutil.virtual_memory()`, `os.cpu_count()`, and available disk at the start of every notebook,
  and record them in the report's environment section. Kaggle's CPU/RAM/runtime/output quotas change;
  do not hard-code assumptions about them.
- **Data attachment:** IEEE-CIS is a *competition* dataset (`ieee-fraud-detection`) — competition
  rules must be accepted before it can be added via "Add Data". PaySim is `ealaxi/paysim1`.
- No GPU is required for HALO. Only the GNN baseline may want one; keep it a baseline, not a
  dependency, and subsample the graph if needed.

## 7.3 Step-by-step run guide

Produce a `RUN_GUIDE.md` that a team member with no context can follow:
account setup → accepting competition rules → creating each notebook → attaching inputs →
expected runtime per stage → what output each stage produces → how to publish it as a Dataset for
the next stage → how to verify a stage succeeded → what to do when a stage times out.
Include exact cell contents or `%run` invocations, not prose descriptions.

## 7.4 The report

`report.py` generates a self-contained report (HTML preferred, Markdown acceptable) containing:
- Environment & reproducibility block (library versions, seeds, resource limits observed, git-style
  hash of `config.py`).
- Dataset descriptives + T2 entity statistics.
- All tables T1–T7 and figures F1–F3, **with real measured values**, mean ± std over seeds, and
  per-fold base rates.
- The PaySim L5 case study.
- The adversarial review (Part 8) with each item marked resolved / mitigated / open.
- The "why it will work" justification (Part 9).
- An explicit **limitations** section.
- Every hypothesised value from §5 shown side-by-side with the measured value, so the reader can see
  where the prediction held and where it failed. **Where it failed, say so plainly** — that is a
  finding, not an embarrassment.

## 7.5 The ZIP deliverable

`package.py` bundles into `/kaggle/working/halo_results.zip`: all code, all result tables (CSV +
the report), all figures (PNG + SVG), the run guide, the adversarial review, `config.py`, the
environment manifest, and a `README.md` explaining the contents. Keep it under Kaggle's output size
cap; if figures push it over, split into `halo_results_code.zip` and `halo_results_figures.zip`.
Confirm the ZIP is downloadable from the notebook's output panel, and state in the final response
exactly where to click to download it.

---

# PART 8 — MANDATORY ADVERSARIAL SELF-REVIEW

Before declaring the pipeline complete, **red-team your own work**. For each item below, state
whether it is **resolved**, **mitigated** (with the mitigation), or **open** (with the risk). Do not
skip items. Add any further vulnerabilities you find.

## Leakage vulnerabilities (the paper is about leakage — a leak here is fatal)
1. **`D1n` uses `TransactionDT`.** Does deriving the UID from a time-based column introduce temporal
   information into the entity definition in a way that leaks? Prove it does not, or fix it.
2. **Itemset mining scope.** Are regimes mined on the training window only, or on the full dataset?
   Full-dataset mining is transductive leakage. Verify.
3. **Empirical Bayes prior fitting.** Are (α₀, β₀) fitted using any data from the test fold? They
   must not be.
4. **Off-by-one at fold boundaries.** Is the Beta posterior update *strictly* causal — is `t − δ`
   exclusive, and are same-timestamp transactions handled correctly?
5. **Frequency/target encodings.** Any encoding fitted across the split boundary?
6. **Scaler/imputer fitting.** Fitted inside folds only?
7. **SMOTE placement.** In the CEP arms, is resampling strictly inside the training fold?
8. **Hyperparameter selection.** Was any hyperparameter chosen using test-fold performance?
   Use a separate inner validation split. τ, γ, and the cost ratio are especially at risk.

## Statistical validity
9. **Base-rate comparability.** AUPRC across folds/protocols with different base rates — handled?
10. **Seed variance.** Are ladder deltas larger than the seed-to-seed standard deviation?
11. **Multiple comparisons.** ~200 fits — is any headline claim a multiple-testing artifact?
12. **Index-AUPRC degeneracy.** After restricting to index events, are there enough positives for
    the metric to be stable? Report the count.
13. **Entity-disjoint split bias.** Removing test entities from training shrinks and shifts the
    training distribution. Is the comparison to the all-entities condition still apples-to-apples,
    or does training-set size need to be controlled?
14. **Dollar-Recall@k ties.** How are equal scores ranked? Does it change the number?

## Method correctness
15. **Monotone constraint sign.** Verified empirically per feature, not assumed?
16. **C-monotonicity with NaNs.** Does the validation metric behave correctly when `C*` is missing?
17. **UID sensitivity.** Do conclusions hold across all three UID variants? If they diverge, say so.
18. **Conflict guard coverage.** What fraction of merges are rejected? Are false merges plausibly
    still present, and what would that do to T2?
19. **Propagated-vs-index labelling.** How is the index event identified when an entity's first
    fraud is itself ambiguous or the entity is truncated by the window boundary?
20. **Reason-code coverage denominator.** Coverage of what, exactly — all alerts, or only alerts
    whose decisive feature was monotone?

## Fairness to baselines
21. **Steel-manning.** Did baselines get equal tuning budget? Document the budget for each.
22. **GNN baseline.** Is it a fair implementation, or a crippled one? If subsampled, does that
    disadvantage it, and is that disclosed?

## Reproducibility
23. Are all seeds fixed and recorded? Is the pipeline deterministic end-to-end? If not, where does
    nondeterminism enter, and is it bounded by the reported variance?

---

# PART 9 — WRITE THE "WHY IT WILL WORK" JUSTIFICATION

Produce a written section arguing, with reference to your **measured** results, why the method and
the finding hold up. The planning document's a-priori reasoning is below — **update each point
against what you actually observed**, and mark any that your results contradict.

1. **The effect is large, not delicate.** Leakage effects of this kind are measured in tens of
   AUPRC points, not tenths. The result does not hinge on a hyperparameter, a seed, or a lucky
   architecture — it is structural.
2. **The central claim is one table away.** If entities are ~99% label-pure (which the Vesta rule
   implies), L3 is proven by a single descriptive statistic — T2.
3. **Block B genuinely transfers to cold entities.** Shared-infrastructure risk is the one signal
   that survives entity disjointness: a brand-new card can still transact from a device or email
   domain that has history. That is how production fraud teams actually operate, and the benchmark's
   leakage has been masking its value.
4. **The paper cannot fail.** If HALO beats the baselines under CEP → method paper. If it does not →
   a stronger negative result: *even a purpose-built method reaches only ~0.4-something, so the
   honest ceiling is far below what the field reports.* Both outcomes are publishable.
5. **It runs on a laptop.** Every block is O(n) single-pass. "0.58 AUPRC at ~40µs/transaction on one
   core" is a competitive claim against the GNN/transformer thread in its own right.
6. **It is data mining, not deep learning.** Entity resolution, closed itemset mining, Bayesian
   shrinkage, damped graph propagation, boosted trees — the right toolbox for a data mining course
   and for ICDM/PKDD reviewers.

---

# PART 10 — WORKING AGREEMENT

- **Smoke-test first.** Never launch the full grid before `--smoke` passes end-to-end.
- **Report honestly.** If a stage fails, times out, or produces a result that contradicts the
  hypothesis, report it plainly with the evidence. Do not quietly drop it or tune until it agrees.
- **No fabricated numbers, ever.** Empty cells with stated reasons are acceptable; invented values
  are not.
- **Checkpoint everything.** Assume any session can die at any moment.
- **Flag scope problems early.** If part of the grid is infeasible within Kaggle's limits, finish
  everything else in full and state explicitly what was left out and why — scaling down is the
  team's call, not yours.
- **Ask before assuming** on genuinely ambiguous design decisions (e.g. exact index-event definition
  at window boundaries); make routine judgement calls yourself and document them.

## Definition of done

- [ ] All modules written, smoke-tested, and run on full data
- [ ] T1–T7 and F1–F3 produced with real measured values, mean ± std over ≥5 seeds
- [ ] PaySim L5 case study complete
- [ ] `RUN_GUIDE.md` written and followable by someone with no context
- [ ] Report generated, including hypothesised-vs-measured comparison and limitations
- [ ] Adversarial review completed, every item marked resolved / mitigated / open
- [ ] "Why it will work" justification written against measured results
- [ ] `halo_results.zip` built, verified downloadable, and its download location stated

---

**Start by confirming your understanding of the hinge fact (§1.3) and the CEP protocol (§3), then
propose your build order before writing code.**
