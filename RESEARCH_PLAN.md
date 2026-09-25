# HALO v2 — Research Plan Toward an A* Submission

Status: proposal. Nothing here is a measured result. Every number marked *(est.)* is a
prior expectation to be tested and reported honestly, including when it fails.

---

## Part 0 — Where v1 actually stands

### 0.1 We are tied with LightGBM, not behind it

| λ (days) | HALO | LightGBM | Gap | HALO std |
|---:|---:|---:|---:|---:|
| 0 | 0.4456 | 0.4630 | −0.0174 | ±0.057 |
| 7 | 0.5089 | 0.5286 | −0.0197 | — |
| 30 | 0.5022 | 0.5118 | −0.0096 | ±0.032 |
| 60 | 0.4945 | 0.4955 | −0.0010 | — |
| 120 | 0.5062 | 0.4941 | **+0.0121** | ±0.050 |

Every gap sits inside one standard deviation. v1 neither loses nor wins in any
statistically defensible sense. To claim a win with paired tests across folds × seeds we
need roughly **+0.03 to +0.04 AUPRC**, not +0.01.

### 0.2 The "we need 120 days" premise is false

HALO at λ=7 (0.5089) is already *better* than HALO at λ=120 (0.5062), and λ=30 is
identical within noise. The λ=120 column is notable only because LightGBM degrades there.
Short-maturity detection is already achieved in absolute terms; what is missing is a
defensible margin at λ ∈ {0, 7, 30}.

### 0.3 The λ sweep tells us where the censoring damage is

Both models peak at λ=7 and decline afterwards. Two forces are visible:

- **λ=0 → 7: +0.063 to +0.066.** Label noise resolving. Most chargebacks land within a week.
- **λ=7 → 120: slow decline.** The cost of discarding recent training data.

Implication: censoring damage is concentrated at **λ=0–7**, and data-starvation damage at
**λ=30–120**. A censoring-aware method attacks both ends and should *flatten the curve*.

---

## Part 1 — Vulnerabilities in v1

| # | Vulnerability | Severity | Location |
|---|---|---|---|
| **V1** | "Damped graph propagation" is not implemented. Code computes `γ·max + (1−γ)·mean` of the *same row's* attribute risks. No neighbour traversal, no hop, no graph. | **Blocking** | `risk.py:224` |
| **V2** | All labels within δ of scoring time are discarded outright rather than modelled as right-censored. | **Blocking for λ<30 goal** | `risk.py`, ingest loop |
| **V3** | Single global τ = 30 d for all 10 attributes. A device's risk half-life ≠ an email domain's. | High | `config.py:85` |
| **V4** | Training objective (constant 20× FN weight) is misaligned with the headline metric (Dollar-Recall@1%). | High | `model.py` vs `metrics.py` |
| **V5** | Monotonicity enforced globally; reason codes are only ever issued where `s ≥ θ`. We pay everywhere, benefit in one region. | Medium-high | `model.py` |
| **V6** | HALO is entirely untuned (τ, γ, cost ratio fixed constants); `run_fold` ignores `tuning_budget` on the HALO branch. | Medium | `experiments.py:123` |
| **V7** | Unseen attribute values and well-evidenced base-rate values both yield `r = α₀/(α₀+β₀)`; only `riskn_` distinguishes them, and it is unconstrained while `risk_` is monotone — a hard interaction for a constrained tree. | Medium | `risk.py` |
| **V8** | No GNN baseline. | Blocking for A* | review item 22 |
| **V9** | Velocity history truncated at 64 events per entity. | Low | `risk.py` |

**V1 must be resolved before any submission.** Either implement real propagation (C2) or
delete the claim from README, HANDOFF_PROMPT and the paper. A reviewer who opens the repo
finds this in under two minutes.

---

## Part 2 — Contributions

### C1 — Latency-aware learning from censored fraud labels *(headline)*

**Problem.** A transaction labelled 0 at age *a* is not a negative; it is right-censored.
v1 discards the last δ days of supervision entirely — at λ=30 that is ~20–33% of the
training window, and specifically the most recent, most distribution-relevant portion.

**Model.** Let `R` be the report delay for a fraudulent transaction, CDF `F`. For a
transaction of age `a` with no report yet:

```
P(fraud | no report by a, x) = p(x)·(1 − F(a)) / (1 − p(x)·F(a))
```

Train with soft targets via EM, alternating between estimating `p(x)` and imputing
censored labels, instead of discarding them.

**Why tractable here.** The IEEE-CIS labelling rule supplies a *known hard censoring
horizon* — anything unreported at 120 days is defined legitimate. `F` therefore has
compact known support and is identifiable from delays observed within the training
window. Worth stating as a proposition.

**Leakage safety (mandatory).** `F` may only be estimated from transactions whose full
120-day window closed *before the training fold ends*. Enforce with a tripwire assertion
`assert_delay_model_is_mature`, in the style of `assert_uid_is_label_free`.

**Prior art to position against:** Chapelle (KDD'14); Ktena et al. (2019); Yasui et al.
(2020); Elkan & Noto (2008). None operate under entity-disjoint evaluation. The novelty is
the intersection plus the identifiability argument.

**Expected effect (est.):** λ=0 **+0.04–0.07**; λ=7 +0.01–0.02; λ=30 +0.005–0.01.
Secondary and more valuable effect: **flattens the λ curve**, decoupling performance from
the maturity choice.

---

### C2 — Leakage-safe multi-hop association propagation *(replaces V1)*

Build what the paper already claims. On the bipartite attribute–entity graph:

```
r^(k) = Σ_{h=1..k} γ^h · (P_{A→E} P_{E→A})^h · r^(0)
```

with the existing time-decayed edge weights.

**The theorem.** Naive multi-hop reintroduces precisely the leakage CEP exists to
eliminate — a 2-hop path can run entity → device → back to the same entity. Define
`P̃^(e)` as the operator with all paths through entity `e` deleted, and prove:

> **Proposition.** The k-hop leave-one-entity-out estimator `r_e^(k)` is independent of
> every label attached to entity `e`, for all k.

This converts the leave-one-out trick from an implementation detail into a stated
guarantee, which is what makes it publishable rather than engineering.

**Expected effect (est.):** +0.005–0.015 overall; larger on cold-entity and Index-AUPRC,
where fraud rings live.

---

### C3 — Uncertainty-aware, gated association risk

`riskn_{a}` (evidence mass) is already emitted; three things are missing.

1. **Emit Beta posterior variance** `αβ / ((α+β)²(α+β+1))` explicitly. A monotone-constrained
   tree cannot easily synthesise it from `risk_` and `riskn_` separately.
2. **Gate on evidence mass**, so the model stops paying the Block B noise tax on warm
   entities where it is net-negative:
   ```
   s(x) = σ(g(n_eff))·f_assoc(x) + (1 − σ(g(n_eff)))·f_behav(x)
   ```
3. **Per-attribute τ_a**, or a multi-scale bank {7, 30, 90} days per attribute with
   model selection.

**Motivation from v1's own ablation:** removing Block B *raises* raw AUPRC (0.5019 →
0.5083) while *lowering* Index-AUPRC (0.5812 → 0.5748). The signal is good but applied
indiscriminately.

**Expected effect (est.):** +0.005–0.012.

---

### C4 — Dollar-aligned objective

We report Dollar-Recall@1% and train with a constant 20× FN weight. Align them:

```
w_i = 1 + κ · y_i · (amount_i / mean_amount)
```

**Expected effect (est.):** ~0 on AUPRC, material on Dollar-Recall@1% — the metric an
actual fraud team optimises.

---

### C5 — Region-restricted certified monotonicity *(the only non-transferable contribution)*

Reason codes are only issued where `s ≥ θ` (`model.py::reason_codes`), yet monotonicity is
enforced across the entire input space at a cost of 0.0110 AUPRC (2.32% relative).
Restrict it:

```
∂s/∂x_j ≥ 0   ∀ x : s(x) ≥ θ − ε,   j ∈ M
```

Identical audit guarantees where reason codes are actually emitted, at strictly lower
accuracy cost.

**Expected effect (est.):** +0.004–0.009, and it is the one component a reviewer cannot
hand to LightGBM.

---

### C6 — Honest tuning

Equal inner-validation budget for HALO and every baseline, on a time-ordered inner split,
never touching the test fold. Currently HALO is untuned while the GBDT baselines have a
tuning path. Neither direction of unfairness is publishable.

**Expected effect (est.):** +0.005–0.02, and removes a guaranteed reviewer objection.

---

## Part 3 — The transfer problem, and how we win anyway

C1, C2, C3, C4 and C6 are all **transferable to LightGBM**. A competent reviewer will
demand "LightGBM + censoring-aware loss + multi-hop features + equal tuning" as a
baseline, and that baseline will land roughly where HALO lands, minus the monotonicity
residual.

**We should therefore not stake the paper on beating vanilla LightGBM on raw AUPRC.**
Three win conditions that are both achievable and more interesting:

### W1 — Win at equal auditability
The correct comparator is **monotone-constrained LightGBM**, not vanilla. Same
auditability contract, and C5 does not transfer. This is a fair fight we can win.

### W2 — Win on λ-robustness
Report the full λ ∈ {0, 7, 14, 30, 60, 120} curve for every method. Claim:
*with censoring-aware training, performance stops depending on the maturity choice.*
Operationally this is worth more than a 0.02 delta, and it is what "mature faster"
actually means — not moving 120 → 30, but making the choice stop mattering.

### W3 — Win at λ=0
Real-time scoring with zero label delay is the most operationally valuable regime and
nobody has made it work under entity-disjoint evaluation. Target 0.4456 → ~0.51 (est.),
which decisively clears LightGBM's current 0.4630.

### The standing bet (report either way)

> **Hypothesis H1.** The value of monotone constraints *increases* as label noise
> increases, because monotonicity is a prior against exactly the spurious non-monotonic
> structure noisy labels induce.

Currently unsupported and confounded: at λ=0 HALO is worst relative to LightGBM, but at
λ=0 the association counters are also starved, so any constraint benefit is buried under
Block B's collapse. **C1 un-starves the counters and makes H1 cleanly testable.**
If it holds: "auditability improves accuracy under label noise" is an A*-grade claim.
If it fails: report it. It is still a finding, and a useful one.

---

## Part 4 — Experimental programme

### Mandatory
- **GNN baselines.** GraphSAGE, PC-GNN, and one recent graph-fraud SOTA. Currently blocking.
- **Delayed-feedback baselines.** Chapelle's DFM, importance weighting. C1 must beat these,
  not merely beat nothing.
- **Datasets ≥ 3.** IEEE-CIS + PaySim is insufficient. Add Sparkov, Elliptic (native
  temporal + entity structure), ideally one industrial partner set. The leakage ladder
  replicating on a second *real* dataset is what turns "an IEEE-CIS quirk" into "a
  field-wide problem."
- **Equal tuning** for every method including HALO.
- **Statistics.** Paired tests across folds × seeds, CIs on every delta, multiple-comparison
  correction for ~200 fits. Any claim under ~0.03 AUPRC is currently indistinguishable
  from noise and must not be stated as a result.

### Ablations
Each of C1–C5 on/off; λ ∈ {0, 7, 14, 30, 60, 120} as the headline sweep; H1 test
(constrained vs unconstrained × λ).

---

## Part 5 — Venue strategy

| Venue | Fit | Requirement |
|---|---|---|
| **NeurIPS D&B** | Very strong | Leakage ladder + CEP + multi-dataset replication as the star. Least additional work. |
| **KDD ADS** | Strong | System + operational metrics + deployment story. C4 and T7 shine. |
| **KDD research** | Plausible | Needs C1 + C2 with the theorem carrying it. |
| **NeurIPS main** | Ambitious | Needs C1 identifiability + C5 certified monotonicity framed learning-theoretically. |

**Recommendation: split into two papers.**

1. **Protocol/benchmark paper** → NeurIPS D&B. Leakage ladder, CEP, multi-dataset
   replication. Nearly ready; its central claim is a measurement and is robust regardless
   of who wins the AUPRC race.
2. **Method paper** → KDD research, once C1 + C2 + C5 have real numbers.

Attempting both in one submission produces a paper that is ambitious on two axes and
convincing on neither.

---

## Part 6 — Order of work

1. **Resolve V1.** Implement C2 properly, or delete the propagation claim everywhere.
   Nothing else should reach a reviewer first.
2. **C1.** Largest payoff, directly targets λ<30, hardest to do correctly. Start the
   delay-model estimator early.
3. **C4 + C6 + C3's per-attribute τ.** Cheap wins; run in parallel with C1.
4. **C3 gating.** Fixes the Block B ablation anomaly.
5. **C5.** The differentiator that survives the transfer-to-LightGBM attack.
6. **GNN baselines + datasets 3 and 4.** Fully parallelisable; can be owned independently.

---

## Part 7 — Failure planning

It is entirely possible the honest ceiling under CEP is ≈0.51 and nothing gets us
convincingly past LightGBM on raw AUPRC.

**That outcome is still publishable, and arguably more interesting:** *"even with
delayed-feedback modelling and multi-hop propagation, the honest ceiling remains far
below what the field reports."* Plan for it now — pre-register W1/W2/W3 as the claims
rather than "we beat LightGBM," so the paper does not depend on a result we cannot
guarantee.
