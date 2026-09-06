# Run Guide — HALO on Kaggle

Written for a team member with no prior context. Follow it top to bottom.

Total wall-clock: roughly **6–10 hours of compute**, spread across 9 notebooks so that no
single Kaggle session runs out of time.

---

## 0. Before you touch Kaggle: run the smoke test locally

This takes minutes and will save you a wasted session.

```bash
cd /path/to/paper
python -m halo.cli smoke --entities 3000 --seeds 0 1
```

It generates synthetic IEEE-CIS-shaped data with known ground truth and runs **every**
stage against it. You should see `PASSED` listing all nine stages. If anything says
`FAILED`, fix it before going near Kaggle.

What the smoke test proves, and why it matters: the synthetic generator produces entities
whose `day − D1` is constant, Vesta-style label propagation, and monotone `C` columns. So
when the pipeline reports entity-resolution pair-F1 = 1.000 and a C-monotonicity violation
rate of 0.00000, that is the code being *correct*, not the data being easy. It is the only
place you can check the audit against a known answer.

---

## 1. Kaggle account setup

1. Create a Kaggle account and **verify your phone number** (Settings → Phone
   Verification). Without this you cannot use accelerators or the Kaggle API.
2. Go to the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection)
   and click **Late Submission** / **Join Competition** to accept the rules. You cannot
   attach the data to a notebook until you have accepted them.
3. Optional, for NB-7: no action needed — `ealaxi/paysim1` is a public dataset.

---

## 2. Upload the `halo/` package as a Kaggle Dataset

Every notebook imports the package, so it needs to live somewhere Kaggle can see.

1. Zip the package folder only:
   ```bash
   cd /path/to/paper
   zip -r halo-src.zip halo -x "*__pycache__*"
   ```
2. On Kaggle: **Datasets → New Dataset**. Title it exactly **`halo-src`**. Upload
   `halo-src.zip`. Kaggle unzips it, so the notebook will see
   `/kaggle/input/halo-src/halo/…`.
3. Set it to Private. Click **Create**.

> Re-upload a new version of this dataset whenever you change the code. Notebooks pick up
> the latest version when you re-attach.

---

## 3. The notebook chain

Nine notebooks. Each one's output feeds the next. Upload the generated `.ipynb` files
from `notebooks/` (**Code → New Notebook → File → Import Notebook**), or copy the cells
by hand.

| # | Notebook | Attach as input | Produces | Rough time |
|---|---|---|---|---|
| 0 | `NB0_setup_and_smoke` | `halo-src` | environment check, smoke pass | 5–10 min |
| 1 | `NB1_entities` | `halo-src`, competition data | **T2 — the go/no-go** | 20–40 min |
| 2 | `NB2_ladder` | + `halo-stage1` | **T1 — leakage ladder** | 2–4 h |
| 3 | `NB3_features` | + `halo-stage2` | Block B/C sanity check | 20–40 min |
| 4 | `NB4_main` | + `halo-stage3` | T3, T5, T7 | 3–5 h |
| 5 | `NB5_ablation_drift` | + `halo-stage4` | T4, F3 | 1–2 h |
| 6 | `NB6_explain` | + `halo-stage5` | T6, **F1** | 40–90 min |
| 7 | `NB7_paysim` | + `halo-stage6`, `ealaxi/paysim1` | L5 | 30–60 min |
| 8 | `NB8_report` | + `halo-stage7` | report + **ZIP** | 5 min |

### Attaching data

In any notebook, right panel → **Add Data**:
- **Competitions** tab → search `ieee-fraud-detection` → Add
- **Datasets** tab → search your `halo-src` → Add
- **Datasets** tab → search your `halo-stageN` → Add

### Publishing a stage's output so the next stage can read it

This is the step people get wrong. After a notebook finishes:

1. Click **Save Version** → *Save & Run All (Commit)*. Wait for it to complete.
2. Open the completed version → **Output** tab.
3. Click **New Dataset** (top-right of the Output panel).
4. Name it exactly **`halo-stage1`** (or `halo-stage2`, …, matching the table above).
5. Create it.

The bootstrap cell in each notebook automatically copies `checkpoints/`, `results/` and
`figures/` forward from any attached dataset whose name starts with `halo-stage`, so you
do not need to change any code.

---

## 4. Verifying each stage succeeded

| Stage | What to check |
|---|---|
| NB-0 | Prints `PASSED` for all nine stages |
| NB-1 | Prints `GO/NO-GO: entity label purity = …` — **record this number** |
| NB-2 | `T1.csv` exists in Output; rung 2 → rung 3 shows a clear drop |
| NB-3 | Prints fold table and Block C column-block sizes (expect large V blocks) |
| NB-4 | `T3.csv`, `T5.csv`, `T7.csv` in Output |
| NB-5 | `T4.csv`, `F3.csv` in Output |
| NB-6 | `T6.csv`, `F1.csv`, and `figures/F1_shap_mass_migration.png` |
| NB-7 | `L5.csv`, `L5_evidence.csv` |
| NB-8 | `halo_results.zip` in Output |

---

## 5. When a stage times out

Kaggle will kill a session that exceeds its runtime limit. Nothing is lost if you have
been committing versions, because every stage checkpoints as it goes.

Recovery, in order of preference:

1. **Reduce seeds.** `main(["run-main", "--seeds", "0", "1", "2"])` instead of five.
   Report the reduced seed count in the paper — it is a real limitation, not a secret.
2. **Split the notebook.** NB-4 is the usual offender; move `run-latency` and `run-cost`
   into their own notebook attached to the same input.
3. **Reduce the latency sweep.** T5 over five δ values is the most expensive single item.
4. **Subsample rows** with `--rows 200000` as a last resort. This changes the base rate
   and the entity-size distribution, so say so explicitly in the report if you do it.

Do **not** respond to a timeout by quietly dropping an experiment. The report marks
missing tables as `NOT RUN` on purpose.

---

## 6. Getting the results out

After NB-8 completes:

1. Open the notebook version → **Output** tab (right-hand panel).
2. Find **`halo_results.zip`**.
3. Click the **download icon** beside it.

Alternatively, **Download All** on the Output tab, or use the API:

```bash
kaggle kernels output <your-username>/<notebook-slug> -p ./halo_output
```

The ZIP contains:

```
halo/           the pipeline source
results/        every result table as CSV, with .meta.json sidecars
figures/        generated figures
notebooks/      these stage notebooks
docs/           run guide, adversarial review, why-it-works, handoff prompt
HALO_report.html    ← open this first
HALO_summary.json   machine-readable summary
MANIFEST.txt        contents and config hash
```

---

## 7. Before you write the paper

Two things from `ADVERSARIAL_REVIEW.md` must be closed first:

- **Item 13 — the training-size control.** Entity-disjoint splitting also shrinks the
  training set, so part of the rung 2 → rung 3 drop could be sample size rather than
  leakage. Re-run rung 2 with the training set randomly subsampled to rung 3's size. If
  the drop persists it is entity leakage. A reviewer *will* ask this.
- **Item 22 — the GNN baseline.** It is not implemented. Either run one, or narrow the
  claim to tabular baselines and say so.

And one standing rule: **the report marks stages that did not run as `NOT RUN`.** Leave
those boxes in. A paper arguing that the field reports inflated numbers cannot itself
quietly fill gaps with estimates.
