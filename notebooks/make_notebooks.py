"""Generate the Kaggle stage notebooks.

Run:  python notebooks/make_notebooks.py

The full grid does not fit in one Kaggle session, so the work is staged across notebooks.
Each stage persists its checkpoints and result tables into /kaggle/working, which you
publish as a Kaggle Dataset; the next notebook attaches that dataset as input. A timeout
then costs one stage, never the whole run.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

BOOT = """\
# --- HALO bootstrap -------------------------------------------------------------
# Attach these datasets to this notebook before running (right panel -> Add Data):
#   1. Competition: "ieee-fraud-detection"      (accept the rules first)
#   2. Your source dataset: "halo-src"           (the halo/ package, see RUN_GUIDE.md)
#   3. For stages after NB-1: the previous stage's output dataset
import os, shutil, sys, subprocess, time

SRC = "/kaggle/input/halo-src"
if os.path.exists(SRC):
    if os.path.exists("/kaggle/working/halo"):
        shutil.rmtree("/kaggle/working/halo")
    shutil.copytree(os.path.join(SRC, "halo"), "/kaggle/working/halo")
sys.path.insert(0, "/kaggle/working")

# Carry forward checkpoints and results from the previous stage, if one is attached.
for prev in sorted(p for p in os.listdir("/kaggle/input") if p.startswith("halo-stage")):
    for sub in ("checkpoints", "results", "figures"):
        s = f"/kaggle/input/{prev}/{sub}"
        if os.path.isdir(s):
            os.makedirs(f"/kaggle/working/{sub}", exist_ok=True)
            for f in os.listdir(s):
                shutil.copy2(os.path.join(s, f), f"/kaggle/working/{sub}/{f}")

from halo.io import environment_manifest
env = environment_manifest()
print("ENVIRONMENT (observed, not assumed):")
for k, v in env.items():
    print(f"  {k:24s} {v}")
"""

SMOKE = """\
# --- Smoke test first. Never launch the full grid before this passes. -------------
# Runs the entire pipeline on generated data with known ground truth in a few minutes.
from halo.cli import main
main(["smoke", "--entities", "3000", "--seeds", "0", "1"])
"""


def cell(src: str, kind: str = "code") -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


STAGES = [
    ("NB0_setup_and_smoke", "NB-0 — Setup and smoke test", """
Verify the environment and prove the pipeline runs end to end **before** spending a
session on the real data. This notebook needs only the `halo-src` dataset.

Expected runtime: 3–8 minutes. If any stage reports FAILED, fix it here — do not proceed.
""", [SMOKE]),

    ("NB1_entities", "NB-1 — Block A and T2 (the go/no-go)", """
Entity resolution, the label-free C-monotonicity validation, and **T2: entity label
purity** — the number the whole paper rests on.

If purity is high, the thesis is proven and everything after this is execution.
If it is low, you learn that now, with time to pivot.

**Publish this notebook's output as a Kaggle Dataset named `halo-stage1`.**
""", ['from halo.cli import main\nmain(["run-entities"])\n']),

    ("NB2_ladder", "NB-2 — T1, the leakage ladder", """
The headline table. Each rung closes one more leakage channel with the model held
constant. Attach `halo-stage1`.

Watch the rung 2 → rung 3 step: that is entity leakage, and it is the paper.

**Publish output as `halo-stage2`.**
""", ['from halo.cli import main\nmain(["run-ladder", "--seeds", "0", "1", "2", "3", "4"])\n']),

    ("NB3_features", "NB-3 — Blocks B and C sanity check", """
Validate the association-risk and regime-mining blocks in isolation before they are
combined into the main results. Attach `halo-stage2`.

**Publish output as `halo-stage3`.**
""", ['''from halo.cli import prepare, main
import argparse, numpy as np
from halo.regimes import RegimeMiner
from halo.risk import AssociationRisk
from halo.protocol import rolling_origin_folds
from halo.config import CFG

args = argparse.Namespace(synthetic=False, entities=6000, rows=None,
                          seeds=list(CFG.seeds), delta=None, uid=CFG.uid_variant,
                          model="lightgbm", tuning_budget=0, no_cache=False)
df, entity, is_index, _ = prepare(args)
folds = rolling_origin_folds(df, entity, latency_delta=CFG.headline_delta,
                             entity_disjoint=True)
print(f"CEP folds: {len(folds)}")
for f in folds:
    print("  ", f.meta())

miner = RegimeMiner().fit(df.iloc[folds[0].train_idx])
print("\\nBlock C:", miner.summary())
print("Largest column blocks:")
for name, members in sorted(miner.column_blocks_.items(),
                            key=lambda kv: -len(kv[1]))[:8]:
    print(f"   {len(members):4d} cols  e.g. {members[:6]}")
'''] ),

    ("NB4_main", "NB-4 — T3, T5, T7 (main results)", """
All baselines and HALO under the Cold-Entity Protocol, the latency sweep, and the
operating-cost table. Attach `halo-stage3`.

This is the longest stage. If it approaches the session limit, split it: run `run-main`
here and move `run-latency` / `run-cost` into a separate notebook.

**Publish output as `halo-stage4`.**
""", ['from halo.cli import main\nmain(["run-main", "--seeds", "0", "1", "2", "3", "4"])\n',
      'main(["run-latency", "--seeds", "0", "1", "2"])\n',
      'main(["run-cost"])\n']),

    ("NB5_ablation_drift", "NB-5 — T4 and F3", """
Block ablation (does each block earn its place?) and the label-free drift loop.
Attach `halo-stage4`.

**Use all 5 seeds here, not fewer.** A synthetic-data check found the ablation ranking
flips completely under a regularization change alone when run at 2 seeds — the deltas
between ablations were smaller than their own seed-to-seed std. Do not trust this table
at reduced seed count; if the session times out, split it rather than cutting seeds.

**Publish output as `halo-stage5`.**
""", ['from halo.cli import main\nmain(["run-ablation", "--seeds", "0", "1", "2", "3", "4"])\n',
      'main(["run-drift"])\n']),

    ("NB6_explain", "NB-6 — T6 and F1 (explainability)", """
Reason-code coverage, the measured price of the monotonicity guarantee, the ROAR
baseline, and **F1 — the SHAP mass-migration figure**. Attach `halo-stage5`.

**Publish output as `halo-stage6`.**
""", ['from halo.cli import main\nmain(["run-shap"])\n',
      'main(["run-faithfulness"])\n']),

    ("NB7_paysim", "NB-7 — L5, the PaySim case study", """
A second benchmark, a different failure mode, the same audit lens. Attach the dataset
`ealaxi/paysim1` **and** `halo-stage6`.

PaySim is **not** used to validate generalisation — it is trivially solvable. It is used
to show the audit method transfers to a different data-generating process.

**Publish output as `halo-stage7`.**
""", ['from halo.cli import main\nmain(["run-paysim", "--seeds", "0", "1", "2"])\n']),

    ("NB8_report", "NB-8 — Report and ZIP bundle", """
Assemble every result table into one HTML report and bundle everything for download.
Attach `halo-stage7`.

When this finishes, open the **Output** tab in the right-hand panel, find
`halo_results.zip`, and click the download icon. That single file contains the code, all
result tables, the figures, the report, and the documentation.
""", ['from halo.cli import main\nmain(["report"])\n',
      '''# Confirm the bundle exists and show where to click.
import os
for f in sorted(os.listdir("/kaggle/working")):
    p = os.path.join("/kaggle/working", f)
    if os.path.isfile(p):
        print(f"{os.path.getsize(p)/1e6:8.2f} MB  {f}")
print("\\nDownload: right-hand panel -> Output -> halo_results.zip -> download icon.")
'''] ),
]


def main() -> None:
    for fname, title, blurb, code_cells in STAGES:
        cells = [cell(f"# {title}\n{blurb}", "markdown"), cell(BOOT)]
        cells += [cell(c) for c in code_cells]
        path = OUT / f"{fname}.ipynb"
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
