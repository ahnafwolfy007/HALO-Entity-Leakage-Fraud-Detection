"""Bundle everything into a single downloadable ZIP.

On Kaggle the bundle lands in /kaggle/working and appears under the notebook's Output
tab. Kaggle caps notebook output, so if the bundle would exceed ``max_mb`` the figures
are split into their own archive rather than silently dropped.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from .config import CFG, FIG_DIR, RESULTS_DIR, WORK_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = ["RUN_GUIDE.md", "ADVERSARIAL_REVIEW.md", "WHY_IT_WORKS.md",
             "HANDOFF_PROMPT.md", "README.md"]


def write_docs() -> list[Path]:
    """Copy the written deliverables next to the results so the ZIP is self-contained."""
    copied = []
    for name in DOC_FILES:
        src = REPO_ROOT / name
        if src.exists():
            dst = WORK_DIR / name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _add_tree(zf: zipfile.ZipFile, root: Path, arc_prefix: str,
              patterns: tuple[str, ...] = ("*",)) -> int:
    n = 0
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, Path(arc_prefix) / p.relative_to(root))
                n += 1
    return n


def build_zip(name: str = "halo_results.zip", max_mb: float = 400.0) -> Path:
    """Code + results + figures + report + docs, in one archive."""
    out = WORK_DIR / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _add_tree(zf, REPO_ROOT / "halo", "halo", ("*.py",))
        _add_tree(zf, RESULTS_DIR, "results", ("*.csv", "*.json"))
        for f in ("HALO_report.html", "HALO_summary.json"):
            p = WORK_DIR / f
            if p.exists():
                zf.write(p, f)
        for d in DOC_FILES:
            p = WORK_DIR / d
            if p.exists():
                zf.write(p, f"docs/{d}")
        nb = REPO_ROOT / "notebooks"
        if nb.exists():
            _add_tree(zf, nb, "notebooks", ("*.py", "*.ipynb", "*.md"))

        size_mb = out.stat().st_size / 1e6 if out.exists() else 0.0
        if size_mb < max_mb:
            _add_tree(zf, FIG_DIR, "figures", ("*.png", "*.svg"))

        zf.writestr("MANIFEST.txt", _manifest())

    if out.stat().st_size / 1e6 > max_mb:
        figs = WORK_DIR / "halo_results_figures.zip"
        with zipfile.ZipFile(figs, "w", zipfile.ZIP_DEFLATED) as zf:
            _add_tree(zf, FIG_DIR, "figures", ("*.png", "*.svg"))
        print(f"  bundle exceeded {max_mb} MB; figures split into {figs.name}")
    return out


def _manifest() -> str:
    tables = sorted(p.name for p in RESULTS_DIR.glob("*.csv"))
    figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
    lines = [
        "HALO -- Entity-Leakage Fraud Detection",
        "=" * 60, "",
        f"config_hash : {CFG.config_hash()}",
        f"seeds       : {list(CFG.seeds)}",
        f"headline_delta_days : {CFG.headline_delta}",
        "",
        "Contents",
        "--------",
        "halo/       pipeline source (importable package)",
        "results/    every result table as CSV, with .meta.json sidecars",
        "figures/    generated figures",
        "notebooks/  Kaggle stage notebooks",
        "docs/       run guide, adversarial review, why-it-works, handoff prompt",
        "HALO_report.html   the assembled report -- open this first",
        "",
        f"Result tables present ({len(tables)}):",
        *[f"  - {t}" for t in tables],
        "",
        f"Figures present ({len(figs)}):",
        *[f"  - {f}" for f in figs],
        "",
        "Integrity note",
        "--------------",
        "Every value in results/ was produced by code in halo/. Stages that did not run",
        "are absent here and are marked NOT RUN in the report, never estimated.",
    ]
    return "\n".join(lines)
