"""Report generation.

Assembles every result table that actually exists into one self-contained HTML file.

Integrity rules enforced here, not merely hoped for:
  * A table that was never produced renders as an explicit "NOT RUN" box naming the
    stage that would have produced it. It is never silently omitted, and it is never
    filled with a plausible-looking placeholder.
  * The hypothesised leakage ladder from the planning document is shown side by side
    with the measured values, so a reader can see where the prediction held and where it
    failed. Where it failed, that is a finding.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, RESULTS_DIR, FIG_DIR, WORK_DIR
from .io import Timer, environment_manifest, load_table

# The planning document's a-priori predictions. NOT results. Shown only for contrast.
HYPOTHESISED_LADDER = {
    "rung0": 0.99, "rung1": 0.88, "rung2": 0.72, "rung3": 0.52, "rung4": 0.43,
}

TABLE_SPECS = [
    ("T1", "T1 &mdash; Leakage ladder", "The headline table. One channel closed per rung, model held constant.", "run-ladder"),
    ("T1b", "T1b &mdash; Training-size control", "Separates 'removed memorisation' from 'trained on less data'. Without this the ladder's key step is confounded.", "run-size-control"),
    ("T1b_attribution", "T1b_attr &mdash; Drop decomposition", "How much of the entity-disjointness drop is leakage, and how much is sample size.", "run-size-control"),
    ("T2", "T2 &mdash; Entity statistics", "The go/no-go. Label purity is the number C1 rests on.", "run-entities"),
    ("T3", "T3 &mdash; Main results under CEP", "All baselines and HALO under the honest protocol.", "run-main"),
    ("T4", "T4 &mdash; HALO ablation", "Does each block earn its place? Memory is separated from behaviour.", "run-ablation"),
    ("T5", "T5 &mdash; Latency sensitivity", "Performance against the label-maturity gate.", "run-latency"),
    ("T6", "T6 &mdash; Faithfulness", "Reason-code coverage and the measured price of the monotonicity guarantee.", "run-faithfulness"),
    ("T6_monotone_verification", "T6b &mdash; Monotone direction check", "Empirical verification that the constraint sign means what we assumed.", "run-faithfulness"),
    ("T6_roar", "T6c &mdash; ROAR curve", "SHAP-guided removal vs random removal. The post-hoc-audit baseline.", "run-faithfulness"),
    ("T7", "T7 &mdash; Operating cost and throughput", "Dollar-Recall@1%, alert precision, microseconds per transaction.", "run-cost"),
    ("F1", "F1 &mdash; SHAP mass migration", "Attribution mass by feature family, leaky protocol vs CEP.", "run-shap"),
    ("F3", "F3 &mdash; Label-free drift", "Regime-distribution shift against AUPRC decay.", "run-drift"),
    ("L5", "L5 &mdash; PaySim generative-determinism audit", "A different benchmark, a different failure mode, the same audit lens.", "run-paysim"),
    ("L5_evidence", "L5b &mdash; PaySim shortcut evidence", "The mechanism, shown before any model is fitted.", "run-paysim"),
]


def _fmt(df: pd.DataFrame, max_rows: int = 60) -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.4f}")
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in d.itertuples(index=False))
    note = (f"<p class='note'>Showing first {max_rows} of {len(df)} rows.</p>"
            if len(df) > max_rows else "")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}"


def _ladder_comparison(t1: pd.DataFrame | None) -> str:
    if t1 is None or "rung" not in t1.columns:
        return _not_run("Hypothesised vs measured ladder", "run-ladder")
    col = next((c for c in ("auprc_mean", "auprc") if c in t1.columns), None)
    if col is None:
        return _not_run("Hypothesised vs measured ladder", "run-ladder")
    rows = []
    for _, r in t1.iterrows():
        key = r["rung"]
        hyp = HYPOTHESISED_LADDER.get(key)
        meas = r[col]
        rows.append({
            "rung": key,
            "label": r.get("rung_label", ""),
            "hypothesised_auprc": hyp if hyp is not None else np.nan,
            "measured_auprc": meas,
            "difference": (meas - hyp) if (hyp is not None and pd.notna(meas)) else np.nan,
            "prediction_held": ("" if hyp is None or pd.isna(meas)
                                else ("yes" if abs(meas - hyp) <= 0.10 else "NO")),
        })
    return _fmt(pd.DataFrame(rows))


def _not_run(title: str, stage: str) -> str:
    return (f"<div class='notrun'><strong>NOT RUN &mdash; {html.escape(title)}</strong>"
            f"<p>No result file was produced. Run <code>python -m halo.cli {stage}</code>. "
            f"This box is deliberate: an unrun experiment is reported as unrun, never as "
            f"an estimate.</p></div>")


CSS = """
:root{--bg:#F4F5F2;--card:#FCFCFA;--ink:#15191B;--ink2:#414B4E;--mut:#6E7A7C;
--rule:#DCDFD8;--teal:#1B6A61;--brass:#8C6310;--soft:#E2EEEA;--warn:#F2E8D2;}
@media(prefers-color-scheme:dark){:root{--bg:#0F1315;--card:#171D1F;--ink:#E9EEEB;
--ink2:#B6C1BE;--mut:#7F8D8A;--rule:#283134;--teal:#55BCAC;--brass:#D3A44A;
--soft:#132E2A;--warn:#2C2313;}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);margin:0;
font:15px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 90px}
h1{font-size:34px;line-height:1.15;margin:0 0 10px;letter-spacing:-.02em}
h2{font-size:22px;margin:44px 0 6px;letter-spacing:-.01em}
h3{font-size:16px;margin:26px 0 6px}
.sub{color:var(--ink2);max-width:70ch;margin:0 0 6px}
.meta{font:12px ui-monospace,Menlo,Consolas,monospace;color:var(--mut);margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:7px;
padding:18px 20px;margin:14px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;font:11px ui-monospace,monospace;color:var(--mut);
text-transform:uppercase;letter-spacing:.06em;padding:0 12px 8px 0;
border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:8px 12px 8px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
tr:last-child td{border-bottom:none}
.notrun{background:var(--warn);border-left:3px solid var(--brass);border-radius:0 6px 6px 0;
padding:14px 18px;margin:14px 0}
.notrun p{margin:6px 0 0;font-size:13.5px;color:var(--ink2)}
.note{font-size:12px;color:var(--mut);margin:6px 0 0}
code{font:12.5px ui-monospace,monospace;background:var(--soft);padding:1px 5px;border-radius:3px}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px 20px;
font:12.5px ui-monospace,monospace}
.kv div{border-bottom:1px solid var(--rule);padding:4px 0}
.kv b{color:var(--mut);font-weight:400}
.warn{border-left:3px solid var(--teal);background:var(--soft);padding:14px 18px;
border-radius:0 6px 6px 0;margin:16px 0}
ol,ul{max-width:78ch}li{margin-bottom:7px}
"""


def build_report(title: str = "HALO — Entity-Leakage Fraud Detection Results",
                 extra_sections: dict[str, str] | None = None,
                 out_path: Path | None = None) -> Path:
    env = environment_manifest()
    t1 = load_table("T1")

    parts = [f"<h1>{html.escape(title)}</h1>"]
    parts.append("<p class='sub'>Automatically generated from the result tables on disk. "
                 "Every value below was produced by code that ran; stages that did not run "
                 "are marked as such rather than estimated.</p>")
    parts.append(f"<p class='meta'>config_hash={CFG.config_hash()} &middot; "
                 f"seeds={list(CFG.seeds)} &middot; headline_delta={CFG.headline_delta}d "
                 f"&middot; generated {env['captured_at']}</p>")

    # Environment ---------------------------------------------------------------------
    parts.append("<h2>Environment</h2>")
    parts.append("<p class='sub'>Observed at runtime, not assumed. Kaggle quotas change; "
                 "these are the limits this run actually saw.</p>")
    kv = "".join(f"<div><b>{html.escape(str(k))}</b><br>{html.escape(str(v))}</div>"
                 for k, v in env.items())
    parts.append(f"<div class='card'><div class='kv'>{kv}</div></div>")

    # Hypothesis vs measurement --------------------------------------------------------
    parts.append("<h2>Hypothesised vs measured &mdash; leakage ladder</h2>")
    parts.append("<p class='sub'>The left column is the planning document's prediction, "
                 "made before any code ran. It is shown for contrast only. Where the "
                 "prediction failed, that is a finding to report, not a number to adjust.</p>")
    parts.append(f"<div class='card'>{_ladder_comparison(t1)}</div>")

    # All tables -----------------------------------------------------------------------
    for tid, title_, blurb, stage in TABLE_SPECS:
        df = load_table(tid)
        parts.append(f"<h2>{title_}</h2><p class='sub'>{blurb}</p>")
        parts.append(f"<div class='card'>{_fmt(df) if df is not None else ''}</div>"
                     if df is not None else _not_run(title_, stage))

    # Figures ---------------------------------------------------------------------------
    figs = sorted(FIG_DIR.glob("*.png"))
    if figs:
        parts.append("<h2>Figures</h2><div class='card'>")
        for f in figs:
            parts.append(f"<h3>{html.escape(f.stem)}</h3>"
                         f"<img src='figures/{html.escape(f.name)}' "
                         f"style='max-width:100%;border:1px solid var(--rule);border-radius:5px'>")
        parts.append("</div>")

    # Runtime ----------------------------------------------------------------------------
    tt = Timer.table()
    if len(tt):
        agg = tt.groupby("stage", as_index=False)["seconds"].agg(["sum", "count"])
        agg.columns = ["stage", "total_seconds", "calls"]
        parts.append("<h2>Runtime</h2><div class='card'>"
                     + _fmt(agg.sort_values("total_seconds", ascending=False)) + "</div>")

    for heading, body in (extra_sections or {}).items():
        parts.append(f"<h2>{html.escape(heading)}</h2><div class='card'>{body}</div>")

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><div class='wrap'>{''.join(parts)}</div></body></html>")

    out = Path(out_path or (WORK_DIR / "HALO_report.html"))
    out.write_text(doc, encoding="utf-8")

    # A machine-readable companion, for anyone who would rather not scrape HTML.
    summary = {"config": CFG.to_dict(), "environment": env,
               "tables_present": [t for t, *_ in TABLE_SPECS if load_table(t) is not None],
               "tables_missing": [t for t, *_ in TABLE_SPECS if load_table(t) is None]}
    (WORK_DIR / "HALO_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                encoding="utf-8")
    return out
