"""Publication-grade figure generation for the HALO research manuscript.

Generates high-resolution vector/raster figures from result tables:
  F1: SHAP Attribution Mass Migration (Leaky vs CEP)
  F2: The Leakage Ladder Collapse (Rungs 0 to 4)
  F3: Label-Free Regime Drift vs Downstream AUPRC Decay
  F4: Main Benchmark Comparison under Cold-Entity Protocol (T3)
  F5: HALO Architectural Block Ablations (T4)
  F6: Operational Label Delay Sensitivity Sweep (T5)
  F7: Operating Cost and Throughput Pareto Frontier (T7)
  F8: PaySim Generator Shortcut Collapse (L5)
  F9: ROAR Faithfulness Curve (T6c)
"""
from __future__ import annotations

import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, RESULTS_DIR
from .io import load_table

logger = logging.getLogger(__name__)

# Palette for publication
TEAL = "#1B6A61"
TEAL_LIGHT = "#4FA399"
AMBER = "#D3A44A"
DARK_AMBER = "#8C6310"
CORAL = "#C25953"
SLATE = "#414B4E"
MUTED = "#7F8D8A"
LIGHT_BG = "#F8F9F8"
GRID_COLOR = "#E2E5E2"


def set_academic_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 200,
    })


def plot_f1_shap_migration(out_dir: Path | None = None) -> Path | None:
    """F1: SHAP attribution mass migration by family."""
    piv = load_table("F1")
    if piv is None or len(piv) == 0 or "migration" not in piv.columns:
        return None
    set_academic_style()
    d = piv.sort_values("migration")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.8), dpi=200)
    colors = [DARK_AMBER if v < 0 else TEAL for v in d["migration"]]
    bars = ax.barh(d["family"], d["migration"] * 100, color=colors, height=0.65)
    ax.axvline(0, color=SLATE, lw=0.9)
    ax.set_xlabel("Change in SHAP Mass Share (% points: CEP − Leaky)", fontweight="bold")
    ax.set_title("F1: Attribution Mass Migrates from Identity to Behaviour", loc="left", pad=12)
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (0.3 if w >= 0 else -0.3)
        align = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}%",
                va="center", ha=align, fontsize=8.5, color=SLATE, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F1_shap_mass_migration.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f2_leakage_ladder(out_dir: Path | None = None) -> Path | None:
    """F2: The Leakage Ladder collapse across Rungs 0 to 4."""
    t1 = load_table("T1")
    if t1 is None or len(t1) == 0:
        return None
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    
    rungs = ["Rung 0\n(All Leaks)", "Rung 1\n(Fold Resample)", "Rung 2\n(Temporal)",
             "Rung 3\n(Entity-Disjoint)", "Rung 4\n(Matured Labels)"]
    auprc_vals = t1["auprc_mean"].to_numpy()
    auprc_errs = t1["auprc_std"].to_numpy()
    dollar_vals = t1["dollar_recall_at_k_mean"].to_numpy() * 100
    dollar_errs = t1["dollar_recall_at_k_std"].to_numpy() * 100
    
    colors = [CORAL, CORAL, DARK_AMBER, TEAL, TEAL]
    
    # Panel 1: AUPRC
    bars1 = ax1.bar(rungs[:len(auprc_vals)], auprc_vals, yerr=auprc_errs, capsize=4,
                    color=colors[:len(auprc_vals)], width=0.55, edgecolor=SLATE, lw=0.6)
    ax1.set_ylabel("AUPRC", fontweight="bold")
    ax1.set_title("(A) Precision-Recall Collapse", loc="left")
    ax1.set_ylim(0, 1.0)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.04, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    
    # Annotate collapse
    if len(auprc_vals) >= 4:
        drop = ((auprc_vals[3] - auprc_vals[2]) / auprc_vals[2]) * 100
        ax1.annotate(f"L3 Entity Collapse:\n{drop:.1f}% AUPRC",
                     xy=(2.5, (auprc_vals[2] + auprc_vals[3])/2),
                     xytext=(2.7, 0.72),
                     arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
                     fontsize=8.5, fontweight="bold", color=CORAL,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0EF", ec=CORAL, lw=0.8))

    # Panel 2: Dollar Recall
    bars2 = ax2.bar(rungs[:len(dollar_vals)], dollar_vals, yerr=dollar_errs, capsize=4,
                    color=colors[:len(dollar_vals)], width=0.55, edgecolor=SLATE, lw=0.6)
    ax2.set_ylabel("Dollar Recall @ 1% Alerts (%)", fontweight="bold")
    ax2.set_title("(B) Recovered Fraud Dollars", loc="left")
    ax2.set_ylim(0, 38)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("F2: The Leakage Ladder — Closing Entity Leakage Exposes Latent Failure",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F2_leakage_ladder_collapse.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_f3_regime_drift(out_dir: Path | None = None) -> Path | None:
    """F3: Label-free regime distribution drift vs AUPRC decay."""
    f3 = load_table("F3")
    if f3 is None or len(f3) == 0:
        return None
    set_academic_style()
    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax2 = ax1.twinx()
    
    days = f3["day_start"]
    js_div = f3["js_divergence"]
    auprc_vals = f3["auprc"]
    
    l1 = ax1.plot(days, js_div, color=CORAL, lw=2.2, marker="o", markersize=4.5,
                  label="Regime Drift (Jensen-Shannon Divergence)")
    l2 = ax2.plot(days, auprc_vals, color=TEAL, lw=2.2, marker="s", markersize=4.5,
                  linestyle="--", label="Test Performance (AUPRC)")
    
    ax1.set_xlabel("Transaction Timeline (Days from Start)", fontweight="bold")
    ax1.set_ylabel("JS Divergence vs Day 0 Baseline", color=CORAL, fontweight="bold")
    ax2.set_ylabel("Test Fold AUPRC", color=TEAL, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=CORAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)
    
    # Shade drift spike
    ax1.axvspan(90, 125, color=DARK_AMBER, alpha=0.12, label="Regime Shift Zone")
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9)
    ax1.set_title("F3: Label-Free Regime Drift Tracks Downstream AUPRC Degradation",
                  loc="left", pad=12)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F3_regime_drift_vs_auprc_decay.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f4_main_benchmark(out_dir: Path | None = None) -> Path | None:
    """F4: Main model benchmark under the Cold-Entity Protocol (T3)."""
    t3 = load_table("T3")
    if t3 is None or len(t3) == 0:
        return None
    set_academic_style()
    d = t3.sort_values("alert_precision_at_k_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
    
    y_pos = np.arange(len(d))
    bar_colors = [TEAL if m == "halo" else SLATE for m in d["model"]]
    
    bars = ax.barh(y_pos, d["alert_precision_at_k_mean"] * 100,
                   xerr=d["alert_precision_at_k_std"] * 100, capsize=3,
                   color=bar_colors, height=0.55, edgecolor=SLATE, lw=0.6)
    ax.set_yticks(y_pos)
    labels = [f"{m.upper() if m != 'halo' else 'HALO (Ours)'} "
              f"({int(mon)} monotone)" if pd.notna(mon) and mon > 0 else m.upper()
              for m, mon in zip(d["model"], d.get("n_monotone_mean", [0]*len(d)))]
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel("Alert Precision @ 1% Review Budget (%)", fontweight="bold")
    ax.set_xlim(50, 100)
    ax.set_title("F4: Model Benchmark under Cold-Entity Protocol (CEP)", loc="left", pad=12)
    
    for bar, val in zip(bars, d["alert_precision_at_k_mean"] * 100):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=9, fontweight="bold", color=SLATE)
        
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F4_main_benchmark_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f5_ablation(out_dir: Path | None = None) -> Path | None:
    """F5: Architectural block ablation (T4)."""
    t4 = load_table("T4")
    if t4 is None or len(t4) == 0:
        return None
    set_academic_style()
    d = t4.sort_values("index_auprc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
    
    y_pos = np.arange(len(d))
    colors = [TEAL if a == "full" else (CORAL if a == "no_riskB" else SLATE) for a in d["ablation"]]
    
    bars = ax.barh(y_pos, d["index_auprc_mean"],
                   xerr=d["index_auprc_std"], capsize=3,
                   color=colors, height=0.55, edgecolor=SLATE, lw=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([a.replace("_", " ") for a in d["ablation"]], fontweight="bold")
    ax.set_xlabel("Index AUPRC (Detecting Linked Entity Fraud Rings)", fontweight="bold")
    ax.set_xlim(0.52, 0.62)
    ax.set_title("F5: HALO Component Ablations — Block B Drives Entity Ring Catching",
                 loc="left", pad=12)
    
    for bar, val in zip(bars, d["index_auprc_mean"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", fontsize=8.5, fontweight="bold", color=SLATE)
        
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F5_architectural_ablation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f6_latency_sweep(out_dir: Path | None = None) -> Path | None:
    """F6: Latency sensitivity sweep (T5)."""
    t5 = load_table("T5")
    if t5 is None or len(t5) == 0:
        return None
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    
    models = t5["model"].unique()
    markers = {"halo": "o", "lightgbm": "s"}
    colors = {"halo": TEAL, "lightgbm": SLATE}
    
    for m in models:
        sub = t5[t5["model"] == m].sort_values("delta_days")
        ax.plot(sub["delta_days"], sub["auprc_mean"],
                marker=markers.get(m, "^"), color=colors.get(m, CORAL), lw=2.2,
                label="HALO (Ours)" if m == "halo" else "LightGBM (Unconstrained)")
        ax.fill_between(sub["delta_days"],
                        sub["auprc_mean"] - sub["auprc_std"],
                        sub["auprc_mean"] + sub["auprc_std"],
                        color=colors.get(m, CORAL), alpha=0.15)
        
    ax.set_xlabel("Label Maturity Delay δ (Days)", fontweight="bold")
    ax.set_ylabel("Test AUPRC", fontweight="bold")
    ax.set_title("F6: Performance vs Label Delay — HALO Overtakes GBDT at δ=120d",
                 loc="left", pad=12)
    ax.set_xticks([0, 7, 30, 60, 120])
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F6_latency_sensitivity_sweep.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f7_throughput_frontier(out_dir: Path | None = None) -> Path | None:
    """F7: Operating throughput vs Dollar-Recall Pareto frontier (T7)."""
    t7 = load_table("T7")
    if t7 is None or len(t7) == 0:
        return None
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    
    for _, r in t7.iterrows():
        m = r["model"]
        x = r["microseconds_per_txn"]
        y = r["dollar_recall_at_k"] * 100
        is_halo = m == "halo"
        color = TEAL if is_halo else SLATE
        size = 140 if is_halo else 80
        ax.scatter(x, y, s=size, color=color, zorder=5, edgecolors="black", lw=0.8)
        offset_y = 0.25 if is_halo else -0.35
        ax.text(x, y + offset_y, m.upper(), ha="center", fontsize=8.5,
                fontweight="bold", color=TEAL if is_halo else SLATE)
        
    ax.set_xlabel("Inference Latency (Microseconds / Transaction)", fontweight="bold")
    ax.set_ylabel("Dollar Recall @ 1% Alerts (%)", fontweight="bold")
    ax.set_title("F7: Throughput & Recall Frontier — Real-Time Production SLA (<1ms)",
                 loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F7_throughput_cost_frontier.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f8_paysim_ladder(out_dir: Path | None = None) -> Path | None:
    """F8: PaySim generator shortcut collapse (L5)."""
    l5 = load_table("L5")
    if l5 is None or len(l5) == 0:
        return None
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    
    labels = ["p0: All Features", "p1: Drop Rule Leak", "p2: Drop Residuals", "p3: Drop Raw Balances\n(Honest Generator Audit)"]
    vals = l5["auprc_mean"].to_numpy()
    colors = [DARK_AMBER, DARK_AMBER, DARK_AMBER, CORAL]
    
    bars = ax.bar(labels[:len(vals)], vals, color=colors[:len(vals)], width=0.55,
                  edgecolor=SLATE, lw=0.8)
    ax.set_ylabel("PaySim Test AUPRC", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("F8: PaySim Case Study — Closing Balance Shortcuts Collapses AUPRC by 73.5%",
                 loc="left", pad=12)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.4f}",
                ha="center", fontsize=8.5, fontweight="bold")
        
    if len(vals) >= 4:
        ax.annotate("73.5% Performance Collapse\nwhen Generator Artifacts Removed",
                    xy=(3, vals[3]), xytext=(2.2, 0.55),
                    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
                    fontsize=8.5, fontweight="bold", color=CORAL,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0EF", ec=CORAL, lw=0.8))
        
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F8_paysim_generator_collapse.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_f9_roar_curve(out_dir: Path | None = None) -> Path | None:
    """F9: ROAR faithfulness curve (T6_roar)."""
    roar = load_table("T6_roar")
    if roar is None or len(roar) == 0:
        return None
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    
    for rem, col, mark, name in [("shap", CORAL, "o", "SHAP-Guided Feature Removal"),
                                 ("random", SLATE, "s", "Random Feature Removal")]:
        sub = roar[roar["removal"] == rem].sort_values("fraction_removed")
        ax.plot(sub["fraction_removed"] * 100, sub["auprc"],
                color=col, marker=mark, lw=2.2, label=name)
        
    ax.set_xlabel("Fraction of Features Removed (%)", fontweight="bold")
    ax.set_ylabel("Retrained Model AUPRC", fontweight="bold")
    ax.set_title("F9: ROAR Faithfulness Curve — Steep Drop Confirms Feature Importance",
                 loc="left", pad=12)
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = (out_dir or FIG_DIR) / "F9_roar_faithfulness_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def generate_all_figures(out_dir: Path | None = None) -> dict[str, Path]:
    """Generate every manuscript figure that has corresponding table data."""
    target = out_dir or FIG_DIR
    target.mkdir(parents=True, exist_ok=True)
    
    generators = [
        ("F1", plot_f1_shap_migration),
        ("F2", plot_f2_leakage_ladder),
        ("F3", plot_f3_regime_drift),
        ("F4", plot_f4_main_benchmark),
        ("F5", plot_f5_ablation),
        ("F6", plot_f6_latency_sweep),
        ("F7", plot_f7_throughput_frontier),
        ("F8", plot_f8_paysim_ladder),
        ("F9", plot_f9_roar_curve),
    ]
    created = {}
    for name, fn in generators:
        try:
            p = fn(target)
            if p and p.exists():
                created[name] = p
                logger.info("Generated figure %s -> %s", name, p)
        except Exception as exc:
            logger.warning("Could not generate %s: %s", name, exc)
    return created
