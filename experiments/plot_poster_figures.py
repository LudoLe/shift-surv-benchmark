"""
Regenerate ESAC poster figures from adaptation_summary.csv.

Fig A: Forest plot — AUROC gap (src→tgt) by scenario × method (95% CI)
Fig B: Grouped bar chart — target AUROC by scenario × method (95% CI)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = Path(__file__).parent.parent
OUT_DIR = _ROOT / "outputs" / "figures"

# ── data ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(_ROOT / "outputs" / "tables" / "adaptation_summary.csv")

METHOD_MAP = {
    "invariant_oracle":  "Oracle",
    "all_observed":      "LR-ERM",
    "coral":             "CORAL",
    "all_observed_mlp":  "MLP-ERM",
    "dann":              "DANN",
    "cdan":              "CDAN",
}
SCENARIO_MAP = {
    "no_shift":               "No shift",
    "covariate_shift":        "Covariate",
    "spurious_shift":         "Spurious",
    "baseline_hazard_shift":  "Baseline hz.",
    "concept_shift":          "Concept",
}
SCENARIO_ORDER = list(SCENARIO_MAP.keys())
METHOD_ORDER   = list(METHOD_MAP.keys())

# colours per method
COLORS = {
    "invariant_oracle":  "#2d6a4f",   # dark green
    "all_observed":      "#1f77b4",   # blue
    "coral":             "#ff7f0e",   # orange
    "all_observed_mlp":  "#9467bd",   # purple
    "dann":              "#d62728",   # red
    "cdan":              "#17becf",   # cyan
}

df["method_label"]   = df["method"].map(METHOD_MAP)
df["scenario_label"] = df["scenario"].map(SCENARIO_MAP)


# ── Fig A: forest plot — AUROC gap ───────────────────────────────────────────

def plot_forest(metric_mean: str, metric_ci: str, xlabel: str,
                title: str, vline: float, fname: str,
                xlim: tuple | None = None, lower_is_better: bool = False):
    n_sc = len(SCENARIO_ORDER)
    n_me = len(METHOD_ORDER)
    row_height = 0.35
    group_gap  = 0.6
    fig_h = n_sc * (n_me * row_height + group_gap) + 0.8

    fig, ax = plt.subplots(figsize=(7.5, fig_h))

    yticks, ylabels = [], []
    y = 0.0

    for sc in SCENARIO_ORDER:
        sc_label = SCENARIO_MAP[sc]
        group_center = y + (n_me - 1) * row_height / 2
        ax.axhspan(y - row_height * 0.5,
                   y + (n_me - 1) * row_height + row_height * 0.5,
                   color="#f7f7f7" if SCENARIO_ORDER.index(sc) % 2 == 0 else "white",
                   zorder=0)

        for i, me in enumerate(METHOD_ORDER):
            row = df[(df.scenario == sc) & (df.method == me)]
            if row.empty:
                y += row_height
                continue
            val = float(row[metric_mean].iloc[0])
            ci  = float(row[metric_ci].iloc[0])
            color = COLORS[me]
            ax.errorbar(val, y, xerr=ci, fmt="o", color=color,
                        ecolor=color, elinewidth=1.4, capsize=3,
                        markersize=5, zorder=3)
            yticks.append(y)
            ylabels.append(METHOD_MAP[me])
            y += row_height

        # scenario label on the right
        ax.text(ax.get_xlim()[1] if xlim is None else xlim[1],
                group_center, sc_label,
                va="center", ha="left", fontsize=8,
                color="#333333", fontweight="bold")
        y += group_gap

    ax.axvline(vline, color="#555555", lw=0.8, ls="--", zorder=2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    if xlim:
        ax.set_xlim(xlim)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    # legend
    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_MAP[m])
               for m in METHOD_ORDER]
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              framealpha=0.8, ncol=2)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


plot_forest(
    metric_mean="gap_mean", metric_ci="gap_ci95",
    xlabel="AUROC gap  (source pooled − target;  ↓ better)",
    title="AUROC degradation from source to target",
    vline=0.0, fname="poster_forest_gap_updated.png",
    xlim=(-0.1, 0.42),
)

plot_forest(
    metric_mean="tgt_ece_mean", metric_ci="tgt_ece_ci95",
    xlabel="Target ECE  (↓ better calibration)",
    title="Target calibration error",
    vline=0.0, fname="poster_forest_ece_updated.png",
    xlim=(-0.02, 0.55), lower_is_better=True,
)



# ── Fig B: grouped bar chart — target AUROC ──────────────────────────────────

def plot_auroc_bars(fname: str = "poster_auroc_bars_updated.png"):
    n_sc = len(SCENARIO_ORDER)
    n_me = len(METHOD_ORDER)
    bar_w = 0.12
    group_w = n_me * bar_w + 0.15

    fig, ax = plt.subplots(figsize=(12, 4.5))

    x_centers = np.arange(n_sc) * group_w
    offsets = np.linspace(-(n_me - 1) / 2, (n_me - 1) / 2, n_me) * bar_w

    for j, me in enumerate(METHOD_ORDER):
        vals, cis = [], []
        for sc in SCENARIO_ORDER:
            row = df[(df.scenario == sc) & (df.method == me)]
            if row.empty:
                vals.append(np.nan); cis.append(0)
            else:
                vals.append(float(row["tgt_auroc_mean"].iloc[0]))
                cis.append(float(row["tgt_auroc_ci95"].iloc[0]))

        xs = x_centers + offsets[j]
        ax.bar(xs, vals, width=bar_w * 0.85,
               color=COLORS[me], label=METHOD_MAP[me],
               alpha=0.88, zorder=3)
        ax.errorbar(xs, vals, yerr=cis,
                    fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=2, zorder=4)

    ax.axhline(0.5, color="#888888", lw=0.7, ls=":", zorder=2)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([SCENARIO_MAP[s] for s in SCENARIO_ORDER],
                       fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Target AUROC  (↑ better)", fontsize=9)
    ax.set_title("Target AUROC by scenario and adaptation method",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0.45, 1.0)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=3)

    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


plot_auroc_bars()


# ── Fig C: combined forest (gap + ECE side by side) ──────────────────────────

def plot_combined_forest(fname: str = "poster_forest_all_methods_updated.png"):
    n_sc = len(SCENARIO_ORDER)
    n_me = len(METHOD_ORDER)
    row_h = 0.32
    grp_gap = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(13, n_sc * (n_me * row_h + grp_gap) + 1.0),
                             sharey=False)

    configs = [
        ("gap_mean",      "gap_ci95",      "AUROC gap (source − target;  ↓ better)", 0.0,  (-0.08, 0.40)),
        ("tgt_ece_mean",  "tgt_ece_ci95",  "Target ECE  (↓ better)",                0.0,  (-0.02, 0.55)),
    ]

    for ax, (mcol, ecol, xlabel, vline, xlim) in zip(axes, configs):
        yticks, ylabels = [], []
        y = 0.0

        for si, sc in enumerate(SCENARIO_ORDER):
            sc_label = SCENARIO_MAP[sc]
            group_center = y + (n_me - 1) * row_h / 2
            ax.axhspan(y - row_h * 0.5,
                       y + (n_me - 1) * row_h + row_h * 0.5,
                       color="#f0f0f0" if si % 2 == 0 else "white", zorder=0)

            for me in METHOD_ORDER:
                row = df[(df.scenario == sc) & (df.method == me)]
                if row.empty:
                    y += row_h
                    yticks.append(y - row_h)
                    ylabels.append("")
                    continue
                val = float(row[mcol].iloc[0])
                ci  = float(row[ecol].iloc[0])
                ax.errorbar(val, y, xerr=ci, fmt="o", color=COLORS[me],
                            ecolor=COLORS[me], elinewidth=1.3, capsize=3,
                            markersize=4.5, zorder=3)
                yticks.append(y)
                ylabels.append(METHOD_MAP[me])
                y += row_h

            ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.01,
                    group_center, sc_label,
                    va="center", ha="left", fontsize=7.5,
                    color="#222222", fontweight="bold")
            y += grp_gap

        ax.axvline(vline, color="#555555", lw=0.8, ls="--", zorder=2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_xlim(xlim)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("AUROC degradation (source → target)", fontsize=9, fontweight="bold")
    axes[1].set_title("Target calibration (ECE)", fontsize=9, fontweight="bold")

    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_MAP[m]) for m in METHOD_ORDER]
    fig.legend(handles=patches, fontsize=8, loc="lower center",
               framealpha=0.9, ncol=6, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Domain adaptation methods across shift scenarios  (mean ± 95% CI, 10 seeds)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 0.87, 1])
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


plot_combined_forest()
print("\nDone.")
