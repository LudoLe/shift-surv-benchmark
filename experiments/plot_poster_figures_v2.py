"""
Poster figures v2 — matching reference style.

Fig A: Forest plot (AUROC gap | ECE gap) — scenarios as horizontal bands,
       methods as distinct markers, style matches poster_forest_all_methods.png.
Fig B: Grouped bar chart (target AUROC) with seed-42 overlay points.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

_ROOT   = Path(__file__).parent.parent
OUT_DIR = _ROOT / "outputs" / "figures"

# ── load data ─────────────────────────────────────────────────────────────────

raw = pd.read_csv(_ROOT / "outputs" / "tables" / "adaptation_raw.csv")

KEEP_SCENARIOS = [
    "no_shift",
    "covariate_shift",
    "spurious_shift",
    "baseline_hazard_shift",
    "concept_shift",
]
raw = raw[raw["scenario"].isin(KEEP_SCENARIOS)].copy()
raw["ece_gap"] = raw["tgt_ece"] - raw["src_ece"]

# derive summary (mean ± 95% CI) from raw for all methods
def ci95(x):
    n = len(x)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=n - 1) * x.std(ddof=1) / np.sqrt(n))

summary = (
    raw.groupby(["scenario", "method"])
    .agg(
        n=("tgt_auroc", "count"),
        gap_mean=("gap", "mean"),
        gap_ci=("gap", ci95),
        ece_gap_mean=("ece_gap", "mean"),
        ece_gap_ci=("ece_gap", ci95),
        tgt_auroc_mean=("tgt_auroc", "mean"),
        tgt_auroc_ci=("tgt_auroc", ci95),
    )
    .reset_index()
)

seed42 = raw[raw["seed"] == 42].copy()

# ── style maps ────────────────────────────────────────────────────────────────

METHOD_ORDER = [
    "invariant_oracle",
    "all_observed",
    "coral",
    "irm",
    "all_observed_mlp",
    "dann",
    "cdan",
]
METHOD_LABEL = {
    "invariant_oracle": "Oracle (inv. only)",
    "all_observed":     "LR-ERM",
    "coral":            "CORAL",
    "irm":              "IRM",
    "all_observed_mlp": "MLP-ERM",
    "dann":             "DANN",
    "cdan":             "CDAN",
}
METHOD_MARKER = {
    "invariant_oracle": "D",   # diamond
    "all_observed":     "o",   # circle
    "coral":            "s",   # square
    "irm":              "P",   # fat plus
    "all_observed_mlp": "v",   # inverted triangle
    "dann":             "^",   # triangle up
    "cdan":             "X",   # x
}
METHOD_COLOR = {
    "invariant_oracle": "#2ca02c",   # green
    "all_observed":     "#1f77b4",   # blue
    "coral":            "#ff7f0e",   # orange
    "irm":              "#17becf",   # teal
    "all_observed_mlp": "#9467bd",   # purple
    "dann":             "#d62728",   # red
    "cdan":             "#e377c2",   # pink
}

SCENARIO_ORDER_FOREST = [
    "concept_shift",
    "baseline_hazard_shift",
    "spurious_shift",
    "covariate_shift",
    "no_shift",
]
SCENARIO_LABEL = {
    "no_shift":              "No Shift\n(control)",
    "covariate_shift":       "Covariate\nShift",
    "spurious_shift":        "Spurious\nShift",
    "baseline_hazard_shift": "Hazard\nShift",
    "concept_shift":         "Concept\nShift",
}
SCENARIO_ANNOT = {
    "no_shift":              "Control regime",
    "covariate_shift":       "Feature-space drift,\nstable risk",
    "spurious_shift":        "Non-transportable\nshortcuts",
    "baseline_hazard_shift": "Incidence shift,\nranking preserved",
    "concept_shift":         "Outcome mechanism\nchanges",
}
SCENARIO_BG = {
    "concept_shift":         "#fff3e0",   # peach
    "baseline_hazard_shift": "#e8f5e9",   # light green
    "spurious_shift":        "#ffffff",   # white
    "covariate_shift":       "#e3f2fd",   # light blue
    "no_shift":              "#ffffff",   # white
}

SCENARIO_ORDER_BAR = [
    "no_shift",
    "covariate_shift",
    "spurious_shift",
    "baseline_hazard_shift",
    "concept_shift",
]
SCENARIO_LABEL_BAR = {
    "no_shift":              "No shift",
    "covariate_shift":       "Covariate",
    "spurious_shift":        "Spurious",
    "baseline_hazard_shift": "Baseline hz.",
    "concept_shift":         "Concept",
}


# ══════════════════════════════════════════════════════════════════════════════
# Fig A — forest plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_forest(fname="poster_forest_v2.png"):
    n_sc  = len(SCENARIO_ORDER_FOREST)
    n_me  = len(METHOD_ORDER)
    step  = 0.22   # y-distance between adjacent methods inside one band
    gap   = 2.2    # y-distance between scenario centres  (must be > n_me*step)

    # ── y positions (bottom = no_shift, top = concept_shift) ─────────────────
    sc_centers: dict[str, float] = {}
    y_pos: dict[tuple, float]    = {}
    y = 0.0
    for sc in reversed(SCENARIO_ORDER_FOREST):       # no_shift first → y=0
        offsets = np.linspace(-(n_me - 1) / 2, (n_me - 1) / 2, n_me) * step
        sc_centers[sc] = y
        for me, off in zip(METHOD_ORDER, offsets):
            y_pos[(sc, me)] = y + off
        y += gap

    # ── band boundaries: exact midpoints between adjacent centres ─────────────
    ordered_centers = sorted(sc_centers.values())    # ascending (bottom→top)
    sc_by_center    = {v: k for k, v in sc_centers.items()}

    band_bounds: dict[str, tuple[float, float]] = {}
    for i, cy in enumerate(ordered_centers):
        lo = (ordered_centers[i - 1] + cy) / 2 if i > 0           else cy - gap / 2
        hi = (cy + ordered_centers[i + 1]) / 2 if i < n_sc - 1    else cy + gap / 2
        band_bounds[sc_by_center[cy]] = (lo, hi)

    ymin = ordered_centers[0]  - gap / 2
    ymax = ordered_centers[-1] + gap / 2

    panels = [
        ("gap_mean",     "gap_ci",
         (-0.12, 0.42),
         "AUROC Gap  (source − target)    ↑ = degradation on target"),
        ("ece_gap_mean", "ece_gap_ci",
         (-0.12, 0.42),
         "ECE Gap  (target − source)    ↑ = calibration worse on target"),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8.5), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    for ax, (mcol, ecol, xlim, xlabel) in zip(axes, panels):
        ax.set_facecolor("white")

        # scenario colour bands (guaranteed non-overlapping)
        for sc in SCENARIO_ORDER_FOREST:
            lo, hi = band_bounds[sc]
            ax.axhspan(lo, hi, color=SCENARIO_BG[sc], zorder=0, lw=0)

        ax.axvline(0, color="#555555", lw=1.0, ls="--", zorder=1)

        for (sc, me), yp in y_pos.items():
            row = summary[(summary.scenario == sc) & (summary.method == me)]
            if row.empty:
                continue
            val = float(row[mcol].iloc[0])
            ci  = float(row[ecol].iloc[0])
            ax.errorbar(
                val, yp, xerr=ci,
                fmt=METHOD_MARKER[me],
                color=METHOD_COLOR[me],
                ecolor=METHOD_COLOR[me],
                elinewidth=1.5, capsize=3.5,
                markersize=7, zorder=4,
                markeredgewidth=0.5,
                markeredgecolor="white",
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.xaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)

    # scenario labels on left y-axis (tick at band centre)
    axes[0].set_yticks([sc_centers[sc] for sc in SCENARIO_ORDER_FOREST])
    axes[0].set_yticklabels(
        [SCENARIO_LABEL[sc] for sc in SCENARIO_ORDER_FOREST],
        fontsize=10.5, fontweight="bold", ha="right",
    )
    axes[0].tick_params(axis="y", pad=6)

    # right-side italic annotations
    for sc in SCENARIO_ORDER_FOREST:
        axes[1].text(
            xlim[1] + 0.015, sc_centers[sc],
            SCENARIO_ANNOT[sc],
            va="center", ha="left",
            fontsize=7.5, fontstyle="italic", color="#444444",
        )

    # legend
    handles = [
        mlines.Line2D([], [], marker=METHOD_MARKER[m], color=METHOD_COLOR[m],
                      markeredgecolor="white", markeredgewidth=0.5,
                      markersize=8, linestyle="None", label=METHOD_LABEL[m])
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(METHOD_ORDER),
        fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.03),
        handletextpad=0.4, columnspacing=1.0,
    )

    fig.suptitle(
        "SHIFT-SURV Benchmark — Adaptation Methods Across Shift Scenarios",
        fontsize=13, fontweight="bold", x=0.5, y=1.02,
    )
    fig.text(0.5, 0.995, r"$n=10$ seeds,  95% CI",
             ha="center", va="top", fontsize=9, color="#555555")

    fig.tight_layout(rect=[0, 0.04, 0.87, 1.0])
    out = OUT_DIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


plot_forest()


# ══════════════════════════════════════════════════════════════════════════════
# Fig B — grouped bar chart + seed-42 overlay
# ══════════════════════════════════════════════════════════════════════════════

# For bars: Oracle, LR-ERM, CORAL, MLP-ERM, DANN, CDAN (no IRM, keeps plot clean)
BAR_METHODS = ["invariant_oracle", "all_observed", "coral",
               "all_observed_mlp", "dann", "cdan"]

def plot_auroc_bars_with_seed(fname="poster_auroc_bars_v2.png"):
    n_sc  = len(SCENARIO_ORDER_BAR)
    n_me  = len(BAR_METHODS)
    bar_w = 0.11
    grp_w = n_me * bar_w + 0.18

    fig, ax = plt.subplots(figsize=(12, 4.8))

    x_centers = np.arange(n_sc) * grp_w
    offsets   = np.linspace(-(n_me - 1) / 2, (n_me - 1) / 2, n_me) * bar_w

    for j, me in enumerate(BAR_METHODS):
        vals, cis = [], []
        for sc in SCENARIO_ORDER_BAR:
            row = summary[(summary.scenario == sc) & (summary.method == me)]
            if row.empty:
                vals.append(np.nan); cis.append(0)
            else:
                vals.append(float(row["tgt_auroc_mean"].iloc[0]))
                cis.append(float(row["tgt_auroc_ci"].iloc[0]))

        xs = x_centers + offsets[j]
        ax.bar(xs, vals, width=bar_w * 0.87,
               color=METHOD_COLOR[me], alpha=0.85,
               label=METHOD_LABEL[me], zorder=3)
        ax.errorbar(xs, vals, yerr=cis,
                    fmt="none", ecolor="#222222",
                    elinewidth=0.9, capsize=2.5, zorder=4)

    ax.axhline(0.5, color="#888888", lw=0.7, ls=":", zorder=2)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [SCENARIO_LABEL_BAR[s] for s in SCENARIO_ORDER_BAR],
        fontsize=9, rotation=18, ha="right",
    )
    ax.set_ylabel("Target AUROC  (↑ better)", fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=3)

    fig.suptitle(
        "SHIFT-SURV Benchmark — Target AUROC by Scenario & Method",
        fontsize=11, fontweight="bold",
    )
    ax.set_title(
        r"mean ± 95% CI  ($n=10$ seeds)",
        fontsize=8.5, color="#555555", pad=4,
    )

    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


plot_auroc_bars_with_seed()
print("\nDone.")
