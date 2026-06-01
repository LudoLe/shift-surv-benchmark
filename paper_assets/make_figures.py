"""
Publication-ready main figures for SHIFT-SURV.
Run from repo root: python paper_assets/make_figures.py
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_summary.csv")
OUT     = os.path.join(ROOT, "paper_assets")

# --- scenario order (no_shift excluded from main figures) ---
SCEN_ORDER = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline hazard"),
    ("concept_shift",         "Concept"),
]
HIGHLIGHT = "baseline_hazard_shift"

# --- global style ---
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.axisbelow":     True,
    "figure.dpi":         180,
    "savefig.dpi":        300,
})

# Palette — passes greyscale
SLATE   = "#546e7a"   # non-highlight bars / ERM dots
RED     = "#c0392b"   # baseline hazard highlight
BLUE    = "#1a5276"   # best DA dots
BLACK   = "#111111"   # oracle dots
LGREY   = "#e0e4e8"   # grid / connector


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["scenario"], r["method"])] = r
    return d


def get(A, sc, m, metric):
    r = A.get((sc, m))
    if not r:
        return np.nan, np.nan
    mn  = r.get(metric + "_mean", "")
    ci  = r.get(metric + "_ci95",  "")
    return (float(mn) if mn not in ("", "nan") else np.nan,
            float(ci)  if ci  not in ("", "nan") else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Discrimination vs Calibration divergence (two-panel horizontal bars)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(A):
    keys   = [k for k, _ in SCEN_ORDER]
    labels = [l for _, l in SCEN_ORDER]
    n      = len(keys)
    # render bottom-to-top so top label = first in list
    yi     = np.arange(n - 1, -1, -1)

    gap    = [get(A, k, "all_observed", "gap")[0]     for k in keys]
    gap_ci = [get(A, k, "all_observed", "gap")[1]     for k in keys]
    ece    = [get(A, k, "all_observed", "tgt_ece")[0] for k in keys]
    ece_ci = [get(A, k, "all_observed", "tgt_ece")[1] for k in keys]

    colors = [RED if k == HIGHLIGHT else SLATE for k in keys]

    fig = plt.figure(figsize=(6.8, 2.4))
    axA = fig.add_axes([0.18, 0.18, 0.35, 0.72])   # left panel
    axB = fig.add_axes([0.58, 0.18, 0.38, 0.72])   # right panel

    BAR_H = 0.52
    EKW   = dict(ecolor="#444444", elinewidth=0.8, capsize=2.5, capthick=0.8)

    for ax, vals, cis, xlabel, panel in [
        (axA, gap, gap_ci, "Source−target AUROC gap", "A"),
        (axB, ece, ece_ci, "Target ECE",              "B"),
    ]:
        ax.barh(yi, vals, xerr=cis, color=colors, height=BAR_H,
                error_kw=EKW, linewidth=0, zorder=2)
        ax.set_xlabel(xlabel, labelpad=4)
        ax.set_yticks(yi)
        ax.set_ylim(-0.6, n - 0.4)
        ax.tick_params(axis="y", left=False, pad=3)
        ax.grid(axis="x", color=LGREY, linewidth=0.5, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.04, 1.05, f"({panel})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")

    axA.set_yticklabels(labels, fontsize=8.5)
    axB.set_yticklabels([])
    axA.axvline(0, color="#888888", lw=0.7, zorder=3)

    handles = [
        mpatches.Patch(color=RED,   label="Baseline hazard"),
        mpatches.Patch(color=SLATE, label="Other shifts"),
    ]
    axB.legend(handles=handles, loc="lower right", frameon=False,
               handlelength=1.2, handleheight=0.9, borderpad=0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_divergence.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Oracle headroom vs practical methods (dot-and-whisker)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2(A):
    keys   = [k for k, _ in SCEN_ORDER]
    labels = [l for _, l in SCEN_ORDER]
    n      = len(keys)
    # same top-to-bottom order as Fig 1
    yi     = np.arange(n - 1, -1, -1)

    # ERM, best DA (max over CORAL/DANN/CDAN), Oracle
    erm_m, erm_ci, da_m, da_ci, orc_m, orc_ci = [], [], [], [], [], []
    for k in keys:
        em, ec = get(A, k, "all_observed",    "tgt_auroc")
        om, oc = get(A, k, "invariant_oracle","tgt_auroc")
        cands  = [get(A, k, m, "tgt_auroc") for m in ("coral","dann","cdan")]
        cands  = [(m, c) for m, c in cands if not np.isnan(m)]
        dm, dc = max(cands, key=lambda t: t[0]) if cands else (np.nan, 0.0)
        erm_m.append(em); erm_ci.append(ec)
        da_m.append(dm);  da_ci.append(dc)
        orc_m.append(om); orc_ci.append(oc)

    fig, ax = plt.subplots(figsize=(5.6, 2.6))

    ROW_H  = 0.75   # vertical space per scenario row
    OFF    = 0.20   # vertical offset between ERM / DA / Oracle within one row
    MS     = 6      # marker size
    EKW = dict(elinewidth=0.9, capsize=2.8, capthick=0.9)

    for i, (k, yy) in enumerate(zip(keys, yi)):
        em = erm_m[i]; om = orc_m[i]

        # grey connector ERM→Oracle (headroom)
        ax.plot([em, om], [yy, yy],
                color=LGREY, lw=2.0, solid_capstyle="round", zorder=1)

        # concept shift: shade the whole row lightly to mark irrecoverability
        if k == "concept_shift":
            ax.axhspan(yy - 0.45, yy + 0.45, color="#fff3cd", alpha=0.6, zorder=0)

        # ERM  — circle, slate
        ax.errorbar(erm_m[i], yy + OFF, xerr=erm_ci[i],
                    fmt="o", color=SLATE, ms=MS,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5, **EKW)
        # best DA — square, blue
        ax.errorbar(da_m[i],  yy,       xerr=da_ci[i],
                    fmt="s", color=BLUE,  ms=MS,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5, **EKW)
        # Oracle — diamond, black
        ax.errorbar(orc_m[i], yy - OFF, xerr=orc_ci[i],
                    fmt="D", color=BLACK, ms=MS-1,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5, **EKW)

    ax.set_yticks(yi)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", left=False)
    ax.set_xlabel("Target AUROC", labelpad=4)
    ax.set_xlim(0.50, 0.90)
    ax.grid(axis="x", color=LGREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # annotation for concept shift — yi maps keys bottom-to-top
    concept_yi = yi[keys.index("concept_shift")]
    ax.text(0.515, concept_yi - 0.38,
            "irrecoverable\nby construction",
            fontsize=6.5, color="#7d6608", style="italic", va="bottom")

    legend_items = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=SLATE,
               markeredgecolor="white", markersize=6, label="ERM"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=6, label="Best DA"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=BLACK,
               markeredgecolor="white", markersize=5, label="Oracle"),
        Line2D([0],[0], color=LGREY, lw=2.5, label="Recoverable headroom"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False,
              ncol=2, columnspacing=1.0, handlelength=1.4, borderpad=0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_headroom.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_headroom.{pdf,png}")


if __name__ == "__main__":
    A = load(SUMMARY)
    make_fig1(A)
    make_fig2(A)
