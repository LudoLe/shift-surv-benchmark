"""
Publication-ready figures for SHIFT-SURV — IEEE conference style.
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

SCEN_ORDER = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline hazard"),
    ("concept_shift",         "Concept"),
]
HIGHLIGHT = "baseline_hazard_shift"

# ── palette (greyscale-safe) ────────────────────────────────────────────────
C_STD  = "#6d8fa8"   # standard bar / ERM dot  (muted steel)
C_HL   = "#b85450"   # baseline-hazard accent   (dusty red)
C_DA   = "#2c5f8a"   # best DA                  (dark blue)
C_ORC  = "#1a1a1a"   # oracle                   (near-black)
C_GRID = "#e8eaed"   # grid lines

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    7.5,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0,
    "xtick.major.size":   2.5,
    "ytick.major.size":   0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.axisbelow":     True,
})


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
    mn = r.get(metric + "_mean", "")
    ci = r.get(metric + "_ci95",  "")
    return (float(mn) if mn not in ("", "nan") else np.nan,
            float(ci) if ci not in ("", "nan") else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1  — two-panel horizontal bars (discrimination vs calibration)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(A):
    keys   = [k for k, _ in SCEN_ORDER]
    labels = [l for _, l in SCEN_ORDER]
    n      = len(keys)
    yi     = np.arange(n - 1, -1, -1, dtype=float)   # top → bottom

    gap    = [get(A, k, "all_observed", "gap")[0]     for k in keys]
    gap_ci = [get(A, k, "all_observed", "gap")[1]     for k in keys]
    ece    = [get(A, k, "all_observed", "tgt_ece")[0] for k in keys]
    ece_ci = [get(A, k, "all_observed", "tgt_ece")[1] for k in keys]
    colors = [C_HL if k == HIGHLIGHT else C_STD for k in keys]

    fig = plt.figure(figsize=(6.5, 2.2))
    # manually placed axes so y-labels don't get clipped
    axA = fig.add_axes([0.175, 0.17, 0.345, 0.75])
    axB = fig.add_axes([0.590, 0.17, 0.385, 0.75])

    BAR  = 0.32
    EKW  = dict(ecolor="#555", elinewidth=0.65, capsize=2.0, capthick=0.65)

    for ax, vals, cis, xlab in [
        (axA, gap, gap_ci, "Source−target AUROC gap"),
        (axB, ece, ece_ci, "Target ECE"),
    ]:
        ax.barh(yi, vals, xerr=cis, color=colors, height=BAR,
                error_kw=EKW, linewidth=0, zorder=2)
        ax.set_xlabel(xlab, labelpad=3)
        ax.set_yticks(yi)
        ax.tick_params(axis="y", left=False, pad=4)
        # very subtle solid grid
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.55, n - 0.45)

    axA.set_yticklabels(labels, linespacing=0.9)
    axB.set_yticklabels([])
    axA.axvline(0, color="#aaa", lw=0.65, zorder=3)

    # panel labels — inside plot, top-left, small caps style
    for ax, lbl in [(axA, "A"), (axB, "B")]:
        ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold",
                va="top", ha="left", color="#333")

    # legend — minimal, two patches
    handles = [mpatches.Patch(facecolor=C_HL,  edgecolor="none", label="Baseline hazard"),
               mpatches.Patch(facecolor=C_STD, edgecolor="none", label="Other shifts")]
    axB.legend(handles=handles, loc="lower right", frameon=False,
               handlelength=0.9, handleheight=0.85, borderpad=0,
               labelspacing=0.3, handletextpad=0.5)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_divergence.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  — dot-and-whisker: oracle headroom vs practical methods
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2(A):
    keys   = [k for k, _ in SCEN_ORDER]
    labels = [l for _, l in SCEN_ORDER]
    n      = len(keys)
    yi     = np.arange(n - 1, -1, -1, dtype=float)

    erm_m, erm_ci, da_m, da_ci, orc_m, orc_ci = [], [], [], [], [], []
    for k in keys:
        em, ec = get(A, k, "all_observed",     "tgt_auroc")
        om, oc = get(A, k, "invariant_oracle", "tgt_auroc")
        cands  = [get(A, k, m, "tgt_auroc") for m in ("coral", "dann", "cdan")]
        cands  = [(mv, cv) for mv, cv in cands if not np.isnan(mv)]
        dm, dc = max(cands, key=lambda t: t[0]) if cands else (np.nan, 0.0)
        erm_m.append(em); erm_ci.append(ec)
        da_m.append(dm);  da_ci.append(dc)
        orc_m.append(om); orc_ci.append(oc)

    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.14)

    OFF = 0.18   # vertical jitter between the three markers
    MS  = 5.5
    EKW = dict(elinewidth=0.75, capsize=2.2, capthick=0.75)

    for i, (k, yy) in enumerate(zip(keys, yi)):
        em, om = erm_m[i], orc_m[i]

        # headroom connector
        ax.plot([em, om], [yy, yy], color="#ccc", lw=1.6,
                solid_capstyle="round", zorder=1)

        # three markers per row
        ax.errorbar(em,      yy + OFF, xerr=erm_ci[i],
                    fmt="o", color=C_STD, ms=MS, zorder=5,
                    markeredgewidth=0, **EKW)
        ax.errorbar(da_m[i], yy,       xerr=da_ci[i],
                    fmt="s", color=C_DA,  ms=MS, zorder=5,
                    markeredgewidth=0, **EKW)
        ax.errorbar(om,      yy - OFF, xerr=orc_ci[i],
                    fmt="D", color=C_ORC, ms=MS - 1, zorder=5,
                    markeredgewidth=0, **EKW)

    # concept-shift note — placed above the Concept row
    concept_yi = yi[keys.index("concept_shift")]
    ax.text(0.505, concept_yi + 0.42,
            "oracle low: irrecoverable by construction",
            fontsize=6.5, color="#999", va="bottom", ha="left", style="italic")

    ax.set_yticks(yi)
    ax.set_yticklabels(labels, linespacing=0.9)
    ax.tick_params(axis="y", left=False, pad=4)
    ax.set_xlabel("Target AUROC", labelpad=3)
    ax.set_xlim(0.50, 0.90)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # clean legend at bottom right
    legend_el = [
        Line2D([0], [0], marker="o", color=C_STD, ls="none",
               ms=5.5, markeredgewidth=0, label="ERM"),
        Line2D([0], [0], marker="s", color=C_DA,  ls="none",
               ms=5.5, markeredgewidth=0, label="Best DA"),
        Line2D([0], [0], marker="D", color=C_ORC, ls="none",
               ms=4.5, markeredgewidth=0, label="Oracle"),
        Line2D([0], [0], color="#ccc", lw=1.8, label="Headroom"),
    ]
    ax.legend(handles=legend_el, loc="lower right", frameon=False,
              ncol=2, columnspacing=1.0, handlelength=1.2,
              labelspacing=0.3, handletextpad=0.4, borderpad=0)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_headroom.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_headroom.{pdf,png}")


if __name__ == "__main__":
    A = load(SUMMARY)
    make_fig1(A)
    make_fig2(A)
