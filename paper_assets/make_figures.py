"""
Publication-ready main figures for SHIFT-SURV.
Run from repo root: python paper_assets/make_figures.py

Fig 1  — scatter: AUROC gap vs target ECE, colored by shift family.
Fig 2  — dot-and-whisker: ERM / best DA / Oracle, sorted by oracle−DA gap.
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_summary.csv")
OUT     = os.path.join(ROOT, "paper_assets")

# Scenarios shown in main figures (no_shift excluded; shape included in Fig1 only)
MAIN = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline hazard"),
    ("concept_shift",         "Concept"),
]
MAIN_AND_SHAPE = MAIN + [("shape_shift", "Shape")]

# Shift family → color  (3 families + shape in grey)
FAM_COLOR = {
    "covariate_shift":       "#4878cf",   # feature-dist shift  — blue
    "observational_shift":   "#4878cf",   # feature-dist shift  — same blue
    "spurious_shift":        "#6aafd2",   # shortcut shift      — lighter blue
    "baseline_hazard_shift": "#c44e52",   # hazard/calib shift  — dusty red
    "concept_shift":         "#1a1a1a",   # mechanism shift     — near-black
    "shape_shift":           "#aaaaaa",   # appendix            — grey
}
FAM_LABEL = {
    "covariate_shift":       "Feature-distribution shift",
    "observational_shift":   "Feature-distribution shift",
    "spurious_shift":        "Shortcut shift",
    "baseline_hazard_shift": "Hazard/calibration shift",
    "concept_shift":         "Mechanism shift",
    "shape_shift":           "Shape shift (appendix)",
}

# Annotation offsets (points) to avoid label collisions
OFFSETS = {
    "covariate_shift":       (7,  -11),
    "observational_shift":   (7,    5),
    "spurious_shift":        (7,    9),
    "baseline_hazard_shift": (-8,   5),
    "concept_shift":         (7,    5),
    "shape_shift":           (7,    4),
}

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":             8,
    "axes.labelsize":        8.5,
    "axes.titlesize":        8.5,
    "xtick.labelsize":       7.5,
    "ytick.labelsize":       7.5,
    "legend.fontsize":       7.5,
    "axes.linewidth":        0.5,
    "xtick.major.width":     0.5,
    "ytick.major.width":     0.5,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.facecolor":        "white",
    "figure.facecolor":      "white",
    "axes.axisbelow":        True,
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
# Figure 1 — scatter: AUROC gap vs ECE, shift families as colors
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(A):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13)

    # very subtle grid — horizontal only
    ax.yaxis.grid(True, color="#ebebeb", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # reference line: x = 0
    ax.axvline(0, color="#cccccc", lw=0.7, zorder=1)

    seen_families = set()
    for sc, lbl in MAIN_AND_SHAPE:
        xm, xc = get(A, sc, "all_observed", "gap")
        ym, yc = get(A, sc, "all_observed", "tgt_ece")
        if np.isnan(xm) or np.isnan(ym):
            continue

        col = FAM_COLOR[sc]
        fam = FAM_LABEL[sc]

        ax.errorbar(
            xm, ym, xerr=xc, yerr=yc,
            fmt="o", color=col, ms=6.5,
            elinewidth=0.7, capsize=2.2, capthick=0.7,
            markeredgewidth=0,
            label=fam if fam not in seen_families else "_nolegend_",
            zorder=3,
        )
        seen_families.add(fam)

        # annotation — small, grey, offset from point
        dx, dy = OFFSETS.get(sc, (6, 4))
        ax.annotate(
            lbl, xy=(xm, ym),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7, color="#444444",
            va="center",
        )

    ax.set_xlabel("Source−target AUROC gap", labelpad=4)
    ax.set_ylabel("Target ECE", labelpad=4)

    # legend — bottom right, no frame, ordered by shift family
    handles, lbs = ax.get_legend_handles_labels()
    # deduplicate order
    seen, h_out, l_out = set(), [], []
    for h, l in zip(handles, lbs):
        if l not in seen:
            seen.add(l); h_out.append(h); l_out.append(l)
    ax.legend(h_out, l_out, loc="lower right", frameon=False,
              markerscale=0.85, handletextpad=0.5, labelspacing=0.35,
              borderpad=0)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_divergence.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — dot-and-whisker, sorted by oracle − best DA (largest gap first)
# ─────────────────────────────────────────────────────────────────────────────
# palette
C_ERM  = "#9ab7d3"   # light steel — ERM
C_DA   = "#2c5f8a"   # dark blue   — best DA
C_ORC  = "#1a1a1a"   # near-black  — Oracle
C_CONN = "#d0d0d0"   # connector

def make_fig2(A):
    keys   = [k for k, _ in MAIN]
    labels_map = dict(MAIN)

    # collect values and sort by oracle − best_DA (descending)
    rows = []
    for k in keys:
        em, ec = get(A, k, "all_observed",     "tgt_auroc")
        om, oc = get(A, k, "invariant_oracle", "tgt_auroc")
        cands  = [get(A, k, m, "tgt_auroc") for m in ("coral", "dann", "cdan")]
        cands  = [(mv, cv) for mv, cv in cands if not np.isnan(mv)]
        dm, dc = max(cands, key=lambda t: t[0]) if cands else (np.nan, 0.0)
        rows.append(dict(key=k, lbl=labels_map[k],
                         em=em, ec=ec, dm=dm, dc=dc, om=om, oc=oc,
                         gap=om - dm))

    # sort: largest oracle−DA gap at top; concept pinned to bottom (irrecoverable)
    rows.sort(key=lambda r: -r["gap"] if r["key"] != "concept_shift" else 999)

    n   = len(rows)
    yi  = np.arange(n - 1, -1, -1, dtype=float)  # top → bottom
    OFF = 0.18

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.91, bottom=0.22)

    ax.xaxis.grid(True, color="#ebebeb", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    EKW = dict(elinewidth=0.7, capsize=2.0, capthick=0.7)

    for i, (row, yy) in enumerate(zip(rows, yi)):
        em, om = row["em"], row["om"]

        # headroom connector
        ax.plot([em, om], [yy, yy], color=C_CONN, lw=1.8,
                solid_capstyle="round", zorder=1)

        ax.errorbar(em,      yy + OFF, xerr=row["ec"],
                    fmt="o", color=C_ERM, ms=5.5, markeredgewidth=0,
                    zorder=4, **EKW)
        ax.errorbar(row["dm"], yy,     xerr=row["dc"],
                    fmt="s", color=C_DA,  ms=5.5, markeredgewidth=0,
                    zorder=4, **EKW)
        ax.errorbar(om,      yy - OFF, xerr=row["oc"],
                    fmt="D", color=C_ORC, ms=4.5, markeredgewidth=0,
                    zorder=4, **EKW)

    # concept note — subtle text above that row
    concept_i = next(i for i, r in enumerate(rows) if r["key"] == "concept_shift")
    concept_yy = yi[concept_i]
    ax.text(0.505, concept_yy + 0.44, "low oracle — irrecoverable",
            fontsize=6, color="#aaaaaa", va="bottom", style="italic")

    ax.set_yticks(yi)
    ax.set_yticklabels([r["lbl"] for r in rows])
    ax.tick_params(axis="y", left=False, pad=3)
    ax.set_xlabel("Target AUROC", labelpad=4)
    ax.set_xlim(0.50, 0.90)

    # title — concise
    ax.set_title("Adaptation methods do not recover oracle performance",
                 fontsize=8, pad=6, loc="left")

    # legend — horizontal strip below x-axis
    legend_el = [
        Line2D([0], [0], marker="o", color=C_ERM, ls="none",
               ms=5.5, markeredgewidth=0, label="ERM"),
        Line2D([0], [0], marker="s", color=C_DA,  ls="none",
               ms=5.5, markeredgewidth=0, label="Best DA"),
        Line2D([0], [0], marker="D", color=C_ORC, ls="none",
               ms=4.5, markeredgewidth=0, label="Oracle"),
        Line2D([0], [0], color=C_CONN, lw=2.0, label="Headroom"),
    ]
    ax.legend(handles=legend_el, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False,
              ncol=4, columnspacing=1.2, handlelength=1.2,
              handletextpad=0.4, borderpad=0)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_headroom.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_headroom.{pdf,png}")


if __name__ == "__main__":
    A = load(SUMMARY)
    make_fig1(A)
    make_fig2(A)
