"""
Generate fig1_divergence: scatter plot AUROC gap (x) vs ECE gap (y).
Each scenario = one dot, colored distinctly, with x/y error bars (95% CI).
Shows that discrimination and calibration degrade independently.
Run from repo root:  python paper_assets/_gen_fig1_dotwhisker.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_summary.csv")
OUT     = os.path.join(ROOT, "paper_assets")

SCENARIOS = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline hazard"),
    ("concept_shift",         "Concept"),
]

# per-scenario colors
COLORS = ["#E74C3C", "#E67E22", "#27AE60", "#2980B9", "#8E44AD"]

# manual label offsets (dx, dy) to avoid overlap with dots
LABEL_OFFSETS = {
    "Covariate":        (  6,  -12),
    "Measurement":      (  6,    5),
    "Spurious":         (  6,   -4),
    "Baseline hazard":  ( -8,   -14),
    "Concept":          (  6,    5),
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":      ["Latin Modern Roman", "Computer Modern Roman",
                        "Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
})


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["scenario"], r["method"])] = r
    return d


def get_val(A, sc, m, key):
    r = A.get((sc, m))
    if not r:
        return np.nan, 0.0
    v  = r.get(key + "_mean", "")
    ci = r.get(key + "_ci95",  "0")
    v  = float(v)  if v  not in ("", "nan") else np.nan
    ci = float(ci) if ci not in ("", "nan") else 0.0
    return v, ci


def make(A):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.12)

    for (sc, lbl), col in zip(SCENARIOS, COLORS):
        auroc_gap, auroc_ci = get_val(A, sc, "all_observed", "gap")
        ece_erm,   ece_ci   = get_val(A, sc, "all_observed", "tgt_ece")
        ece_orc,   _        = get_val(A, sc, "invariant_oracle", "tgt_ece")
        ece_gap = ece_erm - ece_orc   # positive = worse calibration than oracle

        ax.errorbar(auroc_gap, ece_gap,
                    xerr=auroc_ci, yerr=ece_ci,
                    fmt="o", color=col, ms=7, markeredgewidth=0,
                    elinewidth=0.9, capsize=0, zorder=3)

        dx, dy = LABEL_OFFSETS.get(lbl, (6, 4))
        ax.annotate(lbl, (auroc_gap, ece_gap),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=col, va="center")

    ax.axhline(0, color="#cccccc", lw=0.7, ls="--", zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.7, ls="--", zorder=1)

    ax.set_xlabel("AUROC gap (source − target)",  labelpad=3)
    ax.set_ylabel("ECE gap (ERM − Oracle)",       labelpad=3)

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.2f}" if x != 0 else "0"))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:+.2f}" if y != 0 else "0"))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_divergence.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")


if __name__ == "__main__":
    A = load(SUMMARY)
    make(A)
