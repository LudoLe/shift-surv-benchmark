"""
Generate fig1_divergence (dot-and-whisker, Nature style).
Run from repo root:  python paper_assets/_gen_fig1_dotwhisker.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_summary.csv")
OUT     = os.path.join(ROOT, "paper_assets")

# ── scenario config (y-axis order: top → bottom) ──────────────────────────────
SCENARIOS = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline hazard"),
    ("concept_shift",         "Concept"),
]

# alternating band tints (index 0 = topmost row)
BAND_TINTS = ["#eef2f7", "#ffffff", "#eef2f7", "#ffffff", "#eef2f7"]

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":        ["Latin Modern Roman", "Computer Modern Roman",
                          "Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0,
    "xtick.major.size":   3,
    "ytick.major.size":   0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
})

# ── data helpers ───────────────────────────────────────────────────────────────
def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["scenario"], r["method"])] = r
    return d

def get(A, sc, m, base):
    r = A.get((sc, m))
    if not r:
        return np.nan, np.nan, np.nan
    mn = r.get(base + "_mean", "")
    ci = r.get(base + "_ci95",  "0")
    mn = float(mn) if mn not in ("", "nan") else np.nan
    ci = float(ci) if ci not in ("", "nan") else 0.0
    return mn, mn - ci, mn + ci


# ── figure ─────────────────────────────────────────────────────────────────────
def make(A):
    n      = len(SCENARIOS)
    keys   = [k for k, _ in SCENARIOS]
    labels = [l for _, l in SCENARIOS]
    # y positions: top-most scenario = y = n-1, bottom = y = 0
    yi     = np.arange(n - 1, -1, -1, dtype=float)

    # collect data
    gap_m,  gap_lo,  gap_hi  = zip(*[get(A, k, "all_observed", "gap")     for k in keys])
    ece_m,  ece_lo,  ece_hi  = zip(*[get(A, k, "all_observed", "tgt_ece") for k in keys])

    # ── layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5.8, 2.05))
    # manual placement: left panel narrower (gap can be negative), right wider
    axA = fig.add_axes([0.195, 0.13, 0.335, 0.82])
    axB = fig.add_axes([0.565, 0.13, 0.415, 0.82])

    # ── shared horizontal bands ─────────────────────────────────────────────
    half = 0.5
    for ax in (axA, axB):
        for i, (yy, tint) in enumerate(zip(yi, BAND_TINTS)):
            ax.axhspan(yy - half, yy + half, color=tint,
                       linewidth=0, zorder=0)

    # ── dots and whiskers ──────────────────────────────────────────────────
    DOT_COLOR = "#1a1a2e"
    DOT_MS    = 4.8
    EKW       = dict(elinewidth=0.9, capsize=0)   # Nature: no caps

    def draw_dots(ax, means, los, his):
        for yy, mn, lo, hi in zip(yi, means, los, his):
            ax.errorbar(mn, yy,
                        xerr=[[mn - lo], [hi - mn]],
                        fmt="o", color=DOT_COLOR,
                        ms=DOT_MS, markeredgewidth=0,
                        zorder=4, **EKW)

    draw_dots(axA, gap_m, gap_lo, gap_hi)
    draw_dots(axB, ece_m, ece_lo, ece_hi)

    # ── reference lines ────────────────────────────────────────────────────
    axA.axvline(0, color="#aaaaaa", lw=0.7, ls="--", zorder=1)

    # ── axes formatting ────────────────────────────────────────────────────
    for ax in (axA, axB):
        ax.set_yticks(yi)
        ax.set_ylim(-0.55, n - 0.45)
        ax.tick_params(axis="y", left=False, pad=4)
        ax.xaxis.grid(False)
        ax.set_axisbelow(False)

    axA.set_yticklabels(labels)
    axB.set_yticklabels([])

    axA.set_xlabel("AUROC gap (source−target)", labelpad=3)
    axB.set_xlabel("Target ECE",                    labelpad=3)

    # panel labels — top-left of each axes, small, bold
    for ax, lbl in [(axA, "a"), (axB, "b")]:
        ax.text(-0.02, 1.04, f"({lbl})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="right",
                color="#222")

    # ── x-axis tick formatting ─────────────────────────────────────────────
    # AUROC gap: symmetric around 0
    axA.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.2f}" if x != 0 else "0"))
    axA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))

    axB.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    axB.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # ── save ───────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_divergence.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")


if __name__ == "__main__":
    A = load(SUMMARY)
    make(A)
