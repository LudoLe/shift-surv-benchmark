"""
SHIFT-SURV — IEEE conference publication figures.

Fig 1  (fig1_divergence) — two-panel horizontal bars, scenario-aligned.
        Panel A: ERM source-to-target AUROC gap.
        Panel B: ERM target ECE.
        Baseline-hazard shift highlighted as the ECE outlier.
        Sorted by AUROC gap (ascending, top→bottom) so the inverse
        relationship between panels is immediately visible.

Fig 2  (fig2_headroom) — dot-and-whisker per scenario.
        ERM / best deployable DA / Oracle, with 95% CI.
        Sorted by oracle-ERM gap (descending); concept_shift pinned last
        as the irrecoverable exception.

Input  : outputs/tables/benchmark_1s1t_iwcv_summary.csv
         (columns: scenario, method, tgt_auroc_mean/ci95, tgt_ece_mean/ci95,
          gap_mean/ci95)

Output : paper_assets/fig{1,2}_*.{pdf,png}

Run from repo root:
    python paper_assets/make_figures.py
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_summary.csv")
OUT     = os.path.join(ROOT, "paper_assets")

# ── scenario config ────────────────────────────────────────────────────────────
# canonical ordering: sorted by AUROC gap ascending → baseline-hazard at top,
# concept at bottom.  "observational_shift" is labelled "Measurement" per paper.
SCENARIOS = [
    ("baseline_hazard_shift", "Baseline hazard"),
    ("spurious_shift",        "Spurious"),
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("concept_shift",         "Concept"),
]
# scenarios for Fig 2 — sorted at runtime by oracle gap; concept pinned last
FIG2_KEYS = [k for k, _ in SCENARIOS]

DA_METHODS = ("coral", "dann", "cdan")   # deployable DA methods

# ── colour palette (restrained; greyscale-distinguishable) ─────────────────────
#   one accent for the ECE outlier (baseline-hazard), neutral for the rest
C_ACCENT  = "#b85450"   # coral — baseline-hazard (ECE anomaly)
C_NEUTRAL = "#8d9db6"   # muted steel — all other scenarios
C_ERM     = "#8d9db6"   # same muted steel — ERM dots
C_DA      = "#2c5f8a"   # dark blue — best DA dots
C_ORACLE  = "#1a1a1a"   # near-black — oracle dots
C_CONN    = "#d8d8d8"   # very light grey — headroom connector
C_GRID    = "#efefef"   # grid lines

# ── global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":             8,
    "axes.labelsize":        8,
    "axes.titlesize":        8,
    "xtick.labelsize":       7,
    "ytick.labelsize":       8,
    "legend.fontsize":       7,
    "axes.linewidth":        0.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.spines.left":      False,
    "axes.facecolor":        "white",
    "figure.facecolor":      "white",
    "axes.axisbelow":        True,
    "xtick.major.width":     0.5,
    "xtick.major.size":      3,
    "ytick.major.width":     0,
    "ytick.major.size":      0,
})


# ── data helpers ───────────────────────────────────────────────────────────────

def load_summary(path: str) -> dict:
    """Return dict keyed by (scenario, method) → row dict."""
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["scenario"], r["method"])] = r
    return d


def get(A: dict, sc: str, m: str, base: str):
    """
    Return (mean, ci_low, ci_high) for metric `base` (e.g. 'tgt_auroc').
    ci95 is half-width so ci_low = mean − ci95, ci_high = mean + ci95.
    """
    r = A.get((sc, m))
    if not r:
        return np.nan, np.nan, np.nan
    mn  = r.get(base + "_mean", "")
    ci  = r.get(base + "_ci95", "0")
    mn  = float(mn) if mn not in ("", "nan") else np.nan
    ci  = float(ci) if ci not in ("", "nan") else 0.0
    return mn, mn - ci, mn + ci


def best_da(A, sc):
    """Best deployable DA: argmax AUROC among coral/dann/cdan. Returns (mn,lo,hi)."""
    cands = [get(A, sc, m, "tgt_auroc") for m in DA_METHODS]
    cands = [(mn, lo, hi) for mn, lo, hi in cands if not np.isnan(mn)]
    return max(cands, key=lambda t: t[0]) if cands else (np.nan, np.nan, np.nan)


# ── Figure 1: two-panel scenario-aligned horizontal bars ───────────────────────

def make_fig1(A: dict) -> None:
    """
    Panel A — ERM AUROC gap (source − target): positive = discrimination loss.
    Panel B — ERM target ECE: absolute calibration error on target.

    Why two-panel horizontal bars (not a scatter):
    • Aligned y-axes let the viewer follow a single scenario row across both
      panels — the divergent behaviour of baseline-hazard (tiny A bar, long B
      bar) is a pre-attentive visual comparison, not a deductive inference.
    • Sorting by AUROC gap ascending places baseline-hazard at the top; in
      Panel B its bar is longest, creating an "inverted" reading order that
      directly illustrates the independence claim.
    """
    keys   = [k for k, _ in SCENARIOS]
    labels = [l for _, l in SCENARIOS]
    n      = len(keys)
    # y-positions: top row = index 0 → y = n−1 (highest value)
    yi     = np.arange(n - 1, -1, -1, dtype=float)

    gap    = [get(A, k, "all_observed", "gap")[0]     for k in keys]
    gap_lo = [get(A, k, "all_observed", "gap")[1]     for k in keys]
    gap_hi = [get(A, k, "all_observed", "gap")[2]     for k in keys]
    ece    = [get(A, k, "all_observed", "tgt_ece")[0] for k in keys]
    ece_lo = [get(A, k, "all_observed", "tgt_ece")[1] for k in keys]
    ece_hi = [get(A, k, "all_observed", "tgt_ece")[2] for k in keys]

    bar_colors = [C_ACCENT if k == "baseline_hazard_shift" else C_NEUTRAL
                  for k in keys]

    # asymmetric xerr: [[left_err], [right_err]]
    def asym_err(vals, los, his):
        return [[v - lo for v, lo in zip(vals, los)],
                [hi - v for v, hi in zip(vals, his)]]

    # ── layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.5, 2.1))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06,
                            left=0.19, right=0.985, top=0.95, bottom=0.18)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    BAR = 0.44
    EKW = dict(ecolor="#555", elinewidth=0.6, capsize=2.0, capthick=0.6)

    for ax, vals, los, his, xlab, panel_lbl in [
        (axA, gap, gap_lo, gap_hi, "Source$-$target AUROC gap", "A"),
        (axB, ece, ece_lo, ece_hi, "Target ECE",               "B"),
    ]:
        ax.barh(yi, vals,
                xerr=asym_err(vals, los, his),
                color=bar_colors, height=BAR,
                error_kw=EKW, linewidth=0, zorder=2)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
        ax.set_xlabel(xlab, labelpad=3)
        ax.set_yticks(yi)
        ax.tick_params(axis="y", left=False, pad=3)
        ax.set_ylim(-0.6, n - 0.4)
        # panel label — top-left, bold, small
        ax.text(0.02, 0.97, panel_lbl, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", color="#333")

    axA.set_yticklabels(labels)
    axB.set_yticklabels([])
    # reference line at AUROC gap = 0
    axA.axvline(0, color="#aaaaaa", lw=0.7, zorder=3)

    # single colour legend — baseline hazard vs rest
    leg_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_ACCENT,  ec="none",
                       label="Baseline hazard"),
        plt.Rectangle((0, 0), 1, 1, fc=C_NEUTRAL, ec="none",
                       label="Other shifts"),
    ]
    axB.legend(handles=leg_handles, loc="lower right", frameon=False,
               handlelength=0.9, handleheight=0.85, handletextpad=0.5,
               labelspacing=0.3, borderpad=0, fontsize=7)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_divergence.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")


# ── Figure 2: dot-and-whisker — ERM / best DA / Oracle ─────────────────────────

def make_fig2(A: dict) -> None:
    """
    One row per scenario.  Three markers: ERM (circle), best DA (square),
    Oracle (diamond), each with 95% CI.  A thin connector spans ERM→Oracle
    to make the recoverable headroom a physical length.

    Why dot-and-whisker (not grouped bars):
    • The headroom (ERM-to-Oracle) is a segment length — instantly readable.
    • Overlapping CI whiskers show ERM ≈ best DA without arithmetic.
    • CI on the oracle shows it is not trivially easy even with perfect features.
    • Concept-shift's three markers are nearly co-located (irrecoverable):
      visually distinct from the other rows without needing annotation.

    Sort: descending oracle-ERM gap (measurement at top — largest unrecovered
    headroom); concept pinned to the bottom row.
    """
    # build rows, sort
    rows = []
    for k, lbl in SCENARIOS:
        em, elo, ehi = get(A, k, "all_observed",     "tgt_auroc")
        om, olo, ohi = get(A, k, "invariant_oracle", "tgt_auroc")
        dm, dlo, dhi = best_da(A, k)
        rows.append(dict(key=k, lbl=lbl,
                         em=em, elo=elo, ehi=ehi,
                         dm=dm, dlo=dlo, dhi=dhi,
                         om=om, olo=olo, ohi=ohi,
                         headroom=om - em))

    # concept last, rest descending by headroom
    concept = [r for r in rows if r["key"] == "concept_shift"]
    others  = sorted([r for r in rows if r["key"] != "concept_shift"],
                     key=lambda r: -r["headroom"])
    rows    = others + concept
    n       = len(rows)
    yi      = np.arange(n - 1, -1, -1, dtype=float)   # top = largest headroom

    OFF = 0.17   # vertical jitter between the 3 markers per row
    MS  = 5.0
    EKW = dict(elinewidth=0.65, capsize=2.0, capthick=0.65)

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    fig.subplots_adjust(left=0.23, right=0.975, top=0.88, bottom=0.22)

    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)

    for row, yy in zip(rows, yi):
        em, om = row["em"], row["om"]
        # headroom connector (ERM → Oracle)
        ax.plot([em, om], [yy, yy], color=C_CONN, lw=1.8,
                solid_capstyle="round", zorder=1)

        def eb(xm, xlo, xhi, dy, fmt, col):
            ax.errorbar(xm, yy + dy,
                        xerr=[[xm - xlo], [xhi - xm]],
                        fmt=fmt, color=col, ms=MS, zorder=4,
                        markeredgewidth=0, **EKW)

        eb(em,       row["elo"], row["ehi"], +OFF, "o", C_ERM)
        eb(row["dm"],row["dlo"], row["dhi"],  0.0, "s", C_DA)
        eb(om,       row["olo"], row["ohi"], -OFF, "D", C_ORACLE)

    ax.set_yticks(yi)
    ax.set_yticklabels([r["lbl"] for r in rows])
    ax.tick_params(axis="y", left=False, pad=3)
    ax.set_xlabel("Target AUROC", labelpad=3)
    ax.set_xlim(0.50, 0.90)
    ax.set_title("Adaptation methods do not recover oracle performance",
                 fontsize=8, pad=5, loc="left")

    # concept annotation — subtle italic text above that row
    concept_y = yi[next(i for i, r in enumerate(rows)
                        if r["key"] == "concept_shift")]
    ax.text(0.51, concept_y + 0.40,
            "low oracle — irrecoverable by construction",
            fontsize=6, color="#aaa", style="italic", va="bottom")

    # legend — horizontal strip below x-axis
    legend_items = [
        Line2D([0], [0], marker="o", color=C_ERM,    ls="none",
               ms=5, markeredgewidth=0, label="ERM"),
        Line2D([0], [0], marker="s", color=C_DA,     ls="none",
               ms=5, markeredgewidth=0, label="Best DA"),
        Line2D([0], [0], marker="D", color=C_ORACLE, ls="none",
               ms=4, markeredgewidth=0, label="Oracle"),
        Line2D([0], [0], color=C_CONN, lw=2.2, label="Headroom"),
    ]
    ax.legend(handles=legend_items,
              loc="upper center", bbox_to_anchor=(0.5, -0.24),
              frameon=False, ncol=4,
              columnspacing=1.0, handlelength=1.2,
              handletextpad=0.4, borderpad=0)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_headroom.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_headroom.{pdf,png}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    A = load_summary(SUMMARY)
    make_fig1(A)
    make_fig2(A)
