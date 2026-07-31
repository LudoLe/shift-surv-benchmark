"""
Generate fig3_ece_headroom: ECE analogue of fig2.
Oracle = gray background bar (the calibration floor to reach down to).
ERM and Best ECE-DA = vivid narrow bars in front (higher = worse calibration).
The gap between colored bars and oracle shows unrecovered calibration headroom.
Significance stars: paired t-test (ERM vs Best ECE-DA) per scenario.
Run from repo root:  python paper_assets/_gen_fig3_ece_bars.py
"""
import csv, os
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_summary.csv")
RAW     = os.path.join(ROOT, "outputs", "tables", "benchmark_1s1t_iwcv_raw.csv")
OUT     = os.path.join(ROOT, "paper_assets")

SCENARIOS = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline\nhazard"),
    ("concept_shift",         "Concept"),
]

# all methods (including calibration) for best ECE-DA
ECE_METHODS = ("coral", "prior_recalib", "platt_scaling", "dann", "cdan")

C_ERM    = "#2E86C1"   # blue (same as fig2 ERM)
C_DA     = "#E74C3C"   # red  (same as fig2 Best DA)
C_ORACLE = "#D5D8DC"   # light gray (same as fig2 Oracle)
C_ORACLE_EDGE = "#AAB7B8"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":        ["Latin Modern Roman", "Computer Modern Roman",
                          "Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   0,
    "ytick.major.size":   3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
})


def load_summary(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["scenario"], r["method"])] = r
    return d


def load_raw(path):
    """Returns {(scenario, method): [tgt_ece per seed]}."""
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["scenario"], r["method"])
            d.setdefault(key, [])
            try:
                d[key].append(float(r["tgt_ece"]))
            except (ValueError, KeyError):
                pass
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


def best_ece_da(A, sc):
    cands = [(get(A, sc, m, "tgt_ece"), m) for m in ECE_METHODS]
    cands = [(t, m) for t, m in cands if not np.isnan(t[0])]
    return min(cands, key=lambda x: x[0][0])[0] if cands else (np.nan, np.nan, np.nan)


def best_ece_da_method(A, sc):
    cands = [(get(A, sc, m, "tgt_ece")[0], m) for m in ECE_METHODS]
    cands = [(v, m) for v, m in cands if not np.isnan(v)]
    return min(cands)[1] if cands else None


def pvalue_stars(p):
    if p <= 0.001: return "***"
    if p <= 0.01:  return "**"
    if p <= 0.05:  return "*"
    return "ns"


def make(A, R):
    keys   = [k for k, _ in SCENARIOS]
    labels = [l for _, l in SCENARIOS]
    n      = len(keys)
    x      = np.arange(n, dtype=float)

    W_bg = 0.54
    W_fg = 0.20
    sep  = 0.025
    rng  = np.random.default_rng(42)

    Y_MIN = 0.0
    Y_MAX = 0.33

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.14)

    for i, (sc, _) in enumerate(SCENARIOS):
        erm_m, erm_lo, erm_hi = get(A, sc, "all_observed",     "tgt_ece")
        da_m,  da_lo,  da_hi  = best_ece_da(A, sc)
        orc_m, orc_lo, orc_hi = get(A, sc, "invariant_oracle", "tgt_ece")
        da_key = best_ece_da_method(A, sc)

        xi    = x[i]
        x_erm = xi - (W_fg / 2 + sep / 2)
        x_da  = xi + (W_fg / 2 + sep / 2)

        # ── Oracle background bar (gray, the floor to reach) ─────────────
        # Oracle ECE is low (good); gray bar is short — gap above it = headroom
        ax.bar(xi, orc_m, W_bg,
               color=C_ORACLE, edgecolor=C_ORACLE_EDGE,
               linewidth=0.4, zorder=1)

        # ── ERM bar (tall = bad ECE) ──────────────────────────────────────
        ax.bar(x_erm, erm_m, W_fg,
               color=C_ERM, linewidth=0, zorder=3)
        ax.errorbar(x_erm, erm_m,
                    yerr=[[erm_m - erm_lo], [erm_hi - erm_m]],
                    fmt="none", color="black", elinewidth=1.1,
                    capsize=2.8, capthick=1.0, zorder=5)

        # ── Best ECE-DA bar ───────────────────────────────────────────────
        ax.bar(x_da, da_m, W_fg,
               color=C_DA, linewidth=0, zorder=3)
        ax.errorbar(x_da, da_m,
                    yerr=[[da_m - da_lo], [da_hi - da_m]],
                    fmt="none", color="black", elinewidth=1.1,
                    capsize=2.8, capthick=1.0, zorder=5)

        # ── jittered seed dots ─────────────────────────────────────────────
        for jx, key, col in [
            (x_erm, (sc, "all_observed"), C_ERM),
            (x_da,  (sc, da_key),         C_DA),
        ]:
            seeds = R.get(key, [])
            if seeds:
                jitter = rng.uniform(-W_fg * 0.4, W_fg * 0.4, len(seeds))
                ax.scatter(jx + jitter, seeds,
                           s=7, color="white", alpha=0.65,
                           linewidths=0.3, edgecolors=col, zorder=4)

        # ── significance bracket (ERM vs Best ECE-DA) ─────────────────────
        erm_seeds = R.get((sc, "all_observed"), [])
        da_seeds  = R.get((sc, da_key), []) if da_key else []
        if len(erm_seeds) >= 2 and len(da_seeds) >= 2:
            _, p = scipy_stats.ttest_rel(erm_seeds, da_seeds)
            # Bonferroni: best calibration-DA is selected post-hoc among
            # len(ECE_METHODS) candidates, so correct for that multiplicity.
            p = min(1.0, p * len(ECE_METHODS))
            stars = pvalue_stars(p)
            top = max(erm_hi, da_hi) + 0.008
            ax.plot([x_erm, x_erm, x_da, x_da],
                    [top, top + 0.006, top + 0.006, top],
                    color="#555555", linewidth=0.7, zorder=6)
            ax.text((x_erm + x_da) / 2, top + 0.006, stars,
                    ha="center", va="bottom", fontsize=7.5,
                    color="#333333", zorder=6)

    # ── axes ──────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=0.9)
    ax.set_ylabel("Target ECE  (↓ better)", labelpad=4)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── legend ────────────────────────────────────────────────────────────
    leg = [
        plt.Rectangle((0, 0), 1, 1, fc=C_ERM,    ec="none",           label="ERM"),
        plt.Rectangle((0, 0), 1, 1, fc=C_DA,     ec="none",           label="Best ECE-DA"),
        plt.Rectangle((0, 0), 1, 1, fc=C_ORACLE, ec=C_ORACLE_EDGE,
                       linewidth=0.5,                                   label="Oracle (floor)"),
    ]
    ax.legend(handles=leg, loc="upper right", frameon=False,
              ncol=1, fontsize=7.5, handlelength=1.0, handleheight=0.85,
              columnspacing=0.7, handletextpad=0.45, borderpad=0)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_ece_headroom.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_ece_headroom.{pdf,png}")


if __name__ == "__main__":
    A = load_summary(SUMMARY)
    R = load_raw(RAW)
    make(A, R)
