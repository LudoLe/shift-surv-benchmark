"""
Generate fig2_headroom (grouped vertical barplot, Nature style).
Run from repo root:  python paper_assets/_gen_fig2_bars.py
"""
import csv, os, glob
import numpy as np
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
DA_METHODS = ("coral", "dann", "cdan")

C_ERM    = "#566573"
C_DA     = "#2471A3"
C_ORACLE = "#1a1a2e"

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
    """Returns {(scenario, method): [tgt_auroc per seed]}."""
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["scenario"], r["method"])
            d.setdefault(key, [])
            try:
                d[key].append(float(r["tgt_auroc"]))
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


def best_da(A, sc):
    cands = [(get(A, sc, m, "tgt_auroc"), m) for m in DA_METHODS]
    cands = [(t, m) for t, m in cands if not np.isnan(t[0])]
    return max(cands, key=lambda x: x[0][0])[0] if cands else (np.nan, np.nan, np.nan)


def best_da_method(A, sc):
    cands = [(get(A, sc, m, "tgt_auroc")[0], m) for m in DA_METHODS]
    cands = [(v, m) for v, m in cands if not np.isnan(v)]
    return max(cands)[1] if cands else None


def make(A, R):
    keys   = [k for k, _ in SCENARIOS]
    labels = [l for _, l in SCENARIOS]
    n      = len(keys)
    x      = np.arange(n)

    W  = 0.24   # bar width
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.15)

    for i, (sc, _) in enumerate(SCENARIOS):
        erm_m, erm_lo, erm_hi     = get(A, sc, "all_observed",     "tgt_auroc")
        da_m,  da_lo,  da_hi      = best_da(A, sc)
        orc_m, orc_lo, orc_hi     = get(A, sc, "invariant_oracle", "tgt_auroc")
        da_key = best_da_method(A, sc)

        xi = x[i]

        # ── ERM bar ──────────────────────────────────────────────────────
        ax.bar(xi - W, erm_m, W, color=C_ERM,
               linewidth=0, zorder=2)
        ax.errorbar(xi - W, erm_m,
                    yerr=[[erm_m - erm_lo], [erm_hi - erm_m]],
                    fmt="none", color="white", elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=4)

        # ── Best DA bar ───────────────────────────────────────────────────
        ax.bar(xi, da_m, W, color=C_DA,
               linewidth=0, zorder=2)
        ax.errorbar(xi, da_m,
                    yerr=[[da_m - da_lo], [da_hi - da_m]],
                    fmt="none", color="white", elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=4)

        # ── Oracle bar (outline/hatch only) ──────────────────────────────
        ax.bar(xi + W, orc_m, W,
               color="none", edgecolor=C_ORACLE,
               linewidth=1.0, hatch="///", zorder=2)
        ax.errorbar(xi + W, orc_m,
                    yerr=[[orc_m - orc_lo], [orc_hi - orc_m]],
                    fmt="none", color=C_ORACLE, elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=4)

        # ── jittered seed dots (ERM and best DA) ─────────────────────────
        for jx, key, col in [
            (xi - W, (sc, "all_observed"), C_ERM),
            (xi,     (sc, da_key),         C_DA),
        ]:
            seeds = R.get(key, [])
            if seeds:
                jitter = rng.uniform(-0.06, 0.06, len(seeds))
                ax.scatter(jx + jitter, seeds,
                           s=4, color=col, alpha=0.55,
                           linewidths=0, zorder=3)

    # ── axes ──────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=0.9)
    ax.set_ylabel("Target AUROC", labelpad=4)
    ax.set_ylim(0.50, 0.92)
    ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls=":", zorder=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── legend ────────────────────────────────────────────────────────────
    leg = [
        plt.Rectangle((0, 0), 1, 1, fc=C_ERM,  ec="none",     label="ERM"),
        plt.Rectangle((0, 0), 1, 1, fc=C_DA,   ec="none",     label="Best DA"),
        plt.Rectangle((0, 0), 1, 1, fc="none", ec=C_ORACLE,
                       linewidth=1.0, hatch="///",             label="Oracle (headroom)"),
    ]
    ax.legend(handles=leg, loc="lower left", frameon=False,
              ncol=3, fontsize=7.5, handlelength=1.0, handleheight=0.9,
              columnspacing=0.8, handletextpad=0.4, borderpad=0,
              bbox_to_anchor=(0.0, -0.02))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_headroom.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_headroom.{pdf,png}")


if __name__ == "__main__":
    A = load_summary(SUMMARY)
    R = load_raw(RAW)
    make(A, R)
