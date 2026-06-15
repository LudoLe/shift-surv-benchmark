"""
Generate fig4_realdata: I3LUNG consistency check, two panels (like fig1).
One row per center-to-center transfer; point = best DA method vs ERM.
  (a) AUROC gap = AUROC(best-AUROC DA) - ERM        (>0 = DA improves discrimination)
  (b) ECE   gap = ECE(ERM) - ECE(best-calib DA)     (>0 = DA improves calibration)
Each panel picks the best DA method *for its own metric*, so every point is a
single model. No oracle on real data, so ERM is the reference (line at 0).

Uncertainty is the PATIENT-LEVEL BOOTSTRAP (i3lung_*_bootstrap.csv): per-seed
predictions are averaged per patient, then target patients are resampled, so the
bars are 2.5/97.5 percentile CIs over target-cohort sampling and the stars are
the bootstrap two-sided p-value of the gap vs ERM. This replaces the earlier
per-seed t-test, whose variance reflected only the train/val split, not cohort
sampling. CIs are therefore asymmetric.
Run from repo root:  python paper_assets/_gen_fig4_realdata.py
"""
import csv, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(ROOT, "outputs", "tables")
OUT  = os.path.join(ROOT, "paper_assets")

DA_METHODS = ("coral", "dann", "cdan")

# distinct colour per transfer (matches the legend ordering, top -> bottom)
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
          "#9467bd", "#8c564b", "#17becf"]

BAND = "#eef2f7"

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":      ["Latin Modern Roman", "Computer Modern Roman",
                        "Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
})


def load_transfers():
    boot_files = sorted(f for f in glob.glob(os.path.join(TAB, "i3lung_*_*_bootstrap.csv"))
                        if "i3lung_cb" not in os.path.basename(f))
    transfers = []
    for f in boot_files:
        name = os.path.basename(f).replace("i3lung_", "").replace("_bootstrap.csv", "")
        src, tgt = name.split("_")
        rows = {}
        for r in csv.DictReader(open(f)):
            def g(k):
                try:
                    return float(r[k])
                except (ValueError, KeyError):
                    return np.nan
            rows[r["method"]] = {
                "auroc":        g("auroc"),
                "auroc_gap":    g("auroc_gap_vs_ref"),
                "auroc_gap_lo": g("auroc_gap_lo"),
                "auroc_gap_hi": g("auroc_gap_hi"),
                "auroc_gap_p":  g("auroc_gap_p"),
                "ece_gap":      g("ece_gap_vs_ref"),   # erm - method (>0 = DA lower ECE)
                "ece_gap_lo":   g("ece_gap_lo"),
                "ece_gap_hi":   g("ece_gap_hi"),
                "ece_gap_p":    g("ece_gap_p"),
            }
        transfers.append((f"{src}→{tgt}", rows))
    return transfers


def _star(p):
    return "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""


def best_gap(rows, kind):
    """Pick the best DA method for `kind` ('auroc' or 'ece') and return its
    bootstrap gap vs ERM as (gap, lo, hi, star). The gap and its CI/p-value are
    precomputed in the bootstrap file (paired patient resampling vs lr_erm);
    both gap conventions are already oriented so positive = DA better.

    Selection: AUROC -> highest target AUROC; ECE -> largest ECE improvement.
    """
    if kind == "auroc":
        cand = {m: rows[m]["auroc"] for m in DA_METHODS
                if m in rows and not np.isnan(rows[m]["auroc"])}
        if not cand:
            return np.nan, 0.0, 0.0, ""
        m = max(cand, key=cand.get)
        r = rows[m]
        return r["auroc_gap"], r["auroc_gap_lo"], r["auroc_gap_hi"], _star(r["auroc_gap_p"])
    else:
        cand = {m: rows[m]["ece_gap"] for m in DA_METHODS
                if m in rows and not np.isnan(rows[m]["ece_gap"])}
        if not cand:
            return np.nan, 0.0, 0.0, ""
        m = max(cand, key=cand.get)   # largest improvement (erm - method)
        r = rows[m]
        return r["ece_gap"], r["ece_gap_lo"], r["ece_gap_hi"], _star(r["ece_gap_p"])


def make():
    transfers = load_transfers()
    n = len(transfers)
    labels = [t[0] for t in transfers]
    yi = np.arange(n - 1, -1, -1, dtype=float)   # first transfer at top

    # collect best-DA bootstrap gaps: (gap, lo, hi, star)
    A = [best_gap(rows, "auroc") for _, rows in transfers]
    E = [best_gap(rows, "ece")   for _, rows in transfers]

    fig = plt.figure(figsize=(6.4, 2.9))
    axA = fig.add_axes([0.17, 0.16, 0.36, 0.80])
    axB = fig.add_axes([0.60, 0.16, 0.36, 0.80])

    def draw(ax, recs, xlabel):
        for i, yy in enumerate(yi):
            if i % 2 == 0:
                ax.axhspan(yy - 0.5, yy + 0.5, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color="#999999", lw=0.9, ls="--", zorder=1)
        for (m, lo, hi, star), yy, col in zip(recs, yi, COLORS):
            xerr = [[max(0.0, m - lo)], [max(0.0, hi - m)]]   # asymmetric percentile CI
            ax.errorbar(m, yy, xerr=xerr, fmt="o", color=col, ms=6,
                        markeredgewidth=0, elinewidth=1.0, capsize=0, zorder=4)
            if star:
                # place star just beyond the CI on the favourable side
                xpos = (hi + 0.0015) if m >= 0 else (lo - 0.0015)
                ax.text(xpos, yy + 0.18, star, va="center",
                        ha="left" if m >= 0 else "right",
                        fontsize=8, color=col, zorder=5)
        ax.set_yticks(yi)
        ax.set_ylim(-0.6, n - 0.4)
        ax.tick_params(axis="y", left=False, pad=4)
        ax.set_xlabel(xlabel, labelpad=3)

    draw(axA, A, "AUROC gap")
    draw(axB, E, "ECE gap")

    axA.set_yticklabels(labels)
    axB.set_yticklabels([])

    # data-driven limits (bootstrap CIs are wider and asymmetric)
    def _lims(recs, pad=0.006):
        los = [lo for _, lo, hi, _ in recs if not np.isnan(lo)]
        his = [hi for _, lo, hi, _ in recs if not np.isnan(hi)]
        return min(los + [0.0]) - pad, max(his + [0.0]) + pad

    axA.set_xlim(*_lims(A))
    axB.set_xlim(*_lims(E))
    axA.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None))
    axB.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None))
    _fmt = ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 5e-4 else f"{v:+.2f}")
    axA.xaxis.set_major_formatter(_fmt)
    axB.xaxis.set_major_formatter(_fmt)

    # direction hints
    axA.text(0.99, 1.03, "DA better →", transform=axA.transAxes,
             ha="right", va="bottom", fontsize=7, color="#777777")
    axB.text(0.99, 1.03, "DA better →", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=7, color="#777777")

    # panel labels
    for ax, lbl in [(axA, "a"), (axB, "b")]:
        ax.text(-0.02, 1.05, f"({lbl})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="right", color="#222")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_realdata.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_realdata.{pdf,png}")


if __name__ == "__main__":
    make()
