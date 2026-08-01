"""fig3_dissociation — discrimination loss against calibration loss, per regime.

The third of the paper's figures that existed only as a PDF with no script
anywhere in the repo. Its file predates the regime-purity fixes, so it could not
be checked against the current tables; this regenerates it from them.

What it shows: for a source-trained ERM, how much discrimination it loses on
transfer (x) against how much calibration it loses (y), one point per regime. The
claim it supports is that the two axes move independently -- prevalence-moving
regimes sit upper-left with the ranking essentially intact and the worst
calibration, mechanism-moving regimes sit lower-right with the opposite pattern.
A model whose AUROC survives transfer can therefore be badly miscalibrated, which
is why the paper reports the two separately throughout.

Both losses are computed **per seed and then averaged**, not as a difference of
averages. The two agree in the mean but not in the spread: the same seed draws
one source and one target population, so the pair is matched and the paired
standard error is the honest one. Differencing the marginal means would discard
that pairing and inflate the error bars.

no_shift is drawn in grey as the noise floor. Without it the reader has no scale
for how far from zero a regime has to sit to mean anything.

Run from repo root:
    python paper_assets/_gen_fig3_dissociation.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

PREFIX = "benchmark_1s1t_full"          # the only prefix carrying prior_shift too
ERM = "all_observed"

# Grouped by which part of the generator the regime perturbs, because that
# grouping IS the finding: the two groups occupy different quadrants.
GROUPS = {
    "prevalence": (["baseline_hazard_shift", "prior_shift"], "#D95F0E",
                   "Moves event prevalence"),
    "mechanism":  (["observational_shift", "concept_shift"], "#1F6FB4",
                   "Moves observation map or outcome"),
    "covariates": (["covariate_shift", "spurious_shift"], "#009E73",
                   "Moves covariates only"),
}
LABELS = {
    "covariate_shift": "Covariate",
    "observational_shift": "Measurement",
    "spurious_shift": "Spurious",
    "baseline_hazard_shift": "Baseline hazard",
    "prior_shift": "Label-prior",
    "concept_shift": "Concept",
    "no_shift": "No shift",
}
# Nudges so labels do not sit on their marker or on each other. Pure cosmetics;
# no point moves.
OFFSETS = {
    "Covariate": (2, -15), "Measurement": (-34, -17), "Spurious": (4, 8),
    "Baseline hazard": (-20, -18), "Label-prior": (10, -3), "Concept": (-24, 11),
    "No shift": (-32, 7),
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_raw(prefix: str) -> list[dict]:
    p = os.path.join(TABLES, f"{prefix}_raw.csv")
    if not os.path.exists(p):
        raise SystemExit(
            f"manca {p}\n"
            f"  Rigenera con: python -m experiments.run_adaptation "
            f"--include_appendix --out_prefix {prefix}")
    with open(p) as f:
        return list(csv.DictReader(f))


def paired_losses(rows: list[dict]) -> dict:
    """Per-regime mean and paired standard error of the two transfer losses."""
    per = defaultdict(lambda: {"auroc": [], "ece": []})
    for r in rows:
        if r["method"] != ERM:
            continue
        try:
            per[r["scenario"]]["auroc"].append(float(r["src_auroc"]) - float(r["tgt_auroc"]))
            per[r["scenario"]]["ece"].append(float(r["tgt_ece"]) - float(r["src_ece"]))
        except (KeyError, ValueError):
            continue
    out = {}
    for s, d in per.items():
        a, e = np.array(d["auroc"], float), np.array(d["ece"], float)
        if len(a) == 0:
            continue
        out[s] = {
            "auroc_loss": float(a.mean()), "ece_loss": float(e.mean()),
            "auroc_se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "ece_se": float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else 0.0,
            "n_seeds": len(a),
        }
    return out


def make_figure(vals: dict, out_base: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color="#999999", lw=0.8, ls="--", zorder=1)

    drawn = []
    for _key, (scens, colour, legend) in GROUPS.items():
        xs, ys = [], []
        for s in scens:
            if s not in vals:
                continue
            v = vals[s]
            ax.errorbar(v["auroc_loss"], v["ece_loss"],
                        xerr=v["auroc_se"], yerr=v["ece_se"],
                        fmt="o", ms=7, color=colour, ecolor=colour,
                        elinewidth=1.0, capsize=2.5, zorder=3)
            xs.append(v["auroc_loss"])
            ys.append(v["ece_loss"])
            lab = LABELS.get(s, s)
            ax.annotate(lab, (v["auroc_loss"], v["ece_loss"]),
                        textcoords="offset points", xytext=OFFSETS.get(lab, (6, 6)),
                        fontsize=9, color=colour)
        if xs:
            drawn.append((colour, legend))

    if "no_shift" in vals:
        v = vals["no_shift"]
        ax.errorbar(v["auroc_loss"], v["ece_loss"],
                    xerr=v["auroc_se"], yerr=v["ece_se"],
                    fmt="s", ms=6, color="#7F7F7F", ecolor="#7F7F7F",
                    elinewidth=1.0, capsize=2.5, zorder=2)
        ax.annotate(LABELS["no_shift"], (v["auroc_loss"], v["ece_loss"]),
                    textcoords="offset points", xytext=OFFSETS["No shift"],
                    fontsize=9, color="#7F7F7F")
        drawn.append(("#7F7F7F", "No shift (noise floor)"))

    ax.set_xlabel("Discrimination loss on transfer  (AUROC, source $-$ target)")
    ax.set_ylabel("Calibration loss on transfer  (ECE, target $-$ source)")
    # The upper-right corner is empty by construction: nothing loses a lot of
    # discrimination AND a lot of calibration, which is the dissociation itself.
    # The legend goes there rather than forcing the axes wider.
    ax.margins(x=0.14, y=0.12)

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=l)
               for c, l in drawn]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5,
              handletextpad=0.4, borderaxespad=0.2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  scritto {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    vals = paired_losses(load_raw(args.prefix))
    print(f"fig3_dissociation da {args.prefix}_raw.csv\n")
    print(f"  {'regime':<24}{'AUROC loss':>14}{'ECE loss':>14}{'seed':>7}")
    order = [s for _g, (ss, _c, _l) in GROUPS.items() for s in ss] + ["no_shift"]
    for s in order:
        if s not in vals:
            print(f"  {LABELS.get(s, s):<24}{'assente':>14}")
            continue
        v = vals[s]
        print(f"  {LABELS.get(s, s):<24}{v['auroc_loss']:>+9.4f} ±{v['auroc_se']:.4f}"
              f"{v['ece_loss']:>+9.4f} ±{v['ece_se']:.4f}{v['n_seeds']:>7}")
    print()
    make_figure(vals, os.path.join(args.outdir, "fig3_dissociation"))


if __name__ == "__main__":
    main()
