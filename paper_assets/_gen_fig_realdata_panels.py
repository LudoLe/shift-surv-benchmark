"""
Regenerate the four real-data dot-whisker panels at an aspect ratio that suits
their content.

The previous versions were nearly square (about 1.23:1) while carrying only
four to eight rows, so most of each panel was empty vertical space: they read
as oversized and, stacked two per figure, pushed the paper over the page limit.
Height here is proportional to the number of methods plotted, which roughly
halves the vertical footprint at identical width and legibility.

Sources (random-effects pooled estimates produced by experiments/meta_analysis.py):
  i3lung_meta.csv        adaptation + recalibration over the 7 centre transfers
  i3lung_loco_meta.csv   IRM / GroupDRO under leave-one-centre-out (6 folds)
  public_rotterdam_gbsg_meta.csv   registry, both directions x three horizons

Note the I3LUNG panels mix two evaluation designs, k=7 for adaptation and k=6
for the leave-one-centre-out generalization methods; this matches the text and
the previous figure, and the caption should keep saying so.

Run from repo root:
    python paper_assets/_gen_fig_realdata_panels.py
"""
from __future__ import annotations

import argparse
import csv
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.expanduser("~/Downloads/shift-surv-outputs/tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

# Method family -> colour, matching the palette already used in the paper.
ADAPT, GENER, RECAL = "#D95F0E", "#1B9E77", "#E6B422"
FAMILY_LABEL = {ADAPT: "Adaptation", GENER: "Generalization", RECAL: "Recalibration"}

# display name, csv estimand, family colour, which file
M = lambda name, key, colour, src: (name, key, colour, src)

I3_DISC = [
    M("IW-ERM",   "method:iw_erm",     ADAPT, "i3"),
    M("CORAL",    "method:coral",      ADAPT, "i3"),
    M("DANN",     "method:dann",       ADAPT, "i3"),
    M("CDAN",     "method:cdan",       ADAPT, "i3"),
    M("IRM",      "method:irm",        GENER, "loco"),
    M("GroupDRO", "method:group_dro",  GENER, "loco"),
]
I3_CAL = [
    M("IW-ERM",   "method:iw_erm",         ADAPT, "i3"),
    M("CORAL",    "method:coral",          ADAPT, "i3"),
    M("DANN",     "method:dann",           ADAPT, "i3"),
    M("CDAN",     "method:cdan",           ADAPT, "i3"),
    M("Saerens",  "method:prior_recalib",  RECAL, "i3"),
    M("Platt",    "method:platt_scaling",  RECAL, "i3"),
    M("IRM",      "method:irm",            GENER, "loco"),
    M("GroupDRO", "method:group_dro",      GENER, "loco"),
]
REG_DISC = [
    M("IW-ERM", "method:iw_erm", ADAPT, "reg"),
    M("CORAL",  "method:coral",  ADAPT, "reg"),
    M("DANN",   "method:dann",   ADAPT, "reg"),
    M("CDAN",   "method:cdan",   ADAPT, "reg"),
]
REG_CAL = [
    M("IW-ERM",  "method:iw_erm",        ADAPT, "reg"),
    M("CORAL",   "method:coral",         ADAPT, "reg"),
    M("DANN",    "method:dann",          ADAPT, "reg"),
    M("CDAN",    "method:cdan",          ADAPT, "reg"),
    M("Saerens", "method:prior_recalib", RECAL, "reg"),
    M("Platt",   "method:platt_scaling", RECAL, "reg"),
]

from _style import apply as _apply_style, finish as _finish_axes
_apply_style()


def load(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r["metric"], r["estimand"])] = r
    return out


# Geometry shared by all four panels so they compose as a grid. Absolute inches,
# not fractions: the panels hold different numbers of methods, and only a fixed
# left margin and a fixed row pitch make their plot boxes line up on the page.
WIDTH_IN = 3.4
LEFT_IN = 0.92        # room for the longest label ("GroupDRO")
RIGHT_IN = 0.10
BOTTOM_IN = 0.52      # x tick labels + axis label
TOP_IN = 0.10
ROW_IN = 0.26         # vertical pitch per method


def panel(rows, tables, metric, xlabel, outfile, xlim=None, xticks=None):
    n = len(rows)
    height = ROW_IN * n + BOTTOM_IN + TOP_IN
    fig = plt.figure(figsize=(WIDTH_IN, height))
    ax = fig.add_axes([
        LEFT_IN / WIDTH_IN,
        BOTTOM_IN / height,
        (WIDTH_IN - LEFT_IN - RIGHT_IN) / WIDTH_IN,
        (height - BOTTOM_IN - TOP_IN) / height,
    ])

    ys = list(range(n))[::-1]          # first method at the top
    for y, (name, key, colour, src) in zip(ys, rows):
        r = tables[src].get((metric, key))
        if r is None:
            raise KeyError(f"{metric}/{key} not in the {src} table")
        est, lo, hi = float(r["estimate"]), float(r["ci_lo"]), float(r["ci_hi"])
        ax.plot([lo, hi], [y, y], color=colour, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot([est], [y], "o", color=colour, ms=4.6, zorder=3)

    ax.axvline(0.0, color="0.6", ls="--", lw=0.9, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, n - 0.4)
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.tick_params(axis="y", length=0)

    # No per-panel legend. Four panels carried three different legends in three
    # different places, which was the main thing making the block look untidy;
    # the colour key lives in the shared caption instead.

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"{outfile}.{ext}")
        fig.savefig(p, dpi=200)          # no bbox_inches: it would undo the fixed margins
        print(f"wrote {p}  ({height:.2f} in tall, {n} methods)")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default=DOWNLOADS,
                    help="directory with i3lung_loco_meta.csv and the registry meta CSV")
    args = ap.parse_args()

    tables = {
        "i3":   load(os.path.join(ROOT, "outputs", "tables", "i3lung_meta.csv")),
        "loco": load(os.path.join(args.meta_dir, "i3lung_loco_meta.csv")),
        "reg":  load(os.path.join(args.meta_dir, "public_rotterdam_gbsg_meta.csv")),
    }

    # Shared limits per metric so the two cohorts are visually comparable.
    # Few ticks on purpose: at 3.4 in wide a denser grid runs the labels together.
    auroc_lim, auroc_ticks = (-0.055, 0.032), [-0.04, -0.02, 0.0, 0.02]
    ece_lim, ece_ticks = (-0.085, 0.045), [-0.08, -0.04, 0.0, 0.04]

    panel(I3_DISC,  tables, "auroc", "AUROC gap (method $-$ ERM)",
          "fig_i3lung_disc", auroc_lim, auroc_ticks)
    panel(I3_CAL,   tables, "ece",   "ECE gain (ERM $-$ method)",
          "fig_i3lung_cal", ece_lim, ece_ticks)
    panel(REG_DISC, tables, "auroc", "AUROC gap (method $-$ ERM)",
          "fig_registry_disc", auroc_lim, auroc_ticks)
    panel(REG_CAL,  tables, "ece",   "ECE gain (ERM $-$ method)",
          "fig_registry_cal", ece_lim, ece_ticks)


if __name__ == "__main__":
    main()
