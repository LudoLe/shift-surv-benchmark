"""
Alternative layout for the real-data results: TWO forest plots instead of four
panels.

One panel per metric, with both cohorts inside it as labelled blocks:

    fig_real_forest_disc : AUROC gap,  I3LUNG block then Registry block
    fig_real_forest_cal  : ECE gain,   same structure

Why this might read better than the 2x2 version: the comparison the paper
actually makes is "does any method beat ERM", and here every method for a given
metric sits on one shared x-axis, so that comparison is a single vertical scan.
The cost is losing the 2x2 symmetry, and the two panels have different heights
because the calibration panel carries the two recalibration methods as well.

Writes to different filenames than _gen_fig_realdata_panels.py, so both layouts
can coexist while choosing between them. LaTeX for this variant is in
paper_assets/snippet_forest_variant.tex.

Run from repo root:
    python paper_assets/_gen_fig_realdata_forest.py
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.expanduser("~/Downloads/shift-surv-outputs/tables")
OUTDIR = os.path.expanduser("~/Desktop/shift-surv-paper-images")

ADAPT, GENER, RECAL = "#D95F0E", "#1B9E77", "#E6B422"

# (display name, csv estimand, colour, source table)
ROWS_DISC = [
    ("I3LUNG", None, None, None),
    ("IW-ERM",   "method:iw_erm",    ADAPT, "i3"),
    ("CORAL",    "method:coral",     ADAPT, "i3"),
    ("DANN",     "method:dann",      ADAPT, "i3"),
    ("CDAN",     "method:cdan",      ADAPT, "i3"),
    ("IRM",      "method:irm",       GENER, "loco"),
    ("GroupDRO", "method:group_dro", GENER, "loco"),
    ("Registry", None, None, None),
    ("IW-ERM",   "method:iw_erm",    ADAPT, "reg"),
    ("CORAL",    "method:coral",     ADAPT, "reg"),
    ("DANN",     "method:dann",      ADAPT, "reg"),
    ("CDAN",     "method:cdan",      ADAPT, "reg"),
]

ROWS_CAL = [
    ("I3LUNG", None, None, None),
    ("IW-ERM",   "method:iw_erm",        ADAPT, "i3"),
    ("CORAL",    "method:coral",         ADAPT, "i3"),
    ("DANN",     "method:dann",          ADAPT, "i3"),
    ("CDAN",     "method:cdan",          ADAPT, "i3"),
    ("Saerens",  "method:prior_recalib", RECAL, "i3"),
    ("Platt",    "method:platt_scaling", RECAL, "i3"),
    ("IRM",      "method:irm",           GENER, "loco"),
    ("GroupDRO", "method:group_dro",     GENER, "loco"),
    ("Registry", None, None, None),
    ("IW-ERM",   "method:iw_erm",        ADAPT, "reg"),
    ("CORAL",    "method:coral",         ADAPT, "reg"),
    ("DANN",     "method:dann",          ADAPT, "reg"),
    ("CDAN",     "method:cdan",          ADAPT, "reg"),
    ("Saerens",  "method:prior_recalib", RECAL, "reg"),
    ("Platt",    "method:platt_scaling", RECAL, "reg"),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9.5,
})

# Fixed geometry in inches so the two panels share margins and row pitch.
WIDTH_IN = 3.4
LEFT_IN = 0.95
RIGHT_IN = 0.10
BOTTOM_IN = 0.50
TOP_IN = 0.08
ROW_IN = 0.235
HEADER_EXTRA_IN = 0.10     # breathing room above a cohort header


def load(path: str) -> dict:
    with open(path) as fh:
        return {(r["metric"], r["estimand"]): r for r in csv.DictReader(fh)}


def forest(rows, tables, metric, xlabel, outfile, xlim, xticks):
    # Lay rows out top to bottom, giving headers a little extra space above.
    slots, cursor = [], 0.0
    for i, (name, key, colour, src) in enumerate(rows):
        if key is None and i > 0:
            cursor += HEADER_EXTRA_IN / ROW_IN
        slots.append(cursor)
        cursor += 1.0
    total = cursor
    height = ROW_IN * total + BOTTOM_IN + TOP_IN

    fig = plt.figure(figsize=(WIDTH_IN, height))
    ax = fig.add_axes([
        LEFT_IN / WIDTH_IN,
        BOTTOM_IN / height,
        (WIDTH_IN - LEFT_IN - RIGHT_IN) / WIDTH_IN,
        (height - BOTTOM_IN - TOP_IN) / height,
    ])

    label_x = -LEFT_IN / (WIDTH_IN - LEFT_IN - RIGHT_IN)   # figure's left edge

    ticks, ticklabels = [], []
    for slot, (name, key, colour, src) in zip(slots, rows):
        y = total - slot                       # invert: first row at the top
        if key is None:
            # Cohort header: bold, flush left, no marker.
            ax.text(label_x, y, name, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontweight="bold", fontsize=9)
            continue
        r = tables[src].get((metric, key))
        if r is None:
            raise KeyError(f"{metric}/{key} missing from the {src} table")
        est, lo, hi = float(r["estimate"]), float(r["ci_lo"]), float(r["ci_hi"])
        ax.plot([lo, hi], [y, y], color=colour, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot([est], [y], "o", color=colour, ms=4.6, zorder=3)
        ticks.append(y)
        ticklabels.append(name)

    ax.axvline(0.0, color="0.6", ls="--", lw=0.9, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_ylim(total - slots[-1] - 0.7, total - slots[0] + 0.7)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.tick_params(axis="y", length=0)

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"{outfile}.{ext}")
        fig.savefig(p, dpi=200)
        print(f"wrote {p}  ({height:.2f} in tall)")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default=DOWNLOADS)
    args = ap.parse_args()

    tables = {
        "i3":   load(os.path.join(ROOT, "outputs", "tables", "i3lung_meta.csv")),
        "loco": load(os.path.join(args.meta_dir, "i3lung_loco_meta.csv")),
        "reg":  load(os.path.join(args.meta_dir, "public_rotterdam_gbsg_meta.csv")),
    }

    forest(ROWS_DISC, tables, "auroc", "AUROC gap (method $-$ ERM)",
           "fig_real_forest_disc", (-0.055, 0.032), [-0.04, -0.02, 0.0, 0.02])
    forest(ROWS_CAL, tables, "ece", "ECE gain (ERM $-$ method)",
           "fig_real_forest_cal", (-0.085, 0.045), [-0.08, -0.04, 0.0, 0.04])


if __name__ == "__main__":
    main()
