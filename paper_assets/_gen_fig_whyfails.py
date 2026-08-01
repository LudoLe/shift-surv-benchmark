"""
Regenerate the decision-geometry figure ("why alignment fails"), now including
the adversarial methods.

The previous fig_whyfails.pdf predated `--with_dann`, so it showed only ERM,
CORAL, IW-ERM, IRM and GroupDRO. That left the obvious gap: the paper's claim is
about *alignment*, and DANN/CDAN are the methods that align distributions most
aggressively. With them included the result is stronger, not weaker -- they
rotate the decision direction further from the target optimum than ERM does.

Panels are written as two files so LaTeX subcaption supplies "(a)"/"(b)" below,
matching every other multi-panel figure in the paper.

  fig_whyfails_a : decision angle vs the ERM AUROC gap, over the measurement
                   rotation sweep (one point per rotation angle, NOT per regime)
  fig_whyfails_b : covariance distance removed by CORAL per regime, annotated
                   with the residual change in decision angle it buys

Source: outputs/tables/decision_geometry_{sweep,scenarios}.csv, produced by
    python -m experiments.decision_geometry --with_dann

Run from repo root:
    python paper_assets/_gen_fig_whyfails.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

# key in the CSV -> (label, colour). ERM is the baseline, drawn dark.
METHODS = [
    ("erm",       "ERM",      "#3A3A3A"),
    ("coral",     "CORAL",    "#D95F0E"),
    ("iw",        "IW-ERM",   "#D9739F"),
    ("irm",       "IRM",      "#1B9E77"),
    ("group_dro", "GroupDRO", "#E6B422"),
    ("dann",      "DANN",     "#1F6FB4"),
    ("cdan",      "CDAN",     "#7B5AA6"),
]

SCEN_LABEL = {
    "covariate_shift": "Covariate", "observational_shift": "Measurement",
    "spurious_shift": "Spurious", "concept_shift": "Concept",
}
SCEN_ORDER = ["covariate_shift", "observational_shift", "spurious_shift", "concept_shift"]

from _style import apply as _apply_style, finish as _finish_axes
_apply_style()

WIDTH_IN = 3.4
LEFT_IN, RIGHT_IN, BOTTOM_IN, TOP_IN = 0.72, 0.12, 0.52, 0.10


def load(name):
    with open(os.path.join(TABLES, f"decision_geometry_{name}.csv")) as fh:
        return list(csv.DictReader(fh))


def axes(height):
    fig = plt.figure(figsize=(WIDTH_IN, height))
    ax = fig.add_axes([LEFT_IN / WIDTH_IN, BOTTOM_IN / height,
                       (WIDTH_IN - LEFT_IN - RIGHT_IN) / WIDTH_IN,
                       (height - BOTTOM_IN - TOP_IN) / height])
    return fig, ax


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(p, dpi=200)
        print(f"wrote {p}")
    plt.close(fig)


def panel_a():
    rows = sorted(load("sweep"), key=lambda r: float(r["auroc_gap_erm_mean"]))
    x = np.array([float(r["auroc_gap_erm_mean"]) for r in rows])
    fig, ax = axes(2.55)
    for key, label, colour in METHODS:
        y = np.array([float(r[f"angle_{key}_mean"]) for r in rows])
        ci = np.array([float(r[f"angle_{key}_ci95"]) for r in rows])
        ax.plot(x, y, "-o", color=colour, lw=1.5, ms=3.4, label=label, zorder=3)
        ax.fill_between(x, y - ci, y + ci, color=colour, alpha=0.10, lw=0, zorder=1)
    ax.set_xlabel("AUROC gap in transfer (ERM)")
    ax.set_ylabel("Angle to target-optimal ($^\\circ$)")
    ax.legend(ncol=2, loc="upper left", handletextpad=0.4, columnspacing=1.0,
              borderpad=0.1, labelspacing=0.25)
    save(fig, "fig_whyfails_a")


def panel_b():
    rows = {r["scenario"]: r for r in load("scenarios")}
    labels, red, dtheta = [], [], []
    for sc in SCEN_ORDER:
        if sc not in rows:
            continue
        r = rows[sc]
        labels.append(SCEN_LABEL[sc])
        red.append(100.0 * float(r["covdist_reduced_frac"]))
        dtheta.append(float(r["angle_coral_mean"]) - float(r["angle_erm_mean"]))

    fig, ax = axes(2.55)
    xs = np.arange(len(labels))
    ax.bar(xs, red, color="#D95F0E", width=0.62, zorder=2)
    for x, y, d in zip(xs, red, dtheta):
        ax.annotate(f"$\\Delta\\theta={d:+.1f}^\\circ$", xy=(x, y), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color="0.25")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Covariance distance\nremoved by CORAL (%)")
    ax.set_ylim(0, max(red) * 1.30)
    save(fig, "fig_whyfails_b")

    print("\nvalori usati nel pannello b:")
    for l, r_, d in zip(labels, red, dtheta):
        print(f"  {l:<14s} riduzione {r_:5.1f}%   Delta theta CORAL vs ERM {d:+.2f} deg")

    print("\nangoli per scenario (pannello a e' lo sweep; questi vanno nel TESTO):")
    hdr = "  " + f"{'scenario':<14}" + "".join(f"{lab:>10}" for _, lab, _ in METHODS)
    print(hdr)
    for sc in SCEN_ORDER:
        if sc not in rows:
            continue
        vals = "".join(f"{float(rows[sc][f'angle_{k}_mean']):>10.1f}" for k, _, _ in METHODS)
        print(f"  {SCEN_LABEL[sc]:<14}{vals}")


if __name__ == "__main__":
    panel_a()
    panel_b()
