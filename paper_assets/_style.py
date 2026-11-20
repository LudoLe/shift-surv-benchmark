"""Shared visual language for the paper's figures.

The generators grew independently and drifted: font sizes of 9, 10 and 11, two
different dpi settings, three different grid treatments, and legends styled per
file. On a page where four of them sit within two columns of each other, that
reads as carelessness rather than variety.

Chart types still differ -- bars, forest, lines, heatmap -- and should. What is
shared here is everything that is not the chart: type sizes, frame, grid, tick
treatment, legend defaults, and the regime palette.

Import and call `apply()` once at module level, then `finish(ax)` on each axes:

    from _style import apply, finish, REGIME, SERIES
    apply()
    ...
    finish(ax)
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Okabe-Ito, colour-blind safe. One entry per shift regime, used wherever a
# regime is a series: fig3, fig_nsweep, and any per-regime line or point.
REGIME = {
    "covariate_shift":       "#0072B2",
    "observational_shift":   "#D55E00",
    "spurious_shift":        "#009E73",
    "baseline_hazard_shift": "#CC79A7",
    "prior_shift":           "#56B4E9",
    "concept_shift":         "#E69F00",
}

# Method families, used wherever a *method* is a series rather than a regime.
SERIES = {
    "erm":          "#1F6FB4",
    "adaptation":   "#D95F0E",
    "generalization": "#1B9E77",
    "recalibration": "#E6B422",
    "obs_oracle":   "#CFCFCF",
    "lat_oracle":   "#4D4D4D",
}

GRID = "#E8E8E8"
FRAME = "#444444"
# Dash pattern for the latent oracle. A level, not a bar top, and consistent
# across every figure that draws one.
LEVEL_DASH = (0, (4, 2))


def apply() -> None:
    """Rc settings shared by every generator. Call once, at import time."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": FRAME,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        # Grid and tick treatment go here rather than in a per-axes helper, so a
        # generator gets them by importing the module instead of remembering to
        # call something after every plot. That is what drifted before.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": FRAME,
        "ytick.color": FRAME,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })


def finish(ax, grid_axis: str = "y") -> None:
    """Frame, grid and ticks. Applied after the data is drawn, because setting
    the grid first and plotting after leaves it on top of the bars."""
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    ax.tick_params(length=3.5, width=0.8, color=FRAME)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bracket(ax, x0: float, x1: float, y: float, text: str,
            drop: float = 0.007, colour: str = "#555555") -> None:
    """Significance bracket spanning exactly the two series compared.

    Kept here rather than duplicated because the alternative -- a label centred
    over the whole group -- reads as a claim about every bar in it, including
    reference bars no test was run on.
    """
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop],
            color=colour, lw=0.9, zorder=7, solid_capstyle="butt")
    ax.text(0.5 * (x0 + x1), y + drop * 0.6, text, ha="center", va="bottom",
            fontsize=8.5, color="#333333")


def seed_points(ax, xs, ys, colour: str, size: float = 13.0) -> None:
    """Per-seed values as open circles in the series colour.

    Open rather than filled so ten of them over a bar stay legible, and coloured
    rather than grey so a reader can tell which series they belong to when two
    bars sit a few points apart.
    """
    ax.scatter(xs, ys, s=size, facecolors="none", edgecolors=colour,
               linewidths=0.8, alpha=0.85, zorder=6)


# Print sizes. Figures are drawn at the size they are printed in the IEEE
# two-column layout, so a point on the canvas is a point on the page. Drawing
# larger and letting LaTeX shrink the image is what made the camera-ready text
# unreadable: a 4.6 in figure in a 3.5 in column prints 9.5 pt ticks at 7 pt.
COLUMN_WIDTH = 3.5    # in, \columnwidth
TEXT_WIDTH = 7.16     # in, \textwidth


def apply_print() -> None:
    """Type sizes for figures drawn at their printed size (>= 7 pt)."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "savefig.dpi": 300,
        # TrueType rather than Type 3: IEEE PDF eXpress flags Type 3 fonts.
        "pdf.fonttype": 42,
    })
