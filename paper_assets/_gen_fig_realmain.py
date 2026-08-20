"""fig_realmain — method effects on the real cohorts, one panel per endpoint.

Three settings are shown and never pooled together, because their ERM baselines
are fitted on different data and their units of analysis differ:

  I3LUNG adaptation      30 directed transfers, pooled over 6 target cohorts
  I3LUNG generalization  6 leave-one-centre-out folds
  Registry adaptation    2 directed transfers

Pooled estimates and intervals come from experiments/meta_analysis.py, whose
output already carries the sign convention used here: for AUROC the estimate is
method minus ERM, for ECE it is ERM minus method, so positive favours the method
in both panels. Individual-unit points are recomputed from the per-transfer
summaries and are shown faintly; they are the reason the figure exists, since a
pooled interval cannot show how much the effect varies across cohorts.

Three encoding decisions are load-bearing.

Recalibration in the AUROC panel. Platt scaling and prior adjustment are
monotone increasing maps, so ΔAUROC is identically zero, not a measured null. A
filled marker would claim an estimate that was never made; a blank row reads as
missing data. The rows carry a hollow grey diamond at zero plus a footnote,
which is the standard "present, but not of the same kind" convention.

The registry gets no pooled diamond. Two directed transfers cannot support an
interval that a reader would compare against the six-unit intervals above it, so
the block shows the two points and the range between them, marked as such.

One deliberate restriction. i3lung_loco_meta.csv also contains the adaptation
methods and prior adjustment, which the leave-one-centre-out runs happen to
compute. They are not drawn in that block: the generalization protocol does not
observe the target, and prior adjustment reads its covariates, so including it
there would report a result under a premise it violates. The block shows IRM,
GroupDRO and Platt scaling, matching the comparison sets declared in the paper.

Run from repo root:
    python paper_assets/_gen_fig_realmain.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

from _style import apply as _apply_style
_apply_style()

ERM = "lr_erm"
C_ADAPT, C_GENER, C_RECAL = "#D95F0E", "#1B9E77", "#E6B422"
C_LABEL = {C_ADAPT: "#9E440A", C_GENER: "#0F6E56", C_RECAL: "#8A6A0C"}
BAND, GRID, ZERO, GREY = "#F7F7F5", "#E8E8E4", "#3A3A3A", "#8C8C8C"
# Block headings and footnotes start at the left margin of the whole figure,
# not at the left edge of the plotting area: starting them where the data
# begins pushed them into the zero line.
LEFT_EDGE = 0.035
MARGIN = 0.02

NAME = {"iw_erm": "IW-ERM", "coral": "CORAL", "dann": "DANN", "cdan": "CDAN",
        "irm": "IRM", "group_dro": "GroupDRO",
        "prior_recalib": "Prior adjustment", "platt_scaling": "Platt scaling"}
FAMILY = {"iw_erm": C_ADAPT, "coral": C_ADAPT, "dann": C_ADAPT, "cdan": C_ADAPT,
          "irm": C_GENER, "group_dro": C_GENER,
          "prior_recalib": C_RECAL, "platt_scaling": C_RECAL}

# (meta prefix, heading, unit of analysis, methods drawn, pooled)
# pooled=False draws the individual units and their range instead of an
# estimate and interval; see the module docstring.
SETTINGS = [
    ("i3lung_meta", "I3LUNG Adaptation",
     "6 target cohorts, 30 directed transfers",
     ["iw_erm", "coral", "dann", "cdan", "prior_recalib", "platt_scaling"],
     True),
    ("i3lung_loco_meta", "I3LUNG Domain Generalization",
     "6 held-out centres",
     ["irm", "group_dro", "platt_scaling"],
     True),
    ("public_rotterdam_gbsg_meta", "Registry Adaptation",
     "2 directed transfers",
     ["iw_erm", "coral", "dann", "cdan", "prior_recalib", "platt_scaling"],
     False),
]



def load_meta(tables: str, prefix: str) -> dict:
    """(metric, method) -> (estimate, ci_lo, ci_hi, k)"""
    out = {}
    path = os.path.join(tables, f"{prefix}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"manca {path}: rilancia experiments/meta_analysis.py")
    for r in csv.DictReader(open(path)):
        est = r.get("estimand", "")
        if not est.startswith("method:"):
            continue
        try:
            out[(r["metric"], est.split(":", 1)[1])] = (
                float(r["estimate"]), float(r["ci_lo"]), float(r["ci_hi"]),
                int(r["k"]))
        except (KeyError, ValueError):
            continue
    return out


def _rows(path_glob: str, strip: str, skip: str | None):
    for f in glob.glob(path_glob):
        b = os.path.basename(f)
        if skip and skip in b:
            continue
        unit = b.replace("_summary.csv", "").replace(strip, "")
        yield unit, list(csv.DictReader(open(f)))


def _delta(rows, metric):
    """{method: delta against ERM}, signed so positive favours the method."""
    col = "tgt_auroc_mean" if metric == "auroc" else "tgt_ece_mean"
    vals = {}
    for r in rows:
        try:
            vals[r["method"]] = float(r[col])
        except (KeyError, ValueError):
            continue
    if ERM not in vals:
        return {}
    return {m: (v - vals[ERM] if metric == "auroc" else vals[ERM] - v)
            for m, v in vals.items() if m != ERM}


def units(tables: str) -> dict:
    """(setting, metric, method) -> list of per-unit deltas.

    I3LUNG adaptation is averaged within target cohort first, so its points are
    the six units the pooled estimate is computed over rather than the thirty
    transfers, which are not independent.
    """
    out = defaultdict(list)
    per_target = defaultdict(lambda: defaultdict(list))
    for unit, rows in _rows(os.path.join(tables, "i3lung_*_summary.csv"),
                            "i3lung_", "loco"):
        if "_" not in unit:
            continue
        tgt = unit.split("_", 1)[1]
        for metric in ("auroc", "ece"):
            for m, d in _delta(rows, metric).items():
                per_target[(metric, m)][tgt].append(d)
    for (metric, m), d in per_target.items():
        out[("i3lung_meta", metric, m)] = [float(np.mean(v)) for v in d.values()]

    for _unit, rows in _rows(os.path.join(tables, "i3lung_loco_*_summary.csv"),
                             "i3lung_loco_", None):
        for metric in ("auroc", "ece"):
            for m, d in _delta(rows, metric).items():
                out[("i3lung_loco_meta", metric, m)].append(d)

    for _unit, rows in _rows(os.path.join(tables, "public_*_summary.csv"),
                             "public_", None):
        for metric in ("auroc", "ece"):
            for m, d in _delta(rows, metric).items():
                out[("public_rotterdam_gbsg_meta", metric, m)].append(d)
    return out


def layout():
    """Row positions, top to bottom, with a gap between settings."""
    y, rows, bands = 0.0, [], []
    for prefix, head, unit, methods, pooled in SETTINGS:
        top = y
        for m in methods:
            rows.append((y, prefix, m, pooled))
            y += 1.0
        bands.append((top - 0.5, y - 0.5, head, unit))
        y += 1.7
    total = y - 1.7
    return [(total - yy, p, m, k) for yy, p, m, k in rows], \
           [(total - b, total - a, h, u) for a, b, h, u in bands], total


def xrange_for(metric, meta, per_unit, rows, is_auroc):
    """Data limits plus padding, with room kept for the practical margin."""
    vals = []
    for _y, prefix, m, pooled in rows:
        if is_auroc and FAMILY[m] == C_RECAL:
            continue
        vals += list(per_unit.get((prefix, metric, m), []))
        e = meta.get(prefix, {}).get((metric, m))
        if e and pooled:
            vals += [e[1], e[2]]
    if not vals:
        return -0.05, 0.05
    lo, hi = min(vals), max(vals)
    if is_auroc:
        hi = max(hi, MARGIN + 0.004)
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def draw(ax, metric, meta, per_unit, rows, bands, total, xlabel, is_auroc):
    lo_x, hi_x = xrange_for(metric, meta, per_unit, rows, is_auroc)
    ax.set_xlim(lo_x, hi_x)
    ax.set_axisbelow(True)

    # Every setting gets the same band. Alternating them is only legible with
    # many blocks; with three it reads as one block having been forgotten.
    for lo, hi, _h, _u in bands:
        ax.axhspan(lo, hi, color=BAND, zorder=0, lw=0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=1)

    if is_auroc:
        # Neutral grey, not a family colour: shading the practically-relevant
        # zone in the green used for domain generalization reads as if the zone
        # belonged to that family.
        ax.axvspan(MARGIN, hi_x, color="#9AA0A6", alpha=0.07, zorder=1, lw=0)
        ax.axvline(MARGIN, color="#B9B9B9", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.axvline(0.0, color=ZERO, lw=1.0, zorder=3)

    for y, prefix, m, pooled in rows:
        col = FAMILY[m]

        if is_auroc and col == C_RECAL:
            ax.plot([0.0], [y], marker="D", ms=4.6, mfc="none", mec=GREY,
                    mew=0.9, zorder=5)
            continue

        pts = per_unit.get((prefix, metric, m), [])

        if not pooled:
            # Individual transfers are the estimate here, so they are drawn at
            # full weight and joined by their observed range.
            if len(pts) >= 2:
                ax.plot([min(pts), max(pts)], [y, y], color=col, lw=0.9,
                        ls=(0, (2, 2)), zorder=4)
            if pts:
                ax.plot(pts, [y] * len(pts), marker="o", ls="none", ms=4.0,
                        color=col, mec="white", mew=0.6, zorder=5)
            continue

        # Units sit just below the pooled row so the two do not collide.
        if pts:
            ax.scatter(pts, [y - 0.24] * len(pts), s=11, color=col, alpha=0.32,
                       edgecolors="none", zorder=3)
        e = meta.get(prefix, {}).get((metric, m))
        if e:
            est, lo, hi, _k = e
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round",
                    zorder=4)
            ax.plot([est], [y], marker="D", ms=5.2, color=col, mec="white",
                    mew=0.7, zorder=5)

    ax.set_yticks([y for y, _p, _m, _k in rows])
    ax.set_yticklabels([NAME[m] for _y, _p, m, _k in rows])
    ax.set_ylim(-0.9, total + 2.1)
    ax.set_xlabel(xlabel)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    meta = {p: load_meta(args.tables, p) for p, _h, _u, _m, _k in SETTINGS}
    per_unit = units(args.tables)
    rows, bands, total = layout()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.9), sharey=True)
    draw(axes[0], "auroc", meta, per_unit, rows, bands, total,
         r"$\Delta$ AUROC", True)
    draw(axes[1], "ece", meta, per_unit, rows, bands, total,
         r"$\Delta$ ECE", False)

    for lbl in axes[1].get_yticklabels():
        lbl.set_visible(False)

    # The family is carried by the row label itself, which removes the legend:
    # a key would only repeat what the rows already encode.
    for lbl, (_y, _p, m, _k) in zip(axes[0].get_yticklabels(), rows):
        lbl.set_color(C_LABEL[FAMILY[m]])

    # subplots_adjust first: the heading positions are computed from the axes
    # transform, which is only final once the layout is.
    fig.subplots_adjust(left=0.185, right=0.99, top=0.945, bottom=0.125,
                        wspace=0.07)

    # Drawn on the figure, not the left axes: the right panel has an opaque
    # background and covers any heading long enough to overflow.
    for lo, hi, head, unit in bands:
        yfig = fig.transFigure.inverted().transform(
            axes[0].transData.transform((0, hi + 0.35)))[1]
        fig.text(LEFT_EDGE, yfig + 0.015, head, ha="left", va="baseline",
                 fontsize=7.8, fontweight="bold", color="#2E2E2E")

    for ax, title in ((axes[0], "Discrimination"), (axes[1], "Calibration")):
        x0 = ax.get_position().x0
        fig.text(0.5 * (x0 + ax.get_position().x1), 0.965, title,
                 ha="center", va="baseline", fontsize=9.0,
                 fontweight="bold", color="#2E2E2E")


    os.makedirs(args.outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(args.outdir, f"fig_realmain.{ext}"), **kw)
    plt.close(fig)
    print(f"  scritto {args.outdir}/fig_realmain.{{pdf,png}}")
    for p, h, u, ms, pooled in SETTINGS:
        k = next((v[3] for v in meta[p].values()), None)
        tag = f"k={k}" if pooled else "unpooled"
        print(f"    {h:<32}{tag:<12}({u})")


if __name__ == "__main__":
    main()