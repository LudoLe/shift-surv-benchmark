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
MARGIN = 0.02

NAME = {"iw_erm": "IW-ERM", "coral": "CORAL", "dann": "DANN", "cdan": "CDAN",
        "irm": "IRM", "group_dro": "GroupDRO",
        "prior_recalib": "Prior adjustment", "platt_scaling": "Platt scaling"}
FAMILY = {"iw_erm": C_ADAPT, "coral": C_ADAPT, "dann": C_ADAPT, "cdan": C_ADAPT,
          "irm": C_GENER, "group_dro": C_GENER,
          "prior_recalib": C_RECAL, "platt_scaling": C_RECAL}

# (meta prefix, heading, unit of analysis, methods drawn)
SETTINGS = [
    ("i3lung_meta", "I3LUNG adaptation",
     "30 transfers, 6 target cohorts",
     ["iw_erm", "coral", "dann", "cdan", "prior_recalib", "platt_scaling"]),
    ("i3lung_loco_meta", "I3LUNG domain generalization",
     "6 held-out centres",
     ["irm", "group_dro", "platt_scaling"]),
    ("public_rotterdam_gbsg_meta", "Registry adaptation",
     "2 directed transfers",
     ["iw_erm", "coral", "dann", "cdan", "prior_recalib", "platt_scaling"]),
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
    for prefix, head, unit, methods in SETTINGS:
        top = y
        for m in methods:
            rows.append((y, prefix, m))
            y += 1.0
        bands.append((top - 0.5, y - 0.5, head, unit))
        y += 1.5
    total = y - 1.5
    return [(total - yy, p, m) for yy, p, m in rows], \
           [(total - b, total - a, h, u) for a, b, h, u in bands], total


def draw(ax, metric, meta, per_unit, rows, bands, total, xlabel, show_margin):
    for i, (lo, hi, _h, _u) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#F6F6F6", zorder=0, lw=0)
    if show_margin:
        ax.axvline(MARGIN, color="#B9B9B9", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0.0, color="#3A3A3A", lw=1.0, zorder=2)

    for y, prefix, m in rows:
        # Recalibration is a monotone increasing map, so it cannot change the
        # ranking and is not part of the AUROC comparison. Its rows stay blank
        # in that panel rather than showing a point pinned to zero, which would
        # read as a measured null instead of an identity.
        if metric == "auroc" and FAMILY[m] == C_RECAL:
            continue
        col = FAMILY[m]
        pts = per_unit.get((prefix, metric, m), [])
        if pts:
            ax.scatter(pts, [y] * len(pts), s=13, color=col, alpha=0.30,
                       edgecolors="none", zorder=3)
        e = meta.get(prefix, {}).get((metric, m))
        if e:
            est, lo, hi, _k = e
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round",
                    zorder=4)
            ax.plot([est], [y], marker="D", ms=5.2, color=col, mec="white",
                    mew=0.7, zorder=5)

    ax.set_yticks([y for y, _p, _m in rows])
    ax.set_yticklabels([NAME[m] for _y, _p, m in rows])
    ax.set_ylim(-0.9, total + 2.1)
    ax.set_xlabel(xlabel)
    ax.grid(False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    meta = {p: load_meta(args.tables, p) for p, _h, _u, _m in SETTINGS}
    per_unit = units(args.tables)
    rows, bands, total = layout()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.9), sharey=True)
    draw(axes[0], "auroc", meta, per_unit, rows, bands, total,
         r"$\Delta$AUROC   (method $-$ ERM)", True)
    draw(axes[1], "ece", meta, per_unit, rows, bands, total,
         r"$\Delta$ECE   (ERM $-$ method)", False)

    # Setting headings sit above their block on the left panel only, so the row
    # labels are not repeated and nothing needs a legend.
    # subplots_adjust first: the heading positions are computed from the axes
    # transform, which is only final once the layout is.
    fig.subplots_adjust(left=0.185, right=0.99, top=0.985, bottom=0.115,
                        wspace=0.07)

    # Drawn on the figure, not the left axes: the right panel has an opaque
    # background and covers any heading long enough to overflow.
    for lo, hi, head, unit in bands:
        yfig = fig.transFigure.inverted().transform(
            axes[0].transData.transform((0, hi + 0.35)))[1]
        fig.text(0.185, yfig + 0.012, head, ha="left", va="baseline",
                 fontsize=7.8, fontweight="bold", color="#2E2E2E")
        fig.text(0.185, yfig - 0.004, unit, ha="left", va="baseline",
                 fontsize=6.9, color="#777777")

    os.makedirs(args.outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(args.outdir, f"fig_realmain.{ext}"), **kw)
    plt.close(fig)
    print(f"  scritto {args.outdir}/fig_realmain.{{pdf,png}}")
    for p, h, u, ms in SETTINGS:
        k = next((v[3] for v in meta[p].values()), None)
        print(f"    {h:<32}k={k}  ({u})")


if __name__ == "__main__":
    main()
