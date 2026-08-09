"""fig_real — the real-data method effects, and the directional heatmap.

(a) directional heatmap: how much ERM loses going from each source center to
    each target center. A pooled estimate cannot show direction; this can, and
    the asymmetry is the point -- a center can be unusable as a source while
    being unremarkable as a target.
(b) per-method AUROC effect against ERM, one point per target cohort plus the
    pooled estimate.
(c) the same for ECE.

Panels (b) and (c) show the individual units, not only the pooled diamond,
because the spread across target cohorts is what the section argues about: the
pooled number says the average effect is nil, the spread says you cannot tell in
advance which transfer will be which.

Run from repo root:
    python paper_assets/_gen_fig_real3.py
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
ADAPT = ["iw_erm", "coral", "dann", "cdan"]
RECAL = ["prior_recalib", "platt_scaling"]
GENER = ["irm", "group_dro"]
LABEL = {"iw_erm": "IW-ERM", "coral": "CORAL", "dann": "DANN", "cdan": "CDAN",
         "prior_recalib": "Prior adj.", "platt_scaling": "Platt",
         "irm": "IRM", "group_dro": "GroupDRO"}
C_ADAPT, C_RECAL, C_GENER = "#D95F0E", "#E6B422", "#1B9E77"
MARGIN = 0.02


def _pairs(tables: str) -> dict:
    """(source, target) -> {method: (src_auroc, tgt_auroc, tgt_ece)}"""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(tables, "i3lung_*_summary.csv")):
        b = os.path.basename(f)
        if "loco" in b:
            continue
        stem = b.replace("_summary.csv", "").replace("i3lung_", "")
        if "_" not in stem:
            continue
        s, t = stem.split("_", 1)
        for r in csv.DictReader(open(f)):
            m = r.get("method")
            try:
                out[(s, t)][m] = (float(r["src_auroc_mean"]),
                                  float(r["tgt_auroc_mean"]),
                                  float(r["tgt_ece_mean"]))
            except (KeyError, ValueError):
                continue
    return out


def _loco(tables: str) -> dict:
    """target -> {method: (tgt_auroc, tgt_ece)} from the leave-one-center-out folds."""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(tables, "i3lung_loco_*_summary.csv")):
        t = os.path.basename(f).replace("_summary.csv", "").replace("i3lung_loco_", "")
        for r in csv.DictReader(open(f)):
            try:
                out[t][r["method"]] = (float(r["tgt_auroc_mean"]),
                                       float(r["tgt_ece_mean"]))
            except (KeyError, ValueError):
                continue
    return out


def _registry(tables: str) -> dict:
    """(source, target) -> {method: (src_auroc, tgt_auroc, tgt_ece)} for the registry.

    Two directions only, so they are drawn as individual points and no pooled
    estimate is shown: with two units a pooled diamond would suggest a precision
    the design cannot deliver.
    """
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(tables, "public_*_summary.csv")):
        stem = os.path.basename(f).replace("_summary.csv", "")
        parts = stem.split("_")
        s_c, t_c = (parts[-3], parts[-2]) if len(parts) >= 3 else ("src", "tgt")
        for r in csv.DictReader(open(f)):
            try:
                out[(s_c, t_c)][r["method"]] = (float(r["src_auroc_mean"]),
                                                float(r["tgt_auroc_mean"]),
                                                float(r["tgt_ece_mean"]))
            except (KeyError, ValueError):
                continue
    return out


def _reg_deltas(reg, idx):
    acc = defaultdict(list)
    for _k, d in reg.items():
        if ERM not in d:
            continue
        for m in ADAPT + RECAL:
            if m in d:
                acc[m].append(d[m][idx] - d[ERM][idx] if idx == 1
                              else d[ERM][idx] - d[m][idx])
    return acc


def panel_heatmap(ax, pairs, centers):
    """ERM's source-to-target loss, source on rows and target on columns."""
    M = np.full((len(centers), len(centers)), np.nan)
    for i, s in enumerate(centers):
        for j, t in enumerate(centers):
            v = pairs.get((s, t), {}).get(ERM)
            if v:
                M[i, j] = v[0] - v[1]          # source AUROC minus target AUROC
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(centers))); ax.set_xticklabels(centers, rotation=45, ha="right")
    ax.set_yticks(range(len(centers))); ax.set_yticklabels(centers)
    ax.set_xlabel("Target"); ax.set_ylabel("Source")
    for i in range(len(centers)):
        for j in range(len(centers)):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="#F2F2F2", edgecolor="none"))
            else:
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 * lim else "#222222")
    ax.set_xticks(np.arange(-.5, len(centers), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(centers), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ERM AUROC loss", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _by_target(pairs, loco, idx):
    """method -> {target: mean delta against ERM over the transfers into it}.

    idx selects the metric: 1 = AUROC (method minus ERM), 2 = ECE (ERM minus
    method), so that positive favours the method in both cases.
    """
    acc = defaultdict(lambda: defaultdict(list))
    for (s, t), d in pairs.items():
        if ERM not in d:
            continue
        for m in ADAPT + RECAL:
            if m in d:
                acc[m][t].append(d[m][idx] - d[ERM][idx] if idx == 1
                                 else d[ERM][idx] - d[m][idx])
    for t, d in loco.items():
        if ERM not in d:
            continue
        j = idx - 1
        for m in GENER:
            if m in d:
                acc[m][t].append(d[m][j] - d[ERM][j] if idx == 1
                                 else d[ERM][j] - d[m][j])
    return {m: {t: float(np.mean(v)) for t, v in d.items()} for m, d in acc.items()}


def panel_forest(ax, per_unit, reg_vals, methods, xlabel, show_margin):
    ys = np.arange(len(methods))[::-1]
    ax.axvline(0.0, color="#777777", lw=0.9, zorder=1)
    if show_margin:
        ax.axvline(MARGIN, color="#BBBBBB", lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.axvline(-MARGIN, color="#BBBBBB", lw=0.9, ls=(0, (3, 2)), zorder=1)
    for y, m in zip(ys, methods):
        col = C_ADAPT if m in ADAPT else (C_RECAL if m in RECAL else C_GENER)
        vals = list(per_unit.get(m, {}).values())
        if not vals:
            continue
        jit = (np.random.default_rng(abs(hash(m)) % 2**32).random(len(vals)) - .5) * .28
        ax.scatter(vals, y + jit, s=17, facecolors="none", edgecolors=col,
                   linewidths=1.0, zorder=3)
        mu = float(np.mean(vals))
        ax.scatter([mu], [y], marker="D", s=46, color=col, zorder=4)
        rv = reg_vals.get(m, [])
        if rv:
            ax.scatter(rv, [y - 0.30] * len(rv), marker="^", s=20, color=col,
                       alpha=0.85, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([LABEL.get(m, m) for m in methods])
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.7, len(methods) - 0.3)
    ax.grid(axis="x", color="#EDEDED", lw=0.7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    pairs, loco, reg = _pairs(args.tables), _loco(args.tables), _registry(args.tables)
    if not pairs:
        raise SystemExit(f"nessun transfer pairwise in {args.tables}")
    centers = sorted({c for p in pairs for c in p})
    methods = ADAPT + RECAL + GENER
    print(f"  {len(pairs)} transfer, {len(centers)} centri, {len(loco)} fold LOCO, "
          f"{len(reg)} direzioni registry")

    # --- main paper: method effects, two panels -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    panel_forest(axes[0], _by_target(pairs, loco, 1), _reg_deltas(reg, 1),
                 methods, r"$\Delta$AUROC", True)
    axes[0].set_title("(a) discrimination", loc="left", fontsize=9)
    panel_forest(axes[1], _by_target(pairs, loco, 2), _reg_deltas(reg, 2),
                 methods, r"$\Delta$ECE", False)
    axes[1].set_title("(b) calibration", loc="left", fontsize=9)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec="#555555",
               label="I3LUNG target cohort"),
        Line2D([], [], marker="D", ls="", color="#555555", label="pooled"),
        Line2D([], [], marker="^", ls="", color="#555555", label="registry direction"),
    ], loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.4)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig_real.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  scritto {args.outdir}/fig_real.{{pdf,png}}")

    # --- supplement: the directional heatmap --------------------------------
    figh, axh = plt.subplots(figsize=(3.6, 3.1))
    panel_heatmap(axh, pairs, centers)
    figh.tight_layout()
    for ext in ("pdf", "png"):
        figh.savefig(os.path.join(args.outdir, f"fig_heatmap.{ext}"), dpi=200,
                     bbox_inches="tight")
    plt.close(figh)
    print(f"  scritto {args.outdir}/fig_heatmap.{{pdf,png}}")


if __name__ == "__main__":
    main()
