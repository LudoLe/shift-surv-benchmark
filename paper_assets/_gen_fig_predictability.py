"""fig_predictability — what a deployment can measure does not tell it what will fail.

Each point is one directed I3LUNG transfer: on the x axis the covariate distance
between the two centers, measured without labels; on the y axis how much ERM
loses going from source to target. If the quantity adaptation methods minimize
were the quantity that governs transfer, the cloud would rise to the right.

Why this replaces a forest plot of pooled estimates
---------------------------------------------------
A pooled point with an interval answers "how large is the average effect", which
the meta-analysis already states in the text. It cannot show the thing the
section is about: that the spread across transfers is wide, that some transfers
lose a great deal and others nothing, and that the label-free diagnostics do not
separate the two. The association is reported descriptively -- the thirty
transfers share six centers and are not independent -- so the figure carries the
claim the number cannot, which is that there is no usable trend to exploit.

Run from repo root:
    python paper_assets/_gen_fig_predictability.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

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

C_PT = "#1F6FB4"
C_HL = "#D95F0E"


def load(tables: str) -> list[dict]:
    """One row per pairwise transfer: covariate distance, prevalence gap, ERM drop."""
    diag = {}
    for f in glob.glob(os.path.join(tables, "i3lung_*_shiftdiag.csv")):
        if "loco" in os.path.basename(f):
            continue
        name = os.path.basename(f).replace("_shiftdiag.csv", "").replace("i3lung_", "")
        for r in csv.DictReader(open(f)):
            try:
                diag[name] = (float(r["rbf_mmd"]), abs(float(r["prevalence_gap"])),
                              r["source"], r["target"])
            except (KeyError, ValueError):
                pass
    rows = []
    for f in glob.glob(os.path.join(tables, "i3lung_*_summary.csv")):
        if "loco" in os.path.basename(f):
            continue
        name = os.path.basename(f).replace("_summary.csv", "").replace("i3lung_", "")
        if name not in diag:
            continue
        for r in csv.DictReader(open(f)):
            if r.get("method") != "lr_erm":
                continue
            try:
                src_a, tgt_a = float(r["src_auroc_mean"]), float(r["tgt_auroc_mean"])
            except (KeyError, ValueError):
                continue
            mmd, pg, s, t = diag[name]
            rows.append(dict(transfer=name, mmd=mmd, prev_gap=pg,
                             drop=src_a - tgt_a, tgt_auroc=tgt_a, source=s, target=t))
    return rows


def _spearman(a, b) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A = np.array([ra[v] for v in a], float)
    B = np.array([rb[v] for v in b], float)
    A -= A.mean(); B -= B.mean()
    den = (A @ A) ** 0.5 * (B @ B) ** 0.5
    return float(A @ B / den) if den else float("nan")


def panel(ax, rows, xkey, xlabel, highlight):
    x = [r[xkey] for r in rows]
    y = [r["drop"] for r in rows]
    hi = [r["source"] == highlight for r in rows]
    ax.axhline(0.0, color="#999999", lw=0.8, zorder=1)
    ax.scatter([v for v, h in zip(x, hi) if not h],
               [v for v, h in zip(y, hi) if not h],
               s=26, facecolors="none", edgecolors=C_PT, linewidths=1.1, zorder=3)
    ax.scatter([v for v, h in zip(x, hi) if h],
               [v for v, h in zip(y, hi) if h],
               s=34, color=C_HL, zorder=4, label=f"source: {highlight}")
    ax.set_xlabel(xlabel)
    rho = _spearman(x, y)
    ax.text(0.97, 0.04, rf"Spearman $\rho={rho:+.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="#333333")
    ax.grid(axis="both", color="#EDEDED", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return rho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--highlight", default="SZMC")
    args = ap.parse_args()

    rows = load(args.tables)
    if not rows:
        raise SystemExit(f"nessun transfer pairwise trovato in {args.tables}")
    print(f"  {len(rows)} transfer")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    r1 = panel(axes[0], rows, "mmd", "Covariate distance (RBF-MMD)", args.highlight)
    r2 = panel(axes[1], rows, "prev_gap", "Prevalence gap", args.highlight)
    axes[0].set_ylabel("ERM AUROC loss\n(source $-$ target)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig_predictability.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  scritto {args.outdir}/fig_predictability.{{pdf,png}}")
    print(f"  Spearman: MMD {r1:+.3f}   prevalence gap {r2:+.3f}")


if __name__ == "__main__":
    main()
