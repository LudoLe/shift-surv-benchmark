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


PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--highlight", default="SZMC")
    args = ap.parse_args()

    rows = load(args.tables)
    if not rows:
        raise SystemExit(f"nessun transfer pairwise trovato in {args.tables}")
    sources = sorted({r["source"] for r in rows})
    colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sources)}

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(0.0, color="#999999", lw=0.8, zorder=1)
    for c in sources:
        pts = [r for r in rows if r["source"] == c]
        hi = (c == args.highlight)
        ax.scatter([r["mmd"] for r in pts], [r["drop"] for r in pts],
                   s=34 if hi else 24,
                   facecolors=colour[c] if hi else "none",
                   edgecolors=colour[c], linewidths=1.2,
                   zorder=4 if hi else 3, label=c)
    rho = _spearman([r["mmd"] for r in rows], [r["drop"] for r in rows])
    # Reported descriptively and with no fitted line: the thirty transfers share
    # six source centres and are not independent, so a regression line would
    # assert a model the design does not support.
    ax.text(0.97, 0.95, rf"Spearman $\rho={rho:+.2f}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#333333")
    ax.set_xlabel("Source--target RBF-MMD")
    ax.set_ylabel("ERM AUROC loss (source $-$ target)")
    ax.grid(axis="both", color="#F0F0F0", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(title="Source centre", loc="lower left", frameon=False,
              fontsize=6.8, title_fontsize=7, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(args.outdir, f"fig_predictability.{ext}"), **kw)
    plt.close(fig)
    print(f"  scritto {args.outdir}/fig_predictability.{{pdf,png}}   "
          f"{len(rows)} transfer, Spearman {rho:+.3f}")


if __name__ == "__main__":
    main()
