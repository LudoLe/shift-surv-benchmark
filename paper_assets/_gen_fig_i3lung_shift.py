"""fig_i3lung_shift — the two shift mechanisms measurable on I3LUNG without labels.

The last of the paper's figures that existed only as a PDF with no script. Its
file predates the regime-purity fixes and the 30-transfer shift audit, so it
could not be checked against the current tables; this rebuilds it from them.

  (a) pairwise RBF-MMD between the six centres            covariate shift
  (b) six-month survival per centre, with Wilson intervals  label-prior shift

Two things worth knowing about panel (a).

The matrix is **directed**, and deliberately so. The audit computes MMD in a
source-fitted standardised space -- what a deployed model actually sees -- so
A->B and B->A are measured in different spaces and do not agree: the median
disagreement across these centres is about 30% and the largest is over 90%.
Averaging them into a symmetric "pairwise distance" would hide that, and would
assert a symmetry the measurement does not have. Rows are sources, columns are
targets, and the two triangles are separate numbers.

The figure carries no significance marks. The audit bootstraps MMD but does not
compute the permutation test the earlier caption quoted, and MMD is non-negative
so a bootstrap interval never excludes zero in a useful way. Claiming
significance would therefore mean reporting a test this script did not run. If
the claim is wanted in the caption, run the permutation test and add it here.

Wilson rather than Wald intervals in panel (b): with survival up to 0.84 and
centres as small as 191 patients, the normal approximation puts the upper limit
past 1 and understates the asymmetry.

Run from repo root:
    python paper_assets/_gen_fig_i3lung_shift.py
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

AUDIT = "shift_audit_transfers.csv"
C_BAR = "#1F6FB4"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_audit(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"manca {path}\n"
            f"  Rigenera con: python -m experiments.shift_audit --data_dir I3LUNG_DATA")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["transfer"].startswith("i3lung_")]
    if not rows:
        raise SystemExit(f"{path} non contiene transfer i3lung "
                         "(l'audit e' stato girato senza --data_dir?)")
    return rows


def assemble(rows: list[dict]) -> tuple[list[str], np.ndarray, dict]:
    centres = sorted({c for r in rows for c in r["transfer"].split("_")[1:3]})
    idx = {c: i for i, c in enumerate(centres)}
    k = len(centres)

    directed = np.full((k, k), np.nan)
    prev: dict[str, tuple[float, int]] = {}
    for r in rows:
        s, t = r["transfer"].split("_")[1:3]
        directed[idx[s], idx[t]] = float(r["mmd"])
        # Every centre appears as source in k-1 transfers with the same value.
        prev.setdefault(s, (float(r["prev_src_retro"]), int(r["n_src"])))

    # Reported because it is the reason the matrix stays directed.
    asym = [abs(directed[i, j] - directed[j, i]) / max(directed[i, j], directed[j, i])
            for i in range(k) for j in range(i + 1, k)
            if directed[i, j] == directed[i, j] and directed[j, i] == directed[j, i]
            and max(directed[i, j], directed[j, i]) > 0]
    if asym:
        print(f"  asimmetria fra le due direzioni: mediana "
              f"{np.median(asym):.1%}, massimo {np.max(asym):.1%}  "
              f"(per questo la matrice resta direzionale)")
    return centres, directed, prev


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def make_figure(centres: list[str], M: np.ndarray, prev: dict,
                out_base: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- (a) MMD matrix ----
    k = len(centres)
    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F2F2F2")
    im = ax1.imshow(masked, cmap=cmap, vmin=0, vmax=float(np.nanmax(M)))
    ax1.set_xticks(range(k), centres, rotation=45, ha="right")
    ax1.set_yticks(range(k), centres)
    ax1.set_xlabel("Target centre")
    ax1.set_ylabel("Source centre")
    for sp in ax1.spines.values():
        sp.set_visible(False)
    thr = float(np.nanmax(M)) * 0.55
    for i in range(k):
        for j in range(k):
            if i == j or np.isnan(M[i, j]):
                continue
            ax1.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                     color="white" if M[i, j] > thr else "#333333")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("RBF-MMD", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---- (b) prevalence ----
    order = sorted(prev, key=lambda c: prev[c][0])
    vals = [prev[c][0] for c in order]
    ns = [prev[c][1] for c in order]
    los, his = zip(*[wilson(p, n) for p, n in zip(vals, ns)])
    err = np.array([np.array(vals) - np.array(los), np.array(his) - np.array(vals)])

    ax2.bar(range(len(order)), vals, color=C_BAR, width=0.62, zorder=2)
    ax2.errorbar(range(len(order)), vals, yerr=err, fmt="none", ecolor="#333333",
                 elinewidth=1.1, capsize=3, zorder=3)
    ax2.set_xticks(range(len(order)),
                   [f"{c}\n$n$={n}" for c, n in zip(order, ns)], fontsize=8.5)
    ax2.set_ylabel("Six-month survival (positive class)")
    ax2.set_ylim(0, 1.0)
    ax2.axhline(float(np.average(vals, weights=ns)), color="#D95F0E", lw=1.1,
                ls="--", zorder=4)
    ax2.text(len(order) - 0.45, float(np.average(vals, weights=ns)) + 0.025,
             "pooled", color="#D95F0E", fontsize=8.5, ha="right")
    ax2.grid(axis="y", color="#E6E6E6", lw=0.7, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    # Panel labels below the axes, no titles: the style the advisor asked for.
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.5, -0.30, lab, transform=ax.transAxes, ha="center",
                va="top", fontsize=10)
    for ext in ("pdf", "png"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  scritto {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(TABLES, AUDIT))
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_audit(args.audit)
    print(f"fig_i3lung_shift da {os.path.basename(args.audit)} "
          f"({len(rows)} transfer)\n")
    centres, M, prev = assemble(rows)
    print(f"\n  {'centro':<8}{'n':>7}{'survival':>11}{'IC 95%':>18}")
    for c in sorted(prev, key=lambda z: prev[z][0]):
        p, n = prev[c]
        lo, hi = wilson(p, n)
        print(f"  {c:<8}{n:>7}{p:>11.3f}   [{lo:.3f}, {hi:.3f}]")
    off = M[~np.isnan(M)]
    print(f"\n  MMD: min {off.min():.4f}  mediana {np.median(off):.4f}  "
          f"max {off.max():.4f}\n")
    make_figure(centres, M, prev, os.path.join(args.outdir, "fig_i3lung_shift"))


if __name__ == "__main__":
    main()
