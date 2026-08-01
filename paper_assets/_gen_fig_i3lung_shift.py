"""fig_i3lung_shift — the two shift mechanisms measurable on I3LUNG without labels.

Regenerates the figure in the style of the final draft
(shift-surv-figures-FINAL) from the current 30-transfer shift audit. The
previous PDF had no generator anywhere in the repo.

  (a) pairwise RBF-MMD between the six centres            covariate shift
  (b) six-month survival per centre, Wilson intervals     label-prior shift

Centres are ordered by survival in both panels, so the reader can carry the
ordering from one to the other.

Two notes on panel (a). The audit measures MMD in a source-fitted standardised
space -- what a deployed model actually sees -- so MMD(A->B) and MMD(B->A) are
computed in different spaces and disagree, by a median of about 30% across these
centres. The matrix shown is symmetric, each cell averaging the two directions,
which is what the layout requires; the per-cell disagreement is printed to
stdout so it can be checked rather than assumed away.

The figure also carries no significance marks. The audit bootstraps MMD but runs
no permutation test, and MMD is non-negative so a bootstrap interval cannot
exclude zero usefully. Marking significance would mean reporting a test this
script did not run.

Panel (b) plots six-month survival, the positive class of the real cohort, not
event prevalence: the values run 0.58 to 0.84 and the two are complementary, so
labelling them as prevalence inverts the quantity.

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
C_BAR = "#E1A11A"          # the amber of the reference figure

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 200,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
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


def assemble(rows: list[dict]):
    """Symmetric MMD matrix plus per-centre survival, both ordered by survival."""
    prev: dict[str, tuple[float, int]] = {}
    directed: dict[tuple[str, str], float] = {}
    for r in rows:
        s, t = r["transfer"].split("_")[1:3]
        directed[(s, t)] = float(r["mmd"])
        # Every centre appears as source in k-1 transfers with the same value.
        prev.setdefault(s, (float(r["prev_src_retro"]), int(r["n_src"])))

    order = sorted(prev, key=lambda c: prev[c][0])
    k = len(order)
    idx = {c: i for i, c in enumerate(order)}

    M = np.zeros((k, k))
    asym = []
    for (s, t), v in directed.items():
        back = directed.get((t, s))
        if back is None:
            M[idx[s], idx[t]] = M[idx[t], idx[s]] = v
            continue
        M[idx[s], idx[t]] = M[idx[t], idx[s]] = 0.5 * (v + back)
        if max(v, back) > 0:
            asym.append(abs(v - back) / max(v, back))
    if asym:
        print(f"  disaccordo fra le due direzioni: mediana {np.median(asym):.1%}, "
              f"massimo {np.max(asym):.1%}  (la matrice mostra la media)")
    return order, M, prev


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _fmt(v: float) -> str:
    """.02 rather than 0.02: the reference figure's compact cell format."""
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0.") else s


def _save(fig, out_base: str, panel: str) -> None:
    for ext in ("pdf", "png"):
        p = f"{out_base}_{panel}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  scritto {p}")
    plt.close(fig)


def make_figure(order: list[str], M: np.ndarray, prev: dict, out_base: str) -> None:
    """Two files, not one. The panels are composed into a single float with
    subfigure captions in the .tex, exactly as fig_disc and fig_cal are, so the
    caption of each panel lives next to the panel rather than inside the image."""
    k = len(order)

    # ---- (a) MMD matrix ----
    fig, ax1 = plt.subplots(figsize=(4.5, 3.9))
    vmax = float(M.max())
    im = ax1.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax1.set_xticks(range(k), order, rotation=45, ha="right", fontsize=10)
    ax1.set_yticks(range(k), order, fontsize=10)
    ax1.tick_params(length=0)
    ax1.grid(False)
    for sp in ax1.spines.values():
        sp.set_visible(False)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            ax1.text(j, i, _fmt(M[i, j]), ha="center", va="center", fontsize=9,
                     color="white" if M[i, j] > 0.6 * vmax else "#333333")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=2.5)
    fig.tight_layout()
    _save(fig, out_base, "a")

    # ---- (b) survival ----
    fig, ax2 = plt.subplots(figsize=(4.5, 3.9))
    vals = np.array([prev[c][0] for c in order])
    ns = [prev[c][1] for c in order]
    los, his = zip(*[wilson(p, n) for p, n in zip(vals, ns)])
    err = np.array([vals - np.array(los), np.array(his) - vals])

    ax2.bar(range(k), vals, color=C_BAR, width=0.66, zorder=2)
    ax2.errorbar(range(k), vals, yerr=err, fmt="none", ecolor="#333333",
                 elinewidth=1.2, capsize=3.5, zorder=3)
    ax2.axhline(0.5, color="#999999", lw=0.9, ls=":", zorder=1)
    ax2.set_xticks(range(k), order, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("Six-month survival")
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(labelsize=10, length=3.5, width=0.8, color="#444444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_base, "b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(TABLES, AUDIT))
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_audit(args.audit)
    print(f"fig_i3lung_shift da {os.path.basename(args.audit)} "
          f"({len(rows)} transfer)\n")
    order, M, prev = assemble(rows)
    print(f"\n  {'centro':<8}{'n':>7}{'survival':>11}{'IC 95%':>18}")
    for c in order:
        p, n = prev[c]
        lo, hi = wilson(p, n)
        print(f"  {c:<8}{n:>7}{p:>11.3f}   [{lo:.3f}, {hi:.3f}]")
    off = M[~np.eye(len(order), dtype=bool)]
    print(f"\n  MMD: min {off.min():.4f}  mediana {np.median(off):.4f}  "
          f"max {off.max():.4f}\n")
    make_figure(order, M, prev, os.path.join(args.outdir, "fig_i3lung_shift"))


if __name__ == "__main__":
    main()
