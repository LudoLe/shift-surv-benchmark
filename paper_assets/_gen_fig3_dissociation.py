"""fig3_dissociation — discrimination loss against calibration loss, per regime.

Regenerates the figure in the style of the final draft (shift-surv-figures-FINAL)
from the current tables. The previous PDF had no generator anywhere in the repo
and predated the regime-purity fixes, so several of its points moved: covariate
and spurious shift no longer raise the observation noise in the target, which is
what put them above the zero line before.

Style follows the rest of the paper: one Okabe-Ito colour per regime, labels
annotated beside their own point in that colour, no legend, a full frame, and
dashed guides at zero on both axes.

Both losses are computed **per seed and then averaged**, not as a difference of
averages. The two agree in the mean but not in the spread: a seed draws one
source and one target population, so the pair is matched and the paired interval
is the honest one. Intervals use t(0.975, 9) rather than 1.96, since ten seeds is
not asymptotic.

Run from repo root:
    python paper_assets/_gen_fig3_dissociation.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

PREFIX = "benchmark_1s1t_full"          # the only prefix carrying prior_shift too
ERM = "all_observed"
_T9 = 2.262                              # t(0.975, df=9), for the ten-seed protocol

# Okabe-Ito, one per regime, matching the palette used by fig_disc and fig_cal.
REGIMES = [
    ("baseline_hazard_shift", "Baseline hazard", "#CC79A7"),
    ("prior_shift",           "Label-prior",     "#56B4E9"),
    ("spurious_shift",        "Spurious",        "#009E73"),
    ("covariate_shift",       "Covariate",       "#0072B2"),
    ("observational_shift",   "Measurement",     "#D55E00"),
    ("concept_shift",         "Concept",         "#E69F00"),
]
# Label placement in points, relative to the marker. Cosmetic only; no point
# moves. Covariate sits left of Spurious at almost the same height, so its label
# goes below-left rather than right, where it would land on Spurious's marker.
OFFSETS = {
    "Baseline hazard": (14, -4),
    "Label-prior":     (14, 3),
    "Spurious":        (14, 2),
    "Covariate":       (-20, -17),
    "Measurement":     (14, 3),
    "Concept":         (-12, 12),
}

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 200,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
})


def load_raw(prefix: str) -> list[dict]:
    p = os.path.join(TABLES, f"{prefix}_raw.csv")
    if not os.path.exists(p):
        raise SystemExit(
            f"manca {p}\n"
            f"  Rigenera con: python -m experiments.run_adaptation "
            f"--include_appendix --out_prefix {prefix}")
    with open(p) as f:
        return list(csv.DictReader(f))


def paired_losses(rows: list[dict]) -> dict:
    """Per-regime mean and paired 95% interval of the two transfer losses."""
    per = defaultdict(lambda: {"auroc": [], "ece": []})
    for r in rows:
        if r["method"] != ERM:
            continue
        try:
            per[r["scenario"]]["auroc"].append(float(r["src_auroc"]) - float(r["tgt_auroc"]))
            per[r["scenario"]]["ece"].append(float(r["tgt_ece"]) - float(r["src_ece"]))
        except (KeyError, ValueError):
            continue

    def ci(v):
        return float(_T9 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    out = {}
    for s, d in per.items():
        a, e = np.array(d["auroc"], float), np.array(d["ece"], float)
        if len(a) == 0:
            continue
        out[s] = {"auroc_loss": float(a.mean()), "ece_loss": float(e.mean()),
                  "auroc_ci": ci(a), "ece_ci": ci(e), "n_seeds": len(a)}
    return out


def make_figure(vals: dict, out_base: str) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.3))
    ax.axhline(0, color="#CCCCCC", lw=0.9, ls="--", zorder=1)
    ax.axvline(0, color="#CCCCCC", lw=0.9, ls="--", zorder=1)

    for key, label, colour in REGIMES:
        if key not in vals:
            print(f"  ! {key} assente da {PREFIX}_raw.csv")
            continue
        v = vals[key]
        ax.errorbar(v["auroc_loss"], v["ece_loss"],
                    xerr=v["auroc_ci"], yerr=v["ece_ci"],
                    fmt="o", ms=8, color=colour, ecolor=colour,
                    elinewidth=1.4, capsize=0, zorder=3)
        ax.annotate(label, (v["auroc_loss"], v["ece_loss"]),
                    textcoords="offset points", xytext=OFFSETS.get(label, (14, 3)),
                    fontsize=10.5, color=colour)

    ax.set_xlabel("AUROC loss in transfer (source $-$ target)")
    ax.set_ylabel("ECE increase in transfer (target $-$ source)")
    ax.margins(x=0.13, y=0.15)
    ax.tick_params(labelsize=10, length=3.5, width=0.8, color="#444444")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  scritto {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    vals = paired_losses(load_raw(args.prefix))
    print(f"fig3_dissociation da {args.prefix}_raw.csv\n")
    print(f"  {'regime':<18}{'AUROC loss (95% CI)':>24}{'ECE loss (95% CI)':>24}{'seed':>7}")
    for key, label, _c in REGIMES:
        if key not in vals:
            print(f"  {label:<18}{'assente':>24}")
            continue
        v = vals[key]
        print(f"  {label:<18}{v['auroc_loss']:>+15.4f} ±{v['auroc_ci']:.4f}"
              f"{v['ece_loss']:>+15.4f} ±{v['ece_ci']:.4f}{v['n_seeds']:>7}")
    print()
    make_figure(vals, os.path.join(args.outdir, "fig3_dissociation"))


if __name__ == "__main__":
    main()
