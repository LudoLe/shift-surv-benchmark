"""fig_cal_a / fig_cal_b — calibration, per shift regime.

Companion to _gen_fig_disc.py, and the second of the two central figures that
existed only as PDFs with no generator anywhere in the repo.

Three coloured bars per regime rather than two, because calibration has a family
of methods of its own: ERM, the best method of the panel's adaptation family,
and the best post-hoc recalibration (prior adjustment or Platt).

  light grey bar   observed-data target oracle -- the ECE reachable from x
  dark tick        latent oracle, un-recalibrated
  coloured bars    ERM, best DA/DG, best recalibration

Note the axis is inverted relative to fig_disc: ECE is an error, so lower is
better and the oracle references are *floors*, not ceilings. A bar below the
grey bar is therefore better than the target's own covariates would allow --
which happens under label-prior shift, where prior adjustment exploits an
assumption the oracle does not use.

Panel a: single source, best of IW-ERM / CORAL / DANN / CDAN  (benchmark_1s1t_iwcv)
Panel b: three sources, best of IRM / GroupDRO                (benchmark_3s1t)
label-prior comes from its dedicated prefix in both, since prior_shift is in
APPENDIX_SCENARIOS and the main benchmark does not emit it.

Run from repo root:
    python paper_assets/_gen_fig_cal.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _gen_fig_disc import (SCENARIOS, ERM, LATENT_ORACLE, OBSERVED_ORACLE,
                           PRIOR_PREFIX, TABLES, OUTDIR, C_ERM, C_OBS_ORACLE,
                           C_LAT_ORACLE, load_summary, load_raw, load_stars,
                           _f, best_of)

RECAL = ["prior_recalib", "platt_scaling", "isotonic"]
C_RECAL = "#E6B422"

PANELS = {
    "a": dict(prefix="benchmark_1s1t_iwcv",
              family=["iw_erm", "coral", "dann", "cdan"],
              family_label="Best adaptation", colour="#D95F0E"),
    "b": dict(prefix="benchmark_3s1t",
              family=["irm", "group_dro"],
              family_label="Best generalization", colour="#1B9E77"),
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def best_recal(summary: dict, scenario: str) -> str | None:
    """Lowest mean target ECE among the recalibration methods."""
    cands = [(m, _f(summary.get((scenario, m)), "tgt_ece_mean")) for m in RECAL]
    cands = [(m, v) for m, v in cands if v == v]
    return min(cands, key=lambda t: t[1])[0] if cands else None


def best_of_ece(summary: dict, scenario: str, family: list[str]) -> str | None:
    """Best of the adaptation/DG family on ECE -- lowest, not highest.

    Deliberately not reusing best_of from _gen_fig_disc: that one maximises
    AUROC, and silently applying it here would label the *worst*-calibrated
    method as best.
    """
    cands = [(m, _f(summary.get((scenario, m)), "tgt_ece_mean")) for m in family]
    cands = [(m, v) for m, v in cands if v == v]
    return min(cands, key=lambda t: t[1])[0] if cands else None


def draw_panel(ax, panel_key: str) -> None:
    cfg = PANELS[panel_key]
    summary = load_summary(cfg["prefix"])
    raw = load_raw(cfg["prefix"], field="tgt_ece")
    stars = load_stars(cfg["prefix"], metric="ece")

    prior_prefix = PRIOR_PREFIX[panel_key]
    if os.path.exists(os.path.join(TABLES, f"{prior_prefix}_summary.csv")):
        summary.update(load_summary(prior_prefix))
        raw.update(load_raw(prior_prefix, field="tgt_ece"))
        stars.update(load_stars(prior_prefix, metric="ece"))
    else:
        print(f"  ATTENZIONE: manca {prior_prefix}_summary.csv — "
              f"il pannello {panel_key} esce senza il regime label-prior")

    scen = [(k, lbl) for k, lbl in SCENARIOS if (k, ERM) in summary]
    if not scen:
        raise SystemExit(f"nessuno scenario atteso in {cfg['prefix']}_summary.csv")

    x = np.arange(len(scen), dtype=float)
    w = 0.21

    for i, (key, _lbl) in enumerate(scen):
        obs = _f(summary.get((key, OBSERVED_ORACLE)), "tgt_ece_mean")
        lat = _f(summary.get((key, LATENT_ORACLE)), "tgt_ece_mean")
        if obs == obs:
            ax.bar(x[i], obs, width=w * 4.0, color=C_OBS_ORACLE, zorder=1,
                   edgecolor="none")
        if lat == lat:
            ax.plot([x[i] - w * 2.0, x[i] + w * 2.0], [lat, lat],
                    color=C_LAT_ORACLE, lw=1.8, zorder=4, solid_capstyle="butt")

        trio = [(ERM, C_ERM),
                (best_of_ece(summary, key, cfg["family"]), cfg["colour"]),
                (best_recal(summary, key), C_RECAL)]
        for j, (m, colour) in enumerate(trio):
            if m is None:
                continue
            row = summary.get((key, m))
            mean, ci = _f(row, "tgt_ece_mean"), _f(row, "tgt_ece_ci95")
            if mean != mean:
                continue
            pos = x[i] + (j - 1) * w * 1.1
            ax.bar(pos, mean, width=w, color=colour, zorder=2, edgecolor="none")
            if ci == ci:
                ax.errorbar(pos, mean, yerr=ci, fmt="none", ecolor="#333333",
                            elinewidth=1.0, capsize=2.0, zorder=5)
            pts = raw.get((key, m), [])
            if pts:
                jit = (np.random.default_rng(i * 10 + j).random(len(pts)) - 0.5) * w * 0.5
                ax.scatter(pos + jit, pts, s=5, color="#222222", alpha=0.45,
                           zorder=6, linewidths=0)

        if key in stars:
            top = np.nanmax([obs, lat, _f(summary.get((key, ERM)), "tgt_ece_mean")])
            ax.text(x[i], top * 1.06, stars[key], ha="center", va="bottom",
                    fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _k, lbl in scen], rotation=18, ha="right")
    ax.set_ylabel("Target ECE  (lower is better)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    ax.legend(handles=[
        Patch(facecolor=C_OBS_ORACLE, label="Observed-data target oracle"),
        Line2D([0], [0], color=C_LAT_ORACLE, lw=1.8, label="Latent oracle"),
        Patch(facecolor=C_ERM, label="ERM"),
        Patch(facecolor=cfg["colour"], label=cfg["family_label"]),
        Patch(facecolor=C_RECAL, label="Best recalibration"),
    ], loc="upper left", frameon=False, fontsize=8, ncol=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for panel in ("a", "b"):
        fig, ax = plt.subplots(figsize=(7.0, 3.1))
        draw_panel(ax, panel)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.outdir, f"fig_cal_{panel}.{ext}"),
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  scritto {args.outdir}/fig_cal_{panel}.{{pdf,png}}")


if __name__ == "__main__":
    main()
