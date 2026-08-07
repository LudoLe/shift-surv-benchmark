"""fig_disc_a / fig_disc_b — discrimination, per shift regime.

Replaces the figures that had no generator in the repo: fig_disc_{a,b}.pdf
existed only as PDFs, so a re-run could not update them and nothing would
report the mismatch.

Layout (one group per regime):
  light grey bar   observed-data target oracle -- what x predicts in the target
  dark tick        latent oracle -- the biological reference on u_inv
  coloured bars    ERM, and the best method of the panel's family

The two references answer different questions and are drawn differently on
purpose. The gap between them is information unavailable in the observed
covariates, which no method can recover; the gap between the observed-data
oracle and a method is what that method left unrecovered from x. Only the
latter supports a claim about adaptation failing, so it is the one the bars
line up against.

Panel a: single source, best of IW-ERM / CORAL / DANN / CDAN  (benchmark_1s1t_iwcv)
Panel b: three sources, best of IRM / GroupDRO                (benchmark_3s1t)

Significance stars are read from outputs/tables/<prefix>_maxstat.csv when
present (written by experiments/max_statistic_test.py). Absent, the bars are
drawn without stars rather than with hand-entered ones, so the figure can never
assert a significance the current run does not support.

Run from repo root:
    python paper_assets/_gen_fig_disc.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")

SCENARIOS = [
    ("covariate_shift",       "Covariate"),
    ("observational_shift",   "Measurement"),
    ("spurious_shift",        "Spurious"),
    ("baseline_hazard_shift", "Baseline hazard"),
    ("prior_shift",           "Label-prior"),
    ("concept_shift",         "Concept"),
]

ERM = "all_observed"
LATENT_ORACLE = "invariant_oracle"
OBSERVED_ORACLE = "observed_target_oracle"

# prior_shift lives in APPENDIX_SCENARIOS, so run_adaptation's main run does not
# emit it and it has a dedicated prefix. Reading it from there keeps the figure
# on six regimes; without this the panel silently drops to five, which is how
# the omission was found.
PRIOR_PREFIX = {"a": "benchmark_1s1t_prior", "b": "benchmark_3s1t_prior"}

PANELS = {
    "a": dict(prefix="benchmark_1s1t_iwcv",
              family=["iw_erm", "coral", "dann", "cdan"],
              family_label="Best adaptation",
              colour="#D95F0E"),
    "b": dict(prefix="benchmark_3s1t",
              family=["irm", "group_dro"],
              family_label="Best generalization",
              colour="#1B9E77"),
}

C_ERM = "#1F6FB4"
C_OBS_ORACLE = "#CFCFCF"
C_LAT_ORACLE = "#4D4D4D"

from _style import apply as _apply_style, finish as _finish_axes
_apply_style()


def load_summary(prefix: str) -> dict:
    path = os.path.join(TABLES, f"{prefix}_summary.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"manca {path}\n"
            f"  Rigenera con: python -m experiments.run_adaptation --out_prefix {prefix}"
        )
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r["scenario"], r["method"])] = r
    return out


def load_raw(prefix: str, field: str = "tgt_auroc") -> dict:
    """(scenario, method) -> list of per-seed values of `field`, for the dots.

    Parameterised because _gen_fig_cal.py needs tgt_ece from the same tables.
    """
    path = os.path.join(TABLES, f"{prefix}_raw.csv")
    out = defaultdict(list)
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                out[(r["scenario"], r["method"])].append(float(r[field]))
            except (KeyError, ValueError):
                continue
    return out


def load_stars(prefix: str, metric: str = "auroc") -> dict:
    """(scenario) -> star string, from the max-statistic test if it was run.

    Filtering on `metric` is not optional: the CSV holds both the AUROC and the
    ECE test, and without it whichever row came last would win -- this figure
    would be starred from the calibration test and fig_cal from the
    discrimination one.
    """
    path = os.path.join(TABLES, f"{prefix}_maxstat.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("metric", metric) != metric:
                continue
            try:
                p = float(r["p_value"])
            except (KeyError, ValueError):
                continue
            star = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "n.s.")
            # The tested method is carried alongside the star. Without it a
            # figure can only guess which bar the result belongs to, and the
            # guess is wrong whenever the winning family is not the one the
            # bracket was hard-coded to span.
            out[r.get("scenario", "")] = (star, r.get("best", ""))
    return out


def star_of(stars: dict, key: str) -> str:
    return stars[key][0] if key in stars else ""


def tested_best(stars: dict, key: str) -> str:
    return stars[key][1] if key in stars else ""


def _f(row: dict | None, key: str) -> float:
    if not row:
        return float("nan")
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def best_of(summary: dict, scenario: str, family: list[str]) -> str | None:
    """Best method of the family by mean target AUROC. Post-hoc by construction;
    inference about it comes from the max-statistic test, not from these bars."""
    cands = [(m, _f(summary.get((scenario, m)), "tgt_auroc_mean")) for m in family]
    cands = [(m, v) for m, v in cands if v == v]
    return max(cands, key=lambda t: t[1])[0] if cands else None


def draw_panel(ax, panel_key: str) -> None:
    cfg = PANELS[panel_key]
    summary = load_summary(cfg["prefix"])
    raw = load_raw(cfg["prefix"])
    stars = load_stars(cfg["prefix"])

    # Merge in the label-prior regime from its dedicated run.
    prior_prefix = PRIOR_PREFIX[panel_key]
    if os.path.exists(os.path.join(TABLES, f"{prior_prefix}_summary.csv")):
        summary.update(load_summary(prior_prefix))
        raw.update(load_raw(prior_prefix))
        stars.update(load_stars(prior_prefix))
    else:
        print(f"  ATTENZIONE: manca {prior_prefix}_summary.csv — "
              f"il pannello {panel_key} esce senza il regime label-prior")

    scen = [(k, lbl) for k, lbl in SCENARIOS if (k, ERM) in summary]
    if not scen:
        raise SystemExit(f"nessuno scenario atteso in {cfg['prefix']}_summary.csv")

    x = np.arange(len(scen), dtype=float)
    w = 0.26

    for i, (key, _lbl) in enumerate(scen):
        obs = _f(summary.get((key, OBSERVED_ORACLE)), "tgt_auroc_mean")
        lat = _f(summary.get((key, LATENT_ORACLE)), "tgt_auroc_mean")

        # Reference bar: what x can predict in the target.
        if obs == obs:
            ax.bar(x[i], obs, width=w * 3.1, color=C_OBS_ORACLE, zorder=1,
                   edgecolor="none")
        # The latent oracle is a *level*, not a quantity a bar could reach, so it
        # is drawn as a dashed rule. A solid thick tick read as the top edge of a
        # bar, and under concept shift -- where it falls below the observed-data
        # oracle -- as a stray line inside one.
        if lat == lat:
            ax.plot([x[i] - w * 1.62, x[i] + w * 1.62], [lat, lat],
                    color=C_LAT_ORACLE, lw=1.3, ls=(0, (4, 2)), zorder=5,
                    solid_capstyle="butt")

        best = best_of(summary, key, cfg["family"])
        for j, (m, colour) in enumerate([(ERM, C_ERM), (best, cfg["colour"])]):
            if m is None:
                continue
            row = summary.get((key, m))
            mean, ci = _f(row, "tgt_auroc_mean"), _f(row, "tgt_auroc_ci95")
            if mean != mean:
                continue
            pos = x[i] + (j - 0.5) * w * 1.15
            ax.bar(pos, mean, width=w, color=colour, zorder=2, edgecolor="none")
            if ci == ci:
                ax.errorbar(pos, mean, yerr=ci, fmt="none", ecolor="#333333",
                            elinewidth=1.0, capsize=2.5, zorder=5)
            pts = raw.get((key, m), [])
            if pts:
                jit = (np.random.default_rng(i * 10 + j).random(len(pts)) - 0.5) * w * 0.55
                ax.scatter(pos + jit, pts, s=13, facecolors="none",
                           edgecolors=colour, linewidths=0.8, alpha=0.85,
                           zorder=6)

        # The bracket spans the two bars the test actually compared. A label
        # floating over the whole group reads as a statement about the oracle
        # bars too, which no test was run on.
        if key in stars:
            x0, x1 = x[i] - 0.5 * w * 1.15, x[i] + 0.5 * w * 1.15
            tops = []
            for m in (ERM, best):
                if m is None:
                    continue
                row = summary.get((key, m))
                mn, c = _f(row, "tgt_auroc_mean"), _f(row, "tgt_auroc_ci95")
                cand = [mn + (c if c == c else 0.0)]
                pts = raw.get((key, m), [])
                if pts:
                    cand.append(max(pts))
                tops.append(np.nanmax(cand))
            yb = (np.nanmax(tops) if tops else 0.8) + 0.013
            ax.plot([x0, x0, x1, x1], [yb - 0.007, yb, yb, yb - 0.007],
                    color="#555555", lw=0.9, zorder=7, solid_capstyle="butt")
            # The test names whichever method attains the largest STUDENTISED
            # gap; `best` is the largest mean. They usually agree here and did in
            # the run this figure was written for, but where they do not the
            # bracket would silently credit the drawn bar with another method's
            # p-value -- the error fig_cal was fixed for. Naming the winner when
            # it differs keeps the annotation attributable.
            # Name off the panel, into the caption: see _gen_fig_cal.py.
            tested = tested_best(stars, key)
            if tested and tested != best:
                _caption_notes.append((key, tested, star_of(stars, key)))
            ax.text(0.5 * (x0 + x1), yb + 0.004, star_of(stars, key),
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _k, lbl in scen], rotation=30, ha="right")
    ax.set_ylabel("Target AUROC")
    # 0.92 rather than 0.90: the legend needs clearance over the baseline-hazard
    # group, whose latent oracle reaches 0.852. Ticks stop at 0.90 so the axis
    # still reads in fives.
    ax.set_ylim(0.5, 0.92)
    ax.set_yticks(np.arange(0.50, 0.901, 0.05))
    ax.grid(axis="y", color="#E8E8E8", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ax.legend(handles=[
        Patch(facecolor=C_OBS_ORACLE, label="Observed-data target oracle"),
        Line2D([0], [0], color=C_LAT_ORACLE, lw=1.3, ls=(0, (4, 2)),
               label="Latent oracle"),
        Patch(facecolor=C_ERM, label="ERM"),
        Patch(facecolor=cfg["colour"], label=cfg["family_label"]),
    ], loc="upper left", frameon=False, fontsize=7.5, ncol=2,
       handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
       borderaxespad=0.3)



# Regimes whose starred method has no bar of its own, collected while drawing so
# the caption can name it. The name is deliberately not on the panel.
_caption_notes: list = []


def _print_caption_notes(what: str) -> None:
    if not _caption_notes:
        return
    LABEL = {"covariate_shift": "covariate", "observational_shift": "measurement",
             "spurious_shift": "spurious", "baseline_hazard_shift": "baseline-hazard",
             "prior_shift": "label-prior", "concept_shift": "concept"}
    PRETTY = {"dann": "DANN", "cdan": "CDAN", "coral": "CORAL", "iw_erm": "IW-ERM",
              "irm": "IRM", "group_dro": "GroupDRO",
              "prior_recalib": "prior adjustment", "platt_scaling": "Platt scaling",
              "isotonic": "isotonic"}
    parts = [f"{PRETTY.get(m, m)} under {LABEL.get(k, k)} shift ({s})"
             for k, m, s in _caption_notes]
    print(f"\n  RIGA DA AGGIUNGERE ALLA DIDASCALIA di {what}:")
    print("    Stars mark the max-statistic test over the full comparison set; "
          "the method attaining it is " + "; ".join(parts) +
          ", which is not drawn as a separate bar.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for panel in ("a", "b"):
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        draw_panel(ax, panel)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out = os.path.join(args.outdir, f"fig_disc_{panel}.{ext}")
            fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  scritto {args.outdir}/fig_disc_{panel}.{{pdf,png}}")
    _print_caption_notes("fig:disc")


if __name__ == "__main__":
    main()
