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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.expanduser("~/Desktop/shift-surv-paper-images")

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

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


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


def load_stars(prefix: str) -> dict:
    """(scenario) -> star string, from the max-statistic test if it was run."""
    path = os.path.join(TABLES, f"{prefix}_maxstat.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                p = float(r["p_value"])
            except (KeyError, ValueError):
                continue
            out[r.get("scenario", "")] = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            )
    return out


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
        # Reference tick: the biological ceiling on u_inv, drawn as a rule so it
        # never reads as an achievable operating point.
        if lat == lat:
            ax.plot([x[i] - w * 1.55, x[i] + w * 1.55], [lat, lat],
                    color=C_LAT_ORACLE, lw=1.8, zorder=4, solid_capstyle="butt")

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
                ax.scatter(pos + jit, pts, s=5, color="#222222", alpha=0.45,
                           zorder=6, linewidths=0)

        if key in stars:
            top = np.nanmax([obs, lat, _f(summary.get((key, ERM)), "tgt_auroc_mean")])
            ax.text(x[i], top + 0.022, stars[key], ha="center", va="bottom",
                    fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _k, lbl in scen], rotation=18, ha="right")
    ax.set_ylabel("Target AUROC")
    ax.set_ylim(0.5, 0.95)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    ax.legend(handles=[
        Patch(facecolor=C_OBS_ORACLE, label="Observed-data target oracle"),
        Line2D([0], [0], color=C_LAT_ORACLE, lw=1.8, label="Latent oracle"),
        Patch(facecolor=C_ERM, label="ERM"),
        Patch(facecolor=cfg["colour"], label=cfg["family_label"]),
    ], loc="lower left", frameon=False, fontsize=8, ncol=2)


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
            out = os.path.join(args.outdir, f"fig_disc_{panel}.{ext}")
            fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  scritto {args.outdir}/fig_disc_{panel}.{{pdf,png}}")


if __name__ == "__main__":
    main()
