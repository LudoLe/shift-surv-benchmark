"""
Regenerate the sample-size sweep figure for the paper, as TWO separate panels.

Why two files: every multi-panel figure in the paper takes its "(a)"/"(b)"
from LaTeX subcaption, below the panel and without a panel title. The previous
fig5_nsweep baked bold "a"/"b" into the image at the top left, which is the one
remaining figure in the wrong style. These outputs carry no letter and no title.

  fig_nsweep_a : Best DA minus ERM, target AUROC, versus patients per center.
  fig_nsweep_b : Calibration recovery (ERM ECE minus prior-EM ECE) versus n.

Both are paired per seed: the best adaptation method is selected once per
(scenario, n) by mean target AUROC, then differenced against ERM within each
seed. Confidence intervals are t-based over seeds, matching run_adaptation.py.

Data. The five original regimes come from the repo sweep; label-prior was run
later and its CSVs live in the downloaded server outputs. Point --prior_dir
elsewhere if you move them.

Run from repo root:
    python paper_assets/_gen_fig_nsweep.py
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
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_TABLES = os.path.join(ROOT, "outputs", "tables")
PRIOR_TABLES = os.path.expanduser("~/Downloads/shift-surv-outputs/tables")
OUTDIR = os.path.expanduser("~/Desktop/shift-surv-paper-images")

N_GRID = [250, 500, 1000, 2000]

ERM = "all_observed"
RECAL = "prior_recalib"
# Common adaptation set. iw_erm exists only in the label-prior sweep, so
# including it there and nowhere else would make the curves incomparable.
DA_METHODS = ["coral", "dann", "cdan"]

# Regimes reported in the paper, in the order used everywhere else.
SCENARIOS = [
    ("covariate_shift",       "Covariate",       "#1F6FB4"),
    ("observational_shift",   "Measurement",     "#D95F0E"),
    ("spurious_shift",        "Spurious",        "#1B9E77"),
    ("baseline_hazard_shift", "Baseline hazard", "#D9739F"),
    ("prior_shift",           "Label-prior",     "#56B4E9"),
    ("concept_shift",         "Concept",         "#E6B422"),
]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})


def raw_path(scenario: str, n: int, prior_dir: str) -> str:
    if scenario == "prior_shift":
        return os.path.join(prior_dir, f"n_sweep_iwcv_prior_n{n}_raw.csv")
    return os.path.join(REPO_TABLES, f"n_sweep_iwcv_n{n}_raw.csv")


def load(scenario: str, n: int, prior_dir: str) -> dict[str, dict[int, dict]]:
    """method -> seed -> row, restricted to `scenario`."""
    path = raw_path(scenario, n, prior_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {path}\n"
            f"  label-prior sweeps are not in the repo; pass --prior_dir if they moved."
        )
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["scenario"] != scenario:
                continue
            out[r["method"]][int(r["seed"])] = r
    return out


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (float(values.mean()) if values.size else np.nan, np.nan)
    se = values.std(ddof=1) / np.sqrt(values.size)
    return float(values.mean()), float(stats.t.ppf(0.975, values.size - 1) * se)


def paired(by_method: dict, a: str, b: str, field: str) -> np.ndarray:
    """Per-seed a-minus-b, over the seeds both methods share."""
    if a not in by_method or b not in by_method:
        return np.array([])
    seeds = sorted(set(by_method[a]) & set(by_method[b]))
    return np.array([float(by_method[a][s][field]) - float(by_method[b][s][field])
                     for s in seeds])


def series(prior_dir: str):
    """(scenario, n) -> (auroc_diff_mean, ci, ece_recovery_mean, ci)."""
    res: dict[tuple[str, int], tuple] = {}
    for scenario, _, _ in SCENARIOS:
        for n in N_GRID:
            bm = load(scenario, n, prior_dir)

            # Best adaptation method: chosen once, by mean target AUROC.
            available = [m for m in DA_METHODS if m in bm]
            if not available:
                raise RuntimeError(f"no adaptation method for {scenario} n={n}")
            best = max(
                available,
                key=lambda m: np.mean([float(r["tgt_auroc"]) for r in bm[m].values()]),
            )
            a_mean, a_ci = mean_ci(paired(bm, best, ERM, "tgt_auroc"))

            # Calibration recovery: positive means prior-EM lowered the error.
            e_mean, e_ci = mean_ci(paired(bm, ERM, RECAL, "tgt_ece"))

            res[(scenario, n)] = (a_mean, a_ci, e_mean, e_ci)
    return res


def panel(res, idx_mean, idx_ci, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    x = np.array(N_GRID, dtype=float)

    for scenario, label, color in SCENARIOS:
        mean = np.array([res[(scenario, n)][idx_mean] for n in N_GRID])
        ci = np.array([res[(scenario, n)][idx_ci] for n in N_GRID])
        ax.plot(x, mean, "-o", color=color, label=label, lw=1.9, ms=5.5, zorder=3)
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15, lw=0, zorder=1)

    ax.axhline(0.0, color="0.55", ls="--", lw=1.0, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(N_GRID)
    ax.set_xticklabels([str(n) for n in N_GRID])
    ax.minorticks_off()
    ax.set_xlabel("Patients per center")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{outfile}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default=PRIOR_TABLES,
                    help="directory holding n_sweep_iwcv_prior_n*_raw.csv")
    args = ap.parse_args()

    res = series(args.prior_dir)

    panel(res, 0, 1, "Best DA $-$ ERM (target AUROC)", "fig_nsweep_a")
    panel(res, 2, 3, "Calibration recovery (ERM $-$ prior-EM ECE)", "fig_nsweep_b")

    print("\nvalues used (mean [95% CI]):")
    for scenario, label, _ in SCENARIOS:
        for n in N_GRID:
            a, aci, e, eci = res[(scenario, n)]
            print(f"  {label:<16s} n={n:<5d} dAUROC {a:+.4f} [{aci:.4f}]"
                  f"   ECE recovery {e:+.4f} [{eci:.4f}]")


if __name__ == "__main__":
    main()
