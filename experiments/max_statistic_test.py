"""
Exact max-statistic permutation test for "does the best method beat ERM?".

Why this and not Bonferroni. Bonferroni controls the family-wise error over K
*pre-specified* comparisons. Reporting the best method is a different, post-hoc
hypothesis, and Bonferroni also assumes the worst case over the dependence
between methods -- but here the methods are fit on the same data and are
strongly correlated, so it is badly over-conservative. The max-statistic builds
the null distribution of the maximum directly, so it accounts for both the
selection and the correlation.

Construction. For method m and seed s, take the paired difference
d[m,s] = AUROC(m) - AUROC(ERM) on the target. Under the null that no method
differs from ERM, the paired differences are symmetric about zero, so flipping
their signs leaves the distribution unchanged. With S seeds there are 2^S sign
patterns; at S=10 that is 1024, so the null is enumerated EXACTLY rather than
sampled. The statistic is the studentised mean, and the test compares the
observed max over methods against the null distribution of that max.

A null result from this test is stronger than a null from Bonferroni: the test
is more powerful, so failing to reject is harder to attribute to a conservative
correction.

Run from repo root:
    python -m experiments.max_statistic_test
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_TABLES = os.path.join(ROOT, "outputs", "tables")
PRIOR_TABLES = os.path.expanduser("~/Downloads/shift-surv-outputs/tables")

ERM = "all_observed"
ADAPT = ["coral", "dann", "cdan", "iw_erm"]          # discrimination families
RECAL = ["prior_recalib", "platt_scaling", "isotonic"]  # calibration families

SCEN_LABEL = {
    "covariate_shift": "Covariate", "observational_shift": "Measurement",
    "spurious_shift": "Spurious", "baseline_hazard_shift": "Baseline hazard",
    "prior_shift": "Label-prior", "concept_shift": "Concept", "no_shift": "No shift",
}


def load_raw(path: str) -> dict:
    """scenario -> method -> seed -> row"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["scenario"]][r["method"]][int(r["seed"])] = r
    return out


def paired(bysm: dict, method: str, field: str, better: str):
    """Per-seed difference, signed so that positive always means 'better'."""
    if method not in bysm or ERM not in bysm:
        return None
    seeds = sorted(set(bysm[method]) & set(bysm[ERM]))
    if len(seeds) < 3:
        return None
    m = np.array([float(bysm[method][s][field]) for s in seeds])
    e = np.array([float(bysm[ERM][s][field]) for s in seeds])
    d = (m - e) if better == "high" else (e - m)   # AUROC: higher better; ECE: lower
    return d


def studentised(d: np.ndarray) -> float:
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(d.mean() / (sd / np.sqrt(len(d))))


def max_stat_test(diffs: dict[str, np.ndarray]):
    """Exact sign-flip permutation test on the max studentised mean."""
    names = list(diffs)
    D = np.vstack([diffs[m] for m in names])         # (M, S)
    obs = np.array([studentised(d) for d in D])
    obs_max, best = obs.max(), names[int(obs.argmax())]

    S = D.shape[1]
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=S)))   # (2^S, S)
    null = np.empty(len(signs))
    for i, eps in enumerate(signs):
        F = D * eps
        mean = F.mean(axis=1)
        sd = F.std(axis=1, ddof=1)
        t = np.where(sd < 1e-12, 0.0, mean / (sd / np.sqrt(S)))
        null[i] = t.max()
    p_max = float((null >= obs_max).sum() / len(signs))

    # Per-method paired t-tests + Bonferroni, for comparison.
    p_raw = {m: float(stats.ttest_1samp(diffs[m], 0.0).pvalue) for m in names}
    p_bonf = min(1.0, p_raw[best] * len(names))
    return dict(best=best, obs_max=obs_max, mean_gap=float(diffs[best].mean()),
                p_maxstat=p_max, p_bonferroni=p_bonf, n_methods=len(names),
                n_perm=len(signs))


def run(metric_field: str, better: str, methods: list[str], title: str, data: dict,
        metric_name: str = "") -> list[dict]:
    """Print the table and return one row per regime, so the result can be
    written to disk. Previously this only printed: the figures could not read
    the significance and the stars had to be typed into the .tex by hand, which
    is a claim no re-run can invalidate."""
    rows: list[dict] = []
    print(f"\n{'='*76}\n{title}\n{'='*76}")
    print(f"{'regime':<17}{'best':>15}{'mean gap':>11}{'max-stat p':>12}{'Bonf. p':>10}")
    for scen in ["covariate_shift", "observational_shift", "spurious_shift",
                 "baseline_hazard_shift", "prior_shift", "concept_shift"]:
        if scen not in data:
            continue
        diffs = {}
        for m in methods:
            d = paired(data[scen], m, metric_field, better)
            if d is not None:
                diffs[m] = d
        if not diffs:
            continue
        r = max_stat_test(diffs)
        rows.append({
            "scenario": scen, "metric": metric_name, "best": r["best"],
            "mean_gap": round(r["mean_gap"], 5),
            # The figure generators read `p_value`; keeping the max-statistic
            # one under that name means a figure can never be starred from the
            # per-method Bonferroni p, which would ignore the post-hoc selection.
            "p_value": round(r["p_maxstat"], 5),
            "p_bonferroni": round(r["p_bonferroni"], 5),
            "n_methods": r["n_methods"], "n_perm": r["n_perm"],
        })
        star = "*" if r["p_maxstat"] < 0.05 else " "
        print(f"{SCEN_LABEL.get(scen, scen):<17}{r['best']:>15}"
              f"{r['mean_gap']:>+11.4f}{r['p_maxstat']:>11.4f}{star}{r['p_bonferroni']:>10.4f}")
    print(f"\n  exact enumeration of 2^10 = 1024 sign patterns; "
          f"'*' marks max-statistic p < 0.05")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(REPO_TABLES, "benchmark_1s1t_iwcv_raw.csv"))
    ap.add_argument("--prior_raw", default=os.path.join(PRIOR_TABLES, "benchmark_1s1t_prior_raw.csv"))
    ap.add_argument("--out", default=None, help="CSV di output (default: <prefix>_maxstat.csv)")
    args = ap.parse_args()

    data = load_raw(args.raw)
    if os.path.exists(args.prior_raw):
        data.update(load_raw(args.prior_raw))
    else:
        print(f"note: {args.prior_raw} not found -> label-prior omitted")

    rows = run("tgt_auroc", "high", ADAPT,
               "DISCRIMINATION  --  best adaptation method vs ERM (target AUROC)",
               data, metric_name="auroc")
    rows += run("tgt_ece", "low", RECAL + ADAPT,
                "CALIBRATION  --  best method vs ERM (target ECE, lower is better)",
                data, metric_name="ece")

    # Named after the raw file it was computed from, so the stars in a figure
    # can always be traced to the run that produced them.
    prefix = os.path.basename(args.raw).replace("_raw.csv", "")
    out = args.out or os.path.join(REPO_TABLES, f"{prefix}_maxstat.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  scritto {out}  ({len(rows)} righe)")


if __name__ == "__main__":
    main()
