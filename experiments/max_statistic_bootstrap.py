"""
Bootstrap max-statistic test on the real cohorts: does the BEST method beat ERM?

Companion to experiments/max_statistic_test.py, which does the synthetic case by
exact sign-flip permutation over seeds. Here the uncertainty that matters is
target-cohort sampling, so the null is built from the patient-level bootstrap.

Why the per-replicate file is needed
------------------------------------
The `_bootstrap` summary keeps only per-method margins (estimate, CI, p). The
maximum over methods is not a function of the margins: the same resample is
shared by every method, so their gaps are strongly correlated, and ignoring that
correlation would make the test conservative in exactly the way we are trying to
avoid. `run_real_data.py` therefore also writes `<prefix>_bootreps.csv` with one
row per (replicate, method); this script consumes those.

Since the switch to the per-seed bootstrap, a "replicate" is a
(bootstrap resample, training seed) cell rather than a bootstrap resample
alone, so `b` is a composite index and the file is n_boot * n_seeds rows per
method. Nothing here changes: the test only needs each `b` to be shared by
every method (it is -- the same patients and the same seed), and the gaps to
be formed within a cell (they are). The cells are not mutually independent,
since n_seeds of them share a patient resample; that affects how precisely the
null quantile is estimated, not the validity of the comparison, because the
statistic is standardised by the per-cell SD rather than by a standard error
that would need an effective sample size.

Test
----
For method m with bootstrap gaps d[m,b]:
  observed   t_m   = mean_b d[m,b] / sd_b d[m,b]
  null       t~m,b = (d[m,b] - mean_b d[m,b]) / sd_b d[m,b]
Centring imposes the null "no method differs from ERM" while preserving the
joint distribution across methods. For each replicate take the max over methods;
the p-value is the fraction of replicates whose null max reaches the observed
max. Positive gaps always mean "better than ERM" (AUROC: method - ERM;
ECE: ERM - method), so the test is one-sided by construction.

Pooling: transfers are independent bootstraps, so averaging the per-transfer
gaps within a replicate index gives a valid bootstrap distribution of the pooled
(unweighted) effect. Replicate indices are intersected across transfers first.

Run from repo root, after re-running run_real_data.py with the patched code:
    python -m experiments.max_statistic_bootstrap --prefix i3lung
    python -m experiments.max_statistic_bootstrap --prefix i3lung_loco
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")

ERM = "lr_erm"
# Positive = better than ERM for both, by the sign convention in run_real_data.
FIELDS = {"d_auroc": "discrimination (AUROC gap)",
          "d_ece": "calibration (ECE gain)"}

# Prior adjustment and Platt/isotonic scaling are monotone transformations of the
# ERM scores, so their AUROC gap is exactly zero by construction. Including them
# in the discrimination test is not conservative, it is meaningless: a constant
# zero has no sampling distribution, and it wins the max whenever every real
# method is negative. They are compared on calibration only.
RANK_PRESERVING = {"prior_recalib", "platt_scaling", "isotonic"}
EXCLUDE = {"d_auroc": RANK_PRESERVING, "d_ece": set()}


def load_bootreps(path: str) -> dict[str, dict[int, dict[str, float]]]:
    """method -> replicate index -> {d_auroc, d_ece}"""
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["method"]][int(r["b"])] = {
                "d_auroc": float(r["d_auroc"]), "d_ece": float(r["d_ece"])
            }
    return out


def max_stat(D: np.ndarray, names: list[str]):
    """D is (n_methods, n_replicates) of gaps. Returns best, observed t, p."""
    mean = D.mean(axis=1)
    sd = D.std(axis=1, ddof=1)
    sd = np.where(sd < 1e-12, np.inf, sd)          # a constant method cannot win
    t_obs = mean / sd
    k = int(t_obs.argmax())

    null = (D - mean[:, None]) / sd[:, None]        # centred, joint structure kept
    null_max = null.max(axis=0)                     # per replicate
    p = float((null_max >= t_obs[k]).mean())
    return names[k], float(t_obs[k]), float(mean[k]), p


def analyse(prefix: str) -> None:
    paths = sorted(glob.glob(os.path.join(TABLES, f"{prefix}*_bootreps.csv")))
    if not paths:
        raise SystemExit(
            f"nessun file {prefix}*_bootreps.csv in {TABLES}.\n"
            "Rilancia run_real_data.py con il codice aggiornato: le repliche "
            "bootstrap non venivano salvate prima della patch."
        )

    per_transfer = {os.path.basename(p).replace("_bootreps.csv", ""): load_bootreps(p)
                    for p in paths}
    print(f"transfer trovati: {len(per_transfer)}")

    for field, label in FIELDS.items():
        print(f"\n{'='*74}\n{label}\n{'='*74}")
        print(f"{'transfer':<34}{'best':>16}{'gap':>10}{'p':>10}")

        pooled: dict[str, dict[int, float]] = defaultdict(dict)
        for name, bm in sorted(per_transfer.items()):
            methods = sorted(m for m in bm
                             if m != ERM and m not in EXCLUDE[field])
            if not methods:
                continue
            common = set.intersection(*(set(bm[m]) for m in methods))
            common = sorted(common)
            D = np.array([[bm[m][b][field] for b in common] for m in methods])
            best, _, gap, p = max_stat(D, methods)
            star = "*" if p < 0.05 else " "
            print(f"{name:<34}{best:>16}{gap:>+10.4f}{p:>9.4f}{star}")
            for m, row in zip(methods, D):
                for b, v in zip(common, row):
                    pooled[m][b] = pooled[m].get(b, 0.0) + v

        # Pooled across transfers: average per replicate index, then the same test.
        if len(per_transfer) > 1 and pooled:
            methods = sorted(pooled)
            common = sorted(set.intersection(*(set(pooled[m]) for m in methods)))
            n_t = len(per_transfer)
            D = np.array([[pooled[m][b] / n_t for b in common] for m in methods])
            best, _, gap, p = max_stat(D, methods)
            star = "*" if p < 0.05 else " "
            print(f"{'POOLED (media sui transfer)':<34}{best:>16}{gap:>+10.4f}{p:>9.4f}{star}")

    print("\n  '*' = p < 0.05. Test unilaterale: positivo significa meglio di ERM.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="i3lung",
                    help="prefisso dei file <prefix>*_bootreps.csv")
    analyse(ap.parse_args().prefix)


if __name__ == "__main__":
    main()
