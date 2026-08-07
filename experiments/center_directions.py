"""Pairwise cosine between the decision directions each center's model learns.

The question
------------
One I3LUNG center transfers uniformly poorly as a source while behaving normally
as a target, and neither label-free diagnostic flags it: its covariate distance
to the others is below the cohort mean and its prevalence gap close to it. Three
explanations are already excluded by the summary tables -- an uninformative
source model (its in-source AUROC is in line with the others), too few patients
(it is not the smallest cohort), and a symmetric change in P(Y|X) (models fitted
elsewhere work on its patients). What remains is that the direction its model
learns differs from the direction the others agree on. This measures that
directly.

Why a noise floor is part of the measurement
--------------------------------------------
A low cosine between two coefficient vectors can mean they point different ways
or simply that both are imprecisely estimated: at a few hundred patients and
eleven covariates, a logistic-regression coefficient vector carries real
sampling noise, and two independent fits to the SAME distribution will not agree
perfectly either. Reporting a low cosine without that reference invites the
obvious objection and deserves it.

So each center is also split in half and fitted twice, and the cosine between
its own two halves is reported alongside. That is the agreement achievable at
that sample size when the underlying direction is identical by construction. A
between-center cosine only carries information relative to it.

The comparison that matters is not "is the cosine low" but "is one center's
agreement with the others lower than theirs with each other", since all centers
carry comparable noise.

Usage
-----
  python -m experiments.center_directions --data_dir $I3LUNG_DATA
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments import hparams
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"


def _fit_direction(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray | None:
    """Coefficient vector of the same model ERM uses, on standardized features.

    Standardization is per fit, so the direction is expressed in units of that
    center's own covariate spread. Comparing raw coefficients across centers
    would confound a difference in direction with a difference in scale.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
        return None
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    m = LogisticRegression(max_iter=500, solver="lbfgs",
                           C=hparams.get("erm"), random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        m.fit(Xs, y)
    v = m.coef_.ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def run(data_dir: str, n_splits: int = 20, seed: int = 0) -> dict:
    from experiments.run_real_data import load_i3lung_all

    centers, _feats = load_i3lung_all(data_dir)
    names = sorted(centers)

    # One imputer over the pooled cohort: the question is whether the fitted
    # directions differ, and a per-center imputer would add a second reason for
    # them to.
    Xall = np.vstack([centers[c][0].values for c in names])
    imp = SimpleImputer(strategy="median").fit(Xall)
    data = {c: (imp.transform(centers[c][0].values),
                np.asarray(centers[c][1]).astype(int)) for c in names}

    beta = {}
    for c in names:
        v = _fit_direction(*data[c], seed=seed)
        if v is not None:
            beta[c] = v
    if len(beta) < 2:
        raise SystemExit("meno di due centri utilizzabili")

    # Noise floor: two independent halves of the same centre, so the underlying
    # direction is identical and only sampling separates them. Averaged over
    # splits, since a single split is itself noisy.
    floor = {}
    for c in names:
        X, y = data[c]
        cs = []
        for k in range(n_splits):
            try:
                Xa, Xb, ya, yb = train_test_split(
                    X, y, test_size=0.5, random_state=seed + k, stratify=y)
            except ValueError:
                continue
            va, vb = _fit_direction(Xa, ya, seed), _fit_direction(Xb, yb, seed)
            if va is not None and vb is not None:
                cs.append(_cos(va, vb))
        floor[c] = float(np.mean(cs)) if cs else float("nan")

    rows = [{"pair": f"{a}-{b}", "cosine": round(_cos(beta[a], beta[b]), 4),
             "n_a": int(len(data[a][1])), "n_b": int(len(data[b][1]))}
            for a, b in itertools.combinations(sorted(beta), 2)]

    print(f"\n{'='*72}\n  DIREZIONI DECISIONALI PER CENTRO  ({len(beta)} centri)\n{'='*72}")
    print("\n  pavimento di rumore: coseno fra due meta' dello STESSO centro")
    print(f"  {'centro':<10}{'n':>6}{'coseno interno':>17}")
    for c in names:
        print(f"  {c:<10}{len(data[c][1]):>6}{floor.get(c, float('nan')):>17.3f}")

    print("\n  coseno fra centri diversi")
    for r in sorted(rows, key=lambda r: r["cosine"]):
        print(f"  {r['pair']:<16}{r['cosine']:>8.3f}")

    print(f"\n  {'centro':<10}{'coseno medio con gli altri':>28}"
          f"{'pavimento':>12}{'rapporto':>11}")
    summary = []
    for c in sorted(beta):
        others = [_cos(beta[c], beta[d]) for d in beta if d != c]
        m = float(np.mean(others))
        f = floor.get(c, float("nan"))
        ratio = m / f if f == f and f > 0 else float("nan")
        summary.append({"center": c, "mean_cosine_to_others": round(m, 4),
                        "self_split_cosine": round(f, 4) if f == f else "",
                        "ratio": round(ratio, 4) if ratio == ratio else "",
                        "n": int(len(data[c][1]))})
        print(f"  {c:<10}{m:>28.3f}{f:>12.3f}{ratio:>11.3f}")

    print("\n  Il rapporto e' la quantita' da leggere: un centro la cui direzione")
    print("  concorda con le altre quanto concorda con se' stesso ha rapporto ~1.")
    print("  Un valore nettamente piu' basso degli altri centri indica una")
    print("  direzione diversa, non semplicemente una stima piu' rumorosa.")

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / "center_directions_pairs.csv")
    save_table_csv(summary, _TABLES / "center_directions_summary.csv")
    print(f"\n  Scritto: {_TABLES}/center_directions_{{pairs,summary}}.csv")
    return {"pairs": rows, "summary": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.data_dir, n_splits=a.n_splits, seed=a.seed)
