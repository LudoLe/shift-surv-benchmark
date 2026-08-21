"""Can a deployment estimate the target prevalence gap before it has labels?

The question
------------
Across the thirty I3LUNG transfers the gap in outcome prevalence between source
and target tracks how miscalibrated ERM arrives (rho = 0.50), while the
covariate distance between the two cohorts carries no association with it
(rho = -0.15). That is an uncomfortable pair of facts, because computing the
prevalence gap needs the target outcomes, which is exactly what a team planning
a transfer does not have.

There is one way out. The EM procedure of Saerens et al. estimates the target
class prior from the source model's predictions on UNLABELED target covariates.
It is already running inside prior recalibration, and its estimate is discarded.
If that estimate tracks the true gap, the diagnostic that predicts calibration
failure becomes available before deployment, and the practical claim of the
paper changes shape.

If it does not track, the negative result is worth the same page: the quantity
that predicts transfer calibration cannot be estimated from what a deployment
can see, by the one method designed to estimate it.

What this measures
------------------
For every ordered pair of centers: fit ERM on the source exactly as the main
experiment does, apply it to the target covariates, run the EM to convergence,
and compare the estimated target prevalence against the true one. Estimates are
averaged over the same ten seeds, so the split and the fit match the numbers
reported elsewhere.

Usage
-----
  python -m experiments.prior_estimability --data_dir $I3LUNG_DATA
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


def _saerens_em(p_raw: np.ndarray, pi_src: float, n_iter: int = 50) -> float:
    """Target prior implied by the source model's predictions on target x.

    Identical to the loop inside prior recalibration, kept separate here so the
    estimate can be read without also applying the correction.
    """
    pi_tgt = float(p_raw.mean())
    for _ in range(n_iter):
        num = p_raw * (pi_tgt / pi_src)
        den = num + (1.0 - p_raw) * ((1.0 - pi_tgt) / (1.0 - pi_src))
        pi_new = float((num / np.clip(den, 1e-12, None)).mean())
        if abs(pi_new - pi_tgt) < 1e-7:
            return pi_new
        pi_tgt = pi_new
    return pi_tgt


def _one_transfer(Xs, ys, Xt, seed: int) -> float:
    """One seed: the pipeline of experiments.run_real_data._run_seed, up to the
    point where the EM estimate is available."""
    X_tr, _X_val, y_tr, _y_val = train_test_split(
        Xs, ys, test_size=0.2, random_state=seed, stratify=ys)
    imp = SimpleImputer(strategy="median").fit(X_tr)
    sca = StandardScaler().fit(imp.transform(X_tr))
    lr = LogisticRegression(max_iter=500, solver="lbfgs",
                            C=hparams.get("erm"), random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(sca.transform(imp.transform(X_tr)), y_tr)
    p_raw = lr.predict_proba(sca.transform(imp.transform(Xt)))[:, 1]
    return _saerens_em(p_raw, float(y_tr.mean()))


def _spearman(a, b) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A = np.array([ra[v] for v in a], float); A -= A.mean()
    B = np.array([rb[v] for v in b], float); B -= B.mean()
    den = (A @ A) ** 0.5 * (B @ B) ** 0.5
    return float(A @ B / den) if den else float("nan")


def run(data_dir: str, n_seeds: int = 10) -> list[dict]:
    from experiments.run_real_data import load_i3lung_all

    centers, feats = load_i3lung_all(data_dir)
    names = sorted(centers)
    data = {c: (centers[c][0][feats].values.astype(float),
                np.asarray(centers[c][1]).astype(int)) for c in names}

    rows = []
    for s, t in itertools.permutations(names, 2):
        Xs, ys = data[s]
        Xt, yt = data[t]
        if len(np.unique(ys)) < 2:
            continue
        est = [_one_transfer(Xs, ys, Xt, seed) for seed in range(n_seeds)]
        pi_s, pi_t = float(ys.mean()), float(yt.mean())
        pi_hat = float(np.mean(est))
        rows.append({
            "source": s, "target": t,
            "pi_source":        round(pi_s, 4),
            "pi_target_true":   round(pi_t, 4),
            "pi_target_est":    round(pi_hat, 4),
            "pi_target_est_sd": round(float(np.std(est)), 4),
            "gap_true":         round(abs(pi_s - pi_t), 4),
            "gap_est":          round(abs(pi_s - pi_hat), 4),
            "error":            round(pi_hat - pi_t, 4),
        })

    print(f"\n{'='*84}\n  STIMA DELLA PREVALENZA DEL TARGET SENZA ETICHETTE  "
          f"({len(rows)} transfer, {n_seeds} semi)\n{'='*84}")
    print(f"  {'transfer':<16}{'pi sorgente':>12}{'pi target':>11}{'stimata':>10}"
          f"{'errore':>9}{'gap vero':>10}{'gap stim.':>11}")
    for r in sorted(rows, key=lambda r: -r["gap_true"]):
        print(f"  {r['source']+'->'+r['target']:<16}{r['pi_source']:>12.3f}"
              f"{r['pi_target_true']:>11.3f}{r['pi_target_est']:>10.3f}"
              f"{r['error']:>+9.3f}{r['gap_true']:>10.3f}{r['gap_est']:>11.3f}")

    gt = [r["gap_true"] for r in rows]
    ge = [r["gap_est"] for r in rows]
    err = np.array([r["error"] for r in rows])
    print(f"\n  Spearman fra gap vero e gap stimato : {_spearman(gt, ge):+.2f}")
    print(f"  errore medio                        : {err.mean():+.4f}")
    print(f"  errore assoluto mediano             : {np.median(np.abs(err)):.4f}")
    print(f"  intervallo del gap vero             : "
          f"{min(gt):.3f} - {max(gt):.3f}")
    print("\n  Se la correlazione e' alta, la quantita' che predice la scalibratura")
    print("  e' disponibile prima del deployment. Se e' bassa, non lo e'.")

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / "i3lung_prior_estimability.csv")
    print(f"\n  Scritto: {_TABLES}/i3lung_prior_estimability.csv\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    a = ap.parse_args()
    run(a.data_dir, n_seeds=a.n_seeds)
