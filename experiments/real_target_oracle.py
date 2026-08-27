"""The target-data oracle on the real cohorts.

Why this exists
---------------
On synthetic data the paper reports two references, and the distance from ERM to
the target-data oracle is the margin a method could still recover. On the real
cohorts only ERM and the compared methods were reported, which leaves the
central claim weaker than it needs to be: "no method improves on ERM" invites
the answer that there was nothing to improve, and without the reference we
cannot tell the two apart.

The reference is computable here. The latent factors are a construct of the
generator and do not exist outside it, so the latent-feature reference has no
real-data counterpart. The target-data oracle only needs the target cohort's own
covariates and its labels, and both are in the historical record.

What it measures
----------------
For each target center: a logistic regression fitted on that center's covariates
and labels, scored by five-fold out-of-fold predictions so the estimate is not
optimistic. It is a reference and never a compared method, since it reads labels
no deployment has at prediction time. The margin H_obs is its target AUROC minus
ERM's, and it answers how much of the transfer loss the target covariates could
have supported.

Usage
-----
  python -m experiments.real_target_oracle --data_dir $I3LUNG_DATA
  python -m experiments.real_target_oracle --dataset rotterdam_gbsg --tau 36
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments import hparams
from utils.io import save_table_csv
from utils.metrics import auroc, ece

_TABLES = _ROOT / "outputs" / "tables"


def oof_oracle(X: np.ndarray, y: np.ndarray, n_folds: int = 5,
               seed: int = 0) -> tuple[float, float]:
    """Out-of-fold AUROC and ECE of a model fitted on this cohort's own labels.

    Out of fold rather than in sample: an in-sample fit would report how well the
    model memorises the cohort, which is an upper bound on nothing useful.
    """
    y = np.asarray(y).astype(int)
    probs = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sca = StandardScaler().fit(imp.transform(X[tr]))
        m = LogisticRegression(max_iter=500, solver="lbfgs",
                               C=hparams.get("erm"), random_state=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            m.fit(sca.transform(imp.transform(X[tr])), y[tr])
        probs[te] = m.predict_proba(sca.transform(imp.transform(X[te])))[:, 1]
    return float(auroc(y, probs)), float(ece(y, probs))


def _erm_by_target(prefix: str) -> dict[str, list[tuple[str, float, float]]]:
    """ERM's target AUROC and ECE for every transfer, grouped by target center."""
    out: dict[str, list[tuple[str, float, float]]] = {}
    for f in sorted(glob.glob(str(_TABLES / f"{prefix}_*_summary.csv"))):
        b = os.path.basename(f)
        if "loco" in b:
            continue
        m = re.match(rf"{prefix}_(.+?)_(.+?)_summary\.csv$", b)
        if not m:
            continue
        src, tgt = m.groups()
        for r in csv.DictReader(open(f)):
            if r.get("method") != "lr_erm":
                continue
            try:
                out.setdefault(tgt, []).append(
                    (src, float(r["tgt_auroc_mean"]), float(r["tgt_ece_mean"])))
            except (KeyError, ValueError):
                pass
    return out


def run(data_dir: str | None, dataset: str, tau: float | None,
        n_folds: int, seed: int) -> list[dict]:
    if dataset == "i3lung":
        from experiments.run_real_data import load_i3lung_all
        centers, feats = load_i3lung_all(data_dir)
        cohorts = {c: (centers[c][0][feats].values.astype(float),
                       np.asarray(centers[c][1]).astype(int)) for c in centers}
        prefix = "i3lung"
    else:
        from experiments.run_real_data import load_public_transfer
        import importlib
        reg = importlib.import_module(f"data.registry_{dataset}")
        names = list(reg.CENTER_NAMES.values())
        tau = tau if tau is not None else reg.TAU
        cohorts = {}
        for a, b in ((names[0], names[1]), (names[1], names[0])):
            _s, t_df, feats = load_public_transfer(dataset, a, b, tau)
            cohorts[b] = (t_df[feats].values.astype(float),
                          t_df["event"].values.astype(int))
        prefix = f"public_{dataset}"

    erm = _erm_by_target(prefix)
    rows = []
    print(f"\n{'='*84}\n  ORACOLO SUI DATI DEL TARGET  ({len(cohorts)} coorti)\n{'='*84}")
    print(f"  {'target':<10}{'n':>6}{'oracolo AUROC':>15}{'ERM medio':>12}{'H_obs':>9}"
          f"{'oracolo ECE':>13}{'ERM ECE':>10}")
    for c in sorted(cohorts):
        X, y = cohorts[c]
        if len(np.unique(y)) < 2:
            continue
        oa, oe = oof_oracle(X, y, n_folds=n_folds, seed=seed)
        tr = erm.get(c, [])
        ea = float(np.mean([t[1] for t in tr])) if tr else float("nan")
        ee = float(np.mean([t[2] for t in tr])) if tr else float("nan")
        rows.append({"target": c, "n": int(len(y)),
                     "oracle_auroc": round(oa, 4), "oracle_ece": round(oe, 4),
                     "erm_auroc_mean": round(ea, 4), "erm_ece_mean": round(ee, 4),
                     "headroom_auroc": round(oa - ea, 4),
                     "headroom_ece": round(ee - oe, 4),
                     "n_transfers": len(tr)})
        print(f"  {c:<10}{len(y):>6}{oa:>15.3f}{ea:>12.3f}{oa-ea:>+9.3f}"
              f"{oe:>13.3f}{ee:>10.3f}")

    if rows:
        h = [r["headroom_auroc"] for r in rows]
        print(f"\n  margine di discriminazione: mediana {np.median(h):+.3f}, "
              f"intervallo {min(h):+.3f} .. {max(h):+.3f}")
        print("  Positivo significa che le covariate del target sostengono piu'")
        print("  discriminazione di quanta ERM ne raggiunga arrivando da fuori.")

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / f"{prefix}_target_oracle.csv")
    print(f"\n  Scritto: {_TABLES}/{prefix}_target_oracle.csv\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="i3lung")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dataset == "i3lung" and not a.data_dir:
        ap.error("--data_dir e' richiesto per i3lung")
    run(a.data_dir, a.dataset, a.tau, a.n_folds, a.seed)
