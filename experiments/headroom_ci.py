"""Paired patient-level bootstrap for the observed-data headroom.

The headroom table reports point estimates, and the smallest margins invite the
question whether they are distinguishable from zero at all. The oracle and the
transported ERM are evaluated on the same target patients, so their sampling
errors are correlated and an honest interval must be paired: this resamples
target patients and recomputes both sides on each replicate.

Both prediction vectors are reconstructed exactly. The oracle repeats the
five-fold out-of-fold fit of experiments/real_target_oracle.py with the same
seed. ERM repeats the pipeline of run_real_data._run_seed for every incoming
(source, seed) pair, including the per-run IWCV selection of C, which is
deterministic; the recomputed per-transfer means are asserted against the
tracked summaries before any interval is drawn, so a silent divergence aborts
the run instead of shipping.

Resampling is stratified by outcome so every replicate carries both classes.

Usage:  python -m experiments.headroom_ci --data_dir I3LUNG_DATA
"""
from __future__ import annotations

import argparse
import csv
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from experiments import hparams
from experiments.run_real_data import load_i3lung_all, _LR_KWARGS
from experiments.select import select
from utils.metrics import auroc, ece
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"


def oracle_probs(X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                 seed: int = 0) -> np.ndarray:
    """Out-of-fold oracle probabilities, mirroring real_target_oracle exactly."""
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
    return probs


class _View:
    """The dataset view _run_seed hands to select(): imputed, unscaled."""
    def __init__(self, x, y, domain_id):
        self.x = x
        self.y = y.astype(float)
        self.domain_id = domain_id
        self.n = len(y)
        self.u_inv = x
        self.u_spu = x[:, :2]


def erm_probs(Xs, ys, Xt, yt, seed: int) -> np.ndarray:
    """Transported ERM probabilities for one (source, seed), as _run_seed fits
    them: the imputer and scaler live on the source train split, the IWCV
    selection of C sees imputed unscaled features, and the fit sees scaled
    ones."""
    X_tr, _X_val, y_tr, _y_val = train_test_split(
        Xs, ys, test_size=0.2, random_state=seed, stratify=ys)
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_tg_i = imp.transform(Xt)
    sca = StandardScaler().fit(X_tr_i)
    kw = dict(_LR_KWARGS)
    sel = select("erm", [_View(X_tr_i, y_tr, 0)], _View(X_tg_i, yt, 1),
                 seed=seed)
    if sel is not None:
        kw["C"] = float(sel)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(**kw).fit(sca.transform(X_tr_i), y_tr)
    return lr.predict_proba(sca.transform(X_tg_i))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_seeds", type=int, default=10)
    a = ap.parse_args()

    centers, feats = load_i3lung_all(a.data_dir)
    names = sorted(centers)
    data = {c: (centers[c][0][feats].values.astype(float),
                np.asarray(centers[c][1]).astype(int)) for c in names}

    tabled = {}
    for r in csv.DictReader(open(_TABLES / "i3lung_target_oracle.csv")):
        tabled[r["target"]] = (float(r["erm_auroc_mean"]),
                               float(r["headroom_auroc"]))

    rows = []
    rng = np.random.default_rng(0)
    for tgt in names:
        Xt, yt = data[tgt]
        po = oracle_probs(Xt, yt)
        mats, msg = [], []
        for src in names:
            if src == tgt:
                continue
            Xs, ys = data[src]
            for s in range(a.n_seeds):
                mats.append(erm_probs(Xs, ys, Xt, yt, s))
        P = np.vstack(mats)                      # (50, n_t)

        # The recomputation must reproduce the tracked table before any
        # interval is drawn from it.
        erm_mean = float(np.mean([auroc(yt, p) for p in P]))
        tab_erm, tab_h = tabled[tgt]
        assert abs(erm_mean - tab_erm) < 2e-3, (tgt, erm_mean, tab_erm)

        pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
        d_auc, d_ece = [], []
        for _ in range(a.n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                  rng.choice(neg, len(neg), replace=True)])
            yb = yt[idx]
            oa = auroc(yb, po[idx])
            ea = np.mean([auroc(yb, p[idx]) for p in P])
            d_auc.append(oa - ea)
            oe = ece(yb, po[idx])
            ee = np.mean([ece(yb, p[idx]) for p in P])
            d_ece.append(ee - oe)
        lo_a, hi_a = np.percentile(d_auc, [2.5, 97.5])
        lo_e, hi_e = np.percentile(d_ece, [2.5, 97.5])
        rows.append({"target": tgt, "n": len(yt),
                     "headroom_auroc": round(tab_h, 4),
                     "auroc_lo": round(float(lo_a), 4),
                     "auroc_hi": round(float(hi_a), 4),
                     "ece_lo": round(float(lo_e), 4),
                     "ece_hi": round(float(hi_e), 4),
                     "n_boot": a.n_boot})
        star = "*" if lo_a > 0 else " "
        print(f"  {tgt:<6} margine {tab_h:+.3f}  IC95 [{lo_a:+.3f}, {hi_a:+.3f}]{star}"
              f"   ECE [{lo_e:+.3f}, {hi_e:+.3f}]", flush=True)

    save_table_csv(rows, _TABLES / "i3lung_headroom_ci.csv")
    k = sum(1 for r in rows if r["auroc_lo"] > 0)
    print(f"\n  intervalli AUROC che escludono lo zero: {k} / {len(rows)}")
    print(f"  Scritto: {_TABLES}/i3lung_headroom_ci.csv")


if __name__ == "__main__":
    main()
