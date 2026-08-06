"""Select one hyperparameter set per method, once, on a disjoint seed block.

What this fixes
---------------
Only DANN and CDAN had a search; ERM, IW-ERM, CORAL, IRM and GroupDRO each ran
at a single fixed value. A claim that no method beats ERM is unverified for any
method whose one setting was never checked, and the bias could run either way.
Leaving the baseline untuned while tuning the methods would have been generous
to adaptation, so ERM is tuned as well.

The protocol, from configs/hyperparameters.yaml
-----------------------------------------------
Tuning runs on seeds 200--209, disjoint from the screening seeds 0--9 and the
confirmation seeds 100--119: selecting on the seeds you later evaluate on is
leakage whatever the parameter. One set is chosen per method and frozen for
every regime, because a deployable protocol cannot retune when the shift
changes -- it does not know which shift it faces -- and per-regime selection
would fold the choice back into the noise the confirmation stage exists to
remove. Three regimes are used, one from each family of Table I, so the values
are not fitted to a single kind of shift.

The criterion never sees target labels:

  adaptation  importance-weighted AUROC on the held-out source validation
              split, weights from a domain classifier trained on source-train
              against unlabeled target covariates. The target enters only
              through its covariates, exactly as at deployment.

  DG          leave-one-source-out AUROC across source centers. The target is
              not observed at all, in covariates or labels.

Usage
-----
  python -m experiments.tune                    # all methods
  python -m experiments.tune --methods erm coral
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.metrics import auroc

_SPACE = _ROOT / "configs" / "hyperparameters.yaml"
_OUT = _ROOT / "configs" / "hyperparameters_selected.yaml"


# ---------------------------------------------------------------------------
# The two selection criteria
# ---------------------------------------------------------------------------

def _iwcv_score(fit_predict, src, tgt, seed: int) -> float:
    """Importance-weighted AUROC on the source validation split.

    ``fit_predict`` receives (X_tr, y_tr, X_val, X_tgt) already scaled and
    returns validation probabilities. The weights come from a domain classifier
    trained on source-train against target covariates and are evaluated on the
    validation rows, so they are out-of-sample with respect to that classifier.
    """
    from experiments.adaptation_methods import _importance_weights, _weighted_auroc

    X_all = np.vstack([d.x for d in src])
    y_all = np.concatenate([d.y for d in src])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all)
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = sc.transform(X_tr), sc.transform(X_val)
    X_tgt_s = sc.transform(tgt.x)

    w = _importance_weights(X_tr_s, X_tgt_s, X_eval=X_val_s)
    p_val = fit_predict(X_tr_s, y_tr, X_val_s, X_tgt_s)
    return _weighted_auroc(y_val, p_val, w)


def _loso_score(fit_predict_groups, src, seed: int) -> float:
    """Leave-one-source-out AUROC, averaged over the held-out centers.

    Fits on all sources but one and scores on the one left out. The target is
    never touched, which is the only criterion available to a method that does
    not observe it.
    """
    if len(src) < 2:
        return float("nan")
    scores = []
    for k in range(len(src)):
        tr = [d for j, d in enumerate(src) if j != k]
        ho = src[k]
        X_tr = np.vstack([d.x for d in tr])
        sc = StandardScaler().fit(X_tr)
        groups = [(sc.transform(d.x), np.asarray(d.y, dtype=float)) for d in tr]
        p_ho = fit_predict_groups(groups, sc.transform(ho.x))
        s = auroc(np.asarray(ho.y).astype(int), p_ho)
        if not np.isnan(s):
            scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# One candidate evaluation per method
# ---------------------------------------------------------------------------

def _score_erm(C, src, tgt, seed):
    def fp(X_tr, y_tr, X_val, X_tgt):
        m = LogisticRegression(max_iter=500, solver="lbfgs", C=C, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr)
        return m.predict_proba(X_val)[:, 1]
    return _iwcv_score(fp, src, tgt, seed)


def _score_iw_erm(clip, src, tgt, seed):
    from experiments.adaptation_methods import _importance_weights

    def fp(X_tr, y_tr, X_val, X_tgt):
        # The candidate clip is passed through, not applied on top: without the
        # override _importance_weights reads the frozen file, and clipping twice
        # keeps whichever bound is tighter, so the widest candidate can never be
        # evaluated. The scoring weights inside _iwcv_score deliberately keep the
        # frozen clip -- the criterion must not move with the candidate.
        w = _importance_weights(X_tr, X_tgt, X_eval=X_tr, clip=tuple(clip))
        m = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr, sample_weight=w)
        return m.predict_proba(X_val)[:, 1]
    return _iwcv_score(fp, src, tgt, seed)


def _score_coral(shrink, src, tgt, seed):
    from adapt.feature_based import CORAL as _CORAL

    def fp(X_tr, y_tr, X_val, X_tgt):
        m = _CORAL(estimator=LogisticRegression(max_iter=500, C=1.0, random_state=0),
                   Xt=X_tgt, lambda_=shrink, verbose=0, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr)
        return m.estimator_.predict_proba(m.transform(X_val, domain="src"))[:, 1]
    return _iwcv_score(fp, src, tgt, seed)


def _score_irm(lam, src, tgt, seed):
    from experiments.adaptation_methods import fit_irm

    def fpg(groups, X_ho):
        class _D:
            def __init__(self, x, y):
                self.x, self.y = x, y
                self.n = len(y)
                self.domain_id = 0
        srcs = [_D(X, y) for X, y in groups]
        r = fit_irm(srcs, _D(X_ho, np.zeros(len(X_ho))), lam=lam, seed=seed)
        return np.asarray(r["target_probs"])
    return _loso_score(fpg, src, seed)


def _score_group_dro(q_lr, src, tgt, seed):
    from experiments.group_dro import train_group_dro, _sigmoid

    def fpg(groups, X_ho):
        tr, Xv, yv = [], [], []
        for Xg, yg in groups:
            a, b, c, d = train_test_split(Xg, yg, test_size=0.2, random_state=0,
                                          stratify=yg if len(np.unique(yg)) > 1 else None)
            tr.append((a, c.astype(float))); Xv.append(b); yv.append(d)
        th, bb, _ = train_group_dro(tr, X_val=np.vstack(Xv),
                                    y_val=np.concatenate(yv).astype(float), q_lr=q_lr)
        return _sigmoid(X_ho @ th + bb)
    return _loso_score(fpg, src, seed)


_SCORERS = {
    "erm":       ("C",                  _score_erm,       1),
    "iw_erm":    ("clip",               _score_iw_erm,    1),
    "coral":     ("shrinkage",          _score_coral,     1),
    "irm":       ("penalty_weight",     _score_irm,       3),
    "group_dro": ("group_lr",           _score_group_dro, 3),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(methods: list[str] | None = None, verbose: bool = True) -> dict:
    if _OUT.exists():
        print(f"  nota: {_OUT.name} esiste gia'. I valori dentro entrano nel\n"
              f"        criterio di selezione (pesi IWCV, forza di regolarizzazione\n"
              f"        condivisa), quindi il tuning non e' idempotente. Rimuovilo\n"
              f"        per ripartire dai default pre-tuning.\n")
    space = yaml.safe_load(_SPACE.read_text())
    meta, grids = space["meta"], space["grids"]
    seeds, regimes = meta["tuning_seeds"], meta["tuning_regimes"]
    n = meta["n_per_domain"]
    todo = methods or list(_SCORERS)

    selected: dict = {}
    for name in todo:
        if name not in _SCORERS:
            print(f"  {name}: nessun criterio definito, salto"); continue
        key, scorer, n_src = _SCORERS[name]
        cands = grids[name][key]
        if verbose:
            print(f"\n  {name}  ({key}, {len(cands)} candidati, "
                  f"{n_src} sorgente/i, criterio "
                  f"{'LOSO' if n_src > 1 else 'IWCV'})")
        table = {}
        t0 = time.time()
        for c in cands:
            vals = []
            for reg in regimes:
                for s in seeds:
                    d = generate_scenario(reg, n_source_domains=n_src,
                                          n_per_domain=n, seed=s,
                                          base=BaseParams(seed=s))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        v = scorer(c, d["source"], d["target"], s)
                    if not np.isnan(v):
                        vals.append(v)
            table[str(c)] = float(np.mean(vals)) if vals else float("nan")
            if verbose:
                print(f"    {str(c):<14} {table[str(c)]:.4f}")
        best = max(table, key=lambda k: table[k])
        best_val = next(c for c in cands if str(c) == best)
        selected[name] = {"parameter": key, "value": best_val,
                          "criterion_mean": round(table[best], 4),
                          "grid_scores": {k: round(v, 4) for k, v in table.items()},
                          "spread": round(max(table.values()) - min(table.values()), 4)}
        if verbose:
            print(f"    -> scelto {best}  ({time.time()-t0:.0f}s, "
                  f"escursione sulla griglia {selected[name]['spread']:.4f})")

    out = {"selected_on": {k: meta[k] for k in
                           ("tuning_seeds", "tuning_regimes", "n_per_domain")},
           "selected": selected}
    _OUT.write_text(yaml.safe_dump(out, sort_keys=False))

    if verbose:
        print(f"\n{'='*70}\n  SELEZIONE CONGELATA\n{'='*70}")
        print(f"  {'metodo':<12}{'parametro':<18}{'scelto':>14}{'escursione':>14}")
        for k, v in selected.items():
            print(f"  {k:<12}{v['parameter']:<18}{str(v['value']):>14}"
                  f"{v['spread']:>14.4f}")
        print(f"\n  Scritto: {_OUT}")
        print("  La colonna 'escursione' e' la differenza fra il candidato migliore")
        print("  e il peggiore: se e' piccola, quel parametro non guidava il risultato.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Freeze one hyperparameter set per method.")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()
    run(args.methods)
