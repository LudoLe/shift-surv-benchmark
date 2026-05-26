"""
LOOCO: survival models on tabular features.

Methods
-------
  cox_erm      : Cox partial likelihood, L2 reg (baseline, alpha=5e-2)
  cox_enet     : Cox elastic net (L1+L2), alpha tuned by CV on source
  rsf          : Random Survival Forest
  lr_erm       : LR on binary label at τ (reference)

Metrics: AUROC@9, AUROC@12, C-index, iAUC, Brier@9, IBS

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_surv_models
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from data import registry_rwd as reg
from data.preprocess import impute_median
from generator.fitting import fit_risk_weights_cox
from experiments.survival_metrics import c_index, iauc, brier_score, ibs, breslow_predict

TAU       = 9.0
TAU_REF   = 12.0
TAU_GRID  = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]
COX_ALPHA = 5e-2
OUT_DIR   = _ROOT / "outputs" / "tables"

# Elastic net alpha grid for CV
ENET_ALPHAS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1]
L1_RATIO    = 0.5   # equal mix of L1 and L2


def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _lr(X_train, y_train):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                class_weight="balanced", random_state=0)
        lr.fit(X_train, y_train)
    return lr


def _sksurv_y(t, delta):
    """Convert (t, delta) to structured array for sksurv."""
    dt = np.dtype([("event", bool), ("time", float)])
    return np.array([(bool(d), float(ti)) for d, ti in zip(delta, t)], dtype=dt)


# ---------------------------------------------------------------------------
# Elastic net Cox — CV on source to choose alpha
# ---------------------------------------------------------------------------

def _fit_cox_enet(X_src, t_src, delta_src, l1_ratio=L1_RATIO,
                  alphas=ENET_ALPHAS, n_cv=3):
    """
    Fit Cox elastic net. Choose alpha by C-index CV on source.
    Returns fitted model.
    """
    y_src = _sksurv_y(t_src, delta_src)

    best_alpha, best_ci = alphas[0], -1.0
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=0)

    for alpha in alphas:
        cis = []
        for tr_idx, val_idx in kf.split(X_src):
            try:
                m = CoxnetSurvivalAnalysis(
                    alphas=[alpha], l1_ratio=l1_ratio,
                    max_iter=1000, fit_baseline_model=False
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    m.fit(X_src[tr_idx], y_src[tr_idx])
                score = m.predict(X_src[val_idx])
                ci = c_index(t_src[val_idx], delta_src[val_idx], score)
                cis.append(ci)
            except Exception:
                cis.append(float("nan"))
        mean_ci = float(np.nanmean(cis))
        if mean_ci > best_ci:
            best_ci, best_alpha = mean_ci, alpha

    # Refit on all source data with best alpha
    model = CoxnetSurvivalAnalysis(
        alphas=[best_alpha], l1_ratio=l1_ratio,
        max_iter=1000, fit_baseline_model=True
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_src, y_src)

    return model, best_alpha


# ---------------------------------------------------------------------------
# Random Survival Forest
# ---------------------------------------------------------------------------

def _fit_rsf(X_src, t_src, delta_src, n_estimators=300, min_leaf=10):
    y_src = _sksurv_y(t_src, delta_src)
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        rsf.fit(X_src, y_src)
    return rsf


# ---------------------------------------------------------------------------
# Brier/IBS from sksurv model (uses built-in survival function)
# ---------------------------------------------------------------------------

def _sksurv_proba_at_tau(model, X_test, tau):
    """P(T ≤ τ | X) from a fitted sksurv model."""
    try:
        surv_fns = model.predict_survival_function(X_test)
        proba = np.array([
            float(1.0 - fn(min(tau, fn.x[-1]))) for fn in surv_fns
        ])
        return np.clip(proba, 1e-6, 1 - 1e-6)
    except Exception:
        return np.full(len(X_test), float("nan"))


# ---------------------------------------------------------------------------
# Full LOOCO loop
# ---------------------------------------------------------------------------

def run_looco_surv_models(tau=TAU, tau_ref=TAU_REF, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    D = len(centers)
    rows = []

    if verbose:
        print("=== LOOCO: survival models on tabular features ===\n")
        print(f"  τ={tau}  τ_ref={tau_ref}  τ_grid={TAU_GRID}\n")
        hdr = (f"{'method':>10}  {'center':>6}  {'n':>4}  "
               f"{'auc@9':>7}  {'auc@12':>7}  {'c_idx':>7}  "
               f"{'iAUC':>7}  {'brier@9':>8}  {'IBS':>7}")
        print(hdr)
        print("-" * len(hdr))

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        # Impute + scale
        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
                   for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])

        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)

        t_src     = np.concatenate([c["t"]     for c in src_imp])
        delta_src = np.concatenate([c["delta"] for c in src_imp])
        t_tgt     = held["t"]
        delta_tgt = held["delta"]

        y_src_tau = _binary_label(t_src, delta_src, tau)
        y_tgt_tau = _binary_label(t_tgt, delta_tgt, tau)
        y_tgt_ref = _binary_label(t_tgt, delta_tgt, tau_ref)

        # ---- LR ERM ----
        lr_model  = _lr(X_src_s, y_src_tau)
        proba_lr  = lr_model.predict_proba(X_tgt_s)[:, 1]

        # ---- Cox ERM (L2, our best baseline) ----
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_cox = fit_risk_weights_cox(X_src_s, t_src, delta_src, alpha=COX_ALPHA)
        score_cox     = X_tgt_s @ w_cox
        score_cox_src = X_src_s @ w_cox
        proba_cox     = breslow_predict(t_src, delta_src, score_cox_src, score_cox, tau)

        # ---- Cox elastic net ----
        enet_model, best_alpha = _fit_cox_enet(X_src_s, t_src, delta_src)
        score_enet  = enet_model.predict(X_tgt_s)
        proba_enet  = _sksurv_proba_at_tau(enet_model, X_tgt_s, tau)
        n_nonzero   = int(np.sum(enet_model.coef_[:, 0] != 0))

        # ---- RSF ----
        rsf_model  = _fit_rsf(X_src_s, t_src, delta_src)
        score_rsf  = rsf_model.predict(X_tgt_s)
        proba_rsf  = _sksurv_proba_at_tau(rsf_model, X_tgt_s, tau)

        method_list = [
            ("lr_erm",   proba_lr,   proba_lr),
            ("cox_erm",  score_cox,  proba_cox),
            ("cox_enet", score_enet, proba_enet),
            ("rsf",      score_rsf,  proba_rsf),
        ]

        for method, score, proba in method_list:
            auc9  = _auroc_safe(y_tgt_tau, score)
            auc12 = _auroc_safe(y_tgt_ref, score)
            ci    = c_index(t_tgt, delta_tgt, score)
            ia    = iauc(t_tgt, delta_tgt, score, tau_grid=TAU_GRID)
            bs9   = brier_score(t_src, delta_src, t_tgt, delta_tgt, proba, tau)

            if method == "lr_erm":
                def _pfn(tv, _Xs=X_src_s, _Xt=X_tgt_s, _ts=t_src, _ds=delta_src):
                    ys = _binary_label(_ts, _ds, tv)
                    if len(np.unique(ys)) < 2:
                        return np.full(len(t_tgt), float("nan"))
                    return _lr(_Xs, ys).predict_proba(_Xt)[:, 1]
            elif method == "cox_erm":
                def _pfn(tv, _ss=score_cox_src, _st=score_cox,
                         _ts=t_src, _ds=delta_src):
                    return breslow_predict(_ts, _ds, _ss, _st, tv)
            elif method == "cox_enet":
                def _pfn(tv, _m=enet_model, _Xt=X_tgt_s):
                    return _sksurv_proba_at_tau(_m, _Xt, tv)
            else:  # rsf
                def _pfn(tv, _m=rsf_model, _Xt=X_tgt_s):
                    return _sksurv_proba_at_tau(_m, _Xt, tv)

            ibs_val = ibs(t_src, delta_src, t_tgt, delta_tgt, _pfn, tau_grid=TAU_GRID)

            extra = f"  α={best_alpha:.0e} nz={n_nonzero:>2}" if method == "cox_enet" else ""
            row = {
                "method": method, "center": hname, "n": held["n"],
                "auc_tau9": auc9, "auc_tau12": auc12,
                "c_index": ci, "iauc": ia,
                "brier_tau9": bs9, "ibs": ibs_val,
                "enet_alpha": best_alpha if method == "cox_enet" else float("nan"),
                "enet_nonzero": n_nonzero if method == "cox_enet" else float("nan"),
            }
            rows.append(row)

            if verbose:
                print(f"  {method:>10}  {hname:>6}  {held['n']:>4}  "
                      f"{auc9:>7.4f}  {auc12:>7.4f}  {ci:>7.4f}  "
                      f"{ia:>7.4f}  {bs9:>8.4f}  {ibs_val:>7.4f}{extra}")

        if verbose:
            print()

    df = pd.DataFrame(rows)

    if verbose:
        print("=" * 70)
        print("Mean across centers:\n")
        cols = ["auc_tau9", "auc_tau12", "c_index", "iauc", "brier_tau9", "ibs"]
        for method in ["lr_erm", "cox_erm", "cox_enet", "rsf"]:
            sub = df[df["method"] == method][cols].mean()
            vals = "  ".join(f"{c.split('_')[-1]}={v:.4f}" for c, v in sub.items())
            print(f"  {method:>10}: {vals}")

        # Cox enet: how many features selected per fold?
        enet_rows = df[df["method"] == "cox_enet"]
        print(f"\n  cox_enet selected features (mean): "
              f"{enet_rows['enet_nonzero'].mean():.1f} / 22")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "looco_surv_models.csv", index=False)
    if verbose:
        print(f"\nSaved → {out_path}/looco_surv_models.csv")

    return df


if __name__ == "__main__":
    run_looco_surv_models()
