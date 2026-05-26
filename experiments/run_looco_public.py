"""
LOOCO survival benchmark on public multi-centre datasets.

Datasets
--------
  rotterdam_gbsg  : Rotterdam (n=2982) + GBSG2 (n=686), RFS in months
  support2        : SUPPORT2 by dzclass (4 groups, n=9105), OS in months

Methods
-------
  cox_erm   : Cox PH, L2 regularisation (alpha=5e-2)
  cox_enet  : Cox elastic net, alpha tuned by C-index CV on source
  rsf       : Random Survival Forest
  lr_erm    : Logistic regression on binary label at τ (reference)

Metrics: AUROC@τ, AUROC@τ_ref, C-index, iAUC, Brier@τ, IBS

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV

  # single dataset
  python -m experiments.run_looco_public --dataset rotterdam_gbsg
  python -m experiments.run_looco_public --dataset support2

  # both datasets back-to-back
  python -m experiments.run_looco_public --dataset all
"""

from __future__ import annotations
import argparse
import sys
import warnings
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

from data.preprocess import impute_median
from generator.fitting import fit_risk_weights_cox
from experiments.survival_metrics import c_index, iauc, brier_score, ibs, breslow_predict

ENET_ALPHAS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1]
L1_RATIO    = 0.5
COX_ALPHA   = 5e-2


# ---------------------------------------------------------------------------
# Helpers (identical to run_looco_surv_models.py)
# ---------------------------------------------------------------------------

def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2 or np.all(np.isnan(scores)):
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _sksurv_y(t, delta):
    dt = np.dtype([("event", bool), ("time", float)])
    return np.array([(bool(d), float(ti)) for d, ti in zip(delta, t)], dtype=dt)


def _fit_cox_enet(X_src, t_src, delta_src, l1_ratio=L1_RATIO,
                  alphas=ENET_ALPHAS, n_cv=3):
    y_src = _sksurv_y(t_src, delta_src)
    best_alpha, best_ci = alphas[0], -1.0
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=0)

    for alpha in alphas:
        cis = []
        for tr_idx, val_idx in kf.split(X_src):
            try:
                m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1_ratio,
                                           max_iter=1000, fit_baseline_model=False)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    m.fit(X_src[tr_idx], y_src[tr_idx])
                ci = c_index(t_src[val_idx], delta_src[val_idx], m.predict(X_src[val_idx]))
                cis.append(ci)
            except Exception:
                cis.append(float("nan"))
        mean_ci = float(np.nanmean(cis))
        if mean_ci > best_ci:
            best_ci, best_alpha = mean_ci, alpha

    model = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=l1_ratio,
                                    max_iter=1000, fit_baseline_model=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_src, y_src)
    return model, best_alpha


def _fit_rsf(X_src, t_src, delta_src, n_estimators=300, min_leaf=10):
    y_src = _sksurv_y(t_src, delta_src)
    rsf = RandomSurvivalForest(n_estimators=n_estimators,
                               min_samples_leaf=min_leaf,
                               max_features="sqrt",
                               n_jobs=-1, random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        rsf.fit(X_src, y_src)
    return rsf


def _sksurv_proba_at_tau(model, X_test, tau):
    try:
        surv_fns = model.predict_survival_function(X_test)
        proba = np.array([float(1.0 - fn(min(tau, fn.x[-1]))) for fn in surv_fns])
        return np.clip(proba, 1e-6, 1 - 1e-6)
    except Exception:
        return np.full(len(X_test), float("nan"))


def _lr(X_train, y_train):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                class_weight="balanced", random_state=0)
        lr.fit(X_train, y_train)
    return lr


# ---------------------------------------------------------------------------
# Core LOOCO loop
# ---------------------------------------------------------------------------

def run_looco(centers, center_names, tau, tau_ref, tau_grid,
              dataset_name, verbose=True, out_dir=None):
    """
    Run LOOCO evaluation on a pre-loaded list of centre dicts.

    Parameters
    ----------
    centers      : list of dicts {X, t, delta, n, center_id, feature_names}
    center_names : dict  {center_id: str}
    tau          : primary prediction horizon (same units as t)
    tau_ref      : secondary prediction horizon for AUROC comparison
    tau_grid     : list of horizons for iAUC / IBS
    dataset_name : str  used for output directory naming
    """
    if out_dir is None:
        out_dir = _ROOT / "outputs" / "tables" / dataset_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    D    = len(centers)
    rows = []

    if verbose:
        p = centers[0]["X"].shape[1]
        print(f"=== LOOCO: {dataset_name}  ({D} centres, p={p} features) ===\n")
        print(f"  τ={tau}  τ_ref={tau_ref}  τ_grid={tau_grid}\n")
        hdr = (f"{'method':>10}  {'center':>16}  {'n':>5}  "
               f"{'auc@τ':>7}  {'auc@τr':>7}  {'c_idx':>7}  "
               f"{'iAUC':>7}  {'brier@τ':>8}  {'IBS':>7}")
        print(hdr)
        print("-" * len(hdr))

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = center_names.get(held["center_id"], str(held["center_id"]))

        # --- impute & scale ---
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

        # --- LR ERM ---
        lr_model = _lr(X_src_s, y_src_tau)
        proba_lr = lr_model.predict_proba(X_tgt_s)[:, 1]

        # --- Cox ERM ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_cox = fit_risk_weights_cox(X_src_s, t_src, delta_src, alpha=COX_ALPHA)
        score_cox     = X_tgt_s @ w_cox
        score_cox_src = X_src_s @ w_cox
        proba_cox     = breslow_predict(t_src, delta_src, score_cox_src, score_cox, tau)

        # --- Cox elastic net ---
        enet_model, best_alpha = _fit_cox_enet(X_src_s, t_src, delta_src)
        score_enet = enet_model.predict(X_tgt_s)
        proba_enet = _sksurv_proba_at_tau(enet_model, X_tgt_s, tau)
        n_nonzero  = int(np.sum(enet_model.coef_[:, 0] != 0))

        # --- RSF ---
        rsf_model = _fit_rsf(X_src_s, t_src, delta_src)
        score_rsf  = rsf_model.predict(X_tgt_s)
        proba_rsf  = _sksurv_proba_at_tau(rsf_model, X_tgt_s, tau)

        method_list = [
            ("lr_erm",   proba_lr,   proba_lr),
            ("cox_erm",  score_cox,  proba_cox),
            ("cox_enet", score_enet, proba_enet),
            ("rsf",      score_rsf,  proba_rsf),
        ]

        for method, score, proba in method_list:
            auc_tau  = _auroc_safe(y_tgt_tau, score)
            auc_ref  = _auroc_safe(y_tgt_ref, score)
            ci       = c_index(t_tgt, delta_tgt, score)
            ia       = iauc(t_tgt, delta_tgt, score, tau_grid=tau_grid)
            bs_tau   = brier_score(t_src, delta_src, t_tgt, delta_tgt, proba, tau)

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
            else:
                def _pfn(tv, _m=rsf_model, _Xt=X_tgt_s):
                    return _sksurv_proba_at_tau(_m, _Xt, tv)

            ibs_val = ibs(t_src, delta_src, t_tgt, delta_tgt, _pfn, tau_grid=tau_grid)

            rows.append({
                "dataset": dataset_name,
                "method": method,
                "center": hname,
                "n": held["n"],
                "auc_tau": auc_tau,
                "auc_tau_ref": auc_ref,
                "c_index": ci,
                "iauc": ia,
                "brier_tau": bs_tau,
                "ibs": ibs_val,
                "enet_alpha": best_alpha if method == "cox_enet" else float("nan"),
                "enet_nonzero": n_nonzero if method == "cox_enet" else float("nan"),
            })

            if verbose:
                extra = f"  α={best_alpha:.0e} nz={n_nonzero:>2}" if method == "cox_enet" else ""
                print(f"  {method:>10}  {hname:>16}  {held['n']:>5}  "
                      f"{auc_tau:>7.4f}  {auc_ref:>7.4f}  {ci:>7.4f}  "
                      f"{ia:>7.4f}  {bs_tau:>8.4f}  {ibs_val:>7.4f}{extra}")

        if verbose:
            print()

    df = pd.DataFrame(rows)

    if verbose:
        print("=" * 80)
        print(f"Mean across centres ({dataset_name}):\n")
        cols = ["auc_tau", "auc_tau_ref", "c_index", "iauc", "brier_tau", "ibs"]
        for method in ["lr_erm", "cox_erm", "cox_enet", "rsf"]:
            sub  = df[df["method"] == method][cols].mean()
            vals = "  ".join(f"{c.split('_')[-1]}={v:.4f}" for c, v in sub.items())
            print(f"  {method:>10}: {vals}")
        enet_rows = df[df["method"] == "cox_enet"]
        p_total = centers[0]["X"].shape[1]
        print(f"\n  cox_enet selected features (mean): "
              f"{enet_rows['enet_nonzero'].mean():.1f} / {p_total}")

    out_path = out_dir / "looco_surv_models.csv"
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nSaved → {out_path}")

    return df


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "rotterdam_gbsg": {
        "registry": "data.registry_rotterdam_gbsg",
        "tau":      36.0,
        "tau_ref":  48.0,
        "tau_grid": [12.0, 24.0, 36.0, 48.0, 60.0],
    },
    "support2": {
        "registry": "data.registry_support2",
        "tau":      3.0,
        "tau_ref":  6.0,
        "tau_grid": [1.0, 2.0, 3.0, 6.0, 9.0, 12.0],
    },
    "i3lung_vanguri": {
        "registry": "data.registry_i3lung_vanguri",
        "tau":      12.0,
        "tau_ref":  18.0,
        "tau_grid": [6.0, 9.0, 12.0, 15.0, 18.0, 24.0],
    },
}


def run_dataset(dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    import importlib
    cfg = DATASET_CONFIGS[dataset_name]
    reg = importlib.import_module(cfg["registry"])
    centers, _ = reg.load_all_centers()
    return run_looco(
        centers=centers,
        center_names=reg.CENTER_NAMES,
        tau=cfg["tau"],
        tau_ref=cfg["tau_ref"],
        tau_grid=cfg["tau_grid"],
        dataset_name=dataset_name,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOOCO on public survival datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS) + ["all"],
        default="rotterdam_gbsg",
        help="Which dataset to run (default: rotterdam_gbsg)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-row output")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, verbose=not args.quiet)
        print()
