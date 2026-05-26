"""
LOOCO full evaluation: LR ERM vs Cox ERM vs lat_cox.

Metrics
-------
  AUROC   at τ=9  (best horizon from τ sweep)
  AUROC   at τ=12 (original horizon, for reference)
  C-index         (τ-independent concordance)
  iAUC            (mean AUROC over τ ∈ {6,9,12,15,18,24})
  Brier           at τ=9  (LR only — needs calibrated proba)
  IBS             (LR only — integrated Brier over τ grid)

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_full_eval
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
from sklearn.preprocessing import StandardScaler

from data import registry_rwd as reg
from data.preprocess import impute_median
from generator.fitting import fit_risk_weights_cox, fit_reference_generator
from experiments.survival_metrics import c_index, iauc, brier_score, ibs, breslow_predict

TAU      = 9.0    # best from τ sweep
TAU_REF  = 12.0   # reference
TAU_GRID = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]
K_LATENT = 10
COX_ALPHA = 5e-2
OUT_DIR  = _ROOT / "outputs" / "tables"


def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _lr(X_train, y_train):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=0,
        )
        lr.fit(X_train, y_train)
    return lr


def run_full_eval(tau=TAU, tau_ref=TAU_REF, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    D = len(centers)
    rows = []

    if verbose:
        print("=== LOOCO full evaluation ===\n")
        print(f"  Methods: LR ERM | Cox ERM | lat_cox (k={K_LATENT})")
        print(f"  τ_main={tau}  τ_ref={tau_ref}  τ_grid={TAU_GRID}\n")
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

        # --- Cox ERM ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_cox = fit_risk_weights_cox(X_src_s, t_src, delta_src, alpha=COX_ALPHA)
        score_cox = X_tgt_s @ w_cox

        # --- lat_cox k=10 ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A, w_lc, _, _ = fit_reference_generator(
                src_imp, k_total=K_LATENT, tau=tau, time_distribution="cox")
        mu_src = (X_src_s @ A).mean(axis=0)
        score_latcox = (X_tgt_s @ A - mu_src) @ w_lc

        # --- LR ERM ---
        lr_model = _lr(X_src_s, y_src_tau)
        proba_lr  = lr_model.predict_proba(X_tgt_s)[:, 1]
        score_lr  = proba_lr

        # Breslow probabilities for Cox ERM and lat_cox
        score_cox_src    = X_src_s @ w_cox
        proba_cox        = breslow_predict(t_src, delta_src,
                                           score_cox_src, score_cox, tau)

        score_latcox_src = (X_src_s @ A - mu_src) @ w_lc
        proba_latcox     = breslow_predict(t_src, delta_src,
                                           score_latcox_src, score_latcox, tau)

        for method, score, proba in [
            ("lr_erm",  score_lr,      proba_lr),
            ("cox_erm", score_cox,     proba_cox),
            ("lat_cox", score_latcox,  proba_latcox),
        ]:
            auc9  = _auroc_safe(y_tgt_tau, score)
            auc12 = _auroc_safe(y_tgt_ref, score)
            ci    = c_index(t_tgt, delta_tgt, score)
            ia    = iauc(t_tgt, delta_tgt, score, tau_grid=TAU_GRID)

            bs9 = brier_score(t_src, delta_src, t_tgt, delta_tgt, proba, tau)

            # IBS: proba function at each τ
            if method == "lr_erm":
                def _proba_at_tau(t_val, _Xs=X_src_s, _Xt=X_tgt_s,
                                  _ts=t_src, _ds=delta_src):
                    y_s = _binary_label(_ts, _ds, t_val)
                    if len(np.unique(y_s)) < 2:
                        return np.full(len(t_tgt), float("nan"))
                    return _lr(_Xs, y_s).predict_proba(_Xt)[:, 1]
            elif method == "cox_erm":
                def _proba_at_tau(t_val, _sc_src=score_cox_src,
                                  _sc_tgt=score_cox, _ts=t_src, _ds=delta_src):
                    return breslow_predict(_ts, _ds, _sc_src, _sc_tgt, t_val)
            else:  # lat_cox
                def _proba_at_tau(t_val, _sc_src=score_latcox_src,
                                  _sc_tgt=score_latcox, _ts=t_src, _ds=delta_src):
                    return breslow_predict(_ts, _ds, _sc_src, _sc_tgt, t_val)

            ibs_val = ibs(t_src, delta_src, t_tgt, delta_tgt,
                          _proba_at_tau, tau_grid=TAU_GRID)

            row = {
                "method": method, "center": hname, "n": held["n"],
                "auc_tau9": auc9, "auc_tau12": auc12,
                "c_index": ci, "iauc": ia,
                "brier_tau9": bs9, "ibs": ibs_val,
            }
            rows.append(row)

            if verbose:
                bs_str  = f"{bs9:.4f}" if not np.isnan(bs9) else "  n/a "
                ibs_str = f"{ibs_val:.4f}" if not np.isnan(ibs_val) else "  n/a "
                print(f"  {method:>10}  {hname:>6}  {held['n']:>4}  "
                      f"{auc9:>7.4f}  {auc12:>7.4f}  {ci:>7.4f}  "
                      f"{ia:>7.4f}  {bs_str:>8}  {ibs_str:>7}")

        if verbose:
            print()

    df = pd.DataFrame(rows)

    if verbose:
        print("=" * 60)
        print("Mean across centers:\n")
        cols = ["auc_tau9", "auc_tau12", "c_index", "iauc", "brier_tau9", "ibs"]
        for method in ["lr_erm", "cox_erm", "lat_cox"]:
            sub = df[df["method"] == method][cols].mean()
            print(f"  {method:>10}: " +
                  "  ".join(f"{c.split('_')[-1]}={v:.4f}"
                             if not np.isnan(v) else f"{c.split('_')[-1]}=  n/a"
                             for c, v in sub.items()))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "looco_full_eval.csv", index=False)
    if verbose:
        print(f"\nSaved → {out_path}/looco_full_eval.csv")

    return df


if __name__ == "__main__":
    run_full_eval()
