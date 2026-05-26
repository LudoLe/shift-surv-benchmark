"""
LOOCO comparison: Weibull vs log-logistic vs Cox for w estimation, real RWD data.

Four prediction pipelines per fold:
  lr_erm         : standard LR on pooled source X (does not use generator w)
  latent_weibull : score = X_tgt @ A @ w_weibull  (LR on binary y)
  latent_loglog  : score = X_tgt @ A @ w_loglog   (log-logistic MLE on t/delta)
  latent_cox     : score = X_tgt @ A @ w_cox       (Cox partial likelihood on t/delta)

The latent pipelines test whether using full survival information (event times,
not just binary y at τ) gives better risk scores when projected through the
shared PCA latent space A.

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_loglogistic
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from data import registry_rwd as reg
from data.preprocess import impute_median
from generator.fitting import fit_reference_generator


TAU     = 12.0
K_TOTAL = 10
OUT_DIR = _ROOT / "outputs" / "tables"


def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def run_looco_loglogistic(tau=TAU, k_total=K_TOTAL, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    D = len(centers)

    rows = []

    if verbose:
        print("=== LOOCO: Weibull / Log-logistic / Cox  —  w estimation comparison ===\n")
        print(f"  D={D} centers, k={k_total}, τ={tau}\n")
        hdr = (f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  "
               f"{'lat_weib':>8}  {'lat_loglog':>10}  {'lat_cox':>8}  "
               f"{'Δcox-lr':>8}  {'Δcox-w':>8}")
        print(hdr)
        print("-" * len(hdr))

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        # Impute + scale on source
        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
                   for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])
        y_tgt = _binary_label(held["t"], held["delta"], tau)

        # Fit all three generators (same A via PCA, different w)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A, w_w, _, scaler = fit_reference_generator(
                src_imp, k_total=k_total, tau=tau, time_distribution="weibull")
            _, w_l, dp_l, _   = fit_reference_generator(
                src_imp, k_total=k_total, tau=tau, time_distribution="loglogistic")
            _, w_c, _, _      = fit_reference_generator(
                src_imp, k_total=k_total, tau=tau, time_distribution="cox")

        X_tgt_s = scaler.transform(X_held_imp)
        X_src_s = np.vstack([scaler.transform(c["X"]) for c in src_imp])
        y_src   = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])
        mu_src  = (X_src_s @ A).mean(axis=0)   # shared A from Weibull fit

        # Pipeline 1: LR ERM on raw X (no generator)
        from sklearn.linear_model import LogisticRegression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                    class_weight="balanced", random_state=0)
            lr.fit(X_src_s, y_src)
        auroc_lr = _auroc_safe(y_tgt, lr.predict_proba(X_tgt_s)[:, 1])

        # Pipelines 2-4: latent projection  score = (X @ A − μ_src) @ w
        def _lat_auroc(w):
            return _auroc_safe(y_tgt, (X_tgt_s @ A - mu_src) @ w)

        auroc_w = _lat_auroc(w_w)
        auroc_l = _lat_auroc(w_l)
        auroc_c = _lat_auroc(w_c)

        row = {
            "center":      hname,
            "n":           held["n"],
            "nu_ll":       round(dp_l[0].nu, 3),
            "lr_erm":      auroc_lr,
            "lat_weibull": auroc_w,
            "lat_loglog":  auroc_l,
            "lat_cox":     auroc_c,
            "delta_cox_vs_lr": auroc_c - auroc_lr,
            "delta_cox_vs_w":  auroc_c - auroc_w,
        }
        rows.append(row)

        if verbose:
            print(
                f"  {hname:>6}  {held['n']:>4}  "
                f"{auroc_lr:>8.4f}  {auroc_w:>8.4f}  {auroc_l:>10.4f}  "
                f"{auroc_c:>8.4f}  {auroc_c-auroc_lr:>+8.4f}  {auroc_c-auroc_w:>+8.4f}"
            )

    df = pd.DataFrame(rows)

    if verbose:
        print()
        cols = ["lr_erm", "lat_weibull", "lat_loglog", "lat_cox",
                "delta_cox_vs_lr", "delta_cox_vs_w"]
        print("Mean across centers:")
        print(df[cols].mean().round(4).to_string())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "looco_loglogistic.csv", index=False)
    if verbose:
        print(f"\nSaved → {out_path}/looco_loglogistic.csv")

    return df


if __name__ == "__main__":
    run_looco_loglogistic()
