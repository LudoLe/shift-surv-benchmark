"""
LOOCO: three new directions vs LR ERM baseline.

  1. Cox ERM on full X       — Cox partial likelihood directly on X (no latent),
                               uses full survival info without compression
  2. Importance weighting    — density-ratio reweighting: train src/tgt classifier,
                               weight source by p(tgt|x)/p(src|x), retrain LR
  3. Semi-supervised ST      — self-training: pseudo-label high-confidence target
                               samples, retrain LR on source + pseudo-target

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_three
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from data import registry_rwd as reg
from data.preprocess import impute_median
from generator.fitting import fit_risk_weights_cox

TAU     = 12.0
OUT_DIR = _ROOT / "outputs" / "tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _lr(X_train, y_train, sample_weight=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=0,
        )
        lr.fit(X_train, y_train, sample_weight=sample_weight)
    return lr


# ---------------------------------------------------------------------------
# Method 1 — Cox ERM on full X
# ---------------------------------------------------------------------------

def _cox_erm_auroc(X_src, t_src, delta_src, X_tgt, y_tgt):
    """Fit Cox on (X_src, t_src, delta_src), score X_tgt, return AUROC."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # alpha larger because p=22 (more regularisation than k=10)
        w = fit_risk_weights_cox(X_src, t_src, delta_src, alpha=1e-1)
    scores = X_tgt @ w
    return _auroc_safe(y_tgt, scores)


# ---------------------------------------------------------------------------
# Method 2 — Importance weighting (density ratio)
# ---------------------------------------------------------------------------

def _iw_auroc(X_src, y_src, X_tgt, y_tgt, clip=10.0):
    """
    Estimate density ratio p(tgt|x)/p(src|x) via a source/target classifier.
    Clip weights to avoid extreme values, then retrain LR on weighted source.
    """
    n_s, n_t = len(X_src), len(X_tgt)
    X_dom = np.vstack([X_src, X_tgt])
    y_dom = np.concatenate([np.zeros(n_s), np.ones(n_t)])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        dom_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=0)
        dom_clf.fit(X_dom, y_dom)

    # P(target | x) for source samples → importance weights
    p_tgt = dom_clf.predict_proba(X_src)[:, 1]   # P(domain=tgt | x)
    p_src = 1.0 - p_tgt
    weights = np.clip(p_tgt / (p_src + 1e-9), 0.0, clip)
    weights /= weights.mean()   # normalise to mean 1

    lr = _lr(X_src, y_src, sample_weight=weights)
    return _auroc_safe(y_tgt, lr.predict_proba(X_tgt)[:, 1])


# ---------------------------------------------------------------------------
# Method 3 — Semi-supervised self-training
# ---------------------------------------------------------------------------

def _self_train_auroc(X_src, y_src, X_tgt, y_tgt,
                      conf_thresh=0.75, n_iter=5):
    """
    Iterative self-training:
      1. Fit LR on source.
      2. Pseudo-label target samples whose max predicted prob > conf_thresh.
      3. Retrain on source + pseudo-labeled target.
      4. Repeat n_iter times (or until no new pseudo-labels).
    """
    X_train = X_src.copy()
    y_train = y_src.copy()

    for _ in range(n_iter):
        lr = _lr(X_train, y_train)
        proba = lr.predict_proba(X_tgt)          # (n_t, 2)
        max_p = proba.max(axis=1)
        confident = max_p >= conf_thresh

        if confident.sum() == 0:
            break

        pseudo_y = proba[confident].argmax(axis=1)
        X_train = np.vstack([X_src, X_tgt[confident]])
        y_train = np.concatenate([y_src, pseudo_y])

    lr_final = _lr(X_train, y_train)
    return _auroc_safe(y_tgt, lr_final.predict_proba(X_tgt)[:, 1])


# ---------------------------------------------------------------------------
# Main LOOCO loop
# ---------------------------------------------------------------------------

def run_looco_three(tau=TAU, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    D = len(centers)
    rows = []

    if verbose:
        print("=== LOOCO: Cox ERM / Importance Weighting / Self-Training ===\n")
        print(f"  D={D} centers, τ={tau}\n")
        hdr = (f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  "
               f"{'cox_erm':>8}  {'imp_wt':>8}  {'self_tr':>8}  "
               f"{'Δcox':>7}  {'Δiw':>7}  {'Δst':>7}")
        print(hdr)
        print("-" * len(hdr))

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        # Impute
        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
                   for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])

        # Scale
        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)

        # Labels / survival info
        y_src     = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])
        t_src     = np.concatenate([c["t"]     for c in src_imp])
        delta_src = np.concatenate([c["delta"] for c in src_imp])
        y_tgt     = _binary_label(held["t"], held["delta"], tau)

        # LR ERM
        auroc_lr  = _auroc_safe(y_tgt, _lr(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        # Method 1: Cox ERM on full X
        try:
            auroc_cox = _cox_erm_auroc(X_src_s, t_src, delta_src, X_tgt_s, y_tgt)
        except Exception as e:
            auroc_cox = float("nan")
            if verbose:
                print(f"  Cox failed ({hname}): {e}")

        # Method 2: importance weighting
        try:
            auroc_iw = _iw_auroc(X_src_s, y_src, X_tgt_s, y_tgt)
        except Exception as e:
            auroc_iw = float("nan")
            if verbose:
                print(f"  IW failed ({hname}): {e}")

        # Method 3: self-training
        try:
            auroc_st = _self_train_auroc(X_src_s, y_src, X_tgt_s, y_tgt)
        except Exception as e:
            auroc_st = float("nan")
            if verbose:
                print(f"  ST failed ({hname}): {e}")

        row = {
            "center":   hname,
            "n":        held["n"],
            "lr_erm":   auroc_lr,
            "cox_erm":  auroc_cox,
            "imp_wt":   auroc_iw,
            "self_tr":  auroc_st,
            "delta_cox": auroc_cox - auroc_lr,
            "delta_iw":  auroc_iw  - auroc_lr,
            "delta_st":  auroc_st  - auroc_lr,
        }
        rows.append(row)

        if verbose:
            print(
                f"  {hname:>6}  {held['n']:>4}  {auroc_lr:>8.4f}  "
                f"{auroc_cox:>8.4f}  {auroc_iw:>8.4f}  {auroc_st:>8.4f}  "
                f"{auroc_cox-auroc_lr:>+7.4f}  {auroc_iw-auroc_lr:>+7.4f}  "
                f"{auroc_st-auroc_lr:>+7.4f}"
            )

    df = pd.DataFrame(rows)

    if verbose:
        print()
        cols = ["lr_erm", "cox_erm", "imp_wt", "self_tr",
                "delta_cox", "delta_iw", "delta_st"]
        print("Mean across centers:")
        print(df[cols].mean().round(4).to_string())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "looco_three.csv", index=False)
    if verbose:
        print(f"\nSaved → {out_path}/looco_three.csv")

    return df


if __name__ == "__main__":
    run_looco_three()
