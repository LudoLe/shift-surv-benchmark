"""
LOOCO ablations: three directions to improve on latent projection.

Experiments
-----------
1. k sweep    : vary k ∈ K_GRID for latent_weibull (find best compression level)
2. Hybrid     : LR on [X | latent_score] — raw features + w-projected score as extra feature
3. CORAL adapt: per-fold CORAL on full X (no latent compression)

All compared against LR ERM baseline.

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_ablations
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
from generator.fitting import fit_reference_generator, fit_loading_matrix

TAU     = 12.0
K_GRID  = [3, 6, 10, 15, 22]   # 22 = p (no compression)
OUT_DIR = _ROOT / "outputs" / "tables"


def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _lr_fit(X_train, y_train):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=0,
        )
        lr.fit(X_train, y_train)
    return lr


# ---------------------------------------------------------------------------
# Experiment 1 — k sweep (latent_weibull at varying k)
# ---------------------------------------------------------------------------

def run_k_sweep(centers, tau=TAU, verbose=True):
    D = len(centers)
    rows = []
    if verbose:
        k_hdr = "  ".join(f"k={k:>2}" for k in K_GRID)
        print(f"\n--- Experiment 1: k sweep (latent_weibull) ---")
        print(f"{'center':>6}  {'n':>4}  lr_erm  " + k_hdr)
        print("-" * (18 + 8 * len(K_GRID)))

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])} for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])
        y_tgt = _binary_label(held["t"], held["delta"], tau)

        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)
        y_src = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])

        auroc_lr = _auroc_safe(y_tgt, _lr_fit(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        row = {"center": hname, "n": held["n"], "lr_erm": auroc_lr}
        auroc_ks = []
        for k in K_GRID:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                A, w, _, _ = fit_reference_generator(
                    src_imp, k_total=k, tau=tau, time_distribution="weibull")
            mu_src = (X_src_s @ A).mean(axis=0)
            score  = (X_tgt_s @ A - mu_src) @ w
            ak = _auroc_safe(y_tgt, score)
            row[f"k{k}"] = ak
            auroc_ks.append(ak)

        rows.append(row)
        if verbose:
            ks_str = "  ".join(f"{ak:>6.4f}" for ak in auroc_ks)
            print(f"  {hname:>6}  {held['n']:>4}  {auroc_lr:.4f}  {ks_str}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean across centers:")
        cols = ["lr_erm"] + [f"k{k}" for k in K_GRID]
        print(df[cols].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Experiment 2 — Hybrid: LR on [X | latent_score(k=6)]
# ---------------------------------------------------------------------------

def run_hybrid(centers, tau=TAU, k_latent=6, verbose=True):
    D = len(centers)
    rows = []
    if verbose:
        print(f"\n--- Experiment 2: Hybrid LR (X + latent_score, k={k_latent}) ---")
        print(f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  {'latent':>8}  {'hybrid':>8}  "
              f"{'Δhyb-lr':>8}  {'Δhyb-lat':>9}")
        print("-" * 66)

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])} for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])
        y_tgt = _binary_label(held["t"], held["delta"], tau)

        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)
        y_src = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])

        # Fit generator
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A, w, _, _ = fit_reference_generator(
                src_imp, k_total=k_latent, tau=tau, time_distribution="weibull")
        mu_src = (X_src_s @ A).mean(axis=0)

        # Latent scores
        lat_src = ((X_src_s @ A - mu_src) @ w).reshape(-1, 1)
        lat_tgt = ((X_tgt_s @ A - mu_src) @ w).reshape(-1, 1)

        # LR ERM (raw X)
        auroc_lr = _auroc_safe(y_tgt, _lr_fit(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        # Latent only
        auroc_lat = _auroc_safe(y_tgt, lat_tgt.ravel())

        # Hybrid: X concatenated with latent score
        X_src_hyb = np.hstack([X_src_s, lat_src])
        X_tgt_hyb = np.hstack([X_tgt_s, lat_tgt])
        auroc_hyb = _auroc_safe(y_tgt, _lr_fit(X_src_hyb, y_src).predict_proba(X_tgt_hyb)[:, 1])

        row = {
            "center": hname, "n": held["n"],
            "lr_erm": auroc_lr, "latent": auroc_lat, "hybrid": auroc_hyb,
            "delta_hyb_lr": auroc_hyb - auroc_lr,
            "delta_hyb_lat": auroc_hyb - auroc_lat,
        }
        rows.append(row)
        if verbose:
            print(f"  {hname:>6}  {held['n']:>4}  {auroc_lr:>8.4f}  {auroc_lat:>8.4f}  "
                  f"{auroc_hyb:>8.4f}  {auroc_hyb-auroc_lr:>+8.4f}  {auroc_hyb-auroc_lat:>+9.4f}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean across centers:")
        cols = ["lr_erm", "latent", "hybrid", "delta_hyb_lr", "delta_hyb_lat"]
        print(df[cols].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Experiment 3 — CORAL on full X (no latent compression)
# ---------------------------------------------------------------------------

def _coral_transform(X_src, X_tgt, reg=1e-3):
    """Return X_tgt_aligned under CORAL (second-order feature alignment)."""
    p = X_src.shape[1]
    C_s = np.cov(X_src, rowvar=False) + reg * np.eye(p)
    C_t = np.cov(X_tgt, rowvar=False) + reg * np.eye(p)
    # Whitening source, re-colouring with target covariance
    vals_s, vecs_s = np.linalg.eigh(C_s)
    vals_t, vecs_t = np.linalg.eigh(C_t)
    W_s = vecs_s @ np.diag(np.maximum(vals_s, 1e-9) ** -0.5) @ vecs_s.T
    W_t = vecs_t @ np.diag(np.maximum(vals_t, 1e-9) **  0.5) @ vecs_t.T
    T = W_s @ W_t   # (p, p)  maps source → target covariance
    return X_src @ T


def run_coral(centers, tau=TAU, verbose=True):
    D = len(centers)
    rows = []
    if verbose:
        print(f"\n--- Experiment 3: CORAL on full X (p={centers[0]['X'].shape[1]}) ---")
        print(f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  {'coral':>8}  {'Δcoral':>8}")
        print("-" * 48)

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])} for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])
        y_tgt = _binary_label(held["t"], held["delta"], tau)

        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)
        y_src = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])

        # LR ERM baseline
        auroc_lr = _auroc_safe(y_tgt, _lr_fit(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        # CORAL: align source toward target covariance, then retrain
        X_src_coral = _coral_transform(X_src_s, X_tgt_s)
        auroc_coral = _auroc_safe(y_tgt, _lr_fit(X_src_coral, y_src).predict_proba(X_tgt_s)[:, 1])

        row = {
            "center": hname, "n": held["n"],
            "lr_erm": auroc_lr, "coral": auroc_coral,
            "delta_coral": auroc_coral - auroc_lr,
        }
        rows.append(row)
        if verbose:
            print(f"  {hname:>6}  {held['n']:>4}  {auroc_lr:>8.4f}  "
                  f"{auroc_coral:>8.4f}  {auroc_coral-auroc_lr:>+8.4f}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean across centers:")
        cols = ["lr_erm", "coral", "delta_coral"]
        print(df[cols].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(tau=TAU, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_k    = run_k_sweep(centers, tau=tau, verbose=verbose)
    df_hyb  = run_hybrid(centers, tau=tau, verbose=verbose)
    df_coral = run_coral(centers, tau=tau, verbose=verbose)

    df_k.to_csv(out_path / "looco_k_sweep.csv", index=False)
    df_hyb.to_csv(out_path / "looco_hybrid.csv", index=False)
    df_coral.to_csv(out_path / "looco_coral.csv", index=False)

    if verbose:
        print(f"\nSaved to {out_path}/")

    return df_k, df_hyb, df_coral


if __name__ == "__main__":
    run_all()
