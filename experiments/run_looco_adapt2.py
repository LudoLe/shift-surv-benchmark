"""
LOOCO ablations — directions 2 & 3:
  2. TCA on full X (Transfer Component Analysis, kernel MMD alignment)
  3. MLP hybrid: MLP on [X | latent_scores(k=10)]

Both compared against LR ERM baseline.

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_adapt2
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

from data import registry_rwd as reg
from data.preprocess import impute_median
from generator.fitting import fit_reference_generator

TAU      = 12.0
K_LATENT = 10   # updated default
OUT_DIR  = _ROOT / "outputs" / "tables"


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
# TCA — Transfer Component Analysis
# Finds a subspace that minimises MMD between source and target distributions.
# Reference: Pan et al. (2011), IEEE TNN.
#
# Implementation: RBF kernel MMD, top-k eigenvectors of the generalised system.
# ---------------------------------------------------------------------------

def _rbf_kernel(X, Y, sigma=None):
    """Compute RBF kernel matrix K(X, Y) with median bandwidth heuristic."""
    sq_dists = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    if sigma is None:
        sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]) / 2.0)
        sigma = max(sigma, 1e-6)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def tca_transform(X_src, X_tgt, n_components=10, mu=1.0):
    """
    Project X_src and X_tgt into the TCA subspace (n_components dimensions)
    that minimises MMD between the two distributions.

    Parameters
    ----------
    X_src        : (n_s, p) source features
    X_tgt        : (n_t, p) target features
    n_components : TCA subspace dimension
    mu           : regularisation weight (trades off variance vs MMD)

    Returns
    -------
    Z_src : (n_s, n_components)
    Z_tgt : (n_t, n_components)
    """
    n_s, n_t = len(X_src), len(X_tgt)
    n = n_s + n_t
    X_all = np.vstack([X_src, X_tgt])

    # Kernel matrix on all data
    K = _rbf_kernel(X_all, X_all)

    # MMD matrix L
    e = np.zeros(n)
    e[:n_s] =  1.0 / n_s
    e[n_s:] = -1.0 / n_t
    L = np.outer(e, e)

    # Centering matrix H
    H = np.eye(n) - np.ones((n, n)) / n

    # Solve generalised eigen-problem: (K L K + mu I)^{-1} K H K W = lambda W
    A_mat = K @ L @ K + mu * np.eye(n)
    B_mat = K @ H @ K

    # Solve B w = lambda A w  →  A^{-1} B w = lambda w
    try:
        vals, vecs = eigh(B_mat, A_mat, subset_by_index=[n - n_components, n - 1])
    except Exception:
        # fallback: standard eig on A^{-1} B
        A_inv = np.linalg.pinv(A_mat)
        vals, vecs = np.linalg.eigh(A_inv @ B_mat)
        idx = np.argsort(vals)[::-1][:n_components]
        vecs = vecs[:, idx]

    W = vecs[:, ::-1]   # (n, n_components)  — descending eigenvalue order

    # Project: Z = K W
    Z = K @ W                  # (n, n_components)
    Z_src = Z[:n_s]
    Z_tgt = Z[n_s:]

    # Standardise for downstream LR
    mu_z  = Z_src.mean(axis=0)
    std_z = Z_src.std(axis=0) + 1e-9
    Z_src = (Z_src - mu_z) / std_z
    Z_tgt = (Z_tgt - mu_z) / std_z

    return Z_src, Z_tgt


# ---------------------------------------------------------------------------
# Experiment 2 — TCA
# ---------------------------------------------------------------------------

def run_tca(centers, tau=TAU, n_components=10, verbose=True):
    D = len(centers)
    rows = []
    if verbose:
        print(f"\n--- Direction 2: TCA (n_components={n_components}) ---")
        print(f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  {'tca':>8}  {'Δtca':>8}")
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

        # TCA: kernel-MMD alignment then LR in latent space
        try:
            Z_src, Z_tgt = tca_transform(X_src_s, X_tgt_s, n_components=n_components)
            auroc_tca = _auroc_safe(y_tgt, _lr_fit(Z_src, y_src).predict_proba(Z_tgt)[:, 1])
        except Exception as e:
            auroc_tca = float("nan")
            if verbose:
                print(f"    TCA failed for {hname}: {e}")

        row = {
            "center": hname, "n": held["n"],
            "lr_erm": auroc_lr, "tca": auroc_tca,
            "delta_tca": auroc_tca - auroc_lr,
        }
        rows.append(row)
        if verbose:
            print(f"  {hname:>6}  {held['n']:>4}  {auroc_lr:>8.4f}  "
                  f"{auroc_tca:>8.4f}  {auroc_tca-auroc_lr:>+8.4f}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean across centers:")
        cols = ["lr_erm", "tca", "delta_tca"]
        print(df[cols].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Experiment 3 — MLP hybrid: MLP on [X | latent_weibull | latent_cox]
# ---------------------------------------------------------------------------

def run_mlp_hybrid(centers, tau=TAU, k_latent=K_LATENT, verbose=True):
    D = len(centers)
    rows = []
    if verbose:
        print(f"\n--- Direction 3: MLP hybrid (X + latent_weibull + latent_cox, k={k_latent}) ---")
        print(f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  {'mlp_x':>8}  {'mlp_hyb':>9}  "
              f"{'Δhyb-lr':>8}  {'Δhyb-mlp':>9}")
        print("-" * 72)

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

        # Fit weibull + cox generators
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A, w_w, _, _ = fit_reference_generator(
                src_imp, k_total=k_latent, tau=tau, time_distribution="weibull")
            _, w_c, _, _  = fit_reference_generator(
                src_imp, k_total=k_latent, tau=tau, time_distribution="cox")

        mu_src = (X_src_s @ A).mean(axis=0)

        def _lat(X, w):
            return ((X @ A - mu_src) @ w).reshape(-1, 1)

        # Build feature matrices
        lat_src_w  = _lat(X_src_s, w_w)
        lat_tgt_w  = _lat(X_tgt_s, w_w)
        lat_src_c  = _lat(X_src_s, w_c)
        lat_tgt_c  = _lat(X_tgt_s, w_c)

        X_src_hyb = np.hstack([X_src_s, lat_src_w, lat_src_c])
        X_tgt_hyb = np.hstack([X_tgt_s, lat_tgt_w, lat_tgt_c])

        # MLP on raw X
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mlp_x = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=500, random_state=0, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20,
            )
            mlp_x.fit(X_src_s, y_src)
        auroc_mlp_x = _auroc_safe(y_tgt, mlp_x.predict_proba(X_tgt_s)[:, 1])

        # MLP on hybrid [X | latent_w | latent_c]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mlp_hyb = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=500, random_state=0, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20,
            )
            mlp_hyb.fit(X_src_hyb, y_src)
        auroc_mlp_hyb = _auroc_safe(y_tgt, mlp_hyb.predict_proba(X_tgt_hyb)[:, 1])

        # LR ERM baseline
        auroc_lr = _auroc_safe(y_tgt, _lr_fit(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        row = {
            "center": hname, "n": held["n"],
            "lr_erm": auroc_lr, "mlp_x": auroc_mlp_x, "mlp_hybrid": auroc_mlp_hyb,
            "delta_hyb_lr":  auroc_mlp_hyb - auroc_lr,
            "delta_hyb_mlp": auroc_mlp_hyb - auroc_mlp_x,
        }
        rows.append(row)
        if verbose:
            print(f"  {hname:>6}  {held['n']:>4}  {auroc_lr:>8.4f}  {auroc_mlp_x:>8.4f}  "
                  f"{auroc_mlp_hyb:>9.4f}  {auroc_mlp_hyb-auroc_lr:>+8.4f}  "
                  f"{auroc_mlp_hyb-auroc_mlp_x:>+9.4f}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean across centers:")
        cols = ["lr_erm", "mlp_x", "mlp_hybrid", "delta_hyb_lr", "delta_hyb_mlp"]
        print(df[cols].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(tau=TAU, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_tca = run_tca(centers, tau=tau, verbose=verbose)
    df_hyb = run_mlp_hybrid(centers, tau=tau, verbose=verbose)

    df_tca.to_csv(out_path / "looco_tca.csv", index=False)
    df_hyb.to_csv(out_path / "looco_mlp_hybrid.csv", index=False)

    if verbose:
        print(f"\nSaved to {out_path}/")

    return df_tca, df_hyb


if __name__ == "__main__":
    run_all()
