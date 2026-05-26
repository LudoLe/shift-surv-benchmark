"""
Deep dive on Cox ERM: regularisation sweep + ensemble with lat_cox.

Experiments
-----------
1. alpha sweep  : Cox ERM on full X, alpha ∈ ALPHA_GRID
2. Ensemble     : convex combination of Cox ERM score + lat_cox(k=10) score
                  (grid over mixing weight λ ∈ [0,1])
3. Cox + LR     : ensemble of Cox ERM score + LR ERM score

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_cox_deep
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
from generator.fitting import fit_risk_weights_cox, fit_reference_generator

TAU        = 12.0
K_LATENT   = 10
ALPHA_GRID = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 1.0]
LAMBDA_GRID = np.linspace(0.0, 1.0, 11)   # mixing: λ·cox_erm + (1-λ)·other
OUT_DIR    = _ROOT / "outputs" / "tables"


def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
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


def _zscore(v):
    """Standardise a score vector to zero mean, unit std."""
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-9 else 1.0)


# ---------------------------------------------------------------------------
# Experiment 1 — alpha sweep
# ---------------------------------------------------------------------------

def run_alpha_sweep(centers, tau=TAU, verbose=True):
    D = len(centers)
    rows = []

    if verbose:
        alpha_hdr = "  ".join(f"{a:.0e}" for a in ALPHA_GRID)
        print(f"\n--- Exp 1: Cox ERM alpha sweep ---")
        print(f"{'center':>6}  {'n':>4}  lr_erm  " + alpha_hdr)
        print("-" * (18 + 9 * len(ALPHA_GRID)))

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]
        hname  = reg.CENTER_NAMES.get(held["center_id"], str(held["center_id"]))

        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])} for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])

        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)
        y_src     = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])
        t_src     = np.concatenate([c["t"]     for c in src_imp])
        delta_src = np.concatenate([c["delta"] for c in src_imp])
        y_tgt     = _binary_label(held["t"], held["delta"], tau)

        auroc_lr = _auroc_safe(y_tgt, _lr(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])
        row = {"center": hname, "n": held["n"], "lr_erm": auroc_lr}

        auroc_as = []
        for alpha in ALPHA_GRID:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    w = fit_risk_weights_cox(X_src_s, t_src, delta_src, alpha=alpha)
                    a = _auroc_safe(y_tgt, X_tgt_s @ w)
                except Exception:
                    a = float("nan")
            row[f"a{alpha:.0e}"] = a
            auroc_as.append(a)

        rows.append(row)
        if verbose:
            as_str = "  ".join(f"{a:>7.4f}" for a in auroc_as)
            print(f"  {hname:>6}  {held['n']:>4}  {auroc_lr:.4f}  {as_str}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean across centers:")
        cols = ["lr_erm"] + [f"a{a:.0e}" for a in ALPHA_GRID]
        print(df[cols].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Experiment 2 — Cox ERM + lat_cox ensemble
# Exp 3    — Cox ERM + LR ERM ensemble
# ---------------------------------------------------------------------------

def run_ensembles(centers, tau=TAU, best_alpha=1e-1, verbose=True):
    """
    For each fold:
      - Compute Cox ERM score (full X, best alpha)
      - Compute lat_cox score (k=10 latent)
      - Compute LR ERM score
    Then sweep mixing weight λ and report best AUROC per ensemble.
    """
    D = len(centers)

    # Store per-fold scores
    fold_scores = []   # list of dicts with 'y_tgt', 'cox', 'latcox', 'lr'

    for held_idx in range(D):
        held   = centers[held_idx]
        src_cs = [c for i, c in enumerate(centers) if i != held_idx]

        X_src_pool = np.vstack([c["X"] for c in src_cs])
        _, src_med = impute_median(X_src_pool)
        src_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])} for c in src_cs]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])

        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(np.vstack([c["X"] for c in src_imp]))
        X_tgt_s = scaler.transform(X_held_imp)
        y_src     = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in src_imp])
        t_src     = np.concatenate([c["t"]     for c in src_imp])
        delta_src = np.concatenate([c["delta"] for c in src_imp])
        y_tgt     = _binary_label(held["t"], held["delta"], tau)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Cox ERM
            w_cox = fit_risk_weights_cox(X_src_s, t_src, delta_src, alpha=best_alpha)
            score_cox = _zscore(X_tgt_s @ w_cox)

            # lat_cox k=10
            A, w_lc, _, _ = fit_reference_generator(
                src_imp, k_total=K_LATENT, tau=tau, time_distribution="cox")
            mu_src = (X_src_s @ A).mean(axis=0)
            score_latcox = _zscore((X_tgt_s @ A - mu_src) @ w_lc)

            # LR ERM
            lr = _lr(X_src_s, y_src)
            score_lr = _zscore(lr.predict_proba(X_tgt_s)[:, 1])

        fold_scores.append({
            "y_tgt": y_tgt,
            "cox": score_cox,
            "latcox": score_latcox,
            "lr": score_lr,
        })

    def _mean_auroc_ensemble(s1_key, s2_key, lam):
        """Mean AUROC of λ·s1 + (1-λ)·s2 across all folds."""
        aurocs = []
        for fs in fold_scores:
            combined = lam * fs[s1_key] + (1 - lam) * fs[s2_key]
            aurocs.append(_auroc_safe(fs["y_tgt"], combined))
        return float(np.nanmean(aurocs))

    # --- Ensemble A: Cox ERM + lat_cox ---
    best_lam_a, best_auc_a = max(
        ((lam, _mean_auroc_ensemble("cox", "latcox", lam)) for lam in LAMBDA_GRID),
        key=lambda x: x[1]
    )

    # --- Ensemble B: Cox ERM + LR ERM ---
    best_lam_b, best_auc_b = max(
        ((lam, _mean_auroc_ensemble("cox", "lr", lam)) for lam in LAMBDA_GRID),
        key=lambda x: x[1]
    )

    # Baseline aurocs
    auc_cox_only = _mean_auroc_ensemble("cox", "cox", 1.0)
    auc_lr_only  = _mean_auroc_ensemble("lr",  "lr",  1.0)
    auc_lc_only  = _mean_auroc_ensemble("latcox", "latcox", 0.0)

    if verbose:
        print(f"\n--- Exp 2 & 3: Ensembles (alpha={best_alpha:.0e}, k={K_LATENT}) ---")
        print(f"\n  Baselines (mean AUROC across {D} folds):")
        print(f"    LR ERM         : {auc_lr_only:.4f}")
        print(f"    Cox ERM        : {auc_cox_only:.4f}")
        print(f"    lat_cox (k=10) : {auc_lc_only:.4f}")
        print(f"\n  Ensemble A: λ·Cox_ERM + (1-λ)·lat_cox")
        print(f"    Best λ={best_lam_a:.2f}  →  AUROC={best_auc_a:.4f}  "
              f"(Δ vs Cox ERM: {best_auc_a-auc_cox_only:+.4f})")
        print(f"\n  Ensemble B: λ·Cox_ERM + (1-λ)·LR_ERM")
        print(f"    Best λ={best_lam_b:.2f}  →  AUROC={best_auc_b:.4f}  "
              f"(Δ vs Cox ERM: {best_auc_b-auc_cox_only:+.4f})")

        # Print λ curves
        print(f"\n  λ curve — Ensemble A (Cox_ERM + lat_cox):")
        print(f"  {'λ':>6}  {'AUROC':>8}")
        for lam in LAMBDA_GRID:
            a = _mean_auroc_ensemble("cox", "latcox", lam)
            marker = " ←" if abs(lam - best_lam_a) < 1e-9 else ""
            print(f"  {lam:>6.2f}  {a:>8.4f}{marker}")

        print(f"\n  λ curve — Ensemble B (Cox_ERM + LR_ERM):")
        print(f"  {'λ':>6}  {'AUROC':>8}")
        for lam in LAMBDA_GRID:
            a = _mean_auroc_ensemble("cox", "lr", lam)
            marker = " ←" if abs(lam - best_lam_b) < 1e-9 else ""
            print(f"  {lam:>6.2f}  {a:>8.4f}{marker}")

    return {
        "auc_lr": auc_lr_only,
        "auc_cox": auc_cox_only,
        "auc_latcox": auc_lc_only,
        "best_lam_cox_latcox": best_lam_a,
        "best_auc_cox_latcox": best_auc_a,
        "best_lam_cox_lr": best_lam_b,
        "best_auc_cox_lr": best_auc_b,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(tau=TAU, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_alpha = run_alpha_sweep(centers, tau=tau, verbose=verbose)
    df_alpha.to_csv(out_path / "looco_cox_alpha.csv", index=False)

    # Pick best alpha from sweep (max mean AUROC)
    alpha_cols = [f"a{a:.0e}" for a in ALPHA_GRID]
    best_alpha = ALPHA_GRID[int(df_alpha[alpha_cols].mean().values.argmax())]
    if verbose:
        print(f"\n  → Best alpha from sweep: {best_alpha:.0e}")

    ens = run_ensembles(centers, tau=tau, best_alpha=best_alpha, verbose=verbose)

    if verbose:
        print(f"\n=== Summary ===")
        print(f"  LR ERM              : {ens['auc_lr']:.4f}")
        print(f"  Cox ERM (best α)    : {ens['auc_cox']:.4f}")
        print(f"  lat_cox k=10        : {ens['auc_latcox']:.4f}")
        print(f"  Ens(Cox+lat_cox)    : {ens['best_auc_cox_latcox']:.4f}  "
              f"(λ={ens['best_lam_cox_latcox']:.2f})")
        print(f"  Ens(Cox+LR)         : {ens['best_auc_cox_lr']:.4f}  "
              f"(λ={ens['best_lam_cox_lr']:.2f})")

    if verbose:
        print(f"\nSaved → {out_path}/looco_cox_alpha.csv")

    return df_alpha, ens


if __name__ == "__main__":
    run_all()
