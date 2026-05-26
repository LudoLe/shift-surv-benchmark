"""
Adaptation-v3: full-feature LR + latent space only for alignment.

Key difference from v2
----------------------
v2: both prediction AND alignment in latent space (k=6) → loses signal
v3: prediction in full feature space (p=11), latent used ONLY to find θ*

Variants
--------
  S                  source LR on full features, raw target          [baseline]
  trans_mmd          source LR on transductively aligned features    [Exp 1]
  S_ft_real(m)       standard LR fine-tune on [X_src, X_m]          [supervised FT]
  w_refit(m)         regularised w update in feature space           [Exp 2]
  w_refit_aligned(m) regularised w update + transductive alignment   [Exp 3]
  S+real(full)       LR on all target data                           [oracle]

UOC fix
-------
If the inner-LOOCO best AUROC barely beats the source-only baseline
(gap < MIN_ADAPT_GAIN), adaptation is unlikely to help — default to
large λ (strong regularisation) to stay close to w_src.
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median

from generator.fitting import fit_reference_generator
from generator.shift_components import SINGLETON_GRIDS
from experiments.compare_domains import compute_mmd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ADAPT_GAIN = 0.003   # minimum inner-LOOCO improvement to trust λ grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_label(t, delta, tau=12.0):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc(probs, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


# ---------------------------------------------------------------------------
# Source model in full feature space
# ---------------------------------------------------------------------------

def fit_lr(X, y, C=1.0, seed=0):
    if len(np.unique(y)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                               class_weight="balanced", random_state=seed)
        m.fit(X, y)
    return m


def predict_proba(mdl, X):
    if mdl is None:
        return np.full(len(X), 0.5)
    return expit(X @ mdl.coef_[0] + mdl.intercept_[0])


# ---------------------------------------------------------------------------
# Latent-space alignment (θ estimation only — prediction stays in feature space)
# ---------------------------------------------------------------------------

def project_latent(X, A, mu):
    return (np.linalg.pinv(A) @ (X - mu).T).T


def _givens_inv(theta, k):
    R = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    R[0, 0], R[0, 1] =  c,  s
    R[1, 0], R[1, 1] = -s,  c
    return R


def estimate_theta_transductive(z_tgt, z_src, angle_grid):
    """θ* = argmin_θ MMD(R(-θ)@z_tgt, z_src) — full target set, no labels."""
    best_theta, best_mmd = 0.0, float("inf")
    k = z_tgt.shape[1]
    for theta in angle_grid:
        R = _givens_inv(theta, k)
        z_al = (R @ z_tgt.T).T
        mmd = compute_mmd(z_al, z_src)
        if mmd < best_mmd:
            best_mmd, best_theta = mmd, theta
    return best_theta, best_mmd


def align_features(X, A, mu_tgt, mu_src, theta):
    """
    Apply inverse rotation in full feature space:
        x_aligned = μ_src + A @ R(-θ) @ A⁺ @ (x - μ_tgt)
    """
    Xc = X - mu_tgt
    if theta == 0.0:
        return mu_src + Xc
    k   = A.shape[1]
    Ap  = np.linalg.pinv(A)
    M   = A @ _givens_inv(theta, k) @ Ap
    return mu_src + (M @ Xc.T).T


# ---------------------------------------------------------------------------
# Regularised w-refit in full feature space
# ---------------------------------------------------------------------------

def fit_w_regularized(z_src, y_src, z_m, y_m, w0, b0, lambda_reg):
    """
    min_w  mean_src L(w) + mean_tgt L(w)  +  λ·(||w-w0||² + (b-b0)²)

    Works in any feature space (full p=11 or latent k=6) — z_src/z_m are
    just the feature matrices passed in.
    """
    n_src, n_m = len(y_src), len(y_m)
    k = z_src.shape[1]
    Z = np.vstack([z_src, z_m])
    y = np.concatenate([y_src, y_m])
    sw = np.concatenate([
        np.full(n_src, 0.5 / n_src),
        np.full(n_m,   0.5 / max(n_m, 1)),
    ])

    def fg(params):
        w, b = params[:k], params[k]
        logits = Z @ w + b
        probs  = expit(np.clip(logits, -30, 30))
        pc     = np.clip(probs, 1e-9, 1 - 1e-9)
        ce     = -np.dot(sw, y * np.log(pc) + (1 - y) * np.log(1 - pc))
        reg    = lambda_reg * (float(np.dot(w - w0, w - w0)) + (b - b0) ** 2)
        res    = sw * (probs - y)
        dw     = Z.T @ res + 2.0 * lambda_reg * (w - w0)
        db     = float(np.sum(res)) + 2.0 * lambda_reg * (b - b0)
        return ce + reg, np.append(dw, db)

    res = minimize(fg, np.append(w0, b0), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x[:k], float(res.x[k])


# ---------------------------------------------------------------------------
# λ tuning: inner source LOOCO in full feature space
# ---------------------------------------------------------------------------

def tune_lambda_reg(
    sources_imp, scaler, tau, seed,
    m_inner=20,
    lambda_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
    n_seeds=10,
    conservative_lambda=50.0,
):
    """
    Inner LOOCO on source centers in the full (scaled) feature space.

    UOC fix: if best inner AUROC barely beats the source-only baseline
    (< MIN_ADAPT_GAIN), the target center probably has no shift worth
    exploiting → return conservative_lambda.

    Returns (best_lam, auroc_dict, source_only_inner_auroc)
    """
    X_list = [scaler.transform(c["X"]) for c in sources_imp]
    y_list = [_binary_label(c["t"], c["delta"], tau) for c in sources_imp]

    aurocs_by_lam = {lam: [] for lam in lambda_grid}
    source_only_aurocs = []

    for held_i in range(len(sources_imp)):
        X_inner = np.vstack([X_list[j] for j in range(len(sources_imp)) if j != held_i])
        y_inner = np.concatenate([y_list[j] for j in range(len(sources_imp)) if j != held_i])
        X_hi    = X_list[held_i]
        y_hi    = y_list[held_i]
        n_hi    = len(y_hi)

        mdl_in = fit_lr(X_inner, y_inner)
        if mdl_in is None:
            continue
        w0_i = mdl_in.coef_[0].copy()
        b0_i = float(mdl_in.intercept_[0])

        # source-only AUROC on this inner fold
        source_only_aurocs.append(_auroc(predict_proba(mdl_in, X_hi), y_hi))

        for lam in lambda_grid:
            fold_aurocs = []
            for s in range(n_seeds):
                rng   = np.random.default_rng(seed + held_i * 97 + s * 19)
                m_use = min(m_inner, n_hi - 1)
                idx   = rng.choice(n_hi, size=m_use, replace=False)
                X_m_i = X_hi[idx]
                y_m_i = y_hi[idx]
                if len(np.unique(y_m_i)) < 2:
                    continue
                w_a, b_a = fit_w_regularized(X_inner, y_inner, X_m_i, y_m_i,
                                             w0_i, b0_i, lam)
                fold_aurocs.append(_auroc(expit(X_hi @ w_a + b_a), y_hi))
            if fold_aurocs:
                aurocs_by_lam[lam].append(float(np.nanmean(fold_aurocs)))

    mean_aurocs  = {lam: float(np.nanmean(v)) if v else 0.0
                    for lam, v in aurocs_by_lam.items()}
    src_only_auc = float(np.nanmean(source_only_aurocs)) if source_only_aurocs else 0.0

    best_lam     = max(mean_aurocs, key=mean_aurocs.get)
    best_auc     = mean_aurocs[best_lam]

    # Conservative fallback: if adaptation barely helps, don't move from w_src
    if best_auc - src_only_auc < MIN_ADAPT_GAIN:
        best_lam = conservative_lambda

    return best_lam, mean_aurocs, src_only_auc


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_adaptation_v3(
    m_grid=None,
    n_seeds=20,
    k_total=6,
    tau=12.0,
    nu=1.5,
    seed=42,
    lambda_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
    conservative_lambda=50.0,
    out_dir="outputs/tables/",
    out_prefix="adaptation_v3",
    verbose=True,
    load_centers_fn=None,
):
    if m_grid is None:
        m_grid = [5, 10, 20, 30, 50]

    if load_centers_fn is not None:
        centers, _ = load_centers_fn()
    else:
        centers, _ = load_all_centers(registry_module=registry)
    out_path    = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    angle_grid  = SINGLETON_GRIDS["loading_rotation"]

    rows = []

    for held_idx in range(len(centers)):
        held_out = centers[held_idx]
        hid      = held_out["center_id"]
        name     = registry.CENTER_NAMES.get(hid, str(hid))
        n_full   = held_out["n"]

        if verbose:
            print(f"\n=== {name} (n={n_full}) ===")

        # ── source preprocessing ─────────────────────────────────────────────
        sources     = [c for i, c in enumerate(centers) if i != held_idx]
        X_src_pool  = np.vstack([c["X"] for c in sources])
        _, src_med  = impute_median(X_src_pool)
        sources_imp = [
            {**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
            for c in sources
        ]
        X_held_imp = np.where(np.isnan(held_out["X"]), src_med, held_out["X"])

        A, _, _, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
        )

        X_src_all  = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        X_held_s   = scaler.transform(X_held_imp)
        y_src_all  = np.concatenate([
            _binary_label(c["t"], c["delta"], tau) for c in sources_imp
        ])
        y_held     = _binary_label(held_out["t"], held_out["delta"], tau)

        mu_src      = X_src_all.mean(axis=0)
        mu_tgt_full = X_held_s.mean(axis=0)

        # ── latent projections for θ estimation ──────────────────────────────
        z_src       = project_latent(X_src_all, A, mu_src)
        z_held_cent = project_latent(X_held_s,  A, mu_tgt_full)  # remove mean shift

        theta_trans, _ = estimate_theta_transductive(z_held_cent, z_src, angle_grid)

        # ── aligned full-feature target (Exp 1 & 3) ──────────────────────────
        X_held_aligned = align_features(X_held_s, A, mu_tgt_full, mu_src, theta_trans)

        # ── source model in full feature space ───────────────────────────────
        mdl_src = fit_lr(X_src_all, y_src_all, seed=0)
        if mdl_src is None:
            continue
        w0 = mdl_src.coef_[0].copy()
        b0 = float(mdl_src.intercept_[0])

        # ── λ tuning via inner source LOOCO ──────────────────────────────────
        best_lam, lam_scores, src_only_auc = tune_lambda_reg(
            sources_imp, scaler, tau, seed,
            m_inner=20, lambda_grid=lambda_grid, n_seeds=10,
            conservative_lambda=conservative_lambda,
        )
        adapt_gain = max(lam_scores.values()) - src_only_auc
        if verbose:
            lam_str = "  ".join(
                f"λ={l}:{v:.4f}" for l, v in sorted(lam_scores.items())
            )
            print(f"  θ_trans={theta_trans:.3f}  best_λ={best_lam}"
                  f"  inner_gain={adapt_gain:+.4f}  [{lam_str}]")

        # ── fixed variants ────────────────────────────────────────────────────
        a_S      = _auroc(predict_proba(mdl_src, X_held_s),       y_held)
        a_trans  = _auroc(predict_proba(mdl_src, X_held_aligned),  y_held)
        mdl_full = fit_lr(
            np.vstack([X_src_all, X_held_s]),
            np.concatenate([y_src_all, y_held]),
        )
        a_oracle = _auroc(predict_proba(mdl_full, X_held_s), y_held)

        rows += [
            {"center": name, "variant": "S",            "m": 0,      "seed": 0,
             "auroc": a_S,      "theta": 0.0,         "lambda_reg": 0.0},
            {"center": name, "variant": "trans_mmd",    "m": 0,      "seed": 0,
             "auroc": a_trans,  "theta": theta_trans, "lambda_reg": 0.0},
            {"center": name, "variant": "S+real(full)", "m": n_full, "seed": 0,
             "auroc": a_oracle, "theta": 0.0,         "lambda_reg": 0.0},
        ]
        if verbose:
            print(f"  S={a_S:.4f}  trans_mmd={a_trans:.4f}  oracle={a_oracle:.4f}")

        # ── few-shot variants ─────────────────────────────────────────────────
        for m in m_grid:
            if m >= n_full:
                continue

            alist = {k: [] for k in ["S_ft_real", "w_refit", "w_refit_aligned"]}

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                idx = rng.choice(n_full, size=m, replace=False)
                X_m = X_held_s[idx]
                y_m = y_held[idx]

                # Aligned few-shot: use full target mean (transductive centering)
                X_m_aligned = align_features(X_m, A, mu_tgt_full, mu_src, theta_trans)

                # S_ft_real: plain fine-tune in feature space
                mdl_ft = fit_lr(
                    np.vstack([X_src_all, X_m]),
                    np.concatenate([y_src_all, y_m]),
                    seed=s,
                )
                a_ft = _auroc(predict_proba(mdl_ft, X_held_s), y_held)

                # w_refit: regularised update in feature space, raw few-shot
                if len(np.unique(y_m)) >= 2:
                    w_r, b_r = fit_w_regularized(
                        X_src_all, y_src_all, X_m, y_m, w0, b0, best_lam
                    )
                    a_wr = _auroc(expit(X_held_s @ w_r + b_r), y_held)
                else:
                    a_wr = a_S

                # w_refit_aligned: regularised update, aligned few-shot,
                # evaluated on aligned test
                if len(np.unique(y_m)) >= 2:
                    w_ra, b_ra = fit_w_regularized(
                        X_src_all, y_src_all, X_m_aligned, y_m, w0, b0, best_lam
                    )
                    a_wra = _auroc(expit(X_held_aligned @ w_ra + b_ra), y_held)
                else:
                    a_wra = a_trans

                rows += [
                    {"center": name, "variant": "S_ft_real",
                     "m": m, "seed": s, "auroc": a_ft,  "theta": 0.0,         "lambda_reg": best_lam},
                    {"center": name, "variant": "w_refit",
                     "m": m, "seed": s, "auroc": a_wr,  "theta": 0.0,         "lambda_reg": best_lam},
                    {"center": name, "variant": "w_refit_aligned",
                     "m": m, "seed": s, "auroc": a_wra, "theta": theta_trans, "lambda_reg": best_lam},
                ]
                alist["S_ft_real"].append(a_ft)
                alist["w_refit"].append(a_wr)
                alist["w_refit_aligned"].append(a_wra)

            if verbose:
                mft  = float(np.nanmean(alist["S_ft_real"]))
                mwr  = float(np.nanmean(alist["w_refit"]))
                mwra = float(np.nanmean(alist["w_refit_aligned"]))
                print(f"  m={m:4d}  S_ft_real={mft:.4f}  "
                      f"w_refit={mwr:.4f}  w_refit_aligned={mwra:.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out_path / f"{out_prefix}_results.csv", index=False)

    few_vars = ["S_ft_real", "w_refit", "w_refit_aligned"]
    summ = (
        df[df["variant"].isin(few_vars)]
        .groupby(["center", "variant", "m"])
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"))
        .reset_index()
    )
    for var in ["S", "trans_mmd", "S+real(full)"]:
        sub = df[df["variant"] == var].copy()
        sub = sub.rename(columns={"auroc": "auroc_mean"})
        sub["auroc_std"] = 0.0
        summ = pd.concat([summ, sub[["center", "variant", "m", "auroc_mean", "auroc_std"]]])

    summ = summ.sort_values(["center", "variant", "m"]).reset_index(drop=True)
    summ.to_csv(out_path / f"{out_prefix}_summary.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/{out_prefix}_results.csv  and  _summary.csv")
        print("\nMean AUROC by variant and m (averaged over centers and seeds):")
        agg = (
            df.groupby(["variant", "m"])["auroc"]
            .mean().unstack("m").round(4)
        )
        print(agg.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Adaptation-v3")
    p.add_argument("--m_grid",            type=int,   nargs="+", default=[5, 10, 20, 30, 50])
    p.add_argument("--n_seeds",           type=int,   default=20)
    p.add_argument("--k_total",           type=int,   default=6)
    p.add_argument("--tau",               type=float, default=12.0)
    p.add_argument("--nu",                type=float, default=1.5)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lambda_grid",       type=float, nargs="+",
                   default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--conservative_lambda", type=float, default=50.0)
    p.add_argument("--out_dir",           default="outputs/tables/")
    p.add_argument("--no_verbose",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_adaptation_v3(
        m_grid=args.m_grid,
        n_seeds=args.n_seeds,
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        lambda_grid=tuple(args.lambda_grid),
        conservative_lambda=args.conservative_lambda,
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
