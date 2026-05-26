"""
Adaptation-v2: three experiments testing discriminative objectives.

Exp 1 — Transductive MMD alignment (ZERO labels)
   θ* = argmin_θ  MMD( R(-θ) @ z_tgt_all,  z_src )
   Uses ALL target feature observations (no outcomes). Source LR predicts on
   aligned latents.

Exp 2 — Regularised w-refit in latent space (m labels)
   w*(m) = argmin_w  mean_src L(w) + mean_tgt L(w)  +  λ · ||w - w0||²
   λ tuned by inner source LOOCO. No alignment applied.

Exp 3 — Combined: transductive alignment + regularised w-refit (m labels)
   Same as Exp 2 but both training few-shot and test features are first
   aligned via θ* from Exp 1.

All models operate in the k-dimensional PCA latent space z = A⁺(X - μ).

Variants
--------
  S                 source LR in latent space, raw target latents       [baseline]
  trans_mmd         source LR, transductively MMD-aligned latents       [Exp 1]
  S_ft_real(m)      LR retrained on [z_src; z_m], raw latents           [supervised FT]
  w_refit(m)        regularised w update, raw latents                   [Exp 2]
  w_refit_aligned(m) regularised w update + transductive alignment      [Exp 3]
  S+real(full)      LR retrained on full z_held                         [oracle]

Key comparisons
---------------
  trans_mmd         vs S             — Exp 1: unsupervised alignment benefit
  w_refit           vs S_ft_real     — Exp 2: regularised vs plain FT in latent
  w_refit_aligned   vs w_refit       — Exp 3: does alignment help on top of w-refit?
  w_refit           vs S+real(full)  — gap to supervised oracle

Usage
-----
  python -m experiments.run_adaptation_v2_experiment [options]
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
# Helpers
# ---------------------------------------------------------------------------

def _binary_label(t: np.ndarray, delta: np.ndarray, tau: float = 12.0) -> np.ndarray:
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc(probs: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


# ---------------------------------------------------------------------------
# Latent-space geometry
# ---------------------------------------------------------------------------

def project_latent(X: np.ndarray, A: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """z = A⁺ @ (X - mu)   shape: (n, k)"""
    return (np.linalg.pinv(A) @ (X - mu).T).T


def _givens_inv(theta: float, k: int) -> np.ndarray:
    """R(-θ) in the (0,1) plane of k-dim space."""
    R = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    R[0, 0], R[0, 1] =  c,  s
    R[1, 0], R[1, 1] = -s,  c
    return R


def align_latent(z: np.ndarray, theta: float) -> np.ndarray:
    """Apply R(-θ) to rows of z."""
    if theta == 0.0:
        return z
    return (_givens_inv(theta, z.shape[1]) @ z.T).T


def estimate_theta_transductive(
    z_tgt: np.ndarray,
    z_src: np.ndarray,
    angle_grid: list[float],
) -> tuple[float, float]:
    """
    θ* = argmin_θ MMD( R(-θ) @ z_tgt, z_src )

    Uses the FULL target set — stable because no label information required.
    Returns (best_theta, best_mmd).
    """
    best_theta, best_mmd = 0.0, float("inf")
    for theta in angle_grid:
        mmd = compute_mmd(align_latent(z_tgt, theta), z_src)
        if mmd < best_mmd:
            best_mmd, best_theta = mmd, theta
    return best_theta, best_mmd


# ---------------------------------------------------------------------------
# Source model in latent space
# ---------------------------------------------------------------------------

def fit_lr_latent(
    z: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    seed: int = 0,
) -> LogisticRegression | None:
    """LR on k-dimensional latent space."""
    if len(np.unique(y)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mdl = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=seed,
        )
        mdl.fit(z, y)
    return mdl


def _predict_proba_latent(mdl: LogisticRegression | None, z: np.ndarray) -> np.ndarray:
    if mdl is None:
        return np.full(len(z), 0.5)
    return expit(z @ mdl.coef_[0] + mdl.intercept_[0])


# ---------------------------------------------------------------------------
# Regularised w-refit
# ---------------------------------------------------------------------------

def fit_w_regularized(
    z_src: np.ndarray,
    y_src: np.ndarray,
    z_m: np.ndarray,
    y_m: np.ndarray,
    w0: np.ndarray,
    b0: float,
    lambda_reg: float,
) -> tuple[np.ndarray, float]:
    """
    Minimise:
        mean_over_src L(w,b) + mean_over_tgt L(w,b) + λ · (||w-w0||² + (b-b0)²)

    Each sample set contributes equally (mean, not sum), so the balance between
    source and target is determined only by λ, not by set sizes.

    Parameters
    ----------
    w0, b0      : source-model weights — regularisation target
    lambda_reg  : regularisation strength (larger → stay closer to w0)

    Returns
    -------
    (w_adapted, b_adapted)
    """
    n_src, n_m = len(y_src), len(y_m)
    k = z_src.shape[1]

    Z  = np.vstack([z_src, z_m])
    y  = np.concatenate([y_src, y_m])
    # Equal per-set weight: each set contributes 0.5 regardless of size
    sw = np.concatenate([
        np.full(n_src, 0.5 / n_src),
        np.full(n_m,   0.5 / max(n_m, 1)),
    ])

    def loss_and_grad(params: np.ndarray):
        w, b = params[:k], params[k]
        logits  = Z @ w + b
        probs   = expit(np.clip(logits, -30, 30))
        pc      = np.clip(probs, 1e-9, 1 - 1e-9)
        ce      = -np.dot(sw, y * np.log(pc) + (1 - y) * np.log(1 - pc))
        dw_c    = np.zeros(k)
        db_c    = np.zeros(k)     # placeholder shape
        res     = sw * (probs - y)
        dw      = Z.T @ res + 2.0 * lambda_reg * (w - w0)
        db      = float(np.sum(res)) + 2.0 * lambda_reg * (b - b0)
        reg     = lambda_reg * (float(np.dot(w - w0, w - w0)) + (b - b0) ** 2)
        return ce + reg, np.append(dw, db)

    x0  = np.append(w0, b0)
    res = minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x[:k], float(res.x[k])


# ---------------------------------------------------------------------------
# λ tuning: inner source LOOCO
# ---------------------------------------------------------------------------

def tune_lambda_reg(
    sources_imp: list[dict],
    A: np.ndarray,
    mu_src: np.ndarray,
    scaler,
    tau: float,
    seed: int,
    m_inner: int = 20,
    lambda_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_seeds: int = 10,
) -> tuple[float, dict[float, float]]:
    """
    Hold out each source center in turn, treat it as a mini-target with
    m_inner labelled patients, and pick the λ that maximises mean AUROC.

    Reuses the outer A and scaler (fit on all 5 source centers) — small bias
    but avoids refitting the generator 5 more times per outer fold.
    """
    # Pre-project all source centers once
    z_list = [
        project_latent(scaler.transform(c["X"]), A, mu_src)
        for c in sources_imp
    ]
    y_list = [_binary_label(c["t"], c["delta"], tau) for c in sources_imp]

    aurocs_by_lam: dict[float, list[float]] = {lam: [] for lam in lambda_grid}

    for held_i in range(len(sources_imp)):
        z_inner = np.vstack([z_list[j] for j in range(len(sources_imp)) if j != held_i])
        y_inner = np.concatenate([y_list[j] for j in range(len(sources_imp)) if j != held_i])
        z_hi    = z_list[held_i]
        y_hi    = y_list[held_i]
        n_hi    = len(y_hi)

        mdl_in = fit_lr_latent(z_inner, y_inner)
        if mdl_in is None:
            continue
        w0_i = mdl_in.coef_[0].copy()
        b0_i = float(mdl_in.intercept_[0])

        for lam in lambda_grid:
            fold_aurocs: list[float] = []
            for s in range(n_seeds):
                rng   = np.random.default_rng(seed + held_i * 97 + s * 19)
                m_use = min(m_inner, n_hi - 1)
                idx   = rng.choice(n_hi, size=m_use, replace=False)
                z_m_i = z_hi[idx]
                y_m_i = y_hi[idx]
                if len(np.unique(y_m_i)) < 2:
                    continue
                w_a, b_a = fit_w_regularized(
                    z_inner, y_inner, z_m_i, y_m_i, w0_i, b0_i, lam
                )
                fold_aurocs.append(_auroc(expit(z_hi @ w_a + b_a), y_hi))
            if fold_aurocs:
                aurocs_by_lam[lam].append(float(np.nanmean(fold_aurocs)))

    mean_aurocs = {
        lam: float(np.nanmean(v)) if v else 0.0
        for lam, v in aurocs_by_lam.items()
    }
    best_lam = max(mean_aurocs, key=mean_aurocs.get)
    return best_lam, mean_aurocs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_adaptation_v2(
    m_grid: list[int] | None = None,
    n_seeds: int = 20,
    k_total: int = 6,
    tau: float = 12.0,
    nu: float = 1.5,
    seed: int = 42,
    lambda_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    out_dir: str = "outputs/tables/",
    verbose: bool = True,
) -> None:

    if m_grid is None:
        m_grid = [5, 10, 20, 30, 50]

    centers, _ = load_all_centers(registry_module=registry)
    out_path    = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    angle_grid  = SINGLETON_GRIDS["loading_rotation"]

    rows: list[dict] = []

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

        mu_src      = X_src_all.mean(axis=0)     # source feature mean (std space)
        mu_tgt_full = X_held_s.mean(axis=0)      # full target feature mean

        # ── latent projections ────────────────────────────────────────────────
        # Source: centred on source mean  →  z_src ~ N(0, I) approximately
        z_src = project_latent(X_src_all, A, mu_src)

        # Target raw: centred on source mean (preserves mean shift in latent)
        z_held_raw = project_latent(X_held_s, A, mu_src)

        # Target centred on own mean (removes mean shift, used for MMD rotation search)
        z_held_cent = project_latent(X_held_s, A, mu_tgt_full)

        # ── Exp 1: transductive MMD alignment ────────────────────────────────
        theta_trans, mmd_trans = estimate_theta_transductive(
            z_held_cent, z_src, angle_grid
        )
        # Aligned target latents (source frame)
        z_held_aligned = align_latent(z_held_cent, theta_trans)

        # ── source model in latent space ─────────────────────────────────────
        mdl_src = fit_lr_latent(z_src, y_src_all, seed=0)
        if mdl_src is None:
            continue
        w0 = mdl_src.coef_[0].copy()
        b0 = float(mdl_src.intercept_[0])

        # ── λ tuning ─────────────────────────────────────────────────────────
        best_lam, lam_scores = tune_lambda_reg(
            sources_imp, A, mu_src, scaler, tau,
            seed=seed, m_inner=20, lambda_grid=lambda_grid, n_seeds=10,
        )
        if verbose:
            lam_str = "  ".join(f"λ={l}→{v:.4f}" for l, v in sorted(lam_scores.items()))
            print(f"  θ_trans={theta_trans:.3f}  best_λ={best_lam}  [{lam_str}]")

        # ── fixed variants (no m) ─────────────────────────────────────────────
        a_S     = _auroc(expit(z_held_raw     @ w0 + b0), y_held)
        a_trans = _auroc(expit(z_held_aligned @ w0 + b0), y_held)

        # Oracle: retrain on all held-out latents
        mdl_oracle = fit_lr_latent(
            np.vstack([z_src, z_held_raw]),
            np.concatenate([y_src_all, y_held]),
            seed=0,
        )
        a_oracle = _auroc(
            _predict_proba_latent(mdl_oracle, z_held_raw), y_held
        )

        rows += [
            {"center": name, "variant": "S",            "m": 0,      "seed": 0,
             "auroc": a_S,      "theta": 0.0,         "lambda_reg": 0.0},
            {"center": name, "variant": "trans_mmd",    "m": 0,      "seed": 0,
             "auroc": a_trans,  "theta": theta_trans, "lambda_reg": 0.0},
            {"center": name, "variant": "S+real(full)", "m": n_full, "seed": 0,
             "auroc": a_oracle, "theta": 0.0,         "lambda_reg": 0.0},
        ]

        if verbose:
            print(f"  S={a_S:.4f}  trans_mmd={a_trans:.4f}  S+real(full)={a_oracle:.4f}")

        # ── few-shot variants ──────────────────────────────────────────────────
        for m in m_grid:
            if m >= n_full:
                if verbose:
                    print(f"  skipping m={m} (>= n_full={n_full})")
                continue

            alist: dict[str, list[float]] = {
                k: [] for k in ["S_ft_real", "w_refit", "w_refit_aligned"]
            }

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                idx = rng.choice(n_full, size=m, replace=False)

                X_m   = X_held_s[idx]
                y_m   = y_held[idx]

                # Raw few-shot latents (centred on source mean, like z_src)
                z_m_raw = project_latent(X_m, A, mu_src)

                # Aligned few-shot latents (use FULL target mean for centering —
                # transductive: we have all features at deployment time)
                z_m_cent    = project_latent(X_m, A, mu_tgt_full)
                z_m_aligned = align_latent(z_m_cent, theta_trans)

                # ── S_ft_real: plain LR fine-tune on raw latents ──────────────
                mdl_ft = fit_lr_latent(
                    np.vstack([z_src, z_m_raw]),
                    np.concatenate([y_src_all, y_m]),
                    seed=s,
                )
                a_ft = _auroc(_predict_proba_latent(mdl_ft, z_held_raw), y_held)

                # ── w_refit: regularised toward w0, raw latents ───────────────
                if len(np.unique(y_m)) >= 2:
                    w_a, b_a = fit_w_regularized(
                        z_src, y_src_all, z_m_raw, y_m, w0, b0, best_lam
                    )
                    a_wr = _auroc(expit(z_held_raw @ w_a + b_a), y_held)
                else:
                    a_wr = a_S   # degenerate fallback

                # ── w_refit_aligned: regularised toward w0, aligned latents ───
                if len(np.unique(y_m)) >= 2:
                    w_aa, b_aa = fit_w_regularized(
                        z_src, y_src_all, z_m_aligned, y_m, w0, b0, best_lam
                    )
                    a_wra = _auroc(expit(z_held_aligned @ w_aa + b_aa), y_held)
                else:
                    a_wra = a_trans   # degenerate fallback

                rows += [
                    {"center": name, "variant": "S_ft_real",        "m": m, "seed": s,
                     "auroc": a_ft,  "theta": 0.0,         "lambda_reg": best_lam},
                    {"center": name, "variant": "w_refit",           "m": m, "seed": s,
                     "auroc": a_wr,  "theta": 0.0,         "lambda_reg": best_lam},
                    {"center": name, "variant": "w_refit_aligned",   "m": m, "seed": s,
                     "auroc": a_wra, "theta": theta_trans, "lambda_reg": best_lam},
                ]
                alist["S_ft_real"].append(a_ft)
                alist["w_refit"].append(a_wr)
                alist["w_refit_aligned"].append(a_wra)

            if verbose:
                mft  = float(np.nanmean(alist["S_ft_real"]))
                mwr  = float(np.nanmean(alist["w_refit"]))
                mwra = float(np.nanmean(alist["w_refit_aligned"]))
                print(f"  m={m:4d}  S_ft_real={mft:.4f}  w_refit={mwr:.4f}"
                      f"  w_refit_aligned={mwra:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "adaptation_v2_results.csv", index=False)

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
        summ = pd.concat([
            summ, sub[["center", "variant", "m", "auroc_mean", "auroc_std"]]
        ])

    summ = summ.sort_values(["center", "variant", "m"]).reset_index(drop=True)
    summ.to_csv(out_path / "adaptation_v2_summary.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/adaptation_v2_results.csv  and  _summary.csv")
        print("\nMean AUROC by variant and m (averaged over centers and seeds):")
        agg = (
            df.groupby(["variant", "m"])["auroc"]
            .mean()
            .unstack("m")
            .round(4)
        )
        print(agg.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptation-v2: discriminative objectives")
    p.add_argument("--m_grid",       type=int,   nargs="+", default=[5, 10, 20, 30, 50])
    p.add_argument("--n_seeds",      type=int,   default=20)
    p.add_argument("--k_total",      type=int,   default=6)
    p.add_argument("--tau",          type=float, default=12.0)
    p.add_argument("--nu",           type=float, default=1.5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--lambda_grid",  type=float, nargs="+",
                   default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--out_dir",      default="outputs/tables/")
    p.add_argument("--no_verbose",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_adaptation_v2(
        m_grid=args.m_grid,
        n_seeds=args.n_seeds,
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        lambda_grid=tuple(args.lambda_grid),
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
