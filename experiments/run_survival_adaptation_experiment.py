"""
Survival-native adaptation experiment.

Replaces the binary 12-month LR downstream with Cox PH trained on full
(t, delta) survival tuples. This fixes two problems from the binary setting:

  1. Censored-before-12m patients (28% in MH, 25% in SZMC) now contribute
     correctly via the partial likelihood instead of being mislabelled y=0.
  2. All follow-up information is used, not just the binary 12m snapshot.

Downstream model: Cox Proportional Hazards (lifelines.CoxPHFitter)
Metric: Harrell C-index  (concordance of predicted risk with observed order)

Variants
--------
  S                  source Cox on raw target features              [baseline]
  trans_mmd          source Cox on transductively aligned features  [Exp 1]
  S_ft_real(m)       Cox retrained on [src; m raw target]           [supervised FT]
  w_refit(m)         regularised β update, raw features             [Exp 2]
  w_refit_aligned(m) regularised β update + transductive alignment  [Exp 3]
  S+real(full)       Cox on all target data                         [oracle]

Regularised β-refit (Cox version)
----------------------------------
  β*(m) = argmin_β  -PL_src(β) - PL_tgt_m(β) + λ·||β - β_src||²

where PL is the Cox partial log-likelihood. Same structure as LR w_refit
but using survival information — no label binarisation, no censoring bias.
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as ci_lifelines

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median

from generator.fitting import fit_reference_generator
from generator.shift_components import SINGLETON_GRIDS
from experiments.compare_domains import compute_mmd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ADAPT_GAIN = 0.003   # min inner-LOOCO C-index gain to trust λ grid
_COX_PENALIZER = 0.1     # L2 penalizer for CoxPHFitter (prevents divergence)


# ---------------------------------------------------------------------------
# Survival helpers
# ---------------------------------------------------------------------------

def _cindex(risk: np.ndarray, t: np.ndarray, delta: np.ndarray) -> float:
    """Harrell C-index. risk = predicted log-hazard (higher = worse prognosis)."""
    try:
        return float(ci_lifelines(t, -risk, delta))
    except Exception:
        return float("nan")


def _cox_df(X: np.ndarray, t: np.ndarray, delta: np.ndarray,
            feat_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=feat_names)
    df["T"] = t
    df["E"] = delta.astype(int)
    return df


# ---------------------------------------------------------------------------
# Cox PH fitting
# ---------------------------------------------------------------------------

def fit_cox(
    X: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
    feat_names: list[str],
    penalizer: float = _COX_PENALIZER,
) -> CoxPHFitter | None:
    """Fit CoxPHFitter on (X, t, delta). Returns None if degenerate."""
    if delta.sum() < 2:
        return None
    df  = _cox_df(X, t, delta, feat_names)
    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            cph.fit(df, duration_col="T", event_col="E")
        except Exception:
            return None
    return cph


def predict_risk(model: CoxPHFitter | None,
                 X: np.ndarray,
                 feat_names: list[str]) -> np.ndarray:
    """Predicted log-partial-hazard (linear predictor β·x)."""
    if model is None:
        return np.zeros(len(X))
    df = pd.DataFrame(X, columns=feat_names)
    return model.predict_log_partial_hazard(df).values.astype(float)


def _get_beta(model: CoxPHFitter) -> np.ndarray:
    """Extract β coefficients as numpy array (same order as feature columns)."""
    return model.params_.values.astype(float)


# ---------------------------------------------------------------------------
# Latent-space alignment (identical to v3)
# ---------------------------------------------------------------------------

def project_latent(X: np.ndarray, A: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (np.linalg.pinv(A) @ (X - mu).T).T


def _givens_inv(theta: float, k: int) -> np.ndarray:
    R = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    R[0, 0], R[0, 1] =  c,  s
    R[1, 0], R[1, 1] = -s,  c
    return R


def estimate_theta_transductive(
    z_tgt: np.ndarray, z_src: np.ndarray, angle_grid: list[float]
) -> tuple[float, float]:
    best_theta, best_mmd = 0.0, float("inf")
    k = z_tgt.shape[1]
    for theta in angle_grid:
        z_al = (_givens_inv(theta, k) @ z_tgt.T).T
        mmd  = compute_mmd(z_al, z_src)
        if mmd < best_mmd:
            best_mmd, best_theta = mmd, theta
    return best_theta, best_mmd


def align_features(
    X: np.ndarray, A: np.ndarray,
    mu_tgt: np.ndarray, mu_src: np.ndarray, theta: float
) -> np.ndarray:
    Xc = X - mu_tgt
    if theta == 0.0:
        return mu_src + Xc
    Ap = np.linalg.pinv(A)
    M  = A @ _givens_inv(theta, A.shape[1]) @ Ap
    return mu_src + (M @ Xc.T).T


# ---------------------------------------------------------------------------
# Cox partial log-likelihood (for regularised β-refit)
# ---------------------------------------------------------------------------

def _cox_partial_loglik(beta: np.ndarray,
                        X: np.ndarray,
                        t: np.ndarray,
                        delta: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Cox partial log-likelihood and gradient — fully vectorised, O(n·p).

    PL(β) = Σ_{i:δ_i=1} [ β·x_i  -  log Σ_{j:t_j≥t_i} exp(β·x_j) ]

    Returns (negative_loglik, gradient) for minimisation.
    """
    order = np.argsort(t)
    t_s   = t[order]
    d_s   = delta[order].astype(bool)
    X_s   = X[order]                          # (n, p)
    eta_s = X_s @ beta                        # (n,)

    # Numerically stable: subtract max before exp
    eta_s  = eta_s - eta_s.max()
    exp_e  = np.exp(eta_s)                    # (n,)

    # Risk-set sums via reverse cumsum  (risk set at position i = all j >= i)
    risk_sum   = np.cumsum(exp_e[::-1])[::-1]            # (n,)  scalar sum
    risk_sum_X = np.cumsum((exp_e[:, None] * X_s)[::-1],
                            axis=0)[::-1]                # (n, p) weighted sum

    # Log-likelihood (events only)
    ll = np.sum(eta_s[d_s]) - np.sum(np.log(np.maximum(risk_sum[d_s], 1e-300)))

    # Gradient  Σ_{events} [ x_i - E[X | risk set_i] ]
    E_X  = risk_sum_X[d_s] / risk_sum[d_s, None]        # (n_events, p)
    grad = np.sum(X_s[d_s] - E_X, axis=0)               # (p,)

    return -ll, -grad


def fit_cox_regularized(
    X_src: np.ndarray, t_src: np.ndarray, d_src: np.ndarray,
    X_m:   np.ndarray, t_m:   np.ndarray, d_m:   np.ndarray,
    beta0: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    """
    min_β  -PL_src(β) - PL_tgt_m(β)  +  λ·||β - β0||²

    Equal weight on source and target partial likelihoods.
    Returns adapted β.
    """
    p = len(beta0)

    def fg(b: np.ndarray) -> tuple[float, np.ndarray]:
        l_src, g_src = _cox_partial_loglik(b, X_src, t_src, d_src)
        l_m,   g_m   = _cox_partial_loglik(b, X_m,   t_m,   d_m)
        reg           = lambda_reg * float(np.dot(b - beta0, b - beta0))
        dreg          = 2.0 * lambda_reg * (b - beta0)
        # normalise each term so they contribute equally regardless of set size
        return (l_src / len(d_src) + l_m / max(d_m.sum(), 1)
                + reg,
                g_src / len(d_src) + g_m / max(d_m.sum(), 1)
                + dreg)

    res = minimize(fg, beta0.copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x


# ---------------------------------------------------------------------------
# λ tuning: inner source LOOCO (survival version)
# ---------------------------------------------------------------------------

def tune_lambda_reg(
    sources_imp: list[dict],
    scaler,
    feat_names: list[str],
    seed: int,
    m_inner: int = 20,
    lambda_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_seeds: int = 10,
    conservative_lambda: float = 50.0,
) -> tuple[float, dict, float]:

    X_list = [scaler.transform(c["X"]) for c in sources_imp]
    t_list = [c["t"]     for c in sources_imp]
    d_list = [c["delta"] for c in sources_imp]

    aurocs_by_lam: dict[float, list[float]] = {l: [] for l in lambda_grid}
    src_only: list[float] = []

    for held_i in range(len(sources_imp)):
        X_inner = np.vstack([X_list[j] for j in range(len(sources_imp)) if j != held_i])
        t_inner = np.concatenate([t_list[j] for j in range(len(sources_imp)) if j != held_i])
        d_inner = np.concatenate([d_list[j] for j in range(len(sources_imp)) if j != held_i])
        X_hi, t_hi, d_hi = X_list[held_i], t_list[held_i], d_list[held_i]
        n_hi = len(t_hi)

        mdl_in = fit_cox(X_inner, t_inner, d_inner, feat_names)
        if mdl_in is None:
            continue
        beta0_i = _get_beta(mdl_in)
        src_only.append(_cindex(predict_risk(mdl_in, X_hi, feat_names), t_hi, d_hi))

        for lam in lambda_grid:
            fold_ci = []
            for s in range(n_seeds):
                rng   = np.random.default_rng(seed + held_i * 97 + s * 19)
                m_use = min(m_inner, n_hi - 1)
                idx   = rng.choice(n_hi, size=m_use, replace=False)
                X_m_i, t_m_i, d_m_i = X_hi[idx], t_hi[idx], d_hi[idx]
                if d_m_i.sum() < 2:
                    continue
                beta_a = fit_cox_regularized(
                    X_inner, t_inner, d_inner,
                    X_m_i,  t_m_i,   d_m_i,
                    beta0_i, lam,
                )
                fold_ci.append(_cindex(X_hi @ beta_a, t_hi, d_hi))
            if fold_ci:
                aurocs_by_lam[lam].append(float(np.nanmean(fold_ci)))

    mean_ci = {l: float(np.nanmean(v)) if v else 0.0
               for l, v in aurocs_by_lam.items()}
    src_ci  = float(np.nanmean(src_only)) if src_only else 0.0
    best    = max(mean_ci, key=mean_ci.get)

    if mean_ci[best] - src_ci < MIN_ADAPT_GAIN:
        best = conservative_lambda

    return best, mean_ci, src_ci


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_survival_adaptation(
    m_grid: list[int] | None = None,
    n_seeds: int = 20,
    k_total: int = 6,
    nu: float = 1.5,
    seed: int = 42,
    lambda_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    conservative_lambda: float = 50.0,
    out_dir: str = "outputs/tables/",
    out_prefix: str = "survival_adaptation",
    verbose: bool = True,
    load_centers_fn=None,
) -> None:

    if m_grid is None:
        m_grid = [5, 10, 20, 30, 50]

    if load_centers_fn is not None:
        centers, _ = load_centers_fn()
    else:
        centers, _ = load_all_centers(registry_module=registry)
    out_path   = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    angle_grid = SINGLETON_GRIDS["loading_rotation"]

    rows: list[dict] = []

    for held_idx in range(len(centers)):
        held_out = centers[held_idx]
        hid      = held_out["center_id"]
        name     = registry.CENTER_NAMES.get(hid, str(hid))
        n_full   = held_out["n"]
        t_held   = held_out["t"]
        d_held   = held_out["delta"]

        if verbose:
            print(f"\n=== {name} (n={n_full}, events={int(d_held.sum())}) ===")

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
            source_centers=sources_imp, k_total=k_total, tau=12.0, nu=nu,
        )

        X_src_all = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        X_held_s  = scaler.transform(X_held_imp)
        t_src_all = np.concatenate([c["t"]     for c in sources_imp])
        d_src_all = np.concatenate([c["delta"] for c in sources_imp])

        p         = X_src_all.shape[1]
        feat_names = [f"f{i}" for i in range(p)]

        mu_src      = X_src_all.mean(axis=0)
        mu_tgt_full = X_held_s.mean(axis=0)

        # ── latent alignment ─────────────────────────────────────────────────
        z_src       = project_latent(X_src_all, A, mu_src)
        z_held_cent = project_latent(X_held_s,  A, mu_tgt_full)
        theta_trans, _ = estimate_theta_transductive(z_held_cent, z_src, angle_grid)
        X_held_aligned = align_features(X_held_s, A, mu_tgt_full, mu_src, theta_trans)

        # ── source Cox model ─────────────────────────────────────────────────
        mdl_src = fit_cox(X_src_all, t_src_all, d_src_all, feat_names)
        if mdl_src is None:
            if verbose:
                print("  source Cox failed, skipping")
            continue
        beta_src = _get_beta(mdl_src)

        # ── λ tuning ─────────────────────────────────────────────────────────
        best_lam, lam_scores, src_ci = tune_lambda_reg(
            sources_imp, scaler, feat_names, seed,
            m_inner=20, lambda_grid=lambda_grid, n_seeds=10,
            conservative_lambda=conservative_lambda,
        )
        adapt_gain = max(lam_scores.values()) - src_ci
        if verbose:
            lam_str = "  ".join(f"λ={l}:{v:.4f}" for l, v in sorted(lam_scores.items()))
            print(f"  θ_trans={theta_trans:.3f}  best_λ={best_lam}"
                  f"  inner_gain={adapt_gain:+.4f}  [{lam_str}]")

        # ── fixed variants ────────────────────────────────────────────────────
        ci_S     = _cindex(predict_risk(mdl_src, X_held_s,       feat_names), t_held, d_held)
        ci_trans = _cindex(predict_risk(mdl_src, X_held_aligned,  feat_names), t_held, d_held)

        mdl_full = fit_cox(
            np.vstack([X_src_all, X_held_s]),
            np.concatenate([t_src_all, t_held]),
            np.concatenate([d_src_all, d_held]),
            feat_names,
        )
        ci_oracle = _cindex(predict_risk(mdl_full, X_held_s, feat_names), t_held, d_held)

        rows += [
            {"center": name, "variant": "S",            "m": 0,      "seed": 0, "cindex": ci_S,      "theta": 0.0,         "lambda_reg": 0.0},
            {"center": name, "variant": "trans_mmd",    "m": 0,      "seed": 0, "cindex": ci_trans,  "theta": theta_trans, "lambda_reg": 0.0},
            {"center": name, "variant": "S+real(full)", "m": n_full, "seed": 0, "cindex": ci_oracle, "theta": 0.0,         "lambda_reg": 0.0},
        ]
        if verbose:
            print(f"  S={ci_S:.4f}  trans_mmd={ci_trans:.4f}  oracle={ci_oracle:.4f}")

        # ── few-shot variants ─────────────────────────────────────────────────
        for m in m_grid:
            if m >= n_full:
                continue

            alist: dict[str, list] = {
                k: [] for k in ["S_ft_real", "w_refit", "w_refit_aligned"]
            }

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                # Sample m patients NOT used for test evaluation
                idx     = rng.choice(n_full, size=m, replace=False)
                test_mask = np.ones(n_full, dtype=bool)
                test_mask[idx] = False

                X_m  = X_held_s[idx];    t_m  = t_held[idx];   d_m  = d_held[idx]
                X_te = X_held_s[test_mask]; t_te = t_held[test_mask]; d_te = d_held[test_mask]

                X_m_al = align_features(X_m, A, mu_tgt_full, mu_src, theta_trans)
                X_te_al = align_features(X_te, A, mu_tgt_full, mu_src, theta_trans)

                # S_ft_real
                mdl_ft = fit_cox(
                    np.vstack([X_src_all, X_m]),
                    np.concatenate([t_src_all, t_m]),
                    np.concatenate([d_src_all, d_m]),
                    feat_names,
                )
                ci_ft = _cindex(predict_risk(mdl_ft, X_te, feat_names), t_te, d_te)

                # w_refit (raw)
                if d_m.sum() >= 2:
                    beta_r = fit_cox_regularized(
                        X_src_all, t_src_all, d_src_all,
                        X_m, t_m, d_m, beta_src, best_lam,
                    )
                    ci_wr = _cindex(X_te @ beta_r, t_te, d_te)
                else:
                    ci_wr = ci_S

                # w_refit_aligned
                if d_m.sum() >= 2:
                    beta_ra = fit_cox_regularized(
                        X_src_all, t_src_all, d_src_all,
                        X_m_al, t_m, d_m, beta_src, best_lam,
                    )
                    ci_wra = _cindex(X_te_al @ beta_ra, t_te, d_te)
                else:
                    ci_wra = ci_trans

                rows += [
                    {"center": name, "variant": "S_ft_real",       "m": m, "seed": s, "cindex": ci_ft,  "theta": 0.0,         "lambda_reg": best_lam},
                    {"center": name, "variant": "w_refit",          "m": m, "seed": s, "cindex": ci_wr,  "theta": 0.0,         "lambda_reg": best_lam},
                    {"center": name, "variant": "w_refit_aligned",  "m": m, "seed": s, "cindex": ci_wra, "theta": theta_trans, "lambda_reg": best_lam},
                ]
                alist["S_ft_real"].append(ci_ft)
                alist["w_refit"].append(ci_wr)
                alist["w_refit_aligned"].append(ci_wra)

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
        .agg(cindex_mean=("cindex", "mean"), cindex_std=("cindex", "std"))
        .reset_index()
    )
    for var in ["S", "trans_mmd", "S+real(full)"]:
        sub = df[df["variant"] == var].copy()
        sub = sub.rename(columns={"cindex": "cindex_mean"})
        sub["cindex_std"] = 0.0
        summ = pd.concat([summ, sub[["center", "variant", "m", "cindex_mean", "cindex_std"]]])

    summ = summ.sort_values(["center", "variant", "m"]).reset_index(drop=True)
    summ.to_csv(out_path / f"{out_prefix}_summary.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/{out_prefix}_results.csv  and  _summary.csv")
        print("\nMean C-index by variant and m (averaged over centers and seeds):")
        agg = df.groupby(["variant", "m"])["cindex"].mean().unstack("m").round(4)
        print(agg.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Survival-native adaptation")
    p.add_argument("--m_grid",              type=int,   nargs="+", default=[5, 10, 20, 30, 50])
    p.add_argument("--n_seeds",             type=int,   default=20)
    p.add_argument("--k_total",             type=int,   default=6)
    p.add_argument("--nu",                  type=float, default=1.5)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--lambda_grid",         type=float, nargs="+",
                   default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--conservative_lambda", type=float, default=50.0)
    p.add_argument("--out_dir",             default="outputs/tables/")
    p.add_argument("--no_verbose",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_survival_adaptation(
        m_grid=args.m_grid,
        n_seeds=args.n_seeds,
        k_total=args.k_total,
        nu=args.nu,
        seed=args.seed,
        lambda_grid=tuple(args.lambda_grid),
        conservative_lambda=args.conservative_lambda,
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
