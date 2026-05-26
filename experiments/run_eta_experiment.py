"""
Eta experiment: η = U@w as downstream feature + hybrid CE+Cox loss for w.

Key idea
--------
Compress p features to a scalar risk score  η = (X_std @ A) @ w
where A (p×k) is PCA and w (k,) is the generator's latent risk direction.
The downstream predictor is a 1-D logistic regression on η — 1 free
parameter instead of p, directly tied to the generator's learned direction.

Hybrid w loss
-------------
  L(w) = 0.5·CE_src(w)/n_src  +  0.5·CE_tgt(w)/m
        + α_surv · (−PL_Cox(w; U_src, T_src, δ_src)) / n_events_src
        + α_reg  · ‖w − w₀‖²

where CE uses y^12 = 1[t≤12, δ=1] and PL_Cox uses full (T, δ) information.

Variants
--------
  S                  w_src from generator (source LR), 1-D LR on η
  S_ft_real(m)       w fixed, 1-D LR retrained on [src η, tgt η_m]
  w_refit_latent(m)  w updated CE-only (src+tgt), 1-D LR on η     [latent w_refit]
  hybrid_w(m)        w updated CE+Cox (src+tgt), 1-D LR on η       [NEW]
  oracle             w_src, 1-D LR on all target η
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median
from generator.fitting import fit_reference_generator


MIN_ADAPT_GAIN  = 0.003
CONSERVATIVE_LAM = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_label(t, delta, tau=12.0):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc(scores, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _eta(X_std, A, w):
    """Scalar risk score: η = (X_std @ A) @ w  →  (n,)"""
    return X_std @ A @ w


def _fit_1d_lr(eta_src, y_src, eta_m=None, y_m=None, C=1.0):
    """1-D logistic regression on scalar η (or stacked [src, tgt])."""
    if eta_m is not None and len(eta_m) > 0:
        X = np.concatenate([eta_src, eta_m])[:, None]
        y = np.concatenate([y_src, y_m])
    else:
        X = eta_src[:, None]
        y = y_src
    if len(np.unique(y)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(C=C, max_iter=500, solver="lbfgs",
                                fit_intercept=True, random_state=0)
        lr.fit(X, y)
    return lr


def _predict_1d(lr, eta):
    if lr is None:
        return np.full(len(eta), 0.5)
    return expit(eta * float(lr.coef_[0, 0]) + float(lr.intercept_[0]))


# ---------------------------------------------------------------------------
# Cox partial log-likelihood (vectorised, O(n·k))
# ---------------------------------------------------------------------------

def _cox_partial_loglik(w, U, t, delta):
    """Returns (neg_loglik, neg_grad) for minimisation."""
    order = np.argsort(t)
    t_s   = t[order]
    d_s   = delta[order].astype(bool)
    U_s   = U[order]
    eta_s = U_s @ w
    eta_s = eta_s - eta_s.max()
    exp_e = np.exp(eta_s)

    risk_sum   = np.cumsum(exp_e[::-1])[::-1]
    risk_sum_U = np.cumsum((exp_e[:, None] * U_s)[::-1], axis=0)[::-1]

    ll   = np.sum(eta_s[d_s]) - np.sum(np.log(np.maximum(risk_sum[d_s], 1e-300)))
    E_U  = risk_sum_U[d_s] / risk_sum[d_s, None]
    grad = np.sum(U_s[d_s] - E_U, axis=0)

    return -ll, -grad


# ---------------------------------------------------------------------------
# Hybrid w optimisation in latent space
# ---------------------------------------------------------------------------

def fit_w_hybrid(
    U_src, y_src, T_src, d_src,
    U_m,   y_m,
    w0,
    alpha_surv: float,
    alpha_reg:  float,
):
    """
    L(w) = 0.5·CE_src/n_src  +  0.5·CE_tgt/m
          + alpha_surv · (−PL_Cox_src) / n_events
          + alpha_reg  · ‖w − w₀‖²

    alpha_surv=0 → pure CE refit (w_refit_latent)
    alpha_surv>0 → hybrid CE + survival
    """
    k      = len(w0)
    n_src  = len(y_src)
    n_m    = max(len(y_m), 1)
    n_ev   = max(int(d_src.sum()), 1)

    def fg(w):
        # ── CE source ───────────────────────────────────────────────────────
        ls = U_src @ w
        ps = expit(np.clip(ls, -30, 30))
        pc = np.clip(ps, 1e-9, 1 - 1e-9)
        ce_s  = -float(np.mean(y_src * np.log(pc) + (1 - y_src) * np.log(1 - pc)))
        dce_s = (U_src.T @ (ps - y_src)) / n_src

        # ── CE target ───────────────────────────────────────────────────────
        lm = U_m @ w
        pm = expit(np.clip(lm, -30, 30))
        pm_c = np.clip(pm, 1e-9, 1 - 1e-9)
        ce_m  = -float(np.mean(y_m * np.log(pm_c) + (1 - y_m) * np.log(1 - pm_c)))
        dce_m = (U_m.T @ (pm - y_m)) / n_m

        ce   = 0.5 * ce_s + 0.5 * ce_m
        dce  = 0.5 * dce_s + 0.5 * dce_m

        # ── Cox survival term (source only) ─────────────────────────────────
        if alpha_surv > 0.0 and n_ev > 0:
            neg_pl, neg_grad_pl = _cox_partial_loglik(w, U_src, T_src, d_src)
            cox   = alpha_surv * neg_pl   / n_ev
            dcox  = alpha_surv * neg_grad_pl / n_ev
        else:
            cox, dcox = 0.0, np.zeros(k)

        # ── regularisation ──────────────────────────────────────────────────
        diff = w - w0
        reg  = alpha_reg * float(np.dot(diff, diff))
        dreg = 2.0 * alpha_reg * diff

        return ce + cox + reg, dce + dcox + dreg

    res = minimize(fg, w0.copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x


# ---------------------------------------------------------------------------
# Inner LOOCO for alpha_reg (shared by both w_refit_latent and hybrid_w)
# ---------------------------------------------------------------------------

def _tune_alpha_reg(
    sources_imp, scaler, A, w_src,
    T_list, d_list,
    tau, seed,
    alpha_surv,
    alpha_reg_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
    m_inner=20,
    n_seeds=5,
    conservative_lambda=CONSERVATIVE_LAM,
):
    """
    Inner source LOOCO in latent space.
    Returns best alpha_reg for the given alpha_surv.
    """
    X_list = [scaler.transform(c["X"]) for c in sources_imp]
    y_list = [_binary_label(c["t"], c["delta"], tau) for c in sources_imp]

    scores_by_reg = {r: [] for r in alpha_reg_grid}
    src_only_scores = []

    for held_i in range(len(sources_imp)):
        U_inner  = np.vstack([(X_list[j] @ A) for j in range(len(sources_imp)) if j != held_i])
        y_inner  = np.concatenate([y_list[j] for j in range(len(sources_imp)) if j != held_i])
        T_inner  = np.concatenate([T_list[j] for j in range(len(sources_imp)) if j != held_i])
        d_inner  = np.concatenate([d_list[j] for j in range(len(sources_imp)) if j != held_i])

        U_hi = X_list[held_i] @ A
        y_hi = y_list[held_i]
        n_hi = len(y_hi)

        eta_src_inner = U_inner @ w_src
        lr0 = _fit_1d_lr(eta_src_inner, y_inner)
        src_only_scores.append(_auroc(_predict_1d(lr0, U_hi @ w_src), y_hi))

        for reg in alpha_reg_grid:
            fold_scores = []
            for s in range(n_seeds):
                rng   = np.random.default_rng(seed + held_i * 97 + s * 19)
                m_use = min(m_inner, n_hi - 1)
                idx   = rng.choice(n_hi, size=m_use, replace=False)
                U_m_i = U_hi[idx]
                y_m_i = y_hi[idx]
                if len(np.unique(y_m_i)) < 2:
                    continue
                w_new = fit_w_hybrid(
                    U_inner, y_inner, T_inner, d_inner,
                    U_m_i, y_m_i,
                    w_src, alpha_surv=alpha_surv, alpha_reg=reg,
                )
                eta_new = U_hi @ w_new
                lr_new  = _fit_1d_lr(eta_src_inner, y_inner, eta_m=U_m_i @ w_new, y_m=y_m_i)
                fold_scores.append(_auroc(_predict_1d(lr_new, eta_new), y_hi))
            if fold_scores:
                scores_by_reg[reg].append(float(np.nanmean(fold_scores)))

    mean_scores  = {r: float(np.nanmean(v)) if v else 0.0 for r, v in scores_by_reg.items()}
    src_only_auc = float(np.nanmean(src_only_scores)) if src_only_scores else 0.0
    best_reg     = max(mean_scores, key=mean_scores.get)
    if mean_scores[best_reg] - src_only_auc < MIN_ADAPT_GAIN:
        best_reg = conservative_lambda
    return best_reg, mean_scores, src_only_auc


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_eta_experiment(
    m_grid=None,
    n_seeds=10,
    k_total=6,
    tau=12.0,
    nu=1.5,
    seed=42,
    alpha_surv=1.0,
    alpha_reg_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
    out_dir="outputs/tables/",
    out_prefix="eta_experiment",
    verbose=True,
    load_centers_fn=None,
):
    if m_grid is None:
        m_grid = [5, 20, 50]

    if load_centers_fn is not None:
        centers, _ = load_centers_fn()
    else:
        centers, _ = load_all_centers(registry_module=registry)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []

    for held_idx in range(len(centers)):
        held_out = centers[held_idx]
        hid      = held_out["center_id"]
        name     = registry.CENTER_NAMES.get(hid, str(hid))
        n_full   = held_out["n"]

        if verbose:
            print(f"\n=== {name} (n={n_full}, events={int(held_out['delta'].sum())}) ===")

        # ── source preprocessing ─────────────────────────────────────────────
        sources    = [c for i, c in enumerate(centers) if i != held_idx]
        X_src_pool = np.vstack([c["X"] for c in sources])
        _, src_med = impute_median(X_src_pool)
        sources_imp = [
            {**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
            for c in sources
        ]
        X_held_imp = np.where(np.isnan(held_out["X"]), src_med, held_out["X"])

        A, _, _, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
        )

        X_src_all = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        X_held_s  = scaler.transform(X_held_imp)
        y_src_all = np.concatenate([_binary_label(c["t"], c["delta"], tau) for c in sources_imp])
        y_held    = _binary_label(held_out["t"], held_out["delta"], tau)
        T_src_all = np.concatenate([c["t"]     for c in sources_imp])
        d_src_all = np.concatenate([c["delta"] for c in sources_imp])

        # ── latent projections ───────────────────────────────────────────────
        U_src = X_src_all @ A   # (N_src, k)
        U_tgt = X_held_s  @ A   # (N_tgt, k)

        # w_src from generator (source LR in latent space)
        w_src_lr = _fit_1d_lr(U_src @ np.eye(k_total)[:, 0],
                               y_src_all)  # dummy — we want w from fit_reference_generator
        # Use fit_w_hybrid with no target, no Cox, no reg to get CE-based w_src
        # Actually: fit a k-dim LR on U_src as the source w baseline
        from sklearn.linear_model import LogisticRegression as _LR
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            _lr_latent = _LR(fit_intercept=False, max_iter=1000, C=1.0, random_state=0)
            _lr_latent.fit(U_src, y_src_all)
        w_src = _lr_latent.coef_[0] / max(np.linalg.norm(_lr_latent.coef_[0]), 1e-8)

        eta_src = U_src @ w_src
        eta_tgt = U_tgt @ w_src

        # ── source downstream (1-D LR on η_src) ─────────────────────────────
        lr_S    = _fit_1d_lr(eta_src, y_src_all)
        a_S     = _auroc(_predict_1d(lr_S, eta_tgt), y_held)

        # oracle: 1-D LR on all target η
        lr_oracle = _fit_1d_lr(eta_src, y_src_all, eta_m=eta_tgt, y_m=y_held)
        a_oracle  = _auroc(_predict_1d(lr_oracle, eta_tgt), y_held)

        rows += [
            {"center": name, "variant": "S",       "m": 0,      "seed": 0, "auroc": a_S},
            {"center": name, "variant": "oracle",  "m": n_full, "seed": 0, "auroc": a_oracle},
        ]

        if verbose:
            print(f"  S={a_S:.4f}  oracle={a_oracle:.4f}")

        # ── inner LOOCO for alpha_reg ────────────────────────────────────────
        T_list = [c["t"]     for c in sources_imp]
        d_list = [c["delta"] for c in sources_imp]

        # Tune for w_refit_latent (alpha_surv=0)
        best_reg_ce, ce_scores, _ = _tune_alpha_reg(
            sources_imp, scaler, A, w_src,
            T_list, d_list, tau, seed,
            alpha_surv=0.0,
            alpha_reg_grid=alpha_reg_grid,
            m_inner=20, n_seeds=5,
        )
        # Tune for hybrid_w (alpha_surv given)
        best_reg_hyb, hyb_scores, src_auc_inner = _tune_alpha_reg(
            sources_imp, scaler, A, w_src,
            T_list, d_list, tau, seed,
            alpha_surv=alpha_surv,
            alpha_reg_grid=alpha_reg_grid,
            m_inner=20, n_seeds=5,
        )
        if verbose:
            ce_str  = "  ".join(f"r={r}:{v:.4f}" for r, v in sorted(ce_scores.items()))
            hyb_str = "  ".join(f"r={r}:{v:.4f}" for r, v in sorted(hyb_scores.items()))
            print(f"  w_refit_latent  best_reg={best_reg_ce}  [{ce_str}]")
            print(f"  hybrid_w        best_reg={best_reg_hyb}  [{hyb_str}]")

        # ── few-shot variants ─────────────────────────────────────────────────
        for m in m_grid:
            if m >= n_full:
                continue

            alist = {"S_ft_real": [], "w_refit_latent": [], "hybrid_w": []}

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                idx = rng.choice(n_full, size=m, replace=False)

                X_m  = X_held_s[idx]
                y_m  = y_held[idx]
                t_m  = held_out["t"][idx]
                d_m  = held_out["delta"][idx]
                U_m  = X_m @ A
                eta_m_src = U_m @ w_src   # η using source w

                test_mask = np.ones(n_full, dtype=bool)
                test_mask[idx] = False
                U_test    = U_tgt[test_mask]
                y_test    = y_held[test_mask]
                if len(np.unique(y_test)) < 2:
                    continue

                # S_ft_real: source w fixed, retrain 1-D LR on [src η, tgt_m η]
                lr_ft  = _fit_1d_lr(eta_src, y_src_all, eta_m=eta_m_src, y_m=y_m)
                a_ft   = _auroc(_predict_1d(lr_ft, U_test @ w_src), y_test)

                # w_refit_latent: CE only, no Cox
                if len(np.unique(y_m)) >= 2:
                    w_ce = fit_w_hybrid(
                        U_src, y_src_all, T_src_all, d_src_all,
                        U_m, y_m, w_src,
                        alpha_surv=0.0, alpha_reg=best_reg_ce,
                    )
                    lr_ce = _fit_1d_lr(eta_src, y_src_all, eta_m=U_m @ w_ce, y_m=y_m)
                    a_ce  = _auroc(_predict_1d(lr_ce, U_test @ w_ce), y_test)
                else:
                    a_ce = a_S

                # hybrid_w: CE + Cox
                if len(np.unique(y_m)) >= 2:
                    w_hyb = fit_w_hybrid(
                        U_src, y_src_all, T_src_all, d_src_all,
                        U_m, y_m, w_src,
                        alpha_surv=alpha_surv, alpha_reg=best_reg_hyb,
                    )
                    lr_hyb = _fit_1d_lr(eta_src, y_src_all, eta_m=U_m @ w_hyb, y_m=y_m)
                    a_hyb  = _auroc(_predict_1d(lr_hyb, U_test @ w_hyb), y_test)
                else:
                    a_hyb = a_S

                rows += [
                    {"center": name, "variant": "S_ft_real",      "m": m, "seed": s, "auroc": a_ft},
                    {"center": name, "variant": "w_refit_latent",  "m": m, "seed": s, "auroc": a_ce},
                    {"center": name, "variant": "hybrid_w",        "m": m, "seed": s, "auroc": a_hyb},
                ]
                alist["S_ft_real"].append(a_ft)
                alist["w_refit_latent"].append(a_ce)
                alist["hybrid_w"].append(a_hyb)

            if verbose:
                mft  = float(np.nanmean(alist["S_ft_real"]))
                mce  = float(np.nanmean(alist["w_refit_latent"]))
                mhyb = float(np.nanmean(alist["hybrid_w"]))
                print(f"  m={m:4d}  S_ft_real={mft:.4f}  "
                      f"w_refit_latent={mce:.4f}  hybrid_w={mhyb:.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out_path / f"{out_prefix}_results.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/{out_prefix}_results.csv")
        print("\nMean AUROC by variant and m (averaged over centers and seeds):")
        agg = df.groupby(["variant", "m"])["auroc"].mean().unstack("m").round(4)
        print(agg.to_string())


if __name__ == "__main__":
    run_eta_experiment(verbose=True)
