"""
Controlled simulation: does the synthetic→real AUROC gap detect concept shift?

Design
------
Generate D_source (5 domains) with shared w_src via the Weibull PH generator.
Fit a source LR model on pooled D_source.

Then generate a target domain under four shift scenarios:
  no_shift        w=w_src, mu=mu_src, lambda0=lambda0_src
  cov_shift       w=w_src, mu shifted by delta, lambda0=lambda0_src
  lambda_shift    w=w_src, mu=mu_src, lambda0 scaled by factor
  concept_shift   w rotated by angle theta from w_src (varying severity)

For each scenario:
  1. Sample m=20 "real" target patients (from true shifted generator)
  2. Our tool: estimate mu_d, lambda0 from m patients (w_src assumed)
  3. Generate synthetic target population using estimated mu/lambda0, w_src
  4. Evaluate:
       AUROC_real  = source model on real target population
       AUROC_synth = source model on synthetic target population
       gap         = AUROC_synth - AUROC_real

Key prediction:
  - no_shift / cov_shift / lambda_shift → gap ≈ consistent +0.05-0.10
  - concept_shift(theta) → gap grows with theta (AUROC_real drops, synth stays high)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning

from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams, DomainParams
from generator.fitting import (
    fit_loading_matrix, fit_risk_weights,
    fit_lambda0, _km_at_tau,
)


TAU   = 12.0
NU    = 1.5
K     = 4      # latent dims (simple enough for clean simulation)
P     = 11     # observed features
N_SRC = 400    # patients per source domain
N_TGT = 300    # real target patients (evaluation)
M     = 20     # few-shot calibration sample
N_SEEDS = 20


def _binary_label(t, delta, tau=TAU):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc(scores, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _rotate_w(w_src, theta, rng):
    """Rotate w_src by angle theta toward a random perpendicular direction."""
    norm = np.linalg.norm(w_src)
    e1   = w_src / norm
    # random perpendicular vector
    v = rng.standard_normal(len(w_src))
    v -= np.dot(v, e1) * e1
    e2 = v / np.linalg.norm(v)
    return norm * (np.cos(theta) * e1 + np.sin(theta) * e2)


def _fit_source_lr(datasets):
    """Fit LR on pooled source data. Returns (model, scaler_mean, scaler_std)."""
    X = np.vstack([ds.x for ds in datasets])
    y = np.concatenate([ds.y for ds in datasets])
    mu  = X.mean(axis=0);  sd = X.std(axis=0) + 1e-8
    Xs  = (X - mu) / sd
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(C=1.0, max_iter=1000,
                                class_weight="balanced", random_state=0)
        lr.fit(Xs, y)
    return lr, mu, sd


def _estimate_target_params(X_m_raw, t_m, d_m, A, w_fit, source_lambda0_mean, rng_seed):
    """Estimate mu_d and lambda0 from m patients (our tool)."""
    # Standardise with source statistics (passed implicitly through A which was fit on std data)
    mu_m    = X_m_raw.mean(axis=0)
    U_m     = X_m_raw @ A
    mu_d    = U_m.mean(axis=0)
    U_c     = U_m - mu_d
    eta_m   = U_c @ w_fit
    if d_m.sum() >= 3:
        km_s  = _km_at_tau(t_m, d_m, TAU)
        km_prev = float(np.clip(1 - km_s, 0.01, 0.99))
        try:
            lam = fit_lambda0(eta_m, km_prev, tau=TAU, nu=NU)
        except Exception:
            lam = source_lambda0_mean
    else:
        lam = source_lambda0_mean
    return mu_d, lam


def run_concept_shift_simulation(
    theta_grid=None,
    out_dir="outputs/tables/",
    out_prefix="concept_shift_sim",
    verbose=True,
    seed=0,
):
    if theta_grid is None:
        theta_grid = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng_main = np.random.default_rng(seed)

    # ── shared loading matrix and base params ────────────────────────────────
    base = BaseParams(k_inv=K, k_spu=0, p=P, tau=TAU, seed=seed)
    gen  = SyntheticGenerator(base)
    A_shared = gen.shared_A_inv   # (P, K)

    # ── generate source domains ──────────────────────────────────────────────
    # 5 source domains with varying mu and lambda0, shared w_src
    w_src = rng_main.standard_normal(K)
    w_src /= np.linalg.norm(w_src)

    src_params = []
    for d in range(5):
        mu_d = rng_main.standard_normal(K) * 0.5
        dp = DomainParams(
            mu_inv=mu_d, mu_spu=np.zeros(0),
            A_inv=A_shared, sigma_noise=0.5,
            beta0=0.0, w=w_src.copy(), v=np.zeros(0),
            lambda0=rng_main.uniform(0.01, 0.03),
            nu=NU, n=N_SRC, domain_id=d,
        )
        src_params.append(dp)

    src_datasets = [gen.generate_domain(dp, seed=seed+d*100) for d, dp in enumerate(src_params)]

    # Fit source LR (on raw x — no separate standardisation for simplicity)
    lr_src, mu_scl, sd_scl = _fit_source_lr(src_datasets)

    # Fit A and w on pooled source (for the tool's mu/lambda0 estimation)
    X_src_pool = np.vstack([ds.x for ds in src_datasets])
    X_src_std  = (X_src_pool - mu_scl) / sd_scl
    A_fit      = fit_loading_matrix(X_src_std, K)
    y_src_pool = np.concatenate([ds.y for ds in src_datasets])
    U_src_pool = X_src_std @ A_fit
    w_fit      = fit_risk_weights(U_src_pool, y_src_pool)
    lam0_src_mean = float(np.mean([dp.lambda0 for dp in src_params]))

    # Reference: AUROC of source model on a no-shift target (expected upper bound)
    ref_params = DomainParams(
        mu_inv=np.zeros(K), mu_spu=np.zeros(0),
        A_inv=A_shared, sigma_noise=0.5,
        beta0=0.0, w=w_src.copy(), v=np.zeros(0),
        lambda0=lam0_src_mean, nu=NU, n=N_TGT, domain_id=99,
    )

    rows = []

    # ── scenario loop ────────────────────────────────────────────────────────
    scenarios = (
        [("no_shift",     {"theta": 0.0, "mu_scale": 0.0, "lam_scale": 1.0})]
        + [("cov_shift",  {"theta": 0.0, "mu_scale": s,   "lam_scale": 1.0})
           for s in [1.0, 2.0]]
        + [("lambda_shift",{"theta": 0.0,"mu_scale": 0.0, "lam_scale": s})
           for s in [0.3, 3.0]]
        + [("concept_shift",{"theta": th, "mu_scale": 0.0, "lam_scale": 1.0})
           for th in theta_grid[1:]]   # skip theta=0 (= no_shift)
    )

    for scen_name, params in scenarios:
        theta     = params["theta"]
        mu_scale  = params["mu_scale"]
        lam_scale = params["lam_scale"]

        auroc_reals  = []
        auroc_synths = []

        for s in range(N_SEEDS):
            rng = np.random.default_rng(seed + s * 997 + int(theta*1000))

            # Build target w
            w_tgt = _rotate_w(w_src, theta, rng) if theta > 0.0 else w_src.copy()

            # Build target mu
            mu_tgt = rng.standard_normal(K) * mu_scale

            # Build target lambda0
            lam_tgt = lam0_src_mean * lam_scale

            # Generate real target population
            dp_tgt = DomainParams(
                mu_inv=mu_tgt, mu_spu=np.zeros(0),
                A_inv=A_shared, sigma_noise=0.5,
                beta0=0.0, w=w_tgt, v=np.zeros(0),
                lambda0=lam_tgt, nu=NU, n=N_TGT, domain_id=10,
            )
            ds_real = gen.generate_domain(dp_tgt, seed=seed+s*7+1)
            X_real_std = (ds_real.x - mu_scl) / sd_scl
            y_real     = ds_real.y

            if len(np.unique(y_real)) < 2:
                continue

            # AUROC of source model on real target
            a_real = _auroc(lr_src.predict_proba(X_real_std)[:, 1], y_real)

            # Our tool: estimate mu_d and lambda0 from m patients
            idx_m  = rng.choice(N_TGT, size=M, replace=False)
            X_m    = X_real_std[idx_m]
            t_m    = ds_real.t_obs[idx_m]
            d_m    = ds_real.delta[idx_m]

            mu_d_est, lam_est = _estimate_target_params(
                X_m, t_m, d_m, A_fit, w_fit, lam0_src_mean, seed+s
            )

            # Generate synthetic target: tool uses w_src (its assumption), estimated mu/lambda0
            dp_synth = DomainParams(
                mu_inv=mu_d_est, mu_spu=np.zeros(0),
                A_inv=A_shared, sigma_noise=0.5,
                beta0=0.0, w=w_src.copy(),   # ← tool assumes w_src
                v=np.zeros(0),
                lambda0=lam_est, nu=NU, n=N_TGT, domain_id=11,
            )
            ds_synth   = gen.generate_domain(dp_synth, seed=seed+s*13+2)
            X_synth_std = (ds_synth.x - mu_scl) / sd_scl
            y_synth    = ds_synth.y

            if len(np.unique(y_synth)) < 2:
                continue

            a_synth = _auroc(lr_src.predict_proba(X_synth_std)[:, 1], y_synth)

            auroc_reals.append(a_real)
            auroc_synths.append(a_synth)

        a_r = float(np.nanmean(auroc_reals))
        a_s = float(np.nanmean(auroc_synths))
        gap = a_s - a_r

        label = f"{scen_name}(θ={theta:.2f})" if "concept" in scen_name else \
                f"{scen_name}(μ×{mu_scale},λ×{lam_scale})"

        rows.append({
            "scenario": scen_name, "label": label,
            "theta": theta, "mu_scale": mu_scale, "lam_scale": lam_scale,
            "auroc_real": round(a_r, 4), "auroc_synth": round(a_s, 4),
            "gap": round(gap, 4),
        })

        if verbose:
            flag = " ← CONCEPT SHIFT DETECTED" if gap > 0.15 else ""
            print(f"  {label:<35}  "
                  f"real={a_r:.4f}  synth={a_s:.4f}  gap={gap:+.4f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path / f"{out_prefix}_results.csv", index=False)
    if verbose:
        print(f"\nSaved → {out_path}/{out_prefix}_results.csv")

    return df


if __name__ == "__main__":
    print("=== Concept Shift Detection Simulation ===\n")
    run_concept_shift_simulation(verbose=True)
