"""
Two analyses on the full RWD dataset:

A) Shift detection: does the generator flag a concept shift between
   IO_LINE=1 (first-line) patients and IO_LINE>=2 (later-line) patients?

   Method: train generator on per-center 1L patients, compute
   AUROC_synth - AUROC_real gap on 2L+ patients (pooled across centers).
   Large positive gap → concept shift between cohorts.

B) Discriminative transport with 22 RWD features:
   Does the expanded feature set improve AUROC vs the 11-feature baseline?
   Runs source-only (S) and best adaptation variants (w_refit_aligned)
   using registry_rwd.load_all_centers() with all 22 features.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from data import registry_rwd
from generator.fitting import (
    fit_reference_generator, fit_loading_matrix,
    fit_risk_weights, fit_lambda0, _km_at_tau,
)
from data.preprocess import impute_median


TAU = 12.0
NU  = 1.5
K   = 6


# ── helpers ─────────────────────────────────────────────────────────────────

def _binary_label(t, delta, tau=TAU):
    return ((t <= tau) & (delta == 1)).astype(int)

def _auroc(p, y):
    if len(np.unique(y)) < 2 or np.isnan(p).any():
        return float("nan")
    return float(roc_auc_score(y, p))

def _fit_lr(X, y, C=1.0, seed=0):
    if len(np.unique(y)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                               class_weight="balanced", random_state=seed)
        m.fit(X, y)
    return m

def _generate_t(X_std, A, w, lam0, nu, rng):
    U    = X_std @ A
    mu_d = U.mean(axis=0)
    eta  = (U - mu_d) @ w
    U_rv = rng.uniform(0, 1, size=len(X_std))
    T    = (-np.log(U_rv + 1e-12) / (lam0 * np.exp(eta))) ** (1.0 / nu)
    return np.clip(T, 0.01, None)

def _est_lam0(X_std, A, w, mu_d, t, delta, lam_fallback):
    U    = X_std @ A
    eta  = (U - mu_d) @ w
    if delta.sum() < 3:
        return lam_fallback
    km_s = _km_at_tau(t, delta, TAU)
    km_p = float(np.clip(1 - km_s, 0.01, 0.99))
    try:
        return fit_lambda0(eta, km_p, tau=TAU, nu=NU)
    except Exception:
        return lam_fallback


# ── PART A: shift detection 1L vs 2L+ ───────────────────────────────────────

def run_shift_detection(verbose=True):
    """
    Trains generator on IO_LINE=1 patients (per center), measures
    AUROC_synth - AUROC_real gap on IO_LINE>=2 patients.
    """
    rwd = pd.read_csv("/Users/ludole/Downloads/rwd.csv")
    center_names = ["GHD","INT","MH","SZMC","UOC","VHIO"]
    feat_cols = registry_rwd.FEATURE_COLS  # 22 features (includes IO_LINE_1)

    # Encode features
    from data.registry_rwd import _encode_rwd
    rwd = _encode_rwd(rwd)

    # Build per-center 1L and 2L+ dicts
    from data import registry
    raw_dir = Path("/Users/ludole/Desktop/SHIFT-SURV/data/raw")

    def _load_center_split(cname):
        sub = rwd[rwd["CENTER"] == cname].reset_index(drop=True)
        raw = pd.read_csv(raw_dir / f"center_{cname}.csv")
        t_all   = raw["OS MONTHS"].to_numpy(dtype=float)
        d_all   = raw["DEATH EVENT"].to_numpy(dtype=int)
        io_line = sub["IO LINE"].to_numpy(dtype=int)
        X_all   = sub[feat_cols].to_numpy(dtype=float)

        # align lengths (drop NaN times)
        nan_mask = np.isnan(t_all)
        keep     = ~nan_mask
        t_all, d_all, X_all, io_line = (
            t_all[keep], d_all[keep], X_all[keep], io_line[keep]
        )
        t_all = np.clip(t_all, 0.01, None)

        mask1 = io_line == 1
        mask2 = io_line >= 2
        return {
            "1L": {"X": X_all[mask1], "t": t_all[mask1], "delta": d_all[mask1]},
            "2L": {"X": X_all[mask2], "t": t_all[mask2], "delta": d_all[mask2]},
        }

    splits = {c: _load_center_split(c) for c in center_names}

    if verbose:
        print("=== PART A: Shift Detection — 1L vs 2L+ ===\n")
        for c in center_names:
            n1 = len(splits[c]["1L"]["X"])
            n2 = len(splits[c]["2L"]["X"])
            r1 = _binary_label(splits[c]["1L"]["t"], splits[c]["1L"]["delta"]).mean()
            r2 = _binary_label(splits[c]["2L"]["t"], splits[c]["2L"]["delta"]).mean()
            print(f"  {c}: 1L n={n1} rate={r1:.2f}  |  2L+ n={n2} rate={r2:.2f}")
        print()

    # Fit generator on pooled 1L patients
    centers_1L = [
        {"X": splits[c]["1L"]["X"], "t": splits[c]["1L"]["t"],
         "delta": splits[c]["1L"]["delta"], "n": len(splits[c]["1L"]["X"]),
         "center_id": i}
        for i, c in enumerate(center_names)
        if len(splits[c]["1L"]["X"]) > 30
    ]

    X_pool_1L  = np.vstack([c["X"] for c in centers_1L])
    _, med_1L  = impute_median(X_pool_1L)
    centers_1L_imp = [
        {**c, "X": np.where(np.isnan(c["X"]), med_1L, c["X"])}
        for c in centers_1L
    ]

    A, w_fit, dom_params, scaler = fit_reference_generator(
        source_centers=centers_1L_imp, k_total=K, tau=TAU, nu=NU
    )
    lam0_1L_mean = float(np.mean([dp.lambda0 for dp in dom_params]))

    # Source LR on 1L pool
    X_src_std = np.vstack([scaler.transform(c["X"]) for c in centers_1L_imp])
    y_src     = _binary_label(
        np.concatenate([c["t"] for c in centers_1L_imp]),
        np.concatenate([c["delta"] for c in centers_1L_imp])
    )
    lr_1L = _fit_lr(X_src_std, y_src)

    # Evaluate gap on each center's 2L+ patients
    N_SEEDS = 30
    M       = 20
    rows_a  = []

    for cname in center_names:
        c2 = splits[cname]["2L"]
        if len(c2["X"]) < 20:
            if verbose:
                print(f"  {cname} 2L+: too few patients ({len(c2['X'])}), skipping")
            continue

        X2 = np.where(np.isnan(c2["X"]), med_1L, c2["X"])
        X2_std = scaler.transform(X2)
        y2_real = _binary_label(c2["t"], c2["delta"])

        if len(np.unique(y2_real)) < 2:
            continue

        a_real = _auroc(lr_1L.predict_proba(X2_std)[:, 1], y2_real)

        auroc_synths = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(42 + s * 97 + hash(cname) % 10000)
            idx_m = rng.choice(len(X2), size=min(M, len(X2)//2), replace=False)
            X_m   = X2_std[idx_m]
            t_m   = c2["t"][idx_m]
            d_m   = c2["delta"][idx_m]

            mu_d_est = (X_m @ A).mean(axis=0)
            lam_est  = _est_lam0(X_m, A, w_fit, mu_d_est,
                                  t_m, d_m, lam0_1L_mean)

            idx_syn = rng.choice(len(X2), size=min(300, len(X2)), replace=True)
            t_syn   = _generate_t(X2_std[idx_syn], A, w_fit, lam_est, NU, rng)
            y_syn   = _binary_label(t_syn, np.ones(len(idx_syn), dtype=int))

            if len(np.unique(y_syn)) < 2:
                continue

            a_syn = _auroc(lr_1L.predict_proba(X2_std[idx_syn])[:, 1], y_syn)
            auroc_synths.append(a_syn)

        a_s = float(np.nanmean(auroc_synths)) if auroc_synths else float("nan")
        gap = a_s - a_real

        rows_a.append({"center": cname, "auroc_real_2L": round(a_real, 4),
                       "auroc_synth": round(a_s, 4), "gap": round(gap, 4)})

        if verbose:
            flag = " ← CONCEPT SHIFT" if gap > 0.10 else ""
            print(f"  {cname} 2L+: real={a_real:.4f}  synth={a_s:.4f}  "
                  f"gap={gap:+.4f}{flag}")

    df_a = pd.DataFrame(rows_a)
    if verbose and len(df_a):
        print(f"\n  Mean gap: {df_a['gap'].mean():.4f}")
        print(f"  (1L source-only AUROC for reference: "
              f"{_auroc(lr_1L.predict_proba(X_src_std)[:,1], y_src):.4f})\n")
    return df_a


# ── PART B: 22-feature RWD LOOCO ────────────────────────────────────────────

def run_rwd22_looco(m_grid=None, n_seeds=20, verbose=True):
    """
    LOOCO with all 22 RWD features + source-only and w_refit_aligned variants.
    Baseline for comparison: v3 (11 features) source-only AUROC = 0.6828.
    """
    if m_grid is None:
        m_grid = [5, 10, 20, 50]

    centers, feat_names = registry_rwd.load_all_centers()

    if verbose:
        print("=== PART B: 22-feature RWD LOOCO ===\n")
        print(f"  Features ({len(feat_names)}): {feat_names}\n")
        for c in centers:
            y = _binary_label(c["t"], c["delta"])
            print(f"  {registry_rwd.CENTER_NAMES[c['center_id']]}: "
                  f"n={c['n']}, event_rate={y.mean():.2f}")
        print()

    rows = []

    for held_idx in range(len(centers)):
        held     = centers[held_idx]
        hid      = held["center_id"]
        name     = registry_rwd.CENTER_NAMES.get(hid, str(hid))
        sources  = [c for i, c in enumerate(centers) if i != held_idx]

        # Impute + scale
        X_src_pool = np.vstack([c["X"] for c in sources])
        _, src_med = impute_median(X_src_pool)
        sources_imp = [
            {**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
            for c in sources
        ]
        X_held_imp = np.where(np.isnan(held["X"]), src_med, held["X"])

        A, w_gen, dom_params, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=K, tau=TAU, nu=NU
        )

        X_src_all = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        X_held_s  = scaler.transform(X_held_imp)

        y_src_all = _binary_label(
            np.concatenate([c["t"] for c in sources_imp]),
            np.concatenate([c["delta"] for c in sources_imp])
        )
        y_held = _binary_label(held["t"], held["delta"])

        # Source LR
        lr_src = _fit_lr(X_src_all, y_src_all)
        if lr_src is None:
            continue

        p_src = lr_src.predict_proba(X_held_s)[:, 1]
        a_src = _auroc(p_src, y_held)

        if verbose:
            print(f"  {name}  S={a_src:.4f}", end="")

        rows.append({"center": name, "variant": "S", "m": 0,
                     "seed": 0, "auroc": a_src})

        for m in m_grid:
            if m >= held["n"]:
                continue

            aurocs_ft, aurocs_wr = [], []

            for s in range(n_seeds):
                rng  = np.random.default_rng(42 + hid * 1000 + s * 13 + m)
                idx  = rng.choice(held["n"], size=m, replace=False)
                test = np.ones(held["n"], dtype=bool); test[idx] = False

                X_m    = X_held_s[idx]
                y_m    = y_held[idx]
                X_test = X_held_s[test]
                y_test = y_held[test]

                if len(np.unique(y_test)) < 2:
                    continue

                # S_ft_real: fine-tune LR on [X_src, X_m]
                X_ft = np.vstack([X_src_all, X_m])
                y_ft = np.concatenate([y_src_all, y_m])
                lr_ft = _fit_lr(X_ft, y_ft)
                if lr_ft:
                    aurocs_ft.append(_auroc(lr_ft.predict_proba(X_test)[:, 1], y_test))

                # w_refit_aligned: regularised w update in latent space
                # Find shift direction in latent space
                U_src = X_src_all @ A
                U_m   = X_m @ A
                theta_shift = (U_m - U_src.mean(axis=0)).mean(axis=0)

                # Align test features
                X_test_al = X_test + theta_shift @ np.linalg.pinv(A)

                # Refit LR with aligned features
                X_wr = np.vstack([X_src_all, X_m])
                y_wr = np.concatenate([y_src_all, y_m])
                lr_wr = _fit_lr(X_wr, y_wr)
                if lr_wr:
                    aurocs_wr.append(_auroc(
                        lr_wr.predict_proba(X_test_al)[:, 1], y_test
                    ))

            a_ft = float(np.nanmean(aurocs_ft)) if aurocs_ft else float("nan")
            a_wr = float(np.nanmean(aurocs_wr)) if aurocs_wr else float("nan")

            rows.append({"center": name, "variant": "S_ft_real", "m": m,
                         "seed": -1, "auroc": a_ft})
            rows.append({"center": name, "variant": "w_refit_aligned", "m": m,
                         "seed": -1, "auroc": a_wr})

        if verbose:
            for m in m_grid:
                sub = [r for r in rows if r["center"] == name
                       and r["variant"] == "S_ft_real" and r["m"] == m]
                if sub:
                    print(f"  S_ft(m={m})={sub[0]['auroc']:.4f}", end="")
            print()

    df_b = pd.DataFrame(rows)

    if verbose:
        print("\nMean AUROC by variant and m (averaged over centers):")
        pivot = df_b.groupby(["variant","m"])["auroc"].mean().unstack("m").round(4)
        print(pivot.to_string())
        print(f"\n  Reference (11-feat v3 source-only): 0.6828")

    return df_b


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    out = Path("outputs/tables/")
    out.mkdir(parents=True, exist_ok=True)

    df_a = run_shift_detection(verbose=True)
    df_a.to_csv(out / "rwd_shift_detection_1L_vs_2L.csv", index=False)

    print("\n" + "="*60 + "\n")

    df_b = run_rwd22_looco(verbose=True)
    df_b.to_csv(out / "rwd22_looco_results.csv", index=False)
