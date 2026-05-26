"""
Calibration experiment: effect of recalibrating lambda0 (and mu_d) on
Weibull PH predictions for a new center.

Key question
------------
Using the Weibull PH predictor  P(y=1|x) = 1 - exp(-lambda0 * tau^nu * exp(eta))
with eta = (X_std @ A - mu_d) @ w,  how much does recalibrating lambda0
and mu_d from m patients improve predictions?

AUROC: rank-invariant to lambda0 scaling — won't improve with lambda0-only recalibration.
       May improve with mu correction (eta changes).
Brier score: sensitive to probability calibration — should improve markedly.
Calibration error: |mean_pred - observed_event_rate| — direct measure of bias.

Variants (each adds one correction on top of the previous)
----------
  S_weibull          source lambda0_mean + source mu (no target info)
  S_mu               source lambda0_mean + mu_d from m patients
  S_lambda           lambda0 from m patients + source mu
  S_mu_lambda        lambda0 + mu_d both from m patients   [full recalibration]
  oracle             lambda0 + mu_d from ALL target patients

m in {5, 20, 50}, 10 seeds.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median
from generator.fitting import (
    fit_reference_generator,
    fit_lambda0,
    _km_at_tau,
)


TAU = 12.0
NU  = 1.5


def _weibull_prob(eta, lambda0, tau=TAU, nu=NU):
    """P(y=1|eta) = 1 - exp(-lambda0 * tau^nu * exp(eta))"""
    return 1.0 - np.exp(-lambda0 * (tau ** nu) * np.exp(eta))


def _auroc(probs, y):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


def _brier(probs, y):
    """Brier score = mean squared error of probability predictions."""
    return float(np.mean((probs - y) ** 2))


def _calib_error(probs, y):
    """Mean calibration error: |mean predicted prob - observed event rate|."""
    return float(abs(probs.mean() - y.mean()))


def _binary_label(t, delta, tau=TAU):
    return ((t <= tau) & (delta == 1)).astype(int)


def _estimate_lambda0(X_m_std, A, w, mu_d, t_m, delta_m, tau, nu,
                      source_lambda0_mean):
    """Estimate lambda0 from m patients via KM. Falls back to source mean."""
    if delta_m.sum() < 3:
        return source_lambda0_mean
    U_m  = X_m_std @ A
    eta_m = (U_m - mu_d) @ w
    km_s  = _km_at_tau(t_m, delta_m, tau)
    km_prev = float(np.clip(1.0 - km_s, 0.01, 0.99))
    try:
        return fit_lambda0(eta_m, km_prev, tau=tau, nu=nu)
    except Exception:
        return source_lambda0_mean


def run_calibration_experiment(
    m_grid=None,
    n_seeds=10,
    k_total=6,
    tau=TAU,
    nu=NU,
    seed=42,
    out_dir="outputs/tables/",
    out_prefix="calibration",
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

        A, w, domain_params, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
        )

        X_src_all = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        X_held_s  = scaler.transform(X_held_imp)
        y_held    = _binary_label(held_out["t"], held_out["delta"], tau)

        # Source population parameters
        mu_src          = X_src_all @ A  # (N_src, k)
        mu_src_mean     = (X_src_all @ A).mean(axis=0)          # pooled source mean
        lambda0_src_mean = float(np.mean([dp.lambda0 for dp in domain_params]))

        # Oracle: mu and lambda0 from ALL target patients
        U_tgt_full   = X_held_s @ A
        mu_tgt_full  = U_tgt_full.mean(axis=0)
        lambda0_oracle = _estimate_lambda0(
            X_held_s, A, w, mu_tgt_full,
            held_out["t"], held_out["delta"], tau, nu, lambda0_src_mean
        )

        def _predict(X_s, mu_d, lambda0):
            eta = (X_s @ A - mu_d) @ w
            return _weibull_prob(eta, lambda0, tau, nu)

        # S_weibull: source mu + source lambda0
        p_S = _predict(X_held_s, mu_src_mean, lambda0_src_mean)
        # oracle
        p_oracle = _predict(X_held_s, mu_tgt_full, lambda0_oracle)

        for variant, probs in [("S_weibull", p_S), ("oracle", p_oracle)]:
            rows.append({
                "center": name, "variant": variant, "m": 0 if "S" in variant else n_full,
                "seed": 0,
                "auroc":  _auroc(probs, y_held),
                "brier":  _brier(probs, y_held),
                "calib_err": _calib_error(probs, y_held),
            })

        if verbose:
            obs_rate = y_held.mean()
            print(f"  observed event rate: {obs_rate:.3f}")
            print(f"  S_weibull  AUROC={_auroc(p_S,y_held):.4f}  "
                  f"Brier={_brier(p_S,y_held):.4f}  "
                  f"CalibErr={_calib_error(p_S,y_held):.4f}  "
                  f"(mean_pred={p_S.mean():.3f})")
            print(f"  oracle     AUROC={_auroc(p_oracle,y_held):.4f}  "
                  f"Brier={_brier(p_oracle,y_held):.4f}  "
                  f"CalibErr={_calib_error(p_oracle,y_held):.4f}  "
                  f"(mean_pred={p_oracle.mean():.3f})")

        # ── few-shot variants ─────────────────────────────────────────────────
        for m in m_grid:
            if m >= n_full:
                continue

            res = {v: {"auroc": [], "brier": [], "calib_err": []}
                   for v in ["S_mu", "S_lambda", "S_mu_lambda"]}

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                idx = rng.choice(n_full, size=m, replace=False)

                X_m     = X_held_s[idx]
                t_m     = held_out["t"][idx]
                d_m     = held_out["delta"][idx]

                # mu estimated from m patients
                mu_m = (X_m @ A).mean(axis=0)

                # lambda0 estimated from m patients (two versions: with src mu, with m mu)
                lam_src_mu = _estimate_lambda0(
                    X_m, A, w, mu_src_mean, t_m, d_m, tau, nu, lambda0_src_mean
                )
                lam_m_mu = _estimate_lambda0(
                    X_m, A, w, mu_m, t_m, d_m, tau, nu, lambda0_src_mean
                )

                test_mask = np.ones(n_full, dtype=bool)
                test_mask[idx] = False
                X_test = X_held_s[test_mask]
                y_test = y_held[test_mask]
                if len(np.unique(y_test)) < 2:
                    continue

                variants_here = {
                    "S_mu":        _predict(X_test, mu_m,        lambda0_src_mean),
                    "S_lambda":    _predict(X_test, mu_src_mean, lam_src_mu),
                    "S_mu_lambda": _predict(X_test, mu_m,        lam_m_mu),
                }

                for vname, probs in variants_here.items():
                    res[vname]["auroc"].append(_auroc(probs, y_test))
                    res[vname]["brier"].append(_brier(probs, y_test))
                    res[vname]["calib_err"].append(_calib_error(probs, y_test))
                    rows.append({
                        "center": name, "variant": vname, "m": m, "seed": s,
                        "auroc":     _auroc(probs, y_test),
                        "brier":     _brier(probs, y_test),
                        "calib_err": _calib_error(probs, y_test),
                    })

            if verbose:
                for vname in ["S_mu", "S_lambda", "S_mu_lambda"]:
                    a = np.nanmean(res[vname]["auroc"])
                    b = np.nanmean(res[vname]["brier"])
                    c = np.nanmean(res[vname]["calib_err"])
                    print(f"  m={m:3d}  {vname:<14}  "
                          f"AUROC={a:.4f}  Brier={b:.4f}  CalibErr={c:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path / f"{out_prefix}_results.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/{out_prefix}_results.csv")
        print("\nMean metrics by variant and m (averaged over centers and seeds):")
        for metric in ["auroc", "brier", "calib_err"]:
            agg = df.groupby(["variant", "m"])[metric].mean().unstack("m").round(4)
            print(f"\n{metric.upper()}:")
            print(agg.to_string())


if __name__ == "__main__":
    run_calibration_experiment(verbose=True)
