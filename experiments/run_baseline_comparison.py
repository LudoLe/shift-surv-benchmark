"""
Baseline comparison: generator λ₀ recalibration vs simpler alternatives.

The reviewer question: why use a Weibull generator instead of something simpler?

Baselines (all use source model's risk scores, differ only in recalibration)
-----------
  source_only     no adjustment — apply source model directly
  intercept_shift adjust log-odds intercept so mean_pred = observed_rate(m)
                  (1 parameter, purely from event rate of m patients)
  platt_scaling   fit logistic(intercept + slope) on m patients' scores
                  (2 parameters, standard post-hoc calibration)
  generator_lam   our method: recalibrate λ₀ via KM on m patients
                  (1 parameter, uses survival time information not just rate)

Key difference between intercept_shift and generator_lam:
  - intercept_shift: uses only event rate (binary, ignores timing)
  - generator_lam:   uses KM survival curve at tau (uses censored times too)
  With no censoring they are equivalent. The difference appears when
  censoring is heavy (MH: 77% censored, SZMC: 75% censored).

Metrics: Brier score, calibration error |mean_pred - obs_rate|
AUROC: reported for completeness (should be ~equal for all monotone recalibrations)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median
from generator.fitting import fit_reference_generator, fit_lambda0, _km_at_tau


TAU = 12.0
NU  = 1.5


def _binary_label(t, delta, tau=TAU):
    return ((t <= tau) & (delta == 1)).astype(int)

def _auroc(p, y):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

def _brier(p, y):
    return float(np.mean((p - y) ** 2))

def _calib_err(p, y):
    return float(abs(p.mean() - y.mean()))

def _weibull_prob(eta, lam, tau=TAU, nu=NU):
    return 1.0 - np.exp(-lam * (tau ** nu) * np.exp(eta))


# ── recalibration methods ────────────────────────────────────────────────────

def recalib_intercept_shift(logits_src, logits_m, y_m):
    """
    Shift intercept so mean predicted prob = observed event rate.
    Solves: mean(sigmoid(logits + c)) = mean(y_m)  for c.
    """
    target = float(np.clip(y_m.mean(), 0.01, 0.99))
    def eq(c):
        return float(expit(logits_src + c).mean()) - target
    try:
        c = brentq(eq, -10, 10)
    except Exception:
        c = 0.0
    return expit(logits_src + c)


def recalib_platt(logits_src, logits_m, y_m):
    """
    Platt scaling: fit logistic(a + b*score) on m patients.
    """
    if len(np.unique(y_m)) < 2:
        return expit(logits_src)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=0)
        lr.fit(logits_m[:, None], y_m)
    return expit(lr.predict_log_proba(logits_src[:, None])[:, 1]
                 + lr.intercept_[0] + lr.coef_[0, 0] * logits_src)


def recalib_platt_v2(logits_src, logits_m, y_m):
    """Correct Platt: predict_proba after fitting on m."""
    if len(np.unique(y_m)) < 2:
        return expit(logits_src)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=0)
        lr.fit(logits_m[:, None], y_m)
    return lr.predict_proba(logits_src[:, None])[:, 1]


def recalib_generator_lam(eta_held, A, w_fit, X_m_std, t_m, d_m,
                           lam_src_mean, tau=TAU, nu=NU):
    """
    Recalibrate λ₀ via KM on m patients — generator method.
    Uses survival time information (handles censoring).
    """
    U_m   = X_m_std @ A
    mu_m  = U_m.mean(axis=0)
    eta_m = (U_m - mu_m) @ w_fit
    if d_m.sum() >= 3:
        km_s = _km_at_tau(t_m, d_m, tau)
        km_p = float(np.clip(1 - km_s, 0.01, 0.99))
        try:
            lam = fit_lambda0(eta_m, km_p, tau=tau, nu=nu)
        except Exception:
            lam = lam_src_mean
    else:
        lam = lam_src_mean
    return _weibull_prob(eta_held, lam, tau, nu)


# ── main experiment ──────────────────────────────────────────────────────────

def run_baseline_comparison(
    m_grid=None,
    n_seeds=20,
    k_total=6,
    tau=TAU,
    nu=NU,
    seed=42,
    out_dir="outputs/tables/",
    out_prefix="baseline_comparison",
    verbose=True,
    load_centers_fn=None,
):
    if m_grid is None:
        m_grid = [5, 10, 20, 50]

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
        cens_frac = 1.0 - held_out["delta"].mean()

        if verbose:
            print(f"\n=== {name} (n={n_full}, censored={cens_frac:.0%}) ===")

        sources    = [c for i, c in enumerate(centers) if i != held_idx]
        X_pool     = np.vstack([c["X"] for c in sources])
        _, src_med = impute_median(X_pool)
        sources_imp = [{**c,"X":np.where(np.isnan(c["X"]),src_med,c["X"])}
                       for c in sources]
        X_held_imp  = np.where(np.isnan(held_out["X"]), src_med, held_out["X"])

        A, w_gen, domain_params, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu)

        X_src_all  = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        X_held_s   = scaler.transform(X_held_imp)
        y_src_all  = _binary_label(
            np.concatenate([c["t"] for c in sources_imp]),
            np.concatenate([c["delta"] for c in sources_imp]), tau)
        y_held     = _binary_label(held_out["t"], held_out["delta"], tau)

        lam_src_mean = float(np.mean([dp.lambda0 for dp in domain_params]))

        # Source model (LR on full features)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            lr_src = LogisticRegression(C=1.0, max_iter=1000,
                                        class_weight="balanced", random_state=0)
            lr_src.fit(X_src_all, y_src_all)

        logits_held = X_held_s @ lr_src.coef_[0] + lr_src.intercept_[0]
        p_source    = expit(logits_held)

        # Generator: eta for Weibull predictor
        U_held   = X_held_s @ A
        mu_held  = U_held.mean(axis=0)
        eta_held = (U_held - mu_held) @ w_gen

        # Source-only metrics
        for metric_name, val in [
            ("auroc", _auroc(p_source, y_held)),
            ("brier", _brier(p_source, y_held)),
            ("calib_err", _calib_err(p_source, y_held)),
        ]:
            rows.append({"center": name, "variant": "source_only",
                         "m": 0, "seed": 0, "metric": metric_name, "value": val})

        if verbose:
            print(f"  source_only    AUROC={_auroc(p_source,y_held):.4f}  "
                  f"Brier={_brier(p_source,y_held):.4f}  "
                  f"CalibErr={_calib_err(p_source,y_held):.4f}")

        for m in m_grid:
            if m >= n_full:
                continue

            res = {v: {"auroc":[],"brier":[],"calib_err":[]}
                   for v in ["intercept_shift","platt","generator_lam"]}

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid*1000 + s*13 + m)
                idx = rng.choice(n_full, size=m, replace=False)

                test_mask = np.ones(n_full, dtype=bool)
                test_mask[idx] = False
                X_test   = X_held_s[test_mask]
                y_test   = y_held[test_mask]
                if len(np.unique(y_test)) < 2:
                    continue

                X_m  = X_held_s[idx]
                y_m  = y_held[idx]
                t_m  = held_out["t"][idx]
                d_m  = held_out["delta"][idx]

                logits_test = X_test @ lr_src.coef_[0] + lr_src.intercept_[0]
                logits_m    = X_m   @ lr_src.coef_[0] + lr_src.intercept_[0]

                # Baseline 1: intercept shift
                p_int = recalib_intercept_shift(logits_test, logits_m, y_m)

                # Baseline 2: Platt scaling
                p_pl  = recalib_platt_v2(logits_test, logits_m, y_m)

                # Our method: generator λ₀
                eta_test = (X_test @ A - mu_held) @ w_gen
                p_gen    = recalib_generator_lam(
                    eta_test, A, w_gen, X_m, t_m, d_m, lam_src_mean, tau, nu)

                for vname, probs in [
                    ("intercept_shift", p_int),
                    ("platt",           p_pl),
                    ("generator_lam",   p_gen),
                ]:
                    for mname, fn in [("auroc",_auroc),("brier",_brier),("calib_err",_calib_err)]:
                        v = fn(probs, y_test)
                        res[vname][mname].append(v)
                        rows.append({"center":name,"variant":vname,
                                     "m":m,"seed":s,"metric":mname,"value":v})

            if verbose:
                for vname in ["intercept_shift","platt","generator_lam"]:
                    a = np.nanmean(res[vname]["auroc"])
                    b = np.nanmean(res[vname]["brier"])
                    c = np.nanmean(res[vname]["calib_err"])
                    print(f"  m={m:3d} {vname:<18}  "
                          f"AUROC={a:.4f}  Brier={b:.4f}  CalibErr={c:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path / f"{out_prefix}_results.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/{out_prefix}_results.csv")
        pivot = (df.groupby(["variant","m","metric"])["value"]
                 .mean().unstack("metric").round(4))
        for m_val in sorted(df["m"].unique()):
            sub = pivot.xs(m_val, level="m") if m_val in pivot.index.get_level_values("m") else None
            if sub is not None and len(sub):
                print(f"\nm={m_val}:")
                print(sub[["auroc","brier","calib_err"]].to_string())


if __name__ == "__main__":
    run_baseline_comparison(verbose=True)
