"""
Two-stage generative adaptation: A_shared + Δ_d (LOOCO, OS@12m).

Tests whether a rank-1 center-specific correction  Δ_d = u vᵀ  to the
shared loading matrix improves predictions on the held-out target center.

Three variants compared per LOOCO fold:
  V1  No adaptation     — A_shared, λ₀ = source-mean
  V2  Location + scale  — A_shared, λ₀ from KM  (current Stage-2 baseline)
  V3  + Obs correction  — A_d = A_shared + Δ_d (rank-1), λ₀ from KM

For each variant the following metrics are reported:
  AUROC@12m       — discrimination (V3 fit on eval set → optimistic; see note)
  CoxPLL/n        — normalised Cox partial log-likelihood on target
  Calib           — |KM(12m) − Ê[T≤12m]| under Weibull PH

Sensitivity: three regularisation strengths  ρ ∈ {0.1, 1.0, 10.0}  for Δ_d.

Note on AUROC bias
------------------
For V3, Δ_d is fit on the same patients used for evaluation. With rank-1
and ρ ≥ 1 the effective degrees of freedom are small (p+k = 13 vs n ≥ 200),
so overfitting is limited. Nevertheless, treat AUROC as optimistic; prefer
CoxPLL and Calib as primary metrics for the observation-correction claim.

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_two_stage_ada
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.preprocess import impute_median
from generator.fitting import (
    fit_reference_generator,
    fit_delta_rank1,
    fit_lambda0,
    _km_at_tau,
)

TAU_OS  = 12.0
NU      = 1.5
K       = 6
RHO_VALS = [0.1, 1.0, 10.0]   # sensitivity sweep for Δ_d regularisation
OUT_DIR = _ROOT / "outputs" / "tables" / "two_stage_ada"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def _imp(X, meds):
    return np.where(np.isnan(X), meds, X)


def _weibull_loglik(eta: np.ndarray, t: np.ndarray, delta: np.ndarray,
                    lambda0: float, nu: float) -> float:
    """Weibull PH per-sample log-likelihood, summed."""
    t   = np.maximum(t, 1e-8)
    ll  = delta * (np.log(nu) + np.log(lambda0)
                   + (nu - 1) * np.log(t) + eta)
    ll -= lambda0 * (t ** nu) * np.exp(eta)
    return float(ll.sum())


def _cox_pll(eta: np.ndarray, t: np.ndarray, delta: np.ndarray) -> float:
    """Breslow Cox partial log-likelihood."""
    order   = np.argsort(t)
    eta_s   = eta[order];  d_s = delta[order].astype(float)
    eta_sh  = eta_s - eta_s.max()
    exp_eta = np.exp(eta_sh)
    suf_exp = np.cumsum(exp_eta[::-1])[::-1]
    ev      = d_s == 1
    return float((eta_s[ev] - np.log(suf_exp[ev] + 1e-30)).sum())


# ── data loader ───────────────────────────────────────────────────────────────

def load_i3lung() -> list[dict]:
    """i3lung (all IO lines), 7 clinical features + OS survival."""
    rwd = pd.read_csv(_RWD_PATH)
    out = pd.read_csv(_OUTCOMES_PATH)
    df  = rwd.merge(
        out[["Subject", "OS MONTHS", "DEATH EVENT"]],
        on="Subject", how="inner"
    ).dropna(subset=["OS MONTHS", "DEATH EVENT"])

    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        X    = grp[COMMON_FEATS].to_numpy(float)
        t    = grp["OS MONTHS"].to_numpy(float)
        ev   = grp["DEATH EVENT"].to_numpy(float)
        y_os = ((t <= TAU_OS) & (ev == 1)).astype(int)
        centers.append({
            "X": X, "t": t, "delta": ev,
            "y_os": y_os, "n": len(X), "center_id": cid, "name": name,
        })
        print(f"  {name}: n={len(X)}  dead@{TAU_OS:.0f}m={y_os.sum()} "
              f"({y_os.mean()*100:.1f}%)")
    return centers


# ── per-fold metrics for one variant ─────────────────────────────────────────

def _variant_metrics(
    X_tgt_s: np.ndarray,
    t_tgt:   np.ndarray,
    d_tgt:   np.ndarray,
    y_tgt:   np.ndarray,
    A:       np.ndarray,
    w:       np.ndarray,
    lambda0: float,
) -> dict:
    """Compute AUROC, CoxPLL/n, and calibration for one (A, w, λ₀) setting."""
    U   = X_tgt_s @ A
    mu  = U.mean(0)
    eta = (U - mu) @ w

    auroc = _auroc(y_tgt, eta)
    cpll  = _cox_pll(eta, t_tgt, d_tgt) / max(len(t_tgt), 1)
    wll   = _weibull_loglik(eta, t_tgt, d_tgt, lambda0, NU) / max(len(t_tgt), 1)

    # Calibration: |KM(τ) − mean model prediction at τ|
    pred_prev = float(np.mean(1.0 - np.exp(-lambda0 * (TAU_OS ** NU) * np.exp(eta))))
    km_prev   = float(np.clip(1.0 - _km_at_tau(t_tgt, d_tgt, TAU_OS), 0.0, 1.0))
    calib     = abs(km_prev - pred_prev)

    return {"auroc": auroc, "cpll": cpll, "wll": wll,
            "calib": calib, "km_prev": km_prev, "pred_prev": pred_prev}


# ── LOOCO main loop ───────────────────────────────────────────────────────────

def run_looco(i3lung: list[dict], rho: float) -> pd.DataFrame:
    rows = []
    D    = len(i3lung)

    for held_idx in range(D):
        held   = i3lung[held_idx]
        src_cs = [c for i, c in enumerate(i3lung) if i != held_idx]

        y_tgt  = held["y_os"]
        t_tgt  = held["t"]
        d_tgt  = held["delta"]
        if len(np.unique(y_tgt)) < 2:
            continue

        # ── impute with source medians ────────────────────────────────────
        X_src = np.vstack([c["X"] for c in src_cs])
        _, meds = impute_median(X_src)
        src_imp = [{**c, "X": _imp(c["X"], meds)} for c in src_cs]
        X_tgt_imp = _imp(held["X"], meds)

        # ── Stage 1: fit (A, w, domain_params) on source ─────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A, w, dp_list, sc = fit_reference_generator(
                src_imp, k_total=K, tau=TAU_OS, time_distribution="cox",
            )

        X_tgt_s = sc.transform(X_tgt_imp)

        # Source-mean fallback λ₀
        src_lambda0_mean = float(np.mean([dp.lambda0 for dp in dp_list]))

        # KM-calibrated λ₀ for target (used by V2 and V3)
        km_s  = _km_at_tau(t_tgt, d_tgt, TAU_OS)
        km_pv = float(np.clip(1.0 - km_s, 0.01, 0.99))

        # λ₀ for V2 (A_shared, centred eta)
        U_v2 = X_tgt_s @ A;  mu_v2 = U_v2.mean(0)
        eta_v2 = (U_v2 - mu_v2) @ w
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lam0_v2 = fit_lambda0(eta_v2, km_pv, tau=TAU_OS, nu=NU)
        except Exception:
            lam0_v2 = src_lambda0_mean

        # ── Variant 1: no adaptation (source-mean λ₀) ────────────────────
        m1 = _variant_metrics(X_tgt_s, t_tgt, d_tgt, y_tgt,
                               A, w, src_lambda0_mean)

        # ── Variant 2: λ₀ from KM (current Stage-2 baseline) ─────────────
        m2 = _variant_metrics(X_tgt_s, t_tgt, d_tgt, y_tgt,
                               A, w, lam0_v2)

        # ── Variant 3: rank-1 Δ_d + KM λ₀ ───────────────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            delta_d = fit_delta_rank1(
                X_tgt_s, t_tgt, d_tgt, A, w, rho=rho)

        A_d   = A + delta_d
        U_v3  = X_tgt_s @ A_d;  mu_v3 = U_v3.mean(0)
        eta_v3 = (U_v3 - mu_v3) @ w
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lam0_v3 = fit_lambda0(eta_v3, km_pv, tau=TAU_OS, nu=NU)
        except Exception:
            lam0_v3 = src_lambda0_mean

        m3 = _variant_metrics(X_tgt_s, t_tgt, d_tgt, y_tgt,
                               A_d, w, lam0_v3)

        # ΔCoxPLL and ΔCalib vs V2
        delta_cpll  = m3["cpll"]  - m2["cpll"]
        delta_calib = m3["calib"] - m2["calib"]   # negative = better calibration
        delta_auroc = m3["auroc"] - m2["auroc"]

        # Frobenius norm of Δ_d
        delta_norm = float(np.linalg.norm(delta_d, "fro"))

        sign = lambda d, thr=0.005: "▲" if d > thr else ("▼" if d < -thr else "·")

        print(
            f"  {held['name']:>6}  n={held['n']:>3}  "
            f"V1={m1['auroc']:.4f}  V2={m2['auroc']:.4f}  "
            f"V3={m3['auroc']:.4f}  "
            f"ΔAUC={delta_auroc:+.4f}{sign(delta_auroc)}  "
            f"ΔcPLL={delta_cpll:+.4f}  "
            f"Δcal={delta_calib:+.4f}{sign(-delta_calib)}  "  # invert sign: neg=better
            f"|Δ|={delta_norm:.4f}"
        )

        rows.append({
            "center": held["name"], "n": held["n"], "rho": rho,
            "auroc_v1":  round(m1["auroc"], 4),
            "auroc_v2":  round(m2["auroc"], 4),
            "auroc_v3":  round(m3["auroc"], 4),
            "delta_auroc": round(delta_auroc, 4),
            "cpll_v2":   round(m2["cpll"],  5),
            "cpll_v3":   round(m3["cpll"],  5),
            "delta_cpll":  round(delta_cpll, 5),
            "calib_v1":  round(m1["calib"], 4),
            "calib_v2":  round(m2["calib"], 4),
            "calib_v3":  round(m3["calib"], 4),
            "delta_calib": round(delta_calib, 4),
            "delta_norm":  round(delta_norm, 4),
            "km_prev":   round(m2["km_prev"], 4),
        })

    return pd.DataFrame(rows)


# ── summary printer ───────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, rho: float) -> None:
    sub = df[df["rho"] == rho]
    print(f"\n  Summary  ρ={rho}:")
    print(f"    {'Metric':20s}  {'V1':>8}  {'V2':>8}  {'V3':>8}  {'Δ(V3−V2)':>10}")
    print("    " + "-" * 58)

    def _row(label, c1, c2, c3, delta_col):
        print(f"    {label:20s}  {sub[c1].mean():>8.4f}  "
              f"{sub[c2].mean():>8.4f}  {sub[c3].mean():>8.4f}  "
              f"{sub[delta_col].mean():>+10.4f}")

    _row("AUROC@12m (opt*)", "auroc_v1", "auroc_v2", "auroc_v3", "delta_auroc")
    _row("CoxPLL/n",         "cpll_v2",  "cpll_v2",  "cpll_v3",  "delta_cpll")
    _row("Calib error |·|",  "calib_v1", "calib_v2", "calib_v3", "delta_calib")
    print(f"    {'|Δ_d| (mean)':20s}  {'—':>8}  {'—':>8}  "
          f"{sub['delta_norm'].mean():>8.4f}")
    print("    * V3 AUROC is optimistic (Δ fit on eval set)")


# ── entry point ───────────────────────────────────────────────────────────────

def run_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung (all IO lines, OS survival)...")
    i3lung = load_i3lung()

    all_rows = []

    for rho in RHO_VALS:
        print(f"\n{'='*70}")
        print(f"  A_d = A_shared + Δ_d (rank-1, ρ={rho})   k={K}  τ={TAU_OS}m")
        hdr = (f"  {'center':>6}  {'n':>4}  "
               f"{'V1':>7}  {'V2':>7}  {'V3':>7}  "
               f"{'ΔAUC':>8}  {'ΔcPLL':>8}  {'Δcal':>8}  {'|Δ|':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df_rho = run_looco(i3lung, rho=rho)

        all_rows.append(df_rho)
        _print_summary(df_rho, rho)

    df_all = pd.concat(all_rows, ignore_index=True)
    out_csv = OUT_DIR / "two_stage_ada.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── cross-ρ summary ────────────────────────────────────────────────────
    print("\n  Cross-ρ summary  (mean across centers):")
    print(f"  {'ρ':>5}  {'ΔAUC':>8}  {'ΔcPLL':>8}  {'Δcal':>8}  {'|Δ|':>8}")
    print("  " + "-" * 46)
    for rho in RHO_VALS:
        sub = df_all[df_all["rho"] == rho]
        print(f"  {rho:>5}  "
              f"{sub['delta_auroc'].mean():>+8.4f}  "
              f"{sub['delta_cpll'].mean():>+8.4f}  "
              f"{sub['delta_calib'].mean():>+8.4f}  "
              f"{sub['delta_norm'].mean():>8.4f}")

    print("\nInterpretation:")
    print("  If ΔcPLL > 0 and Δcal < 0 → Δ_d corrects observation shift")
    print("  If ΔAUC ≈ 0 (at large ρ)  → gain is calibration, not discrimination")
    print("  If |Δ_d| → 0 as ρ increases → Δ_d is data-driven, not noise")


if __name__ == "__main__":
    run_all()
