"""
LOOCO: three methodological directions.

  1. IPCW-weighted LR  — corrects censoring bias in binary label creation
  2. τ sweep           — find optimal prediction horizon for Cox ERM and LR ERM
  3. BBSE label shift  — label shift correction (Black Box Shift Estimation)

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_looco_methodological
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
from generator.fitting import fit_risk_weights_cox

TAU      = 12.0
TAU_GRID = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]
COX_ALPHA = 5e-2   # best alpha from previous sweep
OUT_DIR  = _ROOT / "outputs" / "tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_label(t, delta, tau):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc_safe(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _lr(X_train, y_train, sample_weight=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=0,
        )
        lr.fit(X_train, y_train, sample_weight=sample_weight)
    return lr


# ---------------------------------------------------------------------------
# IPCW helpers
# ---------------------------------------------------------------------------

def _km_censoring_survival(t, delta):
    """
    Kaplan-Meier estimate of the censoring survival S_C(t).

    Treats censored observations (delta=0) as 'events' for the censoring
    process.  Returns S_C evaluated at each patient's observed time t_i.

    Parameters
    ----------
    t     : (n,) observed times
    delta : (n,) event indicators (1=event, 0=censored)

    Returns
    -------
    S_C : (n,) S_C(t_i) for each patient
    """
    n = len(t)
    unique_times = np.sort(np.unique(t))

    # KM on censoring distribution
    s = 1.0
    km = {}   # time → S_C(t)
    for u in unique_times:
        at_risk  = np.sum(t >= u)
        n_cens   = np.sum((t == u) & (delta == 0))   # censored = event for S_C
        if at_risk > 0:
            s *= (1.0 - n_cens / at_risk)
        km[u] = max(s, 1e-6)   # prevent division by zero later

    return np.array([km[ti] for ti in t])


def _ipcw_weights(t, delta, eps=1e-6):
    """
    IPCW weights: w_i = delta_i / S_C(t_i).

    Only observed events (delta=1) receive non-zero weight.
    """
    S_C = _km_censoring_survival(t, delta)
    return np.where(delta == 1, 1.0 / np.maximum(S_C, eps), 0.0)


# ---------------------------------------------------------------------------
# BBSE helpers
# ---------------------------------------------------------------------------

def _bbse_weights(X_src, y_src, X_tgt, eps=1e-3):
    """
    Black Box Shift Estimation for binary label shift.

    Estimates target prevalence p̂_tgt(y=1) from source classifier
    predictions on the target, then returns per-sample weights for source:
      w(y=1) = p_tgt / p_src
      w(y=0) = (1-p_tgt) / (1-p_src)

    Reference: Lipton et al. (2018) "Detecting and Correcting for Label
    Shift with Black Box Predictors."
    """
    p_src = y_src.mean()

    # Train classifier on source
    lr_dom = _lr(X_src, y_src)
    pred_src = lr_dom.predict_proba(X_src)[:, 1]   # scores on source
    pred_tgt = lr_dom.predict_proba(X_tgt)[:, 1]   # scores on target

    # Estimate confusion matrix on source (using soft predictions)
    # C[i,j] = P(ŷ=i | y=j)  in binary case:
    #   C[1,1] = sensitivity    = E[ŷ | y=1]
    #   C[0,1] = 1-sensitivity
    #   C[1,0] = FPR            = E[ŷ | y=0]
    #   C[0,0] = 1-FPR
    pos = y_src == 1
    neg = y_src == 0

    sensitivity = pred_src[pos].mean() if pos.sum() > 0 else 0.5
    fpr         = pred_src[neg].mean() if neg.sum() > 0 else 0.5

    # Mean prediction on target = p_tgt * sensitivity + (1-p_tgt) * fpr
    # → solve for p_tgt
    denom = sensitivity - fpr
    if abs(denom) < eps:
        return np.ones(len(y_src))   # can't estimate; return uniform

    p_tgt = np.clip(
        (pred_tgt.mean() - fpr) / denom,
        eps, 1.0 - eps
    )

    # Per-sample importance weights
    w_pos = p_tgt / max(p_src, eps)
    w_neg = (1.0 - p_tgt) / max(1.0 - p_src, eps)

    weights = np.where(y_src == 1, w_pos, w_neg)
    weights = np.clip(weights, 0.0, 10.0)
    weights /= weights.mean()
    return weights, float(p_tgt)


# ---------------------------------------------------------------------------
# Shared fold preparation
# ---------------------------------------------------------------------------

def _prep_fold(centers, held_idx, tau):
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

    return hname, held["n"], X_src_s, X_tgt_s, y_src, t_src, delta_src, y_tgt


# ---------------------------------------------------------------------------
# Experiment 1 — IPCW
# ---------------------------------------------------------------------------

def run_ipcw(centers, tau=TAU, verbose=True):
    D = len(centers)
    rows = []

    if verbose:
        print(f"\n--- Exp 1: IPCW-weighted LR (τ={tau}) ---")
        print(f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  {'ipcw_lr':>9}  "
              f"{'Δipcw':>8}  {'p̂_tgt_prev':>11}")
        print("-" * 60)

    for held_idx in range(D):
        hname, n, X_src_s, X_tgt_s, y_src, t_src, delta_src, y_tgt = \
            _prep_fold(centers, held_idx, tau)

        auroc_lr = _auroc_safe(y_tgt, _lr(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        # IPCW weights from source survival data
        ipcw_w = _ipcw_weights(t_src, delta_src)
        # Normalise so weights have mean 1 (scale-invariant for LR)
        ipcw_w = ipcw_w / ipcw_w[ipcw_w > 0].mean() if (ipcw_w > 0).any() else ipcw_w

        # Use IPCW weights directly (class_weight="balanced" already in _lr,
        # but we override with combined weight = ipcw * class_balance)
        class_w = np.where(y_src == 1,
                           1.0 / (y_src.mean() + 1e-9),
                           1.0 / (1.0 - y_src.mean() + 1e-9))
        combined_w = ipcw_w * class_w
        combined_w /= combined_w[combined_w > 0].mean()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lr_ipcw = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                         fit_intercept=True, random_state=0)
            lr_ipcw.fit(X_src_s, y_src, sample_weight=combined_w)
        auroc_ipcw = _auroc_safe(y_tgt, lr_ipcw.predict_proba(X_tgt_s)[:, 1])

        # Estimated target prevalence from IPCW (weighted mean of y in source
        # gives debiased event rate, but here we just report observed target prev)
        tgt_prev = float(y_tgt.mean())

        row = {"center": hname, "n": n, "lr_erm": auroc_lr,
               "ipcw_lr": auroc_ipcw, "delta_ipcw": auroc_ipcw - auroc_lr,
               "tgt_prev": tgt_prev}
        rows.append(row)

        if verbose:
            print(f"  {hname:>6}  {n:>4}  {auroc_lr:>8.4f}  {auroc_ipcw:>9.4f}  "
                  f"{auroc_ipcw-auroc_lr:>+8.4f}  {tgt_prev:>11.3f}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean:")
        print(df[["lr_erm", "ipcw_lr", "delta_ipcw"]].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Experiment 2 — τ sweep
# ---------------------------------------------------------------------------

def run_tau_sweep(centers, tau_grid=TAU_GRID, verbose=True):
    D = len(centers)
    rows = []

    if verbose:
        print(f"\n--- Exp 2: τ sweep ---")
        tau_hdr = "  ".join(f"τ={int(t):>2}" for t in tau_grid)
        print(f"{'method':>8}  {'center':>6}  {'n':>4}  " + tau_hdr)
        print("-" * (26 + 9 * len(tau_grid)))

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
        t_src     = np.concatenate([c["t"]     for c in src_imp])
        delta_src = np.concatenate([c["delta"] for c in src_imp])
        t_tgt     = held["t"]
        delta_tgt = held["delta"]

        # Cox ERM: fit once (τ-independent), evaluate AUROC at each τ
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_cox = fit_risk_weights_cox(X_src_s, t_src, delta_src, alpha=COX_ALPHA)
        score_cox = X_tgt_s @ w_cox

        row_cox = {"method": "cox_erm", "center": hname, "n": held["n"]}
        row_lr  = {"method": "lr_erm",  "center": hname, "n": held["n"]}

        auroc_cox_list, auroc_lr_list = [], []
        for tau in tau_grid:
            y_src_tau = _binary_label(t_src, delta_src, tau)
            y_tgt_tau = _binary_label(t_tgt, delta_tgt, tau)

            # Cox: score fixed, label changes
            a_cox = _auroc_safe(y_tgt_tau, score_cox)

            # LR: refit at each τ
            if len(np.unique(y_src_tau)) < 2:
                a_lr = float("nan")
            else:
                lr = _lr(X_src_s, y_src_tau)
                a_lr = _auroc_safe(y_tgt_tau, lr.predict_proba(X_tgt_s)[:, 1])

            row_cox[f"tau{int(tau)}"] = a_cox
            row_lr[f"tau{int(tau)}"]  = a_lr
            auroc_cox_list.append(a_cox)
            auroc_lr_list.append(a_lr)

        rows.extend([row_cox, row_lr])

        if verbose:
            cox_str = "  ".join(f"{a:>6.4f}" for a in auroc_cox_list)
            lr_str  = "  ".join(f"{a:>6.4f}" for a in auroc_lr_list)
            print(f"  {'cox_erm':>8}  {hname:>6}  {held['n']:>4}  {cox_str}")
            print(f"  {'lr_erm':>8}  {hname:>6}  {held['n']:>4}  {lr_str}")

    df = pd.DataFrame(rows)
    tau_cols = [f"tau{int(t)}" for t in tau_grid]
    if verbose:
        print("\nMean AUROC across centers:")
        for method in ["cox_erm", "lr_erm"]:
            sub = df[df["method"] == method][tau_cols].mean().round(4)
            print(f"  {method:>8}: " + "  ".join(f"{v:.4f}" for v in sub.values))
    return df


# ---------------------------------------------------------------------------
# Experiment 3 — BBSE label shift correction
# ---------------------------------------------------------------------------

def run_bbse(centers, tau=TAU, verbose=True):
    D = len(centers)
    rows = []

    if verbose:
        print(f"\n--- Exp 3: BBSE label shift correction (τ={tau}) ---")
        print(f"{'center':>6}  {'n':>4}  {'lr_erm':>8}  {'bbse_lr':>9}  "
              f"{'Δbbse':>8}  {'p_src':>7}  {'p̂_tgt':>8}  {'p_tgt_obs':>10}")
        print("-" * 76)

    for held_idx in range(D):
        hname, n, X_src_s, X_tgt_s, y_src, t_src, delta_src, y_tgt = \
            _prep_fold(centers, held_idx, tau)

        auroc_lr = _auroc_safe(y_tgt, _lr(X_src_s, y_src).predict_proba(X_tgt_s)[:, 1])

        try:
            bbse_w, p_tgt_est = _bbse_weights(X_src_s, y_src, X_tgt_s)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lr_bbse = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                             fit_intercept=True, random_state=0)
                lr_bbse.fit(X_src_s, y_src, sample_weight=bbse_w)
            auroc_bbse = _auroc_safe(y_tgt, lr_bbse.predict_proba(X_tgt_s)[:, 1])
        except Exception as e:
            auroc_bbse = float("nan")
            p_tgt_est  = float("nan")
            if verbose:
                print(f"  BBSE failed ({hname}): {e}")

        p_src     = float(y_src.mean())
        p_tgt_obs = float(y_tgt.mean())

        row = {
            "center": hname, "n": n,
            "lr_erm": auroc_lr, "bbse_lr": auroc_bbse,
            "delta_bbse": auroc_bbse - auroc_lr,
            "p_src": p_src, "p_tgt_est": p_tgt_est, "p_tgt_obs": p_tgt_obs,
        }
        rows.append(row)

        if verbose:
            print(f"  {hname:>6}  {n:>4}  {auroc_lr:>8.4f}  {auroc_bbse:>9.4f}  "
                  f"{auroc_bbse-auroc_lr:>+8.4f}  {p_src:>7.3f}  "
                  f"{p_tgt_est:>8.3f}  {p_tgt_obs:>10.3f}")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nMean:")
        print(df[["lr_erm", "bbse_lr", "delta_bbse"]].mean().round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(tau=TAU, verbose=True, out_dir=OUT_DIR):
    centers, _ = reg.load_all_centers()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_ipcw = run_ipcw(centers, tau=tau, verbose=verbose)
    df_tau  = run_tau_sweep(centers, verbose=verbose)
    df_bbse = run_bbse(centers, tau=tau, verbose=verbose)

    df_ipcw.to_csv(out_path / "looco_ipcw.csv",  index=False)
    df_tau.to_csv( out_path / "looco_tau_sweep.csv", index=False)
    df_bbse.to_csv(out_path / "looco_bbse.csv",  index=False)

    if verbose:
        print(f"\nSaved to {out_path}/")

    return df_ipcw, df_tau, df_bbse


if __name__ == "__main__":
    run_all()
