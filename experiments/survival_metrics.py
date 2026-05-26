"""
Survival-aware evaluation metrics.

  c_index      : Harrell's concordance index (τ-independent)
  iauc         : AUROC integrated over a τ grid
  brier_score  : IPCW-corrected Brier score at a fixed τ
  ibs          : Integrated Brier Score over a τ grid

All functions accept a generic risk score vector (higher = more risk),
except brier_score / ibs which additionally require calibrated probabilities.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# C-index (Harrell's C)
# ---------------------------------------------------------------------------

def c_index(t: np.ndarray, delta: np.ndarray, scores: np.ndarray) -> float:
    """
    Harrell's concordance index for censored survival data.

    For every admissible pair (i, j) where t_i < t_j and delta_i = 1:
      - concordant  if score_i > score_j
      - discordant  if score_i < score_j
      - tied        if score_i == score_j

    C = (concordant + 0.5 * tied) / total_admissible

    Parameters
    ----------
    t      : (n,) observed times
    delta  : (n,) event indicators (1=event, 0=censored)
    scores : (n,) predicted risk (higher = higher risk of early event)

    Returns
    -------
    float in [0, 1], 0.5 = random
    """
    t = np.asarray(t, float)
    delta = np.asarray(delta, float)
    scores = np.asarray(scores, float)
    n = len(t)

    concordant = 0.0
    discordant = 0.0
    tied       = 0.0

    for i in range(n):
        if delta[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if t[j] <= t[i]:   # j did not survive longer than i
                continue
            # admissible pair: i events before j (or j still alive)
            if scores[i] > scores[j]:
                concordant += 1
            elif scores[i] < scores[j]:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return float("nan")
    return float((concordant + 0.5 * tied) / total)


# ---------------------------------------------------------------------------
# iAUC — integrated AUROC over τ grid
# ---------------------------------------------------------------------------

def iauc(
    t: np.ndarray,
    delta: np.ndarray,
    scores: np.ndarray,
    tau_grid: list[float] | None = None,
) -> float:
    """
    Integrated AUROC over a grid of prediction horizons.

    Computes AUROC(τ) = roc_auc_score(y_τ, scores) at each τ in tau_grid
    where y_τ = (t ≤ τ) & (delta == 1), then returns the simple mean.
    Skips τ values where y_τ has only one class.

    Parameters
    ----------
    t, delta : survival data
    scores   : risk scores (higher = higher risk)
    tau_grid : list of horizons; default [6, 9, 12, 15, 18, 24]

    Returns
    -------
    float : mean AUROC across valid τ values
    """
    if tau_grid is None:
        tau_grid = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]

    aurocs = []
    for tau in tau_grid:
        y = ((t <= tau) & (delta == 1)).astype(int)
        if len(np.unique(y)) < 2:
            continue
        aurocs.append(roc_auc_score(y, scores))

    return float(np.mean(aurocs)) if aurocs else float("nan")


# ---------------------------------------------------------------------------
# KM censoring survival (shared helper)
# ---------------------------------------------------------------------------

def _km_censoring_survival(t: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """S_C(t_i) via Kaplan-Meier on the censoring distribution."""
    unique_times = np.sort(np.unique(t))
    s = 1.0
    km = {}
    for u in unique_times:
        at_risk = np.sum(t >= u)
        n_cens  = np.sum((t == u) & (delta == 0))
        if at_risk > 0:
            s *= (1.0 - n_cens / at_risk)
        km[u] = max(s, 1e-6)
    return np.array([km[ti] for ti in t])


# ---------------------------------------------------------------------------
# Brier score (IPCW-corrected) at fixed τ
# ---------------------------------------------------------------------------

def brier_score(
    t_train: np.ndarray,
    delta_train: np.ndarray,
    t_test: np.ndarray,
    delta_test: np.ndarray,
    proba_test: np.ndarray,
    tau: float,
) -> float:
    """
    IPCW-corrected Brier score at horizon τ.

    BS(τ) = (1/n) Σ_i [
        (1 - p̂_i)² / S_C(t_i) × I(t_i ≤ τ, δ_i=1)
      + (0 - p̂_i)² / S_C(τ)  × I(t_i > τ)
    ]

    S_C estimated from TRAINING data (Kaplan-Meier on censoring).
    p̂_i = predicted P(T ≤ τ | X_i) — so higher = higher risk.

    Parameters
    ----------
    t_train, delta_train : survival data used to estimate S_C
    t_test, delta_test   : test survival data
    proba_test           : (n_test,) predicted P(event by τ)
    tau                  : prediction horizon

    Returns
    -------
    float : Brier score ∈ [0, 0.25], lower = better calibration+discrimination
    """
    # Estimate S_C from training data, evaluate at test times
    t_all = np.concatenate([t_train, t_test])
    d_all = np.concatenate([delta_train, delta_test])
    S_C_all = _km_censoring_survival(t_all, d_all)
    n_train = len(t_train)
    S_C_test = S_C_all[n_train:]

    # S_C at τ (use the KM value at the largest unique time ≤ τ)
    t_train_sorted = np.sort(t_train)
    candidates = t_train_sorted[t_train_sorted <= tau]
    if len(candidates) == 0:
        s_c_tau = 1.0
    else:
        # Rebuild S_C for training only to get S_C(τ)
        S_C_train = _km_censoring_survival(t_train, delta_train)
        t_at_tau = candidates[-1]
        idx = np.where(t_train == t_at_tau)[0][0]
        s_c_tau = max(_km_censoring_survival(t_train, delta_train)[idx], 1e-6)

    n = len(t_test)
    total = 0.0
    for i in range(n):
        p = float(proba_test[i])
        if t_test[i] <= tau and delta_test[i] == 1:
            # observed event before τ → true label = 1
            total += (1.0 - p) ** 2 / S_C_test[i]
        elif t_test[i] > tau:
            # survived past τ → true label = 0
            total += (0.0 - p) ** 2 / s_c_tau

    return total / n


# ---------------------------------------------------------------------------
# IBS — Integrated Brier Score
# ---------------------------------------------------------------------------

def ibs(
    t_train: np.ndarray,
    delta_train: np.ndarray,
    t_test: np.ndarray,
    delta_test: np.ndarray,
    proba_fn,          # callable: tau → (n_test,) predicted P(event by tau)
    tau_grid: list[float] | None = None,
) -> float:
    """
    Integrated Brier Score: trapezoidal integral of BS(τ) over tau_grid.

    Parameters
    ----------
    proba_fn : function that takes tau and returns (n_test,) probability array
    tau_grid : list of horizons to integrate over
    """
    if tau_grid is None:
        tau_grid = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]

    scores = [brier_score(t_train, delta_train, t_test, delta_test,
                          proba_fn(tau), tau)
              for tau in tau_grid]

    # Trapezoidal integration normalised by time range
    tau_arr = np.array(tau_grid, float)
    bs_arr  = np.array(scores, float)
    valid = ~np.isnan(bs_arr)
    if valid.sum() < 2:
        return float("nan")

    return float(np.trapz(bs_arr[valid], tau_arr[valid]) /
                 (tau_arr[valid][-1] - tau_arr[valid][0]))


# ---------------------------------------------------------------------------
# Breslow estimator — convert Cox score to calibrated survival probabilities
# ---------------------------------------------------------------------------

def breslow_predict(
    t_train: np.ndarray,
    delta_train: np.ndarray,
    scores_train: np.ndarray,
    scores_test: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Predict P(T ≤ τ | X) for test samples using the Breslow baseline estimator.

    Given fitted Cox risk scores η = X @ w (train and test), estimates the
    cumulative baseline hazard via Breslow:

        Ĥ₀(t) = Σ_{t_j ≤ t, δ_j=1}  1 / Σ_{k: t_k ≥ t_j} exp(η_k)

    Then:
        P(T ≤ τ | X_i) = 1 − exp(−Ĥ₀(τ) · exp(η_i))

    Parameters
    ----------
    t_train, delta_train : survival data used to estimate Ĥ₀
    scores_train         : (n_train,) Cox scores η on training data
    scores_test          : (n_test,)  Cox scores η on test data
    tau                  : prediction horizon

    Returns
    -------
    proba : (n_test,) predicted P(T ≤ τ | X)
    """
    t_train     = np.asarray(t_train,     float)
    delta_train = np.asarray(delta_train, float)
    scores_train = np.asarray(scores_train, float)
    scores_test  = np.asarray(scores_test,  float)

    # Stable exp: subtract max for numerical safety
    shift = scores_train.max()
    exp_train = np.exp(scores_train - shift)

    # Breslow: iterate over unique event times ≤ τ
    event_times = np.sort(np.unique(t_train[(delta_train == 1) & (t_train <= tau)]))

    H0 = 0.0
    for t_j in event_times:
        n_events_at_tj = float(np.sum((t_train == t_j) & (delta_train == 1)))
        risk_set_sum   = float(np.sum(exp_train[t_train >= t_j]))
        if risk_set_sum > 1e-30:
            H0 += n_events_at_tj / risk_set_sum

    # Predicted survival probability at τ for test samples
    exp_test = np.exp(scores_test - shift)
    proba = 1.0 - np.exp(-H0 * exp_test)
    return np.clip(proba, 1e-6, 1.0 - 1e-6)
