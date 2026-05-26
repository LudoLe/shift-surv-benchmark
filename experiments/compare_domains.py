"""
Metrics for comparing a synthetic held-out domain against a real held-out domain.

Three complementary perspectives:
  1. MMD          — covariate distribution distance (p(x) alignment)
  2. KM distance  — survival curve distance (p(T) alignment)
  3. AUROC-gap Δ  — predictive fidelity (does the synthetic domain reproduce
                    the degradation that the real domain exhibits?)

Composite score (lower = better synthetic fidelity):
  score = 0.4 * |auroc_gap_delta| + 0.3 * mmd + 0.2 * km_distance + 0.1 * |prevalence_gap|
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

from generator.base_generator import DomainDataset


# ---------------------------------------------------------------------------
# 1. MMD (RBF kernel, median-heuristic bandwidth)
# ---------------------------------------------------------------------------

def compute_mmd(
    X1: np.ndarray,
    X2: np.ndarray,
    gamma: Optional[float] = None,
    max_samples: int = 1000,
) -> float:
    """
    Unbiased maximum mean discrepancy with RBF kernel.

    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Parameters
    ----------
    X1, X2   : (n1, p) and (n2, p) feature matrices
    gamma    : RBF bandwidth; None → median heuristic on pooled data
    max_samples: cap on each side to keep O(n²) computation feasible

    Returns
    -------
    mmd_sq : float (can be slightly negative due to finite-sample noise)
    """
    rng = np.random.default_rng(0)
    if len(X1) > max_samples:
        X1 = X1[rng.choice(len(X1), max_samples, replace=False)]
    if len(X2) > max_samples:
        X2 = X2[rng.choice(len(X2), max_samples, replace=False)]

    if gamma is None:
        X_pool = np.vstack([X1, X2])
        dists = np.sum((X_pool[:, None, :] - X_pool[None, :, :]) ** 2, axis=-1)
        gamma = 1.0 / (np.median(dists[dists > 0]) + 1e-8)

    def rbf(A, B):
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq)

    k11 = rbf(X1, X1)
    k22 = rbf(X2, X2)
    k12 = rbf(X1, X2)

    n1, n2 = len(X1), len(X2)
    # Unbiased: zero diagonal for same-set terms
    np.fill_diagonal(k11, 0.0)
    np.fill_diagonal(k22, 0.0)

    mmd2 = (k11.sum() / (n1 * (n1 - 1))
            + k22.sum() / (n2 * (n2 - 1))
            - 2.0 * k12.mean())
    return float(mmd2)


# ---------------------------------------------------------------------------
# 2. Kaplan-Meier distance
# ---------------------------------------------------------------------------

def kaplan_meier(
    t: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nelson-Åalen / KM survival estimator.

    Returns
    -------
    times : (m,) sorted unique event times
    S     : (m,) KM survival probability at each event time
    """
    order = np.argsort(t)
    t_s = t[order]
    d_s = delta[order]

    n = len(t_s)
    at_risk = n
    S = 1.0
    times, S_vals = [], []

    prev_t = None
    for i, (ti, di) in enumerate(zip(t_s, d_s)):
        if ti != prev_t:
            if prev_t is not None:
                times.append(prev_t)
                S_vals.append(S)
            prev_t = ti
        if di == 1:
            S *= (1.0 - 1.0 / at_risk)
        at_risk -= 1

    if prev_t is not None:
        times.append(prev_t)
        S_vals.append(S)

    return np.array(times), np.array(S_vals)


def compute_km_distance(
    t1: np.ndarray,
    d1: np.ndarray,
    t2: np.ndarray,
    d2: np.ndarray,
    n_eval: int = 50,
) -> float:
    """
    Mean absolute difference between two KM survival curves evaluated on a
    common grid of time points.

    Parameters
    ----------
    t1, d1   : times and event indicators for domain 1
    t2, d2   : times and event indicators for domain 2
    n_eval   : number of evaluation points on the common grid

    Returns
    -------
    km_dist : float in [0, 1]
    """
    times1, S1 = kaplan_meier(t1, d1)
    times2, S2 = kaplan_meier(t2, d2)

    t_min = max(times1.min(), times2.min())
    t_max = min(times1.max(), times2.max())
    if t_min >= t_max:
        return 1.0   # non-overlapping time ranges → maximal distance

    grid = np.linspace(t_min, t_max, n_eval)
    S1_grid = np.interp(grid, times1, S1, left=1.0, right=S1[-1])
    S2_grid = np.interp(grid, times2, S2, left=1.0, right=S2[-1])

    return float(np.mean(np.abs(S1_grid - S2_grid)))


# ---------------------------------------------------------------------------
# 3. AUROC-gap delta
# ---------------------------------------------------------------------------

def _fit_lr_auroc(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> float:
    """Fit LR on pooled source x, evaluate on target. Returns target AUROC."""
    X_src = np.vstack([ds.x for ds in source_datasets])
    y_src = np.concatenate([ds.y for ds in source_datasets])

    if len(np.unique(y_src)) < 2:
        return float("nan")

    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src)
    X_tgt_s = scaler.transform(target_dataset.x)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=0)
        lr.fit(X_src_s, y_src)

    if len(np.unique(target_dataset.y)) < 2:
        return 0.5  # degenerate domain — treat as random-chance AUROC
    probs = lr.predict_proba(X_tgt_s)[:, 1]
    return float(roc_auc_score(target_dataset.y, probs))


def compute_auroc_gap_delta(
    src_datasets: list[DomainDataset],
    synth_ds: DomainDataset,
    real_ds: DomainDataset,
) -> float:
    """
    AUROC-gap delta: how well does the synthetic domain reproduce the
    predictive degradation observed on the real held-out domain?

    auroc_gap_delta = |AUROC_gap(real) - AUROC_gap(synth)|

    where AUROC_gap(domain) = AUROC_source - AUROC_target(domain).

    A lower delta means the synthetic domain is a more faithful proxy for
    the real domain in terms of transfer difficulty.

    Parameters
    ----------
    src_datasets : list of source DomainDatasets (same for both comparisons)
    synth_ds     : synthetic held-out domain
    real_ds      : real held-out domain (ground truth)

    Returns
    -------
    delta : float (lower = better)
    """
    # Source AUROC (pooled source, evaluated on itself via the same LR)
    X_src = np.vstack([ds.x for ds in src_datasets])
    y_src = np.concatenate([ds.y for ds in src_datasets])
    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=0)
        lr.fit(X_src_s, y_src)

    if len(np.unique(y_src)) < 2:
        return float("nan")
    probs_src = lr.predict_proba(X_src_s)[:, 1]
    auroc_source = float(roc_auc_score(y_src, probs_src))

    auroc_real  = _fit_lr_auroc(src_datasets, real_ds)
    auroc_synth = _fit_lr_auroc(src_datasets, synth_ds)

    gap_real  = auroc_source - auroc_real
    gap_synth = auroc_source - auroc_synth

    return float(abs(gap_real - gap_synth))


# ---------------------------------------------------------------------------
# Composite domain score
# ---------------------------------------------------------------------------

def km_prevalence_at_tau(ds: DomainDataset, tau: float) -> float:
    """1 - S_KM(tau): censoring-adjusted event probability at horizon tau."""
    t = ds.t_obs if ds.t_obs is not None else ds.T
    d = ds.delta if ds.delta is not None else np.ones(ds.n, dtype=int)
    times, S = kaplan_meier(t, d)
    s_tau = float(np.interp(tau, times, S, left=1.0, right=float(S[-1])))
    return float(np.clip(1.0 - s_tau, 0.0, 1.0))


def compute_prevalence_gap(
    ds1: DomainDataset,
    ds2: DomainDataset,
    tau: Optional[float] = None,
) -> float:
    """
    Absolute KM-adjusted prevalence gap between two datasets.
    Uses 1-KM(tau) so the comparison is censoring-corrected on both sides.
    Falls back to raw y.mean() if tau is not provided.
    """
    if tau is not None:
        return float(abs(km_prevalence_at_tau(ds1, tau) - km_prevalence_at_tau(ds2, tau)))
    return float(abs(ds1.prevalence - ds2.prevalence))


def domain_score(
    synth_ds: DomainDataset,
    real_ds: DomainDataset,
    src_datasets: list[DomainDataset],
    weights: Optional[dict[str, float]] = None,
    tau: Optional[float] = None,
) -> dict:
    """
    Compute all comparison metrics and a composite score.

    Composite (lower = better):
      score = w_auroc * auroc_gap_delta
            + w_mmd   * mmd
            + w_km    * km_distance
            + w_prev  * prevalence_gap

    Parameters
    ----------
    synth_ds    : synthetic held-out DomainDataset
    real_ds     : real held-out DomainDataset (DomainDataset wrapping real data)
    src_datasets: source DomainDatasets (for AUROC-gap computation)
    weights     : dict of metric weights; defaults to paper recommendation

    Returns
    -------
    dict with individual metrics and composite score
    """
    if weights is None:
        weights = {"auroc_gap_delta": 0.4, "mmd": 0.3, "km_distance": 0.2, "prevalence_gap": 0.1}

    mmd = compute_mmd(synth_ds.x, real_ds.x)
    km_dist = compute_km_distance(
        synth_ds.t_obs if synth_ds.t_obs is not None else synth_ds.T,
        synth_ds.delta  if synth_ds.delta  is not None else np.ones(synth_ds.n, dtype=int),
        real_ds.t_obs   if real_ds.t_obs   is not None else real_ds.T,
        real_ds.delta   if real_ds.delta   is not None else np.ones(real_ds.n, dtype=int),
    )
    auroc_delta = compute_auroc_gap_delta(src_datasets, synth_ds, real_ds)
    prev_gap = compute_prevalence_gap(synth_ds, real_ds, tau=tau)

    composite = (
        weights["auroc_gap_delta"] * auroc_delta
        + weights["mmd"]           * max(mmd, 0.0)
        + weights["km_distance"]   * km_dist
        + weights["prevalence_gap"]* prev_gap
    )

    # KM-based prevalence for interpretable reporting
    km_prev_real  = round(km_prevalence_at_tau(real_ds,  tau), 4) if tau else None
    km_prev_synth = round(km_prevalence_at_tau(synth_ds, tau), 4) if tau else None

    return {
        "km_prev_real":    km_prev_real,
        "km_prev_synth":   km_prev_synth,
        "mmd":             round(mmd,         4),
        "km_distance":     round(km_dist,     4),
        "auroc_gap_delta": round(auroc_delta, 4),
        "prevalence_gap":  round(prev_gap,    4),
        "composite":       round(composite,   4),
        "weights":         weights,
        "synth_n":         synth_ds.n,
        "real_n":          real_ds.n,
    }
