"""Lightweight covariate-shift diagnostics.

Used to characterise how far apart two domains are in feature space, to argue
that a recovered performance gain reflects a *covariate / measurement* shift
(a move of p(x)) rather than a concept shift (a move of p(y|x)).

The RBF-MMD here reproduces the metric used by the (now source-less) Paper-1
``compare_domains`` module: an unbiased squared MMD with a median-heuristic
bandwidth, computed on standardised features.
"""
from __future__ import annotations

import numpy as np


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    # ||a-b||^2 = |a|^2 + |b|^2 - 2 a·b
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    sq = np.clip(aa + bb - 2.0 * A @ B.T, 0.0, None)
    return np.exp(-gamma * sq)


def _median_gamma(X: np.ndarray, rng: np.random.Generator, max_n: int = 500) -> float:
    """Median-heuristic bandwidth: gamma = 1 / median(pairwise sq. distances)."""
    n = len(X)
    if n > max_n:
        X = X[rng.choice(n, max_n, replace=False)]
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return 1.0 / med if med > 0 else 1.0


def rbf_mmd(
    Xs: np.ndarray,
    Xt: np.ndarray,
    gamma: float | None = None,
    max_samples: int = 800,
    seed: int = 0,
) -> float:
    """Unbiased squared RBF-MMD between source features Xs and target Xt.

    Both inputs should already be on a common scale (e.g. standardised with a
    source-fit scaler). Subsamples each side to ``max_samples`` for tractability.
    """
    rng = np.random.default_rng(seed)
    Xs = np.asarray(Xs, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    if len(Xs) > max_samples:
        Xs = Xs[rng.choice(len(Xs), max_samples, replace=False)]
    if len(Xt) > max_samples:
        Xt = Xt[rng.choice(len(Xt), max_samples, replace=False)]

    if gamma is None:
        gamma = _median_gamma(np.vstack([Xs, Xt]), rng)

    Kss = _rbf_kernel(Xs, Xs, gamma)
    Ktt = _rbf_kernel(Xt, Xt, gamma)
    Kst = _rbf_kernel(Xs, Xt, gamma)

    m, n = len(Xs), len(Xt)
    # Unbiased estimator: exclude the diagonal self-similarity terms.
    sum_ss = (Kss.sum() - np.trace(Kss)) / (m * (m - 1)) if m > 1 else 0.0
    sum_tt = (Ktt.sum() - np.trace(Ktt)) / (n * (n - 1)) if n > 1 else 0.0
    sum_st = Kst.mean()
    return float(max(0.0, sum_ss + sum_tt - 2.0 * sum_st))


def rbf_mmd_permutation(
    Xs: np.ndarray,
    Xt: np.ndarray,
    n_perm: int = 1000,
    max_samples: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Permutation test for the RBF-MMD: is the source↔target gap larger than
    chance? Returns (observed_mmd, p_value).

    The bandwidth is fixed (pooled median heuristic) across permutations, then
    the pooled sample is repeatedly shuffled and split into two groups of the
    original sizes; p = fraction of permuted MMDs >= observed (add-one smoothed).
    """
    rng = np.random.default_rng(seed)
    Xs = np.asarray(Xs, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    if len(Xs) > max_samples:
        Xs = Xs[rng.choice(len(Xs), max_samples, replace=False)]
    if len(Xt) > max_samples:
        Xt = Xt[rng.choice(len(Xt), max_samples, replace=False)]

    pooled = np.vstack([Xs, Xt])
    gamma = _median_gamma(pooled, rng)
    m = len(Xs)

    def _mmd(a, b):
        Kss = _rbf_kernel(a, a, gamma)
        Ktt = _rbf_kernel(b, b, gamma)
        Kst = _rbf_kernel(a, b, gamma)
        ma, nb = len(a), len(b)
        s_ss = (Kss.sum() - np.trace(Kss)) / (ma * (ma - 1)) if ma > 1 else 0.0
        s_tt = (Ktt.sum() - np.trace(Ktt)) / (nb * (nb - 1)) if nb > 1 else 0.0
        return s_ss + s_tt - 2.0 * Kst.mean()

    observed = _mmd(Xs, Xt)
    count = 0
    N = len(pooled)
    for _ in range(n_perm):
        perm = rng.permutation(N)
        if _mmd(pooled[perm[:m]], pooled[perm[m:]]) >= observed:
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return float(max(0.0, observed)), float(pval)
