"""
Preprocessing utilities for real multi-center survival data.

Stateless helper functions that operate on numpy arrays only.
All scalers are fitted on training data and applied to test data
to avoid data leakage.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on X_train and apply to both splits.

    Returns
    -------
    X_tr_scaled  : (n_train, p) standardized training covariates
    X_te_scaled  : (n_test,  p) standardized test covariates
    scaler       : fitted StandardScaler (for later transform calls)
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    return X_tr_scaled, X_te_scaled, scaler


def clip_times(t: np.ndarray, t_max: float) -> np.ndarray:
    """Clip observed times to [0, t_max] (administrative censoring at t_max)."""
    return np.clip(t, a_min=None, a_max=t_max)



def impute_median(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace NaN with per-column median.  Returns (X_imputed, medians).
    Call with training data to get medians, then apply to test data via
    `np.where(np.isnan(X_test), medians, X_test)`.
    """
    medians = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), medians, X)
    return X_imp, medians


def drop_low_variance(
    X: np.ndarray,
    threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove near-constant columns (var < threshold).

    Returns
    -------
    X_filtered : (n, p') array with low-variance columns removed
    mask       : boolean mask of kept columns (length p)
    """
    mask = X.var(axis=0) >= threshold
    return X[:, mask], mask
