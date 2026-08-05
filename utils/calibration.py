"""Post-hoc probability calibration, shared by the synthetic and real-data paths.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = ["fit_platt", "is_rank_preserving"]


def fit_platt(
    p_val: np.ndarray,
    y_val: np.ndarray,
    p_tgt: np.ndarray,
    guard_monotone: bool = True,
) -> tuple[np.ndarray, np.ndarray, object, bool]:
    """Platt scaling: a 1-D logistic regression on raw probabilities, fitted on
    the source validation split and applied to the target.

    Returns ``(p_val_calibrated, p_tgt_calibrated, calibrator, fell_back)``.

    The bias term is essential: temperature-only scaling cannot correct a
    prior/offset shift between source and target.

    Why the monotonicity guard
    --------------------------
    Platt scaling is by definition a monotone *increasing* map -- it recalibrates
    probabilities, it does not re-rank patients. The fitted map
    ``p -> sigmoid(a*p + b)`` is increasing iff ``a > 0``, but nothing in an
    unconstrained logistic fit enforces that, and with ``C=1e6`` there is
    effectively no regularisation to shrink ``a`` toward zero either. On a small
    validation split whose raw probabilities happen to be weakly or negatively
    associated with the outcome, ``a`` comes out negative, the map becomes
    decreasing, and the ranking is exactly reversed: AUROC flips to ``1 - AUROC``.

    This is not hypothetical. On the I3LUNG GHD->INT transfer, seed 7 of 10
    produced target AUROC 0.276 against a median of 0.728 across the other
    seeds -- 1 - 0.724, an exact inversion -- while its ECE (0.062) sat squarely
    among the others, because calibration error is blind to ranking. It was the
    only such fit across every method, seed and transfer in the run, i.e. a
    degenerate fit on a small calibration set rather than a systematic flaw.

    Two things make it worth guarding rather than tolerating. Downstream code
    (``max_statistic_bootstrap.RANK_PRESERVING``) excludes Platt and prior
    adjustment from the discrimination test *because* their AUROC gap is
    identically zero by construction; a sign flip silently voids that. And
    averaging seeds hides it -- one inverted member barely moves an ensemble --
    so it only surfaces when seeds are evaluated separately.

    When ``a <= 0`` the calibration set does not support a valid Platt fit, so we
    fall back to the uncalibrated probabilities. 
    Set ``guard_monotone=False`` to reproduce the pre-fix behaviour.
    """
    p_val = np.asarray(p_val, dtype=float).ravel()
    p_tgt = np.asarray(p_tgt, dtype=float).ravel()
    y_val = np.asarray(y_val).astype(int).ravel()

    if len(np.unique(y_val)) < 2:
        warnings.warn(
            "Platt scaling requires both classes in the validation split; "
            "falling back to uncalibrated probabilities.",
            RuntimeWarning,
            stacklevel=2,
        )
        return p_val.copy(), p_tgt.copy(), None, True

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
    platt.fit(p_val.reshape(-1, 1), y_val)

    slope = float(np.ravel(platt.coef_)[0])
    if guard_monotone and slope <= 0:
        warnings.warn(
            f"Platt scaling fitted a non-increasing map (slope={slope:.4g}) on "
            f"{len(y_val)} validation points: this reverses the ranking rather "
            "than calibrating it. Falling back to uncalibrated probabilities.",
            RuntimeWarning,
            stacklevel=2,
        )
        return p_val.copy(), p_tgt.copy(), platt, True

    p_val_c = platt.predict_proba(p_val.reshape(-1, 1))[:, 1]
    p_tgt_c = platt.predict_proba(p_tgt.reshape(-1, 1))[:, 1]
    return p_val_c, p_tgt_c, platt, False


def is_rank_preserving(p_raw: np.ndarray, p_cal: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``p_cal`` orders samples the same way as ``p_raw``.

    The invariant that ``RANK_PRESERVING`` methods have an identically zero
    AUROC gap holds only while this does. Cheap enough to assert in tests.
    Ties in ``p_raw`` are allowed to be broken arbitrarily in ``p_cal``; only
    strict inversions count as a violation.
    """
    p_raw = np.asarray(p_raw, dtype=float).ravel()
    p_cal = np.asarray(p_cal, dtype=float).ravel()
    order = np.argsort(p_raw, kind="mergesort")
    d = np.diff(p_cal[order])
    return bool((d >= -tol).all())
