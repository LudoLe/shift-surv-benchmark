"""
Evaluation metrics for fixed-horizon binary prediction.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC; returns NaN if only one class present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUPRC (average precision)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def prevalence(y: np.ndarray) -> float:
    return float(y.mean())


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (equal-width probability bins).

    ECE = Σ_b (n_b / n) * |mean_label_b − mean_prob_b|

    A well-calibrated model has ECE ≈ 0.  Positive ECE always means the
    model's predicted probabilities do not match the observed event rates.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ece_val += (n_b / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece_val)


def summarise_performance(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "",
) -> dict:
    return {
        "label": label,
        "n": len(y_true),
        "prevalence": round(prevalence(y_true), 4),
        "auroc": round(auroc(y_true, y_score), 4),
        "auprc": round(auprc(y_true, y_score), 4),
        "brier": round(brier(y_true, y_score), 4),
    }
