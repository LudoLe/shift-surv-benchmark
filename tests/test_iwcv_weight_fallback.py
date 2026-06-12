"""
Regression test for the IWCV importance-weight wiring (review finding #1).

Background (the bug)
--------------------
fit_dann / fit_cdan used to compute importance weights on the TRAIN split:

    iw = _importance_weights(X_tr_s, X_tgt_s)     # len == len(X_tr_s)

then use them to weight the VALIDATION AUROC:

    score = _weighted_auroc(y_val, p_val, iw)     # len(y_val) == len(X_val_s)

Because len(X_tr_s) != len(X_val_s) (80/20 split), roc_auc_score raised on the
sample_weight length mismatch and _weighted_auroc silently fell back to the
UNWEIGHTED AUROC, making IWCV inert.

The fix
-------
_importance_weights now takes X_eval: the domain classifier is still trained on
(X_tr_s vs X_tgt_s), but the weights are evaluated on X_eval=X_val_s, so they
align with the validation set IWCV scores and stay out-of-sample.

This test asserts:
  * the legacy mis-aligned call still triggers the silent fallback (documents
    why the X_eval argument is necessary), and
  * the fixed call yields val-aligned, non-trivial weights that genuinely
    change the AUROC (the weighting is now live).

Run:  PYTHONPATH=. python tests/test_iwcv_weight_fallback.py   (or: pytest -s)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import experiments.adaptation_methods as m


def _make_source_target(seed: int = 0):
    """Synthetic source/target with a genuine covariate shift so w(x) != const."""
    rng = np.random.default_rng(seed)
    n_src, n_tgt, p = 400, 300, 8

    # Source: standard normal features; label depends on first two features.
    X_src = rng.standard_normal((n_src, p))
    logit = 1.2 * X_src[:, 0] - 0.8 * X_src[:, 1]
    y_src = (rng.uniform(size=n_src) < 1 / (1 + np.exp(-logit))).astype(int)

    # Target: shifted mean -> domain classifier is informative -> weights vary.
    X_tgt = rng.standard_normal((n_tgt, p)) + 1.5
    return X_src, y_src, X_tgt


def _split(X_src, y_src):
    """Reproduce the exact split used inside fit_dann / fit_cdan."""
    return train_test_split(
        X_src, y_src, test_size=0.2, random_state=0, stratify=y_src,
    )


def _val_probs(y_val):
    """Deterministic, non-trivial validation probabilities for scoring."""
    p_val = np.clip(np.linspace(0.01, 0.99, len(y_val)), 0, 1)
    np.random.default_rng(1).shuffle(p_val)
    return p_val


def test_legacy_call_still_silently_falls_back():
    """The old wiring (weights on train) must still degrade to unweighted AUROC."""
    X_src, y_src, X_tgt = _make_source_target()
    X_tr, X_val, y_tr, y_val = _split(X_src, y_src)
    p_val = _val_probs(y_val)

    iw_train = m._importance_weights(X_tr, X_tgt)            # legacy: aligned to train
    assert len(iw_train) == len(X_tr) != len(X_val)
    print(f"[legacy] len(iw)={len(iw_train)}  len(y_val)={len(y_val)} -> mismatch")

    weighted = m._weighted_auroc(y_val, p_val, iw_train)
    unweighted = roc_auc_score(y_val, p_val)
    print(f"[legacy] _weighted_auroc={weighted:.6f}  unweighted={unweighted:.6f}")
    assert weighted == unweighted, "legacy path should fall back to unweighted"


def test_fixed_call_uses_val_aligned_weights():
    """The fix (X_eval=X_val) must produce val-aligned, non-trivial weighting."""
    X_src, y_src, X_tgt = _make_source_target()
    X_tr, X_val, y_tr, y_val = _split(X_src, y_src)
    p_val = _val_probs(y_val)

    # Exact call sequence now used in fit_dann / fit_cdan.
    iw = m._importance_weights(X_tr, X_tgt, X_eval=X_val)

    # (1) weights align with the validation set being scored
    assert len(iw) == len(y_val), "weights must match the val set length"
    print(f"[fixed] len(iw)={len(iw)}  len(y_val)={len(y_val)} -> aligned")

    # (2) _weighted_auroc no longer raises -> weighting is actually applied
    weighted = m._weighted_auroc(y_val, p_val, iw)
    unweighted = roc_auc_score(y_val, p_val)
    print(f"[fixed] _weighted_auroc={weighted:.6f}  unweighted={unweighted:.6f}")
    assert np.isclose(weighted, roc_auc_score(y_val, p_val, sample_weight=iw)), (
        "_weighted_auroc must equal the explicitly weighted score (no fallback)"
    )

    # (3) the weights are non-trivial: they change the score
    assert not np.isclose(weighted, unweighted, atol=1e-9), (
        "weights are non-trivial; the IWCV weighting is now live"
    )
    print(f"[fixed] weighting changes AUROC by {abs(weighted - unweighted):.6f}")


if __name__ == "__main__":
    test_legacy_call_still_silently_falls_back()
    print()
    test_fixed_call_uses_val_aligned_weights()
    print("\nPASS: IWCV weights are now evaluated on the validation set.")
