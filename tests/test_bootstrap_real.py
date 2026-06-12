"""Unit checks for the patient-level bootstrap used on the real-data experiment.

The real I3LUNG data is private, so these tests drive _bootstrap_real_gaps with
synthetic predictions where the right answer is known by construction:
  * a method whose probabilities equal the reference must have a zero gap and a
    non-significant p-value;
  * a method that is strictly better-ranked must show a positive AUROC gap whose
    bootstrap CI excludes 0.

Run:  pytest tests/test_bootstrap_real.py    (or: python tests/test_bootstrap_real.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.run_real_data import _bootstrap_real_gaps


def _make_preds(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Reference: weakly informative score.
    ref = 0.5 + 0.15 * (2 * y - 1) + 0.25 * rng.standard_normal(n)
    ref = np.clip(ref, 0, 1)
    # Strong: cleanly separates the classes (better AUROC).
    strong = np.clip(0.5 + 0.45 * (2 * y - 1) + 0.05 * rng.standard_normal(n), 0, 1)
    # Stack a single "seed" row: shape (n_seeds=1, n).
    return {
        "lr_erm": ref[None, :],
        "copy":   ref[None, :],
        "strong": strong[None, :],
    }, y


def test_identical_method_has_zero_gap():
    preds, y = _make_preds()
    rows = _bootstrap_real_gaps(preds, y, reference="lr_erm", n_boot=1000, seed=1)
    copy = next(r for r in rows if r["method"] == "copy")
    assert abs(copy["auroc_gap_vs_ref"]) < 1e-6
    assert copy["auroc_gap_p"] >= 0.99  # never significant against itself


def test_strong_method_beats_reference():
    preds, y = _make_preds()
    rows = _bootstrap_real_gaps(preds, y, reference="lr_erm", n_boot=2000, seed=1)
    strong = next(r for r in rows if r["method"] == "strong")
    assert strong["auroc_gap_vs_ref"] > 0
    assert strong["auroc_gap_lo"] > 0          # 95% CI excludes 0
    assert strong["auroc_gap_p"] < 0.05


def test_seed_averaging_shapes():
    """Multiple seeds are averaged per patient before resampling."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    P = np.clip(0.5 + 0.2 * (2 * y - 1) + 0.2 * rng.standard_normal((3, n)), 0, 1)
    rows = _bootstrap_real_gaps({"lr_erm": P, "m": P}, y, reference="lr_erm", n_boot=500)
    assert rows and rows[0]["n_tgt"] == n


if __name__ == "__main__":
    test_identical_method_has_zero_gap()
    test_strong_method_beats_reference()
    test_seed_averaging_shapes()
    print("PASS: bootstrap behaves as specified.")
