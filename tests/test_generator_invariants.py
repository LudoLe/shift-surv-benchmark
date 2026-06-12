"""Invariant checks on the synthetic generator.

These guard the design properties the paper relies on, so a future change to the
generator that silently breaks (e.g.) the prevalence-preserving covariate shift
fails loudly instead of quietly altering every downstream result.

Run:  pytest tests/test_generator_invariants.py    (or: python tests/test_generator_invariants.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from generator.scenarios import generate_scenario


def test_reproducible_same_seed():
    """Same scenario + seed must produce identical covariates and labels."""
    a = generate_scenario("covariate_shift", n_source_domains=1, seed=7)
    b = generate_scenario("covariate_shift", n_source_domains=1, seed=7)
    assert np.allclose(a["target"].x, b["target"].x)
    assert np.array_equal(a["target"].y, b["target"].y)
    assert np.allclose(a["source"][0].x, b["source"][0].x)


def test_different_seed_changes_data():
    """Different seeds must produce different draws (no accidental seed pinning)."""
    a = generate_scenario("covariate_shift", n_source_domains=1, seed=1)
    b = generate_scenario("covariate_shift", n_source_domains=1, seed=2)
    assert not np.allclose(a["target"].x, b["target"].x)


def test_no_shift_prevalence_matched():
    """no_shift: source and target are the same DGP -> prevalence within noise."""
    d = generate_scenario("no_shift", n_source_domains=1, seed=0)
    src_prev = d["source"][0].y.mean()
    tgt_prev = d["target"].y.mean()
    assert abs(src_prev - tgt_prev) < 0.08, (src_prev, tgt_prev)


def test_covariate_shift_preserves_prevalence():
    """covariate_shift moves mu_inv orthogonally to w, so prevalence should be
    approximately preserved despite a large P(X) shift (paper Section 5 claim)."""
    diffs = []
    for s in range(5):
        d = generate_scenario("covariate_shift", n_source_domains=1, seed=s)
        diffs.append(abs(d["source"][0].y.mean() - d["target"].y.mean()))
    assert np.mean(diffs) < 0.10, f"mean |prev gap| too large: {np.mean(diffs):.3f}"


def test_concept_shift_flips_selected_weights():
    """concept_shift negates exactly the configured subset of w in the target."""
    d = generate_scenario("concept_shift", n_source_domains=1, seed=0)
    w_src = d["source"][0].params.w
    w_tgt = d["target"].params.w
    # Default flips indices (1, 2); index 0 (dominant factor) is preserved.
    assert np.isclose(w_tgt[0], w_src[0])
    assert np.isclose(w_tgt[1], -w_src[1])
    assert np.isclose(w_tgt[2], -w_src[2])


def test_spurious_shift_breaks_latent_correlation_at_target():
    """spurious_shift: source has cross-correlated latents; target is independent."""
    d = generate_scenario("spurious_shift", n_source_domains=1, seed=0)
    src_cov = d["source"][0].params.latent_cov
    tgt_cov = d["target"].params.latent_cov
    assert src_cov is not None and np.any(np.abs(src_cov - np.eye(src_cov.shape[0])) > 1e-6)
    assert tgt_cov is not None and np.allclose(tgt_cov, np.eye(tgt_cov.shape[0]))


def test_main_benchmark_is_uncensored():
    """The core scenarios carry no censoring: every event is observed (delta==1)."""
    for sc in ["no_shift", "covariate_shift", "concept_shift", "baseline_hazard_shift"]:
        d = generate_scenario(sc, n_source_domains=1, seed=0)
        assert (d["target"].delta == 1).all(), f"{sc} unexpectedly censored"


if __name__ == "__main__":
    test_reproducible_same_seed()
    test_different_seed_changes_data()
    test_no_shift_prevalence_matched()
    test_covariate_shift_preserves_prevalence()
    test_concept_shift_flips_selected_weights()
    test_spurious_shift_breaks_latent_correlation_at_target()
    test_main_benchmark_is_uncensored()
    print("PASS: generator invariants hold.")
