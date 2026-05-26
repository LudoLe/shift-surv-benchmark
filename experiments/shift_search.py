"""
Shift scan utilities: singleton sweep + conditional pair search.

Workflow
--------
1. `singleton_scan`  — for each shift type, sweep its magnitude grid, score
                        each candidate, return a DataFrame ranked by composite.
2. `pair_scan`       — take the two best singleton shift types, form a grid of
                        their paired magnitudes, score each combination.
                        Only called when singleton improvement exceeds a threshold.
3. `best_config`     — pick the single best config from singleton (or pair) results.

Scoring uses `domain_score` from compare_domains.py (MMD + KM distance +
AUROC-gap Δ + KM-prevalence gap weighted composite).
"""

from __future__ import annotations
from itertools import product
from typing import Optional
import numpy as np
import pandas as pd

from generator.base_generator import SyntheticGenerator, DomainDataset
from generator.parameters import BaseParams, DomainParams
from generator.shift_components import apply_shift_config, SINGLETON_GRIDS
from experiments.compare_domains import domain_score


# ---------------------------------------------------------------------------
# Core evaluation primitive
# ---------------------------------------------------------------------------

def _generate_and_score(
    shifted_params: DomainParams,
    base: BaseParams,
    A: np.ndarray,
    src_ds_list: list[DomainDataset],
    real_ds: DomainDataset,
    tau: float,
    seed: int = 42,
) -> dict:
    """Generate one synthetic domain with shifted_params and return scores."""
    gen = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)
    synth_ds = gen.generate_domain(
        shifted_params,
        domain_id=shifted_params.domain_id,
        seed=seed,
    )
    scores = domain_score(synth_ds, real_ds, src_ds_list, tau=tau)
    return {
        k: v for k, v in scores.items()
        if k not in ("weights", "synth_n", "real_n")
    }


# ---------------------------------------------------------------------------
# Singleton scan
# ---------------------------------------------------------------------------

def singleton_scan(
    held_params: DomainParams,
    base: BaseParams,
    A: np.ndarray,
    src_ds_list: list[DomainDataset],
    real_ds: DomainDataset,
    tau: float,
    seed: int = 42,
    grids: Optional[dict[str, list[float]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Sweep each shift type over its magnitude grid and score all candidates.

    Parameters
    ----------
    held_params   : reference DomainParams for the held-out center
    base          : BaseParams for synthetic generation
    A             : (p, k) shared loading matrix
    src_ds_list   : source DomainDatasets for AUROC-gap computation
    real_ds       : real held-out DomainDataset (ground truth)
    tau           : prediction horizon (for KM-prevalence scoring)
    seed          : RNG seed for synthetic generation
    grids         : per-type magnitude grids; defaults to SINGLETON_GRIDS
    verbose       : print progress

    Returns
    -------
    DataFrame with columns:
      shift_type, magnitude, km_prev_real, km_prev_synth,
      mmd, km_distance, auroc_gap_delta, prevalence_gap, composite
    Sorted ascending by composite (best first).
    """
    if grids is None:
        grids = SINGLETON_GRIDS

    rows = []
    for shift_type, grid in grids.items():
        if verbose:
            print(f"    scanning {shift_type} ({len(grid)} points)...")
        for mag in grid:
            config = {shift_type: mag}
            shifted = apply_shift_config(held_params, config)
            scores = _generate_and_score(shifted, base, A, src_ds_list, real_ds, tau, seed)
            rows.append({
                "shift_type": shift_type,
                "magnitude":  round(float(mag), 6),
                **scores,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("composite").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pair scan
# ---------------------------------------------------------------------------

def pair_scan(
    held_params: DomainParams,
    base: BaseParams,
    A: np.ndarray,
    src_ds_list: list[DomainDataset],
    real_ds: DomainDataset,
    tau: float,
    best_singleton_df: pd.DataFrame,
    n_top_per_type: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Combine the two best-performing singleton shift types (by their best magnitude)
    into a pairwise grid and score all combinations.

    Only the top `n_top_per_type` magnitudes from each of the two best shift
    types are crossed, keeping the search tractable (≤ n_top² evaluations).

    Parameters
    ----------
    held_params       : reference DomainParams
    best_singleton_df : output of `singleton_scan`, sorted by composite
    n_top_per_type    : number of top magnitudes to keep per shift type

    Returns
    -------
    DataFrame with columns:
      shift_type_1, magnitude_1, shift_type_2, magnitude_2,
      mmd, km_distance, auroc_gap_delta, prevalence_gap, composite
    Sorted ascending by composite.
    """
    # Identify the two shift types with the best (lowest) composite
    best_per_type = (
        best_singleton_df
        .groupby("shift_type")["composite"]
        .min()
        .sort_values()
    )
    if len(best_per_type) < 2:
        return pd.DataFrame()

    type1, type2 = best_per_type.index[:2]

    # Top magnitudes for each type
    def top_mags(stype: str) -> list[float]:
        sub = best_singleton_df[best_singleton_df["shift_type"] == stype]
        return sub.nsmallest(n_top_per_type, "composite")["magnitude"].tolist()

    mags1 = top_mags(type1)
    mags2 = top_mags(type2)

    if verbose:
        print(f"    pair scan: {type1} × {type2}  ({len(mags1)}×{len(mags2)} = {len(mags1)*len(mags2)} combos)")

    rows = []
    for m1, m2 in product(mags1, mags2):
        config = {type1: m1, type2: m2}
        shifted = apply_shift_config(held_params, config)
        scores = _generate_and_score(shifted, base, A, src_ds_list, real_ds, tau, seed)
        rows.append({
            "shift_type_1": type1,
            "magnitude_1":  round(float(m1), 6),
            "shift_type_2": type2,
            "magnitude_2":  round(float(m2), 6),
            **scores,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("composite").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def best_singleton(df: pd.DataFrame) -> dict:
    """Return the best row from a singleton_scan DataFrame as a dict."""
    if df.empty:
        return {}
    row = df.iloc[0]
    return row.to_dict()


def best_pair(df: pd.DataFrame) -> dict:
    """Return the best row from a pair_scan DataFrame as a dict."""
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def improvement(baseline_composite: float, best_composite: float) -> float:
    """Fractional improvement: (baseline - best) / baseline."""
    if baseline_composite <= 0:
        return 0.0
    return float((baseline_composite - best_composite) / baseline_composite)
