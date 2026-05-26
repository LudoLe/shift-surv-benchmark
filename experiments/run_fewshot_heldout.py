"""
Few-shot held-out calibration experiment.

Question: how many patients from the held-out center are needed to
reproduce it synthetically?

Protocol (per LOODO fold):
  1. Fit shared structure (A, w, scaler) on 5 source centers — unchanged.
  2. Sample m patients from held-out center (calibration set).
  3. Fit held-out domain params from those m patients only.
  4. Generate synthetic held-out (size = full held-out n).
  5. Score against the FULL held-out real domain.

Variants:
  A  covariate-only  — mu_d from X_m; lambda0 = source mean (no outcomes used)
  B  full few-shot   — mu_d from X_m; lambda0 from KM(t_m, delta_m),
                       with fallback to source mean if < 3 events
  C  full info       — current pipeline using all held-out patients (upper bound)

Output files (in --out_dir):
  fewshot_results.csv    — one row per (center, m, seed, variant)
  fewshot_summary.csv    — mean ± std across seeds per (center, m, variant)

Usage
-----
  python -m experiments.run_fewshot_heldout [options]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median

from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams
from generator.fitting import (
    fit_reference_generator,
    fit_domain_params,
    fit_domain_params_fewshot,
    make_real_domain_dataset,
)
from experiments.compare_domains import domain_score

# Reuse the fold-setup helper from shift experiments
from experiments.run_shift_experiments import _prepare_fold


# ---------------------------------------------------------------------------
# Subsampling helper
# ---------------------------------------------------------------------------

def subsample_held_out(
    X_s: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw m patients uniformly at random without replacement.

    Returns (X_m, t_m, delta_m) — the calibration subset.
    """
    n = len(t)
    m = min(m, n)
    idx = rng.choice(n, size=m, replace=False)
    return X_s[idx], t[idx], delta[idx]


def _clean_scores(scores: dict) -> dict:
    """Remove non-scalar entries from domain_score output."""
    return {k: v for k, v in scores.items()
            if k not in ("weights", "synth_n", "real_n")}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_fewshot_experiment(
    m_grid: list[int] | None = None,
    n_seeds: int = 10,
    k_total: int = 6,
    tau: float = 12.0,
    nu: float = 1.5,
    seed: int = 42,
    out_dir: str = "outputs/tables/",
    verbose: bool = True,
) -> None:
    """
    Run the full few-shot calibration experiment across all centers.

    Parameters
    ----------
    m_grid    : list of calibration set sizes to test
    n_seeds   : number of independent random subsamples per (center, m) cell
    k_total   : number of latent dimensions (must match run_shift_experiments)
    tau       : prediction horizon in months
    nu        : Weibull shape (fixed)
    seed      : base RNG seed
    out_dir   : where to write CSV outputs
    """
    if m_grid is None:
        m_grid = [5, 10, 20, 30, 50, 100]

    centers, _ = load_all_centers(registry_module=registry)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []

    for held_idx in range(len(centers)):
        held_out = centers[held_idx]
        hid      = held_out["center_id"]
        name     = registry.CENTER_NAMES.get(hid, str(hid))
        n_full   = held_out["n"]

        if verbose:
            print(f"\n=== Center {name} (n={n_full}) ===")

        # ----------------------------------------------------------------
        # Fit shared structure once per fold (expensive part)
        # ----------------------------------------------------------------
        sources = [c for i, c in enumerate(centers) if i != held_idx]
        X_src_pool = np.vstack([c["X"] for c in sources])
        _, src_medians = impute_median(X_src_pool)
        sources_imp = [
            {**c, "X": np.where(np.isnan(c["X"]), src_medians, c["X"])}
            for c in sources
        ]
        X_held_imp = np.where(np.isnan(held_out["X"]), src_medians, held_out["X"])

        A, w, source_params, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
        )
        p_dim = held_out["X"].shape[1]
        base  = BaseParams(k_inv=k_total, k_spu=0, p=p_dim, tau=tau, seed=seed)
        gen   = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)

        X_held_s = scaler.transform(X_held_imp)

        # Source domain datasets for AUROC-gap computation
        src_ds_list = []
        for sp, sc in zip(source_params, sources_imp):
            X_sc_s = scaler.transform(sc["X"])
            src_ds_list.append(make_real_domain_dataset(
                X_s=X_sc_s, t_obs=sc["t"], delta=sc["delta"],
                A=A, w=w, tau=tau, params=sp, domain_id=sc["center_id"],
            ))

        # Real held-out DomainDataset — evaluation target, NEVER used for calibration
        held_full_params = fit_domain_params(
            X_d=X_held_s, t_d=held_out["t"], delta_d=held_out["delta"],
            A=A, w=w, tau=tau, nu=nu, domain_id=hid, n_synthetic=n_full,
        )
        real_ds = make_real_domain_dataset(
            X_s=X_held_s, t_obs=held_out["t"], delta=held_out["delta"],
            A=A, w=w, tau=tau, params=held_full_params, domain_id=hid,
        )

        # Source lambda0 mean — fallback for Variant A and low-event Variant B
        source_lambda0_mean = float(np.mean([p.lambda0 for p in source_params]))

        # ----------------------------------------------------------------
        # Variant C — full information (upper bound, no subsampling needed)
        # ----------------------------------------------------------------
        synth_C = gen.generate_domain(held_full_params, domain_id=hid)
        c_scores = domain_score(synth_C, real_ds, src_ds_list, tau=tau)
        rows.append({
            "center": name, "m": n_full, "seed": 0, "variant": "C",
            "km_used": True,
            **_clean_scores(c_scores),
        })
        if verbose:
            print(f"  C  m=full  composite={c_scores['composite']:.4f}")

        # ----------------------------------------------------------------
        # Variants A and B — loop over m and seeds
        # ----------------------------------------------------------------
        for m in m_grid:
            if m >= n_full:
                if verbose:
                    print(f"  skipping m={m} (≥ n_full={n_full})")
                continue

            composites_A, composites_B = [], []

            for s in range(n_seeds):
                # Reproducible but independent RNG per (center, m, seed)
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                X_m, t_m, delta_m = subsample_held_out(
                    X_held_s, held_out["t"], held_out["delta"], m, rng,
                )

                for variant in ("A", "B"):
                    params_fs = fit_domain_params_fewshot(
                        X_m=X_m,
                        t_m=t_m,
                        delta_m=delta_m,
                        A=A, w=w,
                        tau=tau, nu=nu,
                        domain_id=hid,
                        n_synthetic=n_full,
                        source_lambda0_mean=source_lambda0_mean,
                        variant=variant,
                    )
                    km_used = getattr(params_fs, "_used_km", variant == "A")
                    synth_fs = gen.generate_domain(
                        params_fs, domain_id=hid,
                        seed=seed + hid * 10000 + s * 100 + m,
                    )
                    scores = domain_score(synth_fs, real_ds, src_ds_list, tau=tau)
                    rows.append({
                        "center":   name,
                        "m":        m,
                        "seed":     s,
                        "variant":  variant,
                        "km_used":  km_used,
                        **_clean_scores(scores),
                    })
                    if variant == "A":
                        composites_A.append(scores["composite"])
                    else:
                        composites_B.append(scores["composite"])

            if verbose:
                mA = float(np.mean(composites_A))
                mB = float(np.mean(composites_B))
                print(f"  m={m:4d}  A composite={mA:.4f}  B composite={mB:.4f}")

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "fewshot_results.csv", index=False)

    # Aggregated summary: mean ± std across seeds (Variants A and B only)
    summary = (
        df[df["variant"].isin(["A", "B"])]
        .groupby(["center", "m", "variant"])
        .agg(
            composite_mean=("composite",    "mean"),
            composite_std=("composite",     "std"),
            km_dist_mean=("km_distance",    "mean"),
            km_dist_std=("km_distance",     "std"),
            mmd_mean=("mmd",                "mean"),
            auroc_delta_mean=("auroc_gap_delta", "mean"),
            km_used_frac=("km_used",        "mean"),
        )
        .reset_index()
    )
    # Append Variant C rows (single seed, no std)
    c_rows = df[df["variant"] == "C"].copy()
    c_rows = c_rows.rename(columns={"composite": "composite_mean", "km_distance": "km_dist_mean",
                                     "mmd": "mmd_mean", "auroc_gap_delta": "auroc_delta_mean"})
    c_rows["composite_std"] = 0.0
    c_rows["km_dist_std"]   = 0.0
    c_rows["km_used_frac"]  = 1.0

    summary.to_csv(out_path / "fewshot_summary.csv", index=False)
    df.to_csv(out_path / "fewshot_results.csv", index=False)

    if verbose:
        print(f"\nSaved fewshot_results.csv and fewshot_summary.csv to {out_path}/")
        print("\nComposite score by center and variant (mean across seeds):")
        pivot = (
            df[df["variant"].isin(["A", "B"])]
            .groupby(["center", "variant", "m"])["composite"]
            .mean()
            .unstack("m")
            .round(4)
        )
        print(pivot.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Few-shot held-out calibration experiment")
    p.add_argument("--m_grid",    type=int, nargs="+", default=[5, 10, 20, 30, 50, 100])
    p.add_argument("--n_seeds",   type=int, default=10)
    p.add_argument("--k_total",   type=int, default=6)
    p.add_argument("--tau",       type=float, default=12.0)
    p.add_argument("--nu",        type=float, default=1.5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default="outputs/tables/")
    p.add_argument("--no_verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_fewshot_experiment(
        m_grid=args.m_grid,
        n_seeds=args.n_seeds,
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
