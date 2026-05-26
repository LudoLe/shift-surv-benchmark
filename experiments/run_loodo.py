"""
Leave-One-Domain-Out (LOODO) evaluation of the fitted synthetic generator.

For each center d in {0, ..., D-1}:
  1. Treat d as the held-out target; the remaining D-1 centers are source.
  2. Fit the generator on source (loading matrix A, risk weights w, per-domain params).
  3. Generate a synthetic dataset with the same DGP fitted on source but shifted
     toward d using the d-specific params.
  4. Compare the synthetic held-out domain against the real held-out domain
     using MMD, KM distance, and AUROC-gap delta.

Usage
-----
  python -m experiments.run_loodo [options]

  -k / --k_total    : number of latent dimensions (default 6)
  --tau             : prediction horizon in months (default 12.0)
  --nu              : Weibull shape parameter (default 1.5)
  --seed            : global RNG seed (default 42)
  --n_synthetic     : synthetic sample size per domain (default = real n)
  --out             : output CSV path (default outputs/tables/loodo_results.csv)
  --verbose         : print per-fold summary (default True)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data import registry
from data.loader import load_all_centers
from data.preprocess import standardize, make_binary_label, impute_median

from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams, DomainParams
from generator.fitting import (
    fit_reference_generator,
    fit_domain_params,
    make_real_domain_dataset,
)
from experiments.compare_domains import domain_score


# ---------------------------------------------------------------------------
# Core LOODO loop
# ---------------------------------------------------------------------------

def run_loodo(
    k_total: int = 6,
    tau: float = 12.0,
    nu: float = 1.5,
    seed: int = 42,
    n_synthetic: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run LOODO across all centers defined in data.registry.

    Parameters
    ----------
    k_total     : number of latent dimensions
    tau         : fixed prediction horizon (months)
    nu          : Weibull shape parameter (fixed across all domains)
    seed        : global RNG seed for synthetic generation
    n_synthetic : synthetic sample size per held-out fold; None = match real n
    verbose     : print per-fold summary to stdout

    Returns
    -------
    results : list of dicts, one per fold, containing center_id and all
              domain_score metrics
    """
    # Load all centers
    centers, feature_names = load_all_centers(registry_module=registry)
    p = centers[0]["X"].shape[1]
    D = len(centers)

    if verbose:
        print(f"Loaded {D} centers, p={p} features, τ={tau} months")

    results = []

    for held_out_idx in range(D):
        held_out_center = centers[held_out_idx]
        source_centers  = [c for i, c in enumerate(centers) if i != held_out_idx]
        hid = held_out_center["center_id"]

        if verbose:
            print(f"\n--- Fold: hold out center {hid} ---")

        # Median imputation: fit on pooled source, apply to held-out
        # (avoids leakage from held-out center into imputed values)
        X_src_pool = np.vstack([c["X"] for c in source_centers])
        _, src_medians = impute_median(X_src_pool)

        source_centers_imp = []
        for c in source_centers:
            X_imp = np.where(np.isnan(c["X"]), src_medians, c["X"])
            source_centers_imp.append({**c, "X": X_imp})

        X_held_imp = np.where(np.isnan(held_out_center["X"]), src_medians, held_out_center["X"])

        # Fit generator on source centers
        A, w, source_params, scaler = fit_reference_generator(
            source_centers=source_centers_imp,
            k_total=k_total,
            tau=tau,
            nu=nu,
        )

        # Standardise held-out center using source scaler
        X_held_s = scaler.transform(X_held_imp)

        # Fit DomainParams for the held-out center
        n_syn = n_synthetic if n_synthetic is not None else held_out_center["n"]
        held_out_params = fit_domain_params(
            X_d=X_held_s,
            t_d=held_out_center["t"],
            delta_d=held_out_center["delta"],
            A=A,
            w=w,
            tau=tau,
            nu=nu,
            domain_id=hid,
            n_synthetic=n_syn,
        )

        # Build BaseParams (k_spu = 0 for MVP)
        base = BaseParams(
            k_inv=k_total,
            k_spu=0,
            p=p,
            tau=tau,
            seed=seed,
        )

        # Generate synthetic held-out domain
        gen = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)
        synth_ds = gen.generate_domain(held_out_params, domain_id=hid)

        # Wrap real held-out data as DomainDataset for comparison
        real_ds = make_real_domain_dataset(
            X_s=X_held_s,
            t_obs=held_out_center["t"],
            delta=held_out_center["delta"],
            A=A,
            w=w,
            tau=tau,
            params=held_out_params,
            domain_id=hid,
        )

        # Wrap source data as DomainDatasets for AUROC-gap computation
        source_ds_list = []
        for sp, sc in zip(source_params, source_centers_imp):
            X_sc_s = scaler.transform(sc["X"])
            src_real = make_real_domain_dataset(
                X_s=X_sc_s,
                t_obs=sc["t"],
                delta=sc["delta"],
                A=A,
                w=w,
                tau=tau,
                params=sp,
                domain_id=sc["center_id"],
            )
            source_ds_list.append(src_real)

        # Compute comparison metrics
        scores = domain_score(synth_ds, real_ds, source_ds_list, tau=tau)

        result = {
            "held_out_center": hid,
            "center_name":     registry.CENTER_NAMES.get(hid, str(hid)),
            "n_real":          real_ds.n,
            "n_synth":         synth_ds.n,
            **{k: v for k, v in scores.items() if k not in ("weights", "synth_n", "real_n")},
        }
        results.append(result)

        if verbose:
            print(
                f"  center={result['center_name']}  n_real={result['n_real']}  n_synth={result['n_synth']}\n"
                f"  km_prev_real={result.get('km_prev_real','?')}  "
                f"km_prev_synth={result.get('km_prev_synth','?')}\n"
                f"  mmd={result['mmd']:.4f}  km={result['km_distance']:.4f}  "
                f"auroc_Δ={result['auroc_gap_delta']:.4f}  "
                f"composite={result['composite']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOODO evaluation of fitted synthetic generator")
    p.add_argument("--k_total",     type=int,   default=6,     help="number of latent dimensions")
    p.add_argument("--tau",         type=float, default=12.0,  help="prediction horizon (months)")
    p.add_argument("--nu",          type=float, default=1.5,   help="Weibull shape parameter")
    p.add_argument("--seed",        type=int,   default=42,    help="global RNG seed")
    p.add_argument("--n_synthetic", type=int,   default=None,  help="synthetic sample size per fold")
    p.add_argument("--out",         type=str,
                   default="outputs/tables/loodo_results.csv",
                   help="output CSV path")
    p.add_argument("--no_verbose",  action="store_true", help="suppress per-fold output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    results = run_loodo(
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        n_synthetic=args.n_synthetic,
        verbose=not args.no_verbose,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))
