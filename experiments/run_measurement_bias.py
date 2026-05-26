"""
Measurement bias experiment — the case where DANN and CORAL should help.

Design
------
All domains share the same causal mechanism: identical w, λ₀, u_inv distribution.
After generating X_true, we add a domain-specific additive bias:

    X_obs_d = X_true_d + scale * bias_d,   bias_d ~ N(0, I_p)

Y / T are generated from u_inv (not from X_obs), so P(Y | u_inv) is identical
across domains. Only P(X_obs) differs — a pure measurement/acquisition bias.

This is exactly the regime DA methods are designed for:
  - LR ERM   : learns source X_obs → Y mapping, fails on target (wrong offset)
  - CORAL     : aligns covariance of X_obs → partial fix (mean shift not corrected)
  - DANN      : aligns full P(X_obs) → should fully recover

Grid on bias_scale: [0.0, 0.5, 1.0, 2.0, 3.0]
N_SEEDS = 10, n_per_domain = 500
"""

from __future__ import annotations
import copy
import sys
import warnings
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from generator.scenarios import generate_scenario
from experiments.baselines import fit_invariant_oracle, fit_all_observed, fit_all_observed_mlp
from experiments.adaptation_methods import fit_coral, fit_dann, fit_irm


BIAS_SCALES  = [0.0, 0.5, 1.0, 2.0, 3.0]
N_SEEDS      = 10
N_PER_DOMAIN = 500
SEEDS        = list(range(42, 42 + N_SEEDS))
OUT_DIR      = _ROOT / "outputs" / "tables"


def _add_bias(dataset, bias_vector, scale):
    """Return a copy of dataset with X_obs = X_true + scale * bias_vector."""
    if scale == 0.0:
        return dataset
    new_x = dataset.x + scale * bias_vector[None, :]   # broadcast over n
    return replace(dataset, x=new_x)


def run_measurement_bias(
    bias_scales=None,
    n_seeds=N_SEEDS,
    n_per_domain=N_PER_DOMAIN,
    verbose=True,
    out_dir=OUT_DIR,
):
    if bias_scales is None:
        bias_scales = BIAS_SCALES

    seeds = list(range(42, 42 + n_seeds))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []

    if verbose:
        print("=== Measurement Bias Experiment ===\n")
        print(f"  bias_scales = {bias_scales}")
        print(f"  n_seeds = {n_seeds},  n_per_domain = {n_per_domain}\n")
        hdr = f"{'scale':>6}  {'inv_oracle':>10}  {'LR ERM':>10}  {'CORAL':>10}  {'IRM':>10}  {'MLP ERM':>10}  {'DANN':>10}"
        print(hdr)
        print("-" * len(hdr))

    for scale in bias_scales:
        res = {m: [] for m in ["inv_oracle", "lr_erm", "coral", "irm", "mlp_erm", "dann"]}

        for seed in seeds:
            # Generate no-shift scenario (same causal mechanism everywhere)
            data = generate_scenario(
                "no_shift", n_per_domain=n_per_domain, seed=seed
            )
            source_true = data["source"]   # list of DomainDataset
            target_true = data["target"]   # DomainDataset

            p = source_true[0].x.shape[1]
            rng = np.random.default_rng(seed + 9999)

            # Fixed target bias direction (same across seeds for reproducibility)
            # Source: no bias.  Target: systematic offset in a fixed direction.
            # This asymmetry is what forces CORAL/DANN to do real work.
            tgt_bias = rng.standard_normal(p)
            tgt_bias /= np.linalg.norm(tgt_bias)   # unit vector, scale controls magnitude

            # Source: no bias (zero offset)
            source_obs = source_true
            # Target: systematic offset
            target_obs = _add_bias(target_true, tgt_bias, scale)

            # Invariant oracle — uses u_inv directly, ignores X_obs bias
            r = fit_invariant_oracle(source_obs, target_obs)
            res["inv_oracle"].append(r["target"]["auroc"])

            # LR ERM — trains on biased X
            r = fit_all_observed(source_obs, target_obs)
            res["lr_erm"].append(r["target"]["auroc"])

            # CORAL
            r = fit_coral(source_obs, target_obs, seed=seed)
            res["coral"].append(r["target"]["auroc"])

            # IRM
            r = fit_irm(source_obs, target_obs, seed=seed)
            res["irm"].append(r["target"]["auroc"])

            # MLP ERM
            r = fit_all_observed_mlp(source_obs, target_obs)
            res["mlp_erm"].append(r["target"]["auroc"])

            # DANN
            r = fit_dann(source_obs, target_obs, seed=seed)
            res["dann"].append(r["target"]["auroc"])

        means = {m: float(np.nanmean(v)) for m, v in res.items()}
        stds  = {m: float(np.nanstd(v))  for m, v in res.items()}

        for m in res:
            for s_idx, v in enumerate(res[m]):
                rows.append({"bias_scale": scale, "method": m,
                             "seed": seeds[s_idx],
                             "tgt_auroc": v})

        if verbose:
            print(
                f"  {scale:>5.1f}  "
                f"{means['inv_oracle']:>10.4f}  "
                f"{means['lr_erm']:>10.4f}  "
                f"{means['coral']:>10.4f}  "
                f"{means['irm']:>10.4f}  "
                f"{means['mlp_erm']:>10.4f}  "
                f"{means['dann']:>10.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path / "measurement_bias_results.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/measurement_bias_results.csv")
        print("\nGain vs LR ERM (AUROC delta):\n")
        pivot = df.groupby(["bias_scale", "method"])["tgt_auroc"].mean().unstack("method")
        pivot["CORAL - LR"]  = pivot["coral"]   - pivot["lr_erm"]
        pivot["DANN  - LR"]  = pivot["dann"]    - pivot["lr_erm"]
        pivot["DANN  - CORAL"] = pivot["dann"]  - pivot["coral"]
        print(pivot[["lr_erm","coral","dann","CORAL - LR","DANN  - LR","DANN  - CORAL"]].round(4).to_string())

    return df


if __name__ == "__main__":
    run_measurement_bias(verbose=True)
