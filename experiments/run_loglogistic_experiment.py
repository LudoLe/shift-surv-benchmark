"""
Log-logistic vs Weibull PH: discriminative signal comparison.

Question
--------
Does modelling a non-proportional, non-monotone hazard (log-logistic) produce
a different discriminative signal than the Weibull PH baseline?

The log-logistic hazard h(t|η) rises to a peak at t* = ((ν-1)/λ₀exp(η))^(1/ν)
then decays to 0 — mimicking delayed treatment effects or post-surgical risk
patterns common in IO.  Crucially, the hazard ratio between two patients is
NOT constant (non-PH):

    HR(t) = [1+(λ₀exp(η₂)t)^ν] / [1+(λ₀exp(η₁)t)^ν]  →  1  as t→∞

Signal comparison
-----------------
For Weibull PH:
    P(y=1|η) = 1 − exp(−λ₀ τ^ν exp(η))   [complementary log-log link]

For log-logistic:
    P(y=1|η) = 1/(1+(λ₀ exp(η) τ)^{−ν}) = σ(ν·(log(λ₀τ) + η))  [logistic link!]

The log-logistic fixed-horizon label is EXACTLY a logistic function of η —
so logistic regression on x should fit it better than under Weibull, where
the true link is cloglog.  We expect LR AUROC ≥ Weibull on log-logistic data.

Grid: shape ν ∈ [0.5, 1.0, 1.5, 2.0, 3.0]
      distribution ∈ {weibull, loglogistic}
N_SEEDS = 10, n_per_domain = 500, τ = 6.0

Usage
-----
  python -m experiments.run_loglogistic_experiment
"""

from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from generator.scenarios import generate_scenario
from generator.parameters import BaseParams
from experiments.baselines import fit_invariant_oracle, fit_all_observed, fit_all_observed_mlp
from experiments.adaptation_methods import fit_irm


NU_GRID   = [0.5, 1.0, 1.5, 2.0, 3.0]
DISTS     = ["weibull", "loglogistic"]
N_SEEDS   = 10
N_PER_DOM = 500
TAU       = 6.0
SEEDS     = list(range(42, 42 + N_SEEDS))
OUT_DIR   = _ROOT / "outputs" / "tables"


def _inject_dist(data: dict, dist: str, nu: float) -> dict:
    """
    Patch generated DomainDatasets to use a specific distribution and shape.
    Re-simulates T and y from the already-drawn latent factors (u_inv, eta),
    so the covariate structure is identical across distributions.
    """
    from scipy.special import expit
    rng = np.random.default_rng(0)

    def _resim(ds, dist, nu, tau):
        n = len(ds.eta)
        # Re-draw U (independent of covariates — randomness in T only)
        U = rng.uniform(0, 1, size=n)
        lam = ds.params.lambda0
        eta = ds.eta

        if dist == "loglogistic":
            U_c = np.clip(U, 1e-9, 1 - 1e-9)
            T_new = (U_c / (1 - U_c)) ** (1.0 / nu) / (lam * np.exp(eta))
        else:
            T_new = (-np.log(np.clip(U, 1e-9, None)) / (lam * np.exp(eta))) ** (1.0 / nu)

        y_new = (T_new <= tau).astype(int)
        return replace(ds, T=T_new, t_obs=T_new, y=y_new)

    new_source = [_resim(ds, dist, nu, TAU) for ds in data["source"]]
    new_target = _resim(data["target"], dist, nu, TAU)
    return {**data, "source": new_source, "target": new_target}


def run_loglogistic_experiment(
    nu_grid=None,
    dists=None,
    n_seeds=N_SEEDS,
    n_per_domain=N_PER_DOM,
    verbose=True,
    out_dir=OUT_DIR,
):
    if nu_grid is None:
        nu_grid = NU_GRID
    if dists is None:
        dists = DISTS

    seeds = list(range(42, 42 + n_seeds))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []

    if verbose:
        print("=== Log-logistic vs Weibull: discriminative signal ===\n")
        print(f"  nu_grid = {nu_grid}")
        print(f"  n_seeds = {n_seeds},  n_per_domain = {n_per_domain},  τ = {TAU}\n")
        hdr = f"{'dist':>12}  {'ν':>5}  {'inv_oracle':>10}  {'LR ERM':>10}  {'MLP ERM':>10}  {'IRM':>10}  {'prev':>6}"
        print(hdr)
        print("-" * len(hdr))

    for dist in dists:
        for nu in nu_grid:
            res = {m: [] for m in ["inv_oracle", "lr_erm", "mlp_erm", "irm", "prev"]}

            for seed in seeds:
                # Generate no_shift scenario (same covariate structure for both distributions)
                data = generate_scenario("no_shift", n_per_domain=n_per_domain, seed=seed)
                # Re-simulate T and y under the chosen distribution and shape
                data = _inject_dist(data, dist, nu)

                source, target = data["source"], data["target"]

                res["prev"].append(float(target.y.mean()))
                res["inv_oracle"].append(
                    fit_invariant_oracle(source, target)["target"]["auroc"])
                res["lr_erm"].append(
                    fit_all_observed(source, target)["target"]["auroc"])
                res["mlp_erm"].append(
                    fit_all_observed_mlp(source, target)["target"]["auroc"])
                res["irm"].append(
                    fit_irm(source, target, seed=seed)["target"]["auroc"])

            means = {m: float(np.nanmean(v)) for m, v in res.items()}

            for m in [k for k in res if k != "prev"]:
                for s_idx, v in enumerate(res[m]):
                    rows.append({
                        "dist": dist, "nu": nu,
                        "method": m, "seed": seeds[s_idx],
                        "tgt_auroc": v,
                        "prevalence": res["prev"][s_idx],
                    })

            if verbose:
                print(
                    f"  {dist:>12}  {nu:>5.1f}  "
                    f"{means['inv_oracle']:>10.4f}  "
                    f"{means['lr_erm']:>10.4f}  "
                    f"{means['mlp_erm']:>10.4f}  "
                    f"{means['irm']:>10.4f}  "
                    f"{means['prev']:>6.3f}"
                )
        if verbose:
            print()

    df = pd.DataFrame(rows)
    df.to_csv(out_path / "loglogistic_results.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/loglogistic_results.csv")
        print("\nLR ERM AUROC — Weibull vs Log-logistic (same ν):\n")
        pivot = (df[df["method"] == "lr_erm"]
                 .groupby(["dist", "nu"])["tgt_auroc"].mean()
                 .unstack("dist")
                 .round(4))
        pivot["loglog−weib"] = pivot.get("loglogistic", np.nan) - pivot.get("weibull", np.nan)
        print(pivot.to_string())

    return df


if __name__ == "__main__":
    run_loglogistic_experiment(verbose=True)
