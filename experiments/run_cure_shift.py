"""
Mixture cure model shift experiment.

Design
------
Source domains: standard Weibull PH (no cure fraction).
Target domain:  mixture cure model — fraction π of patients never event.

    S_tgt(t | X) = π_i + (1 − π_i) · exp(−λ_0 t^ν exp(η_i))

where logit(π_i) = logit(π̄) − 0.5 · w^T u_inv  (high-risk patients less curable).

This is a non-proportional hazard: survival curves plateau on the target
but converge to 0 on the source.  The hazard ratio source/target starts at
its Weibull PH value and decays toward 1 as t → ∞.

Grid: cure_fraction ∈ [0.0, 0.1, 0.2, 0.3, 0.4]
N_SEEDS = 10, n_per_domain = 500

Expected behaviour
------------------
  inv_oracle : AUROC stable (uses u_inv directly, unaffected by cure)
  LR ERM     : degrades with cure_fraction — fixed-horizon label y underestimates
               target prevalence as cured patients always contribute y=0
  CORAL/IRM  : small improvement (no covariate shift to align)
  DANN       : similar to ERM — cure is a label-mechanism shift, not a P(x) shift

Usage
-----
  python -m experiments.run_cure_shift
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from generator.scenarios import generate_scenario
from experiments.baselines import fit_invariant_oracle, fit_all_observed, fit_all_observed_mlp
from experiments.adaptation_methods import fit_coral, fit_dann, fit_irm


CURE_FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40]
N_SEEDS        = 10
N_PER_DOMAIN   = 500
SEEDS          = list(range(42, 42 + N_SEEDS))
OUT_DIR        = _ROOT / "outputs" / "tables"


def run_cure_shift(
    cure_fractions=None,
    n_seeds=N_SEEDS,
    n_per_domain=N_PER_DOMAIN,
    verbose=True,
    out_dir=OUT_DIR,
):
    if cure_fractions is None:
        cure_fractions = CURE_FRACTIONS

    seeds = list(range(42, 42 + n_seeds))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []

    if verbose:
        print("=== Mixture Cure Shift Experiment ===\n")
        print(f"  cure_fractions = {cure_fractions}")
        print(f"  n_seeds = {n_seeds},  n_per_domain = {n_per_domain}\n")
        hdr = (f"{'π̄':>5}  {'inv_oracle':>10}  {'LR ERM':>10}  "
               f"{'CORAL':>10}  {'IRM':>10}  {'MLP ERM':>10}  {'DANN':>10}")
        print(hdr)
        print("-" * len(hdr))

    for cure_frac in cure_fractions:
        res = {m: [] for m in ["inv_oracle", "lr_erm", "coral", "irm", "mlp_erm", "dann"]}

        for seed in seeds:
            data = generate_scenario(
                "cure_shift",
                n_per_domain=n_per_domain,
                seed=seed,
                cure_fraction_target=cure_frac,
                cure_fraction_source=0.0,
            )
            source = data["source"]
            target = data["target"]

            res["inv_oracle"].append(fit_invariant_oracle(source, target)["target"]["auroc"])
            res["lr_erm"].append(fit_all_observed(source, target)["target"]["auroc"])
            res["coral"].append(fit_coral(source, target, seed=seed)["target"]["auroc"])
            res["irm"].append(fit_irm(source, target, seed=seed)["target"]["auroc"])
            res["mlp_erm"].append(fit_all_observed_mlp(source, target)["target"]["auroc"])
            res["dann"].append(fit_dann(source, target, seed=seed)["target"]["auroc"])

        means = {m: float(np.nanmean(v)) for m, v in res.items()}

        for m in res:
            for s_idx, v in enumerate(res[m]):
                rows.append({
                    "cure_fraction": cure_frac,
                    "method": m,
                    "seed": seeds[s_idx],
                    "tgt_auroc": v,
                })

        if verbose:
            print(
                f"  {cure_frac:>4.2f}  "
                f"{means['inv_oracle']:>10.4f}  "
                f"{means['lr_erm']:>10.4f}  "
                f"{means['coral']:>10.4f}  "
                f"{means['irm']:>10.4f}  "
                f"{means['mlp_erm']:>10.4f}  "
                f"{means['dann']:>10.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path / "cure_shift_results.csv", index=False)

    if verbose:
        print(f"\nSaved → {out_path}/cure_shift_results.csv")
        print("\nGain vs LR ERM (AUROC delta):\n")
        pivot = df.groupby(["cure_fraction", "method"])["tgt_auroc"].mean().unstack("method")
        pivot["CORAL−LR"] = pivot["coral"]   - pivot["lr_erm"]
        pivot["IRM−LR"]   = pivot["irm"]     - pivot["lr_erm"]
        pivot["DANN−LR"]  = pivot["dann"]    - pivot["lr_erm"]
        print(pivot[["lr_erm", "coral", "irm", "dann",
                      "CORAL−LR", "IRM−LR", "DANN−LR"]].round(4).to_string())

    return df


if __name__ == "__main__":
    run_cure_shift(verbose=True)
