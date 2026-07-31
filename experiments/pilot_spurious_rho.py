"""Pre-specified pilot: choose rho for the spurious-shift regime.

Rule, fixed before running and not revisited:

    choose the smallest rho for which the source-only spurious oracle
    reaches AUROC >= 0.70

Rationale. With the regime made pure -- only the invariant/spurious
cross-covariance changes, mu_spu no longer moves -- the shortcut it creates was
too weak to affect any method: ERM landed within 0.001 of its no-shift value.
A control that cannot move the thing it is meant to control is not evidence of
anything. But rho must not be tuned by looking at the outcome it is supposed to
test, so the criterion is stated on the source side only.

What keeps this honest:
  * only source data is used -- the spurious oracle's *source* AUROC, never its
    target AUROC;
  * no compared method is fitted, so no method's result can influence rho;
  * pilot seeds are disjoint from the benchmark's seeds 0-9;
  * rho is chosen once and then fixed in configs/generator.yaml.

If no rho in the grid reaches the threshold, the honest conclusion is that the
cross-covariance mechanism alone is a weak control: say so and move the regime
to the appendix rather than raising the threshold or widening the grid.

Run from repo root:
    python -m experiments.pilot_spurious_rho
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from experiments.baselines import fit_spurious_oracle, fit_invariant_oracle

THRESHOLD = 0.70
RHO_GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
PILOT_SEEDS = [1000, 1001, 1002, 1003, 1004]   # disjoint from the benchmark's 0-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_domain", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    print(f"Pilot: rho piu' piccolo con AUROC(oracolo spurio, SORGENTE) >= {args.threshold}")
    print(f"Seed pilota {PILOT_SEEDS} (disgiunti dai seed 0-9 del benchmark)\n")
    print(f"{'rho':>6}{'AUROC spurio (src)':>22}{'AUROC invariante (src)':>26}")
    print("-" * 56)

    chosen = None
    for rho in RHO_GRID:
        spu, inv = [], []
        for s in PILOT_SEEDS:
            d = generate_scenario("spurious_shift", n_source_domains=1,
                                  n_per_domain=args.n_per_domain, seed=s,
                                  base=BaseParams(seed=s), rho_source=rho)
            src = d["source"]
            # source_pooled is the held-out source split: never the target.
            spu.append(fit_spurious_oracle(src, d["target"])["source_pooled"]["auroc"])
            inv.append(fit_invariant_oracle(src, d["target"])["source_pooled"]["auroc"])
        m_spu, m_inv = float(np.mean(spu)), float(np.mean(inv))
        flag = ""
        if chosen is None and m_spu >= args.threshold:
            chosen, flag = rho, "   <-- scelto"
        print(f"{rho:>6.2f}{m_spu:>22.4f}{m_inv:>26.4f}{flag}")

    print()
    if chosen is None:
        print(f"NESSUN rho raggiunge {args.threshold}.")
        print("Il meccanismo della sola cross-covarianza e' un controllo debole:")
        print("dichiararlo come tale e spostare il regime in appendice, senza")
        print("alzare la soglia ne' allargare la griglia dopo aver visto questa tabella.")
    else:
        print(f"rho scelto = {chosen}")
        print("Fissarlo in configs/generator.yaml (rho_source) e non modificarlo piu'.")


if __name__ == "__main__":
    main()
