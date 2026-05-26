"""
Nonlinear-observation diagnostic: compare inv_oracle / LR / MLP AUROC
before (nl_strength=0.0) and after (nl_strength=0.3) the quadratic
observation term is switched on.

Runs only the three lightweight baselines (no CORAL / DANN) for speed.

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_nl_diagnostic
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from generator.parameters import BaseParams
from generator.scenarios import generate_scenario, SCENARIO_NAMES
from experiments.baselines import fit_invariant_oracle, fit_all_observed, fit_all_observed_mlp

_SCENARIOS = [s for s in SCENARIO_NAMES if s != "strong_covariate_shift"]
_SEEDS = list(range(42, 52))          # 10 seeds for stable estimates
_N_PER_DOMAIN = 500

_METHODS = ["invariant_oracle", "all_observed", "all_observed_mlp"]
_LABELS  = {"invariant_oracle": "inv_oracle", "all_observed": "lr_erm", "all_observed_mlp": "mlp_erm"}


def _run(nl_strength: float) -> dict:
    """
    Run all scenarios and seeds for a given nl_strength.
    Returns {(scenario, method): [tgt_auroc per seed]}.
    """
    results: dict[tuple, list] = {(sc, m): [] for sc in _SCENARIOS for m in _METHODS}

    for seed in _SEEDS:
        base = BaseParams(seed=seed, nl_strength=nl_strength)
        for sc in _SCENARIOS:
            data = generate_scenario(sc, n_source_domains=3,
                                     n_per_domain=_N_PER_DOMAIN, seed=seed, base=base)
            src, tgt = data["source"], data["target"]
            for fn, key in [
                (fit_invariant_oracle, "invariant_oracle"),
                (fit_all_observed,     "all_observed"),
                (fit_all_observed_mlp, "all_observed_mlp"),
            ]:
                r = fn(src, tgt)
                results[(sc, key)].append(r["target"]["auroc"])

    return results


def _summarise(results: dict) -> dict:
    out = {}
    for (sc, m), vals in results.items():
        v = np.array(vals)
        out[(sc, m)] = (round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4))
    return out


def main():
    print("Running nl_strength=0.0 (linear)  …", flush=True)
    before = _summarise(_run(0.0))
    print("Running nl_strength=0.3 (nonlinear)…", flush=True)
    after  = _summarise(_run(0.3))

    # -----------------------------------------------------------------------
    sep = "=" * 100
    print(f"\n{sep}")
    print("  NONLINEAR OBSERVATION DIAGNOSTIC  —  target AUROC (mean ± std, 10 seeds)")
    print(f"  Observation model: x = A_inv u_inv + A_spu u_spu + α·B_nl·(u_inv²) + ε")
    print(sep)

    col = 16
    methods = _METHODS
    labels  = [_LABELS[m] for m in methods]

    for sc in _SCENARIOS:
        print(f"\n  Scenario: {sc}")
        header = f"  {'':12s}" + "".join(f"  {'BEFORE':>{col}s}" + f"  {'AFTER':>{col}s}" for _ in labels)
        label_row = f"  {'method':12s}" + "".join(f"  {lbl:>{col}s}  {'':>{col}s}" for lbl in labels)
        print(f"  {'':12s}" + "".join(
            f"  {'α=0.0 (lin)':>{col}s}  {'α=0.3 (nl)':>{col}s}"
            for _ in labels
        ))
        print(f"  {'method':12s}" + "".join(f"  {lbl:>{col*2+5}s}" for lbl in labels))
        print("  " + "-" * (14 + (col * 2 + 7) * len(methods)))
        for m in methods:
            lbl = _LABELS[m]
            b_mean, b_std = before[(sc, m)]
            a_mean, a_std = after[(sc, m)]
            delta = round(a_mean - b_mean, 4)
            delta_str = f"Δ={delta:+.3f}"
            line = f"  {lbl:12s}"
            line += f"  {b_mean:.3f}±{b_std:.3f}  {a_mean:.3f}±{a_std:.3f}  {delta_str}"
            print(line)

    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("  SUMMARY TABLE  (Δ = after − before, positive = improvement)")
    print(sep)
    print(f"  {'scenario':28s}" + "".join(f"  {lbl:>10s}" for lbl in labels))
    print("  " + "-" * (30 + 13 * len(methods)))
    for sc in _SCENARIOS:
        line = f"  {sc:28s}"
        for m in methods:
            b_mean, _ = before[(sc, m)]
            a_mean, _ = after[(sc, m)]
            delta = round(a_mean - b_mean, 4)
            marker = " ←" if abs(delta) > 0.01 else ""
            line += f"  {delta:+.3f}{marker:3s}"
        print(line)

    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("  LR vs MLP gap  (positive = LR wins; smaller = MLP more competitive)")
    print(sep)
    print(f"  {'scenario':28s}  {'before':>10s}  {'after':>10s}  {'Δ gap':>10s}")
    print("  " + "-" * 65)
    for sc in _SCENARIOS:
        lr_b, _ = before[(sc, "all_observed")]
        mlp_b, _ = before[(sc, "all_observed_mlp")]
        lr_a, _ = after[(sc, "all_observed")]
        mlp_a, _ = after[(sc, "all_observed_mlp")]
        gap_b = round(lr_b - mlp_b, 4)
        gap_a = round(lr_a - mlp_a, 4)
        d_gap = round(gap_a - gap_b, 4)
        marker = " ←" if d_gap < -0.01 else ""
        print(f"  {sc:28s}  {gap_b:>10.3f}  {gap_a:>10.3f}  {d_gap:>+10.3f}{marker}")

    print()


if __name__ == "__main__":
    main()
