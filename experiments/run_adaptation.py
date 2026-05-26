"""
Phase-2 experiment runner: domain adaptation methods on all shift scenarios.

Methods compared
----------------
  inv_oracle   — LR oracle on true invariant latent factors u_inv (ceiling)
  all_observed — Pooled ERM LR on observed covariates x (source-only baseline)
  coral        — CORAL with LR estimator
  dann         — Domain-Adversarial Neural Network (TF/Keras)
  cdan         — Conditional DANN (TF/Keras); skipped with --no_cdan

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV

  # Quick smoke-test (1 seed, no CDAN):
  python -m experiments.run_adaptation --single_seed --no_cdan

  # Full 10-seed run:
  python -m experiments.run_adaptation

  # Single scenario, 3 seeds:
  python -m experiments.run_adaptation --scenario covariate_shift --n_seeds 3

  # Skip CDAN:
  python -m experiments.run_adaptation --no_cdan

Outputs
-------
  outputs/tables/adaptation_raw.csv      — one row per (scenario, seed, method)
  outputs/tables/adaptation_summary.csv  — mean ± std ± 95-CI per (scenario, method)
  outputs/tables/adaptation_compact.csv  — pivot: scenarios × methods, target AUROC/ECE/gap
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
from scipy import stats as sp_stats

from generator.scenarios import generate_scenario, SCENARIO_NAMES
from experiments.baselines import fit_invariant_oracle, fit_all_observed, fit_all_observed_mlp
from experiments.adaptation_methods import fit_coral, fit_dann, fit_cdan, fit_irm
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"
DEFAULT_SEEDS = list(range(42, 52))

# Methods in display order — names must match feature_set keys returned by each fitter
_ALL_METHODS = ["invariant_oracle", "all_observed", "coral", "irm", "all_observed_mlp", "dann", "cdan"]

# Short display labels (for column headers)
_DISPLAY = {
    "invariant_oracle":  "inv_oracle",
    "all_observed":      "lr_erm",
    "coral":             "coral",        # linear DA — compare to lr_erm
    "irm":               "irm",          # invariant risk minimization
    "all_observed_mlp":  "mlp_erm",
    "dann":              "dann",         # neural DA — compare to mlp_erm
    "cdan":              "cdan",
}


# ---------------------------------------------------------------------------
# Per-(scenario, seed) runner
# ---------------------------------------------------------------------------

def _run_one(
    scenario: str,
    seed: int,
    n_per_domain: int,
    run_cdan: bool,
    nl_strength: float = 0.3,
) -> list[dict]:
    """Generate one scenario/seed, fit all methods, return flat rows."""
    from generator.parameters import BaseParams
    base = BaseParams(seed=seed, nl_strength=nl_strength)
    data = generate_scenario(scenario, n_source_domains=3,
                             n_per_domain=n_per_domain, seed=seed, base=base)
    src, tgt = data["source"], data["target"]

    results = [
        fit_invariant_oracle(src, tgt),
        fit_all_observed(src, tgt),          # LR ERM  — natural baseline for CORAL
        fit_coral(src, tgt, seed=seed),      # linear DA
        fit_irm(src, tgt, seed=seed),        # invariant risk minimization
        fit_all_observed_mlp(src, tgt),      # MLP ERM — natural baseline for DANN/CDAN
        fit_dann(src, tgt, seed=seed),       # neural DA
    ]
    if run_cdan:
        results.append(fit_cdan(src, tgt, seed=seed))

    return [_flatten(r, scenario, seed) for r in results]


def _flatten(r: dict, scenario: str, seed: int) -> dict:
    """Flatten one method-result dict to a single scalar row."""
    sp  = r.get("source_pooled", {})
    tgt = r.get("target", {})
    sa  = sp.get("auroc", float("nan"))
    ta  = tgt.get("auroc", float("nan"))
    gap = round(sa - ta, 4) if not (np.isnan(sa) or np.isnan(ta)) else float("nan")
    return {
        "scenario": scenario,
        "seed":     seed,
        "method":   r.get("feature_set", "?"),
        # source (held-out val)
        "src_auroc": sa,
        "src_auprc": sp.get("auprc", float("nan")),
        "src_ece":   sp.get("ece",   float("nan")),
        # target
        "tgt_auroc": ta,
        "tgt_auprc": tgt.get("auprc", float("nan")),
        "tgt_ece":   tgt.get("ece",   float("nan")),
        # gap (positive = degradation on target)
        "gap": gap,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["src_auroc", "src_auprc", "src_ece",
                "tgt_auroc", "tgt_auprc", "tgt_ece", "gap"]


def _aggregate(
    raw: list[dict],
    scenarios: list[str],
    methods: list[str],
) -> list[dict]:
    """Compute mean / std / 95-CI across seeds for each (scenario, method)."""
    agg = []
    for sc in scenarios:
        for m in methods:
            subset = [r for r in raw if r["scenario"] == sc and r["method"] == m]
            if not subset:
                continue
            row = {"scenario": sc, "method": m, "n_seeds": len(subset)}
            for key in _METRIC_KEYS:
                vals = np.array([r[key] for r in subset], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    row[f"{key}_mean"] = float("nan")
                    row[f"{key}_std"]  = float("nan")
                    row[f"{key}_ci95"] = float("nan")
                    continue
                mean = float(vals.mean())
                std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                se   = std / np.sqrt(len(vals))
                t_c  = sp_stats.t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
                row[f"{key}_mean"] = round(mean, 4)
                row[f"{key}_std"]  = round(std,  4)
                row[f"{key}_ci95"] = round(t_c * se, 4)
            agg.append(row)
    return agg


def _compact_table(
    agg: list[dict],
    scenarios: list[str],
    methods: list[str],
) -> list[dict]:
    """
    Pivot table: rows = scenarios, columns = methods × {tgt_auroc, gap, tgt_ece}.
    Cell values are formatted as 'mean±CI' strings for readability.
    """
    idx = {(r["scenario"], r["method"]): r for r in agg}

    def _fmt(r, key):
        mean = r.get(f"{key}_mean", float("nan"))
        ci   = r.get(f"{key}_ci95", float("nan"))
        if np.isnan(mean):
            return "—"
        sign = "+" if mean >= 0 and "gap" in key else ""
        return f"{sign}{mean:.3f}±{ci:.3f}"

    rows = []
    for sc in scenarios:
        row = {"scenario": sc}
        for m in methods:
            lbl = _DISPLAY.get(m, m)
            r   = idx.get((sc, m))
            if r is None:
                row[f"{lbl}_tgt_auroc"] = "—"
                row[f"{lbl}_gap"]       = "—"
                row[f"{lbl}_tgt_ece"]   = "—"
            else:
                row[f"{lbl}_tgt_auroc"] = _fmt(r, "tgt_auroc")
                row[f"{lbl}_gap"]       = _fmt(r, "gap")
                row[f"{lbl}_tgt_ece"]   = _fmt(r, "tgt_ece")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _print_table(
    agg: list[dict],
    scenarios: list[str],
    methods: list[str],
    metric: str,
    title: str,
    sign: bool = False,
) -> None:
    idx = {(r["scenario"], r["method"]): r for r in agg}
    col = 20
    labels = [_DISPLAY.get(m, m) for m in methods]
    print(f"\n  {title}")
    print(f"  {'scenario':28s}" + "".join(f"  {lbl:>{col}s}" for lbl in labels))
    print("  " + "-" * (30 + (col + 2) * len(methods)))
    for sc in scenarios:
        line = f"  {sc:28s}"
        for m in methods:
            r = idx.get((sc, m))
            if r is None:
                line += f"  {'—':>{col}s}"
                continue
            mean = r.get(f"{metric}_mean", float("nan"))
            ci   = r.get(f"{metric}_ci95", float("nan"))
            if np.isnan(mean):
                line += f"  {'nan':>{col}s}"
            else:
                s = f"{'+' if sign and mean >= 0 else ''}{mean:.3f} ± {ci:.3f}"
                line += f"  {s:>{col}s}"
        print(line)


def _print_full_report(
    agg: list[dict],
    scenarios: list[str],
    methods: list[str],
) -> None:
    sep = "=" * 110
    print(f"\n{sep}")
    print("  ADAPTATION METHODS — RESULTS ACROSS SCENARIOS  (mean ± 95% CI, 10 seeds)")
    print(sep)
    _print_table(agg, scenarios, methods, "tgt_auroc", "Target AUROC  ↑")
    _print_table(agg, scenarios, methods, "gap",       "AUROC gap  (source pooled − target; positive = degradation)  ↓", sign=True)
    _print_table(agg, scenarios, methods, "tgt_ece",   "Target ECE  ↓ (lower = better calibration)")
    _print_table(agg, scenarios, methods, "tgt_auprc", "Target AUPRC  ↑")


def _print_implementation_report(
    agg: list[dict],
    scenarios: list[str],
    methods: list[str],
) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  IMPLEMENTATION REPORT")
    print(sep)
    idx = {(r["scenario"], r["method"]): r for r in agg}
    # Natural comparisons: CORAL vs LR ERM, DANN/CDAN vs MLP ERM
    da_pairs = {
        "coral": "all_observed",
        "dann":  "all_observed_mlp",
        "cdan":  "all_observed_mlp",
    }
    da_methods = [m for m in methods if m in da_pairs]

    for m in da_methods:
        ref_key = da_pairs[m]
        ref_lbl = _DISPLAY.get(ref_key, ref_key)
        aurocs = []
        nan_count = 0
        for sc in scenarios:
            r = idx.get((sc, m))
            if r:
                v = r.get("tgt_auroc_mean", float("nan"))
                if np.isnan(v):
                    nan_count += 1
                else:
                    aurocs.append(v)

        stable = nan_count == 0 and len(aurocs) == len(scenarios)
        print(f"\n  {m.upper()}  (vs {ref_lbl}):")
        if stable:
            print(f"    Status : STABLE — ran on all {len(scenarios)} scenarios without errors.")
            print(f"    AUROC range (target): [{min(aurocs):.3f}, {max(aurocs):.3f}]")
            ref_aurocs = [idx[(sc, ref_key)].get("tgt_auroc_mean", float("nan"))
                          for sc in scenarios if (sc, ref_key) in idx]
            deltas = [a - b for a, b in zip(aurocs, ref_aurocs)
                      if not (np.isnan(a) or np.isnan(b))]
            if deltas:
                better = sum(1 for d in deltas if d > 0.005)
                worse  = sum(1 for d in deltas if d < -0.005)
                print(f"    vs {ref_lbl:12s}: better on {better}/{len(deltas)} scenarios, "
                      f"worse on {worse}/{len(deltas)}")
        else:
            print(f"    Status : UNSTABLE — {nan_count} scenario(s) returned NaN. "
                  f"Check warnings above.")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    print(f"    Linear track  (LR ERM → CORAL):")
    for sc in scenarios:
        lr  = idx.get((sc, "all_observed"),  {}).get("tgt_auroc_mean", float("nan"))
        co  = idx.get((sc, "coral"),         {}).get("tgt_auroc_mean", float("nan"))
        if not (np.isnan(lr) or np.isnan(co)):
            delta = co - lr
            marker = " ←" if abs(delta) > 0.005 else ""
            print(f"      {sc:30s}  LR={lr:.3f}  CORAL={co:.3f}  Δ={delta:+.3f}{marker}")
    print(f"    Neural track  (MLP ERM → DANN):")
    for sc in scenarios:
        mlp  = idx.get((sc, "all_observed_mlp"), {}).get("tgt_auroc_mean", float("nan"))
        dann = idx.get((sc, "dann"),             {}).get("tgt_auroc_mean", float("nan"))
        if not (np.isnan(mlp) or np.isnan(dann)):
            delta = dann - mlp
            marker = " ←" if abs(delta) > 0.005 else ""
            print(f"      {sc:30s}  MLP={mlp:.3f}  DANN={dann:.3f}  Δ={delta:+.3f}{marker}")
    print()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_adaptation(
    scenarios: list[str] = SCENARIO_NAMES,
    seeds: list[int] = DEFAULT_SEEDS,
    n_per_domain: int = 500,
    run_cdan: bool = True,
    verbose: bool = True,
    nl_strength: float = 0.3,
) -> dict:
    """
    Run all methods across scenarios and seeds; aggregate and save results.

    Returns
    -------
    dict with keys: "raw", "aggregated", "compact"
    """
    methods = ["invariant_oracle", "all_observed", "coral", "all_observed_mlp", "dann"]
    if run_cdan:
        methods.append("cdan")

    all_rows: list[dict] = []
    n_total = len(seeds) * len(scenarios)
    done = 0

    for seed in seeds:
        if verbose:
            print(f"\n[Seed {seed}]", flush=True)
        for sc in scenarios:
            if verbose:
                print(f"  {sc:30s} ...", end=" ", flush=True)
            rows = _run_one(sc, seed, n_per_domain, run_cdan, nl_strength=nl_strength)
            all_rows.extend(rows)
            done += 1
            if verbose:
                # Quick summary of target AUROCs for this cell
                method_aurocs = {r["method"]: r["tgt_auroc"] for r in rows}
                summary = "  ".join(
                    f"{m}={v:.3f}" for m, v in method_aurocs.items()
                    if not np.isnan(v)
                )
                print(f"({done}/{n_total})  {summary}", flush=True)

    agg     = _aggregate(all_rows, scenarios, methods)
    compact = _compact_table(agg, scenarios, methods)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(all_rows, _TABLES / "adaptation_raw.csv")
    save_table_csv(agg,      _TABLES / "adaptation_summary.csv")
    save_table_csv(compact,  _TABLES / "adaptation_compact.csv")

    if verbose:
        _print_full_report(agg, scenarios, methods)
        _print_implementation_report(agg, scenarios, methods)
        print(f"  Saved: {_TABLES}/adaptation_raw.csv")
        print(f"  Saved: {_TABLES}/adaptation_summary.csv")
        print(f"  Saved: {_TABLES}/adaptation_compact.csv")

    return {"raw": all_rows, "aggregated": agg, "compact": compact}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHIFT-SURV phase-2 adaptation benchmark",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help=f"Single scenario to run. Options: {SCENARIO_NAMES}",
    )
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of seeds (starting from 42). Ignored with --single_seed.")
    parser.add_argument("--single_seed", action="store_true",
                        help="Run only seed 42 (quick check).")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Seed to use with --single_seed.")
    parser.add_argument("--n",       type=int, default=500,
                        help="Samples per domain.")
    parser.add_argument("--no_cdan", action="store_true",
                        help="Skip CDAN.")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress verbose output.")
    parser.add_argument("--nl_strength", type=float, default=0.3,
                        help="Quadratic observation nonlinearity α (0.0 = fully linear).")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIO_NAMES
    seeds     = [args.seed] if args.single_seed else list(range(42, 42 + args.n_seeds))

    run_adaptation(
        scenarios=scenarios,
        seeds=seeds,
        n_per_domain=args.n,
        run_cdan=not args.no_cdan,
        verbose=not args.quiet,
        nl_strength=args.nl_strength,
    )
