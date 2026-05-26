"""
Top-level script: run generator validation for all five shift scenarios.

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_sanity_checks
  python -m experiments.run_sanity_checks --scenario spurious_shift
  python -m experiments.run_sanity_checks --multi_seed
  python -m experiments.run_sanity_checks --multi_seed --n_seeds 20

Outputs
-------
  outputs/figures/<scenario>/           — diagnostic plots (single seed)
  outputs/tables/<scenario>/            — CSV and JSON (single seed)
  outputs/tables/all_scenarios_summary.csv
  outputs/tables/multi_seed_summary.csv — mean ± 95% CI across seeds
  outputs/tables/multi_seed_raw.csv     — per-seed scalar rows
  outputs/figures/multi_seed/auroc_gaps.png
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

# Allow imports from project root regardless of working directory
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy import stats as sp_stats

from generator.scenarios import generate_scenario, SCENARIO_NAMES
from experiments.diagnostics import run_diagnostics
from utils.io import tables_dir, save_table_csv, save_table_json


# ---------------------------------------------------------------------------
# Per-scenario interpretation checklist
# ---------------------------------------------------------------------------

EXPECTED_BEHAVIOR = {
    "no_shift": {
        "description": "Control regime — nothing varies across domains.",
        "checks": [
            "Prevalence is similar across source and target (~±5%).",
            "PCA shows overlapping source/target clouds.",
            "AUROC gap (src pooled − target) is small (<0.05) for all_observed.",
            "Invariant oracle ≈ all_observed (no free shortcut to exploit).",
        ],
        "pass_criteria": {
            "prevalence_shift": ("<", 0.06),
            "covariate_shift_score": ("<", 0.30),
            "all_observed_gap": ("<", 0.12),
        },
    },
    "covariate_shift": {
        "description": "Measurement-induced observational shift: A_inv rotated in target; risk mechanism invariant.",
        "checks": [
            "PCA shows source/target clouds in different regions.",
            "Prevalence is stable (mu_inv shift is orthogonal to w).",
            "Invariant oracle is stable (uses u_inv directly, unaffected by A rotation).",
            "all_observed degrades: source model learns wrong measurement basis.",
            "Gap is consistent across seeds (rotation is fixed, not seed-dependent).",
        ],
        "pass_criteria": {
            "covariate_shift_score": (">", 0.30),
            "all_observed_gap": (">", 0.05),
            "invariant_oracle_gap": ("<", "all_observed_gap"),
        },
    },
    "spurious_shift": {
        "description": "Spurious factor distribution flips; invariant mechanism unchanged.",
        "checks": [
            "Spurious oracle (u_spu) performs well on source, degrades on target.",
            "Invariant oracle (u_inv) is stable across source and target.",
            "AUROC gap for spurious_oracle > gap for invariant_oracle.",
            "all_observed degrades: it picks up the spurious shortcut.",
        ],
        "pass_criteria": {
            "spurious_oracle_gap": (">", 0.15),
            "inv_gap_less_than_spu_gap": True,
        },
    },
    "baseline_hazard_shift": {
        "description": "Baseline hazard elevated in target; latent ranking preserved.",
        "checks": [
            "Prevalence is substantially higher in target than sources.",
            "Invariant oracle AUROC is relatively stable (ranking preserved).",
            "PCA shows similar covariate clouds (no p(x) shift).",
            "η distribution shifts little, but T and y shift substantially.",
            "Invariant oracle ECE degrades on target (calibration shift, not ranking shift).",
        ],
        "pass_criteria": {
            "prevalence_shift": (">", 0.15),
            "covariate_shift_score": ("<", 0.30),
            "invariant_oracle_gap": ("<", 0.10),
            "invariant_oracle_ece_gap": (">", 0.05),
        },
    },
    "concept_shift": {
        "description": "w_d partially changes: weaker invariant coefficients flip; dominant preserved.",
        "checks": [
            "Invariant oracle degrades clearly on target (gap > 0.20).",
            "Target AUROC is substantially below source AUROC but stays above chance.",
            "No single source-trained invariant predictor works well on target.",
            "Prevalence and covariate distribution unchanged (pure mechanism shift).",
        ],
        "pass_criteria": {
            "invariant_oracle_gap": (">", 0.20),
            "target_auroc_degraded": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Automated pass/fail checks
# ---------------------------------------------------------------------------

def _check_scenario(name: str, summary: dict) -> list[dict]:
    """
    Apply simple threshold checks to shift indicators.
    Returns list of {check, result, value, threshold}.
    """
    ind = summary["shift_indicators"]
    gaps = ind.get("auroc_gaps", {})
    criteria = EXPECTED_BEHAVIOR.get(name, {}).get("pass_criteria", {})
    check_results = []

    def _gap(fs):
        g = gaps.get(fs, {})
        return g.get("gap", float("nan"))

    def _tgt_auroc(fs):
        g = gaps.get(fs, {})
        return g.get("target_auroc", float("nan"))

    def _ece_gap(fs):
        g = gaps.get(fs, {})
        return g.get("ece_gap", float("nan"))

    for key, condition in criteria.items():
        if key == "inv_gap_less_than_spu_gap":
            inv_g = _gap("invariant_oracle")
            spu_g = _gap("spurious_oracle")
            passed = inv_g < spu_g if not (np.isnan(inv_g) or np.isnan(spu_g)) else False
            check_results.append({
                "check": "inv_gap < spu_gap",
                "passed": passed,
                "inv_gap": round(inv_g, 4),
                "spu_gap": round(spu_g, 4),
            })
        elif key == "target_auroc_degraded":
            tgt_auroc_inv = _tgt_auroc("invariant_oracle")
            passed = tgt_auroc_inv <= 0.75 if not np.isnan(tgt_auroc_inv) else False
            check_results.append({
                "check": "target_auroc_inv_oracle ≤ 0.75 (degraded transport)",
                "passed": passed,
                "value": round(tgt_auroc_inv, 4),
            })
        elif key == "invariant_oracle_gap":
            op, thresh_raw = condition
            val = _gap("invariant_oracle")
            # thresh may be a scalar or a string referencing another gap
            if isinstance(thresh_raw, str):
                thresh = _gap(thresh_raw.replace("_gap", ""))
            else:
                thresh = thresh_raw
            passed = (val > thresh) if op == ">" else (val < thresh) if op == "<" else False
            check_results.append({
                "check": f"invariant_oracle_gap {op} {thresh_raw}",
                "passed": bool(passed),
                "value": round(val, 4),
                "threshold": round(thresh, 4) if isinstance(thresh, float) and not np.isnan(thresh) else thresh,
            })
        elif key == "spurious_oracle_gap":
            op, thresh = condition
            val = _gap("spurious_oracle")
            passed = (val > thresh) if op == ">" else (val < thresh) if op == "<" else False
            check_results.append({
                "check": f"spurious_oracle_gap {op} {thresh}",
                "passed": bool(passed),
                "value": round(val, 4),
            })
        elif key == "invariant_oracle_ece_gap":
            op, thresh = condition
            val = _ece_gap("invariant_oracle")
            passed = (val > thresh) if op == ">" else (val < thresh) if op == "<" else False
            check_results.append({
                "check": f"invariant_oracle_ece_gap {op} {thresh}",
                "passed": bool(passed),
                "value": round(val, 4) if not np.isnan(val) else float("nan"),
                "threshold": thresh,
            })
        elif key in ind:
            op, thresh = condition
            val = ind[key]
            if isinstance(thresh, str):
                # Compare to another metric (e.g., "all_observed_gap")
                thresh_val = _gap(thresh.replace("_gap", "").replace("all_observed", "all_observed"))
                passed = val < thresh_val
                check_results.append({
                    "check": f"{key} {op} {thresh}",
                    "passed": bool(passed),
                    "value": round(val, 4),
                    "threshold": round(thresh_val, 4) if not np.isnan(thresh_val) else "nan",
                })
            else:
                passed = (val > thresh) if op == ">" else (val < thresh)
                check_results.append({
                    "check": f"{key} {op} {thresh}",
                    "passed": bool(passed),
                    "value": round(val, 4),
                    "threshold": thresh,
                })
        elif key == "all_observed_gap":
            op, thresh = condition
            val = _gap("all_observed")
            passed = (val > thresh) if op == ">" else (val < thresh)
            check_results.append({
                "check": f"all_observed_gap {op} {thresh}",
                "passed": bool(passed),
                "value": round(val, 4),
            })

    return check_results


# ---------------------------------------------------------------------------
# Cross-scenario summary
# ---------------------------------------------------------------------------

def _build_cross_scenario_table(all_summaries: dict[str, dict]) -> list[dict]:
    rows = []
    for name, summary in all_summaries.items():
        ind = summary["shift_indicators"]
        gaps = ind.get("auroc_gaps", {})

        def _gap(fs):
            return gaps.get(fs, {}).get("gap", float("nan"))

        def _tgt(fs):
            return gaps.get(fs, {}).get("target_auroc", float("nan"))

        rows.append({
            "scenario": name,
            "src_prevalence": ind.get("source_mean_prevalence"),
            "tgt_prevalence": ind.get("target_prevalence"),
            "prevalence_shift": ind.get("prevalence_shift"),
            "covariate_shift_score": ind.get("covariate_shift_score"),
            "eta_mean_shift": ind.get("eta_mean_shift"),
            # LR baselines
            "inv_oracle_src_auroc": gaps.get("invariant_oracle", {}).get("source_pooled_auroc"),
            "inv_oracle_tgt_auroc": _tgt("invariant_oracle"),
            "inv_oracle_gap": _gap("invariant_oracle"),
            "spu_oracle_src_auroc": gaps.get("spurious_oracle", {}).get("source_pooled_auroc"),
            "spu_oracle_tgt_auroc": _tgt("spurious_oracle"),
            "spu_oracle_gap": _gap("spurious_oracle"),
            "all_obs_src_auroc": gaps.get("all_observed", {}).get("source_pooled_auroc"),
            "all_obs_tgt_auroc": _tgt("all_observed"),
            "all_obs_gap": _gap("all_observed"),
            # MLP baselines
            "inv_oracle_mlp_src_auroc": gaps.get("invariant_oracle_mlp", {}).get("source_pooled_auroc"),
            "inv_oracle_mlp_tgt_auroc": _tgt("invariant_oracle_mlp"),
            "inv_oracle_mlp_gap": _gap("invariant_oracle_mlp"),
            "spu_oracle_mlp_src_auroc": gaps.get("spurious_oracle_mlp", {}).get("source_pooled_auroc"),
            "spu_oracle_mlp_tgt_auroc": _tgt("spurious_oracle_mlp"),
            "spu_oracle_mlp_gap": _gap("spurious_oracle_mlp"),
            "all_obs_mlp_src_auroc": gaps.get("all_observed_mlp", {}).get("source_pooled_auroc"),
            "all_obs_mlp_tgt_auroc": _tgt("all_observed_mlp"),
            "all_obs_mlp_gap": _gap("all_observed_mlp"),
        })
    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(scenarios: list[str], n_per_domain: int = 500, seed: int = 42, verbose: bool = True):
    all_summaries = {}
    all_checks = {}

    for name in scenarios:
        if verbose:
            print(f"\nGenerating scenario: {name} ...")

        scenario_result = generate_scenario(
            name,
            n_source_domains=3,
            n_per_domain=n_per_domain,
            seed=seed,
        )

        summary = run_diagnostics(scenario_result, verbose=verbose)
        all_summaries[name] = summary

        checks = _check_scenario(name, summary)
        all_checks[name] = checks

        expected = EXPECTED_BEHAVIOR.get(name, {})
        if verbose:
            print(f"\n  Expected behavior ({name}):")
            for c in expected.get("checks", []):
                print(f"    ✓ {c}")
            print(f"\n  Automated checks:")
            for chk in checks:
                status = "PASS" if chk.get("passed") else "FAIL"
                print(f"    [{status}] {chk.get('check', '?')}  {chk}")

    # Cross-scenario table
    cross_rows = _build_cross_scenario_table(all_summaries)
    out_path = Path(__file__).parent.parent / "outputs" / "tables" / "all_scenarios_summary.csv"
    save_table_csv(cross_rows, out_path)

    checks_path = Path(__file__).parent.parent / "outputs" / "tables" / "sanity_checks.json"
    save_table_json(all_checks, checks_path)

    if verbose:
        print(f"\n{'='*60}")
        print("  CROSS-SCENARIO SUMMARY")
        print(f"{'='*60}")
        print(f"  {'scenario':25s} {'src_prev':>8s} {'tgt_prev':>8s} {'cov_shift':>9s} "
              f"{'inv_gap':>8s} {'spu_gap':>8s} {'all_gap':>8s}")
        print("  " + "-" * 80)
        for r in cross_rows:
            def _fmt(v):
                return f"{v:8.3f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "     nan"
            print(
                f"  {r['scenario']:25s} "
                f"{_fmt(r['src_prevalence'])} "
                f"{_fmt(r['tgt_prevalence'])} "
                f"{_fmt(r['covariate_shift_score'])} "
                f"{_fmt(r['inv_oracle_gap'])} "
                f"{_fmt(r['spu_oracle_gap'])} "
                f"{_fmt(r['all_obs_gap'])}"
            )
        print(f"\n  Outputs saved to: outputs/figures/ and outputs/tables/")
        print(f"  All-scenarios CSV: {out_path}")
        print(f"  Sanity checks JSON: {checks_path}")

    return all_summaries, all_checks


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = list(range(42, 52))   # seeds 42–51


def _summary_to_scalar_row(name: str, summary: dict) -> dict:
    """Flatten a single-seed summary into a dict of scalar metrics."""
    ind = summary["shift_indicators"]
    gaps = ind.get("auroc_gaps", {})

    def _gap(fs): return gaps.get(fs, {}).get("gap", float("nan"))
    def _src(fs): return gaps.get(fs, {}).get("source_pooled_auroc", float("nan"))
    def _tgt(fs): return gaps.get(fs, {}).get("target_auroc", float("nan"))
    def _ece_gap(fs): return gaps.get(fs, {}).get("ece_gap", float("nan"))

    return {
        "scenario": name,
        "src_prevalence": ind.get("source_mean_prevalence"),
        "tgt_prevalence": ind.get("target_prevalence"),
        "prevalence_shift": ind.get("prevalence_shift"),
        "covariate_shift_score": ind.get("covariate_shift_score"),
        # LR baselines
        "inv_oracle_src_auroc": _src("invariant_oracle"),
        "inv_oracle_tgt_auroc": _tgt("invariant_oracle"),
        "inv_oracle_gap": _gap("invariant_oracle"),
        "spu_oracle_src_auroc": _src("spurious_oracle"),
        "spu_oracle_tgt_auroc": _tgt("spurious_oracle"),
        "spu_oracle_gap": _gap("spurious_oracle"),
        "all_obs_src_auroc": _src("all_observed"),
        "all_obs_tgt_auroc": _tgt("all_observed"),
        "all_obs_gap": _gap("all_observed"),
        # MLP baselines
        "inv_oracle_mlp_src_auroc": _src("invariant_oracle_mlp"),
        "inv_oracle_mlp_tgt_auroc": _tgt("invariant_oracle_mlp"),
        "inv_oracle_mlp_gap": _gap("invariant_oracle_mlp"),
        "spu_oracle_mlp_src_auroc": _src("spurious_oracle_mlp"),
        "spu_oracle_mlp_tgt_auroc": _tgt("spurious_oracle_mlp"),
        "spu_oracle_mlp_gap": _gap("spurious_oracle_mlp"),
        "all_obs_mlp_src_auroc": _src("all_observed_mlp"),
        "all_obs_mlp_tgt_auroc": _tgt("all_observed_mlp"),
        "all_obs_mlp_gap": _gap("all_observed_mlp"),
        # ECE gaps (target_ece − source_pooled_ece; positive = worse calibration on target)
        "inv_oracle_ece_gap": _ece_gap("invariant_oracle"),
        "all_obs_ece_gap": _ece_gap("all_observed"),
        "inv_oracle_mlp_ece_gap": _ece_gap("invariant_oracle_mlp"),
        "all_obs_mlp_ece_gap": _ece_gap("all_observed_mlp"),
    }


def _aggregate_multi_seed(raw: dict[str, list[dict]]) -> list[dict]:
    """
    Compute mean ± 95% CI (t-distribution, ddof=1) across seeds for each
    scenario and metric.  Output columns: <metric>_mean, _std, _ci95_low,
    _ci95_high.
    """
    numeric_keys = [k for k in list(raw.values())[0][0].keys() if k != "scenario"]
    agg_rows = []
    for name, rows in raw.items():
        agg = {"scenario": name, "n_seeds": len(rows)}
        for m in numeric_keys:
            vals = np.array([r[m] for r in rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                for suffix in ("_mean", "_std", "_ci95_low", "_ci95_high"):
                    agg[f"{m}{suffix}"] = float("nan")
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            se = std / np.sqrt(len(vals))
            t_crit = sp_stats.t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
            ci = t_crit * se
            agg[f"{m}_mean"] = round(mean, 4)
            agg[f"{m}_std"] = round(std, 4)
            agg[f"{m}_ci95_low"] = round(mean - ci, 4)
            agg[f"{m}_ci95_high"] = round(mean + ci, 4)
        agg_rows.append(agg)
    return agg_rows


def _print_multi_seed_table(agg_rows: list[dict]) -> None:
    print(f"\n{'='*90}")
    print("  MULTI-SEED SUMMARY  (mean ± 95% CI)")
    print(f"{'='*90}")
    # LR gaps
    lr_metrics = ["inv_oracle_gap", "spu_oracle_gap", "all_obs_gap"]
    # MLP gaps
    mlp_metrics = ["inv_oracle_mlp_gap", "spu_oracle_mlp_gap", "all_obs_mlp_gap"]
    print(f"\n  LR baselines:")
    header = f"  {'scenario':25s}" + "".join(f"  {m[:14]:>16s}" for m in lr_metrics)
    print(header)
    print("  " + "-" * (27 + 18 * len(lr_metrics)))
    for row in agg_rows:
        line = f"  {row['scenario']:25s}"
        for m in lr_metrics:
            mean = row.get(f"{m}_mean", float("nan"))
            ci_lo = row.get(f"{m}_ci95_low", float("nan"))
            ci_hi = row.get(f"{m}_ci95_high", float("nan"))
            if not any(np.isnan([mean, ci_lo, ci_hi])):
                half_ci = (ci_hi - ci_lo) / 2
                line += f"  {mean:+.3f} ± {half_ci:.3f}"
            else:
                line += f"  {'nan':>16s}"
        print(line)
    print(f"\n  MLP baselines:")
    header = f"  {'scenario':25s}" + "".join(f"  {m[:14]:>16s}" for m in mlp_metrics)
    print(header)
    print("  " + "-" * (27 + 18 * len(mlp_metrics)))
    for row in agg_rows:
        line = f"  {row['scenario']:25s}"
        for m in mlp_metrics:
            mean = row.get(f"{m}_mean", float("nan"))
            ci_lo = row.get(f"{m}_ci95_low", float("nan"))
            ci_hi = row.get(f"{m}_ci95_high", float("nan"))
            if not any(np.isnan([mean, ci_lo, ci_hi])):
                half_ci = (ci_hi - ci_lo) / 2
                line += f"  {mean:+.3f} ± {half_ci:.3f}"
            else:
                line += f"  {'nan':>16s}"
        print(line)

    # ECE gaps (calibration)  — positive = worse calibration on target
    ece_metrics = ["inv_oracle_ece_gap", "all_obs_ece_gap", "inv_oracle_mlp_ece_gap", "all_obs_mlp_ece_gap"]
    ece_labels  = ["inv_oracle(LR)", "all_obs(LR)", "inv_oracle(MLP)", "all_obs(MLP)"]
    print(f"\n  ECE gaps (target_ece − source_ece; positive = miscalibrated on target):")
    header = f"  {'scenario':25s}" + "".join(f"  {lbl:>18s}" for lbl in ece_labels)
    print(header)
    print("  " + "-" * (27 + 20 * len(ece_metrics)))
    for row in agg_rows:
        line = f"  {row['scenario']:25s}"
        for m in ece_metrics:
            mean = row.get(f"{m}_mean", float("nan"))
            ci_lo = row.get(f"{m}_ci95_low", float("nan"))
            ci_hi = row.get(f"{m}_ci95_high", float("nan"))
            if not any(np.isnan([mean, ci_lo, ci_hi])):
                half_ci = (ci_hi - ci_lo) / 2
                line += f"  {mean:+.3f} ± {half_ci:.3f}"
            else:
                line += f"  {'nan':>18s}"
        print(line)


def _compute_significance_tests(raw: dict[str, list[dict]]) -> list[dict]:
    """
    For each scenario:
      - One-sample t-test (H0: gap = 0) for each gap metric.
      - Paired t-test (H0: inv_oracle_gap = all_obs_gap) to test whether
        the invariant oracle transfers significantly better than all_observed.
    """
    # One-sample t-tests for AUROC gap keys (LR + MLP) and ECE gap keys
    gap_keys = [
        "inv_oracle_gap", "spu_oracle_gap", "all_obs_gap",
        "inv_oracle_mlp_gap", "spu_oracle_mlp_gap", "all_obs_mlp_gap",
        "inv_oracle_ece_gap", "all_obs_ece_gap",
        "inv_oracle_mlp_ece_gap", "all_obs_mlp_ece_gap",
    ]
    rows = []

    for name, seed_rows in raw.items():
        row = {"scenario": name}

        for key in gap_keys:
            vals = np.array([r.get(key, float("nan")) for r in seed_rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) < 2:
                row[f"{key}_tstat"] = float("nan")
                row[f"{key}_pval"] = float("nan")
                row[f"{key}_sig"] = False
                continue
            tstat, pval = sp_stats.ttest_1samp(vals, popmean=0)
            row[f"{key}_tstat"] = round(float(tstat), 3)
            row[f"{key}_pval"] = round(float(pval), 4)
            row[f"{key}_sig"] = bool(pval < 0.05)

        # Paired t-test: does inv_oracle degrade less than all_observed? (LR)
        inv_vals = np.array([r["inv_oracle_gap"] for r in seed_rows], dtype=float)
        all_vals = np.array([r["all_obs_gap"] for r in seed_rows], dtype=float)
        mask = ~(np.isnan(inv_vals) | np.isnan(all_vals))
        if mask.sum() >= 2:
            tstat, pval = sp_stats.ttest_rel(inv_vals[mask], all_vals[mask])
            row["inv_vs_all_tstat"] = round(float(tstat), 3)
            row["inv_vs_all_pval"] = round(float(pval), 4)
            row["inv_vs_all_sig"] = bool(pval < 0.05)
        else:
            row["inv_vs_all_tstat"] = float("nan")
            row["inv_vs_all_pval"] = float("nan")
            row["inv_vs_all_sig"] = False

        # Paired t-test: inv_oracle_mlp vs all_obs_mlp
        inv_mlp_vals = np.array([r.get("inv_oracle_mlp_gap", float("nan")) for r in seed_rows], dtype=float)
        all_mlp_vals = np.array([r.get("all_obs_mlp_gap", float("nan")) for r in seed_rows], dtype=float)
        mask_mlp = ~(np.isnan(inv_mlp_vals) | np.isnan(all_mlp_vals))
        if mask_mlp.sum() >= 2:
            tstat, pval = sp_stats.ttest_rel(inv_mlp_vals[mask_mlp], all_mlp_vals[mask_mlp])
            row["inv_vs_all_mlp_tstat"] = round(float(tstat), 3)
            row["inv_vs_all_mlp_pval"] = round(float(pval), 4)
            row["inv_vs_all_mlp_sig"] = bool(pval < 0.05)
        else:
            row["inv_vs_all_mlp_tstat"] = float("nan")
            row["inv_vs_all_mlp_pval"] = float("nan")
            row["inv_vs_all_mlp_sig"] = False

        rows.append(row)
    return rows


def _print_significance_table(sig_rows: list[dict]) -> None:
    def _fmt(pval, sig):
        if np.isnan(float(pval)):
            return f"{'nan':>14s}"
        marker = " *" if sig else "  "
        return f"p={float(pval):.4f}{marker}"

    print(f"\n{'='*86}")
    print("  SIGNIFICANCE TESTS  (one-sample t H0: gap=0  |  paired t H0: inv_gap=all_gap)")
    print(f"{'='*86}")
    # LR table
    print(f"\n  LR baselines:")
    print(f"  {'scenario':25s}  {'inv_gap':>14s}  {'spu_gap':>14s}  {'all_gap':>14s}  {'inv_vs_all':>14s}")
    print("  " + "-" * 83)
    for row in sig_rows:
        print(
            f"  {row['scenario']:25s}  "
            f"{_fmt(row['inv_oracle_gap_pval'], row['inv_oracle_gap_sig']):>14s}  "
            f"{_fmt(row['spu_oracle_gap_pval'], row['spu_oracle_gap_sig']):>14s}  "
            f"{_fmt(row['all_obs_gap_pval'], row['all_obs_gap_sig']):>14s}  "
            f"{_fmt(row['inv_vs_all_pval'], row['inv_vs_all_sig']):>14s}"
        )
    # MLP table
    print(f"\n  MLP baselines:")
    print(f"  {'scenario':25s}  {'inv_mlp':>14s}  {'spu_mlp':>14s}  {'all_mlp':>14s}  {'inv_vs_all':>14s}")
    print("  " + "-" * 83)
    for row in sig_rows:
        print(
            f"  {row['scenario']:25s}  "
            f"{_fmt(row['inv_oracle_mlp_gap_pval'], row['inv_oracle_mlp_gap_sig']):>14s}  "
            f"{_fmt(row['spu_oracle_mlp_gap_pval'], row['spu_oracle_mlp_gap_sig']):>14s}  "
            f"{_fmt(row['all_obs_mlp_gap_pval'], row['all_obs_mlp_gap_sig']):>14s}  "
            f"{_fmt(row['inv_vs_all_mlp_pval'], row['inv_vs_all_mlp_sig']):>14s}"
        )
    # ECE gap significance (calibration degradation)
    print(f"\n  ECE gaps (one-sample t H0: ece_gap=0):")
    print(f"  {'scenario':25s}  {'inv(LR)':>14s}  {'all(LR)':>14s}  {'inv(MLP)':>14s}  {'all(MLP)':>14s}")
    print("  " + "-" * 83)
    for row in sig_rows:
        print(
            f"  {row['scenario']:25s}  "
            f"{_fmt(row.get('inv_oracle_ece_gap_pval', float('nan')), row.get('inv_oracle_ece_gap_sig', False)):>14s}  "
            f"{_fmt(row.get('all_obs_ece_gap_pval', float('nan')), row.get('all_obs_ece_gap_sig', False)):>14s}  "
            f"{_fmt(row.get('inv_oracle_mlp_ece_gap_pval', float('nan')), row.get('inv_oracle_mlp_ece_gap_sig', False)):>14s}  "
            f"{_fmt(row.get('all_obs_mlp_ece_gap_pval', float('nan')), row.get('all_obs_mlp_ece_gap_sig', False)):>14s}"
        )
    print("  * p < 0.05")


def run_multi_seed(
    scenarios: list[str] = SCENARIO_NAMES,
    seeds: list[int] = DEFAULT_SEEDS,
    n_per_domain: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Run each scenario across multiple seeds and aggregate with mean ± 95% CI.

    Returns
    -------
    dict with keys "raw" (per-seed rows) and "aggregated" (mean ± CI rows).
    """
    from utils.plotting import plot_multi_seed_gaps

    raw: dict[str, list[dict]] = {name: [] for name in scenarios}

    for s_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n[Seed {seed}  ({s_idx + 1}/{len(seeds)})]", flush=True)
        for name in scenarios:
            scenario_result = generate_scenario(
                name, n_source_domains=3, n_per_domain=n_per_domain, seed=seed,
            )
            summary = run_diagnostics(scenario_result, verbose=False)
            raw[name].append(_summary_to_scalar_row(name, summary))

    agg_rows = _aggregate_multi_seed(raw)
    sig_rows = _compute_significance_tests(raw)

    tables_root = Path(__file__).parent.parent / "outputs" / "tables"
    save_table_csv(agg_rows, tables_root / "multi_seed_summary.csv")
    save_table_csv(sig_rows, tables_root / "significance_tests.csv")
    raw_rows = [row for rows in raw.values() for row in rows]
    save_table_csv(raw_rows, tables_root / "multi_seed_raw.csv")

    fig_dir = Path(__file__).parent.parent / "outputs" / "figures" / "multi_seed"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_multi_seed_gaps(raw, scenarios, save_path=fig_dir / "auroc_gaps.png")

    if verbose:
        _print_multi_seed_table(agg_rows)
        _print_significance_table(sig_rows)
        print(f"\n  Saved: {tables_root / 'multi_seed_summary.csv'}")
        print(f"  Saved: {tables_root / 'multi_seed_raw.csv'}")
        print(f"  Saved: {tables_root / 'significance_tests.csv'}")
        print(f"  Saved: {fig_dir / 'auroc_gaps.png'}")

    return {"raw": raw, "aggregated": agg_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHIFT-SURV generator sanity checks")
    parser.add_argument(
        "--scenario", type=str, default=None,
        help=f"Run a single scenario. Options: {SCENARIO_NAMES}"
    )
    parser.add_argument("--n", type=int, default=500, help="Samples per domain")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (single-seed mode)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--multi_seed", action="store_true",
                        help="Run multi-seed aggregation instead of single-seed diagnostics")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of seeds for multi-seed mode (starting from 42)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIO_NAMES

    if args.multi_seed:
        seeds = list(range(42, 42 + args.n_seeds))
        run_multi_seed(scenarios=scenarios, seeds=seeds,
                       n_per_domain=args.n, verbose=not args.quiet)
    else:
        run_all(scenarios, n_per_domain=args.n, seed=args.seed, verbose=not args.quiet)
