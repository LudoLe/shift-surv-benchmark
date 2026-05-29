"""
Phase-2 experiment runner: domain adaptation methods on all shift scenarios.

Design: 1 source domain + 1 target domain, linear observation model (nl_strength=0.0),
IWCV for DANN/CDAN hyper-parameter selection.

Core methods
------------
  inv_oracle   — LR on true invariant latent factors u_inv (upper bound)
  all_observed — Pooled ERM LR on observed x (source-only baseline)
  coral        — CORAL with LR estimator
  dann         — DANN with IWCV HP selection
  cdan         — Conditional DANN with IWCV HP selection

Optional / appendix
-------------------
  irm          — IRM (excluded from main benchmark; add --include_irm)

Scenarios
---------
  Core    : no_shift, covariate_shift, spurious_shift, baseline_hazard_shift, concept_shift
  Appendix: strong_covariate_shift, cure_shift  (add --include_appendix)

Usage
-----
  cd /path/to/SHIFT-SURV

  # Paper benchmark (1+1, 5 scenarios, IWCV, 10 seeds):
  python -m experiments.run_adaptation

  # Quick smoke-test (1 seed):
  python -m experiments.run_adaptation --n_seeds 1

  # Include appendix scenarios:
  python -m experiments.run_adaptation --include_appendix

  # Covariate-shift IWCV ablation:
  python -m experiments.run_adaptation --scenarios covariate_shift --no_iwcv \\
      --out_prefix ablation_no_iwcv

  # Legacy 3-source run:
  python -m experiments.run_adaptation --n_source_domains 3 --out_prefix legacy_3s1t

Outputs  (in outputs/tables/<out_prefix>_*.csv)
-------
  *_raw.csv       — one row per (scenario, seed, method)
  *_summary.csv   — mean ± std ± 95-CI per (scenario, method)
  *_compact.csv   — pivot: scenarios × methods, tgt_auroc / gap / tgt_ece
  *_hp_sweep.csv  — per-seed IWCV candidate scores for DANN/CDAN
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator.scenarios import generate_scenario, SCENARIO_NAMES
from experiments.baselines import (
    fit_invariant_oracle, fit_all_observed, fit_prior_recalib,
    fit_platt_scaling, fit_isotonic,
)
from experiments.adaptation_methods import fit_coral, fit_dann, fit_cdan
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"

CORE_SCENARIOS = [
    "no_shift",
    "covariate_shift",
    "observational_shift",
    "spurious_shift",
    "baseline_hazard_shift",
    "concept_shift",
    "shape_shift",
]
APPENDIX_SCENARIOS = [
    "strong_covariate_shift",
    "cure_shift",
]

# Short display labels (for column headers)
_DISPLAY = {
    "invariant_oracle": "inv_oracle",
    "all_observed":     "lr_erm",
    "coral":            "coral",
    "prior_recalib":    "prior_recalib",
    "dann":             "dann",
    "cdan":             "cdan",
    "platt_scaling":    "platt",
    "isotonic":         "isotonic",
}


# ---------------------------------------------------------------------------
# Per-(scenario, seed) runner
# ---------------------------------------------------------------------------

def _run_one(
    scenario: str,
    seed: int,
    n_per_domain: int,
    n_source_domains: int,
    use_iwcv: bool,
    nl_strength: float,
    lambda_grid: list | None,
    arch_grid: list | None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Generate one scenario/seed, fit all methods, return (flat_rows, hp_rows, loss_rows, calib_rows)."""
    from generator.parameters import BaseParams
    base = BaseParams(seed=seed, nl_strength=nl_strength)
    data = generate_scenario(
        scenario,
        n_source_domains=n_source_domains,
        n_per_domain=n_per_domain,
        seed=seed,
        base=base,
    )
    src, tgt = data["source"], data["target"]

    results = [
        fit_invariant_oracle(src, tgt),
        fit_all_observed(src, tgt),
        fit_coral(src, tgt, seed=seed),
        fit_prior_recalib(src, tgt),
        fit_platt_scaling(src, tgt),
        fit_isotonic(src, tgt),
        fit_dann(
            src, tgt, seed=seed,
            use_iwcv=use_iwcv,
            lambda_grid=lambda_grid,
            arch_grid=arch_grid,
        ),
        fit_cdan(
            src, tgt, seed=seed,
            use_iwcv=use_iwcv,
            lambda_grid=lambda_grid,
            arch_grid=arch_grid,
        ),
    ]

    flat_rows = [_flatten(r, scenario, seed) for r in results]

    # Collect HP sweep rows from DANN and CDAN
    hp_rows: list[dict] = []
    for r in results:
        if "hp" in r and r["hp"].get("hp_rows"):
            for hp in r["hp"]["hp_rows"]:
                hp_rows.append({
                    "scenario":   scenario,
                    "seed":       seed,
                    "method":     r["feature_set"],
                    "lambda":     hp["lambda"],
                    "arch":       str(hp["arch"]),
                    "iwcv_auroc": hp["iwcv_auroc"],
                    "chosen":     (
                        hp["lambda"] == r["hp"]["lambda"]
                        and str(hp["arch"]) == str(r["hp"]["arch"])
                    ),
                })

    # Collect loss histories from DANN and CDAN
    loss_rows: list[dict] = []
    for r in results:
        hist = r.get("loss_history", {})
        if not hist:
            continue
        n_epochs = len(next(iter(hist.values()), []))
        for ep in range(n_epochs):
            row = {"scenario": scenario, "seed": seed, "method": r["feature_set"], "epoch": ep}
            for key, vals in hist.items():
                if ep < len(vals):
                    row[key] = vals[ep]
            loss_rows.append(row)

    # Collect target probabilities for reliability diagrams
    # calib_data[method] = (y_true, y_prob) both as numpy arrays
    y_tgt = tgt.y if hasattr(tgt, "y") else np.concatenate([ds.y for ds in tgt])
    calib_data: dict[str, tuple] = {}
    for r in results:
        m     = r.get("feature_set", "?")
        probs = r.get("target_probs")
        if probs is not None:
            calib_data[m] = (y_tgt, np.asarray(probs))

    return flat_rows, hp_rows, loss_rows, calib_data


def _flatten(r: dict, scenario: str, seed: int) -> dict:
    """Flatten one method-result dict to a single scalar row."""
    sp  = r.get("source_pooled", {})
    tgt = r.get("target", {})
    sa  = sp.get("auroc", float("nan"))
    ta  = tgt.get("auroc", float("nan"))
    gap = round(sa - ta, 4) if not (np.isnan(sa) or np.isnan(ta)) else float("nan")
    return {
        "scenario":       scenario,
        "seed":           seed,
        "method":         r.get("feature_set", "?"),
        "src_auroc":      sa,
        "src_auprc":      sp.get("auprc",  float("nan")),
        "src_ece":        sp.get("ece",    float("nan")),
        "tgt_auroc":      ta,
        "tgt_auprc":      tgt.get("auprc", float("nan")),
        "tgt_ece":        tgt.get("ece",   float("nan")),
        "gap":            gap,
        "tgt_f1":         tgt.get("f1",          float("nan")),
        "tgt_sensitivity": tgt.get("sensitivity", float("nan")),
        "tgt_specificity": tgt.get("specificity", float("nan")),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["src_auroc", "src_auprc", "src_ece",
                "tgt_auroc", "tgt_auprc", "tgt_ece", "gap",
                "tgt_f1", "tgt_sensitivity", "tgt_specificity"]


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
    """Pivot table: rows = scenarios, columns = methods × {tgt_auroc, gap, tgt_ece}."""
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
    n_seeds: int,
) -> None:
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  ADAPTATION METHODS — RESULTS ACROSS SCENARIOS  (mean ± 95% CI, {n_seeds} seeds)")
    print(sep)
    _print_table(agg, scenarios, methods, "tgt_auroc",      "Target AUROC  ↑")
    _print_table(agg, scenarios, methods, "gap",
                 "AUROC gap  (source pooled − target; positive = degradation)  ↓", sign=True)
    _print_table(agg, scenarios, methods, "tgt_ece",        "Target ECE  ↓ (lower = better calibration)")
    _print_table(agg, scenarios, methods, "tgt_auprc",      "Target AUPRC  ↑")
    _print_table(agg, scenarios, methods, "tgt_f1",         "Target F1  ↑  (threshold = 0.5)")
    _print_table(agg, scenarios, methods, "tgt_sensitivity","Target Sensitivity  ↑  (threshold = 0.5)")
    _print_table(agg, scenarios, methods, "tgt_specificity","Target Specificity  ↑  (threshold = 0.5)")


def _plot_loss_curves(
    loss_rows: list[dict],
    scenarios: list[str],
    out_dir: Path,
    prefix: str,
) -> None:
    """
    One PNG per scenario: 2 subplots (DANN, CDAN).
    Each subplot: per-seed train/val loss as thin lines + mean as thick line.
    """
    if not loss_rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        sc_rows = [r for r in loss_rows if r["scenario"] == sc]
        if not sc_rows:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        fig.suptitle(f"Loss curves — {sc}", fontsize=12)

        for ax, method in zip(axes, ["dann", "cdan"]):
            m_rows = [r for r in sc_rows if r["method"] == method]
            if not m_rows:
                ax.set_title(method.upper())
                ax.set_visible(False)
                continue

            seeds = sorted({r["seed"] for r in m_rows})
            # collect per-seed arrays
            train_curves, val_curves = [], []
            for s in seeds:
                s_rows = sorted([r for r in m_rows if r["seed"] == s], key=lambda x: x["epoch"])
                if not s_rows:
                    continue
                tr = [r.get("loss", float("nan")) for r in s_rows]
                vl = [r.get("val_loss", float("nan")) for r in s_rows]
                train_curves.append(tr)
                if any(not np.isnan(v) for v in vl):
                    val_curves.append(vl)
                ax.plot(tr, color="steelblue", alpha=0.25, linewidth=0.8)
                if any(not np.isnan(v) for v in vl):
                    ax.plot(vl, color="tomato", alpha=0.25, linewidth=0.8)

            # mean curves
            max_ep = max(len(c) for c in train_curves) if train_curves else 0
            if train_curves:
                padded = [c + [float("nan")] * (max_ep - len(c)) for c in train_curves]
                mean_tr = np.nanmean(padded, axis=0)
                ax.plot(mean_tr, color="steelblue", linewidth=2, label="train loss (mean)")
            if val_curves:
                padded_v = [c + [float("nan")] * (max_ep - len(c)) for c in val_curves]
                mean_vl = np.nanmean(padded_v, axis=0)
                ax.plot(mean_vl, color="tomato", linewidth=2, label="val loss (mean)")

            ax.set_title(method.upper())
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = out_dir / f"{prefix}_loss_{sc}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


def _plot_reliability_diagrams(
    all_rows_raw: list[dict],
    scenarios: list[str],
    all_results_by_seed: list[tuple[str, int, list[dict]]],
    out_dir: "Path",
    prefix: str,
) -> None:
    """
    Reliability diagrams: one PNG per scenario.
    Each subplot shows the calibration curve (mean predicted prob vs fraction positive)
    for each method, averaged across seeds.

    Parameters
    ----------
    all_results_by_seed : list of (scenario, seed, results_list)
        The raw result dicts from fit_* calls, needed to get per-sample probabilities.
    """
    pass  # populated below via _plot_reliability_diagrams_impl


def _plot_calibration_curves(
    calib_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    scenarios: list[str],
    methods: list[str],
    out_dir: "Path",
    prefix: str,
) -> None:
    """
    Plot reliability diagrams from pre-aggregated calibration data.

    calib_data[scenario][method] = (mean_pred_prob, fraction_pos)
    One PNG per scenario, one curve per method.
    """
    from sklearn.calibration import calibration_curve as _cal_curve

    colors = {
        "invariant_oracle": "black",
        "all_observed":     "steelblue",
        "coral":            "seagreen",
        "prior_recalib":    "goldenrod",
        "platt_scaling":    "peru",
        "isotonic":         "darkorange",
        "dann":             "tomato",
        "cdan":             "orchid",
    }
    linestyles = {
        "invariant_oracle": "--",
        "all_observed":     "-",
        "coral":            "-",
        "prior_recalib":    "-",
        "platt_scaling":    "-.",
        "isotonic":         "-.",
        "dann":             "-",
        "cdan":             "-",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    n_sc  = len(scenarios)
    ncols = min(4, n_sc)
    nrows = (n_sc + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    for ax_i, sc in enumerate(scenarios):
        ax = axes[ax_i]
        ax.plot([0, 1], [0, 1], "k:", linewidth=1.0, label="perfect")
        for m in methods:
            data = calib_data.get(sc, {}).get(m)
            if data is None:
                continue
            frac_pos, mean_pred = data
            lbl   = _DISPLAY.get(m, m)
            color = colors.get(m, "gray")
            ls    = linestyles.get(m, "-")
            ax.plot(mean_pred, frac_pos, color=color, linestyle=ls,
                    linewidth=1.8, marker="o", markersize=4, label=lbl)
        ax.set_title(sc.replace("_", " "), fontsize=9)
        ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for ax_i in range(len(scenarios), len(axes)):
        axes[ax_i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(methods) + 1),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Reliability diagrams (target domain, mean across seeds)", fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = out_dir / f"{prefix}_reliability.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _print_implementation_report(
    agg: list[dict],
    scenarios: list[str],
    methods: list[str],
) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  IMPLEMENTATION REPORT  (DA methods vs LR ERM baseline)")
    print(sep)
    idx = {(r["scenario"], r["method"]): r for r in agg}
    da_methods = [m for m in methods if m in ("coral", "prior_recalib", "dann", "cdan", "irm")]

    for m in da_methods:
        aurocs, nan_count = [], 0
        for sc in scenarios:
            r = idx.get((sc, m))
            if r:
                v = r.get("tgt_auroc_mean", float("nan"))
                if np.isnan(v):
                    nan_count += 1
                else:
                    aurocs.append(v)

        stable = nan_count == 0 and len(aurocs) == len(scenarios)
        print(f"\n  {m.upper()}  (vs lr_erm):")
        if stable:
            print(f"    Status : STABLE — ran on all {len(scenarios)} scenarios without errors.")
            print(f"    AUROC range (target): [{min(aurocs):.3f}, {max(aurocs):.3f}]")
            ref_aurocs = [
                idx[(sc, "all_observed")].get("tgt_auroc_mean", float("nan"))
                for sc in scenarios if (sc, "all_observed") in idx
            ]
            deltas = [a - b for a, b in zip(aurocs, ref_aurocs)
                      if not (np.isnan(a) or np.isnan(b))]
            if deltas:
                better = sum(1 for d in deltas if d > 0.005)
                worse  = sum(1 for d in deltas if d < -0.005)
                print(f"    vs lr_erm: better on {better}/{len(deltas)} scenarios, "
                      f"worse on {worse}/{len(deltas)}")
        else:
            print(f"    Status : UNSTABLE — {nan_count} scenario(s) returned NaN. "
                  f"Check warnings above.")

    print(f"\n  DELTA vs LR ERM (tgt_auroc, per scenario):")
    for sc in scenarios:
        lr = idx.get((sc, "all_observed"), {}).get("tgt_auroc_mean", float("nan"))
        parts = [f"{sc:30s}  lr_erm={lr:.3f}"]
        for m in da_methods:
            v = idx.get((sc, m), {}).get("tgt_auroc_mean", float("nan"))
            d = v - lr if not (np.isnan(v) or np.isnan(lr)) else float("nan")
            lbl = _DISPLAY.get(m, m)
            parts.append(f"  {lbl}={v:.3f}(Δ{d:+.3f})")
        print("  " + "".join(parts))
    print()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_adaptation(
    scenarios: list[str] = CORE_SCENARIOS,
    seeds: list[int] | None = None,
    n_per_domain: int = 500,
    n_source_domains: int = 1,
    use_iwcv: bool = True,
    nl_strength: float = 0.0,
    out_prefix: str = "benchmark_1s1t",
    lambda_grid: list | None = None,
    arch_grid: list | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run all methods across scenarios and seeds; aggregate and save results.

    Returns
    -------
    dict with keys: "raw", "aggregated", "compact", "hp_sweep"
    """
    if seeds is None:
        seeds = list(range(10))

    methods = [
        "invariant_oracle", "all_observed", "coral", "prior_recalib",
        "platt_scaling", "isotonic", "dann", "cdan",
    ]

    all_rows:  list[dict] = []
    all_hp:    list[dict] = []
    all_loss:  list[dict] = []
    # calib_accum[scenario][method] = list of (y_true, y_prob) tuples across seeds
    calib_accum: dict[str, dict[str, list]] = {sc: {} for sc in scenarios}
    n_total = len(seeds) * len(scenarios)
    done = 0

    for seed in seeds:
        if verbose:
            print(f"\n[Seed {seed}]", flush=True)
        for sc in scenarios:
            if verbose:
                print(f"  {sc:30s} ...", end=" ", flush=True)
            flat_rows, hp_rows, loss_rows, calib_data = _run_one(
                sc, seed, n_per_domain,
                n_source_domains=n_source_domains,
                use_iwcv=use_iwcv,
                nl_strength=nl_strength,
                lambda_grid=lambda_grid,
                arch_grid=arch_grid,
            )
            all_rows.extend(flat_rows)
            all_hp.extend(hp_rows)
            all_loss.extend(loss_rows)
            for m, pair in calib_data.items():
                calib_accum[sc].setdefault(m, []).append(pair)
            done += 1
            if verbose:
                method_aurocs = {r["method"]: r["tgt_auroc"] for r in flat_rows}
                summary = "  ".join(
                    f"{m}={v:.3f}" for m, v in method_aurocs.items()
                    if not np.isnan(v)
                )
                print(f"({done}/{n_total})  {summary}", flush=True)

    agg     = _aggregate(all_rows, scenarios, methods)
    compact = _compact_table(agg, scenarios, methods)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(all_rows,  _TABLES / f"{out_prefix}_raw.csv")
    save_table_csv(agg,       _TABLES / f"{out_prefix}_summary.csv")
    save_table_csv(compact,   _TABLES / f"{out_prefix}_compact.csv")
    if all_hp:
        save_table_csv(all_hp, _TABLES / f"{out_prefix}_hp_sweep.csv")
    if all_loss:
        save_table_csv(all_loss, _TABLES / f"{out_prefix}_loss_history.csv")

    _FIGURES = _ROOT / "outputs" / "figures"
    _plot_loss_curves(all_loss, scenarios, _FIGURES, out_prefix)

    # Build aggregated calibration curves (pool all seeds, then compute)
    from sklearn.calibration import calibration_curve as _cal_curve
    calib_plot_data: dict[str, dict[str, tuple]] = {}
    for sc in scenarios:
        calib_plot_data[sc] = {}
        for m, pairs in calib_accum.get(sc, {}).items():
            all_ytrue = np.concatenate([p[0] for p in pairs])
            all_yprob = np.concatenate([p[1] for p in pairs])
            try:
                frac_pos, mean_pred = _cal_curve(all_ytrue, all_yprob, n_bins=10, strategy="uniform")
                calib_plot_data[sc][m] = (frac_pos, mean_pred)
            except Exception:
                pass
    if any(calib_plot_data[sc] for sc in scenarios):
        _plot_calibration_curves(calib_plot_data, scenarios, methods, _FIGURES, out_prefix)

    if verbose:
        _print_full_report(agg, scenarios, methods, n_seeds=len(seeds))
        _print_implementation_report(agg, scenarios, methods)
        print(f"  Saved: {_TABLES}/{out_prefix}_raw.csv")
        print(f"  Saved: {_TABLES}/{out_prefix}_summary.csv")
        print(f"  Saved: {_TABLES}/{out_prefix}_compact.csv")
        if all_hp:
            print(f"  Saved: {_TABLES}/{out_prefix}_hp_sweep.csv")
        if all_loss:
            print(f"  Saved: {_TABLES}/{out_prefix}_loss_history.csv")
            print(f"  Saved: {_FIGURES}/{out_prefix}_loss_<scenario>.png")

    return {
        "raw":          all_rows,
        "aggregated":   agg,
        "compact":      compact,
        "hp_sweep":     all_hp,
        "loss_history": all_loss,
    }


# ---------------------------------------------------------------------------
# N-sweep: AUROC vs n_per_domain
# ---------------------------------------------------------------------------

def run_n_sweep(
    n_values: list[int],
    scenarios: list[str] = CORE_SCENARIOS,
    seeds: list[int] | None = None,
    use_iwcv: bool = True,
    nl_strength: float = 0.0,
    out_prefix: str = "n_sweep",
    verbose: bool = True,
) -> None:
    """
    Run the full benchmark at each value in *n_values* and save:
      - one raw/summary/compact CSV per n  (prefix_n<n>_*.csv)
      - one aggregated plot: AUROC vs n, per method, per scenario
    """
    methods = [
        "invariant_oracle", "all_observed", "coral", "prior_recalib",
        "platt_scaling", "isotonic", "dann", "cdan",
    ]
    sweep_agg: dict[int, list[dict]] = {}

    for n in n_values:
        if verbose:
            print(f"\n{'='*60}\n  n = {n}\n{'='*60}", flush=True)
        result = run_adaptation(
            scenarios=scenarios,
            seeds=seeds,
            n_per_domain=n,
            use_iwcv=use_iwcv,
            nl_strength=nl_strength,
            out_prefix=f"{out_prefix}_n{n}",
            verbose=verbose,
        )
        sweep_agg[n] = result["aggregated"]

    _plot_n_sweep(sweep_agg, scenarios, methods, out_prefix)


def _plot_n_sweep(
    sweep_agg: dict[int, list[dict]],
    scenarios: list[str],
    methods: list[str],
    out_prefix: str,
) -> None:
    """AUROC vs n per scenario (one subplot each), one line per method."""
    n_values = sorted(sweep_agg.keys())
    colors = {
        "invariant_oracle": "black",
        "all_observed":     "steelblue",
        "coral":            "seagreen",
        "prior_recalib":    "goldenrod",
        "platt_scaling":    "peru",
        "isotonic":         "darkorange",
        "dann":             "tomato",
        "cdan":             "orchid",
    }
    n_sc = len(scenarios)
    ncols = min(4, n_sc)
    nrows = (n_sc + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    for ax_i, sc in enumerate(scenarios):
        ax = axes[ax_i]
        for m in methods:
            lbl = _DISPLAY.get(m, m)
            means, cis = [], []
            for n in n_values:
                row = next((r for r in sweep_agg[n] if r["scenario"] == sc and r["method"] == m), None)
                if row:
                    means.append(float(row.get("tgt_auroc_mean", float("nan"))))
                    cis.append(float(row.get("tgt_auroc_ci95", 0.0)))
                else:
                    means.append(float("nan"))
                    cis.append(0.0)
            means = np.array(means)
            cis   = np.array(cis)
            ax.plot(n_values, means, marker="o", color=colors.get(m, "gray"),
                    linewidth=1.8, label=lbl)
            ax.fill_between(n_values, means - cis, means + cis,
                            alpha=0.12, color=colors.get(m, "gray"))
        ax.set_title(sc.replace("_", " "), fontsize=10)
        ax.set_xlabel("n per domain")
        ax.set_ylabel("Target AUROC")
        ax.set_xscale("log")
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])

    # hide unused subplots
    for ax_i in range(len(scenarios), len(axes)):
        axes[ax_i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Target AUROC vs n per domain  (mean ± 95% CI)", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    _FIGURES = _ROOT / "outputs" / "figures"
    _FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = _FIGURES / f"{out_prefix}_auroc_vs_n.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHIFT-SURV phase-2 adaptation benchmark (1+1 design)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None, metavar="SCENARIO",
        help=f"Scenarios to run (space-separated). Defaults to core 5. "
             f"Available: {CORE_SCENARIOS + APPENDIX_SCENARIOS}",
    )
    parser.add_argument(
        "--include_appendix", action="store_true",
        help="Append strong_covariate_shift and cure_shift to the scenario list.",
    )
    parser.add_argument(
        "--n_source_domains", type=int, default=1,
        help="Number of source domains (1 = benchmark default).",
    )
    parser.add_argument(
        "--n_seeds", type=int, default=10,
        help="Number of seeds.",
    )
    parser.add_argument(
        "--seed_start", type=int, default=0,
        help="First seed value (seeds = range(seed_start, seed_start + n_seeds)).",
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Samples per domain.",
    )
    parser.add_argument(
        "--nl_strength", type=float, default=0.0,
        help="Observation nonlinearity α (0.0 = fully linear).",
    )
    parser.add_argument(
        "--use_iwcv", dest="use_iwcv", action="store_true", default=True,
        help="Use IWCV for DANN/CDAN HP selection (default: on).",
    )
    parser.add_argument(
        "--no_iwcv", dest="use_iwcv", action="store_false",
        help="Disable IWCV; use source-only validation instead.",
    )
    parser.add_argument(
        "--out_prefix", type=str, default="benchmark_1s1t",
        help="Output filename prefix (outputs/tables/<prefix>_*.csv).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output.",
    )
    parser.add_argument(
        "--n_sweep", type=str, default=None, metavar="N1,N2,...",
        help="Comma-separated list of n_per_domain values to sweep over. "
             "Runs the full benchmark at each n and produces an AUROC-vs-n plot. "
             "Example: --n_sweep 250,500,1000,2000",
    )
    args = parser.parse_args()

    scenarios = list(args.scenarios) if args.scenarios else list(CORE_SCENARIOS)
    if args.include_appendix:
        scenarios = scenarios + [s for s in APPENDIX_SCENARIOS if s not in scenarios]

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if args.n_sweep:
        n_values = [int(x.strip()) for x in args.n_sweep.split(",")]
        run_n_sweep(
            n_values=n_values,
            scenarios=scenarios,
            seeds=seeds,
            use_iwcv=args.use_iwcv,
            nl_strength=args.nl_strength,
            out_prefix=args.out_prefix,
            verbose=not args.quiet,
        )
    else:
        run_adaptation(
            scenarios=scenarios,
            seeds=seeds,
            n_per_domain=args.n,
            n_source_domains=args.n_source_domains,
            use_iwcv=args.use_iwcv,
            nl_strength=args.nl_strength,
            out_prefix=args.out_prefix,
            verbose=not args.quiet,
        )
