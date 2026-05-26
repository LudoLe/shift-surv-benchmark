"""
Diagnostic experiments for generator validation.

For each scenario:
  1. Prevalence summary by domain
  2. PCA of observed covariates coloured by domain
  3. Latent factor distribution plots (inv and spu)
  4. Event time distribution plot
  5. Baseline logistic regression: invariant-oracle, spurious-oracle, all-observed
  6. Oracle log-risk diagnostics (η distribution by domain and by label)
  7. Summary table: key shift indicators

All outputs are saved under outputs/figures/<scenario>/ and outputs/tables/<scenario>/
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator.base_generator import DomainDataset
from experiments.baselines import evaluate_all_baselines, baselines_to_flat_rows
from utils.metrics import auroc, prevalence as prev_fn
from utils.plotting import (
    plot_pca_by_domain,
    plot_latent_distributions,
    plot_event_time_distribution,
    plot_prevalence_bar,
    plot_auroc_comparison,
)
from utils.io import figures_dir, tables_dir, save_table_csv, save_table_json


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_diagnostics(
    scenario_result: dict,
    verbose: bool = True,
) -> dict:
    """
    Run full diagnostic suite for a scenario result dict (from generate_scenario).

    Returns a summary dict with all key metrics and interpretation flags.
    """
    name = scenario_result["scenario"]
    sources: list[DomainDataset] = scenario_result["source"]
    target: DomainDataset = scenario_result["target"]
    tau: float = scenario_result["base"].tau

    fig_dir = figures_dir(name)
    tbl_dir = tables_dir(name)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Diagnostics: {name.upper()}")
        print(f"{'='*60}")

    # 1. Prevalence summary
    prev_rows = _prevalence_summary(sources, target, verbose)
    save_table_csv(prev_rows, tbl_dir / "prevalence_by_domain.csv")

    # 2. PCA plot
    plot_pca_by_domain(
        sources, target,
        title=f"[{name}] PCA of observed covariates",
        save_path=fig_dir / "pca_covariates.png",
    )

    # 3. Latent factor distributions
    plot_latent_distributions(
        sources, target, factor="inv",
        title=f"[{name}] Invariant latent factor u^(inv)[0]",
        save_path=fig_dir / "latent_inv_dist.png",
    )
    plot_latent_distributions(
        sources, target, factor="spu",
        title=f"[{name}] Spurious latent factor u^(spu)[0]",
        save_path=fig_dir / "latent_spu_dist.png",
    )

    # 4. Event time distribution
    plot_event_time_distribution(
        sources, target, tau=tau,
        title=f"[{name}] Event time distributions",
        save_path=fig_dir / "event_times.png",
    )

    # 5. Prevalence bar chart
    plot_prevalence_bar(
        sources, target,
        title=f"[{name}] 6-month event prevalence",
        save_path=fig_dir / "prevalence_bar.png",
    )

    # 6. Baselines
    baseline_results = evaluate_all_baselines(sources, target)
    baseline_rows = baselines_to_flat_rows(baseline_results)
    save_table_csv(baseline_rows, tbl_dir / "baseline_results.csv")

    if verbose:
        _print_baseline_table(baseline_rows)

    # 7. Oracle log-risk plot
    _plot_oracle_risk(sources, target, name, fig_dir, verbose)

    # 8. AUROC comparison plot
    auroc_data = _extract_auroc_for_plot(baseline_results, sources, target)
    plot_auroc_comparison(
        auroc_data,
        title=f"[{name}] AUROC by feature set",
        save_path=fig_dir / "auroc_comparison.png",
    )

    # 9. Shift indicators
    indicators = _compute_shift_indicators(sources, target, baseline_results)
    save_table_json(indicators, tbl_dir / "shift_indicators.json")

    if verbose:
        _print_shift_indicators(name, indicators)

    # Compact summary
    summary = {
        "scenario": name,
        "prevalence_rows": prev_rows,
        "baseline_rows": baseline_rows,
        "shift_indicators": indicators,
    }
    save_table_json(summary, tbl_dir / "summary.json")

    return summary


# ---------------------------------------------------------------------------
# Sub-diagnostics
# ---------------------------------------------------------------------------

def _prevalence_summary(
    sources: list[DomainDataset],
    target: DomainDataset,
    verbose: bool,
) -> list[dict]:
    rows = []
    for ds in sources:
        rows.append({
            "domain_id": ds.domain_id,
            "split": "source",
            "n": ds.n,
            "prevalence": round(ds.prevalence, 4),
            "mean_T": round(float(ds.T.mean()), 3),
            "median_T": round(float(np.median(ds.T)), 3),
            "mean_eta": round(float(ds.eta.mean()), 4),
            "std_eta": round(float(ds.eta.std()), 4),
        })
    rows.append({
        "domain_id": target.domain_id,
        "split": "target",
        "n": target.n,
        "prevalence": round(target.prevalence, 4),
        "mean_T": round(float(target.T.mean()), 3),
        "median_T": round(float(np.median(target.T)), 3),
        "mean_eta": round(float(target.eta.mean()), 4),
        "std_eta": round(float(target.eta.std()), 4),
    })
    if verbose:
        print("\n  Prevalence by domain:")
        for r in rows:
            print(f"    D{r['domain_id']} ({r['split']:7s})  "
                  f"prev={r['prevalence']:.3f}  "
                  f"mean_T={r['mean_T']:.2f}  "
                  f"mean_η={r['mean_eta']:.3f}")
    return rows


def _plot_oracle_risk(
    sources: list[DomainDataset],
    target: DomainDataset,
    scenario_name: str,
    fig_dir: Path,
    verbose: bool,
) -> None:
    """
    Overlay η distributions by domain and by label (y=0 vs y=1) to
    verify that the risk mechanism behaves as intended.
    """
    all_ds = sources + [target]
    colors_ds = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: η by domain
    for i, ds in enumerate(all_ds):
        label = f"D{ds.domain_id} ({'src' if i < len(sources) else 'tgt'})"
        axes[0].hist(ds.eta, bins=40, density=True, alpha=0.4,
                     color=colors_ds[i % len(colors_ds)], label=label)
    axes[0].set_xlabel("latent log-risk η")
    axes[0].set_ylabel("density")
    axes[0].set_title("η distribution by domain")
    axes[0].legend(fontsize=7)

    # Right: η by label within target
    mask1 = target.y == 1
    mask0 = target.y == 0
    axes[1].hist(target.eta[mask0], bins=40, density=True, alpha=0.5,
                 color="steelblue", label="target y=0")
    axes[1].hist(target.eta[mask1], bins=40, density=True, alpha=0.5,
                 color="tomato", label="target y=1")
    axes[1].set_xlabel("latent log-risk η")
    axes[1].set_title("Target: η by label")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"[{scenario_name}] Oracle log-risk diagnostics", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "oracle_risk.png", dpi=150)
    plt.close(fig)


def _extract_auroc_for_plot(
    baseline_results: list[dict],
    sources: list[DomainDataset],
    target: DomainDataset,
) -> dict:
    """
    Build {feature_set: {domain_label: auroc}} for the grouped bar plot.
    """
    out = {}
    for r in baseline_results:
        fs = r.get("feature_set", "?")
        if "error" in r:
            continue
        row = {}
        for d_key, scores in r["source_domains"].items():
            row[d_key] = scores.get("auroc", float("nan"))
        row["target"] = r["target"].get("auroc", float("nan"))
        out[fs] = row
    return out


def _compute_shift_indicators(
    sources: list[DomainDataset],
    target: DomainDataset,
    baseline_results: list[dict],
) -> dict:
    """
    Compute scalar indicators that characterise the shift regime.

    Key indicators:
      covariate_shift_score : mean Mahalanobis-like distance between source
                              pooled μ_x and target μ_x (standardised)
      prevalence_shift       : |target_prev - mean_source_prev|
      auroc_gap_all_observed : source_pooled_auroc - target_auroc for all_observed model
      auroc_gap_inv_oracle   : source_pooled_auroc - target_auroc for invariant_oracle
      spurious_degradation   : spurious_oracle src_auroc - target_auroc (how much it degrades)
    """
    # Covariate shift: normalised distance between source and target means in x
    X_src = np.vstack([ds.x for ds in sources])
    mu_src = X_src.mean(axis=0)
    std_src = X_src.std(axis=0) + 1e-8
    mu_tgt = target.x.mean(axis=0)
    cov_shift = float(np.mean(np.abs(mu_tgt - mu_src) / std_src))

    # Prevalence shift
    mean_src_prev = float(np.mean([ds.prevalence for ds in sources]))
    tgt_prev = target.prevalence
    prev_shift = abs(tgt_prev - mean_src_prev)

    # Baseline AUROC + ECE gaps
    gaps = {}
    for r in baseline_results:
        fs = r.get("feature_set", "?")
        if "error" in r:
            gaps[fs] = {
                "source_pooled_auroc": None, "target_auroc": None, "gap": None,
                "source_pooled_ece": None, "target_ece": None, "ece_gap": None,
            }
            continue
        src_auc = r["source_pooled"].get("auroc", float("nan"))
        tgt_auc = r["target"].get("auroc", float("nan"))
        src_ece = r["source_pooled"].get("ece", float("nan"))
        tgt_ece = r["target"].get("ece", float("nan"))
        gaps[fs] = {
            "source_pooled_auroc": src_auc,
            "target_auroc": tgt_auc,
            "gap": round(src_auc - tgt_auc, 4) if not (np.isnan(src_auc) or np.isnan(tgt_auc)) else None,
            "source_pooled_ece": src_ece,
            "target_ece": tgt_ece,
            # ece_gap positive = worse calibration on target
            "ece_gap": round(tgt_ece - src_ece, 4) if not (np.isnan(src_ece) or np.isnan(tgt_ece)) else None,
        }

    # η distribution shift (KL proxy: difference in means / pooled std)
    eta_src = np.concatenate([ds.eta for ds in sources])
    eta_shift = abs(target.eta.mean() - eta_src.mean()) / (eta_src.std() + 1e-8)

    return {
        "covariate_shift_score": round(cov_shift, 4),
        "prevalence_shift": round(prev_shift, 4),
        "eta_mean_shift": round(float(eta_shift), 4),
        "source_mean_prevalence": round(mean_src_prev, 4),
        "target_prevalence": round(tgt_prev, 4),
        "auroc_gaps": gaps,
    }


def _print_baseline_table(rows: list[dict]) -> None:
    print("\n  Baseline results (pooled source → target):")
    header = f"  {'feature_set':22s}  {'split':15s}  {'auroc':>6s}  {'auprc':>6s}  {'prev':>6s}"
    print(header)
    print("  " + "-" * 60)
    for r in rows:
        if "error" in r:
            continue
        if r.get("split") in ("source_pooled", "target"):
            print(
                f"  {r['feature_set']:22s}  {r['split']:15s}  "
                f"{r.get('auroc', float('nan')):6.3f}  "
                f"{r.get('auprc', float('nan')):6.3f}  "
                f"{r.get('prevalence', float('nan')):6.3f}"
            )


def _print_shift_indicators(name: str, ind: dict) -> None:
    print(f"\n  Shift indicators for [{name}]:")
    print(f"    covariate shift score : {ind['covariate_shift_score']:.4f}")
    print(f"    prevalence shift      : {ind['prevalence_shift']:.4f}  "
          f"(src={ind['source_mean_prevalence']:.3f}, tgt={ind['target_prevalence']:.3f})")
    print(f"    η mean shift (stds)   : {ind['eta_mean_shift']:.4f}")
    print("    AUROC gaps (src_pooled − target) / ECE gaps (target − src_pooled):")
    for fs, g in ind["auroc_gaps"].items():
        gap = g.get("gap")
        src = g.get("source_pooled_auroc")
        tgt = g.get("target_auroc")
        ece_gap = g.get("ece_gap")
        src_ece = g.get("source_pooled_ece")
        tgt_ece = g.get("target_ece")
        if gap is not None:
            ece_str = f"  ece: src={src_ece:.3f}, tgt={tgt_ece:.3f}, Δ={ece_gap:+.3f}" if ece_gap is not None else ""
            print(f"      {fs:22s}: auroc src={src:.3f}, tgt={tgt:.3f}, gap={gap:+.3f}{ece_str}")
