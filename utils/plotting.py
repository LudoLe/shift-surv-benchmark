"""
Diagnostic plotting utilities.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats as sp_stats
from sklearn.decomposition import PCA

from generator.base_generator import DomainDataset


PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _domain_colors(n: int):
    cmap = cm.get_cmap("tab10")
    return [cmap(i) for i in range(n)]


# ---------------------------------------------------------------------------
# PCA / covariate distribution
# ---------------------------------------------------------------------------

def plot_pca_by_domain(
    datasets: list[DomainDataset],
    target: DomainDataset,
    title: str = "",
    save_path: Path | None = None,
    label_prefix: str = "Domain",
) -> plt.Figure:
    """
    PCA of observed covariates x coloured by domain.
    Source domains are filled circles; target is open squares.
    """
    all_x = np.vstack([ds.x for ds in datasets] + [target.x])
    pca = PCA(n_components=2, random_state=0)
    all_z = pca.fit_transform(all_x)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = _domain_colors(len(datasets) + 1)

    offset = 0
    for i, ds in enumerate(datasets):
        n = ds.n
        ax.scatter(
            all_z[offset : offset + n, 0],
            all_z[offset : offset + n, 1],
            s=15, alpha=0.4,
            color=colors[i],
            label=f"{label_prefix} {ds.domain_id} (source)",
        )
        offset += n

    n_t = target.n
    ax.scatter(
        all_z[offset : offset + n_t, 0],
        all_z[offset : offset + n_t, 1],
        s=25, alpha=0.5,
        marker="s",
        color=colors[len(datasets)],
        label=f"{label_prefix} {target.domain_id} (target)",
        edgecolors="k", linewidths=0.3,
    )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(title or "PCA of observed covariates")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_latent_distributions(
    datasets: list[DomainDataset],
    target: DomainDataset,
    factor: str = "inv",  # "inv" or "spu"
    title: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Histogram of the first latent factor dimension (invariant or spurious)
    per domain, overlaid.
    """
    attr = "u_inv" if factor == "inv" else "u_spu"
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = _domain_colors(len(datasets) + 1)

    for i, ds in enumerate(datasets):
        vals = getattr(ds, attr)[:, 0]
        ax.hist(vals, bins=40, density=True, alpha=0.35, color=colors[i],
                label=f"Domain {ds.domain_id} (source)")

    vals_t = getattr(target, attr)[:, 0]
    ax.hist(vals_t, bins=40, density=True, alpha=0.5, color=colors[len(datasets)],
            histtype="step", linewidth=2, label=f"Domain {target.domain_id} (target)")

    ax.set_xlabel(f"u^({factor})[0]")
    ax.set_ylabel("density")
    ax.set_title(title or f"Distribution of first {factor} latent factor")
    ax.legend(fontsize=7)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_event_time_distribution(
    datasets: list[DomainDataset],
    target: DomainDataset,
    tau: float = 6.0,
    title: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Histogram of latent event times T per domain with τ marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = _domain_colors(len(datasets) + 1)
    clip = tau * 4

    for i, ds in enumerate(datasets):
        t_clip = np.clip(ds.T, 0, clip)
        ax.hist(t_clip, bins=50, density=True, alpha=0.35, color=colors[i],
                label=f"Domain {ds.domain_id} (src, prev={ds.prevalence:.2f})")

    t_clip = np.clip(target.T, 0, clip)
    ax.hist(t_clip, bins=50, density=True, alpha=0.5, color=colors[len(datasets)],
            histtype="step", linewidth=2,
            label=f"Domain {target.domain_id} (tgt, prev={target.prevalence:.2f})")

    ax.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=f"τ={tau}")
    ax.set_xlabel("Event time T")
    ax.set_ylabel("density")
    ax.set_title(title or "Event time distributions")
    ax.legend(fontsize=7)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_prevalence_bar(
    datasets: list[DomainDataset],
    target: DomainDataset,
    title: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of 6-month event prevalence per domain."""
    all_ds = datasets + [target]
    labels = [f"D{ds.domain_id}\n(src)" for ds in datasets] + [f"D{target.domain_id}\n(tgt)"]
    prevs = [ds.prevalence for ds in all_ds]
    colors = _domain_colors(len(all_ds))
    colors[-1] = "tomato"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, prevs, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.set_ylabel("6-month event prevalence")
    ax.set_ylim(0, min(1.0, max(prevs) * 1.3))
    ax.set_title(title or "Prevalence by domain")
    for bar, p in zip(bars, prevs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{p:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_auroc_comparison(
    results: dict[str, dict],  # {model_name: {domain_label: auroc_value}}
    title: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing AUROC across model types (invariant, spurious,
    all) on source and target domains.
    """
    model_names = list(results.keys())
    domain_labels = list(next(iter(results.values())).keys())
    x = np.arange(len(domain_labels))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = _domain_colors(len(model_names))

    for i, (name, aucs) in enumerate(results.items()):
        vals = [aucs[d] for d in domain_labels]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=name,
               color=colors[i], alpha=0.8, edgecolor="k", linewidth=0.4)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "AUROC by model and domain")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_failure_modes_summary(
    summary_csv: "Path",
    save_path: "Path | None" = None,
) -> "plt.Figure":
    """
    Two-panel forest-plot figure for the paper.

    Left panel  : AUROC gap (source − target)
    Right panel : ECE gap   (target − source)

    Each panel shows one dot + 95% CI per model × scenario.
    Scenarios are arranged on the y-axis, ordered by failure mode group
    (mechanism shift, calibration drift, spurious, observation-driven, control).
    Coloured background bands identify each failure mode group.
    """
    import csv as _csv

    with open(summary_csv) as f:
        rows = {r["scenario"]: r for r in _csv.DictReader(f)}

    # --- scenario order (bottom → top in the plot, so top = most prominent) ---
    # (csv_key, display_label, failure_mode_group)
    scenario_order = [
        ("no_shift",              "No shift",              "Control"),
        ("spurious_shift",        "Spurious shortcut",     "Spurious"),
        ("covariate_shift",       "Covariate shift",       "Observation-driven"),
        ("baseline_hazard_shift", "Baseline hazard shift", "Calibration drift"),
        ("concept_shift",         "Concept shift",         "Mechanism shift"),
    ]

    # --- model specs ---
    models = [
        ("inv_oracle",  "Invariant oracle", "#2166ac"),
        ("all_obs",     "ERM (LR)",         "#d6604d"),
        ("all_obs_mlp", "ERM (MLP)",        "#4dac26"),
    ]

    # --- metrics ---
    metrics = [
        ("_gap",     "AUROC gap  (source \u2212 target)", "AUROC degradation"),
        ("_ece_gap", "ECE gap  (target \u2212 source)",   "Calibration degradation"),
    ]

    # --- y positions: small gap between failure mode groups ---
    y_positions = {}
    y = 0.0
    prev_fm = None
    for sc_key, _, fm in scenario_order:
        if prev_fm is not None and fm != prev_fm:
            y += 0.5
        y_positions[sc_key] = y
        y += 1.2
        prev_fm = fm

    model_offsets = np.linspace(-0.28, 0.28, len(models))

    # --- failure mode background colours ---
    fm_colors = {
        "Mechanism shift":    "#f8d7da",  # soft red
        "Calibration drift":  "#d4edda",  # soft green
        "Spurious":           "#e2e3e5",  # light grey
        "Observation-driven": "#cce5ff",  # soft blue
        "Control":            "#f8f9fa",  # near white
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, (metric_suffix, xlabel, panel_title) in zip(axes, metrics):

        # Background bands per failure mode group
        fm_ys = {}
        for sc_key, _, fm in scenario_order:
            fm_ys.setdefault(fm, []).append(y_positions[sc_key])
        for fm, ys in fm_ys.items():
            ax.axhspan(
                min(ys) - 0.55, max(ys) + 0.55,
                color=fm_colors.get(fm, "white"), alpha=0.35, zorder=0,
            )

        # Zero reference line
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9, zorder=1)

        for b_idx, (model_key, model_label, color) in enumerate(models):
            xs, ys_plot, xerr_lo, xerr_hi = [], [], [], []
            for sc_key, _, _ in scenario_order:
                row = rows.get(sc_key, {})
                mean = float(row.get(f"{model_key}{metric_suffix}_mean", "nan"))
                lo   = float(row.get(f"{model_key}{metric_suffix}_ci95_low",  "nan"))
                hi   = float(row.get(f"{model_key}{metric_suffix}_ci95_high", "nan"))
                xs.append(mean)
                ys_plot.append(y_positions[sc_key] + model_offsets[b_idx])
                xerr_lo.append(mean - lo)
                xerr_hi.append(hi - mean)

            ax.errorbar(
                xs, ys_plot,
                xerr=[xerr_lo, xerr_hi],
                fmt="o", color=color,
                markersize=5.5, capsize=3,
                linewidth=0, elinewidth=1.4, capthick=1.4,
                label=model_label, zorder=3,
            )

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)

    # y-axis tick labels (left panel only — shared axis)
    ytick_pos    = [y_positions[sc] for sc, _, _ in scenario_order]
    ytick_labels = [label for _, label, _ in scenario_order]
    axes[0].set_yticks(ytick_pos)
    axes[0].set_yticklabels(ytick_labels, fontsize=9)
    axes[0].set_ylabel("")

    # Failure mode annotations on the right side of the right panel
    from matplotlib.transforms import blended_transform_factory
    ax_r = axes[1]
    fm_ys_right = {}
    for sc_key, _, fm in scenario_order:
        fm_ys_right.setdefault(fm, []).append(y_positions[sc_key])
    trans = blended_transform_factory(ax_r.transAxes, ax_r.transData)
    for fm, ys in fm_ys_right.items():
        mid_y = (min(ys) + max(ys)) / 2
        ax_r.text(
            1.03, mid_y, fm,
            transform=trans,
            fontsize=7.5, color="#444444",
            va="center", ha="left",
            clip_on=False,
        )

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        fontsize=8, loc="lower center",
        ncol=3, framealpha=0.85, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.06),
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multi_seed_gaps(
    raw: dict[str, list[dict]],
    scenarios: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Grouped bar chart of mean AUROC gaps (src_pooled − target) across scenarios
    with 95% CI error bars (t-distribution, ddof=1).

    One group per scenario; three bars per group:
      invariant_oracle / spurious_oracle / all_observed.
    """
    gap_keys = {
        "inv_oracle": "inv_oracle_gap",
        "spu_oracle": "spu_oracle_gap",
        "all_obs":    "all_obs_gap",
    }
    colors = _domain_colors(len(gap_keys))

    x = np.arange(len(scenarios))
    n_bars = len(gap_keys)
    width = 0.7 / n_bars

    fig, ax = plt.subplots(figsize=(10, 5))

    for b_idx, (label, key) in enumerate(gap_keys.items()):
        means, ci_halfs = [], []
        for name in scenarios:
            vals = np.array([r[key] for r in raw[name]], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                means.append(float("nan"))
                ci_halfs.append(0.0)
                continue
            mean = float(vals.mean())
            if len(vals) > 1:
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                ci_half = sp_stats.t.ppf(0.975, df=len(vals) - 1) * se
            else:
                ci_half = 0.0
            means.append(mean)
            ci_halfs.append(ci_half)

        offset = (b_idx - n_bars / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width * 0.9,
            yerr=ci_halfs, capsize=3,
            label=label, color=colors[b_idx], alpha=0.8,
            edgecolor="k", linewidth=0.4,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("AUROC gap  (src_pooled − target)")
    ax.set_title("AUROC gaps across scenarios — mean ± 95% CI")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
