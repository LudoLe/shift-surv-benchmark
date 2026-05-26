"""
Build publication-quality compound figures for the Results section.

Figures produced:
  outputs/figures/covariate_shift_diagnosis.png
  outputs/figures/baseline_hazard_diagnosis.png
  outputs/figures/concept_shift_diagnosis.png
  outputs/figures/control_secondary.png
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from generator.scenarios import generate_scenario

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
PALETTE = {"oracle": "#2166ac", "lr": "#d6604d", "mlp": "#4dac26"}

# ---------------------------------------------------------------------------
# Helper: read multi-seed summary
# ---------------------------------------------------------------------------

def _read_summary() -> dict[str, dict]:
    path = Path("outputs/tables/multi_seed_summary.csv")
    with open(path) as f:
        return {r["scenario"]: r for r in csv.DictReader(f)}


def _auroc_bars(ax, scenario: str, summary: dict, title: str = "") -> None:
    """
    Grouped bar chart: source vs target AUROC for 3 model classes.
    Data from multi-seed summary (mean ± 95% CI).
    """
    row = summary[scenario]
    models = [
        ("inv_oracle",  "Invariant\noracle",  PALETTE["oracle"]),
        ("all_obs",     "ERM (LR)",           PALETTE["lr"]),
        ("all_obs_mlp", "ERM (MLP)",          PALETTE["mlp"]),
    ]

    x = np.arange(len(models))
    w = 0.35

    for i, (key, label, color) in enumerate(models):
        src_m  = float(row[f"{key}_src_auroc_mean"])
        src_lo = float(row[f"{key}_src_auroc_ci95_low"])
        src_hi = float(row[f"{key}_src_auroc_ci95_high"])
        tgt_m  = float(row[f"{key}_tgt_auroc_mean"])
        tgt_lo = float(row[f"{key}_tgt_auroc_ci95_low"])
        tgt_hi = float(row[f"{key}_tgt_auroc_ci95_high"])

        kw = dict(width=w * 0.88, capsize=3, edgecolor="k",
                  linewidth=0.5, alpha=0.85,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})

        ax.bar(x[i] - w / 2, src_m,
               yerr=[[src_m - src_lo], [src_hi - src_m]],
               color=color, **kw,
               label="Source" if i == 0 else None)
        ax.bar(x[i] + w / 2, tgt_m,
               yerr=[[tgt_m - tgt_lo], [tgt_hi - tgt_m]],
               color=color, hatch="//", **kw,
               label="Target" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models], fontsize=8)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.set_ylim(0.3, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Fig 2: covariate_shift_diagnosis
# ---------------------------------------------------------------------------

def build_covariate_shift(summary: dict) -> None:
    r = generate_scenario("covariate_shift", seed=SEED)
    sources, target = r["source"], r["target"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A — PCA
    ax = axes[0]
    all_x = np.vstack([ds.x for ds in sources] + [target.x])
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(all_x)
    colors_src = ["#4393c3", "#92c5de", "#d1e5f0"]
    offset = 0
    for i, ds in enumerate(sources):
        ax.scatter(z[offset:offset + ds.n, 0], z[offset:offset + ds.n, 1],
                   s=10, alpha=0.35, color=colors_src[i],
                   label=f"Source {ds.domain_id}")
        offset += ds.n
    ax.scatter(z[offset:, 0], z[offset:, 1],
               s=14, alpha=0.55, color="#d6604d", marker="s",
               edgecolors="k", linewidths=0.3, label="Target")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=9)
    ax.set_title("(A)  Observed covariate space", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B — AUROC bars
    _auroc_bars(axes[1], "covariate_shift", summary,
                title="(B)  Source vs target AUROC")

    fig.tight_layout()
    fig.savefig(OUT / "covariate_shift_diagnosis.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: covariate_shift_diagnosis.png")


# ---------------------------------------------------------------------------
# Fig 3: baseline_hazard_diagnosis
# ---------------------------------------------------------------------------

def build_baseline_hazard(summary: dict) -> None:
    r = generate_scenario("baseline_hazard_shift", seed=SEED)
    sources, target = r["source"], r["target"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A — prevalence bars
    ax = axes[0]
    all_ds = sources + [target]
    labels = [f"Src {ds.domain_id}" for ds in sources] + ["Target"]
    prevs  = [ds.prevalence for ds in all_ds]
    colors = ["#4393c3"] * len(sources) + ["#d6604d"]
    bars = ax.bar(labels, prevs, color=colors, alpha=0.85,
                  edgecolor="k", linewidth=0.5)
    for bar, p in zip(bars, prevs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{p:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("6-month event prevalence", fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_title("(A)  Event prevalence by domain", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B — reliability diagram (calibration curve)
    ax = axes[1]

    # Fit LR on pooled source
    X_src = np.vstack([ds.x for ds in sources])
    y_src = np.concatenate([ds.y for ds in sources])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_src)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=0)
        lr.fit(X_s, y_src)

    def _calibration(X, y, label, color, ls="-"):
        probs = lr.predict_proba(scaler.transform(X))[:, 1]
        bins = np.linspace(0, 1, 11)
        bin_means, obs_rates = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 5:
                bin_means.append(probs[mask].mean())
                obs_rates.append(y[mask].mean())
        ax.plot(bin_means, obs_rates, "o-", color=color,
                linestyle=ls, linewidth=1.6, markersize=4, label=label)

    X_src_all = np.vstack([ds.x for ds in sources])
    y_src_all  = np.concatenate([ds.y for ds in sources])
    _calibration(X_src_all, y_src_all, "Source (pooled)", PALETTE["lr"], "-")
    _calibration(target.x, target.y, "Target", "#d6604d", "--")

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.9, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Observed event rate", fontsize=9)
    ax.set_title("(B)  Reliability diagram — ERM (LR)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "baseline_hazard_diagnosis.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: baseline_hazard_diagnosis.png")


# ---------------------------------------------------------------------------
# Fig 4: concept_shift_diagnosis
# ---------------------------------------------------------------------------

def build_concept_shift(summary: dict) -> None:
    r = generate_scenario("concept_shift", seed=SEED)
    sources, target = r["source"], r["target"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A — eta by label, source (pooled) vs target
    ax = axes[0]
    eta_src = np.concatenate([ds.eta for ds in sources])
    y_src   = np.concatenate([ds.y  for ds in sources])

    kw = dict(bins=40, density=True, alpha=0.45)
    ax.hist(eta_src[y_src == 0], color="#4393c3", label="Source  y=0", **kw)
    ax.hist(eta_src[y_src == 1], color="#92c5de", label="Source  y=1",
            histtype="step", linewidth=1.8, **kw)
    ax.hist(target.eta[target.y == 0], color="#d6604d",
            label="Target  y=0", **kw)
    ax.hist(target.eta[target.y == 1], color="#f4a582",
            label="Target  y=1", histtype="step", linewidth=1.8, **kw)
    ax.set_xlabel("Latent log-risk  η", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("(A)  Log-risk distributions by label", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B — AUROC bars
    _auroc_bars(axes[1], "concept_shift", summary,
                title="(B)  Source vs target AUROC")

    fig.tight_layout()
    fig.savefig(OUT / "concept_shift_diagnosis.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: concept_shift_diagnosis.png")


# ---------------------------------------------------------------------------
# Fig 5: control_secondary
# ---------------------------------------------------------------------------

def build_control_secondary(summary: dict) -> None:
    r_ns  = generate_scenario("no_shift", seed=SEED)
    r_spu = generate_scenario("spurious_shift", seed=SEED)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A — no-shift PCA
    ax = axes[0]
    sources_ns, target_ns = r_ns["source"], r_ns["target"]
    all_x = np.vstack([ds.x for ds in sources_ns] + [target_ns.x])
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(all_x)
    colors_src = ["#4393c3", "#92c5de", "#d1e5f0"]
    offset = 0
    for i, ds in enumerate(sources_ns):
        ax.scatter(z[offset:offset + ds.n, 0], z[offset:offset + ds.n, 1],
                   s=10, alpha=0.35, color=colors_src[i],
                   label=f"Source {ds.domain_id}")
        offset += ds.n
    ax.scatter(z[offset:, 0], z[offset:, 1],
               s=14, alpha=0.55, color="#d6604d", marker="s",
               edgecolors="k", linewidths=0.3, label="Target")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=9)
    ax.set_title("(A)  No shift — covariate space", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B — spurious shift: AUROC gaps for 4 model types
    ax = axes[1]
    row = summary["spurious_shift"]
    models = [
        ("inv_oracle",  "Invariant\noracle",  PALETTE["oracle"]),
        ("spu_oracle",  "Spurious\noracle",   "#9970ab"),
        ("all_obs",     "ERM (LR)",           PALETTE["lr"]),
        ("all_obs_mlp", "ERM (MLP)",          PALETTE["mlp"]),
    ]
    x = np.arange(len(models))
    for i, (key, label, color) in enumerate(models):
        m  = float(row[f"{key}_gap_mean"])
        lo = float(row[f"{key}_gap_ci95_low"])
        hi = float(row[f"{key}_gap_ci95_high"])
        ax.bar(x[i], m, yerr=[[m - lo], [hi - m]],
               color=color, alpha=0.85, width=0.6, edgecolor="k",
               linewidth=0.5, capsize=4,
               error_kw={"elinewidth": 1.2, "capthick": 1.2})

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models], fontsize=8)
    ax.set_ylabel("AUROC gap  (source − target)", fontsize=9)
    ax.set_title("(B)  Spurious shortcut — AUROC gaps", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "control_secondary.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: control_secondary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = _read_summary()
    build_covariate_shift(summary)
    build_baseline_hazard(summary)
    build_concept_shift(summary)
    build_control_secondary(summary)
    print("\nAll paper figures saved to outputs/figures/")
