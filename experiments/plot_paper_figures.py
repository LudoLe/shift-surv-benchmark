"""
Paper-ready figures for the domain-grounded generative survival benchmark.

Main-text figures
-----------------
Fig 1 — Fidelity + shift: (A) baseline composite per center, (B) before/after shift
Fig 2 — Shift heatmap: which shift type best explains which center
Fig 3 — KM curves: real vs reference vs best-shift (SZMC, UOC, VHIO)
Fig 4 — Few-shot (main): aggregate learning curve + INT + UOC

Appendix figures
----------------
Fig A1 — Few-shot (full): all 6 per-center learning curves
Fig A2 — Augmentation benchmark

Usage
-----
  python -m experiments.plot_paper_figures            # all main figures
  python -m experiments.plot_paper_figures --fig 3    # one figure
  python -m experiments.plot_paper_figures --appendix # appendix only
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   8,
    "figure.dpi":        150,
}

_CENTERS = ["GHD", "INT", "MH", "SZMC", "UOC", "VHIO"]
_CENTER_CLR = {
    "GHD":  "#e41a1c",
    "INT":  "#377eb8",
    "MH":   "#4daf4a",
    "SZMC": "#984ea3",
    "UOC":  "#ff7f00",
    "VHIO": "#a65628",
}

_CLR_REAL = "#222222"
_CLR_REF  = "#7f7f7f"
_CLR_BEST = "#d62728"

_SHIFT_LABELS = {
    "lambda_shift":     "Hazard",
    "nu_shift":         "Shape (ν)",
    "mu_scale":         "Composition",
    "noise_shift":      "Noise",
    "loading_rotation": "Loading rotation",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    else:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Fidelity overview + shift improvement (2-panel combined)
# ---------------------------------------------------------------------------

def plot_fidelity_and_shift(
    tables_dir: Path,
    out_path: Path,
) -> None:
    """
    Two-panel figure combining the baseline fidelity overview with the
    shift-improvement before/after chart.

    Panel A (left): horizontal dot plot of baseline composite score per center,
      sorted ascending.  A single metric gives the reader one number per center.
    Panel B (right): before/after composite score per center with annotated
      best shift type and fractional improvement.
    """
    base_df = pd.read_csv(tables_dir / "baseline_loodo_results.csv")
    best_df = pd.read_csv(tables_dir / "best_shift_per_center.csv")

    # Shared ordering: both panels sorted by baseline composite, low → high
    # (best-fit center at top, worst at bottom in both panels)
    base_df = base_df.sort_values("composite", ascending=False)
    order = base_df["center_name"].tolist()          # top → bottom: MH … UOC
    order_cat = order[::-1]                          # ascending for Categorical
    for df_ in (base_df, best_df):
        df_["center_name"] = pd.Categorical(
            df_["center_name"], categories=order_cat, ordered=True
        )
    base_df = base_df.sort_values("center_name")
    best_df = best_df.sort_values("center_name")

    with plt.rc_context(_STYLE):
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(12, 4.4),
            gridspec_kw={"width_ratios": [1, 1.5]},
        )

        # ── Panel A: baseline composite dot chart ──────────────────────────
        for i, (_, row) in enumerate(base_df.iterrows()):
            cname = row["center_name"]
            clr   = _CENTER_CLR.get(cname, "#555555")
            ax_a.scatter(row["composite"], i, color=clr, s=70, zorder=3)
            ax_a.text(
                row["composite"] + 0.001, i, f"  n={row['n']}",
                va="center", fontsize=7.5, color="#555555",
            )

        ax_a.set_yticks(range(len(base_df)))
        ax_a.set_yticklabels(base_df["center_name"].tolist(), fontsize=9)
        ax_a.set_xlabel("Composite score (lower = better)", fontsize=9)
        ax_a.set_title("(A)  Reference generator fidelity", fontsize=9, pad=6)
        ax_a.axvline(0, color="#cccccc", lw=0.7, ls="--")
        ax_a.set_xlim(left=0)
        ax_a.grid(axis="x", lw=0.4, alpha=0.4, ls=":")

        # ── Panel B: before/after shift improvement ────────────────────────
        for i, (_, row) in enumerate(best_df.iterrows()):
            cname = row["center_name"]
            b     = row["composite_baseline"]
            best  = row["composite_best"]
            imp   = row["improvement"]
            clr   = _CENTER_CLR.get(cname, "#555555")
            shift = _SHIFT_LABELS.get(row["best_shift_type_1"], row["best_shift_type_1"])

            # Baseline dot always shown
            ax_b.scatter(b, i, color=clr, s=65, zorder=3, marker="o")

            if imp > 0.005:
                # Best-shift diamond + arrow
                ax_b.scatter(best, i, color=clr, s=60, zorder=3, marker="D")
                ax_b.annotate(
                    "", xy=(best, i), xytext=(b, i),
                    arrowprops=dict(arrowstyle="->", color=clr,
                                   lw=1.4, connectionstyle="arc3,rad=0"),
                )
                # Label centred above arrow midpoint → avoids left-edge crowding
                mid_x     = (b + best) / 2.0
                label_str = f"{shift}  {imp:+.0%}"
                ax_b.text(mid_x, i + 0.34, label_str,
                          ha="center", va="bottom", fontsize=7.5, color=clr,
                          clip_on=False)
            else:
                # VHIO: reference is already optimal — suppress "x" marker and
                # add a small inline note to the right of the baseline dot
                ax_b.text(b + 0.0025, i, "ref. best ✓",
                          ha="left", va="center", fontsize=7, color="#888888",
                          style="italic")

        ax_b.set_yticks(range(len(best_df)))
        ax_b.set_yticklabels(best_df["center_name"].tolist(), fontsize=9)
        ax_b.set_xlabel("Composite score (lower = better)", fontsize=9)
        ax_b.set_title("(B)  Best-shift improvement", fontsize=9, pad=6)
        ax_b.set_xlim(left=0)
        ax_b.axvline(0, color="#cccccc", lw=0.7, ls="--")
        # Give the right side enough room for the "ref. best" annotation
        x_max_b = max(best_df["composite_baseline"].max(),
                      best_df["composite_best"].max()) * 1.25
        ax_b.set_xlim(0, x_max_b)

        # Compact legend
        h_ref  = mlines.Line2D([], [], color="#555", marker="o", ls="none",
                               ms=6, label="Reference")
        h_best = mlines.Line2D([], [], color="#555", marker="D", ls="none",
                               ms=6, label="Best shift")
        ax_b.legend(handles=[h_ref, h_best], loc="lower right", fontsize=7.5)

        fig.suptitle(
            "Generator fidelity and shift-calibration improvement across six centers",
            fontsize=10, y=1.01,
        )
        fig.tight_layout()
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2 — Shift heatmap (colorbar and NaN fixes)
# ---------------------------------------------------------------------------

def plot_shift_heatmap(
    tables_dir: Path,
    out_path: Path,
) -> None:
    """
    Heatmap: rows = centers, columns = shift types.
    Values = fractional improvement in composite over baseline for the best
    magnitude of each shift type.

    Fixes vs. original:
    - vmax capped at 0.50 so that moderate improvements are readable; cells
      with raw improvement > 0.50 are annotated with '*'.
    - NaN cells (e.g. VHIO where the shift search returned no valid score)
      are shown in light grey with 'N/A' rather than silently mapped to zero.
    """
    sing_df = pd.read_csv(tables_dir / "singleton_shift_results.csv")
    base_df = pd.read_csv(tables_dir / "baseline_loodo_results.csv")

    base_map = dict(zip(base_df["center_name"], base_df["composite"]))
    shift_types = list(_SHIFT_LABELS.keys())
    centers     = [c for c in _CENTERS if c in sing_df["center_name"].values]

    # Build matrix; track which cells are genuinely missing vs computed-zero
    mat      = np.full((len(centers), len(shift_types)), np.nan)
    raw_vals = np.full((len(centers), len(shift_types)), np.nan)  # uncapped

    for ci, cname in enumerate(centers):
        b   = base_map.get(cname, np.nan)
        sub = sing_df[sing_df["center_name"] == cname]
        for si, st in enumerate(shift_types):
            st_sub = sub[sub["shift_type"] == st]
            if st_sub.empty or np.isnan(b) or b == 0:
                continue  # leave as NaN
            best_comp = float(st_sub["composite"].dropna().min())
            if np.isnan(best_comp):
                continue
            raw_imp = (b - best_comp) / b
            raw_vals[ci, si] = raw_imp
            mat[ci, si] = raw_imp  # will be colour-clipped below

    # Colour scale: cap at ±0.50 so moderate values are distinguishable
    VMAX = 0.50
    mat_clipped = np.clip(mat, -VMAX, VMAX)

    # For imshow, replace NaN with a sentinel outside the colour range so we
    # can draw them separately
    mat_display = np.where(np.isnan(mat_clipped), 0.0, mat_clipped)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))

        im = ax.imshow(mat_display, aspect="auto", cmap="RdYlGn",
                       vmin=-VMAX, vmax=VMAX)

        # White cell borders via minor ticks grid
        ax.set_xticks(np.arange(-0.5, len(shift_types), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(centers), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.8, zorder=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Overlay distinct grey patches for N/A cells (on top of grid)
        for ci in range(len(centers)):
            for si in range(len(shift_types)):
                if np.isnan(mat[ci, si]):
                    ax.add_patch(plt.Rectangle(
                        (si - 0.5, ci - 0.5), 1, 1,
                        facecolor="#d8d8d8", edgecolor="white", lw=1.8,
                        zorder=3,
                    ))

        ax.set_xticks(range(len(shift_types)))
        ax.set_xticklabels([_SHIFT_LABELS[s] for s in shift_types],
                           rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(centers)))
        ax.set_yticklabels(centers, fontsize=9)

        # Cell annotations (on top of patches)
        for ci in range(len(centers)):
            for si in range(len(shift_types)):
                rv = raw_vals[ci, si]
                if np.isnan(rv):
                    ax.text(si, ci, "N/A", ha="center", va="center",
                            fontsize=7, color="#888888", zorder=4,
                            style="italic")
                elif abs(rv) < 0.01:
                    ax.text(si, ci, "—", ha="center", va="center",
                            fontsize=8, color="#666666", zorder=4)
                else:
                    star = "*" if abs(rv) > VMAX else ""
                    txt  = f"{rv:+.0%}{star}"
                    dark = abs(mat_clipped[ci, si]) > VMAX * 0.65
                    ax.text(si, ci, txt, ha="center", va="center",
                            fontsize=7.5, zorder=4,
                            color="white" if dark else "#111111",
                            fontweight="bold" if dark else "normal")

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Fractional improvement\n(+ = better fit,  * = capped at ±50%)",
                       fontsize=7.5)
        cbar.ax.tick_params(labelsize=7.5)
        # Explicit ticks at ±25%, ±50%
        cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
        cbar.set_ticklabels(["-50%", "-25%", "0%", "+25%", "+50%"])

        ax.set_title("Best singleton shift improvement per center × shift type",
                     fontsize=10, pad=10)
        fig.tight_layout()
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3 — KM curves (new center selection + x-axis clip)
# ---------------------------------------------------------------------------

def plot_km_curves(
    tables_dir: Path,
    out_path: Path,
    centers_to_show: list[str] | None = None,
    tau: float = 12.0,
) -> None:
    """
    KM survival curves: real (solid), reference synthetic (dashed),
    best-shift synthetic (dash-dot) for selected centers.

    Default centres: SZMC, UOC, VHIO.
    - SZMC and UOC have the two largest baseline KM distances and show
      visible improvement from shift calibration.
    - VHIO is already well-reproduced by the reference generator (no shift
      needed), providing a deliberate contrast.

    X-axis fix: clipped to real data max × 1.05 to suppress the very long
    synthetic Weibull tails that previously pushed the MH/GHD axes to
    300–600 months.
    """
    if centers_to_show is None:
        centers_to_show = ["SZMC", "UOC", "VHIO"]

    km_data = json.loads((tables_dir / "km_curves.json").read_text())

    n = len(centers_to_show)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), sharey=True)
        if n == 1:
            axes = [axes]

        def _plot_km(ax, entry, color, ls, lw, label):
            t = [0.0] + entry["times"]
            s = [1.0] + entry["S"]
            ax.step(t, s, where="post", color=color, ls=ls, lw=lw, label=label)

        for ax, cname in zip(axes, centers_to_show):
            if cname not in km_data:
                ax.set_title(f"{cname} (no data)")
                continue

            cd = km_data[cname]
            real_max = max(cd["real"]["times"])

            _plot_km(ax, cd["real"],      _CLR_REAL, "-",  1.8, "Real")
            _plot_km(ax, cd["reference"], _CLR_REF,  "--", 1.4, "Reference")
            if "best_pair" in cd:
                _plot_km(ax, cd["best_pair"],      _CLR_BEST, "-.", 1.4, "Best shift (pair)")
            elif "best_singleton" in cd:
                _plot_km(ax, cd["best_singleton"], _CLR_BEST, "-.", 1.4, "Best shift")

            ax.axvline(tau, color="#aaaaaa", lw=0.8, ls=":", label=f"τ={tau:.0f} mo")

            # Clip x-axis to real data range — prevents Weibull tail artifacts
            ax.set_xlim(0, real_max * 1.05)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Months", fontsize=9)
            ax.set_title(cname, fontsize=10, fontweight="bold")
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":")

        axes[0].set_ylabel("Survival probability S(t)", fontsize=9)
        axes[-1].legend(loc="upper right", fontsize=7.5)

        fig.suptitle(
            "KM survival curves: real vs. synthetic (reference and best shift)",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4 — Few-shot main (aggregate + INT + UOC, 3 panels)
# ---------------------------------------------------------------------------

def plot_fewshot_main(
    tables_dir: Path,
    out_path: Path,
) -> None:
    """
    Three-panel main-text few-shot figure.

    Panel A (left): aggregate mean composite across all 6 centers vs m.
      Shows the overall learning curve with ±1 std band and marks the
      Variant-B crossover at m ≈ 20.
    Panel B (centre): INT — fastest convergence, closest to upper bound.
    Panel C (right): UOC — slowest convergence, hardest center.

    Together, the three panels show that the aggregate finding holds even
    in the most challenging case.
    """
    results = pd.read_csv(tables_dir / "fewshot_results.csv")

    _VAR = {
        "A": {"color": "#377eb8", "ls": "-",  "marker": "o",
              "label": "Variant A (covariate-only)"},
        "B": {"color": "#e41a1c", "ls": "--", "marker": "s",
              "label": "Variant B (full few-shot)"},
    }

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

        panels = [("Aggregate (6 centers)", None),
                  ("INT", "INT"),
                  ("UOC", "UOC")]

        for ax, (title, center) in zip(axes, panels):
            if center is None:
                sub = results[results["variant"].isin(["A", "B", "C"])]
            else:
                sub = results[results["center"] == center]

            # Variant C upper-bound line
            c_vals = sub[sub["variant"] == "C"]["composite"]
            if not c_vals.empty:
                c_val = float(c_vals.mean())
                ax.axhline(c_val, color="#4daf4a", ls=":", lw=1.4, alpha=0.9,
                           label="Variant C (full info)")

            for var, style in _VAR.items():
                vsub = sub[sub["variant"] == var]
                if vsub.empty:
                    continue
                grp  = vsub.groupby("m")["composite"].agg(["mean", "std"]).reset_index()
                m_v  = grp["m"].values
                mn   = grp["mean"].values
                sd   = grp["std"].fillna(0).values

                ax.plot(m_v, mn, color=style["color"], ls=style["ls"],
                        lw=1.7, marker=style["marker"], ms=4,
                        label=style["label"])
                ax.fill_between(m_v, mn - sd, mn + sd,
                                color=style["color"], alpha=0.13)

            # Mark B-crosses-A: first m where B mean < A mean
            grp_A = (sub[sub["variant"] == "A"]
                     .groupby("m")["composite"].mean().reset_index()
                     .rename(columns={"composite": "A"}))
            grp_B = (sub[sub["variant"] == "B"]
                     .groupby("m")["composite"].mean().reset_index()
                     .rename(columns={"composite": "B"}))
            merged = grp_A.merge(grp_B, on="m")
            cross  = merged[merged["B"] < merged["A"]]
            if not cross.empty:
                m_cross = int(cross.iloc[0]["m"])
                ax.axvline(m_cross, color="#888888", lw=0.8, ls="--")
                ax.text(m_cross + 1, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.14,
                        f"B < A\n(m={m_cross})", fontsize=6.5, color="#555555", va="top")

            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel("Calibration patients m", fontsize=8.5)
            ax.set_ylabel("Composite score\n(lower = better)", fontsize=8.5)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":")

        axes[0].legend(loc="upper right", fontsize=7.5, ncol=1)

        fig.suptitle(
            "Few-shot calibration: composite score vs. held-out patients used",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Appendix A1 — Few-shot (all 6 centers)
# ---------------------------------------------------------------------------

def plot_fewshot_curves(
    tables_dir: Path,
    out_path: Path,
) -> None:
    """All 6 per-center few-shot learning curves (appendix)."""
    results = pd.read_csv(tables_dir / "fewshot_results.csv")
    centers = [c for c in _CENTERS if c in results["center"].values]

    _VAR_STYLE = {
        "A": {"color": "#377eb8", "ls": "-",  "label": "Variant A (covariate-only)"},
        "B": {"color": "#e41a1c", "ls": "--", "label": "Variant B (full few-shot)"},
        "C": {"color": "#4daf4a", "ls": ":",  "label": "Variant C (full info)"},
    }

    ncols = 3
    nrows = (len(centers) + ncols - 1) // ncols

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 sharey=False)
        axes_flat = axes.flatten()

        for ax, cname in zip(axes_flat, centers):
            sub = results[results["center"] == cname]

            c_row = sub[sub["variant"] == "C"]
            if not c_row.empty:
                c_val = float(c_row["composite"].iloc[0])
                ax.axhline(c_val,
                           **{k: v for k, v in _VAR_STYLE["C"].items() if k != "label"},
                           lw=1.4, alpha=0.8, label=_VAR_STYLE["C"]["label"])

            for var in ("A", "B"):
                vsub = sub[sub["variant"] == var]
                if vsub.empty:
                    continue
                grp   = vsub.groupby("m")["composite"].agg(["mean", "std"]).reset_index()
                style = _VAR_STYLE[var]
                ax.plot(grp["m"], grp["mean"], color=style["color"],
                        ls=style["ls"], lw=1.6, marker="o", ms=4,
                        label=style["label"])
                ax.fill_between(grp["m"],
                                grp["mean"] - grp["std"].fillna(0),
                                grp["mean"] + grp["std"].fillna(0),
                                color=style["color"], alpha=0.15)

            ax.set_title(cname, fontsize=10, fontweight="bold")
            ax.set_xlabel("Calibration set size m", fontsize=8.5)
            ax.set_ylabel("Composite score\n(lower = better)", fontsize=8.5)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":")

        for ax in axes_flat[len(centers):]:
            ax.set_visible(False)

        axes_flat[0].legend(loc="upper right", fontsize=7.5, ncol=1)
        fig.suptitle("Few-shot calibration: all centers (appendix)",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Appendix A2 — Augmentation benchmark
# ---------------------------------------------------------------------------

def plot_augmentation_results(
    tables_dir: Path,
    out_path: Path,
) -> None:
    """
    Two-panel augmentation appendix figure.

    Panel A: aggregate AUROC learning curves (S+real vs S+real+Bsynth).
    Panel B: Δ AUROC vs. generator composite score at m=20 per center,
      with OLS regression line to make the fidelity-predicts-benefit claim
      visually explicit.
    """
    df  = pd.read_csv(tables_dir / "augmentation_results.csv")
    bl  = pd.read_csv(tables_dir / "baseline_loodo_results.csv")

    centers   = [c for c in _CENTERS if c in df["center"].values]
    composite = dict(zip(bl["center_name"], bl["composite"]))

    _VAR_AUG = {
        "S+real":        {"color": "#377eb8", "ls": "-",  "label": "S + real(m)"},
        "S+real+Bsynth": {"color": "#e41a1c", "ls": "--", "label": "S + real(m) + synth"},
    }

    with plt.rc_context(_STYLE):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))

        # Panel A — aggregate learning curves
        for var, style in _VAR_AUG.items():
            sub = df[df["variant"] == var]
            grp = sub.groupby("m")["auroc"].agg(["mean", "std"]).reset_index()
            ax_left.plot(grp["m"], grp["mean"], color=style["color"],
                         ls=style["ls"], lw=1.8, marker="o", ms=4,
                         label=style["label"])
            ax_left.fill_between(grp["m"],
                                 grp["mean"] - grp["std"],
                                 grp["mean"] + grp["std"],
                                 color=style["color"], alpha=0.12)

        s_val = float(df[df["variant"] == "S"]["auroc"].mean())
        ax_left.axhline(s_val, color="#888888", lw=1.0, ls=":",
                        label=f"S only ({s_val:.3f})")
        ax_left.set_xlabel("Calibration patients m", fontsize=9)
        ax_left.set_ylabel("Mean AUROC (12-month)", fontsize=9)
        ax_left.set_title("(A)  Augmentation vs real-only training\n(mean across 6 centers)",
                          fontsize=9)
        ax_left.legend(fontsize=7.5)
        ax_left.grid(axis="y", lw=0.4, alpha=0.4, ls=":")

        # Panel B — Δ AUROC vs composite, with regression line
        m_ref = 20
        xs, ys = [], []
        for cname in centers:
            real_m = df[(df["center"] == cname) & (df["variant"] == "S+real")
                        & (df["m"] == m_ref)]["auroc"].mean()
            comb_m = df[(df["center"] == cname) & (df["variant"] == "S+real+Bsynth")
                        & (df["m"] == m_ref)]["auroc"].mean()
            if np.isnan(real_m) or np.isnan(comb_m):
                continue
            delta_a = comb_m - real_m
            comp    = composite.get(cname, np.nan)
            if np.isnan(comp):
                continue
            clr = _CENTER_CLR.get(cname, "#555555")
            ax_right.scatter(comp, delta_a, color=clr, s=70, zorder=3)
            ax_right.annotate(cname, (comp, delta_a),
                              textcoords="offset points", xytext=(5, 3),
                              fontsize=7.5, color=clr)
            xs.append(comp)
            ys.append(delta_a)

        # OLS regression line
        if len(xs) >= 2:
            xs_a, ys_a = np.array(xs), np.array(ys)
            m_reg, b_reg = np.polyfit(xs_a, ys_a, 1)
            x_line = np.linspace(min(xs_a) * 0.9, max(xs_a) * 1.05, 50)
            ax_right.plot(x_line, m_reg * x_line + b_reg,
                          color="#555555", lw=1.0, ls="--", alpha=0.6,
                          label=f"OLS (slope={m_reg:+.2f})")
            ax_right.legend(fontsize=7.5)

        ax_right.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
        ax_right.set_xlabel("Generator composite score (lower = better fit)", fontsize=9)
        ax_right.set_ylabel(f"Δ AUROC [+synth − real(m={m_ref})]", fontsize=9)
        ax_right.set_title(
            f"(B)  Augmentation benefit vs generator fidelity (m={m_ref})",
            fontsize=9,
        )
        ax_right.grid(lw=0.4, alpha=0.4, ls=":")

        fig.suptitle("Synthetic augmentation: downstream 12-month prediction (appendix)",
                     fontsize=10, y=1.02)
        fig.tight_layout()
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_all_figures(
    tables_dir: str = "outputs/tables/",
    figures_dir: str = "outputs/figures/paper/",
    tau: float = 12.0,
    include_appendix: bool = True,
) -> None:
    t = Path(tables_dir)
    f = Path(figures_dir)

    print("Generating main-text figures...")
    plot_fidelity_and_shift(t,  f / "fig1_fidelity_and_shift.pdf")
    plot_shift_heatmap(t,       f / "fig2_shift_heatmap.pdf")
    plot_km_curves(t,           f / "fig3_km_curves.pdf", tau=tau)
    if (t / "fewshot_results.csv").exists():
        plot_fewshot_main(t,    f / "fig4_fewshot_main.pdf")

    if include_appendix:
        print("Generating appendix figures...")
        if (t / "fewshot_results.csv").exists():
            plot_fewshot_curves(t,         f / "figA1_fewshot_all.pdf")
        if (t / "augmentation_results.csv").exists():
            plot_augmentation_results(t,   f / "figA2_augmentation.pdf")

    print("Done.")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=str, default=None,
                   help="generate one figure: 1, 2, 3, 4, A1, A2")
    p.add_argument("--appendix",    action="store_true",
                   help="generate appendix figures only")
    p.add_argument("--tables_dir",  default="outputs/tables/")
    p.add_argument("--figures_dir", default="outputs/figures/paper/")
    p.add_argument("--tau",         type=float, default=12.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t = Path(args.tables_dir)
    f = Path(args.figures_dir)

    dispatch = {
        "1":  lambda: plot_fidelity_and_shift(t, f / "fig1_fidelity_and_shift.pdf"),
        "2":  lambda: plot_shift_heatmap(t,      f / "fig2_shift_heatmap.pdf"),
        "3":  lambda: plot_km_curves(t,          f / "fig3_km_curves.pdf", tau=args.tau),
        "4":  lambda: plot_fewshot_main(t,       f / "fig4_fewshot_main.pdf"),
        "A1": lambda: plot_fewshot_curves(t,     f / "figA1_fewshot_all.pdf"),
        "A2": lambda: plot_augmentation_results(t, f / "figA2_augmentation.pdf"),
    }

    if args.appendix:
        dispatch["A1"]()
        dispatch["A2"]()
    elif args.fig is not None:
        if args.fig not in dispatch:
            raise ValueError(f"--fig must be one of {list(dispatch)}, got {args.fig!r}")
        dispatch[args.fig]()
    else:
        make_all_figures(args.tables_dir, args.figures_dir, tau=args.tau)
