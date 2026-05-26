"""
Shift-informed test-time domain adaptation experiment.

Idea
----
The shift calibration experiment identified a parametric shift (primarily loading
rotation) that brings the synthetic held-out domain closest to the real one.
Instead of using this for data augmentation (which breaks feature-outcome
consistency when mixed with source data), we apply the INVERSE shift as a
TEST-TIME feature alignment: target features are de-rotated back into the
source feature space before the source-trained classifier sees them.

Alignment formula (loading rotation by angle θ)
------------------------------------------------
If x_tgt was generated as  x_tgt ≈ μ_tgt + A @ R(θ) @ z,
then the source-space equivalent is:
    x_aligned = μ_src + A @ R(-θ) @ A⁺ @ (x_tgt - μ_tgt)

where A⁺ is the Moore-Penrose pseudoinverse of the latent loading matrix A.
For non-rotation shifts (noise, mu_scale) only mean alignment is applied.

Variants
--------
  S                   source model, raw target features  [baseline]
  S_adapt_oracle      source model + oracle rotation (uses full held-out to
                      estimate μ_tgt and known best θ)    [upper bound]
  S_adapt_fs(m)       source model + θ estimated from m patients via grid
                      search maximising source-model AUROC on aligned X_m
  S_ft_real(m)        fine-tune source model on m raw target patients
  S_ft_aligned(m)     fine-tune on m de-rotated target patients,
                      evaluate on de-rotated full target  [consistent pipeline]
  S+real(full)        fine-tune on all target patients    [oracle upper bound]

Key comparisons
---------------
  S_adapt_oracle  vs S            — ceiling of shift-based adaptation
  S_adapt_fs(m)   vs S            — practical few-shot adaptation benefit
  S_adapt_fs(m)   vs S_ft_real(m) — shift adaptation vs supervised fine-tuning
  S_ft_aligned(m) vs S_ft_real(m) — consistent alignment on top of fine-tuning

Usage
-----
  python -m experiments.run_shift_adaptation_experiment [options]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import warnings

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median

from generator.parameters import BaseParams
from generator.fitting import fit_reference_generator
from generator.shift_components import SINGLETON_GRIDS
from experiments.compare_domains import compute_mmd


# ---------------------------------------------------------------------------
# Downstream model helpers  (same as augmentation experiment)
# ---------------------------------------------------------------------------

def _binary_label(t: np.ndarray, delta: np.ndarray, tau: float) -> np.ndarray:
    return ((t <= tau) & (delta == 1)).astype(int)


def _fit_lr(X: np.ndarray, y: np.ndarray, seed: int = 0):
    if len(np.unique(y)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                               random_state=seed, class_weight="balanced")
        m.fit(X, y)
    return m


def _fit_mlp(X: np.ndarray, y: np.ndarray, seed: int = 0):
    if len(np.unique(y)) < 2:
        return None
    early = len(y) >= 20
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          alpha=1e-3, max_iter=500, early_stopping=early,
                          validation_fraction=0.15, n_iter_no_change=15,
                          random_state=seed)
        m.fit(X, y)
    return m


def _auroc(model, X: np.ndarray, y: np.ndarray) -> float:
    if model is None or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


_MODELS = {"lr": _fit_lr, "mlp": _fit_mlp}


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def _givens_inv(theta: float, k: int) -> np.ndarray:
    """R(-θ) in the (0,1) column plane of k-dimensional space."""
    R = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    R[0, 0], R[0, 1] =  c,  s   # note: R(-θ) has +s in [0,1]
    R[1, 0], R[1, 1] = -s,  c
    return R


def align_features(
    X: np.ndarray,
    A: np.ndarray,
    mu_tgt: np.ndarray,
    mu_src: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    De-rotate target features by -θ to align with source feature space.

    x_aligned = μ_src + A @ R(-θ) @ A⁺ @ (X - μ_tgt)

    Parameters
    ----------
    X      : (n, p) target features in standardised space
    A      : (p, k) loading matrix from the reference generator
    mu_tgt : (p,)   target feature mean (empirical or few-shot estimate)
    mu_src : (p,)   source feature mean
    theta  : forward rotation angle that was identified for this domain;
             we apply the inverse (-theta) here
    """
    X_c = X - mu_tgt                          # (n, p)  centre on target mean
    if theta == 0.0:
        # Only mean shift — no rotation to invert
        return mu_src + X_c
    k   = A.shape[1]
    R_i = _givens_inv(theta, k)               # (k, k)  inverse rotation
    Ap  = np.linalg.pinv(A)                   # (k, p)  pseudoinverse
    M   = A @ R_i @ Ap                        # (p, p)  alignment projection
    return mu_src + (M @ X_c.T).T            # (n, p)


def estimate_theta_fewshot(
    X_m: np.ndarray,
    y_m: np.ndarray,
    source_model,
    A: np.ndarray,
    mu_tgt_m: np.ndarray,
    mu_src: np.ndarray,
    angle_grid: list[float],
) -> tuple[float, float]:
    """
    Find θ that maximises source-model AUROC on de-rotated few-shot samples.

    Returns (best_theta, best_auroc).
    Falls back to (0.0, NaN) if y_m is degenerate or source_model is None.
    """
    if source_model is None or len(np.unique(y_m)) < 2:
        return 0.0, float("nan")

    best_theta, best_auroc = 0.0, -1.0
    for theta in angle_grid:
        X_al = align_features(X_m, A, mu_tgt_m, mu_src, theta)
        try:
            probs = source_model.predict_proba(X_al)[:, 1]
            auc   = float(roc_auc_score(y_m, probs))
        except Exception:
            continue
        if auc > best_auroc:
            best_auroc, best_theta = auc, theta

    return best_theta, best_auroc


def estimate_theta_mmd(
    X_m: np.ndarray,
    X_src: np.ndarray,
    A: np.ndarray,
    mu_tgt_m: np.ndarray,
    mu_src: np.ndarray,
    angle_grid: list[float],
) -> tuple[float, float]:
    """
    Find θ that minimises MMD between de-rotated target latents and source latents.

    Operates entirely in the latent space:
        z = A⁺ @ (X - μ)
        minimise MMD(R(-θ) @ z_m, z_src)

    This is unsupervised (no y_m needed) and uses all source samples as a
    stable reference, making it more robust than AUROC search at small m.

    Returns (best_theta, best_mmd).
    """
    A_pinv = np.linalg.pinv(A)                          # (k, p)
    z_src  = (A_pinv @ (X_src  - mu_src).T).T           # (n_src, k)
    z_m    = (A_pinv @ (X_m    - mu_tgt_m).T).T         # (m, k)

    best_theta, best_mmd = 0.0, float("inf")
    k = A.shape[1]
    for theta in angle_grid:
        R_inv    = _givens_inv(theta, k)
        z_al     = (R_inv @ z_m.T).T                    # (m, k)
        mmd_val  = compute_mmd(z_al, z_src)
        if mmd_val < best_mmd:
            best_mmd   = mmd_val
            best_theta = theta

    return best_theta, best_mmd


# ---------------------------------------------------------------------------
# Load best-shift config from CSV
# ---------------------------------------------------------------------------

def _load_shift_configs(tables_dir: Path) -> dict[str, dict]:
    """Returns {center_name: {"loading_rotation": angle, ...}} from CSV."""
    csv = tables_dir / "best_shift_per_center.csv"
    configs: dict[str, dict] = {}
    if not csv.exists():
        warnings.warn(f"best_shift_per_center.csv not found in {tables_dir}")
        return configs
    bdf = pd.read_csv(csv)
    for _, row in bdf.iterrows():
        cname = row["center_name"]
        cfg: dict = {}
        if row["improvement"] > 0.005:
            if pd.notna(row.get("best_shift_type_1")) and row["best_shift_type_1"] != "none":
                cfg[row["best_shift_type_1"]] = float(row["best_magnitude_1"])
            if pd.notna(row.get("best_shift_type_2")) and \
               str(row.get("best_shift_type_2", "")) not in ("", "nan", "none"):
                cfg[row["best_shift_type_2"]] = float(row["best_magnitude_2"])
        configs[cname] = cfg
    return configs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_shift_adaptation_experiment(
    m_grid: list[int] | None = None,
    n_seeds: int = 20,
    k_total: int = 6,
    tau: float = 12.0,
    nu: float = 1.5,
    seed: int = 42,
    out_dir: str = "outputs/tables/",
    verbose: bool = True,
) -> None:

    if m_grid is None:
        m_grid = [5, 10, 20, 30, 50]

    centers, _ = load_all_centers(registry_module=registry)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shift_configs = _load_shift_configs(out_path)
    angle_grid    = SINGLETON_GRIDS["loading_rotation"]   # shared search grid

    rows = []

    for held_idx in range(len(centers)):
        held_out = centers[held_idx]
        hid      = held_out["center_id"]
        name     = registry.CENTER_NAMES.get(hid, str(hid))
        n_full   = held_out["n"]
        shift_cfg = shift_configs.get(name, {})
        # Oracle rotation angle (0 if no rotation shift was identified)
        oracle_theta = float(shift_cfg.get("loading_rotation", 0.0))

        if verbose:
            print(f"\n=== {name} (n={n_full})  oracle_θ={oracle_theta:.3f} ===")

        # ----------------------------------------------------------------
        # Fit shared structure on source centers
        # ----------------------------------------------------------------
        sources    = [c for i, c in enumerate(centers) if i != held_idx]
        X_src_pool = np.vstack([c["X"] for c in sources])
        _, src_med = impute_median(X_src_pool)
        sources_imp = [
            {**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])}
            for c in sources
        ]
        X_held_imp = np.where(np.isnan(held_out["X"]), src_med, held_out["X"])

        A, w, _, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
        )

        X_held_s  = scaler.transform(X_held_imp)
        X_src_all = np.vstack([scaler.transform(c["X"]) for c in sources_imp])
        y_src_all = np.concatenate([
            _binary_label(c["t"], c["delta"], tau) for c in sources_imp
        ])
        y_held    = _binary_label(held_out["t"], held_out["delta"], tau)

        # Feature means in standardised space
        mu_src        = X_src_all.mean(axis=0)           # (p,)
        mu_tgt_oracle = X_held_s.mean(axis=0)            # (p,)  full held-out

        # Oracle-aligned target features (fixed, no m needed)
        X_held_aligned_oracle = align_features(
            X_held_s, A, mu_tgt_oracle, mu_src, oracle_theta
        )

        # ----------------------------------------------------------------
        # Fixed variants (no m) — per model
        # ----------------------------------------------------------------
        for mname, fit_fn in _MODELS.items():
            # S — baseline
            mdl_S = fit_fn(X_src_all, y_src_all, seed=0)
            a_S   = _auroc(mdl_S, X_held_s, y_held)

            # S_adapt_oracle — source model on oracle-aligned test features
            a_oracle_adapt = _auroc(mdl_S, X_held_aligned_oracle, y_held)

            # S+real(full) — fine-tune on all real held-out
            mdl_full = fit_fn(
                np.vstack([X_src_all, X_held_s]),
                np.concatenate([y_src_all, y_held]),
                seed=0,
            )
            a_full = _auroc(mdl_full, X_held_s, y_held)

            rows += [
                {"center": name, "model": mname, "variant": "S",
                 "m": 0, "seed": 0, "auroc": a_S, "theta_used": 0.0},
                {"center": name, "model": mname, "variant": "S_adapt_oracle",
                 "m": 0, "seed": 0, "auroc": a_oracle_adapt,
                 "theta_used": oracle_theta},
                {"center": name, "model": mname, "variant": "S+real(full)",
                 "m": n_full, "seed": 0, "auroc": a_full, "theta_used": 0.0},
            ]

        if verbose:
            a_S_lr  = rows[-6]["auroc"]
            a_oa_lr = rows[-5]["auroc"]
            a_fu_lr = rows[-4]["auroc"]
            print(f"  [lr]  S={a_S_lr:.4f}  S_adapt_oracle={a_oa_lr:.4f}"
                  f"  S+real(full)={a_fu_lr:.4f}")

        # ----------------------------------------------------------------
        # Few-shot variants: loop over m and seeds
        # ----------------------------------------------------------------
        for m in m_grid:
            if m >= n_full:
                if verbose:
                    print(f"  skipping m={m} (>= n_full={n_full})")
                continue

            for mname, fit_fn in _MODELS.items():
                # Re-fit source model once per model type (same across seeds)
                mdl_S = fit_fn(X_src_all, y_src_all, seed=0)

                aurocs: dict[str, list] = {
                    k: [] for k in ["S_adapt_fs", "S_adapt_mmd", "S_ft_real",
                                    "S_ft_aligned", "theta_fs", "theta_mmd"]
                }

                for s in range(n_seeds):
                    rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                    idx = rng.choice(n_full, size=m, replace=False)
                    X_m     = X_held_s[idx]
                    y_m     = y_held[idx]

                    # Few-shot target mean estimate
                    mu_tgt_m = X_m.mean(axis=0)

                    # ── Estimate θ from m patients ──────────────────────
                    theta_fs, _ = estimate_theta_fewshot(
                        X_m, y_m, mdl_S, A, mu_tgt_m, mu_src, angle_grid
                    )

                    # ── Estimate θ from m patients via MMD (unsupervised) ───
                    theta_mmd, _ = estimate_theta_mmd(
                        X_m, X_src_all, A, mu_tgt_m, mu_src, angle_grid
                    )

                    # ── S_adapt_fs: source model on fs-aligned test features ──
                    X_held_aligned_fs = align_features(
                        X_held_s, A, mu_tgt_oracle, mu_src, theta_fs
                    )
                    a_adapt_fs = _auroc(mdl_S, X_held_aligned_fs, y_held)

                    # ── S_adapt_mmd: source model on MMD-aligned test features ──
                    X_held_aligned_mmd = align_features(
                        X_held_s, A, mu_tgt_oracle, mu_src, theta_mmd
                    )
                    a_adapt_mmd = _auroc(mdl_S, X_held_aligned_mmd, y_held)

                    # ── S_ft_real: fine-tune on m raw patients ──────────
                    mseed = seed + s * 7 + (1 if mname == "mlp" else 0)
                    mdl_ft = fit_fn(
                        np.vstack([X_src_all, X_m]),
                        np.concatenate([y_src_all, y_m]),
                        seed=mseed,
                    )
                    a_ft_real = _auroc(mdl_ft, X_held_s, y_held)

                    # ── S_ft_aligned: fine-tune on m aligned patients,
                    #    evaluate on oracle-aligned full target ───────────
                    X_m_aligned = align_features(
                        X_m, A, mu_tgt_m, mu_src, theta_fs
                    )
                    mdl_ft_al = fit_fn(
                        np.vstack([X_src_all, X_m_aligned]),
                        np.concatenate([y_src_all, y_m]),
                        seed=mseed,
                    )
                    # Evaluate on oracle-aligned test set for consistency
                    a_ft_aligned = _auroc(mdl_ft_al, X_held_aligned_oracle, y_held)

                    rows += [
                        {"center": name, "model": mname,
                         "variant": "S_adapt_fs",   "m": m, "seed": s,
                         "auroc": a_adapt_fs,    "theta_used": theta_fs},
                        {"center": name, "model": mname,
                         "variant": "S_adapt_mmd",  "m": m, "seed": s,
                         "auroc": a_adapt_mmd,   "theta_used": theta_mmd},
                        {"center": name, "model": mname,
                         "variant": "S_ft_real",    "m": m, "seed": s,
                         "auroc": a_ft_real,     "theta_used": 0.0},
                        {"center": name, "model": mname,
                         "variant": "S_ft_aligned", "m": m, "seed": s,
                         "auroc": a_ft_aligned,  "theta_used": theta_fs},
                    ]

                    aurocs["S_adapt_fs"].append(a_adapt_fs)
                    aurocs["S_adapt_mmd"].append(a_adapt_mmd)
                    aurocs["S_ft_real"].append(a_ft_real)
                    aurocs["S_ft_aligned"].append(a_ft_aligned)
                    aurocs["theta_fs"].append(theta_fs)
                    aurocs["theta_mmd"].append(theta_mmd)

                if verbose:
                    mA   = float(np.nanmean(aurocs["S_adapt_fs"]))
                    mAm  = float(np.nanmean(aurocs["S_adapt_mmd"]))
                    mR   = float(np.nanmean(aurocs["S_ft_real"]))
                    mAl  = float(np.nanmean(aurocs["S_ft_aligned"]))
                    mTh  = float(np.nanmean(aurocs["theta_fs"]))
                    mThM = float(np.nanmean(aurocs["theta_mmd"]))
                    print(f"  m={m:4d} [{mname:3s}]  adapt_fs={mA:.4f}"
                          f"  adapt_mmd={mAm:.4f}  ft_real={mR:.4f}"
                          f"  ft_aligned={mAl:.4f}"
                          f"  θ_fs={mTh:.3f}  θ_mmd={mThM:.3f}")

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "shift_adaptation_results.csv", index=False)

    few_shot_vars = ["S_adapt_fs", "S_adapt_mmd", "S_ft_real", "S_ft_aligned"]
    summ = (
        df[df["variant"].isin(few_shot_vars)]
        .groupby(["center", "model", "variant", "m"])
        .agg(
            auroc_mean=("auroc",      "mean"),
            auroc_std=("auroc",       "std"),
            theta_mean=("theta_used", "mean"),
        )
        .reset_index()
    )
    # Append fixed variants
    for var in ["S", "S_adapt_oracle", "S+real(full)"]:
        sub = df[df["variant"] == var].copy()
        sub = sub.rename(columns={"auroc": "auroc_mean", "theta_used": "theta_mean"})
        sub["auroc_std"] = 0.0
        summ = pd.concat([summ, sub[["center", "model", "variant", "m",
                                     "auroc_mean", "auroc_std", "theta_mean"]]])

    summ = summ.sort_values(["center", "model", "variant", "m"]).reset_index(drop=True)
    summ.to_csv(out_path / "shift_adaptation_summary.csv", index=False)

    if verbose:
        print(f"\nSaved shift_adaptation_results.csv and shift_adaptation_summary.csv"
              f" → {out_path}/")
        print("\nMean AUROC by variant and model (averaged over centers and seeds):")
        agg = (
            df.groupby(["model", "variant", "m"])["auroc"]
            .mean()
            .unstack("m")
            .round(4)
        )
        print(agg.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Shift-informed test-time domain adaptation")
    p.add_argument("--m_grid",     type=int, nargs="+", default=[5, 10, 20, 30, 50])
    p.add_argument("--n_seeds",    type=int, default=20)
    p.add_argument("--k_total",    type=int, default=6)
    p.add_argument("--tau",        type=float, default=12.0)
    p.add_argument("--nu",         type=float, default=1.5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out_dir",    default="outputs/tables/")
    p.add_argument("--no_verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_shift_adaptation_experiment(
        m_grid=args.m_grid,
        n_seeds=args.n_seeds,
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
