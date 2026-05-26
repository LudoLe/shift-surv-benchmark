"""
Synthetic data augmentation experiment.

Question: does augmenting source training data with generator-calibrated
synthetic held-out samples improve downstream 12-month risk prediction
on the real held-out center?

Outcome & model
---------------
Outcome  : 12-month binary label — y = 1[t ≤ τ AND delta=1] for real data,
           y = 1[T ≤ τ] for synthetic (uncensored, so delta=1 always).
Model    : L2-penalised logistic regression (C=1.0), same family as the
           generator's own risk-weight estimator.  Features = standardised
           p-dim clinical covariates (same space as X_s used throughout).
Metric   : AUROC at 12 months evaluated on the FULL real held-out center.

Training variants
-----------------
  S                    source centers only
  S+src_aug            source + n_full synthetic source-shaped samples
                       (volume-matched control; tests "more data" vs "right shape")
  S+Bsynth(m)          source + n_full synthetic held-out via Variant B
                       (m real patients consumed for KM calibration only)
  S+real(m)            source + m real held-out patients
  S+real(m)+Bsynth(m)  source + m real + n_full synthetic held-out
  S+real(full)         oracle: source + all real held-out patients

Key comparisons
---------------
  S+Bsynth(m) vs S                     — zero-real-data augmentation benefit
  S+real(m)+Bsynth(m) vs S+real(m)     — synthetic on top of real (main test)
  S+Bsynth vs S+src_aug at same n      — shape vs volume

Note on eval leakage: the m real patients used for calibration overlap with the
evaluation set (full held-out center).  This is unavoidable when centers are
small but the effect is tiny at m ≪ n_full; and it affects both S+real(m) arms
equally, so the key comparison S+real(m)+Bsynth vs S+real(m) is unaffected.

Outputs (in --out_dir)
----------------------
  augmentation_results.csv   one row per (center, variant, m, seed)
  augmentation_summary.csv   mean ± std AUROC per (center, variant, m)

Usage
-----
  python -m experiments.run_augmentation_experiment [options]
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

from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams
from generator.fitting import (
    fit_reference_generator,
    fit_domain_params,
    fit_domain_params_fewshot,
    average_domain_params,
)
from generator.shift_components import apply_shift_config

from experiments.run_shift_experiments import _prepare_fold


# ---------------------------------------------------------------------------
# Downstream model helpers
# ---------------------------------------------------------------------------

def _binary_label(t: np.ndarray, delta: np.ndarray, tau: float) -> np.ndarray:
    """y = 1[t ≤ tau AND delta == 1]  (safe for both real and synthetic data)."""
    return ((t <= tau) & (delta == 1)).astype(int)


def _fit_lr(X_train: np.ndarray, y_train: np.ndarray,
            seed: int = 0) -> LogisticRegression | None:
    """L2-penalised LR, C=1.0, consistent with generator's risk-weight estimator."""
    if len(np.unique(y_train)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                random_state=seed, class_weight="balanced")
        lr.fit(X_train, y_train)
    return lr


def _fit_mlp(X_train: np.ndarray, y_train: np.ndarray,
             seed: int = 0) -> MLPClassifier | None:
    """
    Small MLP: two hidden layers (64→32), ReLU, L2=1e-3.
    Early stopping enabled when training set is large enough (n ≥ 20);
    otherwise fixed 300 iterations to avoid validation-split failures on tiny sets.
    """
    if len(np.unique(y_train)) < 2:
        return None
    n = len(y_train)
    early = n >= 20
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            early_stopping=early,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
        mlp.fit(X_train, y_train)
    return mlp


def _auroc(model, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
    """AUROC on the eval set.  Returns NaN if model is None or y_eval is constant."""
    if model is None:
        return float("nan")
    if len(np.unique(y_eval)) < 2:
        return float("nan")
    probs = model.predict_proba(X_eval)[:, 1]
    return float(roc_auc_score(y_eval, probs))


# Map name → fit function (both accept (X, y, seed))
_MODELS = {"lr": _fit_lr, "mlp": _fit_mlp}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_augmentation_experiment(
    m_grid: list[int] | None = None,
    n_seeds: int = 20,
    k_total: int = 6,
    tau: float = 12.0,
    nu: float = 1.5,
    seed: int = 42,
    out_dir: str = "outputs/tables/",
    verbose: bool = True,
) -> None:
    """
    Run the full augmentation benchmark across all LOODO folds.

    Parameters
    ----------
    m_grid   : calibration set sizes (patients from held-out center)
    n_seeds  : independent subsampling seeds per (center, m) cell
    k_total  : latent dimensions (must match earlier experiments)
    tau      : 12-month horizon
    nu       : Weibull shape (fixed)
    seed     : base RNG seed
    out_dir  : output directory
    """
    if m_grid is None:
        m_grid = [5, 10, 20, 30, 50]

    centers, _ = load_all_centers(registry_module=registry)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load best-shift config per center (from shift calibration step)
    # ----------------------------------------------------------------
    best_shift_csv = Path(out_dir) / "best_shift_per_center.csv"
    shift_configs: dict[str, dict] = {}
    if best_shift_csv.exists():
        bdf = pd.read_csv(best_shift_csv)
        for _, brow in bdf.iterrows():
            cname = brow["center_name"]
            cfg: dict = {}
            # Only use calibrated shift when it actually improves fidelity
            if brow["improvement"] > 0.005:
                if pd.notna(brow.get("best_shift_type_1")) and brow["best_shift_type_1"] != "none":
                    cfg[brow["best_shift_type_1"]] = float(brow["best_magnitude_1"])
                if pd.notna(brow.get("best_shift_type_2")) and str(brow.get("best_shift_type_2", "")) not in ("", "nan", "none"):
                    cfg[brow["best_shift_type_2"]] = float(brow["best_magnitude_2"])
            shift_configs[cname] = cfg
    else:
        warnings.warn("best_shift_per_center.csv not found — calibrated variants will use no shift")

    rows = []

    for held_idx in range(len(centers)):
        held_out = centers[held_idx]
        hid      = held_out["center_id"]
        name     = registry.CENTER_NAMES.get(hid, str(hid))
        n_full   = held_out["n"]

        if verbose:
            print(f"\n=== Center {name} (n={n_full}) ===")

        # ----------------------------------------------------------------
        # Fit shared structure on source centers
        # ----------------------------------------------------------------
        sources = [c for i, c in enumerate(centers) if i != held_idx]

        X_src_pool = np.vstack([c["X"] for c in sources])
        _, src_medians = impute_median(X_src_pool)
        sources_imp = [
            {**c, "X": np.where(np.isnan(c["X"]), src_medians, c["X"])}
            for c in sources
        ]
        X_held_imp = np.where(np.isnan(held_out["X"]), src_medians, held_out["X"])

        A, w, source_params, scaler = fit_reference_generator(
            source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
        )

        # Standardised feature matrices
        X_held_s = scaler.transform(X_held_imp)
        X_src_scaled = [scaler.transform(c["X"]) for c in sources_imp]

        # Source training pool (features + labels)
        X_src_all = np.vstack(X_src_scaled)
        y_src_all = np.concatenate([
            _binary_label(c["t"], c["delta"], tau) for c in sources_imp
        ])

        # Evaluation set: FULL real held-out center
        y_held_full = _binary_label(held_out["t"], held_out["delta"], tau)

        # Source-shape params (volume-matched control)
        avg_src_params = average_domain_params(source_params)
        avg_src_params.n = n_full   # same size as held-out

        # Full held-out params (oracle calibration, Variant C equivalent)
        held_full_params = fit_domain_params(
            X_d=X_held_s, t_d=held_out["t"], delta_d=held_out["delta"],
            A=A, w=w, tau=tau, nu=nu, domain_id=hid, n_synthetic=n_full,
        )

        source_lambda0_mean = float(np.mean([p.lambda0 for p in source_params]))

        base = BaseParams(k_inv=k_total, k_spu=0, p=held_out["X"].shape[1],
                          tau=tau, seed=seed)
        gen = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)

        # ----------------------------------------------------------------
        # Per-center shift config (empty dict = no shift, e.g. VHIO)
        # ----------------------------------------------------------------
        shift_cfg = shift_configs.get(name, {})
        if verbose and shift_cfg:
            print(f"  shift config: {shift_cfg}")
        elif verbose:
            print(f"  shift config: none (reference is best)")

        # Calibrated domain params: apply best-shift to full held-out params
        held_calib_params = apply_shift_config(held_full_params, shift_cfg)

        # ----------------------------------------------------------------
        # Pre-generate synthetic data that is shared across models
        # ----------------------------------------------------------------
        synth_src = gen.generate_domain(
            avg_src_params, domain_id=hid, seed=seed + hid * 10000 + 999,
        )
        synth_C = gen.generate_domain(
            held_full_params, domain_id=hid, seed=seed + hid * 10000 + 998,
        )
        synth_C_cal = gen.generate_domain(
            held_calib_params, domain_id=hid, seed=seed + hid * 10000 + 997,
        )

        X_src_aug_pool = np.vstack([X_src_all, synth_src.x])
        y_src_aug_pool = np.concatenate([y_src_all,
                                         _binary_label(synth_src.t_obs,
                                                       synth_src.delta, tau)])
        X_oracle_pool  = np.vstack([X_src_all, X_held_s])
        y_oracle_pool  = np.concatenate([y_src_all, y_held_full])
        X_csynth_pool  = np.vstack([X_src_all, synth_C.x])
        y_csynth_pool  = np.concatenate([y_src_all,
                                          _binary_label(synth_C.t_obs,
                                                        synth_C.delta, tau)])
        X_csynth_cal_pool = np.vstack([X_src_all, synth_C_cal.x])
        y_csynth_cal_pool = np.concatenate([y_src_all,
                                             _binary_label(synth_C_cal.t_obs,
                                                           synth_C_cal.delta, tau)])

        # ----------------------------------------------------------------
        # Fixed-data variants — run for each model
        # ----------------------------------------------------------------
        for mname, fit_fn in _MODELS.items():
            mdl_S           = fit_fn(X_src_all,          y_src_all,          seed=0)
            mdl_src_aug     = fit_fn(X_src_aug_pool,     y_src_aug_pool,     seed=0)
            mdl_oracle      = fit_fn(X_oracle_pool,      y_oracle_pool,      seed=0)
            mdl_csynth      = fit_fn(X_csynth_pool,      y_csynth_pool,      seed=0)
            mdl_csynth_cal  = fit_fn(X_csynth_cal_pool,  y_csynth_cal_pool,  seed=0)

            a_S          = _auroc(mdl_S,          X_held_s, y_held_full)
            a_src_aug    = _auroc(mdl_src_aug,    X_held_s, y_held_full)
            a_oracle     = _auroc(mdl_oracle,     X_held_s, y_held_full)
            a_csynth     = _auroc(mdl_csynth,     X_held_s, y_held_full)
            a_csynth_cal = _auroc(mdl_csynth_cal, X_held_s, y_held_full)

            rows.append({"center": name, "model": mname, "variant": "S",
                         "m": 0, "seed": 0, "auroc": a_S})
            rows.append({"center": name, "model": mname, "variant": "S+src_aug",
                         "m": 0, "seed": 0, "auroc": a_src_aug})
            rows.append({"center": name, "model": mname, "variant": "S+real(full)",
                         "m": n_full, "seed": 0, "auroc": a_oracle})
            rows.append({"center": name, "model": mname, "variant": "S+Csynth",
                         "m": n_full, "seed": 0, "auroc": a_csynth})
            rows.append({"center": name, "model": mname, "variant": "S+Csynth_cal",
                         "m": n_full, "seed": 0, "auroc": a_csynth_cal})

        if verbose:
            a_S_lr  = next(r["auroc"] for r in rows
                           if r["center"]==name and r["model"]=="lr"
                           and r["variant"]=="S")
            a_S_mlp = next(r["auroc"] for r in rows
                           if r["center"]==name and r["model"]=="mlp"
                           and r["variant"]=="S")
            print(f"  S             LR={a_S_lr:.4f}  MLP={a_S_mlp:.4f}")
            a_ora_lr  = next(r["auroc"] for r in rows
                             if r["center"]==name and r["model"]=="lr"
                             and r["variant"]=="S+real(full)")
            a_ora_mlp = next(r["auroc"] for r in rows
                             if r["center"]==name and r["model"]=="mlp"
                             and r["variant"]=="S+real(full)")
            print(f"  S+real(full)  LR={a_ora_lr:.4f}  MLP={a_ora_mlp:.4f}  [oracle]")

        # ----------------------------------------------------------------
        # Few-shot variants: loop over m, seeds, and models
        # ----------------------------------------------------------------
        for m in m_grid:
            if m >= n_full:
                if verbose:
                    print(f"  skipping m={m} (≥ n_full={n_full})")
                continue

            # Accumulate per-model stats for verbose summary
            stats: dict[str, dict[str, list]] = {
                mn: {"bsynth": [], "real": [], "combined": []}
                for mn in _MODELS
            }

            for s in range(n_seeds):
                rng = np.random.default_rng(seed + hid * 1000 + s * 13 + m)
                idx = rng.choice(n_full, size=m, replace=False)
                X_m     = X_held_s[idx]
                t_m     = held_out["t"][idx]
                delta_m = held_out["delta"][idx]
                y_m     = _binary_label(t_m, delta_m, tau)

                # Generate synthetic held-out (shared across models)
                params_B = fit_domain_params_fewshot(
                    X_m=X_m, t_m=t_m, delta_m=delta_m,
                    A=A, w=w, tau=tau, nu=nu, domain_id=hid,
                    n_synthetic=n_full,
                    source_lambda0_mean=source_lambda0_mean,
                    variant="B",
                )
                # Calibrated version: apply best shift to few-shot params
                params_B_cal = apply_shift_config(params_B, shift_cfg)

                synth_B = gen.generate_domain(
                    params_B, domain_id=hid,
                    seed=seed + hid * 10000 + s * 100 + m,
                )
                synth_B_cal = gen.generate_domain(
                    params_B_cal, domain_id=hid,
                    seed=seed + hid * 10000 + s * 100 + m + 1,
                )

                X_synth_B     = synth_B.x
                y_synth_B     = _binary_label(synth_B.t_obs,     synth_B.delta,     tau)
                X_synth_B_cal = synth_B_cal.x
                y_synth_B_cal = _binary_label(synth_B_cal.t_obs, synth_B_cal.delta, tau)

                # Training pools (shared across models)
                X_bsynth_pool     = np.vstack([X_src_all, X_synth_B])
                y_bsynth_pool     = np.concatenate([y_src_all, y_synth_B])
                X_bsynth_cal_pool = np.vstack([X_src_all, X_synth_B_cal])
                y_bsynth_cal_pool = np.concatenate([y_src_all, y_synth_B_cal])
                X_rm_pool         = np.vstack([X_src_all, X_m])
                y_rm_pool         = np.concatenate([y_src_all, y_m])
                X_comb_pool       = np.vstack([X_src_all, X_m, X_synth_B])
                y_comb_pool       = np.concatenate([y_src_all, y_m, y_synth_B])
                X_comb_cal_pool   = np.vstack([X_src_all, X_m, X_synth_B_cal])
                y_comb_cal_pool   = np.concatenate([y_src_all, y_m, y_synth_B_cal])

                for mname, fit_fn in _MODELS.items():
                    mseed = seed + s * 7 + (1 if mname == "mlp" else 0)
                    mdl_b       = fit_fn(X_bsynth_pool,     y_bsynth_pool,     seed=mseed)
                    mdl_b_cal   = fit_fn(X_bsynth_cal_pool, y_bsynth_cal_pool, seed=mseed)
                    mdl_r       = fit_fn(X_rm_pool,         y_rm_pool,         seed=mseed)
                    mdl_c       = fit_fn(X_comb_pool,       y_comb_pool,       seed=mseed)
                    mdl_c_cal   = fit_fn(X_comb_cal_pool,   y_comb_cal_pool,   seed=mseed)

                    a_b       = _auroc(mdl_b,     X_held_s, y_held_full)
                    a_b_cal   = _auroc(mdl_b_cal, X_held_s, y_held_full)
                    a_r       = _auroc(mdl_r,     X_held_s, y_held_full)
                    a_c       = _auroc(mdl_c,     X_held_s, y_held_full)
                    a_c_cal   = _auroc(mdl_c_cal, X_held_s, y_held_full)

                    rows.append({"center": name, "model": mname,
                                 "variant": "S+Bsynth",         "m": m, "seed": s, "auroc": a_b})
                    rows.append({"center": name, "model": mname,
                                 "variant": "S+Bsynth_cal",     "m": m, "seed": s, "auroc": a_b_cal})
                    rows.append({"center": name, "model": mname,
                                 "variant": "S+real",           "m": m, "seed": s, "auroc": a_r})
                    rows.append({"center": name, "model": mname,
                                 "variant": "S+real+Bsynth",    "m": m, "seed": s, "auroc": a_c})
                    rows.append({"center": name, "model": mname,
                                 "variant": "S+real+Bsynth_cal","m": m, "seed": s, "auroc": a_c_cal})

                    stats[mname]["bsynth"].append(a_b)
                    stats[mname]["real"].append(a_r)
                    stats[mname]["combined"].append(a_c)

            if verbose:
                for mname in _MODELS:
                    mB    = float(np.nanmean(stats[mname]["bsynth"]))
                    mR    = float(np.nanmean(stats[mname]["real"]))
                    mC    = float(np.nanmean(stats[mname]["combined"]))
                    print(f"  m={m:4d} [{mname:3s}]  S+Bsynth={mB:.4f}  "
                          f"S+real={mR:.4f}  S+real+Bsynth={mC:.4f}"
                          f"  (Δ={mC-mR:+.4f})")

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "augmentation_results.csv", index=False)

    # Summary: mean ± std across seeds for few-shot variants
    few_shot_vars = ["S+Bsynth", "S+Bsynth_cal", "S+real",
                     "S+real+Bsynth", "S+real+Bsynth_cal"]
    summ = (
        df[df["variant"].isin(few_shot_vars)]
        .groupby(["center", "model", "variant", "m"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc",  "std"),
        )
        .reset_index()
    )
    # Append anchor rows (no seed variance)
    for var in ["S", "S+src_aug", "S+real(full)", "S+Csynth", "S+Csynth_cal"]:
        sub = df[df["variant"] == var].copy()
        sub = sub.rename(columns={"auroc": "auroc_mean"})
        sub["auroc_std"] = 0.0
        summ = pd.concat([summ, sub[["center", "model", "variant", "m",
                                     "auroc_mean", "auroc_std"]]])
    summ = summ.sort_values(["center", "model", "variant", "m"]).reset_index(drop=True)
    summ.to_csv(out_path / "augmentation_summary.csv", index=False)

    if verbose:
        print(f"\nSaved augmentation_results.csv and augmentation_summary.csv → {out_path}/")
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
    p = argparse.ArgumentParser(description="Synthetic augmentation downstream benchmark")
    p.add_argument("--m_grid",    type=int, nargs="+", default=[5, 10, 20, 30, 50])
    p.add_argument("--n_seeds",   type=int, default=20)
    p.add_argument("--k_total",   type=int, default=6)
    p.add_argument("--tau",       type=float, default=12.0)
    p.add_argument("--nu",        type=float, default=1.5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default="outputs/tables/")
    p.add_argument("--no_verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_augmentation_experiment(
        m_grid=args.m_grid,
        n_seeds=args.n_seeds,
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
