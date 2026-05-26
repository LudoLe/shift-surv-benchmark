"""
LOODO generator reconstruction on public multi-centre datasets.

Replicates the i3lung pipeline (run_shift_experiments.py) on:
  rotterdam_gbsg  : Rotterdam (n=2982) + GBSG2 (n=686), RFS in months
  support2        : SUPPORT2 by dzclass (4 groups), OS in months

For each held-out domain:
  1. Fit reference generator on the remaining source domains.
  2. Fit held-out domain parameters (mu_d, lambda0_d).
  3. Generate synthetic held-out domain.
  4. Score: MMD, KM distance, AUROC-gap Δ, prevalence gap → composite.
  5. Singleton shift scan (lambda_shift, nu_shift, mu_scale, noise, loading_rotation).
  6. Conditional pair scan if singleton improves >= pair_threshold.
  7. Risk direction analysis: w vector, feature-space loading A@w,
     per-domain lambda0 and mu_d, w cosine-similarity across folds.

Outputs  (outputs/tables/<dataset>/)
  baseline_loodo_results.csv
  singleton_shift_results.csv
  pair_shift_results.csv         (when triggered)
  best_shift_per_center.csv
  risk_direction.csv             — w, A@w weights, lambda0, mu_d per fold
  km_curves.json

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_loodo_public --dataset rotterdam_gbsg
  python -m experiments.run_loodo_public --dataset support2
  python -m experiments.run_loodo_public --dataset all
"""

from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from data.preprocess import impute_median
from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams
from generator.fitting import (
    fit_reference_generator,
    fit_domain_params,
    make_real_domain_dataset,
)
from generator.shift_components import apply_shift_config
from experiments.compare_domains import domain_score, kaplan_meier
from experiments.shift_search import (
    singleton_scan,
    pair_scan,
    best_singleton,
    best_pair,
    improvement,
)


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "rotterdam_gbsg": {
        "registry":        "data.registry_rotterdam_gbsg",
        "tau":             36.0,
        "nu":              1.5,
        "k_total":         4,
        "pair_threshold":  0.10,
    },
    "support2": {
        "registry":        "data.registry_support2",
        "tau":             3.0,
        "nu":              1.5,
        "k_total":         6,
        "pair_threshold":  0.10,
    },
    "i3lung_vanguri": {
        "registry":        "data.registry_i3lung_vanguri",
        "tau":             12.0,
        "nu":              1.5,
        "k_total":         4,   # 7 features → 4 latent dims
        "pair_threshold":  0.10,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _km_data(ds, label: str) -> dict:
    t = ds.t_obs if ds.t_obs is not None else ds.T
    d = ds.delta if ds.delta is not None else np.ones(ds.n, dtype=int)
    times, S = kaplan_meier(t, d)
    return {"label": label, "times": times.tolist(), "S": S.tolist()}


def _feature_risk_weights(A: np.ndarray, w: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Project risk weights back to feature space: direction = A @ w.
    Returns a DataFrame sorted by absolute loading (most prognostic first).
    """
    feat_w = A @ w   # (p,)
    df = pd.DataFrame({
        "feature":   feature_names,
        "feat_w":    feat_w,
        "abs_feat_w": np.abs(feat_w),
    }).sort_values("abs_feat_w", ascending=False).reset_index(drop=True)
    return df


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))


# ---------------------------------------------------------------------------
# One LOODO fold
# ---------------------------------------------------------------------------

def _prepare_fold(centers, held_idx, k_total, tau, nu, seed, center_names, feature_names):
    held_out = centers[held_idx]
    sources  = [c for i, c in enumerate(centers) if i != held_idx]
    hid      = held_out["center_id"]
    p        = held_out["X"].shape[1]

    X_src_pool = np.vstack([c["X"] for c in sources])
    _, src_med = impute_median(X_src_pool)
    sources_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_med, c["X"])} for c in sources]
    X_held_imp  = np.where(np.isnan(held_out["X"]), src_med, held_out["X"])

    A, w, source_params, scaler = fit_reference_generator(
        source_centers=sources_imp, k_total=k_total, tau=tau, nu=nu,
    )

    X_held_s = scaler.transform(X_held_imp)
    held_params = fit_domain_params(
        X_d=X_held_s, t_d=held_out["t"], delta_d=held_out["delta"],
        A=A, w=w, tau=tau, nu=nu, domain_id=hid,
        n_synthetic=held_out["n"],
    )

    base = BaseParams(k_inv=k_total, k_spu=0, p=p, tau=tau, seed=seed)

    src_ds_list = []
    for sp, sc in zip(source_params, sources_imp):
        X_sc_s = scaler.transform(sc["X"])
        src_ds_list.append(make_real_domain_dataset(
            X_s=X_sc_s, t_obs=sc["t"], delta=sc["delta"],
            A=A, w=w, tau=tau, params=sp, domain_id=sc["center_id"],
        ))

    real_ds = make_real_domain_dataset(
        X_s=X_held_s, t_obs=held_out["t"], delta=held_out["delta"],
        A=A, w=w, tau=tau, params=held_params, domain_id=hid,
    )

    gen = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)
    synth_ds = gen.generate_domain(held_params, domain_id=hid)
    baseline_scores = domain_score(synth_ds, real_ds, src_ds_list, tau=tau)

    hname = center_names.get(hid, str(hid))
    return hname, held_params, base, A, w, src_ds_list, real_ds, baseline_scores, scaler


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loodo(
    dataset_name: str,
    tau: float,
    nu: float,
    k_total: int,
    centers: list[dict],
    center_names: dict,
    feature_names: list[str],
    seed: int = 42,
    pair_threshold: float = 0.10,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> None:

    if out_dir is None:
        out_dir = _ROOT / "outputs" / "tables" / dataset_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    D = len(centers)
    baseline_rows   = []
    singleton_rows  = []
    pair_rows       = []
    best_rows       = []
    risk_rows       = []
    km_curves_data  = {}

    # Collect w vectors across folds for cosine-similarity analysis
    w_per_fold: dict[str, np.ndarray] = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LOODO generator reconstruction: {dataset_name}")
        print(f"  {D} domains  p={feature_names.__len__()}  k={k_total}  τ={tau}  ν={nu}")
        print(f"{'='*60}\n")

    for held_idx in range(D):
        hid   = centers[held_idx]["center_id"]
        hname, held_params, base, A, w, src_ds_list, real_ds, b_scores, scaler = \
            _prepare_fold(centers, held_idx, k_total, tau, nu, seed, center_names, feature_names)

        w_per_fold[hname] = w.copy()
        b_composite = b_scores["composite"]

        if verbose:
            print(f"\n--- Domain: {hname} (id={hid}, n={centers[held_idx]['n']}) ---")
            print(f"  Baseline  composite={b_composite:.4f}  "
                  f"MMD={b_scores['mmd']:.4f}  "
                  f"KM_dist={b_scores['km_distance']:.4f}  "
                  f"AUROC_gap_Δ={b_scores['auroc_gap_delta']:.4f}  "
                  f"prev_gap={b_scores['prevalence_gap']:.4f}")

        # ---- risk direction analysis ----
        fw = _feature_risk_weights(A, w, feature_names)
        risk_rows.append({
            "domain":    hname,
            "lambda0":   round(held_params.lambda0, 6),
            "mu_d_norm": round(float(np.linalg.norm(held_params.mu_inv)), 4),
            **{f"w_{i}": round(float(w[i]), 4) for i in range(len(w))},
            **{f"fw_{feat}": round(float(fw.loc[fw.feature == feat, 'feat_w'].values[0]), 4)
               for feat in feature_names},
        })

        if verbose:
            print(f"  λ₀={held_params.lambda0:.5f}  ‖μ_d‖={np.linalg.norm(held_params.mu_inv):.3f}")
            print(f"  Top feature-space risk weights (A·w):")
            for _, row in fw.head(5).iterrows():
                sign = "↑" if row.feat_w > 0 else "↓"
                print(f"    {sign} {row.feature:<28} {row.feat_w:+.3f}")

        # ---- baseline rows + KM ----
        baseline_rows.append({
            "domain":       hname,
            "domain_id":    hid,
            "n":            centers[held_idx]["n"],
            "lambda0":      round(held_params.lambda0, 6),
            "mu_d_norm":    round(float(np.linalg.norm(held_params.mu_inv)), 4),
            **{k: v for k, v in b_scores.items() if k not in ("weights", "synth_n", "real_n")},
        })

        gen0 = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)
        synth_ref = gen0.generate_domain(held_params, domain_id=hid)
        km_curves_data[hname] = {
            "real":      _km_data(real_ds,   "real"),
            "reference": _km_data(synth_ref, "reference"),
        }

        # ---- singleton scan ----
        if verbose:
            print(f"  Singleton scan (5 shift types)...")
        sing_df = singleton_scan(
            held_params=held_params, base=base, A=A,
            src_ds_list=src_ds_list, real_ds=real_ds,
            tau=tau, seed=seed, verbose=False,
        )
        sing_df.insert(0, "domain", hname)
        sing_df.insert(0, "domain_id", hid)
        singleton_rows.append(sing_df)

        best_sing = best_singleton(sing_df)
        best_sing_composite = float(best_sing.get("composite", b_composite))
        sing_imp = improvement(b_composite, best_sing_composite)

        if verbose:
            print(f"  Best singleton: {best_sing.get('shift_type')} "
                  f"mag={best_sing.get('magnitude'):.4f}  "
                  f"composite={best_sing_composite:.4f}  Δ={sing_imp:+.1%}")

        # best singleton KM
        best_sing_config = {}
        if "shift_type" in best_sing and best_sing["shift_type"] is not None:
            best_sing_config = {best_sing["shift_type"]: best_sing["magnitude"]}
        shifted_p = apply_shift_config(held_params, best_sing_config)
        synth_sing = gen0.generate_domain(shifted_p, domain_id=hid)
        km_curves_data[hname]["best_singleton"] = _km_data(synth_sing, "best_singleton")

        # ---- pair scan ----
        best_p = best_sing
        do_pair = sing_imp >= pair_threshold
        if do_pair:
            if verbose:
                print(f"  Pair scan triggered (Δ={sing_imp:.1%} ≥ {pair_threshold:.0%})...")
            p_df = pair_scan(
                held_params=held_params, base=base, A=A,
                src_ds_list=src_ds_list, real_ds=real_ds,
                tau=tau, best_singleton_df=sing_df, seed=seed, verbose=verbose,
            )
            if not p_df.empty:
                p_df.insert(0, "domain", hname)
                p_df.insert(0, "domain_id", hid)
                pair_rows.append(p_df)

                best_p_row = best_pair(p_df)
                p_composite = float(best_p_row.get("composite", best_sing_composite))
                p_imp = improvement(b_composite, p_composite)

                if verbose:
                    print(f"  Best pair: {best_p_row.get('shift_type_1')} + "
                          f"{best_p_row.get('shift_type_2')}  "
                          f"composite={p_composite:.4f}  Δ={p_imp:+.1%}")

                if p_composite < best_sing_composite:
                    best_p = best_p_row
                    pair_config = {
                        best_p_row["shift_type_1"]: best_p_row["magnitude_1"],
                        best_p_row["shift_type_2"]: best_p_row["magnitude_2"],
                    }
                    sp2 = apply_shift_config(held_params, pair_config)
                    synth_pair = gen0.generate_domain(sp2, domain_id=hid)
                    km_curves_data[hname]["best_pair"] = _km_data(synth_pair, "best_pair")

        # ---- summary row ----
        final_composite = float(best_p.get("composite", b_composite))
        best_rows.append({
            "domain":              hname,
            "n":                   centers[held_idx]["n"],
            "composite_baseline":  round(b_composite, 4),
            "composite_best":      round(final_composite, 4),
            "improvement":         round(improvement(b_composite, final_composite), 4),
            "pair_triggered":      do_pair,
            "best_shift_1":        best_p.get("shift_type") or best_p.get("shift_type_1", "none"),
            "best_mag_1":          best_p.get("magnitude")  or best_p.get("magnitude_1", None),
            "best_shift_2":        best_p.get("shift_type_2", None),
            "best_mag_2":          best_p.get("magnitude_2", None),
            "km_prev_real":        b_scores.get("km_prev_real"),
            "km_prev_synth_best":  best_p.get("km_prev_synth"),
            "km_dist_baseline":    b_scores.get("km_distance"),
            "km_dist_best":        best_p.get("km_distance"),
            "auroc_gap_baseline":  b_scores.get("auroc_gap_delta"),
            "auroc_gap_best":      best_p.get("auroc_gap_delta"),
        })

    # ---- w cosine-similarity across folds ----
    names_list = list(w_per_fold.keys())
    if verbose and len(names_list) > 1:
        print(f"\n{'='*60}")
        print("  Risk direction (w) cosine similarity across LOODO folds:")
        for i in range(len(names_list)):
            for j in range(i + 1, len(names_list)):
                n1, n2 = names_list[i], names_list[j]
                cs = _cosine_sim(w_per_fold[n1], w_per_fold[n2])
                print(f"    cos(w_{n1}, w_{n2}) = {cs:+.4f}")

    # ---- print summary table ----
    if verbose:
        best_df = pd.DataFrame(best_rows)
        print(f"\n{'='*60}")
        print(f"  Best shift per domain ({dataset_name}):")
        print(best_df[[
            "domain", "composite_baseline", "composite_best",
            "improvement", "best_shift_1", "best_mag_1",
        ]].to_string(index=False))

    # ---- save outputs ----
    pd.DataFrame(baseline_rows).to_csv(out_dir / "baseline_loodo_results.csv", index=False)
    pd.concat(singleton_rows, ignore_index=True).to_csv(
        out_dir / "singleton_shift_results.csv", index=False)
    if pair_rows:
        pd.concat(pair_rows, ignore_index=True).to_csv(
            out_dir / "pair_shift_results.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / "best_shift_per_domain.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(out_dir / "risk_direction.csv", index=False)
    with open(out_dir / "km_curves.json", "w") as f:
        json.dump(km_curves_data, f, indent=2)

    if verbose:
        print(f"\nAll outputs saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, seed: int = 42, verbose: bool = True) -> None:
    cfg = DATASET_CONFIGS[dataset_name]
    reg = importlib.import_module(cfg["registry"])
    centers, feature_names = reg.load_all_centers()
    run_loodo(
        dataset_name=dataset_name,
        tau=cfg["tau"],
        nu=cfg["nu"],
        k_total=cfg["k_total"],
        centers=centers,
        center_names=reg.CENTER_NAMES,
        feature_names=feature_names,
        seed=seed,
        pair_threshold=cfg["pair_threshold"],
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOODO generator reconstruction on public datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS) + ["all"],
        default="rotterdam_gbsg",
    )
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, seed=args.seed, verbose=not args.quiet)
