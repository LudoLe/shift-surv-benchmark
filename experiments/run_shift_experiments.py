"""
Full shift experiment pipeline.

For each held-out center (LOODO):
  1. Fit reference generator on the remaining 5 centers.
  2. Run a singleton shift scan (5 types × ~8 magnitudes = ~40 evaluations).
  3. If the best singleton improves composite by > pair_threshold, run a
     pairwise scan of the two best shift types (≤ 9 additional evaluations).
  4. Save all results as CSVs.

Output files (all in --out_dir):
  baseline_loodo_results.csv     — reference generator scores per center
  singleton_shift_results.csv    — all singleton candidates per center
  pair_shift_results.csv         — pairwise candidates (only when triggered)
  best_shift_per_center.csv      — summary: best config, improvement, etc.
  km_curves.json                 — KM curve data for selected centers (plotting)

Usage
-----
  python -m experiments.run_shift_experiments [options]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import registry
from data.loader import load_all_centers
from data.preprocess import impute_median

from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams
from generator.fitting import (
    fit_reference_generator,
    fit_domain_params,
    make_real_domain_dataset,
)
from experiments.compare_domains import domain_score, kaplan_meier
from experiments.shift_search import (
    singleton_scan,
    pair_scan,
    best_singleton,
    best_pair,
    improvement,
)


# ---------------------------------------------------------------------------
# Fold setup (shared between baseline and shift runs)
# ---------------------------------------------------------------------------

def _prepare_fold(
    centers: list[dict],
    held_idx: int,
    k_total: int,
    tau: float,
    nu: float,
    seed: int,
):
    """
    Load, impute, fit and wrap one LOODO fold.

    Returns
    -------
    held_name      : str
    held_params    : DomainParams for the held-out center
    base           : BaseParams
    A              : (p, k) shared loading matrix
    src_ds_list    : list of source DomainDatasets
    real_ds        : DomainDataset wrapping the real held-out center
    baseline_scores: dict from domain_score
    scaler         : fitted StandardScaler
    """
    held_out  = centers[held_idx]
    sources   = [c for i, c in enumerate(centers) if i != held_idx]
    hid       = held_out["center_id"]
    p         = held_out["X"].shape[1]

    # Median imputation fitted on source only
    X_src_pool = np.vstack([c["X"] for c in sources])
    _, src_medians = impute_median(X_src_pool)
    sources_imp = [{**c, "X": np.where(np.isnan(c["X"]), src_medians, c["X"])} for c in sources]
    X_held_imp  = np.where(np.isnan(held_out["X"]), src_medians, held_out["X"])

    # Fit generator on source
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

    # Wrap source domains
    src_ds_list = []
    for sp, sc in zip(source_params, sources_imp):
        X_sc_s = scaler.transform(sc["X"])
        src_ds_list.append(make_real_domain_dataset(
            X_s=X_sc_s, t_obs=sc["t"], delta=sc["delta"],
            A=A, w=w, tau=tau, params=sp, domain_id=sc["center_id"],
        ))

    # Wrap real held-out domain
    real_ds = make_real_domain_dataset(
        X_s=X_held_s, t_obs=held_out["t"], delta=held_out["delta"],
        A=A, w=w, tau=tau, params=held_params, domain_id=hid,
    )

    # Baseline synthetic
    gen = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)
    synth_ds = gen.generate_domain(held_params, domain_id=hid)
    baseline_scores = domain_score(synth_ds, real_ds, src_ds_list, tau=tau)

    return (
        registry.CENTER_NAMES.get(hid, str(hid)),
        held_params, base, A, src_ds_list, real_ds, baseline_scores, scaler,
    )


# ---------------------------------------------------------------------------
# KM curve data for plotting
# ---------------------------------------------------------------------------

def _km_data(ds, label: str) -> dict:
    """Serialise a DomainDataset's KM curve as a plain dict."""
    from generator.base_generator import DomainDataset
    t = ds.t_obs if ds.t_obs is not None else ds.T
    d = ds.delta if ds.delta is not None else np.ones(ds.n, dtype=int)
    times, S = kaplan_meier(t, d)
    return {"label": label, "times": times.tolist(), "S": S.tolist()}


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_shift_experiments(
    k_total: int = 6,
    tau: float = 12.0,
    nu: float = 1.5,
    seed: int = 42,
    pair_threshold: float = 0.10,
    out_dir: str = "outputs/tables/",
    verbose: bool = True,
) -> None:
    """
    Run baseline LOODO + singleton + conditional pair scan for all centers.

    Parameters
    ----------
    pair_threshold : minimum fractional improvement in composite score from
                     the best singleton to trigger a pair scan (default 10%).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    centers, _ = load_all_centers(registry_module=registry)

    baseline_rows   = []
    singleton_rows  = []
    pair_rows       = []
    best_rows       = []
    km_curves_data  = {}

    for held_idx in range(len(centers)):
        hid = centers[held_idx]["center_id"]
        name, held_params, base, A, src_ds_list, real_ds, b_scores, scaler = \
            _prepare_fold(centers, held_idx, k_total, tau, nu, seed)

        b_composite = b_scores["composite"]
        if verbose:
            print(f"\n=== Center {name} (id={hid})  baseline composite={b_composite:.4f} ===")

        # ---- Baseline row ----
        baseline_rows.append({
            "center_id":   hid,
            "center_name": name,
            "n":           centers[held_idx]["n"],
            **{k: v for k, v in b_scores.items() if k not in ("weights", "synth_n", "real_n")},
        })

        # ---- Baseline KM curves ----
        gen0 = SyntheticGenerator(base, shared_A_inv=A, shared_A_spu=None)
        synth_ref = gen0.generate_domain(held_params, domain_id=hid)
        km_curves_data[name] = {
            "real":      _km_data(real_ds,   "real"),
            "reference": _km_data(synth_ref, "reference"),
        }

        # ---- Singleton scan ----
        if verbose:
            print(f"  Singleton scan...")
        sing_df = singleton_scan(
            held_params=held_params, base=base, A=A,
            src_ds_list=src_ds_list, real_ds=real_ds,
            tau=tau, seed=seed, verbose=verbose,
        )
        sing_df.insert(0, "center_name", name)
        sing_df.insert(0, "center_id",   hid)
        singleton_rows.append(sing_df)

        best_sing = best_singleton(sing_df)
        best_sing_composite = float(best_sing.get("composite", b_composite))
        sing_improvement = improvement(b_composite, best_sing_composite)

        if verbose:
            print(f"  Best singleton: {best_sing.get('shift_type')} "
                  f"mag={best_sing.get('magnitude'):.4f}  "
                  f"composite={best_sing_composite:.4f}  "
                  f"Δ={sing_improvement:+.1%}")

        # ---- Best singleton KM curve ----
        from generator.shift_components import apply_shift_config
        best_sing_config = {}
        if "shift_type" in best_sing and best_sing["shift_type"] is not None:
            best_sing_config = {best_sing["shift_type"]: best_sing["magnitude"]}
        shifted_p = apply_shift_config(held_params, best_sing_config)
        synth_sing = gen0.generate_domain(shifted_p, domain_id=hid)
        km_curves_data[name]["best_singleton"] = _km_data(synth_sing, "best_singleton")

        # ---- Conditional pair scan ----
        best_p = best_sing  # default: singleton is the best
        do_pair = sing_improvement >= pair_threshold
        if do_pair:
            if verbose:
                print(f"  Pair scan triggered (improvement={sing_improvement:.1%} ≥ threshold={pair_threshold:.0%})...")
            p_df = pair_scan(
                held_params=held_params, base=base, A=A,
                src_ds_list=src_ds_list, real_ds=real_ds,
                tau=tau, best_singleton_df=sing_df, seed=seed, verbose=verbose,
            )
            if not p_df.empty:
                p_df.insert(0, "center_name", name)
                p_df.insert(0, "center_id",   hid)
                pair_rows.append(p_df)

                best_p_row = best_pair(p_df)
                p_composite = float(best_p_row.get("composite", best_sing_composite))
                p_improvement = improvement(b_composite, p_composite)

                if verbose:
                    print(f"  Best pair: {best_p_row.get('shift_type_1')} + "
                          f"{best_p_row.get('shift_type_2')}  "
                          f"composite={p_composite:.4f}  Δ={p_improvement:+.1%}")

                if p_composite < best_sing_composite:
                    best_p = best_p_row
                    # KM curve for best pair
                    pair_config = {
                        best_p_row["shift_type_1"]: best_p_row["magnitude_1"],
                        best_p_row["shift_type_2"]: best_p_row["magnitude_2"],
                    }
                    sp2 = apply_shift_config(held_params, pair_config)
                    synth_pair = gen0.generate_domain(sp2, domain_id=hid)
                    km_curves_data[name]["best_pair"] = _km_data(synth_pair, "best_pair")

        # ---- Best config summary row ----
        final_composite = float(best_p.get("composite", b_composite))
        best_rows.append({
            "center_id":           hid,
            "center_name":         name,
            "n":                   centers[held_idx]["n"],
            "composite_baseline":  round(b_composite, 4),
            "composite_best":      round(final_composite, 4),
            "improvement":         round(improvement(b_composite, final_composite), 4),
            "pair_scan_triggered": do_pair,
            "best_shift_type_1":   best_p.get("shift_type") or best_p.get("shift_type_1", "none"),
            "best_magnitude_1":    best_p.get("magnitude")  or best_p.get("magnitude_1",  None),
            "best_shift_type_2":   best_p.get("shift_type_2", None),
            "best_magnitude_2":    best_p.get("magnitude_2", None),
            "km_prev_real":        b_scores.get("km_prev_real"),
            "km_prev_synth_best":  best_p.get("km_prev_synth"),
            "km_distance_baseline":b_scores.get("km_distance"),
            "km_distance_best":    best_p.get("km_distance"),
            "auroc_gap_baseline":  b_scores.get("auroc_gap_delta"),
            "auroc_gap_best":      best_p.get("auroc_gap_delta"),
        })

    # ---- Save all outputs ----
    pd.DataFrame(baseline_rows).to_csv(out_path / "baseline_loodo_results.csv", index=False)
    pd.concat(singleton_rows, ignore_index=True).to_csv(
        out_path / "singleton_shift_results.csv", index=False)
    if pair_rows:
        pd.concat(pair_rows, ignore_index=True).to_csv(
            out_path / "pair_shift_results.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_path / "best_shift_per_center.csv", index=False)
    with open(out_path / "km_curves.json", "w") as f:
        json.dump(km_curves_data, f, indent=2)

    if verbose:
        print(f"\nAll results saved to {out_path}/")
        print(pd.DataFrame(best_rows)[
            ["center_name", "composite_baseline", "composite_best", "improvement",
             "best_shift_type_1", "best_magnitude_1"]
        ].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Shift experiment pipeline")
    p.add_argument("--k_total",         type=int,   default=6)
    p.add_argument("--tau",             type=float, default=12.0)
    p.add_argument("--nu",              type=float, default=1.5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--pair_threshold",  type=float, default=0.10,
                   help="min fractional composite improvement to trigger pair scan")
    p.add_argument("--out_dir",         type=str,
                   default="outputs/tables/")
    p.add_argument("--no_verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_shift_experiments(
        k_total=args.k_total,
        tau=args.tau,
        nu=args.nu,
        seed=args.seed,
        pair_threshold=args.pair_threshold,
        out_dir=args.out_dir,
        verbose=not args.no_verbose,
    )
