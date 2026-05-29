"""
Real-data experiment: I3LUNG clinical features (CB), INT → GHD transfer.

Source: INT  (n≈668, after dropping missing OS_6)
Target: GHD  (n=430)
Features: 11 CB features from features.json (clinical + blood)
Label: OS_6 (6-month overall survival, binary)

Protocol mirrors the synthetic benchmark:
  - 10 random seeds, 80/20 stratified source train/val split
  - Median imputation fitted on source train only
  - StandardScaler fitted on source train only
  - Target labels never used during training or HP selection
  - Same methods: lr_erm, coral, prior_recalib, dann, cdan

Usage
-----
  python -m experiments.run_real_data --data_dir /path/to/I3LUNG_DATA
  python -m experiments.run_real_data --data_dir /path/to/I3LUNG_DATA --out_prefix i3lung_cb
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.metrics import auroc, auprc, ece, f1, sensitivity, specificity, prevalence
from utils.io import save_table_csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TABLES  = _ROOT / "outputs" / "tables"
_FIGURES = _ROOT / "outputs" / "figures"
_LR_KWARGS = dict(max_iter=500, solver="lbfgs", C=1.0, random_state=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_i3lung(
    data_dir: str | Path,
    source_center: str = "INT",
    target_center: str = "GHD",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load and merge cb.csv + outcomes.csv; return (source_df, target_df, feature_names).
    Rows with missing OS_6 are dropped.
    """
    data_dir = Path(data_dir)
    cb       = pd.read_csv(data_dir / "cb.csv")
    outcomes = pd.read_csv(data_dir / "outcomes.csv")

    with open(data_dir / "features.json") as f:
        feats = json.load(f)
    cb_feats = feats["CB"]

    df = cb.merge(outcomes[["Subject", "OS_6"]], on="Subject")
    df = df.dropna(subset=["OS_6"])

    src = df[df["CENTER"] == source_center][cb_feats + ["OS_6"]].reset_index(drop=True)
    tgt = df[df["CENTER"] == target_center][cb_feats + ["OS_6"]].reset_index(drop=True)

    return src, tgt, cb_feats


# ---------------------------------------------------------------------------
# Score helper
# ---------------------------------------------------------------------------

def _score(y_true: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "auroc":       round(auroc(y_true, probs), 4),
        "auprc":       round(auprc(y_true, probs), 4),
        "ece":         round(ece(y_true, probs), 4),
        "f1":          round(f1(y_true, probs), 4),
        "sensitivity": round(sensitivity(y_true, probs), 4),
        "specificity": round(specificity(y_true, probs), 4),
        "prevalence":  round(prevalence(y_true), 4),
        "n":           len(y_true),
    }


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def _run_seed(
    src: pd.DataFrame,
    tgt: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    use_iwcv: bool = True,
) -> list[dict]:
    """Fit all methods for one seed; return list of result dicts."""
    from experiments.adaptation_methods import fit_coral, fit_dann, fit_cdan
    from experiments.baselines import fit_prior_recalib

    X_src = src[feature_cols].values
    y_src = src["OS_6"].values.astype(int)
    X_tgt = tgt[feature_cols].values
    y_tgt = tgt["OS_6"].values.astype(int)

    # 80/20 stratified split on source
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_src, y_src, test_size=0.2, random_state=seed, stratify=y_src,
    )

    # Imputer fitted on source train
    imputer = SimpleImputer(strategy="median")
    X_tr  = imputer.fit_transform(X_tr)
    X_val = imputer.transform(X_val)
    X_tgt_imp = imputer.transform(X_tgt)

    # Scaler fitted on source train
    scaler = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_val_s  = scaler.transform(X_val)
    X_tgt_s  = scaler.transform(X_tgt_imp)

    results = []

    # --- LR ERM ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(**_LR_KWARGS)
        lr.fit(X_tr_s, y_tr)
    p_val = lr.predict_proba(X_val_s)[:, 1]
    p_tgt = lr.predict_proba(X_tgt_s)[:, 1]
    results.append({
        "method": "lr_erm",
        "source_pooled": _score(y_val, p_val),
        "target": _score(y_tgt, p_tgt),
    })

    # --- Prior recalibration ---
    pi_src = float(y_tr.mean())
    p_raw  = lr.predict_proba(X_tgt_s)[:, 1]
    pi_tgt = float(p_raw.mean())
    for _ in range(50):
        num   = p_raw * (pi_tgt / pi_src)
        den   = num + (1 - p_raw) * ((1 - pi_tgt) / (1 - pi_src))
        p_adj = num / np.clip(den, 1e-12, None)
        pi_new = float(p_adj.mean())
        if abs(pi_new - pi_tgt) < 1e-7:
            break
        pi_tgt = pi_new
    results.append({
        "method": "prior_recalib",
        "source_pooled": _score(y_val, p_val),
        "target": _score(y_tgt, p_adj),
    })

    # --- Platt scaling ---
    from scipy.optimize import minimize_scalar as _min_scalar
    logits_val = lr.decision_function(X_val_s)
    logits_tgt = lr.decision_function(X_tgt_s)
    def _nll(T):
        p = 1 / (1 + np.exp(-logits_val / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
    T_opt = _min_scalar(_nll, bounds=(0.01, 10.0), method="bounded").x
    p_tgt_platt = 1 / (1 + np.exp(-logits_tgt / T_opt))
    p_val_platt = 1 / (1 + np.exp(-logits_val / T_opt))
    results.append({
        "method": "platt_scaling",
        "source_pooled": _score(y_val, p_val_platt),
        "target": _score(y_tgt, p_tgt_platt),
    })

    # --- CORAL ---
    from adapt.feature_based import CORAL as _CORAL
    coral = _CORAL(
        estimator=LogisticRegression(**_LR_KWARGS),
        Xt=X_tgt_s, lambda_=1.0, verbose=0, random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        coral.fit(X_tr_s, y_tr)
    p_val_c = coral.estimator_.predict_proba(coral.transform(X_val_s, domain="src"))[:, 1]
    p_tgt_c = coral.estimator_.predict_proba(X_tgt_s)[:, 1]
    results.append({
        "method": "coral",
        "source_pooled": _score(y_val, p_val_c),
        "target": _score(y_tgt, p_tgt_c),
    })

    # Build DomainDataset-compatible objects for DANN/CDAN
    # We create minimal objects with .x, .y, .domain_id, .n attributes
    class _DS:
        def __init__(self, x, y, domain_id):
            self.x = x
            self.y = y.astype(float)
            self.domain_id = domain_id
            self.n = len(y)
            # DANN/CDAN also need u_inv, u_spu — not used here
            self.u_inv = x
            self.u_spu = x[:, :2]

    # Pass already-imputed (but not yet scaled) data — fit_dann/fit_cdan
    # will do their own scaling internally. We pass raw imputed arrays.
    src_ds = [_DS(X_tr, y_tr, domain_id=0)]   # 80% source train
    tgt_ds  = _DS(X_tgt_imp, y_tgt, domain_id=1)

    # --- DANN ---
    try:
        r_dann = fit_dann(src_ds, tgt_ds, seed=seed, use_iwcv=use_iwcv)
        results.append({"method": "dann",
                        "source_pooled": r_dann.get("source_pooled", {}),
                        "target": r_dann.get("target", {})})
    except Exception as e:
        warnings.warn(f"DANN failed seed={seed}: {e}")
        results.append({"method": "dann", "source_pooled": {}, "target": {}})

    # --- CDAN ---
    try:
        r_cdan = fit_cdan(src_ds, tgt_ds, seed=seed, use_iwcv=use_iwcv)
        results.append({"method": "cdan",
                        "source_pooled": r_cdan.get("source_pooled", {}),
                        "target": r_cdan.get("target", {})})
    except Exception as e:
        warnings.warn(f"CDAN failed seed={seed}: {e}")
        results.append({"method": "cdan", "source_pooled": {}, "target": {}})

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["src_auroc", "tgt_auroc", "tgt_ece", "tgt_f1",
                "tgt_sensitivity", "tgt_specificity", "gap"]
_METHODS     = ["lr_erm", "prior_recalib", "platt_scaling", "coral", "dann", "cdan"]


def _flatten_real(r: dict, seed: int) -> dict:
    sp  = r.get("source_pooled", {})
    tgt = r.get("target", {})
    sa  = sp.get("auroc",  float("nan"))
    ta  = tgt.get("auroc", float("nan"))
    gap = round(sa - ta, 4) if not (np.isnan(sa) or np.isnan(ta)) else float("nan")
    return {
        "seed":            seed,
        "method":          r["method"],
        "src_auroc":       sa,
        "tgt_auroc":       ta,
        "tgt_ece":         tgt.get("ece",         float("nan")),
        "tgt_f1":          tgt.get("f1",           float("nan")),
        "tgt_sensitivity": tgt.get("sensitivity",  float("nan")),
        "tgt_specificity": tgt.get("specificity",  float("nan")),
        "gap":             gap,
    }


def _aggregate_real(raw: list[dict]) -> list[dict]:
    agg = []
    for m in _METHODS:
        subset = [r for r in raw if r["method"] == m]
        if not subset:
            continue
        row = {"method": m, "n_seeds": len(subset)}
        for key in _METRIC_KEYS:
            vals = np.array([r[key] for r in subset], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_ci95"] = float("nan")
                continue
            mean = float(vals.mean())
            std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            se   = std / np.sqrt(len(vals))
            tc   = sp_stats.t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
            row[f"{key}_mean"] = round(mean, 4)
            row[f"{key}_ci95"] = round(tc * se, 4)
        agg.append(row)
    return agg


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_real_results(agg: list[dict], n_seeds: int, source: str, target: str) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  I3LUNG REAL DATA — {source} → {target}  (mean ± 95% CI, {n_seeds} seeds)")
    print(sep)
    col = 18
    labels = _METHODS
    metrics = [("tgt_auroc", "Target AUROC ↑"),
               ("gap",       "AUROC gap (src−tgt) ↓"),
               ("tgt_ece",   "Target ECE ↓"),
               ("tgt_f1",    "Target F1 ↑")]
    idx = {r["method"]: r for r in agg}
    for metric, title in metrics:
        print(f"\n  {title}")
        print(f"  {'method':20s}" + "".join(f"  {m:>{col}s}" for m in labels))
        print("  " + "-" * (22 + (col + 2) * len(labels)))
        line = f"  {'':20s}"
        for m in labels:
            r = idx.get(m, {})
            mean = r.get(f"{metric}_mean", float("nan"))
            ci   = r.get(f"{metric}_ci95", float("nan"))
            s = f"{mean:.3f}±{ci:.3f}" if not np.isnan(mean) else "—"
            line += f"  {s:>{col}s}"
        print(line)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_real_data(
    data_dir: str | Path,
    source_center: str = "INT",
    target_center: str = "GHD",
    seeds: list[int] | None = None,
    use_iwcv: bool = True,
    out_prefix: str = "i3lung_cb",
    verbose: bool = True,
) -> dict:
    if seeds is None:
        seeds = list(range(10))

    src_df, tgt_df, feature_cols = load_i3lung(data_dir, source_center, target_center)

    if verbose:
        print(f"\nI3LUNG real data: {source_center} (n={len(src_df)}) → {target_center} (n={len(tgt_df)})")
        print(f"Features ({len(feature_cols)}): {feature_cols}")
        print(f"OS_6 prevalence — {source_center}: {src_df['OS_6'].mean():.3f}  "
              f"{target_center}: {tgt_df['OS_6'].mean():.3f}")

    all_rows: list[dict] = []
    for seed in seeds:
        if verbose:
            print(f"  [Seed {seed}]", flush=True)
        results = _run_seed(src_df, tgt_df, feature_cols, seed=seed, use_iwcv=use_iwcv)
        for r in results:
            all_rows.append(_flatten_real(r, seed))

    agg = _aggregate_real(all_rows)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(all_rows, _TABLES / f"{out_prefix}_raw.csv")
    save_table_csv(agg,      _TABLES / f"{out_prefix}_summary.csv")

    if verbose:
        _print_real_results(agg, len(seeds), source_center, target_center)
        print(f"\n  Saved: {_TABLES}/{out_prefix}_raw.csv")
        print(f"  Saved: {_TABLES}/{out_prefix}_summary.csv")

    return {"raw": all_rows, "aggregated": agg}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHIFT-SURV real data experiment — I3LUNG CB features, INT→GHD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to I3LUNG_DATA directory.")
    parser.add_argument("--source", type=str, default="INT",
                        help="Source center name.")
    parser.add_argument("--target", type=str, default="GHD",
                        help="Target center name.")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default="i3lung_cb")
    parser.add_argument("--no_iwcv", dest="use_iwcv", action="store_false", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_real_data(
        data_dir=args.data_dir,
        source_center=args.source,
        target_center=args.target,
        seeds=list(range(args.seed_start, args.seed_start + args.n_seeds)),
        use_iwcv=args.use_iwcv,
        out_prefix=args.out_prefix,
        verbose=not args.quiet,
    )
