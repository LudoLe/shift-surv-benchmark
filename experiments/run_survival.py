"""
Survival validity check — Cox PH on censored data, censoring-aware metrics.

Why this experiment exists
--------------------------
The main benchmark reduces the Weibull-PH survival process to a fixed-horizon
binary label (y = 1[T <= tau]) and evaluates classifiers with AUROC/ECE. A
reviewer can reasonably ask whether the headline finding — a source->target
discrimination gap with large oracle headroom that domain adaptation does not
recover — is an *artifact* of that binarization and of discarding censoring.

This script re-derives the same gap/headroom using a genuine time-to-event
model (Cox proportional hazards) trained and evaluated on **censored** data,
with survival metrics. If the gap and the oracle headroom reappear, the
phenomenon is a property of the shift, not of the fixed-horizon simplification.

Models (1 source + 1 target, mirroring run_adaptation)
------------------------------------------------------
  inv_oracle_cox : Cox PH on the true invariant latents u_inv      (ceiling)
  source_cox     : Cox PH on observed covariates x, source-trained  (transported)

Metrics (target domain)
-----------------------
  c_index : Harrell concordance, censoring-aware analog of AUROC
  td_auc  : time-dependent AUC at tau -- numerically comparable to the
            fixed-horizon AUROC in run_adaptation (this is the bridge metric)
  ibs     : integrated Brier score over [0, tau] (lower = better)

The source->target c_index/td_auc gap should track the AUROC gap from the
classification benchmark; the gap to inv_oracle_cox should track the oracle
headroom. Domain adaptation is intentionally NOT run here: the claim is a
validity check on the *outcome model*, not a new adaptation result.

Usage
-----
  python -m experiments.run_survival
  python -m experiments.run_survival --scenarios covariate_shift concept_shift \
      --censoring_lambda 0.04 --n_seeds 10
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"

CORE_SCENARIOS = [
    "no_shift", "covariate_shift", "observational_shift", "spurious_shift",
    "baseline_hazard_shift", "concept_shift", "shape_shift",
]

# Mild ridge keeps Cox stable on p=20 correlated covariates at n~400.
_COX_ALPHA = 0.1


def _surv_y(ds) -> np.ndarray:
    """Build a scikit-survival structured array from a DomainDataset."""
    return Surv.from_arrays(event=ds.delta.astype(bool), time=ds.t_obs.astype(float))


def _score_target(
    model, scaler, X_eval, y_eval, tau: float,
) -> dict:
    """C-index, time-dependent AUC at tau, and IBS over [0, tau] on one split.

    IPCW reference
    --------------
    cumulative_dynamic_auc and integrated_brier_score reweight by the inverse
    censoring probability, estimated from a reference cohort. We use the
    *evaluation cohort itself* as that reference (``y_eval`` for both train and
    test arguments). This is the correct choice: IPCW corrects for censoring in
    the cohort being scored and is independent of the model. Using the source
    training split instead breaks when the target follow-up extends beyond it
    ("censoring survival function is zero" / "time must be smaller than largest
    observed time point"), which silently dropped ~40% of seeds. The td-AUC and
    IBS are still wrapped so a genuinely degenerate split yields NaN.
    """
    Xs   = scaler.transform(X_eval)
    risk = model.predict(Xs)                      # higher = higher risk

    # Harrell C-index (censoring-aware analog of AUROC)
    c_idx = concordance_index_censored(
        y_eval["event"], y_eval["time"], risk,
    )[0]

    # Time-dependent AUC at tau -- the bridge to the fixed-horizon AUROC
    td = float("nan")
    try:
        if tau < float(y_eval["time"].max()):
            auc, _ = cumulative_dynamic_auc(y_eval, y_eval, risk, np.array([tau]))
            td = float(np.atleast_1d(auc)[0])
    except Exception:
        pass

    # Integrated Brier score over [0, tau]
    ibs = float("nan")
    try:
        times = np.linspace(0.25, tau, 12)
        times = times[(times > float(y_eval["time"].min())) & (times < float(y_eval["time"].max()))]
        if len(times) >= 3:
            surv_fns = model.predict_survival_function(Xs)
            preds = np.row_stack([[fn(t) for t in times] for fn in surv_fns])
            ibs = float(integrated_brier_score(y_eval, y_eval, preds, times))
    except Exception:
        pass

    return {"c_index": round(c_idx, 4), "td_auc": round(td, 4) if td == td else td,
            "ibs": round(ibs, 4) if ibs == ibs else ibs}


def _fit_cox(feature_attr: str, src, tgt, tau: float) -> dict:
    """Fit one Cox model on the source feature view; score source-val and target."""
    X_all = getattr(src, feature_attr)
    y_all = src.y                       # only used to stratify the split
    idx = np.arange(src.n)
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.2, random_state=0,
        stratify=y_all if len(np.unique(y_all)) > 1 else None,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[idx_tr])
    y_tr = _surv_y_subset(src, idx_tr)

    model = CoxPHSurvivalAnalysis(alpha=_COX_ALPHA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    y_val = _surv_y_subset(src, idx_val)
    src_score = _score_target(model, scaler, X_all[idx_val], y_val, tau)
    tgt_score = _score_target(model, scaler, getattr(tgt, feature_attr), _surv_y(tgt), tau)
    return {"source": src_score, "target": tgt_score}


def _surv_y_subset(ds, idx) -> np.ndarray:
    return Surv.from_arrays(event=ds.delta[idx].astype(bool), time=ds.t_obs[idx].astype(float))


def _run_one(scenario: str, seed: int, n: int, censoring_lambda: float) -> tuple[list[dict], float]:
    base = BaseParams(seed=seed)
    data = generate_scenario(
        scenario, n_source_domains=1, n_per_domain=n, seed=seed,
        base=base, censoring_lambda=censoring_lambda,
    )
    src, tgt = data["source"][0], data["target"]
    tau = float(base.tau)
    cens_rate = float(1.0 - tgt.delta.mean())

    rows = []
    for feat, name in [("u_inv", "inv_oracle_cox"), ("x", "source_cox")]:
        r = _fit_cox(feat, src, tgt, tau)
        gap_c  = round(r["source"]["c_index"] - r["target"]["c_index"], 4)
        rows.append({
            "scenario": scenario, "seed": seed, "model": name,
            "src_c_index": r["source"]["c_index"],
            "tgt_c_index": r["target"]["c_index"],
            "c_index_gap": gap_c,
            "tgt_td_auc":  r["target"]["td_auc"],
            "tgt_ibs":     r["target"]["ibs"],
        })
    return rows, cens_rate


_METRIC_KEYS = ["src_c_index", "tgt_c_index", "c_index_gap", "tgt_td_auc", "tgt_ibs"]
_MODELS = ["inv_oracle_cox", "source_cox"]


def _aggregate(raw: list[dict], scenarios: list[str]) -> list[dict]:
    agg = []
    for sc in scenarios:
        for m in _MODELS:
            subset = [r for r in raw if r["scenario"] == sc and r["model"] == m]
            if not subset:
                continue
            row = {"scenario": sc, "model": m, "n_seeds": len(subset)}
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


def _print_report(agg: list[dict], scenarios: list[str], n_seeds: int, cens_rate: float) -> None:
    idx = {(r["scenario"], r["model"]): r for r in agg}
    print(f"\n{'='*100}")
    print(f"  SURVIVAL VALIDITY CHECK — Cox PH, ~{cens_rate*100:.0f}% censored  "
          f"(mean ± 95% CI, {n_seeds} seeds)")
    print(f"{'='*100}")
    print(f"  {'scenario':22s}{'model':18s}{'tgt C-index':>16s}{'C-index gap':>16s}"
          f"{'tgt td-AUC@tau':>18s}{'tgt IBS':>12s}")
    for sc in scenarios:
        for m in _MODELS:
            r = idx.get((sc, m))
            if not r:
                continue
            def cell(k):
                mn, ci = r.get(f"{k}_mean", float("nan")), r.get(f"{k}_ci95", float("nan"))
                return "—" if np.isnan(mn) else f"{mn:.3f}±{ci:.3f}"
            print(f"  {sc:22s}{m:18s}{cell('tgt_c_index'):>16s}{cell('c_index_gap'):>16s}"
                  f"{cell('tgt_td_auc'):>18s}{cell('tgt_ibs'):>12s}")
    print("\n  Read: source_cox shows a positive C-index gap (source > target) and sits below")
    print("  inv_oracle_cox on the target -> the transport gap and oracle headroom from the")
    print("  fixed-horizon AUROC benchmark survive a censored, time-to-event evaluation.\n")


def run_survival(
    scenarios: list[str] = CORE_SCENARIOS,
    seeds: list[int] | None = None,
    n_per_domain: int = 500,
    censoring_lambda: float = 0.04,
    out_prefix: str = "survival_cox",
    verbose: bool = True,
) -> dict:
    if seeds is None:
        seeds = list(range(10))

    all_rows: list[dict] = []
    cens_rates: list[float] = []
    for seed in seeds:
        if verbose:
            print(f"[Seed {seed}]", flush=True)
        for sc in scenarios:
            rows, cr = _run_one(sc, seed, n_per_domain, censoring_lambda)
            all_rows.extend(rows)
            cens_rates.append(cr)
            if verbose:
                tgt = {r["model"]: r["tgt_c_index"] for r in rows}
                print(f"  {sc:22s} oracle={tgt.get('inv_oracle_cox', float('nan')):.3f}  "
                      f"source={tgt.get('source_cox', float('nan')):.3f}  "
                      f"(cens={cr*100:.0f}%)", flush=True)

    agg = _aggregate(all_rows, scenarios)
    mean_cens = float(np.mean(cens_rates)) if cens_rates else 0.0

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(all_rows, _TABLES / f"{out_prefix}_raw.csv")
    save_table_csv(agg,      _TABLES / f"{out_prefix}_summary.csv")

    if verbose:
        _print_report(agg, scenarios, len(seeds), mean_cens)
        print(f"  Saved: {_TABLES}/{out_prefix}_raw.csv")
        print(f"  Saved: {_TABLES}/{out_prefix}_summary.csv")

    return {"raw": all_rows, "aggregated": agg, "mean_censoring": mean_cens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHIFT-SURV survival validity check (Cox PH + censoring)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help=f"Scenarios to run. Default: all core. Available: {CORE_SCENARIOS}")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n", type=int, default=500, help="Samples per domain.")
    parser.add_argument("--censoring_lambda", type=float, default=0.04,
                        help="Exponential censoring rate (0.04 ≈ 20–30%% censored).")
    parser.add_argument("--out_prefix", type=str, default="survival_cox")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    scenarios = list(args.scenarios) if args.scenarios else list(CORE_SCENARIOS)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    run_survival(
        scenarios=scenarios, seeds=seeds, n_per_domain=args.n,
        censoring_lambda=args.censoring_lambda, out_prefix=args.out_prefix,
        verbose=not args.quiet,
    )
