"""Shift-component audit: decompose each real transfer into measurable pieces.

Why this exists
---------------
The synthetic twin currently matches one aggregate number, the RBF-MMD, and
reproduces it by moving the latent means. But MMD is a magnitude, not a
mechanism: the same value is produced by a mean shift, a covariance shift, a
change in the observation map, or any mixture. Attributing it entirely to means
is an assumption, and it is exactly where the twin fails -- on the registry it
predicts importance weighting wins (the right method if the shift is in the
means) while the real transfer is won by CORAL (the right method if it is in the
covariances).

More is measurable without target labels than the twin currently uses: the mean
discrepancy and the covariance discrepancy are separately estimable from
unlabelled covariates. This script measures them, so a twin can reproduce the
observed *composition* of the shift instead of assuming one.

Diagnostics per transfer
------------------------
Geometry, in the source-fitted standardised space:
  d_mean   ||mu_t - mu_s||_2                  first-order discrepancy
  d_cov    ||Sigma_t - Sigma_s||_F            second-order discrepancy
  mmd      RBF-MMD                            aggregate, what the twin used
  cov_share  d_cov / (d_mean + d_cov)         composition of the discrepancy

Domain classifiers, source-vs-target, cross-validated AUROC. These say what
*kind* of boundary separates the domains, which the geometry alone does not:
  linear (LR)      first-order separability
  quadratic (QDA)  needs different covariances to beat linear
  nonlinear (RF)   higher-order structure

The gaps are the informative part. QDA >> LR indicates the discrepancy is
second-order; RF >> QDA indicates structure beyond the first two moments.

Label-dependent quantities (prevalence difference) are computed retrospectively
and reported separately, so nothing here is presented as available before
target labels exist.

All quantities are bootstrapped by resampling patients within source and within
target independently.

Usage
-----
  python -m experiments.shift_audit                        # registry only
  python -m experiments.shift_audit --data_dir I3LUNG_DATA # + i3lung
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils.io import save_table_csv
from utils.shift_diag import rbf_mmd

_TABLES = _ROOT / "outputs" / "tables"


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------

def _standardise(Xs: np.ndarray, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute and scale with statistics fitted on the SOURCE only.

    Source-fitted preprocessing is what a deployed model would use, and it also
    keeps the diagnostics on one footing: the source lands at mean 0 and unit
    variance by construction, so d_mean and d_cov read directly as the target's
    displacement from it.
    """
    imp = SimpleImputer(strategy="median").fit(Xs)
    sc = StandardScaler().fit(imp.transform(Xs))
    return sc.transform(imp.transform(Xs)), sc.transform(imp.transform(Xt))


def geometry(Xs_s: np.ndarray, Xt_s: np.ndarray) -> dict:
    """First- and second-order discrepancies plus the aggregate MMD."""
    mu_s, mu_t = Xs_s.mean(axis=0), Xt_s.mean(axis=0)
    d_mean = float(np.linalg.norm(mu_t - mu_s))

    # Shrinkage covariances: with p up to ~20 and centres as small as ~190
    # patients, the sample covariance is badly conditioned and its Frobenius
    # distance is dominated by estimation noise.
    cov_s = LedoitWolf(assume_centered=False).fit(Xs_s).covariance_
    cov_t = LedoitWolf(assume_centered=False).fit(Xt_s).covariance_
    d_cov = float(np.linalg.norm(cov_t - cov_s, ord="fro"))

    denom = d_mean + d_cov
    return {
        "d_mean": round(d_mean, 5),
        "d_cov": round(d_cov, 5),
        "cov_share": round(d_cov / denom, 5) if denom > 1e-12 else float("nan"),
        "mmd": round(float(rbf_mmd(Xs_s, Xt_s)), 5),
    }


def domain_classifiers(Xs_s: np.ndarray, Xt_s: np.ndarray, seed: int = 0,
                       n_splits: int = 5) -> dict:
    """Cross-validated AUROC of source-vs-target classifiers.

    Domains are usually very unequal in size (555 vs 2894 on the registry), so
    AUROC rather than accuracy, and stratified folds.
    """
    X = np.vstack([Xs_s, Xt_s])
    y = np.r_[np.zeros(len(Xs_s), dtype=int), np.ones(len(Xt_s), dtype=int)]
    if len(np.unique(y)) < 2:
        return {}
    k = max(2, min(n_splits, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    models = {
        "clf_linear": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "clf_quadratic": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "clf_nonlinear": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1),
    }
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, m in models.items():
            try:
                s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
                out[name] = round(float(np.mean(s)), 5)
            except Exception:                                    # noqa: BLE001
                out[name] = float("nan")
    # The gaps carry the mechanism signal, not the levels.
    out["clf_quad_minus_linear"] = round(out.get("clf_quadratic", np.nan)
                                         - out.get("clf_linear", np.nan), 5)
    out["clf_nonlin_minus_quad"] = round(out.get("clf_nonlinear", np.nan)
                                         - out.get("clf_quadratic", np.nan), 5)
    return out


def audit_transfer(src_df: pd.DataFrame, tgt_df: pd.DataFrame,
                   feats: list[str], label: str, name: str,
                   n_boot: int = 500, n_boot_clf: int = 50,
                   seed: int = 0, verbose: bool = True) -> tuple[dict, list[dict]]:
    """Point estimates plus bootstrap CIs for one source->target transfer."""
    Xs = src_df[feats].to_numpy(float)
    Xt = tgt_df[feats].to_numpy(float)
    ys = src_df[label].to_numpy().astype(int)
    yt = tgt_df[label].to_numpy().astype(int)

    Xs_s, Xt_s = _standardise(Xs, Xt)

    row = {"transfer": name, "n_src": len(Xs), "n_tgt": len(Xt),
           "n_features": len(feats)}
    row.update(geometry(Xs_s, Xt_s))
    row.update(domain_classifiers(Xs_s, Xt_s, seed=seed))
    # Retrospective: needs target labels, so it is not available at deployment.
    row["prev_src_retro"] = round(float(ys.mean()), 5)
    row["prev_tgt_retro"] = round(float(yt.mean()), 5)
    row["prev_diff_retro"] = round(float(yt.mean() - ys.mean()), 5)

    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {}
    for b in range(n_boot):
        i = rng.integers(0, len(Xs), len(Xs))
        j = rng.integers(0, len(Xt), len(Xt))
        a, c = _standardise(Xs[i], Xt[j])
        g = geometry(a, c)
        # Classifiers are refit on only the first n_boot_clf replicates: five
        # CV folds of a random forest per replicate would dominate the runtime
        # for a CI on a quantity whose point estimate is already cross-validated.
        if b < n_boot_clf:
            g.update(domain_classifiers(a, c, seed=seed + b))
        g["prev_diff_retro"] = float(yt[j].mean() - ys[i].mean())
        for k, v in g.items():
            boot.setdefault(k, []).append(v)

    for k, vals in boot.items():
        v = np.asarray(vals, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        row[f"{k}_lo"] = round(float(np.percentile(v, 2.5)), 5)
        row[f"{k}_hi"] = round(float(np.percentile(v, 97.5)), 5)

    if verbose:
        print(f"  {name:<28} d_mean {row['d_mean']:.3f}  d_cov {row['d_cov']:.3f}  "
              f"cov_share {row['cov_share']:.3f}  mmd {row['mmd']:.4f}  "
              f"lin {row.get('clf_linear', float('nan')):.3f}  "
              f"quad-lin {row.get('clf_quad_minus_linear', float('nan')):+.3f}  "
              f"nonlin-quad {row.get('clf_nonlin_minus_quad', float('nan')):+.3f}",
              flush=True)

    return row, []


# ---------------------------------------------------------------------------
# Transfer enumeration
# ---------------------------------------------------------------------------

def registry_transfers(tau: float) -> list[tuple]:
    from experiments.run_real_data import load_public_transfer
    out = []
    for s, t in [("Rotterdam", "GBSG2"), ("GBSG2", "Rotterdam")]:
        src, tgt, feats = load_public_transfer("rotterdam_gbsg", s, t, tau=tau)
        out.append((f"registry_{s}_{t}_tau{int(tau)}", src, tgt, feats, "event"))
    return out


def i3lung_transfers(data_dir: str, min_n: int = 150) -> list[tuple]:
    from experiments.run_real_data import load_i3lung, load_i3lung_all
    centers, _feats = load_i3lung_all(data_dir, min_n=min_n)
    names = sorted(centers)
    out = []
    for s, t in itertools.permutations(names, 2):
        src, tgt, feats = load_i3lung(data_dir, s, t)
        out.append((f"i3lung_{s}_{t}", src, tgt, feats, "OS_6"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="I3LUNG_DATA (omesso: solo registry)")
    ap.add_argument("--tau", type=float, default=36.0)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--n_boot_clf", type=int, default=50)
    ap.add_argument("--min_n", type=int, default=150)
    ap.add_argument("--out_prefix", default="shift_audit")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    transfers = registry_transfers(args.tau)
    if args.data_dir:
        transfers += i3lung_transfers(args.data_dir, min_n=args.min_n)
    else:
        print("NOTA: nessuna dir I3LUNG — solo il registry pubblico.\n")

    print(f"Audit di {len(transfers)} transfer "
          f"(n_boot={args.n_boot}, n_boot_clf={args.n_boot_clf})\n")
    rows = []
    for name, src, tgt, feats, label in transfers:
        row, _ = audit_transfer(src, tgt, feats, label, name,
                                n_boot=args.n_boot, n_boot_clf=args.n_boot_clf)
        rows.append(row)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / f"{args.out_prefix}_transfers.csv")

    df = pd.DataFrame(rows)
    pooled = [{
        "n_transfers": len(df),
        "d_mean_median": round(float(df["d_mean"].median()), 5),
        "d_cov_median": round(float(df["d_cov"].median()), 5),
        "cov_share_median": round(float(df["cov_share"].median()), 5),
        "cov_share_min": round(float(df["cov_share"].min()), 5),
        "cov_share_max": round(float(df["cov_share"].max()), 5),
        "mmd_median": round(float(df["mmd"].median()), 5),
        "clf_linear_median": round(float(df["clf_linear"].median()), 5),
        "quad_minus_linear_median": round(float(df["clf_quad_minus_linear"].median()), 5),
        "nonlin_minus_quad_median": round(float(df["clf_nonlin_minus_quad"].median()), 5),
        "n_transfers_quad_beats_linear": int((df["clf_quad_minus_linear"] > 0.02).sum()),
        "n_transfers_nonlin_beats_quad": int((df["clf_nonlin_minus_quad"] > 0.02).sum()),
    }]
    save_table_csv(pooled, _TABLES / f"{args.out_prefix}_pooled.csv")

    print(f"\n  cov_share mediana {pooled[0]['cov_share_median']:.3f} "
          f"(range {pooled[0]['cov_share_min']:.3f}-{pooled[0]['cov_share_max']:.3f})")
    print(f"  QDA batte il lineare di >0.02 in "
          f"{pooled[0]['n_transfers_quad_beats_linear']}/{len(df)} transfer")
    print(f"  il nonlineare batte QDA di >0.02 in "
          f"{pooled[0]['n_transfers_nonlin_beats_quad']}/{len(df)} transfer")
    print(f"\n  Salvato: {_TABLES}/{args.out_prefix}_{{transfers,pooled}}.csv")


if __name__ == "__main__":
    main()
