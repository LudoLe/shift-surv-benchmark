"""
Decision-direction geometry: *why* unsupervised methods fail to recover
discrimination under measurement/observational shift.

The main benchmark shows empirically that no unsupervised method recovers target
discrimination beyond ERM, while the invariant oracle reveals large recoverable
headroom.  This module explains the mechanism, using the known DGP, without any
target labels leaking into training.

Idea
----
A linear classifier is a *decision direction* beta in x-space.  For a domain we
can compute the direction a target-trained model *would* use (beta_tgt*, a LR fit
on target x with TRUE labels -- a diagnostic reference only, never used to train
any method below).  We then ask, for each method, how misaligned its decision
direction is from beta_tgt* (angle in degrees), and how much alignment reduces
the marginal discrepancy (source/target covariance distance).

Methods with an *exact* linear decision direction (compared here):
  ERM, CORAL (feature alignment), IW-ERM (reweighting),
  GroupDRO and IRM (domain generalization; need >=2 source centers).
DANN / CDAN are non-linear (MLP encoder); with --with_dann we add a *linear
surrogate* of their decision function (a LR fit to their target logits) -- an
approximation, clearly labelled, and the only part that needs TensorFlow.

Findings
--------
  1. angle(beta_ERM, beta_tgt*) grows with the A_inv rotation and tracks the
     AUROC gap ~1:1 -- the transport gap *is* a rotation of the decision
     direction.
  2. Every unsupervised method leaves that angle essentially unchanged (all the
     per-method angles overlap with ERM), even though CORAL shrinks the
     covariance distance 40-60%.  Alignment moves the marginal P(x), not the
     label-relevant direction, which is unidentifiable from unlabeled target data.

The linear methods are numpy/scipy only (CORAL/IRM/GroupDRO reimplemented in
closed form / directly), so the default path is unit-testable without the
Keras-2 stack.

Usage
-----
  python -m experiments.decision_geometry                       # synthetic sweep + table + figure
  python -m experiments.decision_geometry --with_dann           # + DANN/CDAN linear surrogate (needs TF)
  python -m experiments.decision_geometry --real --data_dir I3LUNG_DATA          # i3lung LOCO
  python -m experiments.decision_geometry --real --data_dir I3LUNG_DATA --with_dann
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.io import save_table_csv

_TABLES  = _ROOT / "outputs" / "tables"
_FIGURES = _ROOT / "outputs" / "figures"
_LR = dict(max_iter=500, solver="lbfgs", C=1.0)

# Methods with an exact linear decision direction (always computed).
_LINEAR_METHODS = ["erm", "coral", "iw", "group_dro", "irm"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _msqrt(C: np.ndarray, inv: bool = False) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    d = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return V @ np.diag(d) @ V.T


def _coral_align(Xs: np.ndarray, Xt: np.ndarray, lam: float = 1.0) -> np.ndarray:
    Cs = np.cov(Xs, rowvar=False) + lam * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + lam * np.eye(Xt.shape[1])
    return Xs @ (_msqrt(Cs, inv=True) @ _msqrt(Ct))


def _cov_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False)))


def _lr_dir(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return LogisticRegression(**_LR).fit(X, y).coef_[0]


def _iw_direction(Xs, ys, Xt) -> np.ndarray:
    Xd = np.vstack([Xs, Xt])
    yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LogisticRegression(**_LR).fit(Xd, yd)
    p = clf.predict_proba(Xs)[:, 1]
    w = np.clip(p / np.clip(1 - p, 1e-6, None), 0.05, 20.0)
    return LogisticRegression(**_LR).fit(Xs, ys, sample_weight=w).coef_[0]


def _groupdro_direction(groups_scaled, seed=0) -> np.ndarray:
    """Linear GroupDRO direction (theta) via the hardened trainer, in the caller's
    scaled space. 80/20 per-group split gives the source-val early stopping."""
    from experiments.group_dro import train_group_dro
    tr, Xv, yv = [], [], []
    for Xg, yg in groups_scaled:
        Xtr, Xval, ytr, yval = train_test_split(Xg, yg, test_size=0.2, random_state=seed,
                                                stratify=yg if len(np.unique(yg)) > 1 else None)
        tr.append((Xtr, ytr.astype(float)))
        Xv.append(Xval); yv.append(yval)
    theta, b, q = train_group_dro(tr, X_val=np.vstack(Xv),
                                  y_val=np.concatenate(yv).astype(float))
    return theta


def _irm_direction(groups_scaled, lam=1e2, n_iter=500, seed=0) -> np.ndarray:
    """Linear IRMv1 direction (theta), numpy/scipy only (mirrors adaptation_methods.fit_irm)."""
    from scipy.optimize import minimize
    from scipy.special import expit
    envs = [(X, y.astype(float)) for X, y in groups_scaled]
    p = envs[0][0].shape[1]

    def obj_grad(theta):
        loss, grad = 0.0, np.zeros(p)
        for X, y in envs:
            n_e = len(y)
            z = X @ theta
            s = expit(z)
            loss += float(np.mean(np.log1p(np.exp(-(2 * y - 1) * z))))
            grad += X.T @ (s - y) / n_e
            g_e = float(np.mean((s - y) * z))
            loss += lam * g_e * g_e
            grad += lam * 2.0 * g_e * (X.T @ (s * (1 - s) * z + (s - y)) / n_e)
        return loss / len(envs), grad / len(envs)

    rng = np.random.default_rng(seed)
    res = minimize(obj_grad, rng.standard_normal(p) * 0.01, jac=True, method="L-BFGS-B",
                   options={"maxiter": n_iter, "ftol": 1e-9, "gtol": 1e-6})
    return res.x


def _dann_surrogate_direction(sources, tgt, Xt_scaled, method="dann", seed=0) -> np.ndarray:
    """Linear surrogate of a (non-linear) DANN/CDAN decision function: a linear
    regression of the method's target logits onto the scaled target features.
    Approximate; needs TensorFlow (adapt)."""
    from experiments.adaptation_methods import fit_dann, fit_cdan
    r = (fit_dann if method == "dann" else fit_cdan)(sources, tgt, seed=seed, use_iwcv=False)
    p = np.asarray(r.get("target_probs"))
    if p is None or p.size != len(Xt_scaled):
        return np.full(Xt_scaled.shape[1], np.nan)
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return LinearRegression().fit(Xt_scaled, logit).coef_


# ---------------------------------------------------------------------------
# Per (sources, target) geometry
# ---------------------------------------------------------------------------

def _auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def compute_geometry(sources: list, tgt, with_dann: bool = False, seed: int = 0) -> dict:
    """Angle to the target-optimal direction beta_tgt* for every method, the
    covariance distance before/after CORAL, and the ERM AUROC gap.

    *sources* is a list of source domains (>=2 for GroupDRO/IRM to be meaningful);
    ERM/CORAL/IW pool them.  beta_tgt* is a LR on target TRUE labels (reference
    only).  All directions live in the same source-fitted standardized space.
    """
    Xpool = np.vstack([s.x for s in sources])
    ypool = np.concatenate([s.y for s in sources])
    scaler = StandardScaler().fit(Xpool)
    Xs, Xt = scaler.transform(Xpool), scaler.transform(tgt.x)
    groups = [(scaler.transform(s.x), np.asarray(s.y)) for s in sources]

    b_tgt = _lr_dir(Xt, tgt.y)                                   # target-optimal (reference)
    dirs = {
        "erm":       _lr_dir(Xs, ypool),
        "coral":     _lr_dir(_coral_align(Xs, Xt), ypool),
        "iw":        _iw_direction(Xs, ypool, Xt),
        "group_dro": _groupdro_direction(groups, seed=seed),
        "irm":       _irm_direction(groups, seed=seed),
    }
    if with_dann:
        dirs["dann"] = _dann_surrogate_direction(sources, tgt, Xt, "dann", seed)
        dirs["cdan"] = _dann_surrogate_direction(sources, tgt, Xt, "cdan", seed)

    out = {f"angle_{m}": _angle_deg(v, b_tgt) for m, v in dirs.items()}

    # ERM AUROC gap (source-val vs target) for the gap-vs-angle relationship.
    m_erm = LogisticRegression(**_LR).fit(Xs, ypool)
    out["auroc_gap_erm"] = _auroc(ypool, m_erm.predict_proba(Xs)[:, 1]) - \
                           _auroc(tgt.y, m_erm.predict_proba(Xt)[:, 1])
    out["covdist_before"] = _cov_distance(Xs, Xt)
    out["covdist_after"]  = _cov_distance(_coral_align(Xs, Xt), Xt)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_ci(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    m = float(v.mean())
    if len(v) < 2:
        return round(m, 4), 0.0
    se = v.std(ddof=1) / np.sqrt(len(v))
    return round(m, 4), round(float(sp_stats.t.ppf(0.975, len(v) - 1) * se), 4)


def _keys(with_dann):
    ms = _LINEAR_METHODS + (["dann", "cdan"] if with_dann else [])
    return [f"angle_{m}" for m in ms] + ["auroc_gap_erm", "covdist_before", "covdist_after"], ms


# ---------------------------------------------------------------------------
# Synthetic: rotation sweep + scenario table
# ---------------------------------------------------------------------------

def rotation_sweep(degrees, seeds, n, n_sources=3, with_dann=False):
    keys, _ = _keys(with_dann)
    out = []
    for deg in degrees:
        ang = np.radians(deg)
        per = []
        for seed in seeds:
            d = generate_scenario("observational_shift", n_source_domains=n_sources,
                                  n_per_domain=n, seed=seed, base=BaseParams(seed=seed),
                                  a_inv_rotation_angle=ang)
            per.append(compute_geometry(d["source"], d["target"], with_dann=with_dann, seed=seed))
        row = {"rot_deg": deg}
        for k in keys:
            row[f"{k}_mean"], row[f"{k}_ci95"] = _mean_ci(per, k)
        out.append(row)
    return out


def scenario_table(scenarios, seeds, n, n_sources=3, with_dann=False):
    keys, _ = _keys(with_dann)
    out = []
    for sc in scenarios:
        per = []
        for seed in seeds:
            d = generate_scenario(sc, n_source_domains=n_sources, n_per_domain=n,
                                  seed=seed, base=BaseParams(seed=seed))
            per.append(compute_geometry(d["source"], d["target"], with_dann=with_dann, seed=seed))
        row = {"scenario": sc}
        for k in keys:
            row[f"{k}_mean"], row[f"{k}_ci95"] = _mean_ci(per, k)
        cb, ca = row["covdist_before_mean"], row["covdist_after_mean"]
        row["covdist_reduced_frac"] = round((cb - ca) / cb, 4) if cb else float("nan")
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Real data: i3lung leave-one-center-out (multi-source)
# ---------------------------------------------------------------------------

class _Arr:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y).astype(int)
        self.n = len(self.y)


def real_geometry_loco(data_dir, with_dann=False, seed=0):
    """Decision-direction geometry on the i3lung LOCO folds: for each held-out
    target center the others are pooled as sources (>=2, so GroupDRO/IRM apply).
    Median imputer fit on source features only."""
    from sklearn.impute import SimpleImputer
    from experiments.run_real_data import load_i3lung_all

    centers, feats = load_i3lung_all(data_dir)
    names = list(centers.keys())
    keys, _ = _keys(with_dann)
    rows = []
    for tgt_name in names:
        src_names = [c for c in names if c != tgt_name]
        Xpool = np.vstack([centers[c][0].values for c in src_names])
        imp = SimpleImputer(strategy="median").fit(Xpool)
        sources = [_Arr(imp.transform(centers[c][0].values), centers[c][1]) for c in src_names]
        Xt, yt = centers[tgt_name]
        if len(np.unique(yt)) < 2:
            continue
        g = compute_geometry(sources, _Arr(imp.transform(Xt.values), yt),
                             with_dann=with_dann, seed=seed)
        g["fold"] = tgt_name
        g["n_tgt"] = int(len(yt))
        g["prev_tgt"] = round(float(np.asarray(yt).mean()), 3)
        cb, ca = g["covdist_before"], g["covdist_after"]
        g["covdist_reduced_frac"] = round((cb - ca) / cb, 4) if cb else float("nan")
        for k in keys:
            g[k] = round(float(g[k]), 4)
        rows.append(g)
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_COLORS = {"erm": "#c1121f", "coral": "#2a9d8f", "iw": "#e9c46a",
           "group_dro": "#8d99ae", "irm": "#003049", "dann": "#9d4edd", "cdan": "#f4a261"}


def make_sweep_figure(sweep, methods, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = [r["rot_deg"] for r in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: AUROC gap and ERM decision angle track each other.
    axb = ax1.twinx()
    l1, = ax1.plot(deg, [r["auroc_gap_erm_mean"] for r in sweep], "o-",
                   color="#c1121f", label="AUROC gap (ERM)")
    l2, = axb.plot(deg, [r["angle_erm_mean"] for r in sweep], "s--",
                   color="#003049", label=r"angle$(\beta_{ERM},\beta_{tgt}^*)$")
    ax1.set_xlabel(r"$A_{inv}$ rotation (degrees)")
    ax1.set_ylabel("target AUROC gap", color="#c1121f")
    axb.set_ylabel("decision angle (deg)", color="#003049")
    ax1.set_title("The transport gap is a rotation of the decision direction")
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=8)

    # Panel B: every method's decision angle overlaps ERM -> none aligns it.
    for m in methods:
        ax2.plot(deg, [r[f"angle_{m}_mean"] for r in sweep], marker="o", ms=3,
                 color=_COLORS.get(m, "gray"), label=m)
    ax2.set_xlabel(r"$A_{inv}$ rotation (degrees)")
    ax2.set_ylabel(r"angle$(\beta_{method},\beta_{tgt}^*)$ (deg)")
    ax2.set_title("No method moves the decision direction toward the target")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_real_figure(rows, methods, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = [r["fold"] for r in rows]
    x = np.arange(len(folds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Panel A: per-fold decision angle to beta_tgt*, one bar group per method.
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax1.bar(x + (i - len(methods) / 2) * w, [r[f"angle_{m}"] for r in rows], w,
                color=_COLORS.get(m, "gray"), label=m)
    ax1.set_xticks(x); ax1.set_xticklabels(folds, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel(r"angle$(\beta_{method},\beta_{tgt}^*)$ (deg)")
    ax1.set_title("Real (i3lung LOCO): every method's direction stays misaligned")
    ax1.legend(fontsize=7, ncol=2)

    # Panel B: CORAL covariance-distance reduction per fold (the marginal it does align).
    ax2.bar(x, [100 * r["covdist_reduced_frac"] for r in rows], 0.6, color="#2a9d8f")
    ax2.set_xticks(x); ax2.set_xticklabels(folds, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("CORAL covariance-distance reduction (%)")
    ax2.set_title("CORAL does align the marginal (~50%) -- just not the direction")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------

def _print_synth(sweep, table, methods):
    hdr = "".join(f"{m:>11}" for m in methods)
    print("\n  ROTATION SWEEP — angle(beta_method, beta_tgt*) per method (all stay ~ERM)")
    print(f"  {'rot':>4} {'gap_ERM':>8}{hdr}  {'covdist_bef':>11} {'covdist_aft':>11}")
    for r in sweep:
        angs = "".join(f"{r[f'angle_{m}_mean']:>11.1f}" for m in methods)
        print(f"  {r['rot_deg']:>4} {r['auroc_gap_erm_mean']:>8.3f}{angs}  "
              f"{r['covdist_before_mean']:>11.2f} {r['covdist_after_mean']:>11.2f}")

    print("\n  PER-SCENARIO — decision angle per method + CORAL covariance reduction")
    print(f"  {'scenario':22}{hdr}  {'covdist_red%':>12}")
    for r in table:
        angs = "".join(f"{r[f'angle_{m}_mean']:>11.1f}" for m in methods)
        print(f"  {r['scenario']:22}{angs}  {100*r['covdist_reduced_frac']:>11.0f}%")


def _print_real(rows, methods):
    hdr = "".join(f"{m:>11}" for m in methods)
    print("\n  REAL DATA (i3lung LOCO) — angle(beta_method, beta_tgt*) per fold")
    print(f"  {'fold':10}{'n_tgt':>6}{'prev':>6}{hdr}  {'covdist_red%':>12}")
    for r in rows:
        angs = "".join(f"{r[f'angle_{m}']:>11.1f}" for m in methods)
        print(f"  {r['fold']:10}{r['n_tgt']:>6}{r['prev_tgt']:>6.2f}{angs}  "
              f"{100*r['covdist_reduced_frac']:>11.0f}%")
    for m in methods:
        vals = [r[f"angle_{m}"] for r in rows]
        print(f"    mean angle {m:10}: {np.mean(vals):5.1f} deg")
    print("  -> every method's decision direction stays far from the target-optimal; "
          "CORAL still cuts the covariance distance ~50%.")


def main(n, seeds, with_dann):
    degrees = [0, 15, 30, 45, 60, 75, 90]
    scenarios = ["covariate_shift", "observational_shift", "concept_shift", "spurious_shift"]
    _, methods = _keys(with_dann)

    sweep = rotation_sweep(degrees, seeds, n, with_dann=with_dann)
    table = scenario_table(scenarios, seeds, n, with_dann=with_dann)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(sweep, _TABLES / "decision_geometry_sweep.csv")
    save_table_csv(table, _TABLES / "decision_geometry_scenarios.csv")
    make_sweep_figure(sweep, methods, _FIGURES / "decision_geometry.png")
    _print_synth(sweep, table, methods)
    print(f"\n  Saved: {_TABLES}/decision_geometry_sweep.csv, decision_geometry_scenarios.csv")
    print(f"  Saved: {_FIGURES}/decision_geometry.png (+ .pdf)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decision-direction geometry: why alignment fails.")
    ap.add_argument("--n", type=int, default=2000, help="Samples per domain (synthetic).")
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds (synthetic).")
    ap.add_argument("--real", action="store_true", help="Run on the i3lung LOCO folds.")
    ap.add_argument("--data_dir", default=None, help="I3LUNG_DATA directory (with --real).")
    ap.add_argument("--with_dann", action="store_true",
                    help="Also add a linear surrogate of DANN/CDAN (needs TensorFlow).")
    args = ap.parse_args()

    _, methods = _keys(args.with_dann)
    if args.real:
        if not args.data_dir:
            ap.error("--data_dir is required with --real")
        rows = real_geometry_loco(args.data_dir, with_dann=args.with_dann)
        _TABLES.mkdir(parents=True, exist_ok=True)
        save_table_csv(rows, _TABLES / "decision_geometry_real.csv")
        make_real_figure(rows, methods, _FIGURES / "decision_geometry_real.png")
        _print_real(rows, methods)
        print(f"\n  Saved: {_TABLES}/decision_geometry_real.csv")
        print(f"  Saved: {_FIGURES}/decision_geometry_real.png (+ .pdf)")
    else:
        main(args.n, list(range(args.seeds)), args.with_dann)
