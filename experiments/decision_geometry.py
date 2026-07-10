"""
Decision-direction geometry: *why* unsupervised alignment fails to recover
discrimination under measurement/observational shift.

The main benchmark shows empirically that no unsupervised DA method recovers
target discrimination beyond ERM, while the invariant oracle reveals large
recoverable headroom.  This module explains the mechanism, using the known
data-generating process, without any target labels leaking into training.

Idea
----
A linear classifier is a *decision direction* beta in x-space.  For a synthetic
domain we can compute the direction a target-trained model *would* use (the
"target-optimal direction" beta_tgt*, a LR fit on target x with TRUE labels --
used only as a diagnostic reference, never for training).  We then ask, for the
source model and for each unsupervised DA method:

  * how misaligned is its decision direction from beta_tgt*  (angle in degrees),
  * does the method reduce that misalignment,
  * and how much does it reduce the marginal discrepancy (covariance distance)?

Findings the script quantifies (observational shift = A_inv rotated by theta):
  1. angle(beta_src, beta_tgt*) grows with the rotation and tracks the AUROC gap
     almost 1:1 -- the transport gap *is* a rotation of the decision direction.
  2. CORAL (and other unsupervised alignment) shrink the source/target
     covariance distance substantially but leave the decision-direction angle
     essentially unchanged -- they align the marginal P(x), not the label-
     relevant direction, which is not identifiable from unlabeled target data.

Everything here is numpy/sklearn only (no TensorFlow), so CORAL is reimplemented
in closed form and the module is unit-testable without the Keras-2 stack.

Usage
-----
  python -m experiments.decision_geometry                 # sweep + table + figure
  python -m experiments.decision_geometry --n 2000 --seeds 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (degrees) between two decision directions (sign-agnostic up to pi)."""
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _msqrt(C: np.ndarray, inv: bool = False) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    d = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return V @ np.diag(d) @ V.T


def _coral_align(Xs: np.ndarray, Xt: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """CORAL whitening-recoloring of source features to the target covariance
    (Sun & Saenko 2016), closed form: Xs' = Xs @ Cs^{-1/2} @ Ct^{1/2}."""
    Cs = np.cov(Xs, rowvar=False) + lam * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + lam * np.eye(Xt.shape[1])
    return Xs @ (_msqrt(Cs, inv=True) @ _msqrt(Ct))


def _cov_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius distance between the covariance matrices of A and B."""
    return float(np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False)))


def _lr_dir(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return LogisticRegression(**_LR).fit(X, y).coef_[0]


def _iw_direction(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Importance-weighted ERM decision direction (covariate-shift correction):
    reweight source by a domain classifier w(x)=P(tgt|x)/P(src|x)."""
    Xd = np.vstack([Xs, Xt])
    yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LogisticRegression(**_LR).fit(Xd, yd)
    p = clf.predict_proba(Xs)[:, 1]
    w = np.clip(p / np.clip(1 - p, 1e-6, None), 0.05, 20.0)
    return LogisticRegression(**_LR).fit(Xs, ys, sample_weight=w).coef_[0]


# ---------------------------------------------------------------------------
# Per (source, target) geometry
# ---------------------------------------------------------------------------

def _auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def compute_geometry(src, tgt) -> dict:
    """Angles, AUROC gaps and covariance distances for ERM, CORAL, IW-ERM on one
    (source, target) pair.  beta_tgt* is a LR fit on target TRUE labels (a
    reference direction only -- no method below uses target labels)."""
    scaler = StandardScaler().fit(src.x)
    Xs, Xt = scaler.transform(src.x), scaler.transform(tgt.x)

    b_tgt = _lr_dir(Xt, tgt.y)                    # target-optimal direction (reference)
    b_src = _lr_dir(Xs, src.y)                    # ERM
    Xs_coral = _coral_align(Xs, Xt)
    b_coral = _lr_dir(Xs_coral, src.y)            # CORAL (feature alignment)
    b_iw = _iw_direction(Xs, src.y, Xt)           # IW-ERM (reweighting)

    def gap(direction_fit_X, model_X_src, model_X_tgt, y_src):
        m = LogisticRegression(**_LR).fit(direction_fit_X, y_src)
        return _auroc(y_src, m.predict_proba(model_X_src)[:, 1]) - \
               _auroc(tgt.y, m.predict_proba(model_X_tgt)[:, 1])

    return {
        "auroc_gap_erm":   gap(Xs,       Xs,       Xt, src.y),
        "auroc_gap_coral": gap(Xs_coral, Xs_coral, Xt, src.y),
        "angle_src_tgt":   _angle_deg(b_src,   b_tgt),
        "angle_coral_tgt": _angle_deg(b_coral, b_tgt),
        "angle_iw_tgt":    _angle_deg(b_iw,    b_tgt),
        "covdist_before":  _cov_distance(Xs, Xt),
        "covdist_after":   _cov_distance(Xs_coral, Xt),
    }


# ---------------------------------------------------------------------------
# Aggregation over seeds
# ---------------------------------------------------------------------------

def _mean_ci(rows: list[dict], key: str) -> tuple[float, float]:
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    m = float(v.mean())
    if len(v) < 2:
        return round(m, 4), 0.0
    se = v.std(ddof=1) / np.sqrt(len(v))
    return round(m, 4), round(float(sp_stats.t.ppf(0.975, len(v) - 1) * se), 4)


# ---------------------------------------------------------------------------
# Rotation sweep (the mechanism figure)
# ---------------------------------------------------------------------------

def rotation_sweep(degrees: list[int], seeds: list[int], n: int) -> list[dict]:
    out = []
    for deg in degrees:
        ang = np.radians(deg)
        per = []
        for seed in seeds:
            base = BaseParams(seed=seed)
            d = generate_scenario("observational_shift", n_source_domains=1,
                                  n_per_domain=n, seed=seed, base=base,
                                  a_inv_rotation_angle=ang)
            per.append(compute_geometry(d["source"][0], d["target"]))
        row = {"rot_deg": deg}
        for k in ("auroc_gap_erm", "auroc_gap_coral", "angle_src_tgt",
                  "angle_coral_tgt", "angle_iw_tgt", "covdist_before", "covdist_after"):
            m, ci = _mean_ci(per, k)
            row[f"{k}_mean"], row[f"{k}_ci95"] = m, ci
        out.append(row)
    return out


def scenario_table(scenarios: list[str], seeds: list[int], n: int) -> list[dict]:
    out = []
    for sc in scenarios:
        per = []
        for seed in seeds:
            base = BaseParams(seed=seed)
            d = generate_scenario(sc, n_source_domains=1, n_per_domain=n,
                                  seed=seed, base=base)
            per.append(compute_geometry(d["source"][0], d["target"]))
        row = {"scenario": sc}
        for k in ("auroc_gap_erm", "angle_src_tgt", "angle_coral_tgt", "angle_iw_tgt",
                  "covdist_before", "covdist_after"):
            m, ci = _mean_ci(per, k)
            row[f"{k}_mean"], row[f"{k}_ci95"] = m, ci
        # Derived: how much of the misalignment / covariance gap DA removes.
        row["angle_reduced_coral"] = round(row["angle_src_tgt_mean"] - row["angle_coral_tgt_mean"], 4)
        cb, ca = row["covdist_before_mean"], row["covdist_after_mean"]
        row["covdist_reduced_frac"] = round((cb - ca) / cb, 4) if cb else float("nan")
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(sweep: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = [r["rot_deg"] for r in sweep]
    gap_erm = [r["auroc_gap_erm_mean"] for r in sweep]
    ang_src = [r["angle_src_tgt_mean"] for r in sweep]
    ang_cor = [r["angle_coral_tgt_mean"] for r in sweep]
    cov_b = [r["covdist_before_mean"] for r in sweep]
    cov_a = [r["covdist_after_mean"] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: AUROC gap and decision-direction angle track each other.
    axb = ax1.twinx()
    l1, = ax1.plot(deg, gap_erm, "o-", color="#c1121f", label="AUROC gap (ERM)")
    l2, = axb.plot(deg, ang_src, "s--", color="#003049", label=r"angle$(\beta_{src},\beta_{tgt}^*)$")
    ax1.set_xlabel(r"$A_{inv}$ rotation (degrees)")
    ax1.set_ylabel("target AUROC gap", color="#c1121f")
    axb.set_ylabel("decision-direction angle (deg)", color="#003049")
    ax1.set_title("The transport gap is a rotation of the decision direction")
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=8)

    # Panel B: CORAL aligns covariance but not the decision direction.
    axc = ax2.twinx()
    l3, = ax2.plot(deg, ang_src, "s--", color="#003049", label=r"angle$(\beta_{src},\beta_{tgt}^*)$")
    l4, = ax2.plot(deg, ang_cor, "^-", color="#2a9d8f", label=r"angle$(\beta_{CORAL},\beta_{tgt}^*)$")
    l5, = axc.plot(deg, cov_b, ":", color="#8d99ae", label="cov. dist. before")
    l6, = axc.plot(deg, cov_a, "-", color="#8d99ae", label="cov. dist. after CORAL")
    ax2.set_xlabel(r"$A_{inv}$ rotation (degrees)")
    ax2.set_ylabel("decision-direction angle (deg)")
    axc.set_ylabel("source/target covariance distance", color="#8d99ae")
    ax2.set_title("CORAL aligns the marginal, not the decision direction")
    ax2.legend(handles=[l3, l4, l5, l6], loc="upper left", fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _print(sweep, table):
    print("\n  ROTATION SWEEP (observational shift) — angle tracks the AUROC gap; "
          "CORAL leaves the angle put")
    print(f"  {'rot':>4} {'gap_ERM':>9} {'gap_CORAL':>10} {'ang(src,t*)':>12} "
          f"{'ang(coral,t*)':>14} {'covdist_bef':>12} {'covdist_aft':>12}")
    for r in sweep:
        print(f"  {r['rot_deg']:>4} {r['auroc_gap_erm_mean']:>9.3f} "
              f"{r['auroc_gap_coral_mean']:>10.3f} {r['angle_src_tgt_mean']:>12.1f} "
              f"{r['angle_coral_tgt_mean']:>14.1f} {r['covdist_before_mean']:>12.2f} "
              f"{r['covdist_after_mean']:>12.2f}")

    print("\n  PER-SCENARIO — DA reduces the covariance gap, not the decision-direction angle")
    print(f"  {'scenario':22} {'gap':>7} {'ang(src,t*)':>12} {'Δang_CORAL':>11} "
          f"{'covdist_red%':>12}")
    for r in table:
        print(f"  {r['scenario']:22} {r['auroc_gap_erm_mean']:>7.3f} "
              f"{r['angle_src_tgt_mean']:>12.1f} {r['angle_reduced_coral']:>11.1f} "
              f"{100*r['covdist_reduced_frac']:>11.0f}%")


# ---------------------------------------------------------------------------
# Real-data verification of the mechanism (static, no rotation knob)
# ---------------------------------------------------------------------------

# The 7 I3LUNG transfers used in the main real-data table.
I3LUNG_TRANSFERS = [("GHD", "INT"), ("INT", "GHD"), ("INT", "MH"),
                    ("INT", "SZMC"), ("INT", "UOC"), ("INT", "VHIO"), ("VHIO", "INT")]


class _Arr:
    """Lightweight (x, y) holder so compute_geometry runs on real cohorts."""
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y).astype(int)
        self.n = len(self.y)


def real_geometry(data_dir, transfers=I3LUNG_TRANSFERS, label_col="OS_6") -> list[dict]:
    """Run the decision-direction diagnostic on the real I3LUNG transfers.

    beta_tgt* is a LR on the target's TRUE labels -- a diagnostic reference only,
    never used to train any reported method (same protocol as the synthetic
    analysis).  The median imputer is fit on the source features only.
    """
    from sklearn.impute import SimpleImputer
    from experiments.run_real_data import load_i3lung

    rows = []
    for s, t in transfers:
        src_df, tgt_df, feats = load_i3lung(data_dir, s, t)
        Xs, ys = src_df[feats].values, src_df[label_col].values.astype(int)
        Xt, yt = tgt_df[feats].values, tgt_df[label_col].values.astype(int)
        if len(np.unique(ys)) < 2 or len(np.unique(yt)) < 2:
            continue
        imp = SimpleImputer(strategy="median").fit(Xs)
        g = compute_geometry(_Arr(imp.transform(Xs), ys), _Arr(imp.transform(Xt), yt))
        g["transfer"] = f"{s}->{t}"
        g["n_tgt"] = int(len(yt))
        g["prev_tgt"] = round(float(yt.mean()), 3)
        # Round the geometry fields for the CSV.
        for k in ("auroc_gap_erm", "auroc_gap_coral", "angle_src_tgt", "angle_coral_tgt",
                  "angle_iw_tgt", "covdist_before", "covdist_after"):
            g[k] = round(float(g[k]), 4)
        g["angle_reduced_coral"] = round(g["angle_src_tgt"] - g["angle_coral_tgt"], 4)
        g["covdist_reduced_frac"] = round(
            (g["covdist_before"] - g["covdist_after"]) / g["covdist_before"], 4
        ) if g["covdist_before"] else float("nan")
        rows.append(g)
    return rows


def _report_real(rows: list[dict]) -> None:
    print("\n  REAL DATA (I3LUNG) — does the mechanism hold without a rotation knob?")
    print(f"  {'transfer':14} {'n_tgt':>5} {'prev':>5} {'gap':>7} {'ang(src,t*)':>11} "
          f"{'Δang_CORAL':>10} {'covdist_red%':>12}")
    for r in rows:
        print(f"  {r['transfer']:14} {r['n_tgt']:>5} {r['prev_tgt']:>5.2f} "
              f"{r['auroc_gap_erm']:>7.3f} {r['angle_src_tgt']:>11.1f} "
              f"{r['angle_reduced_coral']:>10.1f} {100*r['covdist_reduced_frac']:>11.0f}%")

    gaps = np.array([r["auroc_gap_erm"] for r in rows])
    angs = np.array([r["angle_src_tgt"] for r in rows])
    dred = np.mean([r["angle_reduced_coral"] for r in rows])
    cred = np.mean([r["covdist_reduced_frac"] for r in rows])
    if len(rows) >= 3:
        pr = sp_stats.pearsonr(angs, gaps)
        sr = sp_stats.spearmanr(angs, gaps)
        print(f"\n  angle vs AUROC gap across transfers:  Pearson r={pr[0]:.2f} (p={pr[1]:.3f})"
              f"   Spearman rho={sr.correlation:.2f} (p={sr.pvalue:.3f})")
    print(f"  CORAL: mean covariance-distance reduction = {100*cred:.0f}%,  "
          f"mean decision-angle reduction = {dred:.1f} deg")
    print("  -> alignment shrinks the marginal gap but not the decision-direction angle, "
          "as in the synthetic sweep.")


def main(n: int, seeds: list[int]) -> None:
    degrees = [0, 15, 30, 45, 60, 75, 90]
    scenarios = ["covariate_shift", "observational_shift", "concept_shift", "spurious_shift"]

    sweep = rotation_sweep(degrees, seeds, n)
    table = scenario_table(scenarios, seeds, n)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(sweep, _TABLES / "decision_geometry_sweep.csv")
    save_table_csv(table, _TABLES / "decision_geometry_scenarios.csv")
    make_figure(sweep, _FIGURES / "decision_geometry.png")

    _print(sweep, table)
    print(f"\n  Saved: {_TABLES}/decision_geometry_sweep.csv")
    print(f"  Saved: {_TABLES}/decision_geometry_scenarios.csv")
    print(f"  Saved: {_FIGURES}/decision_geometry.png (+ .pdf)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decision-direction geometry: why alignment fails.")
    ap.add_argument("--n", type=int, default=2000, help="Samples per domain (synthetic).")
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds (synthetic).")
    ap.add_argument("--real", action="store_true",
                    help="Verify the mechanism on the real I3LUNG transfers instead of the synthetic sweep.")
    ap.add_argument("--data_dir", default=None, help="I3LUNG_DATA directory (required with --real).")
    args = ap.parse_args()
    if args.real:
        if not args.data_dir:
            ap.error("--data_dir is required with --real")
        rows = real_geometry(args.data_dir)
        _TABLES.mkdir(parents=True, exist_ok=True)
        save_table_csv(rows, _TABLES / "decision_geometry_real.csv")
        _report_real(rows)
        print(f"\n  Saved: {_TABLES}/decision_geometry_real.csv")
    else:
        main(args.n, list(range(args.seeds)))
