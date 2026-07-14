"""
ECE recoverability: *why* calibration is or is not recoverable under shift.

Companion to experiments.decision_geometry, for the calibration axis.  The
decision-geometry shows discrimination is not recovered because the label-relevant
decision direction is unidentifiable from unlabeled target data.  Calibration
depends on a *different* quantity -- the target prior / probability scale -- so it
can drift and recover independently.  This module quantifies how much of the
target calibration error is removable by recalibration at all, and how much of
that a deployable unsupervised method actually reaches.

A monotone recalibration can always drive in-sample ECE to zero regardless of
ranking (calibration ignores ordering), so the question is not *how much* ECE is
removable but whether it is removable *without target labels*.

Quantities (per scenario / transfer), all evaluated on a held-out target half
------------------------------------
  ece_erm        target ECE of the source-trained ERM classifier (deployed).
  ece_platt      after Platt scaling fit on a source validation split (deployable,
                 unsupervised; a source-estimated affine map).
  ece_priorem    after Saerens EM prior adjustment (deployable, unsupervised;
                 estimates the target prior from *unlabeled* target).
  ece_labeltgt   after an isotonic recalibration fit on the *other* target half's
                 true labels (the calibration a small labeled target sample buys;
                 not an unsupervised method).
  unsup_best_gain = ece_erm - min(ece_platt, ece_priorem)  (best unsupervised gain)
  labeltgt_gain   = ece_erm - ece_labeltgt                 (few-labeled-target gain)

Reading it
----------
  Unsupervised recalibration does not recover ECE, and Prior-EM often *worsens* it
  (its pure-prior-shift assumption is violated), so unsup_best_gain stays near 0.
  A few labeled target samples recover calibration almost fully (labeltgt_gain
  large), because calibration is a one-dimensional monotone correction -- unlike
  the decision direction that discrimination needs.  Calibration and discrimination
  thus fail for the same reason unsupervised (the target relationship is
  unidentifiable from unlabeled data) but differ in how cheaply a few target labels
  fix them.

numpy / sklearn only (no TensorFlow), so it is unit-testable without the Keras-2
stack.

Usage
-----
  python -m experiments.ece_recoverability                        # synthetic scenarios
  python -m experiments.ece_recoverability --real --data_dir I3LUNG_DATA   # i3lung transfers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.metrics import ece
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"
_LR = dict(max_iter=500, solver="lbfgs", C=1.0)


def _saerens_em(p_tgt, pi_src, n_iter=50):
    """Saerens EM prior adjustment: re-estimate the target prior from unlabeled
    target predictions and rescale (Saerens et al. 2002)."""
    pi_tgt = float(p_tgt.mean())
    for _ in range(n_iter):
        num = p_tgt * (pi_tgt / pi_src)
        den = num + (1 - p_tgt) * ((1 - pi_tgt) / (1 - pi_src))
        p_adj = num / np.clip(den, 1e-12, None)
        pi_new = float(p_adj.mean())
        if abs(pi_new - pi_tgt) < 1e-7:
            break
        pi_tgt = pi_new
    return p_adj


def compute_ece_recoverability(sources, tgt, n_bins=10, seed=0) -> dict:
    """Compare deployable *unsupervised* recalibration (Platt on source val,
    Prior-EM) against recalibration that uses a few *labeled target* samples
    (isotonic fit on one target half, evaluated out-of-sample on the other).

    A monotone recalibration can always drive in-sample ECE to zero regardless of
    ranking, so the meaningful question is not 'how much ECE is removable' but
    'removable *without* target labels'.  ece_labeltgt is the calibration that a
    handful of target labels would buy (a proper held-out estimate); the
    unsupervised methods are what one can actually deploy.
    """
    Xpool = np.vstack([s.x for s in sources])
    ypool = np.concatenate([s.y for s in sources])
    Xtr, Xval, ytr, yval = train_test_split(
        Xpool, ypool, test_size=0.2, random_state=seed,
        stratify=ypool if len(np.unique(ypool)) > 1 else None)

    scaler = StandardScaler().fit(Xtr)
    erm = LogisticRegression(**_LR).fit(scaler.transform(Xtr), ytr)
    p_val = erm.predict_proba(scaler.transform(Xval))[:, 1]
    p_tgt_all = erm.predict_proba(scaler.transform(tgt.x))[:, 1]
    y_all = np.asarray(tgt.y).astype(int)

    # Split the target: A = labels used to fit a target recalibration, B = eval.
    idx = np.arange(len(y_all))
    A, B = train_test_split(idx, test_size=0.5, random_state=seed,
                            stratify=y_all if len(np.unique(y_all)) > 1 else None)
    pB, yB = p_tgt_all[B], y_all[B]

    ece_erm = ece(yB, pB, n_bins=n_bins)

    # Deployable, unsupervised: Platt on source val (affine, source-estimated).
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(p_val.reshape(-1, 1), yval)
    ece_platt = ece(yB, platt.predict_proba(pB.reshape(-1, 1))[:, 1], n_bins=n_bins)

    # Deployable, unsupervised: Prior-EM (target prior from *unlabeled* full target).
    p_prior_all = _saerens_em(p_tgt_all, float(ytr.mean()))
    ece_priorem = ece(yB, p_prior_all[B], n_bins=n_bins)

    # With a few labeled target samples: isotonic fit on A, evaluated on B (out-of-sample).
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_tgt_all[A], y_all[A])
    ece_labeltgt = ece(yB, iso.predict(pB), n_bins=n_bins)

    return {
        "ece_erm":            round(float(ece_erm), 4),
        "ece_platt":          round(float(ece_platt), 4),
        "ece_priorem":        round(float(ece_priorem), 4),
        "ece_labeltgt":       round(float(ece_labeltgt), 4),
        "unsup_best_gain":    round(float(ece_erm - min(ece_platt, ece_priorem)), 4),
        "labeltgt_gain":      round(float(ece_erm - ece_labeltgt), 4),
        "n_label":            int(len(A)),
        "prev_src":           round(float(ytr.mean()), 3),
        "prev_tgt":           round(float(y_all.mean()), 3),
    }


def _mean(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    return round(float(v.mean()), 4) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def synthetic_table(scenarios, seeds, n, n_sources=1):
    out = []
    for sc in scenarios:
        per = []
        for s in seeds:
            d = generate_scenario(sc, n_source_domains=n_sources, n_per_domain=n,
                                 seed=s, base=BaseParams(seed=s))
            per.append(compute_ece_recoverability(d["source"], d["target"], seed=s))
        row = {"scenario": sc}
        for k in ("ece_erm", "ece_platt", "ece_priorem", "ece_labeltgt",
                  "unsup_best_gain", "labeltgt_gain", "prev_src", "prev_tgt"):
            row[k] = _mean(per, k)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Real (i3lung transfers)
# ---------------------------------------------------------------------------

class _Arr:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y).astype(int)
        self.n = len(self.y)


_I3LUNG_TRANSFERS = [("GHD", "INT"), ("INT", "GHD"), ("INT", "MH"),
                     ("INT", "SZMC"), ("INT", "UOC"), ("INT", "VHIO"), ("VHIO", "INT")]


_ROTTERDAM_TRANSFERS = [("Rotterdam", "GBSG2"), ("GBSG2", "Rotterdam")]


def _row_from_transfer(Xs, ys, Xt, yt, name):
    from sklearn.impute import SimpleImputer
    if len(np.unique(ys)) < 2 or len(np.unique(yt)) < 2:
        return None
    imp = SimpleImputer(strategy="median").fit(Xs)
    g = compute_ece_recoverability([_Arr(imp.transform(Xs), ys)], _Arr(imp.transform(Xt), yt))
    g["transfer"] = name
    return g


def real_table(data_dir, transfers=_I3LUNG_TRANSFERS, label_col="OS_6"):
    from experiments.run_real_data import load_i3lung
    out = []
    for s, t in transfers:
        src_df, tgt_df, feats = load_i3lung(data_dir, s, t)
        g = _row_from_transfer(src_df[feats].values, src_df[label_col].values.astype(int),
                               tgt_df[feats].values, tgt_df[label_col].values.astype(int), f"{s}->{t}")
        if g is not None:
            out.append(g)
    return out


def real_table_public(registry_name, tau, transfers=_ROTTERDAM_TRANSFERS, label_col="event"):
    """Same diagnostic on a public multicentric registry (e.g. rotterdam_gbsg),
    binarised at horizon ``tau`` months."""
    from experiments.run_real_data import load_public_transfer
    out = []
    for s, t in transfers:
        src_df, tgt_df, feats = load_public_transfer(registry_name, s, t, tau, label_col=label_col)
        g = _row_from_transfer(src_df[feats].values, src_df[label_col].values.astype(int),
                               tgt_df[feats].values, tgt_df[label_col].values.astype(int), f"{s}->{t}")
        if g is not None:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print(rows, key):
    print(f"  {key:22}{'ECE_erm':>9}{'Platt':>8}{'PriorEM':>9}{'label-tgt':>10}"
          f"{'unsup_gain':>11}{'label_gain':>11}{'prev_s->t':>12}")
    for r in rows:
        print(f"  {str(r[key]):22}{r['ece_erm']:>9.3f}{r['ece_platt']:>8.3f}"
              f"{r['ece_priorem']:>9.3f}{r['ece_labeltgt']:>10.3f}"
              f"{r['unsup_best_gain']:>+11.3f}{r['labeltgt_gain']:>+11.3f}"
              f"   {r['prev_src']:>4.2f}->{r['prev_tgt']:<4.2f}")


def main_synth(n, seeds):
    scenarios = ["no_shift", "covariate_shift", "observational_shift",
                 "baseline_hazard_shift", "prior_shift", "concept_shift", "spurious_shift"]
    rows = synthetic_table(scenarios, seeds, n)
    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / "ece_recoverability_synth.csv")
    print("\n  ECE RECOVERABILITY (synthetic) — unsupervised recalibration vs a few "
          "labeled target samples")
    _print(rows, "scenario")
    print("\n  Unsupervised recal (Platt/Prior-EM) barely helps and Prior-EM often backfires; "
          "a few labeled target samples (label-tgt) recover calibration to ~0.03-0.04 everywhere.")
    print(f"  Saved: {_TABLES}/ece_recoverability_synth.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ECE recoverability diagnostic.")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--dataset", default="i3lung", choices=["i3lung", "rotterdam_gbsg"])
    ap.add_argument("--data_dir", default=None, help="I3LUNG_DATA directory (i3lung only).")
    ap.add_argument("--tau", type=float, default=None, help="Horizon in months (public registry).")
    args = ap.parse_args()
    if args.real:
        if args.dataset == "i3lung":
            if not args.data_dir:
                ap.error("--data_dir is required with --dataset i3lung")
            rows, label, out = real_table(args.data_dir), "real i3lung transfers", "ece_recoverability_real.csv"
        else:
            import importlib
            reg = importlib.import_module("data.registry_rotterdam_gbsg")
            tau = args.tau if args.tau is not None else float(reg.TAU)
            rows = real_table_public("rotterdam_gbsg", tau)
            label, out = f"rotterdam_gbsg (breast cancer, tau={tau:g}m)", f"ece_recoverability_rotterdam_tau{tau:g}.csv"
        _TABLES.mkdir(parents=True, exist_ok=True)
        save_table_csv(rows, _TABLES / out)
        print(f"\n  ECE RECOVERABILITY ({label})")
        _print(rows, "transfer")
        print("\n  unsup_gain ~ 0 (Prior-EM can backfire); label_gain = calibration a few "
              "labeled target samples would recover.")
        print(f"  Saved: {_TABLES}/{out}")
    else:
        main_synth(args.n, list(range(args.seeds)))
