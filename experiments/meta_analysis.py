"""
Random-effects meta-analysis of adaptation-vs-ERM gaps across multi-center
transfers (SHIFT-SURV real-data section).

Motivation
----------
The per-transfer patient-level bootstrap (run_real_data._bootstrap_real_gaps)
gives, for each source->target transfer and each method, the gap vs the ERM
reference with a 95% CI that reflects target-cohort sampling uncertainty.  Taken
one transfer at a time these gaps are individually underpowered (wide CIs), so a
reviewer can dismiss any single "no significant difference" as absence of power
rather than evidence of a true null.

This script pools the transfers with a DerSimonian-Laird random-effects model,
turning seven separate n.s. results into a single pooled estimate with a tight
CI, plus a between-transfer heterogeneity summary (Q, I^2, tau^2).  It answers
the question a clinical reviewer actually asks: "aggregated over all centers,
does adaptation buy any discrimination over ERM, and can we rule out a
clinically meaningful gain?"

Two estimands are reported per metric:
  1. Per-method pooled gap vs ERM (no selection bias).  This is the headline.
  2. "Best-DA per transfer" pooled gap: for each transfer pick the method with
     the largest observed gap, then pool.  This deliberately over-credits
     adaptation (winner's curse), so a null even here is a conservative bound.

Effect-sign convention (inherited from the bootstrap CSVs):
  auroc_gap_vs_ref = method - ERM   (positive = method better discrimination)
  ece_gap_vs_ref   = ERM - method   (positive = method better calibration)
So for both metrics, positive = adaptation better.

The per-transfer standard error is recovered from the bootstrap 95% percentile
CI as SE = (hi - lo) / (2 * 1.96): the bootstrap CI is a sampling CI, so this is
the standard inverse-normal back-out used to feed bootstrap effects into a
meta-analysis.

Usage
-----
  python -m experiments.meta_analysis                      # i3lung, default
  python -m experiments.meta_analysis --prefix i3lung
  python -m experiments.meta_analysis --prefix public_rotterdam_gbsg \\
      --clinical_margin 0.02
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TABLES = _ROOT / "outputs" / "tables"

# Methods whose *purpose* is discrimination transport (can move AUROC).
# prior_recalib / platt_scaling are rank-preserving -> excluded from the AUROC
# family by construction (their AUROC gap is ~0), included for ECE.
_DISC_METHODS  = ["iw_erm", "coral", "dann", "cdan", "irm", "group_dro"]
_CALIB_METHODS = ["iw_erm", "prior_recalib", "platt_scaling", "coral", "dann", "cdan", "irm", "group_dro"]
_Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Load per-transfer bootstrap gaps
# ---------------------------------------------------------------------------

def _se_from_ci(lo: float, hi: float) -> float:
    """Recover a Gaussian SE from a 95% percentile CI."""
    return (hi - lo) / (2.0 * _Z95)


def load_transfers(prefix: str, exclude: list[str] | None = None) -> tuple[list[str], dict]:
    """Return (transfer_names, data) where
    data[metric][method][transfer] = (gap, se), metric in {auroc, ece}.

    The glob is a prefix match, so ``--prefix i3lung`` also picks up
    ``i3lung_loco_*``: the six leave-one-center-out folds as well as the seven
    center-to-center transfers. Those two designs are run on the SAME six
    cohorts, so pooling all thirteen re-uses every target patient twice and the
    random-effects model -- which assumes independent units -- returns intervals
    that are too tight. That is a defensible reporting choice only if it is a
    choice; it was previously a side effect of the glob. Pass ``exclude`` (CLI:
    ``--exclude loco``) to keep the designs apart, and read the banner this
    prints, which now names every file that went into the pool.
    """
    paths = sorted(glob.glob(str(_TABLES / f"{prefix}*_bootstrap.csv")))
    for pat in (exclude or []):
        paths = [p for p in paths if pat not in Path(p).name]
    if not paths:
        raise FileNotFoundError(
            f"No '{prefix}*_bootstrap.csv' under {_TABLES}"
            + (f" after excluding {exclude}" if exclude else "")
            + ". Run experiments.run_real_data first."
        )
    data: dict = {"auroc": {}, "ece": {}}
    transfers: list[str] = []
    for p in paths:
        name = Path(p).name[len(prefix):].replace("_bootstrap.csv", "").strip("_")
        transfers.append(name)
        for row in csv.DictReader(open(p)):
            m = row["method"]
            if m == row["reference"]:
                continue  # skip ERM baseline (gap == 0 by definition)
            for metric in ("auroc", "ece"):
                gap = float(row[f"{metric}_gap_vs_ref"])
                se  = _se_from_ci(float(row[f"{metric}_gap_lo"]),
                                  float(row[f"{metric}_gap_hi"]))
                data[metric].setdefault(m, {})[name] = (gap, se)
    return transfers, data


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects pooling
# ---------------------------------------------------------------------------

def dersimonian_laird(y: np.ndarray, se: np.ndarray) -> dict:
    """Random-effects meta-analysis. y = effects, se = their standard errors."""
    y  = np.asarray(y, dtype=float)
    v  = np.asarray(se, dtype=float) ** 2
    v  = np.clip(v, 1e-12, None)
    k  = len(y)

    # Fixed-effect (inverse-variance) pooling — needed for the DL tau^2 estimate.
    w      = 1.0 / v
    y_fix  = float(np.sum(w * y) / np.sum(w))
    Q      = float(np.sum(w * (y - y_fix) ** 2))
    df     = k - 1
    C      = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2   = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Random-effects pooling with the estimated between-transfer variance tau^2.
    w_re   = 1.0 / (v + tau2)
    y_re   = float(np.sum(w_re * y) / np.sum(w_re))
    se_re  = float(np.sqrt(1.0 / np.sum(w_re)))
    z      = y_re / se_re if se_re > 0 else 0.0
    p      = float(2.0 * sp_stats.norm.sf(abs(z)))
    i2     = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0

    return {
        "k":        k,
        "estimate": y_re,
        "se":       se_re,
        "ci_lo":    y_re - _Z95 * se_re,
        "ci_hi":    y_re + _Z95 * se_re,
        "z":        z,
        "p":        p,
        "Q":        Q,
        "I2":       i2,
        "tau2":     tau2,
        "fixed":    y_fix,
    }


# ---------------------------------------------------------------------------
# Pooling drivers
# ---------------------------------------------------------------------------

def pool_per_method(data_metric: dict, methods: list[str], transfers: list[str]) -> list[dict]:
    rows = []
    for m in methods:
        per = data_metric.get(m, {})
        ys  = np.array([per[t][0] for t in transfers if t in per])
        ses = np.array([per[t][1] for t in transfers if t in per])
        if len(ys) < 2:
            continue
        r = dersimonian_laird(ys, ses)
        r["method"] = m
        rows.append(r)
    return rows


def pool_best_per_transfer(data_metric: dict, methods: list[str], transfers: list[str]) -> dict:
    """For each transfer pick the method with the max point-estimate gap, then
    pool those chosen effects.  Over-credits adaptation (winner's curse) -> a
    null here is a conservative upper bound on the adaptation advantage."""
    ys, ses, picks = [], [], []
    for t in transfers:
        cand = [(data_metric[m][t][0], data_metric[m][t][1], m)
                for m in methods if m in data_metric and t in data_metric[m]]
        if not cand:
            continue
        g, s, m = max(cand, key=lambda c: c[0])
        ys.append(g); ses.append(s); picks.append((t, m, g))
    r = dersimonian_laird(np.array(ys), np.array(ses))
    r["picks"] = picks
    return r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(r: dict) -> str:
    return (f"{r['estimate']:+.4f}  [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
            f"p={r['p']:.3f}   I2={r['I2']:4.0f}%  tau2={r['tau2']:.5f}")


def _target_of(name: str) -> str:
    """The target center of a transfer name like ``INT_GHD`` or ``loco_GHD``."""
    return name.split("_")[-1]


def _bootstrap_target_se(prefix: str, target: str, sources: list[str],
                         metric: str, methods: list[str]) -> dict:
    """SE of the within-target mean effect, taken from the joint bootstrap.

    The five transfers into one centre are evaluated on the same patients, so
    their errors are correlated and combining them as independent understates
    the SE of their mean. Measured on I3LUNG the understatement reaches 27% for
    IW-ERM, which propagates straight into the pooled interval the paper's main
    claim rests on.

    The bootstrap replicate index resamples the same target patients in every
    transfer into that centre -- verified: all five files carry the identical
    (b, seed_idx) grid -- so averaging the five effects *within* a replicate and
    taking the spread across replicates reproduces the correlation without
    having to model it. Returns {} when the replicate files are absent, and the
    caller keeps the analytic combination.
    """
    field = "d_auroc" if metric == "auroc" else "d_ece"
    per_source = []
    for s_c in sources:
        path = _TABLES / f"{prefix}_{s_c}_{target}_bootreps.csv"
        if not path.exists():
            return {}
        acc: dict = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                m = r.get("method")
                if m not in methods:
                    continue
                try:
                    acc.setdefault(m, {}).setdefault(int(r["b"]), []).append(
                        float(r[field]))
                except (KeyError, ValueError):
                    continue
        per_source.append({m: {b: float(np.mean(v)) for b, v in d.items()}
                           for m, d in acc.items()})
    if len(per_source) < 2:
        return {}

    out = {}
    for m in methods:
        if not all(m in p for p in per_source):
            continue
        bs = sorted(set.intersection(*(set(p[m]) for p in per_source)))
        if len(bs) < 2:
            continue
        joint = np.array([[p[m][b] for p in per_source] for b in bs])
        se = float(joint.mean(axis=1).std(ddof=1))
        # A method that cannot move the metric -- a monotone recalibration on
        # AUROC -- has an identically zero effect in every replicate. Its SE is
        # zero by construction, which no weighting scheme can use, so it is left
        # to the caller's fallback.
        if se > 0:
            out[m] = se
    return out


def _collapse_by_target(data_metric: dict, transfers: list[str],
                        prefix: str = "", metric: str = "") -> tuple[dict, list[str]]:
    """Average each method's effect over the transfers sharing a target center.

    With all thirty ordered pairs from six centers, every cohort is the target
    five times, so thirty units are nowhere near thirty independent pieces of
    evidence and DerSimonian-Laird would return intervals far too tight. The six
    target cohorts are disjoint patient sets, so collapsing to one value per
    target restores the assumption at the level where it matters. The within-
    target standard errors are combined as if the five estimates were exchangeable
    replicates of the same quantity, which understates their correlation and is
    therefore conservative for the pooled interval.

    Source centers still overlap across targets -- center A is a source for both
    B and C -- so the six units are not fully independent either. That residual
    dependence is between fitted models rather than between measured outcomes,
    and is the smaller of the two.
    """
    # Bootstrap standard errors first, when the replicate files are there.
    boot_se: dict = {}
    if prefix and metric:
        by_target_src: dict = {}
        for t in transfers:
            by_target_src.setdefault(_target_of(t), []).append(t.split("_", 1)[0])
        methods = sorted(data_metric)
        for tgt, srcs in by_target_src.items():
            se = _bootstrap_target_se(prefix, tgt, srcs, metric, methods)
            if se:
                boot_se[tgt] = se
        if boot_se:
            print(f"  [{metric}] errori standard per coorte dal bootstrap "
                  f"congiunto: {len(boot_se)} coorti")

    out: dict = {}
    for m, per in data_metric.items():
        by_t: dict[str, list] = {}
        for t, (g, se) in per.items():
            by_t.setdefault(_target_of(t), []).append((g, se))
        coll = {}
        for tgt, vals in by_t.items():
            gs = np.array([v[0] for v in vals], dtype=float)
            ses = np.array([v[1] for v in vals], dtype=float)
            se = boot_se.get(tgt, {}).get(
                m, float(np.sqrt(np.sum(ses ** 2)) / len(ses)))
            coll[tgt] = (float(gs.mean()), float(se))
        out[m] = coll
    targets = sorted({_target_of(t) for t in transfers})
    return out, targets


def _design_of(name: str) -> str:
    """Which experimental design a transfer belongs to.

    'loco_GHD' is a leave-one-center-out fold; 'INT_GHD' is a directed
    center-to-center transfer. They share target cohorts, so pooling both means
    the same patients enter the model twice.
    """
    return "leave-one-center-out" if name.startswith("loco") else "center-to-center"


def run(prefix: str, clinical_margin: float, out_prefix: str | None,
        exclude: list[str] | None = None, by_target: bool = False) -> None:
    transfers, data = load_transfers(prefix, exclude=exclude)
    if by_target:
        for metric in ("auroc", "ece"):
            data[metric], transfers_t = _collapse_by_target(
                data[metric], transfers, prefix=prefix, metric=metric)
        transfers = transfers_t
        print(f"\n  COLLASSATO PER CENTRO TARGET: {len(transfers)} unita' "
              f"({', '.join(transfers)})")
        print("  Le coorti target sono disgiunte, quindi queste unita' soddisfano")
        print("  l'assunzione di indipendenza al livello dove conta.")
    designs: dict[str, list[str]] = {}
    for t in transfers:
        designs.setdefault(_design_of(t), []).append(t)

    print(f"\n{'='*92}")
    print(f"  RANDOM-EFFECTS META-ANALYSIS (DerSimonian-Laird) — prefix '{prefix}'")
    print(f"  {len(transfers)} transfers: {', '.join(transfers)}")
    for d, ts in sorted(designs.items()):
        print(f"    {d:22s} k={len(ts):2d}  {', '.join(ts)}")
    if len(designs) > 1:
        print("  ! POOLING TWO DESIGNS. They are run on the same cohorts, so target")
        print("    patients are counted more than once and DerSimonian-Laird's")
        print("    independence assumption is violated: the pooled CIs below are")
        print("    anticonservative. Either report the designs separately")
        print(f"    (--exclude loco / --prefix {prefix}_loco) or state the dependence.")
    print(f"  Effect = adaptation - ERM (positive = adaptation better). "
          f"Clinical margin = {clinical_margin:+.3f}")
    print('='*92)

    out_rows: list[dict] = []

    for metric, methods, arrow in (("auroc", _DISC_METHODS, "AUROC"),
                                   ("ece",   _CALIB_METHODS, "ECE")):
        print(f"\n  [{arrow}]  per-method pooled gap vs ERM "
              f"(positive = adaptation better {'discrimination' if metric=='auroc' else 'calibration'})")
        print(f"  {'method':16s}{'pooled gap [95% CI]':40s}{'p':>8s}  {'I2':>5s}")
        print("  " + "-"*72)
        per_method = pool_per_method(data[metric], methods, transfers)
        for r in per_method:
            print(f"  {r['method']:16s}{_fmt(r)}")
            out_rows.append({"metric": metric, "estimand": f"method:{r['method']}",
                             **{k: r[k] for k in ("k","estimate","se","ci_lo","ci_hi","p","I2","tau2")}})

        best = pool_best_per_transfer(data[metric], methods, transfers)
        print(f"  {'BEST-per-transfer':16s}{_fmt(best)}   (winner's-curse upper bound)")
        picks = ", ".join(f"{t}:{m}({g:+.3f})" for t, m, g in best["picks"])
        print(f"    picks: {picks}")
        out_rows.append({"metric": metric, "estimand": "best_per_transfer",
                         **{k: best[k] for k in ("k","estimate","se","ci_lo","ci_hi","p","I2","tau2")}})

        # Interpretation vs the clinical margin.
        top = max(per_method, key=lambda r: r["estimate"]) if per_method else None
        if top is not None:
            rules_out = best["ci_hi"] < clinical_margin
            print(f"\n    -> Best single method by point estimate: {top['method']} "
                  f"({top['estimate']:+.4f}).")
            print(f"    -> Even the winner's-curse best-per-transfer upper CI is "
                  f"{best['ci_hi']:+.4f}; a {clinical_margin:+.3f} gain is "
                  f"{'RULED OUT' if rules_out else 'NOT ruled out'}.")

    if out_prefix:
        out_path = _TABLES / f"{out_prefix}_meta.csv"
        keys = ["metric","estimand","k","estimate","se","ci_lo","ci_hi","p","I2","tau2"]
        with open(out_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for row in out_rows:
                wr.writerow({k: (round(row[k], 6) if isinstance(row[k], float) else row[k]) for k in keys})
        print(f"\n  Saved: {out_path}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Random-effects meta-analysis of adaptation-vs-ERM gaps.")
    ap.add_argument("--prefix", default="i3lung", help="Bootstrap-CSV prefix (default: i3lung).")
    ap.add_argument("--clinical_margin", type=float, default=0.02,
                    help="AUROC/ECE gain considered clinically meaningful (default: 0.02).")
    ap.add_argument("--out_prefix", default=None,
                    help="If set, write <out_prefix>_meta.csv (default: <prefix>).")
    ap.add_argument("--by_target", action="store_true",
                    help="collassa i transfer che condividono il centro target "
                         "in un'unita' sola prima di poolare (raccomandato con "
                         "tutte e 30 le coppie ordinate)")
    ap.add_argument("--exclude", nargs="*", default=None, metavar="SUBSTR",
                    help="Drop bootstrap files whose name contains any of these "
                         "substrings, e.g. --exclude loco to keep the "
                         "center-to-center design separate from the LOCO folds.")
    args = ap.parse_args()
    run(args.prefix, args.clinical_margin, args.out_prefix or args.prefix,
        exclude=args.exclude, by_target=args.by_target)
