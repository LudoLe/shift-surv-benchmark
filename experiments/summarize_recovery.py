"""
Recovery verdict across transfers, corrected for the post-hoc "best DA" selection.

For each transfer we ran ~6 competing methods against the ERM reference and,
in the figures, report the *best* one per metric. Selecting the best of several
methods and quoting its raw bootstrap p-value inflates the false-positive rate
(with 4 candidates at alpha=0.05 the chance of at least one spurious hit is
~19%). This script therefore applies a **Holm step-down correction within each
method family** and reports, per transfer and per metric, the best-improving
method with its family-wise-adjusted p-value and a corrected significance flag.

Families
--------
  AUROC  (discrimination) : adaptation methods {iw_erm, coral, dann, cdan}.
      Calibration-only methods (platt, prior_recalib) are monotone transforms of
      the ERM scores, so they leave AUROC unchanged and are excluded here.
  ECE    (calibration)    : all six non-reference methods, since any of them may
      end up being the best-calibrated on the target.

Reads every ``outputs/tables/*_bootstrap.csv`` (I3LUNG and public), enriches with
the matching ``*_shiftdiag.csv`` when present, and writes
``outputs/tables/recovery_verdict.csv``.

Usage
-----
  python -m experiments.summarize_recovery
  python -m experiments.summarize_recovery --glob "public_*_bootstrap.csv"
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.io import save_table_csv

_ROOT   = Path(__file__).parent.parent
_TABLES = _ROOT / "outputs" / "tables"

AUROC_FAMILY = ["iw_erm", "coral", "dann", "cdan"]
ECE_FAMILY   = ["iw_erm", "prior_recalib", "platt_scaling", "isotonic",
                "coral", "dann", "cdan"]
ALPHA        = 0.05


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for a family of tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        val = (m - i) * p
        running = max(running, val)         # enforce monotonicity
        adj[name] = float(min(1.0, running))
    return adj


def _best_in_family(
    boot: pd.DataFrame, family: list[str], gap_col: str, p_col: str,
) -> dict:
    """Best-improving method in a family with a Holm-adjusted p over the family."""
    sub = boot[boot["method"].isin(family)]
    if sub.empty:
        return {}
    pvals = {r["method"]: float(r[p_col]) for _, r in sub.iterrows()}
    adj = _holm(pvals)
    # "Recovery" = a positive gap (method better than ERM). Pick the largest gain.
    gains = sub[sub[gap_col] > 0]
    pool = gains if not gains.empty else sub
    best = pool.loc[pool[gap_col].idxmax()]
    m = best["method"]
    return {
        "method":   m,
        "gap":      round(float(best[gap_col]), 4),
        "p_raw":    round(float(best[p_col]), 4),
        "p_holm":   round(float(adj[m]), 4),
        "sig":      bool(best[gap_col] > 0 and adj[m] < ALPHA),
        "n_family": len(sub),
    }


def _tau_from(prefix: str, diag: pd.DataFrame | None) -> str:
    if diag is not None and "tau" in diag.columns:
        v = diag["tau"].iloc[0]
        if pd.notna(v) and str(v) != "":
            return str(v)
    m = re.search(r"_tau([0-9.]+)", prefix)
    return m.group(1) if m else ""


def summarize(pattern: str = "*_bootstrap.csv", verbose: bool = True) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(_TABLES / pattern))):
        prefix = Path(path).name[: -len("_bootstrap.csv")]
        boot = pd.read_csv(path)

        diag_path = _TABLES / f"{prefix}_shiftdiag.csv"
        diag = pd.read_csv(diag_path) if diag_path.exists() else None
        src = diag["source"].iloc[0] if diag is not None else ""
        tgt = diag["target"].iloc[0] if diag is not None else ""
        mmd   = float(diag["rbf_mmd"].iloc[0]) if diag is not None else np.nan
        mmd_p = float(diag["rbf_mmd_p"].iloc[0]) if (diag is not None and "rbf_mmd_p" in diag.columns) else np.nan
        prev  = float(diag["prevalence_gap"].iloc[0]) if diag is not None else np.nan
        n_tgt = int(boot["n_tgt"].iloc[0]) if "n_tgt" in boot.columns else np.nan

        au = _best_in_family(boot, AUROC_FAMILY, "auroc_gap_vs_ref", "auroc_gap_p")
        ec = _best_in_family(boot, ECE_FAMILY,   "ece_gap_vs_ref",   "ece_gap_p")

        rows.append({
            "transfer":        prefix,
            "source":          src,
            "target":          tgt,
            "tau":             _tau_from(prefix, diag),
            "n_tgt":           n_tgt,
            "mmd":             round(mmd, 4)   if not np.isnan(mmd)  else "",
            "mmd_p":           round(mmd_p, 4) if not np.isnan(mmd_p) else "",
            "prevalence_gap":  round(prev, 4)  if not np.isnan(prev) else "",
            "auroc_best":      au.get("method", ""),
            "auroc_gap":       au.get("gap", ""),
            "auroc_p_raw":     au.get("p_raw", ""),
            "auroc_p_holm":    au.get("p_holm", ""),
            "auroc_recovers":  au.get("sig", ""),
            "ece_best":        ec.get("method", ""),
            "ece_gap":         ec.get("gap", ""),
            "ece_p_raw":       ec.get("p_raw", ""),
            "ece_p_holm":      ec.get("p_holm", ""),
            "ece_recovers":    ec.get("sig", ""),
        })

    if verbose:
        _print(rows)
    if rows:
        save_table_csv(rows, _TABLES / "recovery_verdict.csv")
        if verbose:
            print(f"\n  Saved: {_TABLES}/recovery_verdict.csv")
    return rows


def _print(rows: list[dict]) -> None:
    print("\n" + "=" * 118)
    print("  RECOVERY VERDICT  (best method per family, Holm-corrected within family; ✓ = corrected p < 0.05 with a gain)")
    print("=" * 118)
    hdr = (f"  {'transfer':40s}{'n_tgt':>6s}  {'MMD(p)':>12s}  "
           f"{'AUROC: best (Δ, p_holm)':>34s}  {'ECE: best (Δ, p_holm)':>34s}")
    print(hdr)
    print("  " + "-" * 114)
    for r in rows:
        mmd = f"{r['mmd']}" + (f"({r['mmd_p']})" if r['mmd_p'] != "" else "")
        au = (f"{r['auroc_best']} {r['auroc_gap']:+} p={r['auroc_p_holm']}"
              f"{' ✓' if r['auroc_recovers'] is True else ''}") if r['auroc_best'] else "—"
        ec = (f"{r['ece_best']} {r['ece_gap']:+} p={r['ece_p_holm']}"
              f"{' ✓' if r['ece_recovers'] is True else ''}") if r['ece_best'] else "—"
        label = r["transfer"][:40]
        print(f"  {label:40s}{str(r['n_tgt']):>6s}  {mmd:>12s}  {au:>34s}  {ec:>34s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recovery verdict with multiplicity correction.")
    parser.add_argument("--glob", type=str, default="*_bootstrap.csv",
                        help="Filename pattern under outputs/tables/ to summarise.")
    args = parser.parse_args()
    summarize(pattern=args.glob)
