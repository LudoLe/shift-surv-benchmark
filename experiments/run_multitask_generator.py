"""
Multitask generator experiment.

Tests whether fitting the shared loading matrix A jointly on
  — i3lung source OS (main task)
  — MSK external PFS (auxiliary task, weight λ)
improves LOOCO OS@12m performance versus the pure OS lat_cox baseline.

Four conditions:
  λ=0     : OS only  → standard lat_cox (should match run_looco_full_eval.py
             with 7 features; here the 7-feature variant for fair comparison)
  λ=0.3   : mild PFS influence
  λ=0.5   : equal contribution (default recommendation)
  λ=1.0   : strong PFS influence (risk of negative transfer)

For each λ, LOOCO AUROC@12m is computed per centre and averaged.

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_multitask_generator
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data.preprocess import impute_median
from generator.fitting import fit_reference_generator

TAU_OS  = 12.0
K       = 7           # latent dims — slightly larger than 6 to give more room
OUT_DIR = _ROOT / "outputs" / "tables" / "msk_genomic"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")
_CLINICAL_CSV  = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_GENOMIC_CSV   = Path("/Users/ludole/Downloads/Job-8338737215013712236690668870.csv")
_PDL1_PARQUET  = Path("/Users/ludole/Downloads/PDL1_SCORE.parquet")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]

LAMBDAS = [0.0, 0.3, 0.5, 1.0]


# ── helpers ───────────────────────────────────────────────────────────────────

def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def _imp(X, meds):
    return np.where(np.isnan(X), meds, X)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_i3lung() -> list[dict]:
    """i3lung centres, all IO lines, 7 common clinical features + OS survival."""
    rwd = pd.read_csv(_RWD_PATH)
    out = pd.read_csv(_OUTCOMES_PATH)
    df  = rwd.merge(
        out[["Subject", "OS MONTHS", "DEATH EVENT"]],
        on="Subject", how="inner"
    ).dropna(subset=["OS MONTHS", "DEATH EVENT"])

    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        X    = grp[COMMON_FEATS].to_numpy(float)
        t    = grp["OS MONTHS"].to_numpy(float)
        ev   = grp["DEATH EVENT"].to_numpy(float)
        y_os = ((t <= TAU_OS) & (ev == 1)).astype(int)
        centers.append({
            "X": X, "t": t, "delta": ev,
            "y_os": y_os, "n": len(X), "center_id": cid, "name": name,
        })
        print(f"  {name}: n={len(X)}  dead@{TAU_OS:.0f}m={y_os.sum()} "
              f"({y_os.mean()*100:.1f}%)")
    return centers


def load_msk_pfs() -> dict:
    """MSK clinical features + PFS survival times (all n=368 patients)."""
    clin = pd.read_csv(_CLINICAL_CSV)

    sex     = (clin["sex"] == 1).astype(float)
    ecog    = pd.to_numeric(clin["ecog"],  errors="coerce")
    nlr     = pd.to_numeric(clin["dnlr"],  errors="coerce")
    smk_cur = (clin["smoking_status"] == 1).astype(float)
    smk_for = (clin["smoking_status"] == 2).astype(float)
    smk_nev = (clin["smoking_status"] == 0).astype(float)
    try:
        pdl1_df = pd.read_parquet(_PDL1_PARQUET)
        pdl1_df.index.name = "main_index"
        pdl1_df = pdl1_df.reset_index()
        m2      = clin.merge(pdl1_df, on="main_index", how="left")
        score   = pd.to_numeric(m2["Sauter PD-L1 Score"], errors="coerce")
        pdl1    = np.where(score==0, 0.,
                  np.where((score>0)&(score<50), 1.,
                  np.where(score>=50, 2., np.nan)))
    except Exception:
        pdl1 = np.full(len(clin), np.nan)

    X_clin = np.column_stack([sex, ecog, nlr, smk_cur, smk_for, smk_nev, pdl1])
    pfs    = pd.to_numeric(clin["pfs"],         errors="coerce").to_numpy(float)
    pfs_e  = pd.to_numeric(clin["pfs_censor"],  errors="coerce").to_numpy(float)

    valid  = ~np.isnan(pfs) & ~np.isnan(pfs_e)
    print(f"  MSK PFS: n={valid.sum()}")
    return {
        "X_clin":    X_clin[valid],
        "t_pfs":     pfs[valid],
        "delta_pfs": pfs_e[valid],
    }


# ── LOOCO evaluation ──────────────────────────────────────────────────────────

def looco_auroc(
    i3lung:       list[dict],
    msk_pfs:      dict,
    lam:          float,
    k:            int = K,
    verbose:      bool = True,
) -> pd.DataFrame:

    rows = []
    ext_arg = None if lam == 0.0 else msk_pfs   # pass MSK only when λ > 0

    for held_idx in range(len(i3lung)):
        held   = i3lung[held_idx]
        src_cs = [c for i, c in enumerate(i3lung) if i != held_idx]
        y_tgt  = held["y_os"]
        if len(np.unique(y_tgt)) < 2:
            continue

        # ── impute source + target with source medians ────────────────────
        X_src = np.vstack([c["X"] for c in src_cs])
        _, meds = impute_median(X_src)
        src_imp = [
            {**c, "X": _imp(c["X"], meds)}
            for c in src_cs
        ]
        X_tgt_imp = _imp(held["X"], meds)

        # ── prepare external_pfs dict with source-imputed MSK features ────
        if ext_arg is not None:
            ext = {
                "X_clin":    _imp(msk_pfs["X_clin"], meds),
                "t_pfs":     msk_pfs["t_pfs"],
                "delta_pfs": msk_pfs["delta_pfs"],
            }
        else:
            ext = None

        # ── fit generator ─────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A, w, _, sc = fit_reference_generator(
                src_imp, k_total=k, tau=TAU_OS,
                time_distribution="cox",
                external_pfs=ext, lam_pfs=lam,
            )

        X_src_s = sc.transform(_imp(X_src, meds))
        X_tgt_s = sc.transform(X_tgt_imp)
        mu_src  = (X_src_s @ A).mean(0)
        score   = (X_tgt_s @ A - mu_src) @ w
        auc     = _auroc(y_tgt, score)

        rows.append({
            "center": held["name"], "n": held["n"],
            "lam": lam, "auroc": round(auc, 4),
        })
        if verbose:
            print(f"    {held['name']:>6}  n={held['n']:>3}  "
                  f"λ={lam}  AUROC={auc:.4f}")

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung (all IO lines, OS outcome)...")
    i3lung = load_i3lung()

    print("\nLoading MSK PFS (all patients)...")
    msk_pfs = load_msk_pfs()

    all_rows = []
    summary  = []

    print(f"\n=== Multitask generator — LOOCO OS@{TAU_OS:.0f}m  (k={K}) ===")
    hdr = (f"  {'λ':>4}  {'mean AUROC':>10}  "
           + "  ".join(f"{c['name']:>6}" for c in i3lung))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for lam in LAMBDAS:
        lbl = "OS-only (baseline)" if lam == 0.0 else f"λ={lam}"
        print(f"\n  -- {lbl} --")
        df_lam = looco_auroc(i3lung, msk_pfs, lam=lam, k=K)
        all_rows.append(df_lam)

        mean_auc = df_lam["auroc"].mean()
        center_aucs = df_lam.set_index("center")["auroc"]
        row_str = f"  {lam:>4}  {mean_auc:>10.4f}  "
        row_str += "  ".join(f"{center_aucs.get(c['name'], float('nan')):>6.4f}"
                              for c in i3lung)
        summary.append(row_str)

    print(f"\n  Summary (k={K}):")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in summary:
        print(row)

    df_all = pd.concat(all_rows, ignore_index=True)
    out_csv = OUT_DIR / "multitask_generator.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # Report delta vs baseline (λ=0)
    base = df_all[df_all["lam"] == 0.0].set_index("center")["auroc"]
    print("\n  Δ AUROC vs OS-only baseline:")
    for lam in LAMBDAS[1:]:
        sub  = df_all[df_all["lam"] == lam].set_index("center")["auroc"]
        delta = (sub - base).mean()
        print(f"    λ={lam}  mean Δ={delta:+.4f}")


if __name__ == "__main__":
    run_all()
