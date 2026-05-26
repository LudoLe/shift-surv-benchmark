"""
Cross-modal genomic latent transfer (LOOCO, OS@12m).

Revised framing: since joint PCA on [clin+geno] produces loading components
dominated by genomic variance (the clinical submatrix A_clin[:7,:] carries
near-zero signal), we instead keep the latent space in the clinical subspace
and vary ONLY the supervision signal used to fit the risk weights.

Four variants
-------------
  A  A_src / OS   : PCA on i3lung source [clin], Cox on i3lung source OS
                    — standard lat_cox baseline (from run_looco_full_eval.py)
  B  A_MSK / OS   : PCA on MSK clinical (k dims), LR on MSK OS binary
                    — "MSK clinical OS model transferred to i3lung"
  C  A_MSK / PFS  : PCA on MSK clinical (k dims), Cox on MSK PFS
                    — main hypothesis: PFS supervision from MSK captures
                      more relevant clinical-risk signal (via partial
                      clinical-genomic correlation) than i3lung OS supervision
  D  Direct MSK   : no latent space; raw Cox weights from MSK clinical + PFS
                    — simplest transfer baseline

Deltas are relative to Variant A (lat_cox baseline).

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_genomic_latent_transfer
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
from generator.fitting import (
    fit_loading_matrix,
    fit_risk_weights,
    fit_risk_weights_cox,
    fit_reference_generator,
)

TAU_OS  = 12.0
K       = 6           # latent dimensions
OUT_DIR = _ROOT / "outputs" / "tables" / "msk_genomic"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")
_CLINICAL_CSV  = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_GENOMIC_CSV   = Path("/Users/ludole/Downloads/Job-8338737215013712236690668870.csv")
_PDL1_PARQUET  = Path("/Users/ludole/Downloads/PDL1_SCORE.parquet")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def _imp(X, meds):
    return np.where(np.isnan(X), meds, X)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_i3lung() -> list[dict]:
    """
    Load i3lung centres (all IO lines), 7 clinical features + OS survival data.
    """
    rwd = pd.read_csv(_RWD_PATH)
    out = pd.read_csv(_OUTCOMES_PATH)
    df  = rwd.merge(
        out[["Subject", "OS MONTHS", "DEATH EVENT"]],
        on="Subject", how="inner"
    ).dropna(subset=["OS MONTHS", "DEATH EVENT"])

    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        X        = grp[COMMON_FEATS].to_numpy(float)
        t_os     = grp["OS MONTHS"].to_numpy(float)
        delta_os = grp["DEATH EVENT"].to_numpy(float)
        y_os     = ((t_os <= TAU_OS) & (delta_os == 1)).astype(int)
        centers.append({
            "X": X, "t": t_os, "delta": delta_os,
            "y_os": y_os, "n": len(X), "center_id": cid, "name": name,
        })
        print(f"  {name}: n={len(X)}  dead@{TAU_OS:.0f}m={y_os.sum()} "
              f"({y_os.mean()*100:.1f}%)")
    return centers


def load_msk() -> dict:
    """
    Load MSK with 7 clinical features, PFS survival times, and OS binary label.
    """
    clin = pd.read_csv(_CLINICAL_CSV)
    geno = pd.read_csv(_GENOMIC_CSV)  # needed only to flag has_geno

    geno_ids = geno[["Patient_ID"]].copy()
    merged   = clin.merge(geno_ids, left_on="dmp_pt_id",
                          right_on="Patient_ID", how="left")
    has_geno = (~merged["Patient_ID"].isna()).to_numpy()

    sex     = (merged["sex"] == 1).astype(float)
    ecog    = pd.to_numeric(merged["ecog"],  errors="coerce")
    nlr     = pd.to_numeric(merged["dnlr"],  errors="coerce")
    smk_cur = (merged["smoking_status"] == 1).astype(float)
    smk_for = (merged["smoking_status"] == 2).astype(float)
    smk_nev = (merged["smoking_status"] == 0).astype(float)
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

    pfs   = pd.to_numeric(clin["pfs"],          errors="coerce").to_numpy(float)
    pfs_e = pd.to_numeric(clin["pfs_censor"],   errors="coerce").to_numpy(float)
    os_m  = pd.to_numeric(clin["os_int"],       errors="coerce").to_numpy(float)
    vital = pd.to_numeric(clin["vital_status"], errors="coerce").to_numpy(float)

    valid = ~np.isnan(pfs) & ~np.isnan(pfs_e)
    X_clin   = X_clin[valid]
    pfs      = pfs[valid];    pfs_e = pfs_e[valid]
    os_m     = os_m[valid];   vital = vital[valid]
    has_geno = has_geno[valid]

    valid_os = ~np.isnan(os_m) & ~np.isnan(vital)
    y_os     = np.where(valid_os,
                        ((vital == 1) & (os_m <= TAU_OS)).astype(float), np.nan)

    print(f"  MSK: n={valid.sum()}  genomic subset={has_geno.sum()}")
    print(f"  MSK valid OS: {valid_os.sum()}  "
          f"dead@{TAU_OS:.0f}m={int(np.nansum(y_os))}")

    return {
        "X_clin":   X_clin,
        "t_pfs":    pfs,    "delta_pfs": pfs_e,
        "y_os":     y_os,   "valid_os":  valid_os,
        "has_geno": has_geno,
        "n":        int(valid.sum()),
    }


# ── main experiment ───────────────────────────────────────────────────────────

def run_looco_latent_variants(
    i3lung: list[dict],
    msk:    dict,
    k:      int = K,
) -> pd.DataFrame:
    """
    LOOCO on i3lung centres, OS@12m.

    Variants:
      A  A_src/OS   — PCA on i3lung source, Cox on i3lung OS  (lat_cox baseline)
      B  A_MSK/OS   — PCA on MSK clinical,  LR  on MSK OS binary
      C  A_MSK/PFS  — PCA on MSK clinical,  Cox on MSK PFS    (main hypothesis)
      D  Direct/PFS — no PCA; raw Cox weights from MSK clinical + PFS
    """
    print(f"\n=== Latent transfer variants  (k={k}, τ_OS={TAU_OS}m) ===")
    print(f"  A=src/OS (baseline)  B=MSK/OS  C=MSK/PFS  D=Direct/PFS")
    hdr = (f"  {'center':>6}  {'n':>4}  {'A_src':>7}  "
           f"{'B_mOS':>7}  {'C_mPFS':>7}  {'D_dir':>7}  "
           f"{'ΔB':>6}  {'ΔC':>6}  {'ΔD':>6}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    D    = len(i3lung)
    rows = []

    # Pre-extract MSK data for variants B, C, D (constant across folds)
    # --- OS subset (valid OS, full MSK) ---
    mask_os  = msk["valid_os"]
    X_msk_os = msk["X_clin"][mask_os]
    y_os_msk = msk["y_os"][mask_os].astype(int)

    # --- PFS (all MSK with valid PFS) ---
    X_msk_pfs   = msk["X_clin"]
    t_pfs_msk   = msk["t_pfs"]
    d_pfs_msk   = msk["delta_pfs"]

    for held_idx in range(D):
        held   = i3lung[held_idx]
        src_cs = [c for i, c in enumerate(i3lung) if i != held_idx]
        y_tgt  = held["y_os"]
        if len(np.unique(y_tgt)) < 2:
            continue

        X_src = np.vstack([c["X"]     for c in src_cs])
        t_src = np.concatenate([c["t"]     for c in src_cs])
        d_src = np.concatenate([c["delta"] for c in src_cs])

        # ── impute: i3lung source medians for all clinical features ────────
        _, meds_src = impute_median(X_src)
        X_src_imp   = _imp(X_src,        meds_src)
        X_tgt_imp   = _imp(held["X"],    meds_src)
        X_msk_os_imp  = _imp(X_msk_os,  meds_src)
        X_msk_pfs_imp = _imp(X_msk_pfs, meds_src)

        # ── Variant A: standard lat_cox (i3lung source, OS) ───────────────
        src_imp_a = [
            {**c, "X": _imp(c["X"], meds_src)}
            for c in src_cs
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A_a, w_a, _, sc_a = fit_reference_generator(
                src_imp_a, k_total=k, tau=TAU_OS, time_distribution="cox")
        X_src_sa = sc_a.transform(X_src_imp)
        X_tgt_sa = sc_a.transform(X_tgt_imp)
        mu_a     = (X_src_sa @ A_a).mean(0)
        score_A  = (X_tgt_sa @ A_a - mu_a) @ w_a

        # ── Common clinical scale for MSK-based variants: pool i3lung + MSK ─
        sc_j = StandardScaler().fit(
            np.vstack([X_src_imp, X_msk_pfs_imp]))  # all MSK clinical
        X_tgt_sj      = sc_j.transform(X_tgt_imp)
        X_msk_os_sj   = sc_j.transform(X_msk_os_imp)
        X_msk_pfs_sj  = sc_j.transform(X_msk_pfs_imp)

        # ── Variant B: PCA on MSK clinical, LR on MSK OS ──────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A_b   = fit_loading_matrix(X_msk_os_sj, k)
            U_msk_b = X_msk_os_sj @ A_b
            if len(np.unique(y_os_msk)) >= 2:
                w_b = fit_risk_weights(U_msk_b, y_os_msk)
            else:
                w_b = np.ones(k) / np.sqrt(k)
        mu_b    = U_msk_b.mean(0)
        score_B = (X_tgt_sj @ A_b - mu_b) @ w_b

        # ── Variant C: PCA on MSK clinical, Cox on MSK PFS ────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A_c   = fit_loading_matrix(X_msk_pfs_sj, k)
            U_msk_c = X_msk_pfs_sj @ A_c
            w_c   = fit_risk_weights_cox(U_msk_c, t_pfs_msk, d_pfs_msk)
        mu_c    = U_msk_c.mean(0)
        score_C = (X_tgt_sj @ A_c - mu_c) @ w_c

        # ── Variant D: direct Cox on MSK clinical features, no latent ─────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_d   = fit_risk_weights_cox(X_msk_pfs_sj, t_pfs_msk, d_pfs_msk)
        score_D = X_tgt_sj @ w_d

        # ── evaluate ──────────────────────────────────────────────────────
        a_A = _auroc(y_tgt, score_A)
        a_B = _auroc(y_tgt, score_B)
        a_C = _auroc(y_tgt, score_C)
        a_D = _auroc(y_tgt, score_D)

        sign = lambda d: "▲" if d > 0.005 else ("▼" if d < -0.005 else "·")
        print(f"  {held['name']:>6}  {held['n']:>4}  "
              f"{a_A:>7.4f}  {a_B:>7.4f}  {a_C:>7.4f}  {a_D:>7.4f}  "
              f"{a_B-a_A:>+5.3f}{sign(a_B-a_A)}  "
              f"{a_C-a_A:>+5.3f}{sign(a_C-a_A)}  "
              f"{a_D-a_A:>+5.3f}{sign(a_D-a_A)}")

        rows.append({
            "center":        held["name"],
            "n":             held["n"],
            "auroc_src_os":  round(a_A, 4),
            "auroc_msk_os":  round(a_B, 4),
            "auroc_msk_pfs": round(a_C, 4),
            "auroc_dir_pfs": round(a_D, 4),
            "delta_msk_os":  round(a_B - a_A, 4),
            "delta_msk_pfs": round(a_C - a_A, 4),
            "delta_dir_pfs": round(a_D - a_A, 4),
        })

    df = pd.DataFrame(rows)
    print("\n  Mean AUROC across i3lung centres:")
    for col, label in [
        ("auroc_src_os",  "A  A_src/OS (baseline)"),
        ("auroc_msk_os",  "B  A_MSK/OS           "),
        ("auroc_msk_pfs", "C  A_MSK/PFS          "),
        ("auroc_dir_pfs", "D  Direct/PFS         "),
    ]:
        print(f"    {label} = {df[col].mean():.4f}")
    print()
    for col, label in [
        ("delta_msk_os",  "Δ(B−A)"),
        ("delta_msk_pfs", "Δ(C−A)"),
        ("delta_dir_pfs", "Δ(D−A)"),
    ]:
        print(f"    {label} = {df[col].mean():+.4f}")
    return df


# ── entry point ───────────────────────────────────────────────────────────────

def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung (all IO lines, OS outcome, 7 clinical features)...")
    i3lung = load_i3lung()

    print("\nLoading MSK (clinical features, PFS+OS outcomes)...")
    msk = load_msk()

    df = run_looco_latent_variants(i3lung, msk, k=K)
    out_csv = OUT_DIR / "cross_modal_latent_transfer.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    run_all()
