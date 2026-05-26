"""
Two-stage PFS proxy with lat_cox generator.

Stage 1 (constant): fit MSK → PFS Cox model (clinical only, or clin+geno)
Stage 2 (LOOCO):    augment i3lung clinical features with the PFS_score,
                    fit lat_cox (PCA + Cox OS) on the 8-feature matrix,
                    compare with 7-feature lat_cox baseline.

Variants
--------
  A  lat_cox 7    — lat_cox on 7 clinical features (baseline)
  B  lat_cox+PFS  — lat_cox on [7 clin | PFS_clin_score]  (MSK clinical Cox)
  C  lat_cox+PFSg — lat_cox on [7 clin | PFS_geno_score]  (MSK clin+geno Cox,
                    geno features set to 0 when applied to i3lung)

The PFS_score is a scalar computed by the MSK-trained PFS Cox model.
Adding it as an 8th dimension gives the generator a "genomically-informed
clinical direction" from MSK PFS data.

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_latcox_pfs_proxy
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
    fit_risk_weights_cox,
)

TAU_OS  = 12.0
K       = 6
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
    """i3lung, all IO lines, OS survival + 7 clinical features."""
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


def load_msk() -> dict:
    """MSK with 7 clinical + 7 genomic features and PFS survival times."""
    clin = pd.read_csv(_CLINICAL_CSV)
    geno = pd.read_csv(_GENOMIC_CSV)

    geno_sub = geno[["Patient_ID", "TMB", "EGFR_driver", "ERBB2_driver",
                      "BRAF_driver", "MET_driver", "STK11_driver", "ARID1A_driver"]]
    merged = clin.merge(geno_sub, left_on="dmp_pt_id",
                        right_on="Patient_ID", how="left")

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

    tmb_log = np.log1p(
        pd.to_numeric(merged["TMB"], errors="coerce").to_numpy(float))
    drivers = merged[["EGFR_driver", "ERBB2_driver", "BRAF_driver",
                       "MET_driver",  "STK11_driver",  "ARID1A_driver"]
                     ].to_numpy(float)
    tmb_med   = float(np.nanmedian(tmb_log))
    tmb_log   = np.where(np.isnan(tmb_log), tmb_med, tmb_log)
    drivers   = np.where(np.isnan(drivers), 0., drivers)
    X_geno    = np.column_stack([tmb_log.reshape(-1, 1), drivers])
    X_full    = np.column_stack([X_clin, X_geno])

    pfs   = pd.to_numeric(clin["pfs"],          errors="coerce").to_numpy(float)
    pfs_e = pd.to_numeric(clin["pfs_censor"],   errors="coerce").to_numpy(float)
    valid = ~np.isnan(pfs) & ~np.isnan(pfs_e)

    print(f"  MSK: n_total={valid.sum()}")
    return {
        "X_clin": X_clin[valid], "X_full": X_full[valid],
        "t_pfs":  pfs[valid],    "delta_pfs": pfs_e[valid],
        "n":      int(valid.sum()),
    }


# ── lat_cox helper ────────────────────────────────────────────────────────────

def _latcox_score(
    X_src_s: np.ndarray, t_src: np.ndarray, d_src: np.ndarray,
    X_tgt_s: np.ndarray, k: int,
) -> np.ndarray:
    """Fit lat_cox on source, return scores on target."""
    A   = fit_loading_matrix(X_src_s, k)
    U   = X_src_s @ A
    w   = fit_risk_weights_cox(U, t_src, d_src)
    mu  = U.mean(0)
    return (X_tgt_s @ A - mu) @ w


# ── main experiment ───────────────────────────────────────────────────────────

def run_latcox_proxy(i3lung: list[dict], msk: dict, k: int = K) -> pd.DataFrame:
    """
    LOOCO OS@12m.

    A  lat_cox 7 features (baseline)
    B  lat_cox 8 = 7 clin + PFS_score from MSK clinical Cox
    C  lat_cox 8 = 7 clin + PFS_score from MSK clin+geno Cox (geno=0 on i3lung)
    """
    print(f"\n=== Two-stage PFS proxy with lat_cox  (k={k}, τ={TAU_OS}m) ===")
    print(f"  A=lat_cox/7  B=+PFS_clin  C=+PFS_geno")
    hdr = (f"  {'center':>6}  {'n':>4}  {'A_base':>7}  "
           f"{'B_+PFSc':>7}  {'C_+PFSg':>7}  {'ΔB':>6}  {'ΔC':>6}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    D    = len(i3lung)
    rows = []

    # ── Stage 1: fit two MSK PFS Cox models (constant across LOOCO folds) ──
    # We fit on MSK using MSK-internal scale here; re-scale per fold below
    # using a joint i3lung+MSK scaler — so we store raw MSK data and re-fit
    # the weights per fold to ensure consistent scaling.
    # (w_pfs direction is mostly stable across scaler choices since it's
    #  fit on a large fixed dataset; this approach is exact, not approximate.)

    for held_idx in range(D):
        held   = i3lung[held_idx]
        src_cs = [c for i, c in enumerate(i3lung) if i != held_idx]
        y_tgt  = held["y_os"]
        if len(np.unique(y_tgt)) < 2:
            continue

        X_src = np.vstack([c["X"]     for c in src_cs])
        t_src = np.concatenate([c["t"]     for c in src_cs])
        d_src = np.concatenate([c["delta"] for c in src_cs])

        # ── impute ────────────────────────────────────────────────────────
        _, meds_src  = impute_median(X_src)
        X_src_imp    = _imp(X_src,        meds_src)
        X_tgt_imp    = _imp(held["X"],    meds_src)
        X_msk_c_imp  = _imp(msk["X_clin"], meds_src)

        # Genomic features: MSK internal imputation
        _, meds_g = impute_median(msk["X_full"][:, 7:])
        X_geno_imp  = _imp(msk["X_full"][:, 7:], meds_g)
        X_msk_f_imp = np.column_stack([X_msk_c_imp, X_geno_imp])

        # ── joint clinical scale: pool i3lung source + MSK clinical ───────
        sc = StandardScaler().fit(np.vstack([X_src_imp, X_msk_c_imp]))
        X_src_s    = sc.transform(X_src_imp)
        X_tgt_s    = sc.transform(X_tgt_imp)
        X_msk_c_s  = sc.transform(X_msk_c_imp)

        # Genomic scale: MSK internal
        sc_g       = StandardScaler().fit(X_geno_imp)
        X_geno_s   = sc_g.transform(X_geno_imp)
        X_msk_f_s  = np.column_stack([X_msk_c_s, X_geno_s])  # (n_msk, 14)

        # ── Stage 1 (per-fold): MSK PFS Cox models ─────────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # clinical-only PFS model
            w_pfs_clin = fit_risk_weights_cox(X_msk_c_s, msk["t_pfs"], msk["delta_pfs"])
            # clin+geno PFS model
            w_pfs_full = fit_risk_weights_cox(X_msk_f_s, msk["t_pfs"], msk["delta_pfs"])

        # PFS scores for i3lung (clinical only)
        pfs_score_c_src = (X_src_s @ w_pfs_clin).reshape(-1, 1)
        pfs_score_c_tgt = (X_tgt_s @ w_pfs_clin).reshape(-1, 1)

        # Geno variant: apply clinical part of clin+geno model to i3lung
        w_pfs_clin_part = w_pfs_full[:7]  # ignore genomic dimensions
        pfs_score_g_src = (X_src_s @ w_pfs_clin_part).reshape(-1, 1)
        pfs_score_g_tgt = (X_tgt_s @ w_pfs_clin_part).reshape(-1, 1)

        # ── Variant A: lat_cox on 7 clinical ──────────────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            score_A = _latcox_score(X_src_s, t_src, d_src, X_tgt_s, k)

        # ── Variant B: lat_cox on 8 = 7 clin + PFS_clin ───────────────────
        X_src_b = np.hstack([X_src_s, pfs_score_c_src])
        X_tgt_b = np.hstack([X_tgt_s, pfs_score_c_tgt])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            score_B = _latcox_score(X_src_b, t_src, d_src, X_tgt_b, k)

        # ── Variant C: lat_cox on 8 = 7 clin + PFS_geno_clinical_part ────
        X_src_c = np.hstack([X_src_s, pfs_score_g_src])
        X_tgt_c = np.hstack([X_tgt_s, pfs_score_g_tgt])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            score_C = _latcox_score(X_src_c, t_src, d_src, X_tgt_c, k)

        # ── evaluate ──────────────────────────────────────────────────────
        a_A = _auroc(y_tgt, score_A)
        a_B = _auroc(y_tgt, score_B)
        a_C = _auroc(y_tgt, score_C)

        sign = lambda d: "▲" if d > 0.005 else ("▼" if d < -0.005 else "·")
        print(f"  {held['name']:>6}  {held['n']:>4}  "
              f"{a_A:>7.4f}  {a_B:>7.4f}  {a_C:>7.4f}  "
              f"{a_B-a_A:>+5.3f}{sign(a_B-a_A)}  "
              f"{a_C-a_A:>+5.3f}{sign(a_C-a_A)}")

        rows.append({
            "center":       held["name"],
            "n":            held["n"],
            "auroc_base":   round(a_A, 4),
            "auroc_pfs_c":  round(a_B, 4),
            "auroc_pfs_g":  round(a_C, 4),
            "delta_pfs_c":  round(a_B - a_A, 4),
            "delta_pfs_g":  round(a_C - a_A, 4),
        })

    df = pd.DataFrame(rows)
    print(f"\n  Mean AUROC:")
    print(f"    A  lat_cox/7   = {df['auroc_base'].mean():.4f}")
    print(f"    B  +PFS_clin   = {df['auroc_pfs_c'].mean():.4f}")
    print(f"    C  +PFS_geno   = {df['auroc_pfs_g'].mean():.4f}")
    print(f"    Δ(B−A)         = {df['delta_pfs_c'].mean():+.4f}")
    print(f"    Δ(C−A)         = {df['delta_pfs_g'].mean():+.4f}")
    return df


# ── entry point ───────────────────────────────────────────────────────────────

def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung (all IO lines, OS survival)...")
    i3lung = load_i3lung()

    print("\nLoading MSK (clinical+genomic, PFS survival)...")
    msk = load_msk()

    df = run_latcox_proxy(i3lung, msk, k=K)
    out_csv = OUT_DIR / "latcox_pfs_proxy.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    run_all()
