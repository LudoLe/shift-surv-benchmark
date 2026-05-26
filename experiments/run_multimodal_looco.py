"""
Multimodal generator LOOCO experiment.

Goal
----
Test whether fitting a shared latent space on multiple modalities improves
AUROC@12m over a clinical-only baseline.

Three conditions (all LOOCO on 6 i3lung centres):

  RWD    k=6  clinical (11-dim)                         — 100% coverage
  MM_rad k=6  clinical + PyRadiomics (128-dim)           —  42% coverage  (*)
  MM_geo k=4  clinical + genomic flags (4-dim)           —  82% coverage

(*) MH has only 3% radiomics coverage and is expected to match RWD.

Architecture
------------
Each modality has its own loading matrix A_m (p_m × k) that maps into the
SAME shared k-dimensional latent space.  Missing patients are handled by
masked-average fusion — no imputation:

    U_i = (Σ_m mask_im · X_m_i @ A_m) / (Σ_m mask_im)
    η_i = U_i @ w

Evaluation
----------
  AUROC on ALL held-out patients            (main comparison)
  AUROC on MODALITY-POSITIVE subset         (patients who have the extra modality)
  AUROC on MODALITY-NEGATIVE subset         (clinical-only patients)

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_multimodal_looco
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

from generator.fitting import (
    fit_loading_matrix,
    fit_risk_weights_cox,
    fit_multimodal_loading_matrices,
    score_multimodal,
)

TAU     = 12.0
K_RWD   = 6   # clinical-only and rwd+rad (min(11,128) = 11 ≥ 6)
K_GEO   = 4   # rwd+geno constrained by dim(geno) = 4
OUT_DIR = _ROOT / "outputs" / "tables" / "multimodal"

_MM_DIR      = _ROOT / "data_multimodal"
_RWD_CSV     = _MM_DIR / "rwd_processed.csv"
_GENO_CSV    = _MM_DIR / "genomics_processed.csv"
_OUTCOMES_CSV = _MM_DIR / "outcomes.csv"
_PARQUET     = _MM_DIR / "features_dataset_radpy_fixed.parquet"

RWD_FEATS  = ["SEX", "ECOG PS", "log_LDH", "log_NLR",
              "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
              "PDL1 CATEGORY",
              "PARENCHYMAL BRAIN METS AT IO START",
              "LIVER METS AT IO START", "BONE METS AT IO START"]
GENO_FEATS = ["DRIVER", "KRAS", "P53", "STK11"]
P_RWD, P_GEO = len(RWD_FEATS), len(GENO_FEATS)
P_RAD = 128   # from parquet mod2


# ── helpers ───────────────────────────────────────────────────────────────────

def _auroc(y, s):
    if len(y) < 5 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def _scaler_fit_transform(X_src, X_tgt, mask_src=None, mask_tgt=None):
    """
    Fit StandardScaler on source (optionally only on mask_src rows),
    transform both source and target.  mask_tgt rows not in mask are left as-is
    (their values won't be used in the masked fusion).
    """
    if mask_src is not None:
        idx = mask_src.astype(bool)
        sc = StandardScaler().fit(X_src[idx])
        X_src_s = X_src.copy()
        X_src_s[idx] = sc.transform(X_src[idx])
    else:
        sc = StandardScaler().fit(X_src)
        X_src_s = sc.transform(X_src)

    if mask_tgt is not None:
        idx_t = mask_tgt.astype(bool)
        X_tgt_s = X_tgt.copy()
        if idx_t.any():
            X_tgt_s[idx_t] = sc.transform(X_tgt[idx_t])
    else:
        X_tgt_s = sc.transform(X_tgt)

    return X_src_s, X_tgt_s


# ── data loader ───────────────────────────────────────────────────────────────

def load_i3lung_multimodal() -> list[dict]:
    """
    Load i3lung multimodal data per centre.  Each centre dict contains:
      X_rwd       : (n, 11)  clinical features — always present
      X_geno      : (n,  4)  genomic flags     — NaN-free; use mask_geno
      mask_geno   : (n,) float  1 = patient has genomic data
      X_rad       : (n, 128) PyRadiomics       — zeros for missing; use mask_rad
      mask_rad    : (n,) float  1 = patient has radiomics
      t, delta, y_os, n, center_id, name
    """
    rwd  = pd.read_csv(_RWD_CSV)
    geno = pd.read_csv(_GENO_CSV)
    out  = pd.read_csv(_OUTCOMES_CSV)
    pq   = pd.read_parquet(_PARQUET)[["Subject", "mod2"]]  # radiomics

    df = (
        rwd
        .merge(geno[["Subject"] + GENO_FEATS], on="Subject", how="left")
        .merge(out[["Subject", "OS MONTHS", "DEATH EVENT"]], on="Subject", how="inner")
        .merge(pq, on="Subject", how="left")
        .dropna(subset=["OS MONTHS", "DEATH EVENT"])
    )

    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        # ── clinical ──────────────────────────────────────────────────────
        X_rwd = grp[RWD_FEATS].to_numpy(float)

        # ── genomic ───────────────────────────────────────────────────────
        X_geno_raw = grp[GENO_FEATS].to_numpy(float)
        mask_geno  = (~np.isnan(X_geno_raw).any(axis=1)).astype(float)
        X_geno     = np.where(np.isnan(X_geno_raw), 0.0, X_geno_raw)

        # ── radiomics ─────────────────────────────────────────────────────
        rad_arrays = grp["mod2"].tolist()
        mask_rad   = np.array([1.0 if v is not None else 0.0 for v in rad_arrays])
        X_rad      = np.zeros((len(grp), P_RAD))
        for i, v in enumerate(rad_arrays):
            if v is not None:
                X_rad[i] = np.array(v, dtype=float)

        t    = grp["OS MONTHS"].to_numpy(float)
        ev   = grp["DEATH EVENT"].to_numpy(float)
        y_os = ((t <= TAU) & (ev == 1)).astype(int)

        centers.append({
            "name": name, "center_id": cid, "n": len(grp),
            "X_rwd": X_rwd, "X_geno": X_geno, "mask_geno": mask_geno,
            "X_rad":  X_rad,  "mask_rad":  mask_rad,
            "t": t, "delta": ev, "y_os": y_os,
        })
        print(f"  {name}: n={len(grp)}"
              f"  geno={int(mask_geno.sum())} ({mask_geno.mean()*100:.0f}%)"
              f"  rad={int(mask_rad.sum())} ({mask_rad.mean()*100:.0f}%)"
              f"  dead@{TAU:.0f}m={y_os.sum()} ({y_os.mean()*100:.0f}%)")
    return centers


# ── score helper using single A ───────────────────────────────────────────────

def _score_single(X_s, A, w, mu_src):
    return (X_s @ A - mu_src) @ w


# ── LOOCO fold ────────────────────────────────────────────────────────────────

def looco_fold(held: dict, src_cs: list[dict]) -> dict:
    y_held = held["y_os"]
    if len(np.unique(y_held)) < 2:
        return {}

    # ── pool source arrays ────────────────────────────────────────────────
    X_rwd_src  = np.vstack([c["X_rwd"]  for c in src_cs])
    X_geno_src = np.vstack([c["X_geno"] for c in src_cs])
    X_rad_src  = np.vstack([c["X_rad"]  for c in src_cs])
    mask_geno_src = np.concatenate([c["mask_geno"] for c in src_cs])
    mask_rad_src  = np.concatenate([c["mask_rad"]  for c in src_cs])
    t_src  = np.concatenate([c["t"]     for c in src_cs])
    d_src  = np.concatenate([c["delta"] for c in src_cs])
    N_src  = len(t_src)

    # ── standardise feature blocks ────────────────────────────────────────
    # RWD: pooled (always present)
    sc_rwd = StandardScaler().fit(X_rwd_src)
    X_rwd_src_s  = sc_rwd.transform(X_rwd_src)
    X_rwd_held_s = sc_rwd.transform(held["X_rwd"])

    # Genomic: fit on geno-positive source patients only
    X_geno_src_s, X_geno_held_s = _scaler_fit_transform(
        X_geno_src, held["X_geno"],
        mask_src=mask_geno_src, mask_tgt=held["mask_geno"],
    )

    # Radiomics: fit on rad-positive source patients only
    X_rad_src_s, X_rad_held_s = _scaler_fit_transform(
        X_rad_src, held["X_rad"],
        mask_src=mask_rad_src, mask_tgt=held["mask_rad"],
    )

    # Shorthands
    mg_src = mask_geno_src;  mg_held = held["mask_geno"]
    mr_src = mask_rad_src;   mr_held = held["mask_rad"]
    ones   = np.ones(N_src)

    result = {}

    # ════════════════════════════════════════════════════════════════════
    # CONDITION RWD  (clinical-only, k=K_RWD)
    # ════════════════════════════════════════════════════════════════════
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        A_rwd = fit_loading_matrix(X_rwd_src_s, K_RWD)
        U_src_rwd = X_rwd_src_s @ A_rwd
        w_rwd = fit_risk_weights_cox(U_src_rwd, t_src, d_src)

    mu_rwd   = U_src_rwd.mean(0)
    s_rwd    = _score_single(X_rwd_held_s, A_rwd, w_rwd, mu_rwd)
    result["rwd_all"] = _auroc(y_held, s_rwd)

    for lbl, mask in [("gp", mg_held.astype(bool)),
                       ("gn", (~mg_held.astype(bool)))]:
        if mask.sum() >= 5:
            result[f"rwd_geo{lbl}"] = _auroc(y_held[mask], s_rwd[mask])
    for lbl, mask in [("rp", mr_held.astype(bool)),
                       ("rn", (~mr_held.astype(bool)))]:
        if mask.sum() >= 5:
            result[f"rwd_rad{lbl}"] = _auroc(y_held[mask], s_rwd[mask])

    mods_rad = [
        {"key": "rwd", "X": X_rwd_src_s, "mask": ones},
        {"key": "rad", "X": X_rad_src_s,  "mask": mr_src},
    ]

    # ════════════════════════════════════════════════════════════════════
    # CONDITION MM_rad_avg  (rwd + radiomics, masked-average, k=K_RWD)
    # ════════════════════════════════════════════════════════════════════
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        A_avg, w_avg, _ = fit_multimodal_loading_matrices(
            mods_rad, t_src, d_src, k_total=K_RWD, fusion="average",
        )

    mu_avg   = (X_rwd_src_s @ A_avg["rwd"]).mean(0)
    s_avg    = score_multimodal(
        X_dict={"rwd": X_rwd_held_s, "rad": X_rad_held_s},
        mask={"rwd": np.ones(held["n"]), "rad": mr_held},
        A_dict=A_avg, w=w_avg, mu_src=mu_avg,
    )
    result["mmavg_all"] = _auroc(y_held, s_avg)
    for lbl, mask in [("rp", mr_held.astype(bool)), ("rn", ~mr_held.astype(bool))]:
        if mask.sum() >= 5:
            result[f"mmavg_{lbl}"] = _auroc(y_held[mask], s_avg[mask])

    # ════════════════════════════════════════════════════════════════════
    # CONDITION MM_rad_add  (rwd + radiomics, additive, k=K_RWD)
    # ════════════════════════════════════════════════════════════════════
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        A_add, w_add, alp_add = fit_multimodal_loading_matrices(
            mods_rad, t_src, d_src, k_total=K_RWD, fusion="additive",
        )

    mu_add   = (X_rwd_src_s @ A_add["rwd"]).mean(0)
    s_add    = score_multimodal(
        X_dict={"rwd": X_rwd_held_s, "rad": X_rad_held_s},
        mask={"rwd": np.ones(held["n"]), "rad": mr_held},
        A_dict=A_add, w=w_add, mu_src=mu_add, alphas=alp_add,
    )
    result["mmadd_all"] = _auroc(y_held, s_add)
    result["alpha_rad"] = float(alp_add.get("rad", np.nan))
    for lbl, mask in [("rp", mr_held.astype(bool)), ("rn", ~mr_held.astype(bool))]:
        if mask.sum() >= 5:
            result[f"mmadd_{lbl}"] = _auroc(y_held[mask], s_add[mask])

    # ════════════════════════════════════════════════════════════════════
    # CONDITION MM_geo  (rwd + genomic flags, average, k=K_GEO)  — reference
    # ════════════════════════════════════════════════════════════════════
    mods_geo = [
        {"key": "rwd",  "X": X_rwd_src_s,  "mask": ones},
        {"key": "geno", "X": X_geno_src_s, "mask": mg_src},
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        A_geo, w_geo, _ = fit_multimodal_loading_matrices(
            mods_geo, t_src, d_src, k_total=K_GEO, fusion="average",
        )

    mu_geo   = (X_rwd_src_s @ A_geo["rwd"]).mean(0)
    s_geo    = score_multimodal(
        X_dict={"rwd": X_rwd_held_s, "geno": X_geno_held_s},
        mask={"rwd": np.ones(held["n"]), "geno": mg_held},
        A_dict=A_geo, w=w_geo, mu_src=mu_geo,
    )
    result["mmgeo_all"] = _auroc(y_held, s_geo)

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def run_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung multimodal data...")
    i3lung = load_i3lung_multimodal()
    D = len(i3lung)

    print(f"\nK_RWD={K_RWD}  K_GEO={K_GEO}  τ={TAU}m")
    print(f"\n{'='*90}")
    hdr = (f"  {'centre':>6}  "
           f"{'RWD':>7}  {'avg_all':>7}  {'Δavg':>7}  "
           f"{'add_all':>7}  {'Δadd':>7}  {'α_rad':>7}  "
           f"{'avg_rp':>7}  {'add_rp':>7}  {'geo_all':>7}")
    print(hdr)
    print("  " + "-" * 88)

    rows = []
    for held_idx in range(D):
        held   = i3lung[held_idx]
        src_cs = [c for i, c in enumerate(i3lung) if i != held_idx]

        res = looco_fold(held, src_cs)
        if not res:
            continue

        def _f(k):
            v = res.get(k, float("nan"))
            return f"{v:7.4f}" if not np.isnan(v) else f"{'—':>7}"
        def _d(a, b):
            va, vb = res.get(a, np.nan), res.get(b, np.nan)
            return f"{vb-va:+7.4f}" if (not np.isnan(va) and not np.isnan(vb)) else f"{'—':>7}"

        print(f"  {held['name']:>6}  "
              f"{_f('rwd_all')}  {_f('mmavg_all')}  {_d('rwd_all','mmavg_all')}  "
              f"{_f('mmadd_all')}  {_d('rwd_all','mmadd_all')}  {_f('alpha_rad')}  "
              f"{_f('mmavg_rp')}  {_f('mmadd_rp')}  {_f('mmgeo_all')}")

        row = {"center": held["name"], "n": held["n"],
               "rad_frac":  float(held["mask_rad"].mean()),
               "geno_frac": float(held["mask_geno"].mean())}
        row.update(res)
        rows.append(row)

    df = pd.DataFrame(rows)

    def _mg(col):
        return df[col].mean() if col in df.columns else float("nan")
    def _f(v): return f"{v:7.4f}" if not np.isnan(v) else f"{'—':>7}"
    def _d(v): return f"{v:+7.4f}" if not np.isnan(v) else f"{'—':>7}"

    print(f"\n{'='*90}")
    print("  Summary: mean AUROC over centres")
    print(hdr)
    print("  " + "-" * 88)
    print(f"  {'mean':>6}  "
          f"{_f(_mg('rwd_all'))}  {_f(_mg('mmavg_all'))}  "
          f"{_d(_mg('mmavg_all')-_mg('rwd_all'))}  "
          f"{_f(_mg('mmadd_all'))}  {_d(_mg('mmadd_all')-_mg('rwd_all'))}  "
          f"{_f(_mg('alpha_rad'))}  "
          f"{_f(_mg('mmavg_rp'))}  {_f(_mg('mmadd_rp'))}  {_f(_mg('mmgeo_all'))}")

    out_csv = OUT_DIR / "multimodal_looco.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print("\nInterpretation:")
    print("  Δavg  → masked-average fusion vs RWD")
    print("  Δadd  → additive fusion vs RWD  (clinical signal preserved)")
    print("  α_rad → learned radiomics weight in additive fusion (>0 = rad helps training)")
    print("  *_rp  → AUROC on patients WITH radiomics only")
    print("  MH has only 3% rad coverage → Δrad ≈ 0 expected")


if __name__ == "__main__":
    run_all()
