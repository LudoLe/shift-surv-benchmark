"""
PFS transfer learning experiment.

Exp 1 — i3lung LOOCO, PFS > 6m, clinical features only
Exp 2 — MSK internal CV: clinical / genomic / clin+genomic → PFS > 6m
Exp 3 — LOOCO i3lung + MSK (clinical), PFS > 6m
Exp 4 — PFS as proxy for OS (two-stage transfer):
         Stage 1: train MSK clinical+genomic → PFS model
         Stage 2: apply to i3lung to get predicted PFS score
                  use clin + PFS_score to predict OS on i3lung (LOOCO)
         Compare to: clin only → OS (baseline)

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_pfs_transfer
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data.preprocess import impute_median

TAU_PFS = 6.0
TAU_OS  = 12.0
OUT_DIR = _ROOT / "outputs" / "tables" / "msk_genomic"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")
_CLINICAL_CSV  = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_GENOMIC_CSV   = Path("/Users/ludole/Downloads/Job-8338737215013712236690668870.csv")
_PDL1_PARQUET  = Path("/Users/ludole/Downloads/PDL1_SCORE.parquet")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]

_LR_KW  = dict(C=1.0, max_iter=500, solver="lbfgs",
               class_weight="balanced", random_state=0)
_MLP_KW = dict(hidden_layer_sizes=(32,16), activation="relu",
               solver="lbfgs", alpha=0.01, max_iter=1000, random_state=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _long_resp(pfs, pfs_ev, tau=TAU_PFS):
    return ((pfs > tau) | ((pfs <= tau) & (pfs_ev == 0))).astype(int)

def _dead_at_tau(os_m, death_ev, tau=TAU_OS):
    return ((os_m <= tau) & (death_ev == 1)).astype(int)

def _auroc(y, s):
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, s))

def _fit_predict(X_tr, y_tr, X_te, cls, kw):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m = cls(**kw); m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]

def _impute_scale(X_src, X_targets):
    _, meds = impute_median(X_src)
    imp = lambda X: np.where(np.isnan(X), meds, X)
    sc  = StandardScaler().fit(imp(X_src))
    return [sc.transform(imp(X)) for X in [X_src] + X_targets]


# ── data loaders ──────────────────────────────────────────────────────────────

def load_i3lung() -> list[dict]:
    """Load i3lung centers with both PFS and OS outcomes."""
    rwd = pd.read_csv(_RWD_PATH)
    out = pd.read_csv(_OUTCOMES_PATH)
    df  = rwd.merge(
        out[["Subject","PFS MONTHS","PROGRESSION EVENT",
             "OS MONTHS","DEATH EVENT"]],
        on="Subject", how="inner"
    ).dropna(subset=["PFS MONTHS","PROGRESSION EVENT","OS MONTHS","DEATH EVENT"])

    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        X     = grp[COMMON_FEATS].to_numpy(float)
        y_pfs = _long_resp(grp["PFS MONTHS"].to_numpy(float),
                           grp["PROGRESSION EVENT"].to_numpy(float))
        y_os  = _dead_at_tau(grp["OS MONTHS"].to_numpy(float),
                             grp["DEATH EVENT"].to_numpy(float))
        centers.append({
            "X": X, "y_pfs": y_pfs, "y_os": y_os,
            "n": len(grp), "center_id": cid, "name": name,
        })
    return centers


def load_msk() -> dict:
    """Load MSK with both PFS and OS outcomes + genomic features."""
    clin = pd.read_csv(_CLINICAL_CSV)
    geno = pd.read_csv(_GENOMIC_CSV)

    geno_sub = geno[["Patient_ID","TMB","EGFR_driver","ERBB2_driver",
                      "BRAF_driver","MET_driver","STK11_driver","ARID1A_driver"]]
    merged = clin.merge(geno_sub, left_on="dmp_pt_id",
                        right_on="Patient_ID", how="left")

    sex     = (merged["sex"] == 1).astype(float)
    ecog    = pd.to_numeric(merged["ecog"],           errors="coerce")
    nlr     = pd.to_numeric(merged["dnlr"],           errors="coerce")
    smk_cur = (merged["smoking_status"] == 1).astype(float)
    smk_for = (merged["smoking_status"] == 2).astype(float)
    smk_nev = (merged["smoking_status"] == 0).astype(float)
    try:
        pdl1_df = pd.read_parquet(_PDL1_PARQUET)
        pdl1_df.index.name = "main_index"
        pdl1_df = pdl1_df.reset_index()
        m2      = merged.merge(pdl1_df, on="main_index", how="left")
        score   = pd.to_numeric(m2["Sauter PD-L1 Score"], errors="coerce")
        pdl1    = np.where(score==0, 0., np.where((score>0)&(score<50), 1.,
                  np.where(score>=50, 2., np.nan)))
    except Exception:
        pdl1 = np.full(len(merged), np.nan)

    X_clin = np.column_stack([sex, ecog, nlr, smk_cur, smk_for, smk_nev, pdl1])

    tmb_log = np.log1p(pd.to_numeric(merged["TMB"], errors="coerce").to_numpy(float))
    drivers = merged[["EGFR_driver","ERBB2_driver","BRAF_driver",
                       "MET_driver","STK11_driver","ARID1A_driver"]].to_numpy(float)
    tmb_med = float(np.nanmedian(tmb_log))
    tmb_log = np.where(np.isnan(tmb_log), tmb_med, tmb_log)
    drivers = np.where(np.isnan(drivers), 0., drivers)
    X_geno  = np.column_stack([tmb_log.reshape(-1,1), drivers])
    X_full  = np.column_stack([X_clin, X_geno])

    pfs   = pd.to_numeric(merged["pfs"],         errors="coerce").to_numpy(float)
    pfs_e = pd.to_numeric(merged["pfs_censor"],  errors="coerce").to_numpy(float)
    vital = pd.to_numeric(merged["vital_status"],errors="coerce").to_numpy(float)
    os_m  = pd.to_numeric(merged["os_int"],      errors="coerce").to_numpy(float)

    valid = ~np.isnan(pfs) & ~np.isnan(pfs_e)
    X_clin, X_full = X_clin[valid], X_full[valid]
    pfs, pfs_e     = pfs[valid], pfs_e[valid]
    vital          = vital[valid]
    os_m           = os_m[valid]
    has_geno       = (~merged["TMB"].isna()).to_numpy()[valid]

    y_pfs = _long_resp(pfs, pfs_e)
    # OS label: dead within TAU_OS — pfs used as proxy for alive censoring
    y_os  = ((vital == 1) & (os_m <= TAU_OS)).astype(int)
    # For alive, set os_time=pfs (censoring proxy, consistent with registry_vanguri)

    return {"X_clin": X_clin, "X_full": X_full,
            "y_pfs": y_pfs, "y_os": y_os,
            "has_geno": has_geno, "n": len(pfs), "name": "MSK"}


# ── Exp 1: i3lung LOOCO PFS clinical only ────────────────────────────────────

def run_exp1_i3lung_pfs(centers: list[dict]) -> pd.DataFrame:
    print("=== Exp 1: i3lung LOOCO — PFS > 6m, clinical only ===")
    hdr = f"  {'center':>6}  {'model':>4}  {'AUROC':>7}  {'prev':>5}  {'n':>5}"
    print(hdr); print("  " + "-"*(len(hdr)-2))
    rows = []
    for held_idx in range(len(centers)):
        held   = centers[held_idx]
        src_cs = [c for i,c in enumerate(centers) if i != held_idx]
        X_src  = np.vstack([c["X"]     for c in src_cs])
        y_src  = np.concatenate([c["y_pfs"] for c in src_cs])
        y_tgt  = held["y_pfs"]
        if len(np.unique(y_tgt)) < 2: continue
        sc = _impute_scale(X_src, [held["X"]])
        for name, cls, kw in [("lr",LogisticRegression,_LR_KW),
                               ("mlp",MLPClassifier,_MLP_KW)]:
            a = _auroc(y_tgt, _fit_predict(sc[0], y_src, sc[1], cls, kw))
            rows.append({"center":held["name"],"model":name,
                         "auroc":round(a,4),"prev":round(y_tgt.mean(),3),
                         "n":held["n"]})
            print(f"  {held['name']:>6}  {name:>4}  {a:>7.4f}  "
                  f"{y_tgt.mean():>5.3f}  {held['n']:>5}")
    df = pd.DataFrame(rows)
    print(f"\n  Mean AUROC:  lr={df[df.model=='lr']['auroc'].mean():.4f}  "
          f"mlp={df[df.model=='mlp']['auroc'].mean():.4f}\n")
    return df


# ── Exp 2: MSK CV PFS ────────────────────────────────────────────────────────

def run_exp2_msk_cv_pfs(msk: dict, n_splits: int = 5) -> pd.DataFrame:
    mask = msk["has_geno"]
    y    = msk["y_pfs"][mask]
    print(f"=== Exp 2: MSK CV — PFS > 6m  (genomic subset n={mask.sum()}, "
          f"long_resp={y.sum()} {y.mean()*100:.1f}%) ===")
    rows = []
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    geno_only = msk["X_full"][mask][:, 7:]
    for feat, X in [("clinical",    msk["X_clin"][mask]),
                    ("genomic",     geno_only),
                    ("clin+genomic",msk["X_full"][mask])]:
        for name, cls, kw in [("lr",LogisticRegression,_LR_KW),
                               ("mlp",MLPClassifier,_MLP_KW)]:
            aucs = []
            for tr,te in skf.split(X, y):
                sc = _impute_scale(X[tr],[X[te]])
                aucs.append(_auroc(y[te], _fit_predict(sc[0],y[tr],sc[1],cls,kw)))
            rows.append({"features":feat,"model":name,
                         "auroc_mean":round(np.mean(aucs),4),
                         "auroc_std": round(np.std(aucs), 4)})
            print(f"  {feat:>14}  {name}  "
                  f"AUROC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print()
    return pd.DataFrame(rows)


# ── Exp 3: LOOCO i3lung + MSK, PFS ───────────────────────────────────────────

def run_exp3_looco_msk_pfs(centers: list[dict], msk: dict) -> pd.DataFrame:
    print("=== Exp 3: LOOCO PFS + MSK — A(src) / B(+MSKclin) / C(+MSKgeno) ===")
    hdr = (f"  {'center':>6}  {'model':>4}  {'A':>7}  {'B':>7}  {'C':>7}  "
           f"{'ΔB':>6}  {'ΔC':>6}")
    print(hdr); print("  "+"-"*(len(hdr)-2))
    n_geno = msk["X_full"].shape[1] - msk["X_clin"].shape[1]
    rows = []
    gz = lambda X: np.hstack([X, np.zeros((len(X), n_geno))])

    for held_idx in range(len(centers)):
        held   = centers[held_idx]
        src_cs = [c for i,c in enumerate(centers) if i != held_idx]
        X_src  = np.vstack([c["X"]     for c in src_cs])
        y_src  = np.concatenate([c["y_pfs"] for c in src_cs])
        y_tgt  = held["y_pfs"]
        if len(np.unique(y_tgt)) < 2: continue

        sc_a = _impute_scale(X_src, [held["X"]])
        sc_b = _impute_scale(np.vstack([X_src, msk["X_clin"]]),
                             [held["X"]])
        sc_c = _impute_scale(np.vstack([gz(X_src), msk["X_full"]]),
                             [gz(held["X"])])
        y_ab = np.concatenate([y_src, msk["y_pfs"]])

        sign = lambda d: "▲" if d>0.005 else ("▼" if d<-0.005 else "·")
        for name, cls, kw in [("lr",LogisticRegression,_LR_KW),
                               ("mlp",MLPClassifier,_MLP_KW)]:
            a_A = _auroc(y_tgt, _fit_predict(sc_a[0], y_src, sc_a[1], cls, kw))
            a_B = _auroc(y_tgt, _fit_predict(sc_b[0], y_ab,  sc_b[1], cls, kw))
            a_C = _auroc(y_tgt, _fit_predict(sc_c[0], y_ab,  sc_c[1], cls, kw))
            rows.append({"center":held["name"],"model":name,
                         "auroc_A":round(a_A,4),"auroc_B":round(a_B,4),
                         "auroc_C":round(a_C,4),
                         "delta_B":round(a_B-a_A,4),"delta_C":round(a_C-a_A,4)})
            print(f"  {held['name']:>6}  {name:>4}  {a_A:>7.4f}  {a_B:>7.4f}  "
                  f"{a_C:>7.4f}  {a_B-a_A:>+6.3f}{sign(a_B-a_A)}  "
                  f"{a_C-a_A:>+6.3f}{sign(a_C-a_A)}")
        print()

    df = pd.DataFrame(rows)
    print("  Mean Δ:")
    for name, sub in df.groupby("model"):
        print(f"    {name}: B={sub['delta_B'].mean():+.4f}  C={sub['delta_C'].mean():+.4f}")
    print()
    return df


# ── Exp 4: PFS as proxy for OS ────────────────────────────────────────────────

def run_exp4_pfs_proxy_os(centers: list[dict], msk: dict) -> pd.DataFrame:
    """
    Two-stage transfer:
      Stage 1: train MSK (clin+genomic) → PFS model
      Stage 2: apply to i3lung to get PFS_score
               LOOCO: predict OS using clin alone vs clin + PFS_score
    """
    print("=== Exp 4: PFS proxy for OS — two-stage transfer ===")
    print("  Stage 1: train MSK (clin+genomic) → PFS model")

    # Fit PFS model on full MSK (all patients, clin+genomic)
    sc_msk = _impute_scale(msk["X_full"], [])
    X_msk_s = sc_msk[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pfs_model = LogisticRegression(**_LR_KW)
        pfs_model.fit(X_msk_s, msk["y_pfs"])
    print(f"  MSK PFS model trained on n={len(msk['y_pfs'])} patients")
    print()

    # For i3lung: we need to scale clinical features using MSK scaler
    # but MSK has 14 features, i3lung has 7 — fit a separate scaler on clinical only
    _, meds_msk = impute_median(msk["X_clin"])
    imp_msk = lambda X: np.where(np.isnan(X), meds_msk, X)
    sc_clin  = StandardScaler().fit(imp_msk(msk["X_clin"]))

    # Also fit a PFS model on MSK clinical only (for comparison)
    X_msk_clin_s = sc_clin.transform(imp_msk(msk["X_clin"]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pfs_clin_model = LogisticRegression(**_LR_KW)
        pfs_clin_model.fit(X_msk_clin_s, msk["y_pfs"])

    def _get_pfs_score(X_clin, model, scaler, meds):
        X_imp = np.where(np.isnan(X_clin), meds, X_clin)
        X_s   = scaler.transform(X_imp)
        # For full model: pad with zeros for genomic features
        if model is pfs_model:
            n_geno = msk["X_full"].shape[1] - msk["X_clin"].shape[1]
            X_s = np.hstack([X_s, np.zeros((len(X_s), n_geno))])
        return model.predict_proba(X_s)[:, 1]

    print("  Stage 2: LOOCO OS prediction — baseline vs + PFS_score")
    hdr = (f"  {'center':>6}  {'model':>4}  {'OS_base':>8}  "
           f"{'OS+PFS_c':>9}  {'OS+PFS_g':>9}  {'Δclin':>7}  {'Δgeno':>7}")
    print(hdr); print("  "+"-"*(len(hdr)-2))
    rows = []

    for held_idx in range(len(centers)):
        held   = centers[held_idx]
        src_cs = [c for i,c in enumerate(centers) if i != held_idx]
        X_src  = np.vstack([c["X"]    for c in src_cs])
        y_os_src = np.concatenate([c["y_os"] for c in src_cs])
        y_os_tgt = held["y_os"]
        if len(np.unique(y_os_tgt)) < 2: continue

        # PFS scores from MSK model (genomic) and MSK clin model
        pfs_score_geno_src = _get_pfs_score(X_src,      pfs_model,      sc_clin, meds_msk).reshape(-1,1)
        pfs_score_geno_tgt = _get_pfs_score(held["X"],  pfs_model,      sc_clin, meds_msk).reshape(-1,1)
        pfs_score_clin_src = _get_pfs_score(X_src,      pfs_clin_model, sc_clin, meds_msk).reshape(-1,1)
        pfs_score_clin_tgt = _get_pfs_score(held["X"],  pfs_clin_model, sc_clin, meds_msk).reshape(-1,1)

        X_src_aug_c = np.hstack([X_src,     pfs_score_clin_src])
        X_tgt_aug_c = np.hstack([held["X"], pfs_score_clin_tgt])
        X_src_aug_g = np.hstack([X_src,     pfs_score_geno_src])
        X_tgt_aug_g = np.hstack([held["X"], pfs_score_geno_tgt])

        sc_base = _impute_scale(X_src,       [held["X"]])
        sc_c    = _impute_scale(X_src_aug_c, [X_tgt_aug_c])
        sc_g    = _impute_scale(X_src_aug_g, [X_tgt_aug_g])

        sign = lambda d: "▲" if d>0.005 else ("▼" if d<-0.005 else "·")
        for name, cls, kw in [("lr",LogisticRegression,_LR_KW),
                               ("mlp",MLPClassifier,_MLP_KW)]:
            a_base = _auroc(y_os_tgt, _fit_predict(sc_base[0], y_os_src, sc_base[1], cls, kw))
            a_c    = _auroc(y_os_tgt, _fit_predict(sc_c[0],    y_os_src, sc_c[1],    cls, kw))
            a_g    = _auroc(y_os_tgt, _fit_predict(sc_g[0],    y_os_src, sc_g[1],    cls, kw))
            rows.append({"center":held["name"],"model":name,
                         "auroc_base":round(a_base,4),
                         "auroc_pfs_clin":round(a_c,4),
                         "auroc_pfs_geno":round(a_g,4),
                         "delta_clin":round(a_c-a_base,4),
                         "delta_geno":round(a_g-a_base,4),
                         "delta_geno_vs_clin":round(a_g-a_c,4)})
            print(f"  {held['name']:>6}  {name:>4}  {a_base:>8.4f}  "
                  f"{a_c:>9.4f}  {a_g:>9.4f}  "
                  f"{a_c-a_base:>+7.4f}{sign(a_c-a_base)}  "
                  f"{a_g-a_base:>+7.4f}{sign(a_g-a_base)}")
        print()

    df = pd.DataFrame(rows)
    print("  Mean Δ OS AUROC (vs clinical baseline):")
    for name, sub in df.groupby("model"):
        print(f"    {name}: +PFS_clin={sub['delta_clin'].mean():+.4f}  "
              f"+PFS_geno={sub['delta_geno'].mean():+.4f}  "
              f"[geno-clin]={sub['delta_geno_vs_clin'].mean():+.4f}")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data (all IO lines)...")
    centers = load_i3lung()
    msk     = load_msk()
    print(f"  i3lung: {sum(c['n'] for c in centers)} patients across "
          f"{len(centers)} centres")
    print(f"  MSK:    {msk['n']} patients  "
          f"(genomic subset: {msk['has_geno'].sum()})\n")

    df1 = run_exp1_i3lung_pfs(centers)
    df2 = run_exp2_msk_cv_pfs(msk)
    df3 = run_exp3_looco_msk_pfs(centers, msk)
    df4 = run_exp4_pfs_proxy_os(centers, msk)

    df1.to_csv(OUT_DIR / "exp1_i3lung_pfs.csv",    index=False)
    df2.to_csv(OUT_DIR / "exp2_msk_cv_pfs.csv",    index=False)
    df3.to_csv(OUT_DIR / "exp3_looco_msk_pfs.csv", index=False)
    df4.to_csv(OUT_DIR / "exp4_pfs_proxy_os.csv",  index=False)
    print(f"\nSaved → {OUT_DIR}/")


if __name__ == "__main__":
    run_all()
