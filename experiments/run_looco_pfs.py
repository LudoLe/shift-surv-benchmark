"""
LOOCO experiment: long responder prediction (PFS > 6m).

Outcome shared across i3lung + MSK:
    y = 1  if PFS > 6m  (or censored after 6m without progression event)
    y = 0  if progressed / died within 6m

Feature set: 7 common clinical features
    SEX, ECOG PS, NLR, SMOKING CURRENT, SMOKING FORMER, SMOKING NEVER, PDL1 CATEGORY

Conditions (for each held-out i3lung centre):
    A  source only        : 5 i3lung centres, clinical
    B  source + MSK_clin  : add MSK with clinical features
    C  source + MSK_geno  : add MSK with clinical + genomic (i3lung → genomic=0)

Also reports MSK internal 5-fold CV (clinical vs clin+genomic) with PFS outcome.

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_looco_pfs
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

TAU     = 6.0
OUT_DIR = _ROOT / "outputs" / "tables" / "msk_genomic"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")
_CLINICAL_CSV  = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_GENOMIC_CSV   = Path("/Users/ludole/Downloads/Job-8338737215013712236690668870.csv")
_PDL1_PARQUET  = Path("/Users/ludole/Downloads/PDL1_SCORE.parquet")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]

GENOMIC_COLS = ["TMB_log1p", "EGFR_driver", "ERBB2_driver", "BRAF_driver",
                "MET_driver", "STK11_driver", "ARID1A_driver"]

CENTER_NAMES = {0:"GHD", 1:"INT", 2:"MH", 3:"SZMC", 4:"UOC", 5:"VHIO", 6:"MSK"}

_LR_KW  = dict(C=1.0, max_iter=500, solver="lbfgs",
               class_weight="balanced", random_state=0)
_MLP_KW = dict(hidden_layer_sizes=(32,16), activation="relu",
               solver="lbfgs", alpha=0.01, max_iter=1000, random_state=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _long_responder(pfs, pfs_event, tau=TAU):
    """y=1 if PFS > tau OR censored after tau (no event by tau)."""
    return ((pfs > tau) | ((pfs <= tau) & (pfs_event == 0))).astype(int)

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


# ── load i3lung centers with PFS outcome ─────────────────────────────────────

def load_i3lung_pfs(io_line: int | None = 1) -> list[dict]:
    rwd  = pd.read_csv(_RWD_PATH)
    out  = pd.read_csv(_OUTCOMES_PATH)

    df = rwd.merge(
        out[["Subject","PFS MONTHS","PROGRESSION EVENT"]],
        on="Subject", how="inner"
    ).dropna(subset=["PFS MONTHS","PROGRESSION EVENT"])

    if io_line is not None:
        before = len(df)
        df = df[df["IO LINE"] == io_line]
        print(f"  IO_LINE={io_line} filter: {len(df)}/{before} patients kept")

    # PDL1 CATEGORY: use from rwd if available
    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        X = grp[COMMON_FEATS].to_numpy(float)
        pfs   = grp["PFS MONTHS"].to_numpy(float)
        pfs_e = grp["PROGRESSION EVENT"].to_numpy(float)
        y     = _long_responder(pfs, pfs_e)
        centers.append({
            "X": X, "y": y, "pfs": pfs, "pfs_event": pfs_e,
            "n": len(y), "center_id": cid, "name": name,
        })
        print(f"  {name}: n={len(y)}  long_resp={y.sum()} ({y.mean()*100:.1f}%)")
    return centers


# ── load MSK with PFS outcome ─────────────────────────────────────────────────

def load_msk_pfs(io_line: int | None = 1) -> dict:
    clin = pd.read_csv(_CLINICAL_CSV)
    geno = pd.read_csv(_GENOMIC_CSV)

    if io_line is not None:
        before = len(clin)
        clin = clin[clin["io_line"] == io_line].copy()
        print(f"  IO_LINE={io_line} filter: {len(clin)}/{before} MSK patients kept")

    geno_sub = geno[["Patient_ID","TMB","EGFR_driver","ERBB2_driver",
                      "BRAF_driver","MET_driver","STK11_driver","ARID1A_driver"]]
    merged = clin.merge(geno_sub, left_on="dmp_pt_id",
                        right_on="Patient_ID", how="left")

    # clinical features
    sex     = (merged["sex"] == 1).astype(float)
    ecog    = pd.to_numeric(merged["ecog"], errors="coerce")
    nlr     = pd.to_numeric(merged["dnlr"], errors="coerce")
    smk_cur = (merged["smoking_status"] == 1).astype(float)
    smk_for = (merged["smoking_status"] == 2).astype(float)
    smk_nev = (merged["smoking_status"] == 0).astype(float)
    try:
        pdl1_df  = pd.read_parquet(_PDL1_PARQUET)
        pdl1_df.index.name = "main_index"
        pdl1_df  = pdl1_df.reset_index()
        m2       = merged.merge(pdl1_df, on="main_index", how="left")
        score    = pd.to_numeric(m2["Sauter PD-L1 Score"], errors="coerce")
        pdl1_cat = np.where(score==0, 0., np.where((score>0)&(score<50), 1.,
                   np.where(score>=50, 2., np.nan)))
    except Exception:
        pdl1_cat = np.full(len(merged), np.nan)

    X_clin = np.column_stack([sex, ecog, nlr, smk_cur, smk_for, smk_nev, pdl1_cat])

    # genomic features
    tmb_log = np.log1p(pd.to_numeric(merged["TMB"], errors="coerce").to_numpy(float))
    drivers = merged[["EGFR_driver","ERBB2_driver","BRAF_driver",
                       "MET_driver","STK11_driver","ARID1A_driver"]].to_numpy(float)
    tmb_med = float(np.nanmedian(tmb_log))
    tmb_log = np.where(np.isnan(tmb_log), tmb_med, tmb_log)
    drivers = np.where(np.isnan(drivers), 0., drivers)
    X_geno  = np.column_stack([tmb_log.reshape(-1,1), drivers])
    X_full  = np.column_stack([X_clin, X_geno])

    # PFS outcome — pfs_censor=1 means event occurred
    pfs   = pd.to_numeric(merged["pfs"],        errors="coerce").to_numpy(float)
    pfs_e = pd.to_numeric(merged["pfs_censor"], errors="coerce").to_numpy(float)

    valid = ~np.isnan(pfs) & ~np.isnan(pfs_e)
    pfs, pfs_e = pfs[valid], pfs_e[valid]
    X_clin, X_full = X_clin[valid], X_full[valid]
    has_geno = (~merged["TMB"].isna()).to_numpy()[valid]

    y = _long_responder(pfs, pfs_e)
    print(f"  MSK: n={len(y)}  long_resp={y.sum()} ({y.mean()*100:.1f}%)")

    return {"X_clin": X_clin, "X_full": X_full, "y": y,
            "pfs": pfs, "pfs_event": pfs_e,
            "has_geno": has_geno, "n": len(y), "name": "MSK"}


# ── Experiment 1: MSK internal CV (PFS outcome) ───────────────────────────────

def run_msk_cv_pfs(msk: dict, n_splits: int = 5) -> pd.DataFrame:
    print("\n=== MSK internal CV — PFS > 6m (genomic subset) ===")
    mask = msk["has_geno"]
    y    = msk["y"][mask]
    print(f"  n={mask.sum()}  long_resp={y.sum()} ({y.mean()*100:.1f}%)")

    rows = []
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    geno_only = msk["X_full"][mask][:, 7:]
    for feat_name, X in [("clinical",     msk["X_clin"][mask]),
                          ("genomic_only", geno_only),
                          ("clin+genomic", msk["X_full"][mask])]:
        fold_aucs = {"lr": [], "mlp": []}
        for tr, te in skf.split(X, y):
            sc = _impute_scale(X[tr], [X[te]])
            for name, cls, kw in [("lr", LogisticRegression, _LR_KW),
                                   ("mlp", MLPClassifier, _MLP_KW)]:
                fold_aucs[name].append(
                    _auroc(y[te], _fit_predict(sc[0], y[tr], sc[1], cls, kw)))
        for name in ["lr","mlp"]:
            aucs = fold_aucs[name]
            rows.append({"features": feat_name, "model": name,
                         "auroc_mean": round(np.mean(aucs),4),
                         "auroc_std":  round(np.std(aucs), 4)})
            print(f"  {feat_name:>14}  {name}  "
                  f"AUROC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return pd.DataFrame(rows)


# ── Experiment 2: LOOCO i3lung — conditions A / B / C ────────────────────────

def run_looco_pfs(i3lung: list[dict], msk: dict) -> pd.DataFrame:
    print("\n=== LOOCO PFS > 6m — A(src) / B(+MSK_clin) / C(+MSK_geno) ===")
    hdr = (f"  {'center':>6}  {'model':>4}  {'A:src':>8}  "
           f"{'B:+MSKc':>8}  {'C:+MSKg':>8}  {'ΔB':>6}  {'ΔC':>6}")
    print(hdr); print("  " + "-"*(len(hdr)-2))

    D = len(i3lung); rows = []
    n_geno = msk["X_full"].shape[1] - msk["X_clin"].shape[1]

    for held_idx in range(D):
        held   = i3lung[held_idx]
        src_cs = [c for i,c in enumerate(i3lung) if i != held_idx]
        y_tgt  = held["y"]
        if len(np.unique(y_tgt)) < 2: continue

        X_src = np.vstack([c["X"] for c in src_cs])
        y_src = np.concatenate([c["y"] for c in src_cs])

        # A: source only
        sc_a = _impute_scale(X_src, [held["X"]])
        X_src_a, X_tgt_a = sc_a[0], sc_a[1]

        # B: + MSK clinical
        X_aug_b = np.vstack([X_src, msk["X_clin"]])
        y_aug_b = np.concatenate([y_src, msk["y"]])
        sc_b = _impute_scale(X_aug_b, [held["X"]])
        X_aug_b_s, X_tgt_b_s = sc_b[0], sc_b[1]

        # C: + MSK clinical+genomic (i3lung genomic=0)
        gz = lambda X: np.hstack([X, np.zeros((len(X), n_geno))])
        X_aug_c = np.vstack([gz(X_src), msk["X_full"]])
        y_aug_c = np.concatenate([y_src, msk["y"]])
        sc_c = _impute_scale(X_aug_c, [gz(held["X"])])
        X_aug_c_s, X_tgt_c_s = sc_c[0], sc_c[1]

        sign = lambda d: "▲" if d>0.005 else ("▼" if d<-0.005 else "·")

        for name, cls, kw in [("lr",  LogisticRegression, _LR_KW),
                               ("mlp", MLPClassifier,      _MLP_KW)]:
            a_A = _auroc(y_tgt, _fit_predict(X_src_a,   y_src,   X_tgt_a,   cls, kw))
            a_B = _auroc(y_tgt, _fit_predict(X_aug_b_s, y_aug_b, X_tgt_b_s, cls, kw))
            a_C = _auroc(y_tgt, _fit_predict(X_aug_c_s, y_aug_c, X_tgt_c_s, cls, kw))

            rows.append({"center": held["name"], "model": name,
                         "n_tgt": held["n"], "prev_tgt": round(y_tgt.mean(),3),
                         "auroc_A": round(a_A,4), "auroc_B": round(a_B,4),
                         "auroc_C": round(a_C,4),
                         "delta_B": round(a_B-a_A,4),
                         "delta_C": round(a_C-a_A,4),
                         "delta_C_vs_B": round(a_C-a_B,4)})

            print(f"  {held['name']:>6}  {name:>4}  {a_A:>8.4f}  "
                  f"{a_B:>8.4f}  {a_C:>8.4f}  "
                  f"{a_B-a_A:>+6.3f}{sign(a_B-a_A)}  "
                  f"{a_C-a_A:>+6.3f}{sign(a_C-a_A)}")
        print()

    df = pd.DataFrame(rows)
    print("  Mean Δ across i3lung centres:")
    for name, sub in df.groupby("model"):
        print(f"    {name}:  B(+MSK_clin)={sub['delta_B'].mean():+.4f}  "
              f"C(+MSK_geno)={sub['delta_C'].mean():+.4f}  "
              f"[C-B]={sub['delta_C_vs_B'].mean():+.4f}")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung centers (PFS outcome, IO_LINE=1)...")
    i3lung = load_i3lung_pfs(io_line=1)

    print("Loading MSK (PFS outcome, all IO lines)...")
    msk = load_msk_pfs(io_line=None)

    df1 = run_msk_cv_pfs(msk)
    df2 = run_looco_pfs(i3lung, msk)

    df1.to_csv(OUT_DIR / "msk_cv_pfs.csv",    index=False)
    df2.to_csv(OUT_DIR / "looco_pfs.csv",      index=False)
    print(f"\nSaved → {OUT_DIR}/")


if __name__ == "__main__":
    run_all()
