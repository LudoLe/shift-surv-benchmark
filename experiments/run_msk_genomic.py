"""
MSK genomic transportability experiment.

Experiment 1: MSK internal CV
    5-fold CV on MSK patients with genomic data (n=247).
    Compare: clinical (7 feat) vs clinical + genomic (7+7 feat).
    Models: LR, MLP.

Experiment 2: LOOCO enrichment — i3lung with MSK genomic
    For each held-out i3lung centre:
        A) source only       : 5 i3lung centres, clinical features
        B) source + MSK_clin : add MSK with clinical features only
        C) source + MSK_geno : add MSK with clinical + genomic features
                               (i3lung train/test: genomic imputed to 0)
        D) source + MSK_proxy: add MSK with clinical + genomic features;
                               for i3lung, predict genomic proxies from
                               clinical features using MSK as training set,
                               then use clin + predicted proxies at test time.
    Compare AUROC@τ=12m across A, B, C, D.

Genomic features used (ALK/ROS1/RET dropped — zero prevalence):
    TMB_log1p, EGFR_driver, ERBB2_driver, BRAF_driver,
    MET_driver, STK11_driver, ARID1A_driver

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_msk_genomic
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data import registry_i3lung_vanguri as reg
from data.preprocess import impute_median

TAU     = 12.0
OUT_DIR = _ROOT / "outputs" / "tables" / "msk_genomic"

_CLINICAL_CSV = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_GENOMIC_CSV  = Path("/Users/ludole/Downloads/Job-8338737215013712236690668870.csv")

GENOMIC_COLS = [
    "TMB_log1p",
    "EGFR_driver", "ERBB2_driver", "BRAF_driver",
    "MET_driver",  "STK11_driver", "ARID1A_driver",
]

_LR_KW  = dict(C=1.0, max_iter=500, solver="lbfgs",
               class_weight="balanced", random_state=0)
_MLP_KW = dict(hidden_layer_sizes=(32, 16), activation="relu",
               solver="lbfgs", alpha=0.01, max_iter=1000, random_state=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _binary_label(t, delta, tau=TAU):
    return ((t <= tau) & (delta == 1)).astype(int)

def _auroc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))

def _fit_predict(X_tr, y_tr, X_te, model_cls, kw):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m = model_cls(**kw)
        m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]

def _impute_scale(X_src, X_targets: list[np.ndarray]):
    _, meds = impute_median(X_src)
    imp = lambda X: np.where(np.isnan(X), meds, X)
    X_src_imp = imp(X_src)
    sc = StandardScaler().fit(X_src_imp)
    return [sc.transform(imp(X)) for X in [X_src] + X_targets]


# ── load MSK with genomics ────────────────────────────────────────────────────

def load_msk_genomic() -> dict:
    """
    Returns a dict with two feature matrices for MSK:
        clin_X  : (n, 7)  clinical features only
        full_X  : (n, 14) clinical + genomic features
        t, delta, n
    For patients without genomics, TMB_log1p is imputed to training median
    and driver flags are set to 0 (wildtype assumed).
    """
    clin = pd.read_csv(_CLINICAL_CSV)
    geno = pd.read_csv(_GENOMIC_CSV)

    geno = geno[["Patient_ID","TMB",
                 "EGFR_driver","ALK_driver","ERBB2_driver","ROS1_driver",
                 "RET_driver","BRAF_driver","MET_driver",
                 "STK11_driver","ARID1A_driver"]].copy()

    merged = clin.merge(geno, left_on="dmp_pt_id", right_on="Patient_ID", how="left")

    # ── clinical features (same as registry_vanguri) ──────────────────────
    sex      = (merged["sex"] == 1).astype(float)
    ecog     = pd.to_numeric(merged["ecog"], errors="coerce")
    nlr      = pd.to_numeric(merged["dnlr"], errors="coerce")
    smk_cur  = (merged["smoking_status"] == 1).astype(float)
    smk_for  = (merged["smoking_status"] == 2).astype(float)
    smk_nev  = (merged["smoking_status"] == 0).astype(float)

    # PDL1 from separate parquet (same as registry_vanguri)
    try:
        pdl1_df  = pd.read_parquet(
            Path("/Users/ludole/Downloads/PDL1_SCORE.parquet"))
        pdl1_df.index.name = "main_index"
        pdl1_df = pdl1_df.reset_index()
        merged2 = merged.merge(pdl1_df, on="main_index", how="left")
        score   = pd.to_numeric(merged2["Sauter PD-L1 Score"], errors="coerce")
        pdl1_cat = pd.Series(np.nan, index=score.index)
        pdl1_cat[score == 0]                      = 0.0
        pdl1_cat[(score > 0) & (score < 50)]      = 1.0
        pdl1_cat[score >= 50]                     = 2.0
    except Exception:
        pdl1_cat = pd.Series(np.nan, index=merged.index)

    clin_X = np.column_stack([
        sex.to_numpy(float), ecog.to_numpy(float),
        nlr.to_numpy(float), smk_cur.to_numpy(float),
        smk_for.to_numpy(float), smk_nev.to_numpy(float),
        pdl1_cat.to_numpy(float),
    ])

    # ── genomic features ─────────────────────────────────────────────────
    tmb     = pd.to_numeric(merged["TMB"], errors="coerce")
    tmb_log = np.log1p(tmb.to_numpy(float))

    driver_cols = ["EGFR_driver","ERBB2_driver","BRAF_driver",
                   "MET_driver","STK11_driver","ARID1A_driver"]
    drivers = merged[driver_cols].to_numpy(float)

    # Impute TMB_log to median for patients without genomics
    tmb_med = float(np.nanmedian(tmb_log))
    tmb_log = np.where(np.isnan(tmb_log), tmb_med, tmb_log)
    # Impute binary drivers to 0 (wildtype assumed)
    drivers = np.where(np.isnan(drivers), 0.0, drivers)

    geno_X = np.column_stack([tmb_log.reshape(-1,1), drivers])

    full_X = np.column_stack([clin_X, geno_X])

    # ── outcome ───────────────────────────────────────────────────────────
    vital  = pd.to_numeric(merged["vital_status"], errors="coerce").to_numpy()
    os_int = pd.to_numeric(merged["os_int"],       errors="coerce").to_numpy()
    pfs    = pd.to_numeric(merged["pfs"],          errors="coerce").to_numpy()

    t_raw = np.where(vital == 1, os_int, pfs)
    delta = np.where(vital == 1, 1, 0).astype(int)

    valid = (~np.isnan(vital)) & (~np.isnan(t_raw))
    t = np.clip(t_raw[valid], 0.01, None)
    delta = delta[valid]
    clin_X = clin_X[valid]
    full_X = full_X[valid]

    # flag which patients had real genomics
    has_geno = (~merged["TMB"].isna()).to_numpy()[valid]

    return {
        "clin_X":   clin_X,
        "full_X":   full_X,
        "t":        t,
        "delta":    delta,
        "n":        int(len(t)),
        "has_geno": has_geno,
        "center_id": 6,
    }


# ── Experiment 1: MSK internal CV ────────────────────────────────────────────

def run_msk_cv(msk: dict, n_splits: int = 5) -> pd.DataFrame:
    print("=== Exp 1: MSK internal CV (genomic subset only, n={}) ===".format(
        msk["has_geno"].sum()))

    mask = msk["has_geno"]
    y    = _binary_label(msk["t"][mask], msk["delta"][mask])
    print(f"  Prevalence y=1: {y.mean():.3f}  n={len(y)}")

    rows = []
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    geno_only = msk["full_X"][mask][:, 7:]   # genomic columns only

    for feat_name, X_full in [("clinical",     msk["clin_X"][mask]),
                               ("genomic_only", geno_only),
                               ("clin+genomic", msk["full_X"][mask])]:
        fold_aucs = {"lr": [], "mlp": []}
        for tr, te in skf.split(X_full, y):
            scaled = _impute_scale(X_full[tr], [X_full[te]])
            X_tr_s, X_te_s = scaled[0], scaled[1]

            for name, cls, kw in [("lr", LogisticRegression, _LR_KW),
                                   ("mlp", MLPClassifier, _MLP_KW)]:
                proba = _fit_predict(X_tr_s, y[tr], X_te_s, cls, kw)
                fold_aucs[name].append(_auroc(y[te], proba))

        for name in ["lr", "mlp"]:
            aucs = fold_aucs[name]
            rows.append({
                "experiment": "msk_cv",
                "features": feat_name,
                "model": name,
                "auroc_mean": np.mean(aucs),
                "auroc_std":  np.std(aucs),
                "n_features": X_full.shape[1],
            })
            print(f"  {feat_name:>14}  {name}  AUROC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    print()
    return pd.DataFrame(rows)


# ── Genomic proxy predictor ───────────────────────────────────────────────────

def _fit_genomic_predictor(X_msk_clin: np.ndarray,
                            X_msk_geno: np.ndarray) -> list:
    """
    For each genomic feature, train a predictor on MSK clinical features.
    Binary drivers → LogisticRegression (predict P(driver=1)).
    TMB_log1p     → Ridge regression.
    Returns list of fitted models in GENOMIC_COLS order:
        [TMB_model, EGFR_model, ERBB2_model, BRAF_model,
         MET_model, STK11_model, ARID1A_model]
    """
    models = []
    for j, name in enumerate(GENOMIC_COLS):
        y_g = X_msk_geno[:, j]
        valid = ~np.isnan(y_g)
        if valid.sum() < 10:
            models.append(None)
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if name == "TMB_log1p":
                m = Ridge(alpha=1.0)
                m.fit(X_msk_clin[valid], y_g[valid])
            else:
                if y_g[valid].sum() < 3:   # too rare to predict
                    models.append(None)
                    continue
                m = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                       random_state=0)
                m.fit(X_msk_clin[valid], y_g[valid].astype(int))
        models.append(m)
    return models


def _predict_genomic_proxies(models: list,
                              X_clin: np.ndarray) -> np.ndarray:
    """
    Apply genomic predictor models to clinical features.
    Returns (n, n_genomic_features) array of soft predictions.
    """
    cols = []
    for j, (m, name) in enumerate(zip(models, GENOMIC_COLS)):
        if m is None:
            cols.append(np.zeros(len(X_clin)))
        elif name == "TMB_log1p":
            cols.append(m.predict(X_clin))
        else:
            cols.append(m.predict_proba(X_clin)[:, 1])
    return np.column_stack(cols)


# ── Experiment 2: LOOCO enrichment with genomics ──────────────────────────────

def run_looco_genomic(i3lung_centers: list[dict], msk: dict) -> pd.DataFrame:
    print("=== Exp 2: LOOCO enrichment — A/B/C/D ===")
    hdr = (f"  {'center':>6}  {'model':>4}  {'A:src':>8}  {'B:+MSKc':>8}  "
           f"{'C:+MSKg':>8}  {'D:proxy':>8}  {'ΔB':>6}  {'ΔC':>6}  {'ΔD':>6}")
    print(hdr)
    print("  " + "-"*(len(hdr)-2))

    D    = len(i3lung_centers)
    rows = []

    for held_idx in range(D):
        held     = i3lung_centers[held_idx]
        src_cs   = [c for i,c in enumerate(i3lung_centers) if i != held_idx]
        hname    = reg.CENTER_NAMES[held["center_id"]]

        y_tgt = _binary_label(held["t"], held["delta"])
        if len(np.unique(y_tgt)) < 2:
            continue

        # ── condition A: source only (i3lung, clinical) ───────────────────
        X_src = np.vstack([c["X"] for c in src_cs])
        t_src = np.concatenate([c["t"]   for c in src_cs])
        d_src = np.concatenate([c["delta"] for c in src_cs])
        y_src = _binary_label(t_src, d_src)

        sc_a = _impute_scale(X_src, [held["X"]])
        X_src_a, X_tgt_a = sc_a[0], sc_a[1]

        # ── condition B: +MSK clinical only ──────────────────────────────
        X_aug_b = np.vstack([X_src, msk["clin_X"]])
        t_aug_b = np.concatenate([t_src, msk["t"]])
        d_aug_b = np.concatenate([d_src, msk["delta"]])
        y_aug_b = _binary_label(t_aug_b, d_aug_b)

        sc_b = _impute_scale(X_aug_b, [held["X"]])
        X_aug_b_s, X_tgt_b_s = sc_b[0], sc_b[1]

        n_geno_feat = msk["full_X"].shape[1] - msk["clin_X"].shape[1]

        # ── condition C: +MSK clinical+genomic, i3lung genomic=0 ─────────
        geno_zeros = lambda X: np.hstack([X, np.zeros((len(X), n_geno_feat))])
        X_aug_c  = np.vstack([geno_zeros(X_src), msk["full_X"]])
        y_aug_c  = _binary_label(np.concatenate([t_src, msk["t"]]),
                                 np.concatenate([d_src, msk["delta"]]))
        sc_c = _impute_scale(X_aug_c, [geno_zeros(held["X"])])
        X_aug_c_s, X_tgt_c_s = sc_c[0], sc_c[1]

        # ── condition D: +MSK clinical+genomic, i3lung genomic=proxies ───
        # Train genomic predictor on MSK patients that have real genomics
        msk_has = msk["has_geno"]
        _, meds_msk = impute_median(msk["clin_X"][msk_has])
        imp_msk = lambda X: np.where(np.isnan(X), meds_msk, X)
        sc_proxy = StandardScaler().fit(imp_msk(msk["clin_X"][msk_has]))
        X_msk_clin_s = sc_proxy.transform(imp_msk(msk["clin_X"][msk_has]))
        X_msk_geno   = msk["full_X"][msk_has, 7:]   # genomic columns

        proxy_models = _fit_genomic_predictor(X_msk_clin_s, X_msk_geno)

        # Generate proxies for i3lung training centres and held-out centre
        def _get_proxies(X_clin):
            _, meds = impute_median(X_clin)
            X_imp = np.where(np.isnan(X_clin), meds, X_clin)
            X_s   = sc_proxy.transform(X_imp)
            return _predict_genomic_proxies(proxy_models, X_s)

        X_src_proxy  = np.hstack([X_src, _get_proxies(X_src)])
        X_tgt_proxy  = np.hstack([held["X"], _get_proxies(held["X"])])
        X_aug_d      = np.vstack([X_src_proxy, msk["full_X"]])
        y_aug_d      = y_aug_c.copy()
        sc_d = _impute_scale(X_aug_d, [X_tgt_proxy])
        X_aug_d_s, X_tgt_d_s = sc_d[0], sc_d[1]

        sign = lambda d: "▲" if d > 0.005 else ("▼" if d < -0.005 else "·")

        for name, cls, kw in [("lr", LogisticRegression, _LR_KW),
                               ("mlp", MLPClassifier, _MLP_KW)]:
            a_A = _auroc(y_tgt, _fit_predict(X_src_a,   y_src,   X_tgt_a,   cls, kw))
            a_B = _auroc(y_tgt, _fit_predict(X_aug_b_s, y_aug_b, X_tgt_b_s, cls, kw))
            a_C = _auroc(y_tgt, _fit_predict(X_aug_c_s, y_aug_c, X_tgt_c_s, cls, kw))
            a_D = _auroc(y_tgt, _fit_predict(X_aug_d_s, y_aug_d, X_tgt_d_s, cls, kw))

            rows.append({
                "center": hname, "model": name, "n_tgt": held["n"],
                "auroc_A": round(a_A, 4), "auroc_B": round(a_B, 4),
                "auroc_C": round(a_C, 4), "auroc_D": round(a_D, 4),
                "delta_B": round(a_B - a_A, 4),
                "delta_C": round(a_C - a_A, 4),
                "delta_D": round(a_D - a_A, 4),
                "delta_D_vs_C": round(a_D - a_C, 4),
            })

            print(f"  {hname:>6}  {name:>4}  {a_A:>8.4f}  {a_B:>8.4f}  "
                  f"{a_C:>8.4f}  {a_D:>8.4f}  "
                  f"{a_B-a_A:>+6.3f}{sign(a_B-a_A)}  "
                  f"{a_C-a_A:>+6.3f}{sign(a_C-a_A)}  "
                  f"{a_D-a_A:>+6.3f}{sign(a_D-a_A)}")
        print()

    df = pd.DataFrame(rows)
    print("  Mean Δ across i3lung centres (vs source-only baseline):")
    for name, sub in df.groupby("model"):
        print(f"    {name}:  B(+MSK_clin)={sub['delta_B'].mean():+.4f}  "
              f"C(+MSK_geno_0)={sub['delta_C'].mean():+.4f}  "
              f"D(+MSK_proxy)={sub['delta_D'].mean():+.4f}  "
              f"[D-C]={sub['delta_D_vs_C'].mean():+.4f}")
    return df


# ── Experiment 3: long responder prediction (PFS > 6m) ───────────────────────

def run_long_responder_cv(n_splits: int = 5) -> pd.DataFrame:
    """
    5-fold CV on all MSK patients (n=367) predicting PFS > 6 months.
    pfs_censor=1 → event occurred; long responder = pfs > 6 OR (pfs ≤ 6 AND pfs_censor=0).
    Compares: clinical / genomic_only / clin+genomic.
    Only patients with genomic data are used (n=246) to allow fair comparison.
    """
    clin = pd.read_csv(_CLINICAL_CSV)
    geno = pd.read_csv(_GENOMIC_CSV)

    geno_sub = geno[["Patient_ID","TMB",
                      "EGFR_driver","ERBB2_driver","BRAF_driver",
                      "MET_driver","STK11_driver","ARID1A_driver"]].copy()
    merged = clin.merge(geno_sub, left_on="dmp_pt_id",
                        right_on="Patient_ID", how="inner")  # only with genomics

    # ── label: long responder = PFS > 6m (or censored after 6m) ─────────
    pfs      = pd.to_numeric(merged["pfs"], errors="coerce").to_numpy()
    pfs_ev   = pd.to_numeric(merged["pfs_censor"], errors="coerce").to_numpy()
    # y=1 if NOT progressed by 6m: either no event by 6m, or censored after 6m
    y = ((pfs > 6) | ((pfs <= 6) & (pfs_ev == 0))).astype(int)

    print(f"=== Exp 3: Long responder (PFS > 6m) — genomic subset n={len(y)} ===")
    print(f"  Long responders: {y.sum()} ({y.mean()*100:.1f}%)")

    # ── clinical features ────────────────────────────────────────────────
    sex     = (merged["sex"] == 1).astype(float).to_numpy()
    ecog    = pd.to_numeric(merged["ecog"], errors="coerce").to_numpy()
    nlr     = pd.to_numeric(merged["dnlr"], errors="coerce").to_numpy()
    smk_cur = (merged["smoking_status"] == 1).astype(float).to_numpy()
    smk_for = (merged["smoking_status"] == 2).astype(float).to_numpy()
    smk_nev = (merged["smoking_status"] == 0).astype(float).to_numpy()

    try:
        pdl1_df  = pd.read_parquet(Path("/Users/ludole/Downloads/PDL1_SCORE.parquet"))
        pdl1_df.index.name = "main_index"
        pdl1_df  = pdl1_df.reset_index()
        merged2  = merged.merge(pdl1_df, on="main_index", how="left")
        score    = pd.to_numeric(merged2["Sauter PD-L1 Score"], errors="coerce")
        pdl1_cat = np.where(score == 0, 0.0,
                   np.where((score > 0) & (score < 50), 1.0,
                   np.where(score >= 50, 2.0, np.nan)))
    except Exception:
        pdl1_cat = np.full(len(merged), np.nan)

    X_clin = np.column_stack([sex, ecog, nlr, smk_cur, smk_for, smk_nev, pdl1_cat])

    # ── genomic features ─────────────────────────────────────────────────
    tmb_log  = np.log1p(pd.to_numeric(merged["TMB"], errors="coerce").to_numpy())
    drivers  = merged[["EGFR_driver","ERBB2_driver","BRAF_driver",
                        "MET_driver","STK11_driver","ARID1A_driver"]].to_numpy(float)
    X_geno   = np.column_stack([tmb_log.reshape(-1,1), drivers])
    X_full   = np.column_stack([X_clin, X_geno])

    # ── CV ───────────────────────────────────────────────────────────────
    rows = []
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    for feat_name, X in [("clinical",     X_clin),
                          ("genomic_only", X_geno),
                          ("clin+genomic", X_full)]:
        fold_aucs = {"lr": [], "mlp": []}
        for tr, te in skf.split(X, y):
            scaled = _impute_scale(X[tr], [X[te]])
            X_tr_s, X_te_s = scaled[0], scaled[1]
            for name, cls, kw in [("lr",  LogisticRegression, _LR_KW),
                                   ("mlp", MLPClassifier,      _MLP_KW)]:
                proba = _fit_predict(X_tr_s, y[tr], X_te_s, cls, kw)
                fold_aucs[name].append(_auroc(y[te], proba))

        for name in ["lr", "mlp"]:
            aucs = fold_aucs[name]
            rows.append({
                "experiment": "long_responder_cv",
                "features":   feat_name,
                "model":      name,
                "auroc_mean": round(np.mean(aucs), 4),
                "auroc_std":  round(np.std(aucs),  4),
                "n_features": X.shape[1],
            })
            print(f"  {feat_name:>14}  {name}  "
                  f"AUROC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print()
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    centers, _ = reg.load_all_centers()
    i3lung_cs  = [c for c in centers if c["center_id"] != 6]
    msk_std    = next(c for c in centers if c["center_id"] == 6)

    msk = load_msk_genomic()
    # Sanity check: msk["clin_X"] should match msk_std["X"] in shape
    print(f"  MSK standard registry:  n={msk_std['n']}")
    print(f"  MSK genomic loader:     n={msk['n']}, with_geno={msk['has_geno'].sum()}")
    print()

    df1 = run_msk_cv(msk)
    df2 = run_looco_genomic(i3lung_cs, msk)
    df3 = run_long_responder_cv()

    df1.to_csv(OUT_DIR / "msk_cv_genomic.csv",        index=False)
    df2.to_csv(OUT_DIR / "msk_looco_genomic.csv",     index=False)
    df3.to_csv(OUT_DIR / "msk_long_responder_cv.csv", index=False)
    print(f"\nSaved → {OUT_DIR}/")


if __name__ == "__main__":
    run_all()
