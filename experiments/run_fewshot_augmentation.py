"""
Few-shot transfer learning with synthetic augmentation (LOOCO, OS@12m).

Goal
----
Test whether the Weibull PH generator improves predictions on a new centre
that has only m << n labeled patients, by augmenting the few-shot training
set with synthetic patients generated from the fitted generator.

The generator acts as a PRIOR informed by all source centres: it captures
the population-level survival geometry (A_shared, w_shared) and adapts the
domain-specific parameters (μ_d, λ₀_d) from the m calibration patients.

Design
------
For each held-out centre (LOOCO):
  1. Randomly draw m patients as calibration set; rest = evaluation set.
  2. Stage 1: fit (A, w) on 5 source centres.
  3. Stage 2: fit DomainParams on m calibration patients.
  4. Generate n_syn synthetic patients.
  5. Train downstream Cox in latent space on:
       (A) m real only                 — few-shot baseline
       (B) m real + n_syn synthetic    — generator-augmented
  6. Evaluate AUROC@12m on evaluation set.

Averaged over N_SEEDS random cal/eval splits per centre per condition.

Conditions
----------
  cond_OS    : OS-only A  (k=6), pooled standardisation   ← current pipeline
  cond_PC    : OS-only A  (k=6), per-centre std            ← NLR scale fix
  cond_MT    : multitask A (k=3, PFS aux λ=0.3), pooled   ← genomic prior
               (requires MSK clinical + PFS data)

m ∈ {10, 20, 50}   n_syn = 300   N_SEEDS = 5

Per-centre standardisation (cond_PC)
-------------------------------------
Each source centre is z-scored within itself before pooling.  This removes
inter-centre NLR scale heterogeneity (confirmed: MH/SZMC have NLR std ≈1.4
vs INT/VHIO ≈0.65 on the pooled scale — see run_feature_geometry_diagnostic).
The target centre is also z-scored within itself using all n patients (feature
data, not labels — acceptable in deployment).

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_fewshot_augmentation
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
    fit_reference_generator,
    fit_domain_params_fewshot,
    fit_loading_matrix,
    fit_risk_weights_cox,
    fit_multitask_loading_matrix,
    fit_lambda0,
    _km_at_tau,
    compute_genomic_risk_scores,
)
from generator.base_generator import SyntheticGenerator
from generator.parameters import BaseParams, DomainParams

TAU_OS   = 12.0
NU       = 1.5
K_OS     = 6          # latent dims for OS-only conditions
K_MT     = 3          # latent dims for multitask (bottleneck regime)
LAM_MT   = 0.3        # PFS auxiliary weight
LAM_GENO = 0.3        # genomic distillation weight
N_SYN    = 300        # synthetic patients per fold
N_SEEDS  = 5          # random cal/eval splits per fold
M_VALS   = [10, 20, 50]
OUT_DIR  = _ROOT / "outputs" / "tables" / "fewshot_aug"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")
_CLINICAL_CSV  = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_GENOMIC_CSV   = Path("/Users/ludole/Downloads/Job-8338737215013712236690668870.csv")
_PDL1_PARQUET  = Path("/Users/ludole/Downloads/PDL1_SCORE.parquet")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]
P = len(COMMON_FEATS)


# ── helpers ───────────────────────────────────────────────────────────────────

def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def _imp(X, meds):
    return np.where(np.isnan(X), meds, X)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_i3lung() -> list[dict]:
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


def load_msk_pfs() -> dict | None:
    """MSK clinical features + PFS.  Returns None if files missing."""
    if not _CLINICAL_CSV.exists():
        return None
    try:
        import pandas as pd
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
            m2    = clin.merge(pdl1_df, on="main_index", how="left")
            score = pd.to_numeric(m2["Sauter PD-L1 Score"], errors="coerce")
            pdl1  = np.where(score==0, 0.,
                    np.where((score>0)&(score<50), 1.,
                    np.where(score>=50, 2., np.nan)))
        except Exception:
            pdl1 = np.full(len(clin), np.nan)

        X_clin = np.column_stack([sex, ecog, nlr, smk_cur, smk_for, smk_nev, pdl1])
        pfs    = pd.to_numeric(clin["pfs"],        errors="coerce").to_numpy(float)
        pfs_e  = pd.to_numeric(clin["pfs_censor"], errors="coerce").to_numpy(float)
        valid  = ~np.isnan(pfs) & ~np.isnan(pfs_e)
        print(f"  MSK PFS: n={valid.sum()}")
        return {"X_clin": X_clin[valid], "t_pfs": pfs[valid], "delta_pfs": pfs_e[valid]}
    except Exception as e:
        print(f"  MSK load failed: {e}")
        return None


def load_msk_genomic_pfs() -> dict | None:
    """
    MSK clinical + genomic features + PFS for patients with BOTH data sources.

    Joins clinical (main_index) ↔ genomic (Patient_ID) — 247 patients in common.
    Returns None if either file is missing.

    Genomic features: TMB (continuous) + 9 binary driver flags
      → X_geno shape (N, 10)
    Clinical features (same 7 as COMMON_FEATS):
      → X_clin shape (N, 7)
    """
    if not _CLINICAL_CSV.exists() or not _GENOMIC_CSV.exists():
        return None
    try:
        clin = pd.read_csv(_CLINICAL_CSV)
        geno = pd.read_csv(_GENOMIC_CSV)

        merged = clin.merge(geno, left_on="main_index", right_on="Patient_ID", how="inner")

        sex     = (merged["sex"] == 1).astype(float)
        ecog    = pd.to_numeric(merged["ecog"], errors="coerce")
        nlr     = pd.to_numeric(merged["dnlr"], errors="coerce")
        smk_cur = (merged["smoking_status"] == 1).astype(float)
        smk_for = (merged["smoking_status"] == 2).astype(float)
        smk_nev = (merged["smoking_status"] == 0).astype(float)
        try:
            pdl1_df = pd.read_parquet(_PDL1_PARQUET)
            pdl1_df.index.name = "main_index"
            pdl1_df = pdl1_df.reset_index()
            m2    = merged.merge(pdl1_df, on="main_index", how="left")
            score = pd.to_numeric(m2["Sauter PD-L1 Score"], errors="coerce")
            pdl1  = np.where(score==0, 0.,
                    np.where((score>0)&(score<50), 1.,
                    np.where(score>=50, 2., np.nan)))
        except Exception:
            pdl1 = np.full(len(merged), np.nan)

        X_clin = np.column_stack([sex, ecog, nlr, smk_cur, smk_for, smk_nev, pdl1])

        # Genomic: TMB + 9 driver mutation flags
        driver_cols = [c for c in geno.columns if c.endswith("_driver")]
        tmb  = pd.to_numeric(merged["TMB"], errors="coerce").to_numpy(float)
        driv = merged[driver_cols].astype(float).to_numpy()
        X_geno = np.column_stack([tmb[:, None], driv])  # (N, 10)

        pfs   = pd.to_numeric(merged["pfs"],        errors="coerce").to_numpy(float)
        pfs_e = pd.to_numeric(merged["pfs_censor"], errors="coerce").to_numpy(float)

        valid = ~np.isnan(pfs) & ~np.isnan(pfs_e) & ~np.isnan(tmb)
        print(f"  MSK genomic+PFS: n={valid.sum()} "
              f"(joined from {len(merged)} clinical+genomic patients)")
        return {
            "X_clin":    X_clin[valid],
            "X_geno":    X_geno[valid],
            "t_pfs":     pfs[valid],
            "delta_pfs": pfs_e[valid],
        }
    except Exception as e:
        print(f"  MSK genomic load failed: {e}")
        return None


# ── standardisation helpers ───────────────────────────────────────────────────

def _scale_pooled(src_imp_list, tgt_imp):
    """Pooled standardisation (current pipeline)."""
    sc = StandardScaler().fit(np.vstack(src_imp_list))
    return [sc.transform(X) for X in src_imp_list], sc.transform(tgt_imp), sc


def _scale_percenter(src_imp_list, tgt_imp):
    """
    Per-centre standardisation.
    Each source centre and the target centre are z-scored within themselves,
    then a light global centering is applied to the pooled result.
    This removes inter-centre NLR scale heterogeneity before fitting A.
    """
    pc_src = [StandardScaler().fit_transform(X) for X in src_imp_list]
    pc_tgt = StandardScaler().fit_transform(tgt_imp)
    all_data = np.vstack(pc_src + [pc_tgt])
    sc_global = StandardScaler()
    all_s = sc_global.fit_transform(all_data)
    sizes = [len(X) for X in pc_src] + [len(pc_tgt)]
    cuts  = np.cumsum(sizes)[:-1]
    chunks = np.split(all_s, cuts)
    return chunks[:-1], chunks[-1], sc_global


# ── synthetic generation ──────────────────────────────────────────────────────

def _generate_synthetic(dp: DomainParams, k: int, n_syn: int, seed: int) -> tuple:
    """
    Generate n_syn synthetic patients from DomainParams.
    Returns (u_syn, t_syn, d_syn) — latent factors + survival times.
    """
    base = BaseParams(k_inv=k, k_spu=0, p=P, tau=TAU_OS, seed=seed, nl_strength=0.0)
    gen  = SyntheticGenerator(base)
    dp_gen = DomainParams(
        mu_inv=dp.mu_inv,
        mu_spu=np.zeros(0),
        A_inv=dp.A_inv,
        A_spu=None,
        sigma_noise=dp.sigma_noise,
        beta0=dp.beta0,
        w=dp.w,
        v=np.zeros(0),
        lambda0=dp.lambda0,
        nu=dp.nu,
        n=n_syn,
        domain_id=dp.domain_id,
        censoring_lambda=0.0,
    )
    ds = gen.generate_domain(dp_gen, domain_id=dp.domain_id, seed=seed)
    return ds.u_inv, ds.t_obs, ds.delta


# ── single-fold evaluation ────────────────────────────────────────────────────

def _downstream_cox_auroc(
    U_train: np.ndarray, t_train: np.ndarray, d_train: np.ndarray,
    U_eval:  np.ndarray, y_eval: np.ndarray,
) -> float:
    """Train latent Cox on train set, evaluate AUROC on eval set."""
    if int(d_train.sum()) < 3 or len(np.unique(y_eval)) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        w_ds = fit_risk_weights_cox(U_train, t_train, d_train)
    mu = U_train.mean(0)
    score = (U_eval - mu) @ w_ds
    return _auroc(y_eval, score)


def run_fold(
    held:          dict,
    src_cs:        list[dict],
    meds_src:      np.ndarray,
    msk_pfs:       dict | None,
    m:             int,
    seed:          int,
    msk_geno_pfs:  dict | None = None,
) -> dict:
    """
    Run one LOOCO fold with cal/eval split seeded by `seed`.
    Returns dict with AUROC for each condition × (no-aug / aug).
    """
    n_tgt = held["n"]
    if n_tgt <= m + 5:
        return {}

    # ── cal/eval split ────────────────────────────────────────────────────
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_tgt)
    cal_idx  = perm[:m]
    eval_idx = perm[m:]

    X_tgt      = held["X"]
    t_tgt      = held["t"]
    d_tgt      = held["delta"]
    y_tgt      = held["y_os"]

    X_tgt_imp  = _imp(X_tgt, meds_src)
    X_cal_imp  = X_tgt_imp[cal_idx]
    t_cal      = t_tgt[cal_idx];   d_cal  = d_tgt[cal_idx]
    y_eval     = y_tgt[eval_idx]

    if int(d_cal.sum()) < 2:   # too few events in cal set
        return {}

    # ── source imputed features ───────────────────────────────────────────
    X_src_imp_list = [_imp(c["X"], meds_src) for c in src_cs]
    t_src_all = np.concatenate([c["t"]     for c in src_cs])
    d_src_all = np.concatenate([c["delta"] for c in src_cs])
    src_imp_dicts = [{**c, "X": _imp(c["X"], meds_src)} for c in src_cs]

    result = {}

    # ══ CONDITION: OS-only (pooled std) ══════════════════════════════════
    src_s_list, tgt_s, sc_p = _scale_pooled(X_src_imp_list, X_tgt_imp)
    X_cal_s_p   = tgt_s[cal_idx]
    X_eval_s_p  = tgt_s[eval_idx]
    X_src_s_p   = np.vstack(src_s_list)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        src_p_dicts = [{**c, "X": s} for c, s in zip(src_cs, src_s_list)]
        A_os, w_os, dp_list_os, _ = fit_reference_generator(
            src_p_dicts, k_total=K_OS, tau=TAU_OS,
            time_distribution="cox",
        )

    lam0_src_mean = float(np.mean([dp.lambda0 for dp in dp_list_os]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        dp_os = fit_domain_params_fewshot(
            X_cal_s_p, A_os, w_os, TAU_OS, NU,
            domain_id=held["center_id"], n_synthetic=N_SYN,
            source_lambda0_mean=lam0_src_mean,
            t_m=t_cal, delta_m=d_cal, variant="B",
        )

    U_cal_os  = X_cal_s_p  @ A_os
    U_eval_os = X_eval_s_p @ A_os

    # No-aug
    result["OS_noaug"] = _downstream_cox_auroc(
        U_cal_os, t_cal, d_cal, U_eval_os, y_eval)

    # Augmented
    u_syn_os, t_syn_os, d_syn_os = _generate_synthetic(dp_os, K_OS, N_SYN, seed=seed+100)
    U_aug_os = np.vstack([U_cal_os, u_syn_os])
    t_aug_os = np.concatenate([t_cal, t_syn_os])
    d_aug_os = np.concatenate([d_cal, d_syn_os])
    result["OS_aug"] = _downstream_cox_auroc(
        U_aug_os, t_aug_os, d_aug_os, U_eval_os, y_eval)

    # ══ CONDITION: OS-only (per-centre std) ══════════════════════════════
    src_s_list_pc, tgt_s_pc, _ = _scale_percenter(X_src_imp_list, X_tgt_imp)
    X_cal_s_pc   = tgt_s_pc[cal_idx]
    X_eval_s_pc  = tgt_s_pc[eval_idx]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        src_pc_dicts = [{**c, "X": s} for c, s in zip(src_cs, src_s_list_pc)]
        A_pc, w_pc, dp_list_pc, _ = fit_reference_generator(
            src_pc_dicts, k_total=K_OS, tau=TAU_OS,
            time_distribution="cox",
        )

    lam0_pc_mean = float(np.mean([dp.lambda0 for dp in dp_list_pc]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        dp_pc = fit_domain_params_fewshot(
            X_cal_s_pc, A_pc, w_pc, TAU_OS, NU,
            domain_id=held["center_id"], n_synthetic=N_SYN,
            source_lambda0_mean=lam0_pc_mean,
            t_m=t_cal, delta_m=d_cal, variant="B",
        )

    U_cal_pc  = X_cal_s_pc  @ A_pc
    U_eval_pc = X_eval_s_pc @ A_pc

    result["PC_noaug"] = _downstream_cox_auroc(
        U_cal_pc, t_cal, d_cal, U_eval_pc, y_eval)

    u_syn_pc, t_syn_pc, d_syn_pc = _generate_synthetic(dp_pc, K_OS, N_SYN, seed=seed+200)
    U_aug_pc = np.vstack([U_cal_pc, u_syn_pc])
    t_aug_pc = np.concatenate([t_cal, t_syn_pc])
    d_aug_pc = np.concatenate([d_cal, d_syn_pc])
    result["PC_aug"] = _downstream_cox_auroc(
        U_aug_pc, t_aug_pc, d_aug_pc, U_eval_pc, y_eval)

    # ══ CONDITION: multitask (k=3, PFS aux) — only if MSK available ══════
    if msk_pfs is not None:
        ext = {
            "X_clin":    _imp(msk_pfs["X_clin"], meds_src),
            "t_pfs":     msk_pfs["t_pfs"],
            "delta_pfs": msk_pfs["delta_pfs"],
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A_mt, w_mt, dp_list_mt, sc_mt = fit_reference_generator(
                src_p_dicts, k_total=K_MT, tau=TAU_OS,
                time_distribution="cox",
                external_pfs=ext, lam_pfs=LAM_MT,
            )

        lam0_mt_mean = float(np.mean([dp.lambda0 for dp in dp_list_mt]))
        X_cal_s_mt  = sc_mt.transform(X_cal_imp)
        X_eval_s_mt = sc_mt.transform(X_tgt_imp[eval_idx])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dp_mt = fit_domain_params_fewshot(
                X_cal_s_mt, A_mt, w_mt, TAU_OS, NU,
                domain_id=held["center_id"], n_synthetic=N_SYN,
                source_lambda0_mean=lam0_mt_mean,
                t_m=t_cal, delta_m=d_cal, variant="B",
            )

        U_cal_mt  = X_cal_s_mt  @ A_mt
        U_eval_mt = X_eval_s_mt @ A_mt

        result["MT_noaug"] = _downstream_cox_auroc(
            U_cal_mt, t_cal, d_cal, U_eval_mt, y_eval)

        u_syn_mt, t_syn_mt, d_syn_mt = _generate_synthetic(dp_mt, K_MT, N_SYN, seed=seed+300)
        U_aug_mt = np.vstack([U_cal_mt, u_syn_mt])
        t_aug_mt = np.concatenate([t_cal, t_syn_mt])
        d_aug_mt = np.concatenate([d_cal, d_syn_mt])
        result["MT_aug"] = _downstream_cox_auroc(
            U_aug_mt, t_aug_mt, d_aug_mt, U_eval_mt, y_eval)

    # ══ CONDITION: MT_geno (k=3, PFS + genomic distillation) ═════════════
    if msk_geno_pfs is not None:
        ext_g = {
            "X_clin":    _imp(msk_geno_pfs["X_clin"], meds_src),
            "t_pfs":     msk_geno_pfs["t_pfs"],
            "delta_pfs": msk_geno_pfs["delta_pfs"],
        }
        X_geno_raw = msk_geno_pfs["X_geno"]
        geno_meds  = np.nanmedian(X_geno_raw, axis=0)
        X_geno_imp = _imp(X_geno_raw, geno_meds)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            A_mg, w_mg, dp_list_mg, sc_mg = fit_reference_generator(
                src_p_dicts, k_total=K_MT, tau=TAU_OS,
                time_distribution="cox",
                external_pfs=ext_g, lam_pfs=LAM_MT,
                genomic_distill={"X_geno": X_geno_imp, "lam_geno": LAM_GENO},
            )

        lam0_mg_mean = float(np.mean([dp.lambda0 for dp in dp_list_mg]))
        X_cal_s_mg   = sc_mg.transform(X_cal_imp)
        X_eval_s_mg  = sc_mg.transform(X_tgt_imp[eval_idx])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dp_mg = fit_domain_params_fewshot(
                X_cal_s_mg, A_mg, w_mg, TAU_OS, NU,
                domain_id=held["center_id"], n_synthetic=N_SYN,
                source_lambda0_mean=lam0_mg_mean,
                t_m=t_cal, delta_m=d_cal, variant="B",
            )

        U_cal_mg  = X_cal_s_mg  @ A_mg
        U_eval_mg = X_eval_s_mg @ A_mg

        result["MTG_noaug"] = _downstream_cox_auroc(
            U_cal_mg, t_cal, d_cal, U_eval_mg, y_eval)

        u_syn_mg, t_syn_mg, d_syn_mg = _generate_synthetic(dp_mg, K_MT, N_SYN, seed=seed+400)
        U_aug_mg = np.vstack([U_cal_mg, u_syn_mg])
        t_aug_mg = np.concatenate([t_cal, t_syn_mg])
        d_aug_mg = np.concatenate([d_cal, d_syn_mg])
        result["MTG_aug"] = _downstream_cox_auroc(
            U_aug_mg, t_aug_mg, d_aug_mg, U_eval_mg, y_eval)

    return result


# ── main loop ─────────────────────────────────────────────────────────────────

def run_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung (all IO lines)...")
    i3lung = load_i3lung()
    D = len(i3lung)

    print("\nLoading MSK PFS (for multitask condition)...")
    msk_pfs = load_msk_pfs()
    if msk_pfs is None:
        print("  MSK not available — multitask condition skipped")

    print("\nLoading MSK genomic + PFS (for genomic distillation condition)...")
    msk_geno_pfs = load_msk_genomic_pfs()
    if msk_geno_pfs is None:
        print("  MSK genomic not available — MT_geno condition skipped")

    print(f"\nn_syn={N_SYN}  k_OS={K_OS}  k_MT={K_MT}  "
          f"lam_MT={LAM_MT}  lam_geno={LAM_GENO}  seeds={N_SEEDS}")

    all_rows = []

    for m in M_VALS:
        print(f"\n{'='*100}")
        print(f"  m = {m} calibration patients")
        print(f"  {'centre':>6}  {'OS_noaug':>9}  {'OS_aug':>9}  "
              f"{'ΔOS':>7}  {'PC_noaug':>9}  {'PC_aug':>9}  "
              f"{'ΔPC':>7}  {'MT_noaug':>9}  {'MT_aug':>9}  {'ΔMT':>7}  "
              f"{'MTG_noaug':>9}  {'MTG_aug':>9}  {'ΔMTG':>7}")
        print("  " + "-" * 100)

        for held_idx in range(D):
            held   = i3lung[held_idx]
            src_cs = [c for i, c in enumerate(i3lung) if i != held_idx]

            # Source imputation medians (constant across seeds)
            X_src = np.vstack([c["X"] for c in src_cs])
            _, meds_src = impute_median(X_src)

            # Collect results across seeds
            fold_results: list[dict] = []
            for seed in range(N_SEEDS):
                r = run_fold(held, src_cs, meds_src, msk_pfs, m=m, seed=seed,
                             msk_geno_pfs=msk_geno_pfs)
                if r:
                    fold_results.append(r)

            if not fold_results:
                continue

            # Average over seeds
            keys = fold_results[0].keys()
            avg  = {k: float(np.nanmean([r[k] for r in fold_results if k in r]))
                    for k in keys}

            os_d  = avg.get("OS_aug",   float("nan")) - avg.get("OS_noaug",  float("nan"))
            pc_d  = avg.get("PC_aug",   float("nan")) - avg.get("PC_noaug",  float("nan"))
            mt_d  = avg.get("MT_aug",   float("nan")) - avg.get("MT_noaug",  float("nan"))
            mtg_d = avg.get("MTG_aug",  float("nan")) - avg.get("MTG_noaug", float("nan"))

            def _f(v): return f"{v:9.4f}" if not np.isnan(v) else f"{'—':>9}"
            def _d(v): return f"{v:+7.4f}" if not np.isnan(v) else f"{'—':>7}"

            print(f"  {held['name']:>6}  "
                  f"{_f(avg.get('OS_noaug',  float('nan')))}  "
                  f"{_f(avg.get('OS_aug',    float('nan')))}  {_d(os_d)}  "
                  f"{_f(avg.get('PC_noaug',  float('nan')))}  "
                  f"{_f(avg.get('PC_aug',    float('nan')))}  {_d(pc_d)}  "
                  f"{_f(avg.get('MT_noaug',  float('nan')))}  "
                  f"{_f(avg.get('MT_aug',    float('nan')))}  {_d(mt_d)}  "
                  f"{_f(avg.get('MTG_noaug', float('nan')))}  "
                  f"{_f(avg.get('MTG_aug',   float('nan')))}  {_d(mtg_d)}")

            for r in fold_results:
                row = {"center": held["name"], "n": held["n"], "m": m}
                row.update(r)
                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  Summary: mean AUROC over centres (averaged over seeds)")
    print(f"  {'m':>4}  {'OS_noaug':>9}  {'OS_aug':>9}  {'ΔOS':>7}  "
          f"{'PC_noaug':>9}  {'PC_aug':>9}  {'ΔPC':>7}  "
          f"{'MT_noaug':>9}  {'MT_aug':>9}  {'ΔMT':>7}  "
          f"{'MTG_noaug':>9}  {'MTG_aug':>9}  {'ΔMTG':>7}")
    print("  " + "-" * 100)

    for m in M_VALS:
        sub = df[df["m"] == m]
        if sub.empty:
            continue
        def _mg(col):
            return sub[col].mean() if col in sub.columns else float("nan")

        os_d  = _mg("OS_aug")  - _mg("OS_noaug")
        pc_d  = _mg("PC_aug")  - _mg("PC_noaug")
        mt_d  = _mg("MT_aug")  - _mg("MT_noaug")
        mtg_d = _mg("MTG_aug") - _mg("MTG_noaug")

        def _f(v): return f"{v:9.4f}" if not np.isnan(v) else f"{'—':>9}"
        def _d(v): return f"{v:+7.4f}" if not np.isnan(v) else f"{'—':>7}"

        print(f"  {m:>4}  "
              f"{_f(_mg('OS_noaug'))}  {_f(_mg('OS_aug'))}  {_d(os_d)}  "
              f"{_f(_mg('PC_noaug'))}  {_f(_mg('PC_aug'))}  {_d(pc_d)}  "
              f"{_f(_mg('MT_noaug'))}  {_f(_mg('MT_aug'))}  {_d(mt_d)}  "
              f"{_f(_mg('MTG_noaug'))}  {_f(_mg('MTG_aug'))}  {_d(mtg_d)}")

    # ── save ──────────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "fewshot_augmentation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── interpretation guide ──────────────────────────────────────────────
    print("\nInterpretation:")
    print("  ΔOS > 0        → synthetic augmentation helps few-shot OS prediction")
    print("  ΔPC − ΔOS > 0  → per-centre std improves synthetic data quality")
    print("  ΔMT − ΔOS > 0  → PFS transfer (MSK) improves the synthetic prior")
    print("  ΔMTG − ΔMT > 0 → genomic distillation adds signal beyond PFS-only")
    print("  Effect should be largest at m=10 (most data-hungry regime)")


if __name__ == "__main__":
    run_all()
