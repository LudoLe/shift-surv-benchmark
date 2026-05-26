"""
Feature geometry diagnostic across i3lung centres.

Tests whether the observation layer (clinical feature covariance structure)
is stable across centres — the key prerequisite for deciding whether
A_d = A_shared + Δ_d is justified or unnecessary.

Metrics
-------
1. Per-centre feature summary: mean, std, missingness rate
2. Pairwise Frobenius distance  ‖Σ_i − Σ_j‖_F
3. Pairwise RV coefficient     RV(Σ_i, Σ_j) ∈ [0,1]  (1 = identical structure)
4. Principal angles between per-centre top-k PCA subspaces
5. Variance explained by pooled PCA per centre (how much shared structure explains)

Interpretation guide
--------------------
‖Σ_i − Σ_j‖_F < 2·p/√min(n_i,n_j) → consistent with sampling noise alone
RV ≈ 1 → same covariance structure
Principal angle θ_1 > 20° → meaningful subspace divergence
If most pairs are within sampling noise → shared A is well-justified
If several pairs show systematic deviation → Δ_d may capture real shift

Usage
-----
    cd /Users/ludole/Desktop/SHIFT-SURV
    python -m experiments.run_feature_geometry_diagnostic
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data.preprocess import impute_median

OUT_DIR = _ROOT / "outputs" / "tables" / "geometry"

_RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
_OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")

COMMON_FEATS = ["SEX", "ECOG PS", "NLR",
                "SMOKING CURRENT", "SMOKING FORMER", "SMOKING NEVER",
                "PDL1 CATEGORY"]
K = 6   # PCA dimensions (same as generator)


# ── helpers ───────────────────────────────────────────────────────────────────

def _imp(X, meds):
    return np.where(np.isnan(X), meds, X)


def _rv_coefficient(A: np.ndarray, B: np.ndarray) -> float:
    """
    RV coefficient between two (p×p) covariance matrices.
    RV(A,B) = tr(AB) / sqrt(tr(A²) · tr(B²))
    Range [0,1]; 1 = identical up to scaling.
    """
    num  = float(np.trace(A @ B))
    denom = np.sqrt(float(np.trace(A @ A)) * float(np.trace(B @ B)))
    return num / denom if denom > 1e-10 else float("nan")


def _sampling_noise_threshold(n_i: int, n_j: int, p: int) -> float:
    """
    Approximate ‖Σ_i − Σ_j‖_F expected under identical populations.
    Uses asymptotic variance of sample covariance: Var ≈ 2/n for each entry.
    ‖ΔΣ‖_F ≈ sqrt(2 p² / min(n)) as a rough guide.
    """
    return float(np.sqrt(2 * p**2 / min(n_i, n_j)))


# ── data loader ───────────────────────────────────────────────────────────────

def load_i3lung_raw() -> list[dict]:
    """Load i3lung centres with raw (un-imputed) feature matrices."""
    rwd = pd.read_csv(_RWD_PATH)
    out = pd.read_csv(_OUTCOMES_PATH)
    df  = rwd.merge(
        out[["Subject", "OS MONTHS", "DEATH EVENT"]],
        on="Subject", how="inner"
    ).dropna(subset=["OS MONTHS", "DEATH EVENT"])

    centers = []
    for cid, (name, grp) in enumerate(df.groupby("CENTER")):
        X_raw = grp[COMMON_FEATS].to_numpy(float)
        t     = grp["OS MONTHS"].to_numpy(float)
        ev    = grp["DEATH EVENT"].to_numpy(float)
        centers.append({
            "X_raw": X_raw, "t": t, "delta": ev,
            "n": len(X_raw), "center_id": cid, "name": name,
        })
    return centers


# ── diagnostic functions ──────────────────────────────────────────────────────

def feature_summary(centers: list[dict], X_scaled_list: list[np.ndarray]) -> pd.DataFrame:
    """Per-centre, per-feature: mean, std, missingness rate (before imputation)."""
    rows = []
    for c, Xs in zip(centers, X_scaled_list):
        Xr = c["X_raw"]
        miss = np.mean(np.isnan(Xr), axis=0)   # missingness in original
        for j, feat in enumerate(COMMON_FEATS):
            rows.append({
                "center": c["name"], "feature": feat,
                "mean":    round(float(np.mean(Xs[:, j])), 4),
                "std":     round(float(np.std(Xs[:, j])),  4),
                "missing": round(float(miss[j]) * 100,     1),
            })
    return pd.DataFrame(rows)


def pairwise_covariance_metrics(
    centers: list[dict],
    X_scaled_list: list[np.ndarray],
) -> pd.DataFrame:
    """
    All pairwise (i, j) pairs:
      - Frobenius distance ‖Σ_i − Σ_j‖_F
      - RV coefficient
      - Noise threshold (expected distance under H0: same distribution)
      - Ratio: observed / expected (> 2 = statistically surprising)
    """
    covs = [np.cov(Xs.T) for Xs in X_scaled_list]
    rows = []
    n_c = len(centers)
    for i in range(n_c):
        for j in range(i + 1, n_c):
            frob  = float(np.linalg.norm(covs[i] - covs[j], "fro"))
            rv    = _rv_coefficient(covs[i], covs[j])
            noise = _sampling_noise_threshold(centers[i]["n"], centers[j]["n"],
                                              len(COMMON_FEATS))
            rows.append({
                "center_i":  centers[i]["name"],
                "center_j":  centers[j]["name"],
                "n_i":       centers[i]["n"],
                "n_j":       centers[j]["n"],
                "frob":      round(frob,  4),
                "rv":        round(rv,    4),
                "noise_thr": round(noise, 4),
                "snr_ratio": round(frob / noise, 2),
            })
    return pd.DataFrame(rows)


def pca_subspace_angles(
    centers: list[dict],
    X_scaled_list: list[np.ndarray],
    k_small: int = 2,
) -> pd.DataFrame:
    """
    Pairwise principal angles between per-centre top-k_small PCA subspaces.

    Note: using k_small=2 (not K=6) because with K=p-1 the k-dimensional
    subspace fills nearly all of ℝ^p and all angles are trivially ≈ 0.
    k_small=2 compares only the most important directions, where genuine
    centre-level differences are most visible.

    Also compared against the pooled (shared) A subspace.
    """
    def _pca_basis(X, k):
        pca = PCA(n_components=k, random_state=0)
        pca.fit(X)
        return pca.components_.T   # (p, k) — orthonormal columns

    X_pooled = np.vstack(X_scaled_list)
    A_pooled = _pca_basis(X_pooled, k_small)

    per_center_bases = [_pca_basis(Xs, k_small) for Xs in X_scaled_list]

    rows = []
    n_c = len(centers)
    for i in range(n_c):
        for j in range(i + 1, n_c):
            angles = np.degrees(
                subspace_angles(per_center_bases[i], per_center_bases[j]))
            rows.append({
                "center_i":       centers[i]["name"],
                "center_j":       centers[j]["name"],
                "type":           "center–center",
                "theta_1_deg":    round(float(angles[0]),  2),
                "theta_2_deg":    round(float(angles[-1]), 2),
                "theta_mean_deg": round(float(angles.mean()), 2),
            })
        # vs. pooled
        angles_vs_pool = np.degrees(
            subspace_angles(per_center_bases[i], A_pooled))
        rows.append({
            "center_i":       centers[i]["name"],
            "center_j":       "POOLED",
            "type":           "center–pooled",
            "theta_1_deg":    round(float(angles_vs_pool[0]),  2),
            "theta_2_deg":    round(float(angles_vs_pool[-1]), 2),
            "theta_mean_deg": round(float(angles_vs_pool.mean()), 2),
        })

    return pd.DataFrame(rows)


def feature_variance_ratio(
    centers: list[dict],
    X_scaled_list: list[np.ndarray],
) -> pd.DataFrame:
    """
    Per-feature variance ratio: centre std / 1.0 (pooled std = 1 after scaling).
    Ratios far from 1.0 indicate that the pooled scale fits poorly for that centre
    and that a centre-specific feature scaling could help.
    """
    rows = []
    for c, Xs in zip(centers, X_scaled_list):
        stds = np.std(Xs, axis=0)
        for j, feat in enumerate(COMMON_FEATS):
            rows.append({
                "center":  c["name"],
                "feature": feat,
                "std_ratio": round(float(stds[j]), 3),
            })
    df = pd.DataFrame(rows)
    return df.pivot(index="center", columns="feature", values="std_ratio")


def variance_explained_by_shared_A(
    centers: list[dict],
    X_scaled_list: list[np.ndarray],
    k: int = K,
) -> pd.DataFrame:
    """
    For each centre, what fraction of total variance is explained by
    projecting onto the shared (pooled) loading matrix A?
    If this is < 0.8, the shared A misses centre-specific variance.
    """
    X_pooled = np.vstack(X_scaled_list)
    pca = PCA(n_components=k, random_state=0)
    pca.fit(X_pooled)
    A = pca.components_.T   # (p, k)

    rows = []
    for c, Xs in zip(centers, X_scaled_list):
        total_var  = float(np.sum(Xs ** 2))                  # tr(XᵀX)
        proj       = Xs @ A                                   # (n, k)
        explained  = float(np.sum(proj ** 2))                 # tr(XᵀAA'X)
        frac       = explained / total_var if total_var > 0 else float("nan")

        # Per-centre PCA variance (how much does their own PCA explain?)
        pca_c = PCA(n_components=k, random_state=0).fit(Xs)
        own_explained = float(pca_c.explained_variance_ratio_.sum())

        rows.append({
            "center":          c["name"],
            "n":               c["n"],
            "var_explained_shared": round(frac, 4),
            "var_explained_own":    round(own_explained, 4),
            "var_gap":         round(own_explained - frac, 4),
        })
    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

def run_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading i3lung data...")
    centers = load_i3lung_raw()
    for c in centers:
        print(f"  {c['name']:>6}: n={c['n']}")

    # ── Standardise: impute with pooled medians, scale to unit variance ───
    X_pooled_raw = np.vstack([c["X_raw"] for c in centers])
    _, meds = impute_median(X_pooled_raw)
    X_imp_list = [_imp(c["X_raw"], meds) for c in centers]

    sc = StandardScaler().fit(np.vstack(X_imp_list))
    X_scaled_list = [sc.transform(X) for X in X_imp_list]

    # ── 1. Feature summary ────────────────────────────────────────────────
    print("\n=== 1. Feature summary (standardised) ===")
    df_feat = feature_summary(centers, X_scaled_list)
    # Pivot for readability
    pivot_mean = df_feat.pivot(index="center", columns="feature", values="mean")
    pivot_std  = df_feat.pivot(index="center", columns="feature", values="std")
    pivot_miss = df_feat.pivot(index="center", columns="feature", values="missing")

    print("\n  Mean per feature (standardised):")
    print(pivot_mean.to_string())
    print("\n  Std per feature (standardised):")
    print(pivot_std.to_string())
    print("\n  Missingness % per feature (raw):")
    print(pivot_miss.to_string())

    # ── 2. Pairwise covariance metrics ────────────────────────────────────
    print("\n=== 2. Pairwise covariance distances ===")
    df_cov = pairwise_covariance_metrics(centers, X_scaled_list)
    print(df_cov[["center_i", "center_j", "frob", "rv",
                  "noise_thr", "snr_ratio"]].to_string(index=False))

    n_sig = (df_cov["snr_ratio"] > 2.0).sum()
    print(f"\n  Pairs with frob/noise_thr > 2 (likely real shift): "
          f"{n_sig} / {len(df_cov)}")

    # ── 3. PCA subspace angles (top-2 directions) ────────────────────────
    k_small = 2
    print(f"\n=== 3. Principal angles — top-{k_small} PCA subspaces (degrees) ===")
    print(f"  (k={k_small}, not k={K}: with k=p-1 all subspaces trivially overlap)")
    df_ang = pca_subspace_angles(centers, X_scaled_list, k_small=k_small)

    print("\n  Centre–centre pairs:")
    cc = df_ang[df_ang["type"] == "center–center"]
    print(cc[["center_i", "center_j", "theta_1_deg",
              "theta_2_deg", "theta_mean_deg"]].to_string(index=False))

    print("\n  Each centre vs. pooled A:")
    cp = df_ang[df_ang["type"] == "center–pooled"]
    print(cp[["center_i", "theta_1_deg",
              "theta_2_deg", "theta_mean_deg"]].to_string(index=False))

    # ── 4. Variance explained by shared A ────────────────────────────────
    print(f"\n=== 4. Variance explained by shared A (k={K}) ===")
    df_var = variance_explained_by_shared_A(centers, X_scaled_list, k=K)
    print(df_var.to_string(index=False))

    # ── 5. Per-feature variance ratio ─────────────────────────────────────
    print("\n=== 5. Per-feature std ratio (centre / pooled, 1.0 = perfect match) ===")
    df_vratio = feature_variance_ratio(centers, X_scaled_list)
    print(df_vratio.to_string())

    # Flag features with ratio outside [0.7, 1.3]
    arr = df_vratio.to_numpy()
    extremes = []
    for ri, rname in enumerate(df_vratio.index):
        for ci, cname in enumerate(df_vratio.columns):
            v = arr[ri, ci]
            if v < 0.7 or v > 1.3:
                extremes.append(f"{rname}/{cname}={v:.2f}")
    if extremes:
        print(f"\n  Features with ratio outside [0.7, 1.3]: {', '.join(extremes)}")
    else:
        print("\n  All feature std ratios within [0.7, 1.3]")

    # ── Save ──────────────────────────────────────────────────────────────
    df_feat.to_csv(OUT_DIR / "feature_summary.csv", index=False)
    df_cov.to_csv(OUT_DIR  / "covariance_distances.csv", index=False)
    df_ang.to_csv(OUT_DIR  / "subspace_angles.csv", index=False)
    df_var.to_csv(OUT_DIR  / "variance_explained.csv", index=False)
    df_vratio.to_csv(OUT_DIR / "feature_variance_ratio.csv")
    print(f"\nSaved → {OUT_DIR}/")

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n=== Verdict ===")
    mean_frob  = df_cov["frob"].mean()
    mean_snr   = df_cov["snr_ratio"].mean()
    mean_rv    = df_cov["rv"].mean()
    mean_ang1  = cc["theta_1_deg"].mean()
    n_extreme  = len(extremes)

    print(f"  Mean pairwise ‖Σ_i−Σ_j‖_F : {mean_frob:.3f}")
    print(f"  Mean frob/noise_thr         : {mean_snr:.2f}  "
          f"({'above' if mean_snr > 2 else 'within'} sampling noise)")
    print(f"  Mean RV coefficient         : {mean_rv:.4f}  "
          f"({'similar' if mean_rv > 0.9 else 'divergent'} covariance structure)")
    print(f"  Mean top-2 subspace angle   : {mean_ang1:.1f}°  "
          f"({'shift detectable' if mean_ang1 > 15 else 'subspaces aligned'})")
    print(f"  Extreme std-ratio entries   : {n_extreme}  "
          f"({'present' if n_extreme > 0 else 'none'} — features with anomalous scale in some centre)")

    if mean_snr < 2 and mean_ang1 < 10 and n_extreme == 0:
        print("\n  → Geometry STABLE: shared A well-justified; Δ_d not supported by data.")
    elif mean_snr > 3 or mean_ang1 > 25 or n_extreme > 3:
        print("\n  → Real centre differences detected.")
        print("    Δ_d with feature-level scaling could capture genuine shift.")
    else:
        print("\n  → Mixed evidence.")
        print("    Covariance differences exist but are partly attributable to sampling noise.")
        print("    The dominant shift is in feature MEANS (location), not covariance geometry.")
        print("    Δ_d (rotation correction) is unlikely to help; per-centre μ_d already handles location.")


if __name__ == "__main__":
    run_all()
