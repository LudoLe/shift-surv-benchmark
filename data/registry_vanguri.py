"""
Vanguri et al. (Nature Cancer 2022) — MSK NSCLC immunotherapy cohort.

Source : Synapse syn26642505
         Clinical file : Job-832174535646262024340388944.csv  (n=368)
         PDL1 scores   : PDL1_SCORE.parquet                  (n=201)

Single centre: Memorial Sloan Kettering (MSK), centre_id=6.

Seven features aligned to the i3lung standard set
--------------------------------------------------
  SEX              : 1=male  0=female   (Vanguri sex: 1=M 2=F)
  ECOG PS          : 0/1/2
  NLR              : dNLR used as proxy (derived NLR, same direction, different scale)
  SMOKING CURRENT  : 1 if smoking_status==1
  SMOKING FORMER   : 1 if smoking_status==2
  SMOKING NEVER    : 1 if smoking_status==0
  PDL1 CATEGORY    : 0=<1%  1=1-49%  2=≥50%  (NaN where PDL1 score missing → imputed)

Outcome
-------
  Overall survival in months:
    dead   (vital_status=1) : t = os_int,  delta = 1
    alive  (vital_status=0) : t = pfs,     delta = 0  (censored at last known follow-up)
  tau = 12 months  (matches i3lung)

File paths (hardcoded — edit if you move the downloads)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

_CLINICAL_CSV  = Path("/Users/ludole/Downloads/Job-832174535646262024340388944.csv")
_PDL1_PARQUET  = Path("/Users/ludole/Downloads/PDL1_SCORE.parquet")

CENTER_NAMES: dict[int, str] = {6: "MSK"}

FEATURE_COLS: list[str] = [
    "SEX",
    "ECOG PS",
    "NLR",
    "SMOKING CURRENT",
    "SMOKING FORMER",
    "SMOKING NEVER",
    "PDL1 CATEGORY",
]

TAU_GRID = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]
TAU      = 12.0
TAU_REF  = 18.0


def _pdl1_category(score_series: pd.Series) -> pd.Series:
    """Bin continuous Sauter PDL1 score (0-100) to i3lung ordinal (0/1/2)."""
    out = pd.Series(np.nan, index=score_series.index, dtype=float)
    numeric = pd.to_numeric(score_series, errors="coerce")
    out[numeric == 0]                        = 0.0
    out[(numeric > 0) & (numeric < 50)]      = 1.0
    out[numeric >= 50]                       = 2.0
    return out


def load_all_centers(registry_module=None) -> tuple[list[dict], list[str]]:
    """
    Load Vanguri MSK cohort as a single centre dict.

    Returns
    -------
    centers       : list with one element (centre_id=6, MSK)
    feature_names : FEATURE_COLS
    """
    clin = pd.read_csv(_CLINICAL_CSV)
    pdl1 = pd.read_parquet(_PDL1_PARQUET)

    # Merge PDL1 on main_index (left join — keeps all clinical rows)
    pdl1.index.name = "main_index"
    pdl1 = pdl1.reset_index()
    merged = clin.merge(pdl1, on="main_index", how="left")

    # ── features ────────────────────────────────────────────────────────────
    sex      = (merged["sex"] == 1).astype(float)                      # 1=M
    ecog     = pd.to_numeric(merged["ecog"], errors="coerce")
    nlr      = pd.to_numeric(merged["dnlr"], errors="coerce")          # dNLR proxy
    smk_cur  = (merged["smoking_status"] == 1).astype(float)
    smk_for  = (merged["smoking_status"] == 2).astype(float)
    smk_nev  = (merged["smoking_status"] == 0).astype(float)
    pdl1_cat = _pdl1_category(merged["Sauter PD-L1 Score"])            # NaN where missing

    X = np.column_stack([
        sex.to_numpy(dtype=np.float64),
        ecog.to_numpy(dtype=np.float64),
        nlr.to_numpy(dtype=np.float64),
        smk_cur.to_numpy(dtype=np.float64),
        smk_for.to_numpy(dtype=np.float64),
        smk_nev.to_numpy(dtype=np.float64),
        pdl1_cat.to_numpy(dtype=np.float64),
    ])

    # ── outcome ─────────────────────────────────────────────────────────────
    vital  = pd.to_numeric(merged["vital_status"], errors="coerce")
    os_int = pd.to_numeric(merged["os_int"],       errors="coerce")
    pfs    = pd.to_numeric(merged["pfs"],          errors="coerce")

    # dead → t = os_int ; alive → t = pfs (censoring proxy)
    t_raw = np.where(vital == 1, os_int, pfs)
    delta = np.where(vital == 1, 1, 0).astype(int)

    # Drop rows with missing vital_status or missing time
    valid = (~np.isnan(vital)) & (~np.isnan(t_raw))
    X, t_raw, delta = X[valid], t_raw[valid], delta[valid]
    t = np.clip(t_raw, 0.01, None)

    center = {
        "X":             X,
        "t":             t,
        "delta":         delta,
        "n":             int(len(t)),
        "center_id":     6,
        "feature_names": FEATURE_COLS,
    }
    return [center], FEATURE_COLS
