"""
Registry for the extended RWD feature set (22 features vs 11 original).

Source: /Users/ludole/Downloads/rwd.csv
Outcomes: OS MONTHS, DEATH EVENT from per-center CSVs in data/raw/

New features vs original 11
----------------------------
+ IO LINE (binary: 1=first line)
+ AGE AT IO START (ordinal: 0=<62, 1=62-69, 2=>69)
+ HISTOLOGY ADENOCARCINOMA, HISTOLOGY SQUAMOUS
+ NEUTROPHYL, LYMPHOCYTES, MONOCYTE
+ STAGE AT IO START (binary: 1=stage 3, 0=stage ≤2)
+ ADRENAL METS AT IO START, LUNG METS AT IO START
+ BMI (ordinal: 0=<23, 1=23-27, 2=>27)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT      = Path(__file__).parent
_RWD_PATH  = Path("/Users/ludole/Downloads/rwd.csv")

TIME_COL  = "OS MONTHS"
EVENT_COL = "DEATH EVENT"

CENTER_NAMES: dict[int, str] = {
    0: "GHD", 1: "INT", 2: "MH", 3: "SZMC", 4: "UOC", 5: "VHIO",
}

_CENTER_ORDER = ["GHD", "INT", "MH", "SZMC", "UOC", "VHIO"]

FEATURE_COLS: list[str] = [
    # ── original 11 ──────────────────────────────────────────────────────
    "SEX",
    "ECOG PS",
    "LDH",
    "NLR",
    "SMOKING CURRENT",
    "SMOKING FORMER",
    "SMOKING NEVER",
    "PDL1 CATEGORY",
    "PARENCHYMAL BRAIN METS AT IO START",
    "LIVER METS AT IO START",
    "BONE METS AT IO START",
    # ── new features ─────────────────────────────────────────────────────
    "IO_LINE_1",               # 1 = first line IO
    "AGE_IO_ORD",              # ordinal: 0=<62, 1=62-69, 2=>69
    "HISTOLOGY ADENOCARCINOMA",
    "HISTOLOGY SQUAMOUS",
    "NEUTROPHYL",
    "LYMPHOCYTES",
    "MONOCYTE",
    "STAGE3_AT_IO",            # 1 = stage 3 (metastatic)
    "ADRENAL METS AT IO START",
    "LUNG METS AT IO START",
    "BMI_ORD",                 # ordinal: 0=<23, 1=23-27, 2=>27
]


def _encode_rwd(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the RWD dataframe."""
    out = df.copy()

    # IO LINE: 1 = first line, 0 = later
    out["IO_LINE_1"] = (out["IO LINE"] == 1).astype(float)

    # AGE AT IO START: ordinal
    age_map = {"<62": 0, "62-69": 1, ">69": 2}
    out["AGE_IO_ORD"] = out["AGE AT IO START"].map(age_map)

    # STAGE AT IO START: binary metastatic
    out["STAGE3_AT_IO"] = (out["STAGE AT IO START"] == 3).astype(float)

    # BMI: ordinal
    bmi_map = {"<23": 0, "23-27": 1, ">27": 2}
    out["BMI_ORD"] = out["BMI"].map(bmi_map)

    return out


def load_all_centers(registry_module=None, io_line_1_only: bool = False):
    """
    Load extended RWD features + outcomes from per-center CSVs.

    Parameters
    ----------
    registry_module  : ignored (kept for API compatibility)
    io_line_1_only   : if True, keep only IO LINE == 1 patients

    Returns
    -------
    centers      : list of center dicts (same format as data.loader)
    feature_names: list[str]
    """
    rwd = pd.read_csv(_RWD_PATH)
    rwd = _encode_rwd(rwd)

    if io_line_1_only:
        rwd = rwd[rwd["IO LINE"] == 1].reset_index(drop=True)

    # When filtered to IO LINE=1, IO_LINE_1 is constant → drop it
    feat_cols = [f for f in FEATURE_COLS
                 if not (io_line_1_only and f == "IO_LINE_1")]

    centers = []
    for cid, cname in enumerate(_CENTER_ORDER):
        raw_path = _ROOT / "raw" / f"center_{cname}.csv"
        raw      = pd.read_csv(raw_path)

        # Row-aligned join (verified: LDH/NLR identical between rwd and raw)
        sub = rwd[rwd["CENTER"] == cname].reset_index(drop=True)

        if io_line_1_only:
            # Need to subset raw outcomes to match filtered rows
            # Use the original (unfiltered) rwd to get row indices within center
            rwd_full = pd.read_csv(_RWD_PATH)
            center_mask = rwd_full["CENTER"] == cname
            io1_mask    = rwd_full["IO LINE"] == 1
            keep_idx    = np.where((center_mask & io1_mask).values[
                center_mask.values
            ])[0]
            raw = raw.iloc[keep_idx].reset_index(drop=True)

        t      = raw[TIME_COL].to_numpy(dtype=np.float64)
        delta  = raw[EVENT_COL].to_numpy(dtype=int)
        X_cols = sub[feat_cols].to_numpy(dtype=np.float64)

        # Drop rows with NaN time
        nan_mask = np.isnan(t)
        if nan_mask.any():
            keep = ~nan_mask
            t, delta, X_cols = t[keep], delta[keep], X_cols[keep]

        # Clip non-positive times
        t = np.clip(t, 0.01, None)

        centers.append({
            "X":            X_cols,
            "t":            t,
            "delta":        delta,
            "n":            int(len(t)),
            "center_id":    cid,
            "feature_names": feat_cols,
        })

    return centers, FEATURE_COLS
