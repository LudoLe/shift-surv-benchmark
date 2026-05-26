"""
Two-center breast cancer registry: Rotterdam + GBSG2.

Centers
-------
  0 : Rotterdam  n=2982  Dutch multi-hospital breast-cancer study (1978–1993)
  1 : GBSG2      n=686   German Breast Cancer Study Group (1984–1989)

Outcome
-------
  Recurrence-free survival (RFS) in months.
  Event = first recurrence or death (whichever comes first).
  Time converted from days: months = days / 30.44.

Harmonised features (8)
-----------------------
  age        : age at surgery (years)
  meno       : menopausal status  0=pre  1=post
  nodes      : number of positive lymph nodes
  pgr        : progesterone receptor level (fmol/l)
  er         : estrogen receptor level (fmol/l)
  hormon     : hormonal therapy received  0/1
  grade_high : high tumour grade  0=(grade 1–2)  1=(grade 3)
  size_ord   : tumour size ordinal  0=(≤20 mm)  1=(20–50 mm)  2=(>50 mm)

Data source
-----------
  Rotterdam : downloaded from Rdatasets mirror of R `survival` package
              stored at data/public/rotterdam.csv
  GBSG2     : loaded via sksurv.datasets.load_gbsg2()

Usage
-----
  from data import registry_rotterdam_gbsg as reg
  centers, feature_names = reg.load_all_centers()
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2

_PUBLIC = Path(__file__).parent / "public"
_ROTTERDAM_CSV = _PUBLIC / "rotterdam.csv"

TIME_COL  = "rfs_months"
EVENT_COL = "recur"

CENTER_NAMES: dict[int, str] = {
    0: "Rotterdam",
    1: "GBSG2",
}

FEATURE_COLS: list[str] = [
    "age",
    "meno",
    "nodes",
    "pgr",
    "er",
    "hormon",
    "grade_high",
    "size_ord",
]

# Tau grid (months) recommended for this dataset
TAU_GRID = [12.0, 24.0, 36.0, 48.0, 60.0]
TAU      = 36.0
TAU_REF  = 48.0


def _load_rotterdam() -> dict:
    df = pd.read_csv(_ROTTERDAM_CSV)

    t      = df["rtime"].to_numpy(dtype=np.float64) / 30.44
    delta  = df["recur"].to_numpy(dtype=int)

    # grade: 2→0, 3→1
    grade_high = (df["grade"] == 3).astype(float).to_numpy()

    # size: <=20→0, 20-50→1, >50→2
    size_map = {"<=20": 0.0, "20-50": 1.0, ">50": 2.0}
    size_ord = df["size"].map(size_map).to_numpy(dtype=np.float64)

    X = np.column_stack([
        df["age"].to_numpy(dtype=np.float64),
        df["meno"].to_numpy(dtype=np.float64),
        df["nodes"].to_numpy(dtype=np.float64),
        df["pgr"].to_numpy(dtype=np.float64),
        df["er"].to_numpy(dtype=np.float64),
        df["hormon"].to_numpy(dtype=np.float64),
        grade_high,
        size_ord,
    ])

    # Drop rows with NaN time
    valid = ~np.isnan(t)
    X, t, delta = X[valid], t[valid], delta[valid]
    t = np.clip(t, 0.01, None)

    return {
        "X": X,
        "t": t,
        "delta": delta,
        "n": int(len(t)),
        "center_id": 0,
        "feature_names": FEATURE_COLS,
    }


def _load_gbsg2() -> dict:
    X_df, y = load_gbsg2()

    t      = y["time"].astype(np.float64) / 30.44
    delta  = y["cens"].astype(int)

    meno       = (X_df["menostat"] == "Post").astype(float).to_numpy()
    grade_high = (X_df["tgrade"]   == "III").astype(float).to_numpy()
    hormon     = (X_df["horTh"]    == "yes").astype(float).to_numpy()

    tsize = X_df["tsize"].to_numpy(dtype=np.float64)
    size_ord = np.where(tsize <= 20, 0.0, np.where(tsize <= 50, 1.0, 2.0))

    X = np.column_stack([
        X_df["age"].to_numpy(dtype=np.float64),
        meno,
        X_df["pnodes"].to_numpy(dtype=np.float64),
        X_df["progrec"].to_numpy(dtype=np.float64),
        X_df["estrec"].to_numpy(dtype=np.float64),
        hormon,
        grade_high,
        size_ord,
    ])

    valid = ~np.isnan(t)
    X, t, delta = X[valid], t[valid], delta[valid]
    t = np.clip(t, 0.01, None)

    return {
        "X": X,
        "t": t,
        "delta": delta,
        "n": int(len(t)),
        "center_id": 1,
        "feature_names": FEATURE_COLS,
    }


def load_all_centers(registry_module=None) -> tuple[list[dict], list[str]]:
    """
    Load Rotterdam and GBSG2 as two centres.

    Returns
    -------
    centers       : list of 2 centre dicts with keys X, t, delta, n, center_id, feature_names
    feature_names : list[str] shared feature names
    """
    rotterdam = _load_rotterdam()
    gbsg2     = _load_gbsg2()
    centers   = [rotterdam, gbsg2]
    return centers, FEATURE_COLS
