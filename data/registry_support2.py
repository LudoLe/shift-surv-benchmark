"""
SUPPORT2 registry: four clinical-domain groups as pseudo-centres.

SUPPORT (Study to Understand Prognoses and Preferences for Outcomes and Risks
of Treatments) enrolled 9105 critically ill adults across 5 US hospitals.
The public CSV does not contain a hospital identifier; instead we use
`dzclass` (disease class) as the domain — each class has a distinct clinical
profile and prognostic structure.

Domains (dzclass → centre id)
------------------------------
  0 : ARF/MOSF             n≈4227  acute respiratory/multi-organ failure
  1 : COPD/CHF/Cirrhosis   n≈2862  chronic organ disease
  2 : Cancer               n≈1420  advanced malignancy
  3 : Coma                 n≈596   neurological emergency

Outcome
-------
  Overall survival in months (d.time / 30.44), event = death.

Features (16)
-------------
  age, sex_male, meanbp, hrt, resp, temp, wblc, alb, bili, crea,
  sod, ph, num_co, scoma, diabetes, dementia

  High-missingness variables (glucose ≥49%, bun ≥48%, urine ≥53%,
  adlp ≥62%) are excluded; remaining NaNs handled by median imputation
  in the LOOCO runner.

Data source
-----------
  CSV stored at data/public/support2.csv
  (downloaded from https://hbiostat.org/data/repo/support2csv.zip)

Usage
-----
  from data import registry_support2 as reg
  centers, feature_names = reg.load_all_centers()
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

_PUBLIC = Path(__file__).parent / "public"
_SUPPORT2_CSV = _PUBLIC / "support2.csv"

TIME_COL  = "d.time"
EVENT_COL = "death"

DZCLASS_TO_ID: dict[str, int] = {
    "ARF/MOSF":           0,
    "COPD/CHF/Cirrhosis": 1,
    "Cancer":             2,
    "Coma":               3,
}

CENTER_NAMES: dict[int, str] = {
    0: "ARF/MOSF",
    1: "COPD/CHF/Cirr",
    2: "Cancer",
    3: "Coma",
}

FEATURE_COLS: list[str] = [
    "age",
    "sex_male",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "wblc",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "num_co",
    "scoma",
    "diabetes",
    "dementia",
]

# Tau grid (months) recommended for this dataset
TAU_GRID = [1.0, 2.0, 3.0, 6.0, 9.0, 12.0]
TAU      = 3.0
TAU_REF  = 6.0


def load_all_centers(registry_module=None) -> tuple[list[dict], list[str]]:
    """
    Load SUPPORT2 split by dzclass into four pseudo-centres.

    Returns
    -------
    centers       : list of 4 centre dicts
    feature_names : list[str] shared feature names
    """
    df = pd.read_csv(_SUPPORT2_CSV)

    # Encode sex
    df["sex_male"] = (df["sex"] == "male").astype(float)

    # Rename num.co → num_co (dot in column name causes issues)
    df["num_co"] = df["num.co"].astype(float)

    # Convert diabetes / dementia to numeric (they may be 0/1 already)
    df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")
    df["dementia"] = pd.to_numeric(df["dementia"], errors="coerce")

    centers = []
    for dzclass, cid in sorted(DZCLASS_TO_ID.items(), key=lambda x: x[1]):
        mask = df["dzclass"] == dzclass
        sub  = df[mask].reset_index(drop=True)

        X = sub[FEATURE_COLS].to_numpy(dtype=np.float64)
        t = (sub[TIME_COL].to_numpy(dtype=np.float64)) / 30.44   # days → months
        delta = sub[EVENT_COL].to_numpy(dtype=int)

        # Drop rows with NaN outcome time
        valid = ~np.isnan(t)
        if not valid.all():
            X, t, delta = X[valid], t[valid], delta[valid]

        t = np.clip(t, 0.01, None)

        centers.append({
            "X": X,
            "t": t,
            "delta": delta,
            "n": int(len(t)),
            "center_id": cid,
            "feature_names": FEATURE_COLS,
        })

    return centers, FEATURE_COLS
