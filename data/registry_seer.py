"""SEER multi-registry survival registry.

Genuine multicentric: the SEER geographic **registries** are the centers, so a
source->target transfer is a real cross-site generalisation.

This module expects a *cleaned* CSV at ``data/public/seer.csv`` exported from
SEER*Stat (a case listing). See ``SEER_INSTRUCTIONS.md`` for how to obtain access
and produce it. Expected columns (rename SEER*Stat variables to these):

  registry          : SEER registry name (the center)          [str]
  survival_months   : Survival months (overall survival)       [numeric]
  vital_status      : 1 = dead (event), 0 = alive (censored)   [0/1 or "Dead"/"Alive"]
  age               : age at diagnosis, years                  [numeric]
  sex               : 0 = male, 1 = female                     [0/1]
  year_dx           : year of diagnosis                        [numeric]
  stage             : ordinal stage (e.g. 0=localized,1=regional,2=distant)
  grade             : tumour grade (1..4)                      [numeric]
  tumor_size        : tumour size, mm                          [numeric]

Defaults are configured for a solid-tumour cohort (e.g. lung, to mirror I3LUNG).
For a different site, edit FEATURE_COLS / TAU and re-export the matching columns.
Unparseable / coded-missing feature values become NaN and are median-imputed
downstream (source-fit) exactly like the I3LUNG pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PUBLIC   = Path(__file__).parent / "public"
_SEER_CSV = _PUBLIC / "seer.csv"

TIME_COL     = "survival_months"
EVENT_COL    = "vital_status"
REGISTRY_COL = "registry"

FEATURE_COLS: list[str] = ["age", "sex", "year_dx", "stage", "grade", "tumor_size"]

# Fixed-horizon grid (months). Solid tumours -> longer horizons than I3LUNG's 6m.
TAU_GRID = [6.0, 12.0, 24.0, 36.0, 60.0]
TAU      = 12.0
TAU_REF  = 24.0

# Registries with fewer than this many patients are dropped (unstable centers).
MIN_N = 500

# Populated by load_all_centers(); read by run_real_data.load_public_transfer.
CENTER_NAMES: dict[int, str] = {}


def _coerce_event(s: pd.Series) -> np.ndarray:
    """Map vital status to 1=dead / 0=alive, accepting numeric or SEER strings."""
    if s.dtype == object:
        m = s.astype(str).str.strip().str.lower()
        return m.map(lambda v: 1 if v.startswith("dead") else 0).to_numpy(dtype=int)
    return pd.to_numeric(s, errors="coerce").fillna(0).round().astype(int).to_numpy()


def load_all_centers(registry_module=None) -> tuple[list[dict], list[str]]:
    """Load SEER as one center per registry (filtered to >= MIN_N patients).

    Returns (centers, feature_names) with the six-key center dict schema used by
    the rest of the pipeline; times in months, delta = 1 for death.
    """
    if not _SEER_CSV.exists():
        raise FileNotFoundError(
            f"{_SEER_CSV} not found. Export a SEER*Stat case listing and clean it "
            f"to the expected schema — see SEER_INSTRUCTIONS.md."
        )

    df = pd.read_csv(_SEER_CSV)
    required = [TIME_COL, EVENT_COL, REGISTRY_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"seer.csv is missing columns {missing}. Present: {list(df.columns)}. "
            f"Rename the SEER*Stat variables to the expected names (see the module docstring)."
        )

    # Outcome must be parseable; drop rows without survival time or registry.
    df = df.dropna(subset=[REGISTRY_COL]).copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    global CENTER_NAMES
    CENTER_NAMES = {}
    centers: list[dict] = []
    cid = 0
    for name, g in df.groupby(REGISTRY_COL, sort=True):
        if len(g) < MIN_N:
            continue
        # Features -> numeric (coded-missing -> NaN, imputed downstream).
        X = np.column_stack([
            pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            for c in FEATURE_COLS
        ])
        t = np.clip(g[TIME_COL].to_numpy(dtype=float), 0.01, None)
        delta = _coerce_event(g[EVENT_COL])
        CENTER_NAMES[cid] = str(name)
        centers.append({
            "X": X, "t": t, "delta": delta, "n": int(len(t)),
            "center_id": cid, "feature_names": FEATURE_COLS,
        })
        cid += 1

    if not centers:
        raise ValueError(
            f"No SEER registry has >= MIN_N={MIN_N} patients after cleaning. "
            f"Lower MIN_N or check the export."
        )
    return centers, FEATURE_COLS
