"""
Load and align real multi-center clinical data.

Each call to `load_center` returns a dict with arrays ready for downstream use.
`load_all_centers` loads all centers, validates feature alignment, and returns
a list of per-center dicts plus metadata about the shared feature set.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-center loader
# ---------------------------------------------------------------------------

def load_center(
    path: str | Path,
    center_id: int,
    time_col: str,
    event_col: str,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Load one center's CSV and return aligned arrays.

    Parameters
    ----------
    path        : path to CSV file
    center_id   : integer identifier for this center
    time_col    : column name for follow-up / event time
    event_col   : column name for event indicator (1=event, 0=censored)
    feature_cols: list of covariate column names; None = all except time/event

    Returns
    -------
    dict with keys:
      X            : (n, p) float64 covariate matrix
      t            : (n,)   float64 observed times
      delta        : (n,)   int     event indicators
      n            : int    sample size
      center_id    : int
      feature_names: list[str]
    """
    df = pd.read_csv(path)

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in {path}")
    if event_col not in df.columns:
        raise ValueError(f"event_col '{event_col}' not found in {path}")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (time_col, event_col)]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    t = df[time_col].to_numpy(dtype=np.float64)
    delta = df[event_col].to_numpy(dtype=int)

    # Drop rows where time is NaN (cannot be used in any downstream computation)
    nan_mask = np.isnan(t)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        import warnings
        warnings.warn(
            f"Center {center_id}: {n_nan} NaN times in '{time_col}' — rows dropped"
        )
        keep = ~nan_mask
        X, t, delta = X[keep], t[keep], delta[keep]

    # Clip zero/negative times to a small positive value (data entry artefacts)
    if np.any(t <= 0):
        n_bad = int((t <= 0).sum())
        import warnings
        warnings.warn(
            f"Center {center_id}: {n_bad} non-positive times in '{time_col}' — clipped to 0.01"
        )
        t = np.clip(t, a_min=0.01, a_max=None)
    if not set(np.unique(delta)).issubset({0, 1}):
        raise ValueError(f"Center {center_id}: event column '{event_col}' must be binary (0/1)")

    return {
        "X": X,
        "t": t,
        "delta": delta,
        "n": int(len(t)),
        "center_id": center_id,
        "feature_names": feature_cols,
    }


# ---------------------------------------------------------------------------
# Multi-center loader
# ---------------------------------------------------------------------------

def load_all_centers(
    registry_module=None,
    center_paths: Optional[dict[int, str | Path]] = None,
    time_col: Optional[str] = None,
    event_col: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
) -> tuple[list[dict], list[str]]:
    """
    Load all centers and validate that they share the same feature set.

    Pass either a `registry_module` (imports CENTER_PATHS, TIME_COL, etc.)
    or the individual keyword arguments.

    Returns
    -------
    centers      : list of dicts, one per center (output of `load_center`)
    feature_names: list[str] — shared feature names in canonical order
    """
    if registry_module is not None:
        center_paths = registry_module.CENTER_PATHS
        time_col     = registry_module.TIME_COL
        event_col    = registry_module.EVENT_COL
        if feature_cols is None:
            feature_cols = registry_module.FEATURE_COLS

    if center_paths is None or time_col is None or event_col is None:
        raise ValueError("Provide either registry_module or (center_paths, time_col, event_col)")

    centers = []
    for cid, path in sorted(center_paths.items()):
        c = load_center(path, center_id=cid, time_col=time_col,
                        event_col=event_col, feature_cols=feature_cols)
        centers.append(c)

    # Validate feature alignment
    ref_features = centers[0]["feature_names"]
    for c in centers[1:]:
        if c["feature_names"] != ref_features:
            raise ValueError(
                f"Feature mismatch between center {centers[0]['center_id']} "
                f"and center {c['center_id']}.\n"
                f"Expected: {ref_features}\nGot: {c['feature_names']}"
            )

    return centers, ref_features
