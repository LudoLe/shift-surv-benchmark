"""
One-time data preparation script.

Reads the raw source files, joins covariates with outcomes, selects the
hypothesis-driven RWD feature set, and writes one CSV per center to data/raw/.

Run from the project root:
    python -m data.prepare_data

Input files (edit paths below if needed):
    RWD_PATH      : patient-level covariates with CENTER and SET columns
    OUTCOMES_PATH : OS MONTHS and DEATH EVENT per patient
    FEATURES_PATH : features.json — selects the "RWD" feature list

Output:
    data/raw/center_<NAME>.csv   — one file per center, with NaN preserved
    (imputation happens at LOODO time, fitted on source split only)
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------
RWD_PATH      = Path("/Users/ludole/Downloads/rwd.csv")
OUTCOMES_PATH = Path("/Users/ludole/Downloads/outcomes.csv")
FEATURES_PATH = Path("/Users/ludole/Downloads/features.json")

OUT_DIR = Path(__file__).parent / "raw"

# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
SUBJECT_COL = "Subject"
CENTER_COL  = "CENTER"
TIME_COL    = "OS MONTHS"
EVENT_COL   = "DEATH EVENT"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    rwd  = pd.read_csv(RWD_PATH)
    out  = pd.read_csv(OUTCOMES_PATH)
    with open(FEATURES_PATH) as f:
        feature_groups = json.load(f)
    feature_cols: list[str] = feature_groups["RWD"]

    # Join
    df = rwd.merge(
        out[[SUBJECT_COL, EVENT_COL, TIME_COL]],
        on=SUBJECT_COL, how="inner"
    )

    # Validate columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns not found in rwd.csv: {missing_cols}")

    keep_cols = [CENTER_COL, TIME_COL, EVENT_COL] + feature_cols
    df = df[keep_cols].copy()

    # Sanity checks
    assert df[TIME_COL].min() >= 0, "Negative OS MONTHS found"
    assert set(df[EVENT_COL].dropna().unique()).issubset({0, 1}), \
        f"DEATH EVENT must be 0/1, got: {df[EVENT_COL].unique()}"

    # Write per-center CSVs (NaN preserved — imputation done at LOODO time)
    centers = sorted(df[CENTER_COL].unique())
    for center in centers:
        sub = df[df[CENTER_COL] == center].drop(columns=[CENTER_COL])
        out_path = OUT_DIR / f"center_{center}.csv"
        sub.to_csv(out_path, index=False)
        n_events = int(sub[EVENT_COL].sum())
        print(
            f"  {center:6s}  n={len(sub):4d}  events={n_events:4d} "
            f"({n_events/len(sub)*100:.1f}%)  → {out_path}"
        )

    print(f"\nDone. {len(centers)} centers written to {OUT_DIR}/")
    print("Feature columns:", feature_cols)


if __name__ == "__main__":
    main()
