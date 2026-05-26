"""
Registry of real multi-center clinical data sources.

Centers: GHD, INT, MH, SZMC, UOC, VHIO  (n=2075 total)
Feature set: hypothesis-driven RWD covariates from features.json
Outcome: overall survival (OS MONTHS, DEATH EVENT)

To regenerate the per-center CSVs from the raw source files:
    python -m data.prepare_data
"""

from __future__ import annotations
from pathlib import Path

_ROOT = Path(__file__).parent  # data/

# ---------------------------------------------------------------------------
# Paths — one CSV per center (written by data/prepare_data.py)
# ---------------------------------------------------------------------------

CENTER_PATHS: dict[int, Path] = {
    0: _ROOT / "raw" / "center_GHD.csv",
    1: _ROOT / "raw" / "center_INT.csv",
    2: _ROOT / "raw" / "center_MH.csv",
    3: _ROOT / "raw" / "center_SZMC.csv",
    4: _ROOT / "raw" / "center_UOC.csv",
    5: _ROOT / "raw" / "center_VHIO.csv",
}

CENTER_NAMES: dict[int, str] = {
    0: "GHD",
    1: "INT",
    2: "MH",
    3: "SZMC",
    4: "UOC",
    5: "VHIO",
}

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

TIME_COL:  str = "OS MONTHS"
EVENT_COL: str = "DEATH EVENT"

# 11 hypothesis-driven RWD covariates (from features.json, key "RWD")
FEATURE_COLS: list[str] = [
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
]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

CENTER_META: dict[int, dict] = {
    0: {"name": "GHD",  "n": 430,  "event_rate": 0.979, "median_OS_months": 7.7},
    1: {"name": "INT",  "n": 684,  "event_rate": 0.785, "median_OS_months": 9.6},
    2: {"name": "MH",   "n": 231,  "event_rate": 0.390, "median_OS_months": 11.8},
    3: {"name": "SZMC", "n": 273,  "event_rate": 0.392, "median_OS_months": 12.0},
    4: {"name": "UOC",  "n": 251,  "event_rate": 0.498, "median_OS_months": 12.9},
    5: {"name": "VHIO", "n": 206,  "event_rate": 0.850, "median_OS_months": 12.4},
}
