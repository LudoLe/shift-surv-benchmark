"""
Joint registry: 6 i3lung centres + MSK (Vanguri), restricted to 7 common features.

Seven common features
---------------------
  SEX, ECOG PS, NLR, SMOKING CURRENT, SMOKING FORMER, SMOKING NEVER, PDL1 CATEGORY

All centres share this feature set.
i3lung centres: GHD(0), INT(1), MH(2), SZMC(3), UOC(4), VHIO(5)
MSK (Vanguri):  id=6

Outcome: OS in months, tau=12 months.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from data import registry as i3lung_reg
from data.loader import load_center
from data import registry_vanguri as vanguri_reg

_ROOT = Path(__file__).parent

FEATURE_COLS: list[str] = [
    "SEX",
    "ECOG PS",
    "NLR",
    "SMOKING CURRENT",
    "SMOKING FORMER",
    "SMOKING NEVER",
    "PDL1 CATEGORY",
]

CENTER_NAMES: dict[int, str] = {
    0: "GHD",
    1: "INT",
    2: "MH",
    3: "SZMC",
    4: "UOC",
    5: "VHIO",
    6: "MSK",
}

TAU_GRID = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0]
TAU      = 12.0
TAU_REF  = 18.0


def load_all_centers(registry_module=None) -> tuple[list[dict], list[str]]:
    """
    Load all 7 centres (6 i3lung + MSK) with the 7 common features.

    Returns
    -------
    centers       : list of 7 centre dicts
    feature_names : FEATURE_COLS (7 features)
    """
    # ── i3lung centres (subset to 7 common features) ────────────────────────
    i3lung_centers = []
    for cid, path in sorted(i3lung_reg.CENTER_PATHS.items()):
        c = load_center(
            path=path,
            center_id=cid,
            time_col=i3lung_reg.TIME_COL,
            event_col=i3lung_reg.EVENT_COL,
            feature_cols=FEATURE_COLS,          # restrict to 7 features
        )
        i3lung_centers.append(c)

    # ── MSK (Vanguri) ────────────────────────────────────────────────────────
    vanguri_centers, _ = vanguri_reg.load_all_centers()

    return i3lung_centers + vanguri_centers, FEATURE_COLS
