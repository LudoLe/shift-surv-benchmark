"""
Save/load utilities for figures and summary tables.
"""
from __future__ import annotations
from pathlib import Path
import json
import csv


OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


def figures_dir(scenario: str) -> Path:
    p = FIGURES_DIR / scenario
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir(scenario: str) -> Path:
    p = TABLES_DIR / scenario
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_table_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
