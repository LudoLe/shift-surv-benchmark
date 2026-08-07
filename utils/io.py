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
    """Write rows to CSV, taking the columns from the union of all their keys.

    Taking them from rows[0] alone loses a whole table when a later row carries
    an extra key: DictWriter tolerates missing keys (they come out blank) but
    raises on extra ones, and it raises inside writerows, after part of the file
    is already on disk. That is how a sweep shard ended up with eleven of its
    thirteen rows and a non-zero exit. Heterogeneous rows are normal here --
    a configuration that failed reports different fields from one that
    succeeded -- so the union is what the caller means.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows[1:]:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_table_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
