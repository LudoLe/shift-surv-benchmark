"""Rebuild the real-cohort summary tables from the raw per-seed tables.

Fit results flow into three tables. The raw table keeps every row, the
bootstrap collects methods dynamically, and the summary aggregates over the
explicit _METHODS list. Isotonic calibration was added to the fits and reached
the first two, while _METHODS still lacked it, so a finished rerun produced
summaries without the isotonic row. The fits do not need repeating: this
rebuilds every summary from its raw table with the same aggregation function
the runner uses, and reports whether anything besides the isotonic row moved.

Usage
-----
  python -m experiments.rebuild_summaries
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.run_real_data import _aggregate_real  # noqa: E402
from utils.io import save_table_csv  # noqa: E402

_TABLES = _ROOT / "outputs" / "tables"


def main() -> None:
    raws = sorted(glob.glob(str(_TABLES / "i3lung_*_raw.csv"))) + \
           sorted(glob.glob(str(_TABLES / "public_*_raw.csv")))
    n_iso = n_changed = 0
    for rf in raws:
        prefix = os.path.basename(rf)[: -len("_raw.csv")]
        sf = _TABLES / f"{prefix}_summary.csv"
        rows = list(csv.DictReader(open(rf)))
        if not rows:
            print(f"  {prefix}: raw vuoto, salto")
            continue
        agg = _aggregate_real(rows)

        # Everything except the isotonic row must reproduce the summary the
        # runner wrote from the same raw results in memory.
        diffs = []
        if sf.exists():
            old = {r["method"]: r for r in csv.DictReader(open(sf))}
            for row in agg:
                m = row["method"]
                if m == "isotonic" or m not in old:
                    continue
                for k, v in row.items():
                    ov = old[m].get(k)
                    if ov is not None and str(v) != str(ov):
                        diffs.append(f"{m}.{k}: {ov} -> {v}")
        has_iso = any(r["method"] == "isotonic" for r in agg)
        n_iso += has_iso
        n_changed += bool(diffs)
        save_table_csv(agg, sf)
        flag = "isotonic OK" if has_iso else "senza isotonic (atteso per loco)"
        extra = f"   ATTENZIONE {len(diffs)} campi cambiati: {diffs[:3]}" if diffs else ""
        print(f"  {prefix}: {flag}{extra}")
    print(f"\n  {len(raws)} summary ricostruiti, {n_iso} con isotonic, "
          f"{n_changed} con differenze fuori da isotonic (atteso 0)")


if __name__ == "__main__":
    main()
