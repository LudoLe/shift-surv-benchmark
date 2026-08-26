"""Merge the sharded sensitivity sweep and report the counts the paper quotes.

The sweep runs as one process per (mode, source count, regime, axis or world
slice) so that an interruption costs one shard rather than the run. This puts the
shards back together and prints the three things the paper states: how the
configurations divide across verdicts, how often each method improves
calibration in each regime, and which configurations still need confirmation.

Run from repo root:
    python -m experiments.collect_sweep
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter

REGS = ["covariate_shift", "observational_shift", "spurious_shift",
        "baseline_hazard_shift", "prior_shift", "concept_shift"]
METHODS = ["dann", "cdan", "platt", "prior", "iw_erm", "coral", "irm", "group_dro"]
TABLES = os.path.join("outputs", "tables")


def _merge(pattern: str, out: str) -> list[dict]:
    rows, cols = [], []
    for f in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(f)):
            rows.append(r)
            for c in r:
                if c not in cols:
                    cols.append(c)
    if not rows:
        return []
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return rows


def _verdicts(rows: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'regime':<24}{'verdetti':<52}{'gain max':>10}")
    for reg in REGS:
        sub = [r for r in rows if r.get("regime") == reg]
        if not sub:
            continue
        c = Counter(r.get("verdict", "?") for r in sub)
        hi = [float(r["adapt_gain_hi"]) for r in sub
              if r.get("adapt_gain_hi") not in (None, "", "nan")]
        v = " ".join(f"{k}={n}" for k, n in c.most_common())
        print(f"  {reg:<24}{v:<52}{max(hi) if hi else float('nan'):>10.4f}")
    tot = Counter(r.get("verdict", "?") for r in rows)
    print(f"  {'TOTALE':<24}{' '.join(f'{k}={n}' for k, n in tot.most_common())}")


def _calibration(rows: list[dict], label: str) -> None:
    print(f"\n  {label}   (intervallo che esclude lo zero)")
    for reg in REGS:
        sub = [r for r in rows if r.get("regime") == reg]
        if not sub:
            continue
        line = []
        for m in METHODS:
            up = dn = 0
            for r in sub:
                try:
                    lo, hi = float(r[f"d_{m}_lo"]), float(r[f"d_{m}_hi"])
                except (KeyError, ValueError):
                    continue
                if lo > 0:
                    up += 1
                elif hi < 0:
                    dn += 1
            if up or dn:
                line.append(f"{m} {up}/{dn}")
        if line:
            print(f"    {reg:<24}{'  '.join(line)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=TABLES)
    args = ap.parse_args()
    t = args.tables

    allrows = {}
    for ns in (1, 3):
        oat = _merge(os.path.join(t, f"oat_tuneall_{ns}src_*.csv"),
                     os.path.join(t, f"sens_tuneall_oat_{ns}src.csv"))
        wor = _merge(os.path.join(t, f"sens_tuneall_{ns}src_*_s*.csv"),
                     os.path.join(t, f"sens_tuneall_worlds_{ns}src.csv"))
        allrows[ns] = (oat, wor)
        print(f"\n{'='*78}")
        print(f"  {ns} SORGENTE/I — {len(oat)} configurazioni OAT + {len(wor)} worlds "
              f"= {len(oat)+len(wor)}")
        print('='*78)
        if oat:
            _verdicts(oat, "OAT — discriminazione")
        if wor:
            _verdicts(wor, "WORLDS — discriminazione")
        if wor:
            _calibration(wor, "WORLDS — calibrazione, migliora/peggiora")

    grand = sum(len(o) + len(w) for o, w in allrows.values())
    every = [r for o, w in allrows.values() for r in o + w]
    c = Counter(r.get("verdict", "?") for r in every)
    print(f"\n{'='*78}\n  TOTALE COMPLESSIVO: {grand} configurazioni\n{'='*78}")
    for k, n in c.most_common():
        print(f"    {k:<18}{n:>5}")
    br = [r for r in every if r.get("verdict") in ("BREAK", "candidate_break")]
    inv = sum(1 for r in br
              if r.get("erm_auroc") not in (None, "", "nan") and float(r["erm_auroc"]) < 0.5)
    print(f"\n  da portare in conferma: {len(br)}")
    print(f"    di cui con ERM sotto 0.5 (ordinamento invertito): {inv}")
    print(f"    con ERM sopra 0.5: {len(br) - inv}")


if __name__ == "__main__":
    main()
