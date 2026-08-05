"""
Compare each synthetic twin with the real transfer it was matched to.

The twin contains only the two mechanisms measurable without target labels, so
it is the most favourable world alignment could face. The question this answers
is whether the real transfer falls below it, i.e. whether the measurable
mechanisms bound what adaptation actually achieves.

Run from repo root:  python -m experiments.twin_vs_real
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")

ERM = "lr_erm"
ADAPT = ["coral", "dann", "cdan", "iw_erm"]          # discrimination competitors
RANK_PRESERVING = {"prior_recalib", "platt_scaling", "isotonic"}


def real_gaps(bootreps_path):
    """method -> mean bootstrap gap, for AUROC and ECE."""
    auc, ece = defaultdict(list), defaultdict(list)
    with open(bootreps_path) as fh:
        for r in csv.DictReader(fh):
            auc[r["method"]].append(float(r["d_auroc"]))
            ece[r["method"]].append(float(r["d_ece"]))
    return ({m: float(np.mean(v)) for m, v in auc.items() if m != ERM},
            {m: float(np.mean(v)) for m, v in ece.items() if m != ERM})


def twin_gaps(twin_path):
    auc, ece = {}, {}
    with open(twin_path) as fh:
        for r in csv.DictReader(fh):
            auc[r["method"]] = float(r["twin_auroc_gap"])
            ece[r["method"]] = float(r["twin_ece_gap"])
    return auc, ece


def match(twin_path):
    """Locate the bootreps file of the real transfer a twin was built from."""
    tag = os.path.basename(twin_path)[len("twin_"):-len(".csv")]
    for cand in (f"i3lung_{tag}_bootreps.csv",
                 f"public_rotterdam_gbsg_{tag}_bootreps.csv"):
        p = os.path.join(TABLES, cand)
        if os.path.exists(p):
            return tag, p
    return tag, None


def main() -> None:
    rows = []
    for tp in sorted(glob.glob(os.path.join(TABLES, "twin_*.csv"))):
        tag, rp = match(tp)
        if rp is None:
            print(f"  ! nessun bootreps per {tag}")
            continue
        t_auc, t_ece = twin_gaps(tp)
        r_auc, r_ece = real_gaps(rp)
        shared = [m for m in ADAPT if m in t_auc and m in r_auc]
        if not shared:
            continue
        rows.append(dict(
            tag=tag,
            twin_best=max(t_auc[m] for m in shared),
            real_best=max(r_auc[m] for m in shared),
            twin_ece=max(t_ece[m] for m in t_ece),
            real_ece=max(r_ece[m] for m in r_ece if m in t_ece),
            per_method={m: (t_auc[m], r_auc[m]) for m in shared},
        ))

    print(f"{'transfer':<30}{'AUROC gemello':>15}{'AUROC reale':>13}{'gemello>reale':>15}")
    up = 0
    for r in rows:
        flag = "si" if r["twin_best"] >= r["real_best"] else "NO"
        up += flag == "si"
        print(f"{r['tag']:<30}{r['twin_best']:>+15.4f}{r['real_best']:>+13.4f}{flag:>15}")
    print(f"\n  il gemello fa almeno quanto il reale in {up}/{len(rows)} transfer")

    print(f"\n{'transfer':<30}{'ECE gemello':>15}{'ECE reale':>13}{'gemello>reale':>15}")
    upe = 0
    for r in rows:
        flag = "si" if r["twin_ece"] >= r["real_ece"] else "NO"
        upe += flag == "si"
        print(f"{r['tag']:<30}{r['twin_ece']:>+15.4f}{r['real_ece']:>+13.4f}{flag:>15}")
    print(f"\n  il gemello fa almeno quanto il reale in {upe}/{len(rows)} transfer")

    print("\nper metodo, guadagno AUROC (gemello / reale):")
    print(f"  {'transfer':<28}" + "".join(f"{m:>20}" for m in ADAPT))
    for r in rows:
        cells = "".join(
            f"{r['per_method'][m][0]:>+9.4f}/{r['per_method'][m][1]:>+9.4f}"
            if m in r["per_method"] else f"{'—':>20}" for m in ADAPT)
        print(f"  {r['tag']:<28}" + cells)


if __name__ == "__main__":
    main()
