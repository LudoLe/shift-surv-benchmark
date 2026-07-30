"""Summarise the per-seed bootstrap: variance split, and what changes vs the ensemble.

Exists so the analysis is a `git pull` away instead of a multi-line heredoc pasted
into a terminal, which is fragile enough to have eaten several attempts.

Reads, for every transfer, the pair
    outputs/tables/<prefix>_bootstrap.csv           (per-seed: cohort + training)
    outputs/tables/<prefix>_bootstrap_ensemble.csv  (seeds averaged out: cohort only)

Run from the repo root:
    python -m experiments.summarize_perseed
    python -m experiments.summarize_perseed --prefix i3lung_loco
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "outputs", "tables")


def load(path: str) -> dict[str, dict[str, str]]:
    with open(path) as fh:
        return {r["method"]: r for r in csv.DictReader(fh)}


def f(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def significant(r: dict) -> bool:
    """CI excludes zero (both bounds on the same side)."""
    lo, hi = f(r, "auroc_gap_lo"), f(r, "auroc_gap_hi")
    return lo == lo and hi == hi and lo * hi > 0


def gapfmt(r: dict) -> str:
    return f"{f(r,'auroc_gap_vs_ref'):+.3f}[{f(r,'auroc_gap_lo'):+.3f},{f(r,'auroc_gap_hi'):+.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="only transfers whose name starts with this")
    ap.add_argument("--tables", default=TABLES,
                    help="directory holding the *_bootstrap.csv files "
                         "(e.g. an outputs/_archive_* dir, to re-read an older run)")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.tables, "*_bootstrap.csv"))
                   if not p.endswith("_bootstrap_ensemble.csv")
                   and os.path.basename(p).startswith(args.prefix))
    if not paths:
        raise SystemExit(f"nessuna tabella *_bootstrap.csv in {args.tables} con prefisso '{args.prefix}'")

    decomp, flips, adaptive = [], [], []
    n_old = 0

    for p in paths:
        name = os.path.basename(p)[: -len("_bootstrap.csv")]
        new = load(p)
        if not new or "auroc_gap_sd_within" not in next(iter(new.values())):
            n_old += 1
            continue
        ref = next(iter(new.values())).get("reference", "lr_erm")

        ens_path = p.replace("_bootstrap.csv", "_bootstrap_ensemble.csv")
        old = load(ens_path) if os.path.exists(ens_path) else {}

        for m, r in new.items():
            if m == ref:
                continue
            w, b = f(r, "auroc_gap_sd_within"), f(r, "auroc_gap_sd_between")
            decomp.append((name, m, w, b, b / w if w > 1e-9 else float("nan")))

            if m in ("group_dro", "irm", "dann", "cdan", "coral", "iw_erm"):
                adaptive.append((name, m, f(r, "auroc"), gapfmt(r),
                                 f(r, "auroc_gap_p"), significant(r)))

            if m in old:
                so, sn = significant(old[m]), significant(r)
                if so or sn:
                    flips.append((name, m, gapfmt(old[m]), gapfmt(r), so, sn))

    if n_old:
        print(f"NOTA: {n_old} tabella/e priva/e delle colonne per-seed (run vecchio), ignorate.\n")

    print("=" * 96)
    print("METODI DI ADATTAMENTO — gap vs ERM (per-seed: coorte + addestramento)")
    print("=" * 96)
    print(f"{'transfer':<34}{'method':<12}{'AUROC':>8}{'gap vs ERM [95% CI]':>26}{'p':>8}  sig")
    for name, m, au, g, p, sig in adaptive:
        print(f"{name:<34}{m:<12}{au:>8.3f}{g:>26}{p:>8.3f}  {'SI' if sig else '--'}")

    print()
    print("=" * 96)
    print("VECCHIO (ensemble) vs NUOVO (per-seed) — solo dove almeno uno e' significativo")
    print("=" * 96)
    if not flips:
        print("  nessun caso significativo in nessuna delle due versioni")
    n_lost = n_gained = 0
    for name, m, go, gn, so, sn in flips:
        if so and not sn:
            tag, n_lost = "<-- PERDE significativita'", n_lost + 1
        elif sn and not so:
            tag, n_gained = "<-- nuovo significativo", n_gained + 1
        else:
            tag = "regge"
        print(f"{name:<34}{m:<12} ens {go:>24}   seed {gn:>24}  {tag}")
    print(f"\n  persi: {n_lost}   nuovi: {n_gained}   invariati: {len(flips)-n_lost-n_gained}")

    print()
    print("=" * 96)
    print("DECOMPOSIZIONE DELLA VARIANZA — ratio = between/within")
    print("  within  = ricampionamento pazienti, a training fisso")
    print("  between = riaddestramento, a coorte fissa (cio' che l'ensemble buttava via)")
    print("=" * 96)
    print(f"{'transfer':<34}{'method':<12}{'within':>9}{'between':>10}{'ratio':>8}")
    ranked = sorted(decomp, key=lambda t: (t[4] != t[4], -(t[4] if t[4] == t[4] else 0)))
    for name, m, w, b, ratio in ranked[:20]:
        rs = f"{ratio:>8.2f}" if ratio == ratio else f"{'n/a':>8}"
        print(f"{name:<34}{m:<12}{w:>9.4f}{b:>10.4f}{rs}")
    if len(ranked) > 20:
        print(f"  ... e altre {len(ranked)-20} righe")

    fin = sorted(t[4] for t in decomp if t[4] == t[4])
    if fin:
        print(f"\n  ratio mediano {fin[len(fin)//2]:.2f}   max {fin[-1]:.2f}   "
              f"sopra 1: {sum(x > 1 for x in fin)}/{len(fin)}")
        print("  (ratio n/a = metodo rank-preserving: gap AUROC identicamente zero, corretto)")


if __name__ == "__main__":
    main()
