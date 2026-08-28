"""Does CORAL win where the covariance gap is largest?

Across the thirty I3LUNG transfers and four adaptation methods, one interval
excludes zero in the favourable direction on discrimination: CORAL on
UOC to VHIO, at +0.028. Six such intervals are expected by chance across the
hundred and twenty comparisons, so the finding is within noise. It is still worth
asking whether it points at the mechanism CORAL corrects, because if it did, the
reading would be that the method works when its assumption holds and the single
result is the visible tip of that.

CORAL aligns second moments. Its objective is the squared Frobenius distance
between the source and target covariance matrices, and that distance is
computable without labels. If the win reflects the mechanism, UOC to VHIO should
sit near the top of a ranking by that distance. If it sits in the middle, the
result carries no information about the mechanism and the noise reading stands.

The mean distance is reported alongside, since a transfer can differ in location
without differing in shape.

Usage
-----
  python -m experiments.coral_covariance_check --data_dir $I3LUNG_DATA
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"


def _cov_dist(a: np.ndarray, b: np.ndarray) -> float:
    """CORAL's objective: squared Frobenius distance between covariances.

    Features are standardised by the source, which is the frame the method works
    in, so the number is the quantity CORAL drives towards zero rather than a
    generic distance.
    """
    sc = StandardScaler().fit(a)
    Ca = np.cov(sc.transform(a), rowvar=False)
    Cb = np.cov(sc.transform(b), rowvar=False)
    return float(np.linalg.norm(Ca - Cb, "fro") ** 2)


def _mean_dist(a: np.ndarray, b: np.ndarray) -> float:
    sc = StandardScaler().fit(a)
    return float(np.linalg.norm(sc.transform(a).mean(0) - sc.transform(b).mean(0)))


def run(data_dir: str, highlight: str = "UOC->VHIO") -> list[dict]:
    from experiments.run_real_data import load_i3lung_all

    centers, feats = load_i3lung_all(data_dir)
    names = sorted(centers)
    X_all = np.vstack([centers[c][0][feats].values.astype(float) for c in names])
    imp = SimpleImputer(strategy="median").fit(X_all)
    X = {c: imp.transform(centers[c][0][feats].values.astype(float)) for c in names}

    rows = [{"transfer": f"{s}->{t}", "source": s, "target": t,
             "cov_distance": round(_cov_dist(X[s], X[t]), 4),
             "mean_distance": round(_mean_dist(X[s], X[t]), 4)}
            for s, t in itertools.permutations(names, 2)]
    rows.sort(key=lambda r: -r["cov_distance"])
    for i, r in enumerate(rows, 1):
        r["cov_rank"] = i

    print(f"\n{'='*66}\n  DISTANZA DI COVARIANZA FRA CENTRI  ({len(rows)} transfer)\n{'='*66}")
    print(f"  {'transfer':<16}{'covarianza':>13}{'medie':>10}{'rango':>7}")
    for r in rows:
        mark = "   <--" if r["transfer"] == highlight else ""
        print(f"  {r['transfer']:<16}{r['cov_distance']:>13.2f}"
              f"{r['mean_distance']:>10.3f}{r['cov_rank']:>7}{mark}")

    hit = next((r for r in rows if r["transfer"] == highlight), None)
    if hit:
        print(f"\n  {highlight} e' al posto {hit['cov_rank']} su {len(rows)}.")
        print("  In alto: il risultato di CORAL sarebbe coerente con il meccanismo.")
        print("  A meta': non porta informazione sul meccanismo.")

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / "i3lung_covariance_distance.csv")
    print(f"\n  Scritto: {_TABLES}/i3lung_covariance_distance.csv\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--highlight", default="UOC->VHIO")
    a = ap.parse_args()
    run(a.data_dir, a.highlight)
