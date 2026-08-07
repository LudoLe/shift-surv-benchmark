"""Per-run hyperparameter selection, by the same criterion the global tuning uses.

Why this exists
---------------
DANN and CDAN re-run an importance-weighted cross-validation search on every
(transfer, seed) they are fitted to. ERM, IW-ERM, CORAL, IRM and GroupDRO used
a single value selected once on synthetic data. The comparison was therefore
asymmetric in the direction that matters least for a positive claim and most
for a negative one: the paper says no method beats ERM, and the obvious question
is whether each method was given a fair chance. Answering "we searched hardest
for the two that lost worst" is the wrong shape of answer, however defensible
each individual choice was.

This module gives the other five the same treatment, under the same criterion,
so ``use_iwcv=True`` now means the same thing for every method.

What it does not change
-----------------------
The sensitivity sweep and the confirmation stage call every fit with
``use_iwcv=False`` and are unaffected: there, all methods run at the frozen
values and the comparison is already symmetric. Re-running them was not needed
and their results stand.

The criterion
-------------
Reused verbatim from experiments/tune.py rather than reimplemented, so the two
cannot drift apart:

  adaptation (erm, iw_erm, coral)
      importance-weighted AUROC on the held-out source validation split, with
      weights from a domain classifier trained on source-train against the
      unlabeled target covariates.

  domain generalization (irm, group_dro)
      leave-one-source-out AUROC across source centers; the target is not
      observed at all.

Target labels are never read. With a single source center the LOSO criterion is
undefined, and those two methods fall back to their frozen value -- with one
source GroupDRO reduces to ERM anyway.

Selection runs per (transfer, seed), which is what DANN and CDAN already do, so
the unit of selection is the same across the compared family.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

_GRIDS: dict | None = None
_WARNED: set[str] = set()


def _grids() -> dict:
    global _GRIDS
    if _GRIDS is not None:
        return _GRIDS
    from pathlib import Path
    import yaml
    path = Path(__file__).parent.parent / "configs" / "hyperparameters.yaml"
    _GRIDS = (yaml.safe_load(path.read_text()) or {}).get("grids", {}) or {}
    return _GRIDS


def candidates(method: str) -> tuple[str, list] | None:
    """(parameter name, grid) for a method, or None if it has no searchable grid."""
    from experiments.tune import _SCORERS
    if method not in _SCORERS:
        return None
    key = _SCORERS[method][0]
    grid = (_grids().get(method) or {}).get(key)
    if not grid:
        return None
    return key, list(grid)


def select(method: str, source_datasets: list, target_dataset, seed: int = 0) -> Any:
    """Choose one value for ``method`` on this transfer, or None to keep the frozen one.

    Returns None -- meaning "caller should use hparams.get(method)" -- whenever
    the criterion cannot be evaluated: no grid, fewer than two source centers
    for a DG method, or every candidate scoring NaN because the split is
    degenerate. Falling back silently would hide that; the caller keeps the
    frozen value, which is a defined quantity, and a note is printed once.
    """
    from experiments.tune import _SCORERS
    cg = candidates(method)
    if cg is None:
        return None
    _, grid = cg
    _key, scorer, n_src_needed = _SCORERS[method]

    if n_src_needed > 1 and len(source_datasets) < 2:
        if method not in _WARNED:
            _WARNED.add(method)
            print(f"  nota: {method} con una sola sorgente -- criterio LOSO non "
                  f"definito, resta al valore congelato")
        return None

    scores: list[float] = []
    for c in grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                v = scorer(c, source_datasets, target_dataset, seed)
            except Exception:
                v = float("nan")
        scores.append(float(v))

    if not np.isfinite(scores).any():
        if method not in _WARNED:
            _WARNED.add(method)
            print(f"  nota: {method} -- ogni candidato ha dato NaN, resta al "
                  f"valore congelato")
        return None
    # np.nanargmax over a list keeps ties resolved by grid order, which is the
    # same rule experiments/tune.py uses, so a tie selects the same value here
    # as it did in the global tuning.
    return grid[int(np.nanargmax(scores))]
