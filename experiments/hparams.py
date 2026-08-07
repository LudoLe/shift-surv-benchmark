"""Frozen hyperparameters, read from the selection rather than hardcoded.

``configs/hyperparameters_selected.yaml`` is written once by
experiments/tune.py on a disjoint seed block and is then fixed for every
benchmark, sensitivity and confirmation run. Reading from it here means the
value a method uses is the value the selection procedure chose, and that the
two cannot drift apart: a hardcoded default that no longer matches the frozen
file is the kind of discrepancy nobody notices until a reviewer recomputes
something.

Every accessor falls back to the value the benchmark used before tuning, so the
code still runs if the file is absent -- the same contract configs/generator.yaml
has with generator/config.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
_PATH = _ROOT / "configs" / "hyperparameters_selected.yaml"

# Pre-tuning values, kept so the module is self-contained and so the fallback is
# visible rather than implied.
_FALLBACK: dict[str, Any] = {
    "erm":       1.0,            # C
    "iw_erm":    (0.05, 20.0),   # clip
    "coral":     1.0,            # shrinkage
    "irm":       1e2,            # penalty weight
    "group_dro": 0.02,           # q_lr
}

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    sel: dict = {}
    if _PATH.exists():
        try:
            import yaml
            sel = (yaml.safe_load(_PATH.read_text()) or {}).get("selected", {}) or {}
        except ImportError:
            sel = {}
    _CACHE = sel
    return sel


def get(method: str, override=None):
    """Selected value for ``method``, unless the caller passed one explicitly.

    An explicit argument always wins, which is what lets the sensitivity sweep
    and the tuning run itself vary a parameter the frozen file also names.
    """
    if override is not None:
        return override
    entry = _load().get(method)
    if entry is None or "value" not in entry:
        return _FALLBACK[method]
    v = entry.get("value", _FALLBACK[method])
    return tuple(v) if isinstance(v, list) else v


def provenance() -> str:
    """One line naming where the values came from, for run logs."""
    return (f"hyperparameters from {_PATH.name}"
            if _PATH.exists() else "hyperparameters at pre-tuning defaults")
