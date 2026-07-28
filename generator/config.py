"""
Single source of truth for generator parameters: ``configs/generator.yaml``.

Every structural constant (BaseParams), per-domain default (DomainParams) and
per-scenario shift parameter is declared in that file.  The Python defaults are
kept in place and are numerically identical, so the code still runs if the YAML
is absent (e.g. when the package is vendored elsewhere); the YAML is what a
reader edits and what the paper's reproducibility statement points at.

Resolution order for the config path:
  1. explicit ``path`` argument
  2. ``$SHIFT_SURV_CONFIG``
  3. ``<repo_root>/configs/generator.yaml``

Angles are written in DEGREES in the YAML (readability) and converted to the
radian kwarg names the scenario functions expect: ``<name>_deg`` -> ``<name>``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "generator.yaml")

# Cache keyed by resolved path so repeated scenario calls do not re-read the file.
_CACHE: dict[str, dict] = {}


def config_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("SHIFT_SURV_CONFIG") or DEFAULT_CONFIG_PATH


def load_config(path: Optional[str] = None, reload: bool = False) -> dict:
    """Return the parsed config, or ``{}`` when the file is unavailable."""
    resolved = config_path(path)
    if not reload and resolved in _CACHE:
        return _CACHE[resolved]

    cfg: dict = {}
    if os.path.exists(resolved):
        try:
            import yaml
        except ImportError:  # pyyaml not installed -> fall back to code defaults
            cfg = {}
        else:
            with open(resolved, "r") as fh:
                cfg = yaml.safe_load(fh) or {}

    _CACHE[resolved] = cfg
    return cfg


def base_kwargs(cfg: Optional[dict] = None) -> dict:
    """BaseParams fields declared in the config (``base:`` block)."""
    cfg = load_config() if cfg is None else cfg
    return dict(cfg.get("base") or {})


def domain_defaults(cfg: Optional[dict] = None) -> dict:
    """
    DomainParams fields declared in the config (``domain_defaults:`` block).

    List-valued fields (``w``, ``v``) are converted to arrays so callers can use
    them directly as DomainParams arguments.
    """
    cfg = load_config() if cfg is None else cfg
    out = dict(cfg.get("domain_defaults") or {})
    for key in ("w", "v"):
        if key in out and out[key] is not None:
            out[key] = np.asarray(out[key], dtype=float)
    return out


def scenario_kwargs(name: str, cfg: Optional[dict] = None) -> dict:
    """
    Shift parameters declared for ``name`` under ``scenarios:``.

    ``*_deg`` keys are converted to radians and renamed to the kwarg the
    scenario function actually takes (``a_inv_rotation_angle_deg`` ->
    ``a_inv_rotation_angle``).  Tuple-typed kwargs are normalised from the YAML
    lists.
    """
    cfg = load_config() if cfg is None else cfg
    raw = (cfg.get("scenarios") or {}).get(name) or {}

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith("_deg"):
            out[key[: -len("_deg")]] = np.deg2rad(float(value))
        elif key == "w_flip_indices":
            out[key] = tuple(int(i) for i in value)
        else:
            out[key] = value
    return out


def experiment_config(cfg: Optional[dict] = None) -> dict:
    """Protocol block (``experiment:``): seeds, sweeps, clinical margin."""
    cfg = load_config() if cfg is None else cfg
    return dict(cfg.get("experiment") or {})
