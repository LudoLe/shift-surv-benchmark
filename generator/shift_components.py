"""
Elementary shift components for the domain-grounded generator.

Each function takes a fitted DomainParams and returns a modified copy with
one parameter perturbed.  Shifts are applied on top of the reference-fitted
params, so magnitude=baseline_value recovers the unperturbed generator.

Shift types
-----------
lambda_shift    : multiply lambda0 by a factor  (baseline factor = 1.0)
nu_shift        : replace nu with a new value   (baseline nu = fitted value)
mu_scale        : scale mu_inv by a scalar       (baseline scale = 1.0;
                  0.0 collapses composition to source mean)
noise_shift     : replace sigma_noise            (baseline = fitted value)
loading_rotation: rotate A_inv in the (0,1) column plane by angle in radians
                  (baseline angle = 0.0)

Grid definitions (SINGLETON_GRIDS) are the per-type magnitude grids used by
the singleton scan.  The baseline value is always included so the no-shift
case is scored as a reference point.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from generator.parameters import DomainParams


# ---------------------------------------------------------------------------
# Magnitude grids
# ---------------------------------------------------------------------------

SINGLETON_GRIDS: dict[str, list[float]] = {
    "lambda_shift":     [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
    "nu_shift":         [0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.8],
    "mu_scale":         [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "noise_shift":      [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0],
    "loading_rotation": [0.0, np.pi / 8, np.pi / 6, np.pi / 4,
                         np.pi / 3, np.pi / 2, 2 * np.pi / 3],
}

# Baseline magnitude for each shift type (= no-op value)
BASELINE_MAGNITUDE: dict[str, float] = {
    "lambda_shift":     1.0,
    "nu_shift":         1.5,   # updated per-domain if nu was fitted
    "mu_scale":         1.0,
    "noise_shift":      None,  # baseline = fitted sigma_noise (domain-specific)
    "loading_rotation": 0.0,
}


# ---------------------------------------------------------------------------
# Individual shift functions
# ---------------------------------------------------------------------------

def apply_lambda_shift(params: DomainParams, factor: float) -> DomainParams:
    """
    Multiply baseline hazard by `factor`.

    factor < 1 → lower event rate (fewer events by τ)
    factor > 1 → higher event rate (more events by τ)
    factor = 1 → no change (baseline)
    """
    p = params.copy()
    p.lambda0 = params.lambda0 * factor
    return p


def apply_nu_shift(params: DomainParams, nu: float) -> DomainParams:
    """
    Replace Weibull shape parameter.

    nu < 1 → decreasing hazard over time
    nu = 1 → constant hazard (exponential)
    nu > 1 → increasing hazard over time (default 1.5)
    """
    p = params.copy()
    p.nu = nu
    return p


def apply_mu_scale(params: DomainParams, scale: float) -> DomainParams:
    """
    Scale mu_inv by `scale`.

    scale = 0  → erase the composition shift (mu_inv → 0, i.e. source mean)
    scale = 1  → keep fitted mu_inv unchanged (baseline)
    scale > 1  → amplify the composition shift
    scale < 0  → invert the composition direction
    """
    p = params.copy()
    p.mu_inv = params.mu_inv * scale
    return p


def apply_noise_shift(params: DomainParams, sigma: float) -> DomainParams:
    """Replace observation noise standard deviation."""
    p = params.copy()
    p.sigma_noise = sigma
    return p


def apply_loading_rotation(
    params: DomainParams,
    angle: float,
    col_pair: tuple[int, int] = (0, 1),
) -> DomainParams:
    """
    Rotate A_inv by `angle` radians in the column subspace defined by `col_pair`.

    A_inv_new = A_inv @ R  where R is a Givens rotation in (col_pair[0], col_pair[1]).
    Column norms are preserved (R is orthogonal), so the loading magnitudes are
    unchanged — only the directions of the top-2 latent dimensions are rotated.
    """
    if params.A_inv is None:
        raise ValueError("A_inv must be set before applying a loading rotation")
    k = params.A_inv.shape[1]
    i, j = col_pair
    if i >= k or j >= k:
        raise ValueError(f"col_pair {col_pair} out of range for A_inv with k={k}")

    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(k, dtype=float)
    R[i, i], R[i, j] =  c, -s
    R[j, i], R[j, j] =  s,  c

    p = params.copy()
    p.A_inv = params.A_inv @ R
    return p


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

def apply_shift_config(
    params: DomainParams,
    config: dict,
) -> DomainParams:
    """
    Apply a dict of shift components to params.  Keys are shift-type strings,
    values are the corresponding magnitudes.  Shifts are applied in a fixed
    canonical order so combined configs are deterministic.

    Example
    -------
    apply_shift_config(params, {"lambda_shift": 2.0, "nu_shift": 1.8})
    """
    p = params
    # Canonical application order
    if "lambda_shift" in config:
        p = apply_lambda_shift(p, config["lambda_shift"])
    if "nu_shift" in config:
        p = apply_nu_shift(p, config["nu_shift"])
    if "mu_scale" in config:
        p = apply_mu_scale(p, config["mu_scale"])
    if "noise_shift" in config:
        p = apply_noise_shift(p, config["noise_shift"])
    if "loading_rotation" in config:
        p = apply_loading_rotation(p, config["loading_rotation"])
    return p
