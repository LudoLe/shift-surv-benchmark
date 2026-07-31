"""
Domain-specific parameter tuples Θ_d and shared base configuration.

Notation follows Section 4 of the paper:
  Θ_d = (μ_d^inv, μ_d^spu, A_d^inv, A_d^spu, Σ_d, β_{0,d}, w_d, v_d, λ_{0,d}, ν_d)

Observation model
-----------------
Invariant and spurious latent factors are projected into observed covariate
space through *separate* loading matrices, with an optional tanh nonlinear
observation component:

    x_{i,d} = A_d^(inv) u_{i,d}^(inv)
             + A_d^(spu) u_{i,d}^(spu)
             + α · B_d^(nl) tanh(u_{i,d}^(inv))                 [if nl_strength > 0]
             + ε_{i,d}

  A_d^(inv)  : (p × k_inv) — maps invariant factors to observed features
  A_d^(spu)  : (p × k_spu) — maps spurious factors to observed features
  B_d^(nl)   : (p × k_inv) — maps tanh(invariant factors); column-normalised
  α          : scalar nonlinearity strength (BaseParams.nl_strength)
  ε_{i,d}    ~ N(0, σ_d² I_p) — observation noise

The nonlinear term introduces observation-level complexity while keeping the
survival mechanism (η, T, y) fully linear. B^(nl) is shared across domains by
default (same nonlinear loading everywhere); per-domain overrides are possible
via DomainParams.B_nl.

The tanh term is correlated with η = w^T u_inv (since Cor(tanh(u), u) ≈ 0.96
for u ~ N(0,1)), so it carries genuine invariant signal via a saturating pathway.
LR can only use the linear projection of tanh(u_inv) onto x; MLP can learn to
invert the saturation and recover better estimates of u_inv.  The improvement
is most pronounced in scenarios where u_inv deviates from zero.

Separating the three loadings makes the following structure explicit:
  - which observed features carry invariant prognostic signal (A^(inv))
  - which carry spurious / non-transportable structure (A^(spu))
  - which carry saturating observation complexity (B^(nl)) — invariant channel only
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BaseParams:
    """
    Shared structural parameters common across all domains.

    k_inv       : number of invariant latent factors
    k_spu       : number of spurious latent factors
    p           : dimension of observed covariate vector x
    tau         : fixed-horizon prediction window (months)
    seed        : global RNG seed
    nl_strength : strength α of the tanh nonlinear observation term
                  (x += α · B_nl tanh(u_inv)).
                  0.0 = fully linear observation model (original behaviour).
                  1.0 = tanh term variance ≈ linear term variance; creates a
                        genuine saturating observation pathway that LR can only
                        partially exploit (Cor(tanh(u),u) ≈ 0.96 but nonlinear).
    """
    k_inv: int = 3
    k_spu: int = 2
    p: int = 20
    tau: float = 6.0
    seed: int = 42
    nl_strength: float = 0.0


@dataclass
class DomainParams:
    """
    Per-domain parameter tuple Θ_d as defined in Section 4.

    mu_inv   : μ_d^(inv)  — mean of invariant latent factors        (k_inv,)
    mu_spu   : μ_d^(spu)  — mean of spurious latent factors         (k_spu,)
    A_inv    : A_d^(inv)  — invariant factor loading matrix         (p, k_inv)
               None → use the shared invariant loading matrix
    A_spu    : A_d^(spu)  — spurious factor loading matrix          (p, k_spu)
               None → use the shared spurious loading matrix
    sigma_noise : scalar noise std for Σ_d = sigma_noise² · I_p
    latent_cov  : optional joint covariance for [u_inv; u_spu] — enables
                  within-domain spurious correlation (spurious-shortcut setup).
                  None → identity (independent factors)
    beta0    : β_{0,d}    — domain-specific intercept in log-risk
    w        : w_d         — invariant factor risk coefficients       (k_inv,)
    v        : v_d         — spurious factor risk coefficients        (k_spu,)
               Zero vector means spurious factors do not enter the risk.
    lambda0  : λ_{0,d}    — Weibull baseline rate (scale parameter)
    nu       : ν_d         — Weibull shape parameter
    n        : number of samples to draw from this domain
    domain_id: integer domain index (assigned at generation time if -1)
    """
    mu_inv: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mu_spu: np.ndarray = field(default_factory=lambda: np.zeros(2))
    A_inv: Optional[np.ndarray] = None   # None → use shared invariant loading
    A_spu: Optional[np.ndarray] = None   # None → use shared spurious loading
    B_nl:  Optional[np.ndarray] = None   # None → use shared nonlinear loading
    sigma_noise: float = 0.5
    latent_cov: Optional[np.ndarray] = None  # None → identity (independent)
    beta0: float = 0.0
    w: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5, -0.5]))
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lambda0: float = 0.02
    nu: float = 1.5
    n: int = 500
    domain_id: int = -1
    censoring_lambda: float = 0.0   # Exp(λ_c) censoring rate; 0 = no censoring
    censoring_beta: float = 0.0     # risk dependence of censoring: rate_i = λ_c·exp(β·η_i).
                                    # 0 = administrative (independent of risk); >0 = the
                                    # sicker are lost to follow-up first, which is what makes
                                    # excluding unknown-status patients a biased selection
                                    # rather than a reduction in sample size.
    # ── Mixture cure model (optional) ──────────────────────────────────────
    # cure_fraction : scalar π̄ ∈ [0, 1) — average probability of being a
    #                 long-term survivor.  0.0 = standard Weibull PH (no cure).
    # w_cure        : (k_inv,) — patient-specific logit adjustment.
    #   logit(π_i) = logit(π̄) + u_inv_i @ w_cure
    #   None → homogeneous cure fraction (all patients share π̄).
    # Non-proportionality arises because S(t|X) = π_i + (1-π_i)·S_W(t|η_i)
    # makes the hazard ratio time-dependent: it converges to 1 as t→∞ for
    # cured patients and diverges between domains with different cure fractions.
    cure_fraction: float = 0.0
    w_cure: Optional[np.ndarray] = None
    # ── Time-to-event distribution ──────────────────────────────────────────
    # "weibull"     : Weibull PH (default).  Hazard h(t) = λ₀ν t^(ν-1) exp(η).
    #                 Proportional hazards; nu > 1 → increasing, nu < 1 → decreasing.
    # "loglogistic" : Log-logistic (AFT, non-PH).  Hazard rises then falls.
    #                 h(t) ∝ t^(ν-1)/(1+(λ₀ exp(η) t)^ν); peak at t* = ((ν-1)/λ₀exp(η))^(1/ν).
    #                 HR between two patients is not constant → non-proportional.
    #                 Inverse CDF: T = (U/(1-U))^(1/ν) / (λ₀·exp(η)).
    time_distribution: str = "weibull"

    def copy(self) -> "DomainParams":
        return DomainParams(
            mu_inv=self.mu_inv.copy(),
            mu_spu=self.mu_spu.copy(),
            A_inv=self.A_inv.copy() if self.A_inv is not None else None,
            A_spu=self.A_spu.copy() if self.A_spu is not None else None,
            B_nl=self.B_nl.copy() if self.B_nl is not None else None,
            sigma_noise=self.sigma_noise,
            latent_cov=self.latent_cov.copy() if self.latent_cov is not None else None,
            beta0=self.beta0,
            w=self.w.copy(),
            v=self.v.copy(),
            lambda0=self.lambda0,
            nu=self.nu,
            n=self.n,
            domain_id=self.domain_id,
            censoring_lambda=self.censoring_lambda,
            censoring_beta=self.censoring_beta,
            cure_fraction=self.cure_fraction,
            w_cure=self.w_cure.copy() if self.w_cure is not None else None,
            time_distribution=self.time_distribution,
        )


def make_shared_loading_matrices(
    base: BaseParams, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Construct the three fixed loading matrices for the observation model:

      A_inv : (p × k_inv) — invariant factor → observed feature projection
      A_spu : (p × k_spu) — spurious factor → observed feature projection
      B_nl  : (p × k_inv) — tanh(invariant factor) → observed feature
              projection (nonlinear term); None when base.nl_strength == 0.

    A_inv and A_spu are drawn from a single RNG stream (identical to the
    previous combined matrix — splitting at column k_inv gives the same
    numerical values).  B_nl is drawn from the same stream immediately
    after, ensuring deterministic reproducibility.

    All columns are normalised to unit L2 norm.
    """
    rng = np.random.default_rng(seed)
    k = base.k_inv + base.k_spu
    A = rng.standard_normal((base.p, k))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    A_inv = A[:, : base.k_inv]   # (p, k_inv)
    A_spu = A[:, base.k_inv :]   # (p, k_spu)

    if base.nl_strength > 0.0:
        B_nl = rng.standard_normal((base.p, base.k_inv))
        B_nl /= np.linalg.norm(B_nl, axis=0, keepdims=True)
    else:
        B_nl = None

    return A_inv, A_spu, B_nl


def default_source_params(
    base: BaseParams,
    A_inv: np.ndarray,
    A_spu: np.ndarray,
    n_domains: int = 3,
    n_per_domain: int = 500,
) -> list[DomainParams]:
    """
    Return a list of identical source domain parameters (no-shift baseline).
    All domains share the same Θ_d; shift scenarios override specific fields.

    Field values come from the ``domain_defaults`` block of
    configs/generator.yaml, falling back to the dataclass defaults below when
    the config is unavailable.
    """
    from .config import domain_defaults

    defaults = domain_defaults()

    default_w = np.asarray(defaults.get("w", [1.0, 0.5, -0.5]), dtype=float)[: base.k_inv]
    if len(default_w) < base.k_inv:
        default_w = np.pad(default_w, (0, base.k_inv - len(default_w)))

    params = []
    for d in range(n_domains):
        p = DomainParams(
            mu_inv=np.zeros(base.k_inv),
            mu_spu=np.zeros(base.k_spu),
            A_inv=A_inv.copy(),
            A_spu=A_spu.copy(),
            sigma_noise=defaults.get("sigma_noise", 0.5),
            beta0=defaults.get("beta0", 0.0),
            w=default_w.copy(),
            v=np.zeros(base.k_spu),
            lambda0=defaults.get("lambda0", 0.02),
            nu=defaults.get("nu", 1.5),
            n=n_per_domain,
            domain_id=d,
        )
        params.append(p)
    return params
