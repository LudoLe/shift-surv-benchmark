"""
Scenario interface — Section 5 / Table 1 of the paper.

Each scenario is defined by which subset of Θ_d varies across domains
while the rest is held fixed.  The core scenarios are:

  no_shift              — nothing varies; control regime
  covariate_shift       — μ^(inv), μ^(spu), Σ vary; A and outcome mechanism fixed
  observational_shift   — covariate_shift + A_inv rotated in the target
  spurious_shift        — μ^(spu), Σ vary; v_d = 0 (spurious factors enter
                          covariates differently but never directly the risk)
  baseline_hazard_shift — λ_{0,d} varies; latent ranking preserved
  concept_shift         — w_d changes sign; conditional outcome mechanism shifts
  shape_shift           — Weibull shape ν varies (PH still holds)

Appendix scenarios: strong_covariate_shift, cure_shift.

Usage
-----
  datasets = generate_scenario("no_shift")
  # returns {"source": [DomainDataset, ...], "target": DomainDataset}
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .parameters import BaseParams, DomainParams, default_source_params, make_shared_loading_matrices
from .base_generator import SyntheticGenerator, DomainDataset
from .config import base_kwargs, load_config, scenario_kwargs


SCENARIO_NAMES = [
    "no_shift",
    "covariate_shift",
    "observational_shift",
    "spurious_shift",
    "baseline_hazard_shift",
    "shape_shift",
    "concept_shift",
    "prior_shift",
    "strong_covariate_shift",
    "cure_shift",
    "censoring_shift",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_scenario(
    name: str,
    n_source_domains: int = 3,
    n_per_domain: int = 500,
    seed: int = 42,
    base: Optional[BaseParams] = None,
    censoring_lambda: float = 0.0,
    **kwargs,
) -> dict:
    """
    Generate a multi-domain dataset for the named shift scenario.

    Returns
    -------
    dict with keys:
      "source"   : list of DomainDataset (labeled source centers)
      "target"   : DomainDataset         (unlabeled target center)
      "scenario" : str                   (scenario name)
      "base"     : BaseParams
      "shared_A" : np.ndarray            (shared loading matrix)
    """
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{name}'.  Choose from: {SCENARIO_NAMES}")

    # Config-declared values fill in whatever the caller did not pass explicitly;
    # explicit arguments always win.  See configs/generator.yaml.
    cfg = load_config()
    if base is None:
        base = BaseParams(**{**base_kwargs(cfg), "seed": seed})

    for key, value in scenario_kwargs(name, cfg).items():
        kwargs.setdefault(key, value)

    # Pure label/prior-shift target prevalence (used only by prior_shift).
    prevalence_target = kwargs.pop("prevalence_target", 0.60)

    shared_A_inv, shared_A_spu, shared_B_nl = make_shared_loading_matrices(base, seed=seed)
    gen = SyntheticGenerator(
        base,
        shared_A_inv=shared_A_inv,
        shared_A_spu=shared_A_spu,
        shared_B_nl=shared_B_nl,
    )

    source_params = default_source_params(
        base, shared_A_inv, shared_A_spu,
        n_domains=n_source_domains, n_per_domain=n_per_domain,
    )
    target_id = n_source_domains   # target domain index = K

    if name == "no_shift":
        source_p, target_p = _no_shift(source_params, target_id, n_per_domain)
    elif name == "covariate_shift":
        source_p, target_p = _covariate_shift(source_params, target_id, n_per_domain, base, seed, **kwargs)
    elif name == "observational_shift":
        source_p, target_p = _observational_shift(source_params, target_id, n_per_domain, base, seed, **kwargs)
    elif name == "spurious_shift":
        source_p, target_p = _spurious_shift(source_params, target_id, n_per_domain, base, seed, **kwargs)
    elif name == "baseline_hazard_shift":
        source_p, target_p = _baseline_hazard_shift(source_params, target_id, n_per_domain, **kwargs)
    elif name == "shape_shift":
        source_p, target_p = _shape_shift(source_params, target_id, n_per_domain, base, **kwargs)
    elif name == "concept_shift":
        source_p, target_p = _concept_shift(source_params, target_id, n_per_domain, base, **kwargs)
    elif name == "prior_shift":
        source_p, target_p = _prior_shift(source_params, target_id, n_per_domain)
    elif name == "strong_covariate_shift":
        source_p, target_p = _strong_covariate_shift(source_params, target_id, n_per_domain, base, seed, **kwargs)
    elif name == "censoring_shift":
        source_p, target_p = _censoring_shift(source_params, target_id, n_per_domain, **kwargs)
    elif name == "cure_shift":
        source_p, target_p = _cure_shift(source_params, target_id, n_per_domain, base, **kwargs)

    # Optional independent exponential censoring (C ~ Exp(censoring_lambda)).
    # The fixed-horizon benchmark leaves this at 0 (delta == 1 for all); the
    # survival validity check (experiments/run_survival.py) turns it on so a
    # Cox model is trained and scored on genuinely censored data.
    if censoring_lambda > 0.0:
        # The fixed-horizon label is y = 1[t_obs <= tau AND delta == 1], i.e.
        # "event observed by tau". Without censoring that is exactly 1[T <= tau],
        # the estimand the binary benchmark targets. With censoring it is not:
        # a patient censored before tau has an unknown Y(tau) and is silently
        # labelled 0, which biases the endpoint downward. Every binary
        # experiment therefore leaves this at 0 -- only run_survival.py turns it
        # on, and it fits a Cox model on (t_obs, delta) rather than using y.
        # Warned rather than blocked because the parameter is legitimate for
        # survival-side use; the point is that it must not reach the binary path.
        warnings.warn(
            f"censoring_lambda={censoring_lambda} is set. The fixed-horizon "
            "label y then means 'event observed by tau', NOT 1[T <= tau]: "
            "patients censored before tau are labelled 0 with unknown true "
            "status. Use survival-aware evaluation (see run_survival.py), not "
            "the binary endpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        for sp in source_p:
            sp.censoring_lambda = censoring_lambda
        target_p.censoring_lambda = censoring_lambda

    source_datasets = gen.generate_domains(source_p)
    if name == "prior_shift":
        target_dataset = _generate_prior_shift_target(
            gen, target_p, target_id, n_per_domain, prevalence_target, seed)
    else:
        target_dataset = gen.generate_domain(target_p, domain_id=target_id)

    return {
        "source": source_datasets,
        "target": target_dataset,
        "scenario": name,
        "base": base,
        "shared_A_inv": shared_A_inv,
        "shared_A_spu": shared_A_spu,
        "shared_B_nl":  shared_B_nl,
    }


# ---------------------------------------------------------------------------
# Scenario parameterisations
# ---------------------------------------------------------------------------

def _no_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
) -> tuple[list[DomainParams], DomainParams]:
    """
    No shift — all Θ_d are identical across source and target.
    Control regime: adaptation should not be necessary.
    """
    ref = source_params[0].copy()
    target = ref.copy()
    target.domain_id = target_id
    target.n = n
    return source_params, target


def _covariate_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    seed: int,
    mu_inv_shift_scale: float = 1.5,
    mu_spu_shift_scale: float = 1.5,
    sigma_noise_target: float | None = None,
    source_mu_scale: float = 0.3,
    source_sigma_base: float = 0.5,
    source_sigma_jitter: float = 0.1,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Pure covariate shift — only the latent means shift; the observation model
    (A_inv and the noise scale) and the outcome mechanism are unchanged.

    P(X) shifts rigidly (location shift in latent space), P(Y|u_inv) is
    identical across domains. Analogous to the two-moons problem: same
    decision boundary, different region of input space.

    mu_inv is shifted orthogonally to w so that mean(eta) is unchanged
    and prevalence is stable across source and target.

    ``sigma_noise_target`` defaults to the source noise level, matching
    ``_spurious_shift``. It previously defaulted to 0.8 against a source of
    ~0.5, which raised the observation-noise variance by 2.6x on top of the
    mean displacement. That made the regime a mixture of case-mix and
    measurement shift, and the two are not interchangeable for this benchmark's
    purpose: a mean displacement moves information around, so importance
    weighting and feature alignment can in principle undo it, whereas added
    observation noise destroys information that no unsupervised method can
    recover. Attributing the resulting unrecovered gap to "covariate shift"
    therefore admitted an alternative explanation the benchmark is meant to
    rule out. The noise increase belongs to ``_observational_shift``, which
    keeps sigma at 0.8 alongside the A_inv rotation.
    """
    rng = np.random.default_rng(seed + 10)

    for sp in source_params:
        sp.mu_inv = rng.standard_normal(base.k_inv) * source_mu_scale
        sp.mu_spu = rng.standard_normal(base.k_spu) * source_mu_scale
        sp.sigma_noise = source_sigma_base + rng.uniform(-source_sigma_jitter,
                                                          source_sigma_jitter)

    # Shift direction orthogonal to w → w^T mu_inv = 0 → stable prevalence
    w_src = source_params[0].w.copy()
    d = np.ones(base.k_inv, dtype=float)
    d -= (np.dot(d, w_src) / np.dot(w_src, w_src)) * w_src
    d /= np.linalg.norm(d) + 1e-8

    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    # Guarded, not unconditional: at scale 0 the assignment would set the target
    # means to zero while the sources keep mu ~ N(0,1)*source_mu_scale, so "no
    # shift" would still displace them. That is invisible at the benchmark's
    # scale of 1.5, but it is exactly the setting the twin uses for its identity
    # control and for its covariance-only variant, where it silently added a
    # mean shift of ~0.6 to a regime that was supposed to have none.
    if mu_inv_shift_scale:
        target.mu_inv = d * mu_inv_shift_scale
    if mu_spu_shift_scale:
        target.mu_spu = np.ones(base.k_spu) * mu_spu_shift_scale
    # sigma_noise and A_inv are deliberately NOT assigned: the target inherits
    # them from source_params[0].copy(), so Sigma is identical to the source's
    # realised value rather than merely close to it. Setting it to a constant
    # (even the source's nominal base) would leave a residual difference,
    # because the sources draw sigma with jitter.
    if sigma_noise_target is not None:
        target.sigma_noise = sigma_noise_target

    return source_params, target


def _observational_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    seed: int,
    mu_inv_shift_scale: float = 0.0,
    mu_spu_shift_scale: float = 0.0,
    sigma_noise_target: float | None = None,
    a_inv_rotation_angle: float = np.pi / 3,   # 60 degrees
    source_mu_scale: float = 0.3,
    source_sigma_base: float = 0.5,
    source_sigma_jitter: float = 0.1,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Measurement-induced observational shift — A_inv rotated in the target.

    The same underlying biology (u_inv, w, lambda0) is expressed differently
    across centers due to different measurement protocols (assays, lab
    pipelines). Formally: A_inv_target = A_inv_source @ R where R is a
    Givens rotation of 60° in the (0,1) column pair.

    A source-trained LR predicts w^T R u_inv instead of w^T u_inv.
    The correlation between the two is ρ = w^T R w / ||w||² ≈ 0.58 for
    w=[1, 0.5, -0.5] and 60°, giving a consistent AUROC gap of ~0.10-0.15.

    The invariant oracle is unaffected (uses u_inv directly, bypasses A).
    """
    rng = np.random.default_rng(seed + 10)

    for sp in source_params:
        sp.mu_inv = rng.standard_normal(base.k_inv) * source_mu_scale
        sp.mu_spu = rng.standard_normal(base.k_spu) * source_mu_scale
        sp.sigma_noise = source_sigma_base + rng.uniform(-source_sigma_jitter,
                                                          source_sigma_jitter)

    w_src = source_params[0].w.copy()
    d = np.ones(base.k_inv, dtype=float)
    d -= (np.dot(d, w_src) / np.dot(w_src, w_src)) * w_src
    d /= np.linalg.norm(d) + 1e-8

    c, s = np.cos(a_inv_rotation_angle), np.sin(a_inv_rotation_angle)
    R = np.eye(base.k_inv, dtype=float)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s,  c

    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    # Pure measurement shift: ONLY the observation map changes. The latent means
    # and the noise scale are inherited from source_params[0].copy(), so this
    # regime is not covariate shift with a rotation bolted on. R is an
    # orthogonal (invertible) rotation and sigma is unchanged, so no information
    # is destroyed -- the target-optimal rule stays recoverable from x in
    # principle, which is what makes the unrecovered gap attributable to the
    # methods rather than to the observation process.
    if mu_inv_shift_scale:
        target.mu_inv = d * mu_inv_shift_scale
    if mu_spu_shift_scale:
        target.mu_spu = np.ones(base.k_spu) * mu_spu_shift_scale
    if sigma_noise_target is not None:
        target.sigma_noise = sigma_noise_target
    target.A_inv = source_params[0].A_inv @ R

    return source_params, target


def _spurious_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    seed: int,
    rho_source: float = 0.70,
    spu_mean_target: float | None = None,
    sigma_noise_target: float | None = None,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Spurious shortcut shift.  v_d = 0 throughout (spurious factors never
    directly enter the risk), but within source domains u_inv and u_spu are
    correlated (latent_cov off-diagonal = ρ > 0).  This creates a genuine
    within-domain shortcut: models that see x can use the spurious component
    of x as a proxy for the invariant signal.

    In the target domain, (a) u_inv and u_spu are independent (ρ = 0), and
    (b) the mean of u_spu flips sign.  Both effects break the spurious
    shortcut, so spurious-reliant models degrade.  Invariant-only models are
    stable because w_d and the invariant distribution are unchanged.

    The within-domain cross-covariance is the standard mechanism for spurious
    correlation in the representation-learning literature (akin to Colored
    MNIST, where label and spurious feature are statistically associated
    within training domains but not at test time).
    """
    k = base.k_inv + base.k_spu

    def _build_latent_cov(rho: float) -> np.ndarray:
        """
        Block covariance for [u_inv; u_spu] with controlled cross-correlation ρ.

        Only the first k_min = min(k_inv, k_spu) invariant/spurious dimensions
        are paired (Cross-block[i, k_inv+i] = ρ for i < k_min); the rest stay
        independent.  Each paired (u_inv_i, u_spu_i) forms a 2×2 block
        [[1, ρ], [ρ, 1]] with eigenvalues 1±ρ.  So the spectrum is:

            1+ρ  (multiplicity k_min)
            1−ρ  (multiplicity k_min)
            1    (multiplicity k_inv + k_spu − 2·k_min, the unpaired dims)

        With the defaults k_inv=3, k_spu=2 → k_min=2: eigenvalues 1±ρ each with
        multiplicity 2, plus one unpaired invariant dim with eigenvalue 1.
        This is PD for |ρ| < 1 — no PSD correction required.

        Previous design (all-ones cross-block) was only PD for ρ < 1/k_inv,
        so the eigenvalue-clipping fix silently reduced the achieved correlation
        to ~0.44 regardless of the requested ρ value.
        """
        k_min = min(base.k_inv, base.k_spu)
        C = np.eye(k)
        for i in range(k_min):
            C[i, base.k_inv + i] = rho
            C[base.k_inv + i, i] = rho
        return C

    cov_src = _build_latent_cov(rho_source)  # u_inv ↔ u_spu correlated in source
    cov_tgt = np.eye(k)                       # independent in target

    for i, sp in enumerate(source_params):
        sp.v = np.zeros(base.k_spu)       # spurious factors NEVER directly enter risk
        sp.mu_spu = np.zeros(base.k_spu)  # mean unchanged (μ^spu fixed across domains)
        sp.latent_cov = cov_src            # within-domain shortcut structure

    # Target: independent latent factors, reversed spurious mean
    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    target.v = np.zeros(base.k_spu)
    target.latent_cov = cov_tgt   # ONLY the cross-covariance changes
    # mu_spu is inherited (zeros, as set on the sources above): moving it as
    # well would make this a compound spurious + covariate shift, which is a
    # different regime and should be named as one.
    if spu_mean_target is not None:
        target.mu_spu = np.full(base.k_spu, spu_mean_target)
    if sigma_noise_target is not None:
        target.sigma_noise = sigma_noise_target

    return source_params, target


def _baseline_hazard_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    lambda0_target: float = 0.06,
    nu_target: float = 1.5,
    beta0_target: float = 0.0,
    source_lambda0_base: float = 0.020,
    source_lambda0_step: float = 0.002,
    source_nu: float = 1.5,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Baseline hazard shift — λ_{0,d} varies; β_{0,d}=0 and ν_d held fixed.
    μ^(inv), μ^(spu), A, Σ, w, v all held fixed.

    The latent risk ranking is preserved (same w, same factor distributions),
    but the absolute probability of event by τ changes because the baseline
    rate is elevated in the target (~3× higher than source mean).

    With the default lambda0=0.02 in DomainParams, source prevalence is ~0.33.
    The target lambda0=0.06 raises prevalence to ~0.58, a shift of ~0.25.
    This is a pure rate shift: β₀=0 everywhere, ν=1.5 everywhere.

    The shift is deliberately isolated to λ₀ only — no intercept change —
    so it is cleanly interpretable as a "higher-risk population" scenario
    where the ranking mechanism is identical but the absolute hazard differs.
    """
    # Sources: mild variation in baseline rate (~±10% around 0.02)
    for i, sp in enumerate(source_params):
        sp.beta0 = 0.0
        sp.lambda0 = source_lambda0_base + i * source_lambda0_step   # 0.020, 0.022, 0.024
        sp.nu = source_nu

    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    # Target: 3× higher baseline rate → prevalence ~0.58 vs source ~0.33
    target.beta0 = beta0_target
    target.lambda0 = lambda0_target
    target.nu = nu_target
    # Covariate distribution and risk coefficients unchanged:
    # mu_inv, mu_spu, A, sigma_noise, w, v all copied from source[0]

    return source_params, target


def _shape_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    nu_target: float = 0.7,
    source_nu: float = 1.5,
    source_lambda0: float = 0.02,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Weibull shape shift — nu varies, lambda0 and everything else fixed.

    nu controls the shape of the hazard over time:
      nu > 1 : increasing hazard (source default: 1.5)
      nu = 1 : constant hazard (exponential)
      nu < 1 : decreasing hazard (target default: 0.7)

    With nu_target=0.7 the target hazard is decreasing — patients who survive
    the early period face a lower risk later. This is qualitatively different
    from the source (increasing hazard) and cannot be corrected by any method
    that only aligns P(X) or adjusts the intercept.

    The latent ranking w^T u_inv is preserved, but the mapping from risk score
    to event probability at horizon tau changes non-linearly. A source-trained
    model uses the wrong shape assumption, miscalibrating predictions even when
    discrimination (AUROC) is partially preserved.

    lambda0 is adjusted so that source and target have similar prevalence:
    same mean eta, same tau, but different nu → adjust lambda0 to compensate.
    """
    for sp in source_params:
        sp.nu = source_nu
        sp.lambda0 = source_lambda0

    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    target.nu = nu_target
    # Adjust lambda0 so prevalence stays roughly comparable:
    # P(T<=tau) = 1 - exp(-lambda0 * tau^nu * exp(mean_eta))
    # mean_eta ~ 0 for zero-mean latents → exp(mean_eta) ~ 1
    # Solve: lambda0_tgt * tau^nu_tgt = lambda0_src * tau^nu_src
    tau = base.tau
    target.lambda0 = source_lambda0 * (tau ** source_nu) / (tau ** nu_target)

    return source_params, target


def _concept_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    w_flip_indices: tuple = (1, 2),
) -> tuple[list[DomainParams], DomainParams]:
    """
    Partial concept shift — a subset of invariant risk coefficients changes sign.

    Only the components listed in w_flip_indices are negated in the target;
    the remaining components are unchanged.  With the default source weights
    w = [1.0, 0.5, -0.5] and w_flip_indices=(1, 2), the target weights become
    [1.0, -0.5, 0.5].

    The cosine similarity between source and target w is ρ = 0.5/1.5 = 1/3,
    giving a theoretical oracle AUROC of ≈ 0.61 on the target — a clear but
    non-catastrophic transport failure.  The source model degrades substantially
    (gap > 0.25) while the target task remains informative and non-inverted.

    Interpretation: the dominant risk factor (index 0) is the same across
    centers; the weaker subsidiary factors reverse their roles.  This is more
    realistic than a full sign-flip (ρ = −1, AUROC → 0) and easier to diagnose.
    """
    ref_w = source_params[0].w.copy()

    # Sources share the same w (invariant among source domains only)
    for sp in source_params:
        sp.w = ref_w.copy()
        sp.v = np.zeros(base.k_spu)

    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    # Target: negate only the selected components of w
    w_tgt = ref_w.copy()
    for i in w_flip_indices:
        if i < len(w_tgt):
            w_tgt[i] = -w_tgt[i]
    target.w = w_tgt
    target.v = np.zeros(base.k_spu)

    return source_params, target


def _prior_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Pure label / prior shift (P(Y) changes, P(X|Y) invariant).

    The target uses exactly the same data-generating process as the source, so
    the class-conditional feature distribution P(X|Y) is identical; only the
    class prevalence P(Y) is changed, by class-conditional resampling in
    ``_generate_prior_shift_target``.  This is the dual of covariate shift
    (which changes P(X) with P(Y|X) fixed).

    It is the assumption class of Saerens/EM prior adjustment, so it serves as a
    *positive control* for calibration recovery: unsupervised prior adjustment
    should recover calibration here, and its failure under baseline-hazard,
    measurement, or concept shift is then attributable to a violated assumption
    rather than to a broken method.
    """
    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    return source_params, target


def _generate_prior_shift_target(
    gen: SyntheticGenerator,
    target_p: DomainParams,
    target_id: int,
    n: int,
    pi_target: float,
    seed: int,
) -> DomainDataset:
    """Generate a source-identical target pool, then resample within each class to
    the target prevalence ``pi_target`` -- P(X|Y) invariant, P(Y) shifted."""
    pool_p = target_p.copy()
    pool_p.n = max(n * 6, 3000)
    pool = gen.generate_domain(pool_p, domain_id=target_id,
                               seed=gen.base.seed + target_id * 1000 + 555)

    rng = np.random.default_rng((seed + 777) & 0xFFFFFFFF)
    pos = np.where(pool.y == 1)[0]
    neg = np.where(pool.y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:   # degenerate pool -> fall back to no resampling
        return gen.generate_domain(target_p, domain_id=target_id)

    n_pos = int(round(n * pi_target))
    n_neg = n - n_pos
    sel = np.concatenate([
        rng.choice(pos, size=n_pos, replace=len(pos) < n_pos),
        rng.choice(neg, size=n_neg, replace=len(neg) < n_neg),
    ])
    rng.shuffle(sel)

    def _pick(a):
        return a[sel] if a is not None else None

    return DomainDataset(
        domain_id=target_id,
        u_inv=pool.u_inv[sel], u_spu=pool.u_spu[sel],
        x=pool.x[sel], eta=pool.eta[sel], T=pool.T[sel], y=pool.y[sel],
        params=pool.params,
        C=_pick(pool.C), t_obs=_pick(pool.t_obs), delta=_pick(pool.delta),
    )


def _strong_covariate_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    seed: int,
    spu_mean_target: float = 3.0,
    sigma_noise_target: float = 1.5,
    a_spu_rotation_angle: float = np.pi / 2,   # 90 degrees
    source_mu_spu_scale: float = 0.2,
    source_sigma_noise: float = 0.5,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Strong observational covariate shift — positive control for DA methods.

    This scenario is designed so that p(x) shifts dramatically between source
    and target while p(y | u_inv) remains identical.  It serves as a sanity
    check: DA methods that can align p(x) should reduce the AUROC gap here.

    Three simultaneous shifts in x-space (all v_d = 0, so y is unaffected):

    1. Large mean shift of u_spu (mu_spu_tgt = spu_mean_target * 1).
       The A_spu columns project this shift into a large constant offset in x,
       making source and target means in x visibly different.

    2. 90° rotation of A_spu in the target.
       A_spu_tgt = A_spu_src @ R_90.  Since k_spu = 2 this is a full
       quarter-turn of the spurious loading plane.  The covariance of the
       A_spu contribution to x flips completely (eigenspaces exchange).
       CORAL detects this as a second-order distributional difference.

    3. Moderate increase in observation noise (sigma 0.5 → 1.5).
       This reduces the signal-to-noise ratio in x, degrading ERM performance.
       Note: irreducible noise cannot be recovered by any DA method — this
       gives the 'floor' below which no method can go.

    Why ERM degrades
    ----------------
    A source model trained with sigma=0.5 and small spurious contribution sees
    a much noisier, spurious-dominated x on target.  The decision boundary is
    placed with the wrong confidence (wrong noise assumption), and the large
    mean offset in x (A_spu @ mu_spu_tgt) is invisible to a source-trained LR.

    What DA methods can fix
    -----------------------
    CORAL: aligns the second-order statistics (covariance of x changes due to
           A_spu rotation and sigma increase).
    DANN : aligns the full marginal distribution.  The large mean shift and
           covariance change are easily detectable by the discriminator, giving
           a strong adversarial training signal.  The gradient reversal should
           guide the encoder to ignore A_spu directions and the noise offset.

    What no method can fix
    ----------------------
    The sigma increase is irreducible — the signal-to-noise ratio in u_inv is
    lower on target and no unsupervised alignment can recover it fully.
    The invariant oracle (uses u_inv directly, bypasses x noise) sets the ceiling.

    Expected result
    ---------------
    invariant_oracle: stable (uses u_inv, unaffected by x-space changes)
    ERM (all_observed): clear AUROC gap (SNR drop + distributional mismatch)
    CORAL: reduces gap (aligns covariance change from A_spu rotation + sigma)
    DANN:  reduces gap further (aligns full distribution including mean shift)
    Gap remaining after DA: reflects irreducible SNR loss from sigma increase
    """
    rng = np.random.default_rng(seed + 99)

    # Source domains: standard parameters, small inter-center noise variation
    for sp in source_params:
        sp.v = np.zeros(base.k_spu)
        sp.mu_spu = rng.standard_normal(base.k_spu) * source_mu_spu_scale
        sp.sigma_noise = source_sigma_noise

    # 90° rotation matrix in the k_spu = 2 dimensional spurious loading space
    c, s = np.cos(a_spu_rotation_angle), np.sin(a_spu_rotation_angle)
    R_spu = np.array([[c, -s], [s, c]], dtype=float)   # (k_spu, k_spu)

    # Target: rotate A_spu, shift mu_spu, increase noise; keep A_inv and w intact
    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    target.v         = np.zeros(base.k_spu)
    target.mu_spu    = np.full(base.k_spu, spu_mean_target)
    target.sigma_noise = sigma_noise_target
    # A_spu_tgt = A_spu_src @ R_90  (rotate in loading space, not in x-space)
    A_spu_src        = source_params[0].A_spu      # (p, k_spu)
    target.A_spu     = A_spu_src @ R_spu           # (p, k_spu): rotated

    return source_params, target


# ---------------------------------------------------------------------------
# Mixture cure shift
# ---------------------------------------------------------------------------


def _censoring_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    censoring_lambda_target: float = 0.06,
    censoring_beta_target: float = 0.8,
) -> tuple[list[DomainParams], DomainParams]:
    """Outcome-ascertainment shift: the target has shorter, risk-dependent follow-up.

    Every other regime perturbs how patients or their covariates differ. This one
    perturbs how the *outcome is observed*, which no other regime covers: the
    benchmark has a measurement shift on x and nothing on y.

    The true label 1[T <= tau] is unchanged in distribution -- the source and
    target populations are identical and the outcome mechanism is untouched.
    What changes is the recorded label y = 1[t_obs <= tau AND delta == 1], and
    which patients have a known status at all.

    This is motivated by the real cohort rather than added for completeness. In
    I3LUNG median follow-up runs from 7.7 to 12.9 months across the six centres,
    and the fraction of patients whose six-month status is unobserved -- and who
    are therefore dropped from the analysis -- ranges from 0% to 17%. Centres are
    thus compared on differently selected subsets.

    censoring_beta_target makes the censoring rate depend on the latent risk,
    rate_i = lambda_c * exp(beta * eta_i). At beta = 0 censoring is
    administrative and the exclusion costs only sample size; at beta > 0 the
    sicker are lost first and the exclusion selects on the outcome. That
    distinction is the clinically interesting one and is not identifiable from
    the recorded data, which is why it needs a simulator.

    The source is left uncensored so the regime isolates the target's
    ascertainment; a symmetric variant with both centres censored at different
    rates is more realistic but confounds the source model's own labels.
    """
    target = source_params[0].copy()
    target.domain_id = target_id
    target.n = n
    target.censoring_lambda = censoring_lambda_target
    target.censoring_beta = censoring_beta_target
    return source_params, target


def _cure_shift(
    source_params: list[DomainParams],
    target_id: int,
    n: int,
    base: BaseParams,
    cure_fraction_target: float = 0.25,
    cure_fraction_source: float = 0.0,
    w_cure_scale: float = 0.5,
) -> tuple[list[DomainParams], DomainParams]:
    """
    Mixture cure shift — non-proportional hazard via a cure fraction.

    Motivation (IO context)
    -----------------------
    In immunotherapy trials a subset of patients achieve durable responses
    and are effectively 'cured' (no event within any practical follow-up).
    Standard Weibull PH forces all survival curves to converge to 0 and
    cannot capture this plateau.  The mixture cure model is:

        S(t | X) = π_i + (1 − π_i) · exp(−λ_0 t^ν exp(η_i))

    where π_i ∈ (0,1) is the patient-specific cure probability.

    Design
    ------
    Source domains: standard Weibull PH (cure_fraction_source = 0.0 by
                    default). Set cure_fraction_source > 0 for a source-
                    side cure fraction too.
    Target domain:  mixture cure model with cure_fraction_target (default
                    0.25, i.e. 25% long-term survivors).

    Patient-level cure probability in the target:
        logit(π_i) = logit(π̄) + u_inv_i @ w_cure
    where w_cure is a small perturbation of the shared risk weights w,
    scaled by w_cure_scale, so that the same latent factors that drive
    prognosis also modulate curability (biologically motivated).

    Non-proportionality
    -------------------
    The hazard ratio between source and target individuals is no longer
    constant: it starts at the Weibull PH value and decays toward 1 as
    t → ∞ (cured target patients 'pull' the hazard to zero).
    Kaplan-Meier curves plateau instead of reaching zero — a signature
    visible in real IO data.

    Impact on adaptation methods
    ----------------------------
    LR ERM    : AUROC degraded because the fixed-horizon label y = 1[T≤τ]
                underestimates target prevalence (cured patients inflate y=0).
                The decision boundary learned on source miscalibrates.
    CORAL/IRM : partially compensate for covariate distribution shift but
                cannot recover the cure-induced label mechanism shift.
    DANN      : aligns P(x) but the label noise from the cure mechanism is
                not covariate-driven → adversarial alignment cannot fix it.
    Invariant oracle: unaffected by cure mechanism when u_inv is used.
    """
    for sp in source_params:
        sp.cure_fraction = cure_fraction_source
        sp.w_cure        = None   # homogeneous when source has cure fraction

    target = source_params[0].copy()
    target.domain_id     = target_id
    target.n             = n
    target.cure_fraction = cure_fraction_target

    # w_cure: small vector aligned with w so that high-risk patients are
    # less likely to be cured (negative correlation between η and π)
    w_norm = target.w / (np.linalg.norm(target.w) + 1e-8)
    target.w_cure = -w_cure_scale * w_norm   # high η → lower logit(π) → lower cure prob

    return source_params, target
