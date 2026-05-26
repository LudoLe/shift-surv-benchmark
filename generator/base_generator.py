"""
Core synthetic generator — Section 4 of the paper.

Data-generating process per individual i in domain d:

  u^(inv)_{i,d} ~ N(μ_d^(inv), I_{k_inv})
  u^(spu)_{i,d} ~ N(μ_d^(spu), I_{k_spu})

  x_{i,d} = A_d^(inv) u^(inv) + A_d^(spu) u^(spu)
            + α · B_d^(nl) tanh(u^(inv))            [if nl_strength > 0]
            + ε,   ε ~ N(0, σ² I_p)

  The two loading matrices are kept separate so the code makes explicit which
  part of x carries invariant prognostic signal (A^(inv)) and which carries
  spurious / non-transportable structure (A^(spu)).  Domain-specific shifts in
  how latent factors are observed can be induced through either matrix
  independently — useful for covariate-shift and spurious-shift scenarios.

  η_{i,d}  = β_{0,d} + w_d^T u^(inv) + v_d^T u^(spu) -  eta è un vettore di lunghezza n (uno scalare per paziente) che contiene il log-rischio latente di  
  ogni individuo nel dominio.

    β_{0,d} — intercetta del dominio. Uno scalare che sposta il rischio di tutti i pazienti del centro 
    d verso l'alto o verso il basso, indipendentemente dalle loro caratteristiche individuali. È il    
    "rischio basale medio" del centro. Cambia tra domini nel baseline_hazard_shift. 

    w_d^T u^(inv)_{i,d} — contributo dei fattori invarianti. w_d è un vettore di pesi (k_inv           
    componenti) che dice quanto conta ciascun fattore causale nel determinare il rischio. u^(inv)_{i,d}
    è il valore di quei fattori per il paziente i. Il prodotto scalare somma i contributi: fattori con
    peso positivo aumentano il rischio, negativi lo abbassano. Nel concept_shift è w_d a cambiare.

    v_d^T u^(spu)_{i,d} — contributo dei fattori spurious. Stessa struttura di prima, ma per i fattori 
    non causali. Nel generatore v_d = 0 per default ovunque: i fattori spurious non entrano mai in η.
    Questo è il punto chiave — esistono in x ma sono causalmente disconnessi dall'outcome. 

  U_{i,d}  ~ Uniform(0,1)
  T_{i,d}  = (- log U / (λ_{0,d} exp(η)))^{1/ν_d}   [Weibull PH inverse CDF]

  CDF -  Se vuoi campionare una variabile aleatoria con CDF F(t), puoi prendere U ~ Uniform(0,1) e calcolare T = F⁻¹(U). È garantito che T ha la distribuzione giusta.  
  in questo caso: 
    Quindi la CDF è F(t) = 1 - S(t). Campionare T ~ F equivale a risolvere:                                                                                                                                
    U = S(T)      con U ~ Uniform(0,1)                                                                                                                                                                  
    (U e 1−U hanno la stessa distribuzione uniforme, quindi si usa direttamente S invece di F).        
    Risolvendo per T:                                                                                                                                                                                
        U = exp(- λ₀ · T^ν · exp(η))
        log U = - λ₀ · T^ν · exp(η)                                                                        
        T^ν = - log U / (λ₀ · exp(η))
        T = (- log U / (λ₀ · exp(η)))^{1/ν}  

    - −log U è sempre positivo (U ∈ (0,1) → log U < 0) e ha distribuzione Esponenziale(1). Introduce la
   casualità individuale — due pazienti con lo stesso η hanno tempi diversi.                         
    - λ₀ · exp(η) è il rischio totale del paziente: scala quanto velocemente arriva l'evento. Più è
    alto, più T è piccolo.                                                                             
    - 1/ν controlla la forma: comprime o espande la scala temporale in modo non lineare.                                   
    
  S(t | η) = P(T > t) = exp(- λ₀ · t^ν · exp(η))
  y^(τ)_{i,d} = 1[T_{i,d} ≤ τ]

   No, non usiamo Cox. Usiamo il Weibull PH.                                                          
                                                                                                     
  Condividono la struttura PH — il log-rischio η moltiplica la baseline hazard — ma differiscono su  
  cosa è quella baseline:                                                                            
                                                                                                     
  Cox:       h(t|η) = h₀(t) · exp(η)      h₀(t) non specificata
  Weibull PH: h(t|η) = λ₀ · ν · t^{ν-1} · exp(η)    h₀(t) = λ₀ · ν · t^{ν-1}                         
                                                                                                     
  Cox è un modello di stima — utile quando non vuoi assumere una forma per h₀(t). Weibull PH è un    
  modello generativo — utile quando devi campionare T, perché hai una formula chiusa per l'inversa   
  della CDF. Per un generatore sintetico serve necessariamente la seconda. 
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .parameters import BaseParams, DomainParams, make_shared_loading_matrices


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class DomainDataset:
    """
    All variables generated for a single domain.

    Fields (n = number of samples):
      domain_id : int — domain index d
      u_inv     : (n, k_inv)  — invariant latent factors
      u_spu     : (n, k_spu)  — spurious latent factors
      x         : (n, p)      — observed covariates
      eta       : (n,)        — latent log-risk 
      T         : (n,)        — latent event time
      y         : (n,)        — fixed-horizon label  y^(τ) = 1[T ≤ τ]
      params    : DomainParams — the Θ_d used to generate this domain
    """
    domain_id: int
    u_inv: np.ndarray
    u_spu: np.ndarray
    x: np.ndarray
    eta: np.ndarray
    T: np.ndarray
    y: np.ndarray
    params: DomainParams
    C:     Optional[np.ndarray] = None   # censoring times (None = no censoring)
    t_obs: Optional[np.ndarray] = None   # min(T, C) observed time
    delta: Optional[np.ndarray] = None   # 1[T <= C] event indicator

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    def summary(self) -> dict:
        return {
            "domain_id": self.domain_id,
            "n": self.n,
            "prevalence": round(self.prevalence, 4),
            "mean_T": round(float(self.T.mean()), 3),
            "median_T": round(float(np.median(self.T)), 3),
            "mean_eta": round(float(self.eta.mean()), 4),
            "std_eta": round(float(self.eta.std()), 4),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """
    Latent-factor Weibull PH generator for multi-center shift benchmarks.

    Parameters
    ----------
    base : BaseParams
        Structural parameters shared across all domains (k_inv, k_spu, p, tau).
    shared_A_inv : np.ndarray, optional
        Shared p × k_inv loading matrix for invariant factors.  If None it is
        constructed from base.seed.  DomainParams.A_inv overrides per domain.
    shared_A_spu : np.ndarray, optional
        Shared p × k_spu loading matrix for spurious factors.  If None it is
        constructed from base.seed (same RNG stream, drawn after A_inv).
        DomainParams.A_spu overrides per domain.
    """

    def __init__(
        self,
        base: BaseParams,
        shared_A_inv: Optional[np.ndarray] = None,
        shared_A_spu: Optional[np.ndarray] = None,
        shared_B_nl:  Optional[np.ndarray] = None,
    ):
        self.base = base
        if shared_A_inv is None or shared_A_spu is None or (
            base.nl_strength > 0.0 and shared_B_nl is None
        ):
            _inv, _spu, _b_nl = make_shared_loading_matrices(base, seed=base.seed)
            if shared_A_inv is None:
                shared_A_inv = _inv
            if shared_A_spu is None:
                shared_A_spu = _spu
            if shared_B_nl is None and base.nl_strength > 0.0:
                shared_B_nl = _b_nl
        self.shared_A_inv = shared_A_inv
        self.shared_A_spu = shared_A_spu
        self.shared_B_nl  = shared_B_nl   # None when nl_strength == 0

    def generate_domain(
        self,
        params: DomainParams,
        domain_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DomainDataset:
        """
        Generate one domain dataset from parameter tuple Θ_d.

        Parameters
        ----------
        params : DomainParams
        domain_id : int, optional — overrides params.domain_id when set
        seed : int, optional — per-domain RNG seed
        """
        d_id = domain_id if domain_id is not None else params.domain_id
        # Each domain gets a deterministic but independent seed derived from the
        # global seed and domain index, ensuring reproducibility across runs.
        rng_seed = seed if seed is not None else (self.base.seed + d_id * 1000)
        rng = np.random.default_rng(rng_seed)

        n = params.n
        k_inv = self.base.k_inv
        k_spu = self.base.k_spu

        # ---- latent factors ------------------------------------------------
        # Sample invariant and spurious factors jointly from a single multivariate
        # Gaussian so that off-diagonal covariance blocks can induce cross-group
        # correlation (u_inv <-> u_spu).  Sampling separately would force
        # Cov(u_inv, u_spu) = 0 by construction and make spurious shortcuts
        # impossible to simulate.  In most scenarios latent_cov = None and the
        # joint covariance reduces to the identity (independent factors).
        k = k_inv + k_spu
        mu_joint = np.concatenate([params.mu_inv, params.mu_spu])
        cov_joint = params.latent_cov if params.latent_cov is not None else np.eye(k)
        u_joint = rng.multivariate_normal(mu_joint, cov_joint, size=n)
        u_inv = u_joint[:, :k_inv]   # (n, k_inv) — causal factors driving outcome
        u_spu = u_joint[:, k_inv:]   # (n, k_spu) — non-causal factors entering x only

        # ---- observed covariates -------------------------------------------
        # x = A^(inv) u_inv + A^(spu) u_spu + eps
        #
        # The two loading matrices are kept separate to make the structure
        # explicit: A^(inv) carries prognostically relevant signal;
        # A^(spu) carries spurious / non-transportable structure.
        # Domain-specific overrides are supported per matrix independently —
        # a scenario can shift only the spurious loading (spurious_shift) or
        # only the invariant loading (covariate_shift) without touching the
        # other.  None falls back to the shared matrix.
        A_inv = params.A_inv if params.A_inv is not None else self.shared_A_inv
        eps = rng.normal(0, params.sigma_noise, size=(n, self.base.p))
        x = u_inv @ A_inv.T + eps   # (n, p)
        if k_spu > 0:
            A_spu = params.A_spu if params.A_spu is not None else self.shared_A_spu
            x = x + u_spu @ A_spu.T

        # ---- optional tanh nonlinear observation term ----------------------
        # x += α · B_nl tanh(u_inv)
        # tanh(u_inv) is a monotone, saturating function of the invariant factors.
        # Unlike the squared term, tanh(u_inv) IS correlated with u_inv (and
        # hence with η), so it adds a nonlinear signal pathway that LR can only
        # partially exploit via its linear projection.  MLP can learn to invert
        # the saturating mapping and recover better estimates of u_inv.
        # B_nl is (p, k_inv); tanh(u_inv) is (n, k_inv) → contribution is (n, p).
        # For u ~ N(0,1): Cor(tanh(u), u) ≈ 0.96; Var(tanh(u)) ≈ 0.43 vs 1.0.
        # With α ≥ 1 the tanh term has variance comparable to the linear term,
        # creating a genuinely nonlinear observation without dominating the signal.
        if self.base.nl_strength > 0.0:
            B_nl = params.B_nl if params.B_nl is not None else self.shared_B_nl
            if B_nl is not None:
                x = x + self.base.nl_strength * (np.tanh(u_inv) @ B_nl.T)  # (n, p)

        # ---- latent log-risk -----------------------------------------------
        # η_{i,d} = β_{0,d} + w_d^T u^(inv) + v_d^T u^(spu)
        # beta0 is a domain-level intercept (shifts all patients' risk uniformly).
        # w weights the causal factors; v weights the spurious ones (v=0 by default,
        # so spurious factors never directly drive the outcome).
        eta = params.beta0 + u_inv @ params.w + u_spu @ params.v  # (n,)

        # ---- Weibull PH event time (inverse CDF) ---------------------------
        # T = (-log U / (λ_0 exp(η)))^{1/ν}
        # U ~ Uniform(0,1) introduces individual-level randomness: two patients
        # with the same η will have different event times.  -log U ~ Exp(1).
        # lambda0 controls the baseline event rate (scale); nu controls the
        # shape of the hazard over time (nu > 1: increasing hazard, nu < 1:
        # decreasing).  Higher eta compresses T toward zero — higher risk,
        # earlier event.
        U = rng.uniform(0, 1, size=n)
        dist = getattr(params, "time_distribution", "weibull")
        if dist == "loglogistic":
            # Log-logistic inverse CDF: T = (U/(1-U))^(1/ν) / (λ₀·exp(η))
            # Survival: S(t) = 1/(1+(λ₀ exp(η) t)^ν)
            # Hazard: non-monotone for ν>1 — peaks at t*=((ν-1)/λ₀exp(η))^(1/ν)
            # Non-PH: HR(t) = [1+(λ₀exp(η₂)t)^ν] / [1+(λ₀exp(η₁)t)^ν] ≠ const
            U_clip = np.clip(U, 1e-9, 1 - 1e-9)
            T = (U_clip / (1.0 - U_clip)) ** (1.0 / params.nu) / (params.lambda0 * np.exp(eta))
        else:
            # Weibull PH (default): T = (-log U / λ₀ exp(η))^(1/ν)
            T = (-np.log(U) / (params.lambda0 * np.exp(eta))) ** (1.0 / params.nu)

        # ---- mixture cure model (optional) ---------------------------------
        # When cure_fraction > 0 a fraction π of patients are long-term
        # survivors: their true event time is ∞, so they appear as censored.
        # Patient-specific cure probability:
        #   logit(π_i) = logit(π̄) + u_inv_i @ w_cure   (if w_cure is set)
        #              = logit(π̄)                        (homogeneous otherwise)
        # Non-proportional hazards arise because the marginal survival function
        #   S(t|X) = π_i + (1 - π_i) · exp(-λ₀ t^ν exp(η_i))
        # makes the hazard ratio between any two patients time-dependent:
        # it starts at the Weibull PH value and decays toward 1 as t → ∞.
        if params.cure_fraction > 0.0:
            from scipy.special import expit as _expit
            pi_bar = float(np.clip(params.cure_fraction, 1e-6, 1 - 1e-6))
            logit_pi_bar = float(np.log(pi_bar / (1.0 - pi_bar)))
            if params.w_cure is not None:
                logit_pi = logit_pi_bar + u_inv @ params.w_cure   # (n,)
            else:
                logit_pi = np.full(n, logit_pi_bar)
            pi_i = _expit(logit_pi)                                # (n,) ∈ (0,1)
            # Draw susceptibility: 1 = will eventually have event, 0 = cured
            susceptible = rng.binomial(1, 1.0 - pi_i).astype(bool)  # (n,)
            # Cured patients: T = inf (never event → always censored)
            T = np.where(susceptible, T, np.inf)

        # ---- independent exponential censoring ----------------------------
        # C ~ Exp(lambda_c); t_obs = min(T, C); delta = 1[T <= C].
        # When censoring_lambda == 0 there is no censoring: C = inf, delta = 1.
        if params.censoring_lambda > 0.0:
            C_arr = rng.exponential(scale=1.0 / params.censoring_lambda, size=n)
        else:
            C_arr = np.full(n, np.inf)
        t_obs = np.minimum(T, C_arr)
        delta = (T <= C_arr).astype(int)

        # ---- fixed-horizon label -------------------------------------------
        # y = 1[t_obs <= tau AND delta == 1]
        # Consistent with real-data labelling: event observed within the window.
        y = ((t_obs <= self.base.tau) & (delta == 1)).astype(int)

        p_copy = params.copy()
        p_copy.domain_id = d_id
        return DomainDataset(
            domain_id=d_id,
            u_inv=u_inv,
            u_spu=u_spu,
            x=x,
            eta=eta,
            T=T,
            y=y,
            params=p_copy,
            C=C_arr if params.censoring_lambda > 0.0 else None,
            t_obs=t_obs,
            delta=delta,
        )

    def generate_domains(
        self,
        domain_params: list[DomainParams],
        base_seed_offset: int = 0,
    ) -> list[DomainDataset]:
        """Generate one dataset per element of domain_params.

        Each domain receives a unique seed derived from the global seed, its
        domain index, and an optional offset — keeping domains reproducible and
        mutually independent across calls.
        """
        datasets = []
        for i, dp in enumerate(domain_params):
            d_id = dp.domain_id if dp.domain_id >= 0 else i
            ds = self.generate_domain(dp, domain_id=d_id, seed=self.base.seed + d_id * 1000 + base_seed_offset)
            datasets.append(ds)
        return datasets
