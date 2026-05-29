"""
Fit a synthetic Weibull PH generator to real multi-center survival data.

Pipeline (two-stage MVP):
  1. Fit a shared PCA loading matrix A (p × k) from pooled source data.
  2. Fit shared risk weights w (k,) from pooled (U, y) via logistic regression.
  3. Fit per-domain baseline hazard λ₀_d via 1-D bisection so that
     synthetic prevalence matches real prevalence at horizon τ.
  4. Fit per-domain censoring rate λ_c_d from observed (t, delta).
  5. Return DomainParams for each source domain + a reference set.

k_spu is set to 0 throughout (single latent pool: all k dimensions treated as
"invariant"). The inv/spu split is deferred to a later phase.

Notation
--------
U      : (n, k) latent factor matrix (PCA scores, centred)
eta    : (n,)   latent log-risk  = U @ w
lambda0: scalar baseline Weibull rate
nu     : scalar Weibull shape parameter (fixed default: 1.5)
tau    : fixed prediction horizon (same unit as t)
"""

from __future__ import annotations
from typing import Optional
import warnings
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit as _sigmoid
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from generator.base_generator import DomainDataset
from generator.parameters import BaseParams, DomainParams


# ---------------------------------------------------------------------------
# Step 1 — loading matrix
# ---------------------------------------------------------------------------

def fit_loading_matrix(
    X_pooled: np.ndarray,
    k_total: int,
) -> np.ndarray:
    """
    Fit a (p × k_total) loading matrix from pooled source data via PCA.

    Columns are the top-k_total principal components of X_pooled, scaled so
    each column has unit L2 norm (matching the synthetic generator convention).

    Parameters
    ----------
    X_pooled : (N, p) array — pooled and standardized source covariates
    k_total  : number of latent dimensions

    Returns
    -------
    A : (p, k_total) loading matrix with unit-norm columns
    """
    pca = PCA(n_components=k_total, random_state=0)
    pca.fit(X_pooled)
    A = pca.components_.T   # (p, k_total) — each column is a principal direction
    # Normalise columns to unit L2 norm (convention in parameters.py)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


# ---------------------------------------------------------------------------
# Step 2 — risk weights
# ---------------------------------------------------------------------------

def fit_risk_weights(
    U_pooled: np.ndarray,
    y_pooled: np.ndarray,
) -> np.ndarray:
    """
    Fit logistic regression on pooled latent factors U to predict y.

    The fitted coefficient vector is used as the shared risk weight vector w.
    Intercept is dropped (absorbed into per-domain lambda0).

    Parameters
    ----------
    U_pooled : (N, k) latent factor matrix (PCA scores)
    y_pooled : (N,)   binary fixed-horizon labels

    Returns
    -------
    w : (k,) risk weight vector (normalised to unit L2 norm)
    """
    if len(np.unique(y_pooled)) < 2:
        raise ValueError("y_pooled has only one class — cannot fit risk weights")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(
            fit_intercept=False,
            max_iter=500,
            solver="lbfgs",
            C=1.0,
            random_state=0,
        )
        lr.fit(U_pooled, y_pooled)

    w = lr.coef_[0]   # (k,)
    norm = np.linalg.norm(w)
    if norm < 1e-8:
        raise ValueError("Fitted risk weights are near-zero — check U and y alignment")
    return w / norm


# ---------------------------------------------------------------------------
# Step 2b — risk weights via Cox partial likelihood
# ---------------------------------------------------------------------------

def fit_risk_weights_cox(
    U_pooled: np.ndarray,
    t_pooled: np.ndarray,
    delta_pooled: np.ndarray,
    alpha: float = 1e-2,
    n_iter: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """
    Fit risk weights w via Cox proportional hazards partial likelihood.

    Maximises the Breslow log-partial-likelihood:

        l(w) = Σ_{i: δ_i=1} [ U_i @ w − log Σ_{j: t_j ≥ t_i} exp(U_j @ w) ]

    plus an optional L2 penalty  − α/2 · ||w||²  for regularisation.

    Advantages over LR on binary y and log-logistic MLE
    -----------------------------------------------------
    - Uses ALL observed event times, not just the binary label at a fixed τ.
    - Semiparametric: does not assume a parametric baseline hazard shape.
      Compatible with any underlying distribution (Weibull, log-logistic, …).
    - Standard model for survival data; C-statistic optimised by this objective
      is the rank-equivalent of AUROC.

    Gradient (closed form, efficient suffix-sum implementation)
    -----------------------------------------------------------
    ∂l/∂w = Σ_{i:δ=1} [ U_i − E_R(t_i)[U · softmax(η)] ]
    where E_R[·] is the risk-set weighted average at event time t_i.

    Parameters
    ----------
    U_pooled    : (N, k) PCA latent factors
    t_pooled    : (N,)   observed survival times
    delta_pooled: (N,)   event indicators
    alpha       : L2 regularisation strength (default 1e-2)
    n_iter      : L-BFGS-B iterations
    seed        : RNG seed for initialisation

    Returns
    -------
    w : (k,) risk weights, unit-norm
    """
    # Sort ascending by time (required for suffix-sum trick)
    order = np.argsort(t_pooled)
    U  = U_pooled[order].astype(float)
    d  = delta_pooled[order].astype(float)
    k, n = U.shape[1], len(d)

    def _neg_loglik_and_grad(w):
        eta = U @ w                               # (n,)
        # Numerically stable exp: subtract running max from the right
        eta_shift = eta - eta.max()
        exp_eta   = np.exp(eta_shift)             # (n,)

        # Suffix sums from largest time (index n-1) to smallest (index 0)
        suf_exp  = np.cumsum(exp_eta[::-1])[::-1]          # (n,)
        suf_expU = np.cumsum((exp_eta[:, None] * U)[::-1],
                             axis=0)[::-1]                  # (n, k)

        ev = d == 1                               # event mask
        log_denom = np.log(suf_exp[ev] + 1e-30)
        ll = (eta[ev] - log_denom).sum()

        # Gradient of partial log-likelihood
        wm_U  = suf_expU[ev] / suf_exp[ev, None]  # (n_ev, k) risk-set weighted mean
        grad_ll = (U[ev] - wm_U).sum(axis=0)       # (k,)

        # L2 penalty
        ll   -= alpha * 0.5 * float(w @ w) * n
        grad  = grad_ll - alpha * w * n

        return -ll / n, -grad / n

    rng = np.random.default_rng(seed)
    w0  = rng.standard_normal(k) * 0.01
    res = minimize(_neg_loglik_and_grad, w0, jac=True, method="L-BFGS-B",
                   options={"maxiter": n_iter, "ftol": 1e-10, "gtol": 1e-6})

    w_opt = res.x
    norm  = np.linalg.norm(w_opt)
    if norm < 1e-8:
        raise ValueError("Cox MLE: fitted w is near-zero — check U/t alignment")
    return w_opt / norm


# ---------------------------------------------------------------------------
# Step 2b-aux — Cox score function  ∂CoxPLL/∂η  (used for multitask A fitting)
# ---------------------------------------------------------------------------

def _cox_residuals_eta(
    eta:   np.ndarray,
    t:     np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """
    Compute ∂CoxPLL / ∂η at the given linear predictor η.

    Returns the (N,) Cox Schoenfeld / score residuals:
        r_i = δ_i  −  exp(η_i) · Σ_{j: δ_j=1, t_j ≤ t_i}  1 / Σ_{m: t_m ≥ t_j} exp(η_m)

    This is ∂l/∂η_i in the partial log-likelihood, obtained via the efficient
    suffix-sum trick (same as in fit_risk_weights_cox).
    """
    n     = len(eta)
    order = np.argsort(t)
    eta_s = eta[order]
    d_s   = delta[order].astype(float)
    inv_order = np.argsort(order)

    eta_shift = eta_s - eta_s.max()
    exp_eta   = np.exp(eta_shift)
    suf_exp   = np.cumsum(exp_eta[::-1])[::-1]   # (n,) suffix sums

    # cumulative sum of 1/suf_exp[j] at event positions only
    inv_at_ev = np.zeros(n)
    ev_mask   = d_s == 1
    inv_at_ev[ev_mask] = 1.0 / (suf_exp[ev_mask] + 1e-30)
    cum_inv   = np.cumsum(inv_at_ev)             # (n,) running accumulator

    r_sorted = d_s - exp_eta * cum_inv
    return r_sorted[inv_order]                   # restore original patient order


# ---------------------------------------------------------------------------
# Step 1-mm — multimodal loading matrices  (shared latent space, masked fusion)
# ---------------------------------------------------------------------------

def fit_multimodal_loading_matrices(
    modalities:   list,
    t_pooled:     np.ndarray,
    delta_pooled: np.ndarray,
    k_total:      int,
    alpha:        float = 1e-2,
    n_outer:      int   = 40,
    lr:           float = 0.05,
    seed:         int   = 0,
    fusion:       str   = "average",
) -> tuple:
    """
    Fit per-modality loading matrices that share a k-dimensional latent space.

    Two fusion strategies (controlled by `fusion`):

    "average" (default) — masked average:
        U_i = (Σ_m mask_im · X_m_i @ A_m) / (Σ_m mask_im)
        Patients with more modalities: each contributes with weight 1/n_avail.
        Problem: the primary (clinical) signal is diluted for multi-modal patients.

    "additive" — primary + scaled auxiliary contributions:
        U_i = X_pri_i @ A_pri  +  Σ_{m≠pri} α_m · mask_im · X_m_i @ A_m
        where α_m are per-modality scalars, initialised to 0 and jointly optimised.
        Clinical representation is never diluted; auxiliary modalities ADD to it.
        The first entry of `modalities` is treated as the primary modality.
        α_m are optimised via gradient ascent alongside the A_m matrices.

    Cox partial log-likelihood is maximised jointly over all {A_m} via
    alternating gradient ascent on the Stiefel manifold (QR retraction).
    Modalities missing for patient i are excluded from U_i — no imputation.

    Constraint:  k_total ≤ min(p_m) for all m  (QR retraction requires p_m ≥ k).

    Parameters
    ----------
    modalities   : list of dicts, each containing:
                     "key"  : str identifier
                     "X"    : (N, p_m) standardised feature matrix (NaN-free)
                     "mask" : (N,) float — 1.0 if patient i has this modality,
                              0.0 otherwise.  Every patient must have at least
                              one modality (mask sum ≥ 1 for all i).
    t_pooled     : (N,) survival times
    delta_pooled : (N,) event indicators
    k_total      : shared latent dimension
    alpha        : Cox L2 regularisation strength
    n_outer      : outer alternating iterations
    lr           : Stiefel gradient-ascent step size

    Returns
    -------
    A_dict : dict {"key": (p_m, k_total)} — one loading matrix per modality
    w      : (k_total,) shared Cox risk weights (unit-norm)
    alphas : dict {"key": float} — per-modality scale (only for fusion="additive";
             primary modality has alpha=1.0, auxiliary alphas are learned)
    """
    if fusion not in ("average", "additive"):
        raise ValueError(f"fusion must be 'average' or 'additive', got '{fusion}'")

    N = len(t_pooled)

    # Initialise each A_m via PCA on patients who have that modality
    A_dict = {}
    for m in modalities:
        idx = m["mask"].astype(bool)
        A_dict[m["key"]] = fit_loading_matrix(m["X"][idx], k_total)

    masks_f = [m["mask"].astype(float) for m in modalities]   # list of (N,)
    n_avail = np.array(masks_f).sum(axis=0)                    # (N,)  ≥ 1

    # For additive fusion: learned per-auxiliary scalar α_m (init small)
    pri_key = modalities[0]["key"]
    aux_keys = [m["key"] for m in modalities[1:]]
    alphas = {k: 0.01 for k in aux_keys}   # initialise near-zero

    for _ in range(n_outer):
        # ── fused latent representation ──────────────────────────────────
        if fusion == "average":
            U = np.zeros((N, k_total))
            for m, mf in zip(modalities, masks_f):
                idx = mf.astype(bool)
                U[idx] += m["X"][idx] @ A_dict[m["key"]]
            U /= n_avail[:, None]

        else:  # additive
            # U_i = X_pri @ A_pri + Σ_{aux} α_m · mask_im · X_m @ A_m
            U = X_pri_part = modalities[0]["X"] @ A_dict[pri_key]  # (N, k)
            U = U.copy()
            for m, mf in zip(modalities[1:], masks_f[1:]):
                idx = mf.astype(bool)
                alp = alphas[m["key"]]
                U[idx] += alp * (m["X"][idx] @ A_dict[m["key"]])

        # ── Cox risk weights ──────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w = fit_risk_weights_cox(U, t_pooled, delta_pooled, alpha=alpha)

        # ── Cox score residuals  r_i = ∂CoxPLL/∂η_i ──────────────────────
        r = _cox_residuals_eta(U @ w, t_pooled, delta_pooled)  # (N,)

        # ── gradient steps ────────────────────────────────────────────────
        if fusion == "average":
            for m, mf in zip(modalities, masks_f):
                idx    = mf.astype(bool)
                X_sub  = m["X"][idx]
                r_sub  = r[idx]
                ni_sub = n_avail[idx]
                grad = X_sub.T @ ((r_sub / ni_sub)[:, None] * w[None, :]) / N
                A_m  = A_dict[m["key"]] + lr * grad
                Q, _ = np.linalg.qr(A_m)
                A_dict[m["key"]] = Q[:, :k_total]

        else:  # additive
            # Primary A_pri: gradient from all patients
            grad_pri = modalities[0]["X"].T @ (r[:, None] * w[None, :]) / N
            A_m = A_dict[pri_key] + lr * grad_pri
            Q, _ = np.linalg.qr(A_m)
            A_dict[pri_key] = Q[:, :k_total]

            # Auxiliary A_m and α_m
            for m, mf in zip(modalities[1:], masks_f[1:]):
                idx   = mf.astype(bool)
                alp   = alphas[m["key"]]
                X_sub = m["X"][idx]           # (N_m, p_m)
                r_sub = r[idx]                # (N_m,)
                U_aux = X_sub @ A_dict[m["key"]]  # (N_m, k)

                # ∂l/∂A_m = α · X_m.T @ (r[:, None] * w) / N
                grad_Am = alp * (X_sub.T @ (r_sub[:, None] * w[None, :]) / N)
                A_m = A_dict[m["key"]] + lr * grad_Am
                Q, _ = np.linalg.qr(A_m)
                A_dict[m["key"]] = Q[:, :k_total]

                # ∂l/∂α_m = (U_aux @ w) · r  summed over patients with this mod
                grad_alp = float((U_aux @ w) @ r_sub) / N
                alphas[m["key"]] = alphas[m["key"]] + lr * grad_alp

    alphas[pri_key] = 1.0   # primary always has weight 1
    return A_dict, w, alphas


def score_multimodal(
    X_dict:  dict,
    mask:    dict,
    A_dict:  dict,
    w:       np.ndarray,
    mu_src:  Optional[np.ndarray] = None,
    alphas:  Optional[dict] = None,
) -> np.ndarray:
    """
    Compute per-patient scores using whatever modalities are available.

    Parameters
    ----------
    X_dict  : {"key": (N, p_m)} — scaled feature matrices per modality
    mask    : {"key": (N,) float} — availability masks (1 = present)
    A_dict  : {"key": (p_m, k)} — loading matrices
    w       : (k,) shared risk weights
    mu_src  : (k,) optional source-domain mean to subtract before scoring
    alphas  : optional dict {"key": float} — per-modality scales.
              None → masked-average fusion (divide by n_avail).
              Provided → additive fusion: U = Σ_m alphas[m] * mask_m * X_m @ A_m.

    Returns
    -------
    score : (N,) linear predictor  (U − mu_src) @ w
    """
    keys = list(A_dict.keys())
    N    = next(iter(X_dict.values())).shape[0]
    k    = w.shape[0]

    if alphas is None:
        # ── masked-average fusion ─────────────────────────────────────────
        U       = np.zeros((N, k))
        n_avail = np.zeros(N)
        for key in keys:
            mf  = mask[key].astype(float)
            idx = mf.astype(bool)
            if idx.any():
                U[idx]      += X_dict[key][idx] @ A_dict[key]
                n_avail[idx] += 1.0
        safe = n_avail > 0
        U[safe] /= n_avail[safe, None]
    else:
        # ── additive fusion ───────────────────────────────────────────────
        # U_i = Σ_m alphas[m] · mask_im · X_m_i @ A_m
        U = np.zeros((N, k))
        for key in keys:
            mf  = mask[key].astype(float)
            alp = float(alphas.get(key, 1.0))
            U  += alp * mf[:, None] * (X_dict[key] @ A_dict[key])

    if mu_src is not None:
        U = U - mu_src
    return U @ w


# ---------------------------------------------------------------------------
# Step 1-multi — multitask loading matrix  (PFS-auxiliary + OS-main)
# ---------------------------------------------------------------------------

def compute_genomic_risk_scores(
    X_geno:      np.ndarray,
    t_pfs:       np.ndarray,
    delta_pfs:   np.ndarray,
    k_geno:      int = 3,
) -> np.ndarray:
    """
    Stage A of genomic distillation: learn a genomic encoder supervised by PFS
    and return the genomic risk score η_geno for each MSK patient.

    Pipeline:
      1. StandardScale X_geno within MSK.
      2. PCA(X_geno_scaled, k_geno) → A_geno  (p_geno × k_geno)
      3. Cox(X_geno_scaled @ A_geno, t_pfs, δ_pfs) → w_geno
      4. η_geno = (X_geno_scaled @ A_geno) @ w_geno   (N_msk,)

    The resulting η_geno is a continuous genomic risk score: patients with
    higher η_geno have worse PFS prognosis according to their genomic profile.
    It serves as the distillation target in fit_multitask_loading_matrix.

    Parameters
    ----------
    X_geno    : (N_msk, p_geno) raw genomic features (TMB + driver mutations)
    t_pfs     : (N_msk,) PFS times
    delta_pfs : (N_msk,) PFS event indicators
    k_geno    : number of genomic latent dimensions (default 3)

    Returns
    -------
    eta_geno : (N_msk,) genomic risk scores (not unit-normalised — magnitude matters)
    """
    from sklearn.preprocessing import StandardScaler as _SS
    X_g = _SS().fit_transform(X_geno)
    A_g = fit_loading_matrix(X_g, k_geno)
    U_g = X_g @ A_g
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        w_g = fit_risk_weights_cox(U_g, t_pfs, delta_pfs)
    return U_g @ w_g   # (N_msk,)


def fit_multitask_loading_matrix(
    X_src:        np.ndarray,
    t_src:        np.ndarray,
    delta_src:    np.ndarray,
    X_ext:        np.ndarray,
    t_ext:        np.ndarray,
    delta_ext:    np.ndarray,
    k_total:      int,
    lam:          float = 0.5,
    alpha:        float = 1e-2,
    n_outer:      int   = 40,
    lr:           float = 0.05,
    seed:         int   = 0,
    eta_geno_ext: Optional[np.ndarray] = None,
    lam_geno:     float = 0.3,
) -> np.ndarray:
    """
    Fit a shared (p × k_total) loading matrix A that is jointly supervised by
    up to three tasks:

      Task 1  i3lung source OS  (main task,   weight 1)         — CoxPLL
      Task 2  external PFS      (aux task,    weight lam)        — CoxPLL
      Task 3  genomic distill   (distil task, weight lam_geno)  — MSE regression
              [active only when eta_geno_ext is provided]

    Task 3 is knowledge distillation from a pre-trained genomic encoder:

        η_geno = (X_geno @ A_geno) @ w_geno   (precomputed, frozen)

    A is trained so that (X_clin_msk @ A) @ w_distill ≈ η_geno,
    forcing A to find clinical directions that correlate with genomic risk.
    Since i3lung shares the same clinical feature space, the genomic signal
    is implicitly transferred via A_shared even though i3lung has no genomics.

    Algorithm: alternating minimisation on Stiefel manifold.

      Initialise A ← PCA(X_src)
      Repeat n_outer times:
        1. w_os      ← Cox(X_src @ A, t_src, δ_src)
        2. w_pfs     ← Cox(X_ext @ A, t_ext, δ_ext)
        3. w_distill ← OLS(X_ext @ A, η_geno)     [if eta_geno_ext given]
        4. grad_A =   X_src.T @ (r_os  ⊗ w_os)  / N_src
                    + lam      · X_ext.T @ (r_pfs ⊗ w_pfs) / N_ext
                    + lam_geno · X_ext.T @ (r_dist ⊗ w_distill) / N_ext
           where r_dist = X_ext @ A @ w_distill − η_geno   (regression residual)
        5. A ← QR-retract( A + lr · grad_A )

    Parameters
    ----------
    X_src, t_src, delta_src : i3lung source features (scaled) + OS survival
    X_ext, t_ext, delta_ext : MSK clinical features (scaled) + PFS survival
    k_total                  : latent dimension
    lam                      : weight of PFS task
    alpha                    : L2 regularisation for Cox risk weights
    n_outer                  : alternating iterations
    lr                       : gradient-ascent step size on A
    eta_geno_ext             : (N_ext,) precomputed genomic risk scores for MSK
                               patients (from compute_genomic_risk_scores).
                               None → skip Task 3 (equivalent to lam_geno=0)
    lam_geno                 : weight of genomic distillation task

    Returns
    -------
    A : (p, k_total) loading matrix with orthonormal columns
    """
    A      = fit_loading_matrix(X_src, k_total)
    N_src  = len(t_src)
    N_ext  = len(t_ext)

    for _ in range(n_outer):
        U_src = X_src @ A
        U_ext = X_ext @ A

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            w_os  = fit_risk_weights_cox(U_src, t_src, delta_src, alpha=alpha)
            w_pfs = fit_risk_weights_cox(U_ext, t_ext, delta_ext, alpha=alpha)

        r_os  = _cox_residuals_eta(U_src @ w_os,  t_src, delta_src)
        r_pfs = _cox_residuals_eta(U_ext @ w_pfs, t_ext, delta_ext)

        grad_A  = (X_src.T @ (r_os[:, None]  * w_os[None,  :])) / N_src
        grad_A += lam * (X_ext.T @ (r_pfs[:, None] * w_pfs[None, :])) / N_ext

        # ── Task 3: genomic distillation (optional) ───────────────────────
        if eta_geno_ext is not None:
            # OLS: w_distill = (U_extᵀ U_ext)⁻¹ U_extᵀ η_geno
            w_dist, _, _, _ = np.linalg.lstsq(U_ext, eta_geno_ext, rcond=None)
            r_dist = U_ext @ w_dist - eta_geno_ext          # (N_ext,) residual
            # ∂MSE/∂A = (1/N_ext) · X_ext.T @ (r_dist[:,None] * w_dist[None,:])
            # We subtract (gradient ascent on neg-MSE = gradient descent on MSE)
            grad_A -= lam_geno * (X_ext.T @ (r_dist[:, None] * w_dist[None, :])) / N_ext

        A = A + lr * grad_A
        Q, _ = np.linalg.qr(A)
        A    = Q[:, :k_total]

    return A


# ---------------------------------------------------------------------------
# Step 1-ada — rank-1 center-specific observation correction  Δ = u vᵀ
# ---------------------------------------------------------------------------

def fit_delta_rank1(
    X_tgt:       np.ndarray,
    t_tgt:       np.ndarray,
    delta_tgt:   np.ndarray,
    A:           np.ndarray,
    w:           np.ndarray,
    rho:         float = 1.0,
    n_restarts:  int   = 3,
    n_iter:      int   = 300,
    seed:        int   = 0,
) -> np.ndarray:
    """
    Fit a rank-1 center-specific correction  Δ_d = u vᵀ  (p × k)  such that

        A_d = A_shared + Δ_d

    maximises the regularised Cox partial log-likelihood on target data:

        l(u, v) = CoxPLL( (X @ (A + u vᵀ)) @ w ) − ρ/2 · (‖u‖² + ‖v‖²) · n

    With Δ = u vᵀ the linear predictor decomposes as

        η_i  =  η_shared_i  +  c · h_i
        η_shared_i = (x_i @ A) @ w   (fixed)
        c = vᵀ @ w                   (scalar)
        h_i = x_i @ u                (patient-specific loading)

    Gradients (closed-form):
        ∂l/∂u = c · Xᵀ r − ρ · u · n
        ∂l/∂v = (hᵀ r) · w − ρ · v · n
    where r = ∂CoxPLL/∂η  (computed via _cox_residuals_eta).

    Multiple random restarts are used to avoid saddle points; the solution
    with the highest regularised CoxPLL is returned.

    Parameters
    ----------
    X_tgt, t_tgt, delta_tgt : (n, p) scaled target features, survival times,
                               event indicators
    A                        : (p, k) shared loading matrix (frozen)
    w                        : (k,) shared risk weights (frozen)
    rho                      : L2 regularisation on u and v (default 1.0)
    n_restarts               : random restarts (default 3)
    n_iter                   : L-BFGS-B iterations per restart
    seed                     : base RNG seed

    Returns
    -------
    delta : (p, k) rank-1 correction matrix  u vᵀ
    """
    p, k = A.shape
    n    = len(t_tgt)

    # Fixed shared predictor (does not change during optimisation)
    eta_base = X_tgt @ A @ w   # (n,)

    def _neg_ll_grad(params):
        u = params[:p]
        v = params[p:]
        c = float(v @ w)
        h = X_tgt @ u          # (n,)
        eta = eta_base + c * h  # (n,)

        # Cox PLL value (for convergence tracking only)
        order     = np.argsort(t_tgt)
        eta_s     = eta[order]
        d_s       = delta_tgt[order].astype(float)
        eta_shift = eta_s - eta_s.max()
        exp_eta   = np.exp(eta_shift)
        suf_exp   = np.cumsum(exp_eta[::-1])[::-1]
        ev        = d_s == 1
        ll        = (eta_s[ev] - np.log(suf_exp[ev] + 1e-30)).sum()

        # Score residuals  r = ∂CoxPLL/∂η
        r = _cox_residuals_eta(eta, t_tgt, delta_tgt)   # (n,)

        # Gradients w.r.t. u and v
        grad_u = c * (X_tgt.T @ r) - rho * u * n
        grad_v = float(h @ r) * w   - rho * v * n

        # Regularisation on likelihood
        ll -= rho * 0.5 * (float(u @ u) + float(v @ v)) * n

        grad = np.concatenate([grad_u, grad_v])
        return -ll / n, -grad / n

    best_val, best_uv = np.inf, None
    rng = np.random.default_rng(seed)

    for _ in range(n_restarts):
        params0 = rng.standard_normal(p + k) * 0.01
        res = minimize(
            _neg_ll_grad, params0, jac=True, method="L-BFGS-B",
            options={"maxiter": n_iter, "ftol": 1e-10, "gtol": 1e-6},
        )
        if res.fun < best_val:
            best_val, best_uv = res.fun, res.x

    u_opt = best_uv[:p]
    v_opt = best_uv[p:]
    return np.outer(u_opt, v_opt)   # (p, k)


# ---------------------------------------------------------------------------
# Step 2c — risk weights via log-logistic survival likelihood
# ---------------------------------------------------------------------------

def fit_risk_weights_loglogistic(
    U_pooled: np.ndarray,
    t_pooled: np.ndarray,
    delta_pooled: np.ndarray,
    n_iter: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """
    Fit risk weights w via MLE of a log-logistic AFT survival model on
    pooled (U, t, delta) data.

    Model
    -----
    S(t | u) = 1 / (1 + (λ₀ · exp(u @ w) · t)^ν)

    Let η_i = U_i @ w,  z_i = log(λ₀) + η_i + log(t_i).  Then:

      log f(t_i | u_i) = log ν − log t_i + ν z_i − 2 · softplus(ν z_i)
      log S(t_i | u_i) = − softplus(ν z_i)

    Total log-likelihood:
      L = Σ_i [ δ_i · (log ν + ν z_i − log t_i) − (1 + δ_i) · softplus(ν z_i) ]

    Advantages over LR on binary y
    --------------------------------
    - Uses actual event times, not just y = 1[T ≤ τ].  A patient who events
      at t=0.5 and one at t=5.9 both have y=1 but carry different information.
    - Link function P(y=1|η) = σ(ν·(log(λ₀τ)+η)) is exactly logistic → no
      link misspecification when log-logistic DGP is correct.
    - Jointly estimates ν, giving a calibrated shape parameter for the generator.

    Parameters
    ----------
    U_pooled  : (N, k) PCA latent factors
    t_pooled  : (N,)   observed survival times
    delta_pooled: (N,) event indicators (1=event, 0=censored)

    Returns
    -------
    w      : (k,) risk weights, unit-norm
    lambda0: float, global baseline hazard rate
    nu     : float, shape parameter
    """
    k = U_pooled.shape[1]
    log_t = np.log(np.clip(t_pooled, 1e-8, None))
    delta = delta_pooled.astype(float)

    def _softplus(x):
        # numerically stable: log(1 + exp(x))
        return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))

    def _neg_loglik_and_grad(params):
        w       = params[:k]
        log_lam = params[k]      # log λ₀
        log_nu  = params[k + 1]  # log ν  (ensures ν > 0)
        nu      = np.exp(log_nu)

        eta = U_pooled @ w                              # (N,)
        z   = log_lam + eta + log_t                     # (N,)
        nuz = nu * z                                    # (N,)
        sp  = _softplus(nuz)                            # log(1+(λ₀ exp(η) t)^ν)
        sig = _sigmoid(nuz)                             # σ(ν·z)

        # log-likelihood (we minimise the negative)
        ll  = (delta * (np.log(nu) + nuz - log_t) - (1.0 + delta) * sp).sum()

        # --- gradients ---
        # ∂ll/∂w: chain through η_i → z_i → nuz_i
        r_w = delta * nu - (1.0 + delta) * sig * nu    # (N,) residuals × ν
        grad_w = U_pooled.T @ r_w                       # (k,)

        # ∂ll/∂log_λ₀: ∂z_i/∂log_lam = 1
        grad_lam = (delta * nu - (1.0 + delta) * sig * nu).sum()

        # ∂ll/∂log_ν via chain: ∂ll/∂ν × ν
        # ∂ll/∂ν = Σ [δ_i·(1/ν + z_i) − (1+δ_i)·σ(νz_i)·z_i]
        dll_dnu = (delta * (1.0 / nu + z) - (1.0 + delta) * sig * z).sum()
        grad_nu = dll_dnu * nu   # ∂ll/∂log_ν = ∂ll/∂ν · ν

        grad = np.concatenate([grad_w, [grad_lam, grad_nu]])
        return -ll / len(delta), -grad / len(delta)

    rng = np.random.default_rng(seed)
    w0      = rng.standard_normal(k) * 0.01
    lam0    = np.log(1.0 / (t_pooled[delta == 1].mean() + 1e-8))
    nu0     = 0.0   # log(1) = 0 → ν=1 start

    x0 = np.concatenate([w0, [lam0, nu0]])
    res = minimize(
        _neg_loglik_and_grad, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": n_iter, "ftol": 1e-10, "gtol": 1e-6},
    )

    w_opt   = res.x[:k]
    lam_opt = float(np.exp(res.x[k]))
    nu_opt  = float(np.exp(res.x[k + 1]))

    norm = np.linalg.norm(w_opt)
    if norm < 1e-8:
        raise ValueError("Log-logistic MLE: fitted w is near-zero — check U/t alignment")
    return w_opt / norm, lam_opt, nu_opt


# ---------------------------------------------------------------------------
# Step 3 — per-domain baseline hazard
# ---------------------------------------------------------------------------

def _weibull_prevalence(lambda0: float, eta: np.ndarray, tau: float, nu: float) -> float:
    """Expected prevalence under Weibull PH with given lambda0 and individual etas."""
    return float(np.mean(1.0 - np.exp(-lambda0 * (tau ** nu) * np.exp(eta))))


def _loglogistic_prevalence(lambda0: float, eta: np.ndarray, tau: float, nu: float) -> float:
    """Expected prevalence under log-logistic with given lambda0 and individual etas."""
    x = (lambda0 * np.exp(eta) * tau) ** nu
    return float(np.mean(x / (1.0 + x)))


def _km_at_tau(t: np.ndarray, delta: np.ndarray, tau: float) -> float:
    """
    Kaplan-Meier survival probability S(tau).

    Returns the probability of surviving past tau, accounting for censoring.
    Used to calibrate lambda0 against the censoring-adjusted event probability
    rather than the raw observed proportion (which is deflated by censoring).
    """
    order = np.argsort(t)
    t_s, d_s = t[order], delta[order]
    n = len(t_s)
    S = 1.0
    at_risk = n
    for ti, di in zip(t_s, d_s):
        if ti > tau:
            break
        if di == 1:
            S *= 1.0 - 1.0 / max(at_risk, 1)
        at_risk -= 1
    return float(S)


def fit_lambda0(
    eta: np.ndarray,
    observed_prevalence: float,
    tau: float,
    nu: float = 1.5,
    bracket: tuple[float, float] = (1e-6, 1e2),
) -> float:
    """
    Find lambda0 such that E[1 - exp(-lambda0 * tau^nu * exp(eta))] == observed_prevalence.

    Uses Brent's method (guaranteed convergence for monotone function on bracket).

    Parameters
    ----------
    eta                 : (n,) latent log-risk values for this domain
    observed_prevalence : target event probability at horizon tau
    tau                 : fixed horizon
    nu                  : Weibull shape parameter
    bracket             : search interval for lambda0

    Returns
    -------
    lambda0 : float
    """
    if not (0.0 < observed_prevalence < 1.0):
        raise ValueError(f"observed_prevalence must be in (0,1); got {observed_prevalence:.4f}")

    f = lambda lam: _weibull_prevalence(lam, eta, tau, nu) - observed_prevalence  # noqa: E731

    # Check that the bracket brackets the root
    f_lo = f(bracket[0])
    f_hi = f(bracket[1])
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Bracket {bracket} does not straddle the root "
            f"(f_lo={f_lo:.4f}, f_hi={f_hi:.4f}). "
            "Try a wider bracket or check prevalence/eta values."
        )

    return float(brentq(f, bracket[0], bracket[1], xtol=1e-8, maxiter=200))


def fit_lambda0_loglogistic(
    eta: np.ndarray,
    observed_prevalence: float,
    tau: float,
    nu: float = 1.5,
    bracket: tuple[float, float] = (1e-6, 1e2),
) -> float:
    """
    Find lambda0 for a log-logistic model such that
    E[1/(1+(λ₀·exp(η)·τ)^ν)] matches 1 − observed_prevalence.
    Uses Brent's method (same interface as fit_lambda0).
    """
    if not (0.0 < observed_prevalence < 1.0):
        raise ValueError(f"observed_prevalence must be in (0,1); got {observed_prevalence:.4f}")
    f = lambda lam: _loglogistic_prevalence(lam, eta, tau, nu) - observed_prevalence  # noqa: E731
    f_lo, f_hi = f(bracket[0]), f(bracket[1])
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Bracket {bracket} does not straddle the root "
            f"(f_lo={f_lo:.4f}, f_hi={f_hi:.4f})."
        )
    return float(brentq(f, bracket[0], bracket[1], xtol=1e-8, maxiter=200))


# ---------------------------------------------------------------------------
# Step 4 — per-domain censoring rate
# ---------------------------------------------------------------------------

def fit_censoring_rate(
    t_obs: np.ndarray,
    delta: np.ndarray,
    min_censored_fraction: float = 0.05,
) -> float:
    """
    Estimate exponential censoring rate λ_c from observed (t_obs, delta).

    Method: MLE for Exp(λ_c) fitted to the censoring times of censored
    individuals only.  Returns 0.0 if the censored fraction is too small
    to estimate reliably.

    Parameters
    ----------
    t_obs               : (n,) observed times = min(T, C)
    delta               : (n,) event indicators (1=event, 0=censored)
    min_censored_fraction: minimum fraction censored to bother estimating

    Returns
    -------
    lambda_c : float (0.0 = no censoring in this domain)
    """
    censored_mask = (delta == 0)
    censored_fraction = float(censored_mask.mean())

    if censored_fraction < min_censored_fraction:
        return 0.0

    # For Exp(λ_c): MLE is λ_c = 1 / mean(C)
    # For censored observations t_obs[i] is a lower bound on C[i]: C[i] >= t_obs[i].
    # Under exponential censoring, t_obs[censored] == C[censored] exactly.
    c_times = t_obs[censored_mask]
    if len(c_times) == 0 or c_times.mean() < 1e-8:
        return 0.0

    return float(1.0 / c_times.mean())


# ---------------------------------------------------------------------------
# Step 5 — per-domain DomainParams
# ---------------------------------------------------------------------------

def fit_domain_params(
    X_d: np.ndarray,
    t_d: np.ndarray,
    delta_d: np.ndarray,
    A: np.ndarray,
    w: np.ndarray,
    tau: float,
    nu: float,
    domain_id: int,
    n_synthetic: Optional[int] = None,
) -> DomainParams:
    """
    Fit DomainParams for one domain given its real data and shared (A, w).

    Parameters
    ----------
    X_d        : (n_d, p) standardized covariates for this domain
    t_d        : (n_d,)   observed survival times
    delta_d    : (n_d,)   event indicators
    A          : (p, k)   shared loading matrix
    w          : (k,)     shared risk weights
    tau        : prediction horizon
    nu         : Weibull shape (fixed)
    domain_id  : integer domain index
    n_synthetic: number of synthetic samples to draw; defaults to n_d

    Returns
    -------
    DomainParams with mu_inv, lambda0, censoring_lambda set from data.
    k_spu = 0: only A_inv is used (single latent pool, no spurious split).
    """
    k = A.shape[1]
    n_d = len(t_d)
    n_syn = n_synthetic if n_synthetic is not None else n_d

    # Project covariates onto latent space
    U_d = X_d @ A   # (n_d, k)

    # Per-domain latent mean (captures mean-shift between domains)
    mu_d = U_d.mean(axis=0)   # (k,)

    # Centre U for risk computation (eta uses centred factors + intercept via lambda0)
    U_centred = U_d - mu_d
    eta_d = U_centred @ w   # (n_d,)

    # Censoring-adjusted prevalence via KM estimator at tau.
    # Using raw observed proportion would be deflated by censoring (the MH bug):
    # patients censored before tau lower the observed rate even if their true
    # event probability is high.  1 - S_KM(tau) corrects for this.
    km_survival = _km_at_tau(t_d, delta_d, tau)
    km_prevalence = float(np.clip(1.0 - km_survival, 0.01, 0.99))

    # Fit baseline hazard.  This helper assumes a Weibull PH model; the
    # distribution-aware path (log-logistic) lives in fit_reference_generator.
    lambda0_d = fit_lambda0(eta_d, km_prevalence, tau=tau, nu=nu)

    # Synthetic domains are generated WITHOUT censoring.
    # lambda0 is already calibrated to the KM-adjusted true event probability,
    # so the synthetic empirical CDF at tau matches S_KM_real(tau) by design.
    # Applying censoring again would deflate prev_synth below the calibrated target.
    # KM comparison on the real side already handles real-data censoring correctly.
    return DomainParams(
        mu_inv=mu_d,
        mu_spu=np.zeros(0),      # k_spu = 0
        A_inv=A.copy(),
        A_spu=None,
        sigma_noise=float(np.std(X_d - U_d @ A.T)),
        beta0=0.0,
        w=w.copy(),
        v=np.zeros(0),           # k_spu = 0
        lambda0=lambda0_d,
        nu=nu,
        n=n_syn,
        domain_id=domain_id,
        censoring_lambda=0.0,    # uncensored synthetic
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def fit_reference_generator(
    source_centers:  list[dict],
    k_total:         int,
    tau:             float,
    nu:              float = 1.5,
    time_distribution: str = "weibull",
    external_pfs:    Optional[dict] = None,
    lam_pfs:         float = 0.5,
    genomic_distill: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[DomainParams], "StandardScaler"]:
    """
    Full fitting pipeline on pooled source centers.

    Parameters
    ----------
    source_centers    : list of dicts (keys: X, t, delta, n, center_id)
    k_total           : number of latent dimensions
    tau               : fixed prediction horizon
    nu                : shape parameter. For "weibull": fixed ν. For
                        "loglogistic": used as initial guess; MLE estimates ν
                        jointly with w and overwrites this value.
    time_distribution : "weibull" (default) or "loglogistic".
                        Controls both how w is estimated and how λ₀ is
                        calibrated per domain.
    external_pfs      : optional dict with keys
                          "X_clin"    : (N_ext, p) raw clinical features (same p as source)
                          "t_pfs"     : (N_ext,) PFS times
                          "delta_pfs" : (N_ext,) PFS event indicators
                        If provided, the loading matrix A is fit via multitask
                        optimisation (OS source + PFS external), replacing the
                        default unsupervised PCA.  All other pipeline steps
                        (risk-weight fitting, per-domain λ₀) remain unchanged.
    lam_pfs           : weight of the auxiliary PFS task in the multitask
                        loading fit (ignored when external_pfs is None).
    genomic_distill   : optional dict with keys
                          "X_geno"   : (N_ext, p_geno) genomic features aligned
                                       row-by-row with external_pfs patients
                          "lam_geno" : float weight for distillation task (default 0.3)
                        Activates Task 3 in fit_multitask_loading_matrix: a
                        genomic encoder is trained on X_geno supervised by PFS,
                        and the resulting η_geno scores are distilled into A via
                        OLS regression.  Requires external_pfs to also be set.

    Returns
    -------
    A             : (p, k_total) shared loading matrix
    w             : (k_total,) shared risk weights
    domain_params : list of DomainParams, one per source center
    scaler        : fitted StandardScaler
    """
    from sklearn.preprocessing import StandardScaler

    # --- Pool and standardise ---
    X_pooled = np.vstack([c["X"] for c in source_centers])
    scaler   = StandardScaler()
    X_pooled_s = scaler.fit_transform(X_pooled)

    sizes = [c["n"] for c in source_centers]
    X_scaled_list, idx = [], 0
    for n in sizes:
        X_scaled_list.append(X_pooled_s[idx: idx + n])
        idx += n

    # --- Step 1: shared loading matrix ---
    if external_pfs is not None:
        # Impute external features with source medians, then scale with source scaler
        X_ext_raw = external_pfs["X_clin"]
        src_meds  = np.nanmedian(X_pooled, axis=0)
        X_ext_imp = np.where(np.isnan(X_ext_raw), src_meds, X_ext_raw)
        X_ext_s   = scaler.transform(X_ext_imp)

        t_src_all = np.concatenate([c["t"]     for c in source_centers])
        d_src_all = np.concatenate([c["delta"] for c in source_centers])

        eta_geno = None
        lam_geno = 0.3
        if genomic_distill is not None:
            X_geno_raw = genomic_distill["X_geno"]
            geno_meds  = np.nanmedian(X_geno_raw, axis=0)
            X_geno_imp = np.where(np.isnan(X_geno_raw), geno_meds, X_geno_raw)
            eta_geno = compute_genomic_risk_scores(
                X_geno_imp,
                external_pfs["t_pfs"],
                external_pfs["delta_pfs"],
            )
            lam_geno = float(genomic_distill.get("lam_geno", 0.3))

        A = fit_multitask_loading_matrix(
            X_pooled_s, t_src_all, d_src_all,
            X_ext_s,    external_pfs["t_pfs"], external_pfs["delta_pfs"],
            k_total, lam=lam_pfs,
            eta_geno_ext=eta_geno, lam_geno=lam_geno,
        )
    else:
        A = fit_loading_matrix(X_pooled_s, k_total)

    U_pooled = X_pooled_s @ A   # (N, k)

    # --- Step 2: risk weights ---
    t_pooled     = np.concatenate([c["t"]     for c in source_centers])
    delta_pooled = np.concatenate([c["delta"] for c in source_centers])

    if time_distribution == "loglogistic":
        # Log-logistic MLE on (t, delta) — estimates w and ν jointly
        w, _lambda0_global, nu = fit_risk_weights_loglogistic(
            U_pooled, t_pooled, delta_pooled
        )
        _fit_lam0 = fit_lambda0_loglogistic
    elif time_distribution == "cox":
        # Cox partial likelihood — semiparametric, no baseline hazard assumed
        # λ₀ and ν still calibrated per-domain via Weibull KM matching (pragmatic)
        w = fit_risk_weights_cox(U_pooled, t_pooled, delta_pooled)
        _fit_lam0 = fit_lambda0
    else:
        # Weibull PH: LR on binary label y (default)
        y_pooled = ((t_pooled <= tau) & (delta_pooled == 1)).astype(int)
        w = fit_risk_weights(U_pooled, y_pooled)
        _fit_lam0 = fit_lambda0

    # --- Step 3: per-domain params (lambda0 calibrated per domain) ---
    domain_params = []
    for c, X_s in zip(source_centers, X_scaled_list):
        U_d   = X_s @ A
        mu_d  = U_d.mean(axis=0)
        eta_d = (U_d - mu_d) @ w

        km_survival   = _km_at_tau(c["t"], c["delta"], tau)
        km_prevalence = float(np.clip(1.0 - km_survival, 0.01, 0.99))
        lambda0_d     = _fit_lam0(eta_d, km_prevalence, tau=tau, nu=nu)

        dp = DomainParams(
            mu_inv=mu_d,
            mu_spu=np.zeros(0),
            A_inv=A.copy(),
            A_spu=None,
            sigma_noise=float(np.std(X_s - U_d @ A.T)),
            beta0=0.0,
            w=w.copy(),
            v=np.zeros(0),
            lambda0=lambda0_d,
            nu=nu,
            n=c["n"],
            domain_id=c["center_id"],
            censoring_lambda=0.0,
            time_distribution=time_distribution,
        )
        domain_params.append(dp)

    return A, w, domain_params, scaler


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_real_domain_dataset(
    X_s: np.ndarray,
    t_obs: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    w: np.ndarray,
    tau: float,
    params: DomainParams,
    domain_id: int,
) -> DomainDataset:
    """
    Wrap real domain data in a DomainDataset-like structure for comparison.

    The latent factors are estimated from data (U = X_s @ A); T and eta are
    reconstructed approximations.  u_spu is empty (k_spu = 0).

    Returns a DomainDataset with:
      u_inv  = X_s @ A (estimated latent factors)
      u_spu  = (n, 0) empty array
      x      = X_s
      eta    = estimated log-risk
      t_obs, delta set from real data
      y      = fixed-horizon label from real data
    """
    U = X_s @ A   # (n, k)
    mu = U.mean(axis=0)
    U_centred = U - mu
    eta = U_centred @ w

    # Reconstruct a plausible T (approximate; use t_obs as stand-in for observed)
    y = ((t_obs <= tau) & (delta == 1)).astype(int)

    return DomainDataset(
        domain_id=domain_id,
        u_inv=U_centred,
        u_spu=np.zeros((len(y), 0)),
        x=X_s,
        eta=eta,
        T=t_obs.copy(),   # approximate: real T is latent
        y=y,
        params=params,
        t_obs=t_obs.copy(),
        delta=delta.copy(),
    )


def fit_domain_params_fewshot(
    X_m: np.ndarray,
    A: np.ndarray,
    w: np.ndarray,
    tau: float,
    nu: float,
    domain_id: int,
    n_synthetic: int,
    source_lambda0_mean: float,
    t_m: Optional[np.ndarray] = None,
    delta_m: Optional[np.ndarray] = None,
    variant: str = "B",
    min_events_for_km: int = 3,
) -> DomainParams:
    """
    Few-shot domain parameter estimation from m held-out patients.

    variant="A"  covariate-only: mu_d from X_m; lambda0 = source_lambda0_mean.
    variant="B"  full: mu_d from X_m; lambda0 from KM(t_m, delta_m).
                 Falls back to source_lambda0_mean when fewer than
                 min_events_for_km events are in the m-patient sample
                 (KM is unreliable with very few events).

    Parameters
    ----------
    X_m                  : (m, p) standardised covariates of the calibration set
    A                    : (p, k) shared loading matrix (from source fit)
    w                    : (k,)   shared risk weights (from source fit)
    tau                  : prediction horizon
    nu                   : Weibull shape (fixed)
    domain_id            : integer center id
    n_synthetic          : number of patients to generate (typically = full n)
    source_lambda0_mean  : mean lambda0 across source domains — fallback value
    t_m                  : (m,) observed times — required for variant="B"
    delta_m              : (m,) event indicators — required for variant="B"
    variant              : "A" (covariate-only) or "B" (full few-shot)
    min_events_for_km    : minimum events in the m-sample before using KM
    """
    U_m = X_m @ A
    mu_d = U_m.mean(axis=0)
    U_centred = U_m - mu_d
    eta_m = U_centred @ w
    sigma_noise = float(np.std(X_m - U_m @ A.T))

    if variant == "A":
        lambda0_d = source_lambda0_mean
        used_km = False

    elif variant == "B":
        n_events = int(delta_m.sum()) if delta_m is not None else 0
        if n_events >= min_events_for_km and t_m is not None:
            km_survival = _km_at_tau(t_m, delta_m, tau)
            km_prev = float(np.clip(1.0 - km_survival, 0.01, 0.99))
            try:
                lambda0_d = fit_lambda0(eta_m, km_prev, tau=tau, nu=nu)
                used_km = True
            except Exception:
                lambda0_d = source_lambda0_mean
                used_km = False
        else:
            lambda0_d = source_lambda0_mean
            used_km = False
    else:
        raise ValueError(f"variant must be 'A' or 'B', got '{variant}'")

    p = DomainParams(
        mu_inv=mu_d,
        mu_spu=np.zeros(0),
        A_inv=A.copy(),
        A_spu=None,
        sigma_noise=sigma_noise,
        beta0=0.0,
        w=w.copy(),
        v=np.zeros(0),
        lambda0=lambda0_d,
        nu=nu,
        n=n_synthetic,
        domain_id=domain_id,
        censoring_lambda=0.0,
    )
    # Attach metadata so callers can inspect whether KM was actually used
    p._used_km = used_km  # type: ignore[attr-defined]
    return p


def average_domain_params(params_list: list[DomainParams]) -> DomainParams:
    """
    Return a DomainParams whose continuous fields are the mean of the list.
    Useful for constructing a 'pooled source' reference parameter set.
    """
    ref = params_list[0].copy()
    ref.mu_inv    = np.mean([p.mu_inv    for p in params_list], axis=0)
    ref.lambda0   = float(np.mean([p.lambda0   for p in params_list]))
    ref.sigma_noise = float(np.mean([p.sigma_noise for p in params_list]))
    ref.censoring_lambda = float(np.mean([p.censoring_lambda for p in params_list]))
    return ref
