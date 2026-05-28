"""
Domain adaptation method wrappers for the SHIFT-SURV benchmark.

Methods
-------
CORAL  — CORrelation ALignment (Sun & Saenko 2016), sklearn-compatible.
         Aligns second-order statistics of source features to target covariance;
         downstream estimator is a Logistic Regression.

DANN   — Domain-Adversarial Neural Network (Ganin et al. 2016), TF/Keras.
         Gradient reversal on a domain discriminator during task training.

CDAN   — Conditional Domain-Adversarial Network (Long et al. 2018), TF/Keras.
         Conditions domain discrimination on the task-network prediction via a
         multilinear map; otherwise identical architecture to DANN.

Training protocol (identical to baselines.py)
---------------------------------------------
  - Pooled source data split 80/20 (stratified) → train / held-out val.
  - StandardScaler fitted on source train split only; applied to val and target.
  - Target labels are NEVER seen during training or model selection.
  - Return-dict format matches baselines._fit_and_score():
      {feature_set, source_domains, source_pooled, target, model, scaler}

Dependencies
------------
  pip install adapt==0.4.5    (installs scikeras, cvxopt as well)
  TensorFlow 2.13 (already present in the project environment)
"""
from __future__ import annotations

import os
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from generator.base_generator import DomainDataset
from utils.metrics import auroc, auprc, ece, prevalence, f1, sensitivity, specificity

# Suppress TF C++ startup messages before the first TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from adapt.feature_based import CORAL as _CORAL   # noqa: E402
from adapt.feature_based import DANN  as _DANN    # noqa: E402
from adapt.feature_based import CDAN  as _CDAN    # noqa: E402


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

_LR_KWARGS = dict(max_iter=500, solver="lbfgs", C=1.0, random_state=0)

# Encoder layers (input → hidden → embedding)
_ENC_UNITS   = (64, 32)
# Discriminator hidden layer
_DISC_HIDDEN = 32
# DANN / CDAN training defaults
_DANN_EPOCHS  = 50
_DANN_BATCH   = 64
_DANN_LAMBDA  = 1.0
_CDAN_LAMBDA  = 0.1

# IWCV grid-search defaults
_DANN_EPOCHS_SEARCH  = 30    # epochs per candidate during HP search
_DANN_EPOCHS_FINAL   = 100   # epochs for final retraining of best config
_DEFAULT_LAMBDA_GRID = [0.001, 0.01, 0.1, 0.3, 1.0]
_DEFAULT_ARCH_GRID   = [(32, 16), (64, 32)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pool_sources(datasets: list[DomainDataset]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([ds.x for ds in datasets])
    y = np.concatenate([ds.y for ds in datasets])
    return X, y


def _score(y_true: np.ndarray, probs: np.ndarray, n: int) -> dict:
    return {
        "auroc":       round(auroc(y_true, probs), 4),
        "auprc":       round(auprc(y_true, probs), 4),
        "ece":         round(ece(y_true, probs), 4),
        "f1":          round(f1(y_true, probs), 4),
        "sensitivity": round(sensitivity(y_true, probs), 4),
        "specificity": round(specificity(y_true, probs), 4),
        "prevalence":  round(prevalence(y_true), 4),
        "n": n,
    }


def _nan_score(n: int = 0) -> dict:
    d = {k: float("nan") for k in ["auroc", "auprc", "ece", "f1", "sensitivity", "specificity", "prevalence"]}
    d["n"] = n
    return d


def _importance_weights(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """Domain-classifier importance weights w(x) = P(tgt|x)/P(src|x), clipped to [0.05, 20]."""
    X = np.vstack([X_src, X_tgt])
    y = np.array([0] * len(X_src) + [1] * len(X_tgt), dtype=float)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=0, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X, y)
    p_tgt = clf.predict_proba(X_src)[:, 1]
    p_src = np.clip(1.0 - p_tgt, 1e-6, None)
    return np.clip(p_tgt / p_src, 0.05, 20.0)


def _weighted_auroc(y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    """AUROC with importance weights; falls back to unweighted on failure."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probs, sample_weight=weights))
    except Exception:
        return float(roc_auc_score(y_true, probs))


# ---------------------------------------------------------------------------
# CORAL
# ---------------------------------------------------------------------------

def fit_coral(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    lambda_: float = 1.0,
    seed: int = 0,
) -> dict:
    """
    CORAL: CORrelation ALignment.

    CORAL computes a linear whitening–recoloring transformation W such that
    transformed source features share the same covariance as the target.
    The LR estimator is then trained on the aligned source features.

    Prediction convention:
      Source val   → coral.transform(X_val_scaled) → estimator_.predict_proba
      Target       → estimator_.predict_proba(X_tgt_scaled) directly
                     (target features are already in target-space; transform
                     would over-rotate them and degrade performance)
    """
    X_all, y_all = _pool_sources(source_datasets)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all,
    )

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_tgt_s = scaler.transform(target_dataset.x)

    model = _CORAL(
        estimator=LogisticRegression(**_LR_KWARGS),
        Xt=X_tgt_s,
        lambda_=lambda_,
        verbose=0,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_tr_s, y_tr)

    # Source val: apply whitening–recoloring (domain="src" required by ADAPT API)
    probs_val = model.estimator_.predict_proba(model.transform(X_val_s, domain="src"))[:, 1]
    # Target: already lives in target-covariance space
    probs_tgt = model.estimator_.predict_proba(X_tgt_s)[:, 1]

    source_scores = {}
    for ds in source_datasets:
        X_d_s = scaler.transform(ds.x)
        p_d   = model.estimator_.predict_proba(model.transform(X_d_s, domain="src"))[:, 1]
        source_scores[f"source_d{ds.domain_id}"] = _score(ds.y, p_d, ds.n)

    return {
        "feature_set":    "coral",
        "source_domains": source_scores,
        "source_pooled":  _score(y_val, probs_val, len(y_val)),
        "target":         _score(target_dataset.y, probs_tgt, target_dataset.n),
        "model":          model,
        "scaler":         scaler,
    }


# ---------------------------------------------------------------------------
# Shared Keras architecture factory for DANN / CDAN
# ---------------------------------------------------------------------------

def _build_networks(
    input_dim: int,
    seed: int = 0,
    task_softmax: bool = False,
    enc_units: tuple = _ENC_UNITS,
):
    """
    Build fresh Keras encoder, task, and discriminator Sequential models.

    Called once per (method, seed) to avoid state leakage across runs.

    Architecture
    ~~~~~~~~~~~~
    encoder      : Dense(enc_units[0], relu) → Dense(enc_units[1], relu)
    task (DANN)  : Dense(1, sigmoid)
    task (CDAN)  : Dense(2, softmax)
    discriminator: Dense(32, relu) → Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dense

    tf.random.set_seed(seed)

    encoder = Sequential([
        Input(shape=(input_dim,)),
        Dense(enc_units[0], activation="relu"),
        Dense(enc_units[1], activation="relu"),
    ], name="encoder")

    if task_softmax:
        task = Sequential([Dense(2, activation="softmax")], name="task")
    else:
        task = Sequential([Dense(1, activation="sigmoid")], name="task")

    discriminator = Sequential([
        Dense(_DISC_HIDDEN, activation="relu"),
        Dense(1, activation="sigmoid"),
    ], name="discriminator")

    return encoder, task, discriminator


def _squeeze_proba(model, X: np.ndarray) -> np.ndarray:
    """Call model.predict; return positive-class probability (col 1 for softmax, squeeze for sigmoid)."""
    out = model.predict(X, verbose=0)
    out = np.squeeze(out).astype(float)
    if out.ndim == 2:
        return out[:, 1]
    return out


# ---------------------------------------------------------------------------
# DANN
# ---------------------------------------------------------------------------

def _fit_dann_single(
    X_tr_s, y_tr, X_tgt_s, lambda_, enc_units, epochs, batch_size, seed,
    X_val_s=None, y_val=None, early_stopping_patience=0,
):
    """Train one DANN candidate; returns (model, None) or (None, error_str)."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(seed)
    encoder, task, disc = _build_networks(
        X_tr_s.shape[1], seed=seed, enc_units=enc_units,
    )
    model = _DANN(
        encoder=encoder, task=task, discriminator=disc,
        Xt=X_tgt_s, lambda_=lambda_, verbose=0, random_state=seed,
        optimizer=tf.keras.optimizers.RMSprop(),
    )
    fit_kwargs = {"epochs": epochs, "batch_size": batch_size}
    if early_stopping_patience > 0 and X_val_s is not None:
        fit_kwargs["validation_data"] = (X_val_s, y_val.astype("float32"))
        fit_kwargs["callbacks"] = [EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience,
            restore_best_weights=True, min_delta=1e-4,
        )]
    try:
        model.fit(X_tr_s, y_tr.astype("float32"), **fit_kwargs)
        hist = {k: [float(v) for v in vals]
                for k, vals in getattr(model, "history_", {}).items()
                if isinstance(vals, (list, np.ndarray))}
        return model, None, hist
    except Exception as exc:
        return None, str(exc), {}


def fit_dann(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    lambda_: float = _DANN_LAMBDA,
    epochs: int = _DANN_EPOCHS,
    batch_size: int = _DANN_BATCH,
    seed: int = 0,
    # IWCV grid-search parameters (None → legacy single-fit behaviour)
    lambda_grid: list | None = None,
    arch_grid: list | None = None,
    use_iwcv: bool = True,
    epochs_search: int = _DANN_EPOCHS_SEARCH,
    epochs_final: int = _DANN_EPOCHS_FINAL,
) -> dict:
    """
    DANN: Domain-Adversarial Neural Network.

    When *lambda_grid* and *arch_grid* are provided (or *use_iwcv* is True with
    defaults), the function performs an IWCV grid search:
      1. Estimate importance weights w(x) via a domain logistic classifier.
      2. For each (lambda, arch) candidate, train for *epochs_search* epochs and
         score the source-validation set under weights w — a proxy for
         target AUROC without using target labels.
      3. Retrain the best config for *epochs_final* epochs.

    When both grids are None and *use_iwcv* is False, falls back to the
    original single-fit behaviour for backward compatibility.
    """
    import tensorflow as tf

    X_all, y_all = _pool_sources(source_datasets)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all,
    )

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_tgt_s = scaler.transform(target_dataset.x).astype("float32")

    # Resolve whether to run IWCV
    run_iwcv = use_iwcv or (lambda_grid is not None) or (arch_grid is not None)
    if run_iwcv:
        lam_grid  = lambda_grid if lambda_grid is not None else _DEFAULT_LAMBDA_GRID
        a_grid    = arch_grid   if arch_grid   is not None else _DEFAULT_ARCH_GRID
        iw        = _importance_weights(X_tr_s, X_tgt_s)
        best_score, best_cfg = -1.0, (lam_grid[0], a_grid[0])
        hp_rows = []
        for lam in lam_grid:
            for arch in a_grid:
                m, err, _ = _fit_dann_single(
                    X_tr_s, y_tr, X_tgt_s, lam, arch, epochs_search, batch_size, seed,
                    X_val_s=X_val_s, y_val=y_val, early_stopping_patience=5,
                )
                if m is None:
                    score = float("nan")
                else:
                    p_val = _squeeze_proba(m, X_val_s)
                    score = _weighted_auroc(y_val, p_val, iw)
                hp_rows.append({"lambda": lam, "arch": arch, "iwcv_auroc": score})
                if not np.isnan(score) and score > best_score:
                    best_score, best_cfg = score, (lam, arch)
        best_lam, best_arch = best_cfg
        model, err, loss_history = _fit_dann_single(
            X_tr_s, y_tr, X_tgt_s, best_lam, best_arch, epochs_final, batch_size, seed,
            X_val_s=X_val_s, y_val=y_val, early_stopping_patience=10,
        )
        chosen_lambda, chosen_arch = best_lam, best_arch
    else:
        hp_rows = []
        model, err, loss_history = _fit_dann_single(
            X_tr_s, y_tr, X_tgt_s, lambda_, _ENC_UNITS, epochs, batch_size, seed,
        )
        chosen_lambda, chosen_arch = lambda_, _ENC_UNITS

    if model is None:
        warnings.warn(f"DANN training failed (seed={seed}): {err}")
        return {
            "feature_set":    "dann",
            "error":          err,
            "source_domains": {},
            "source_pooled":  _nan_score(len(y_val)),
            "target":         _nan_score(target_dataset.n),
            "hp": {"lambda": chosen_lambda, "arch": chosen_arch, "hp_rows": hp_rows},
            "loss_history":   {},
        }

    probs_val = _squeeze_proba(model, X_val_s)
    probs_tgt = _squeeze_proba(model, X_tgt_s)

    source_scores = {}
    for ds in source_datasets:
        X_d_s = scaler.transform(ds.x).astype("float32")
        source_scores[f"source_d{ds.domain_id}"] = _score(ds.y, _squeeze_proba(model, X_d_s), ds.n)

    return {
        "feature_set":    "dann",
        "source_domains": source_scores,
        "source_pooled":  _score(y_val, probs_val, len(y_val)),
        "target":         _score(target_dataset.y, probs_tgt, target_dataset.n),
        "model":          model,
        "scaler":         scaler,
        "hp":             {"lambda": chosen_lambda, "arch": chosen_arch, "hp_rows": hp_rows},
        "loss_history":   loss_history,
    }


# ---------------------------------------------------------------------------
# CDAN
# ---------------------------------------------------------------------------

def _fit_cdan_single(
    X_tr_s, y_tr, X_tgt_s, lambda_, enc_units, epochs, batch_size, seed,
    X_val_s=None, y_val=None, early_stopping_patience=0,
):
    """Train one CDAN candidate; returns (model, None) or (None, error_str)."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(seed)
    encoder, task, disc = _build_networks(
        X_tr_s.shape[1], seed=seed, task_softmax=False, enc_units=enc_units,
    )
    model = _CDAN(
        encoder=encoder, task=task, discriminator=disc,
        Xt=X_tgt_s, lambda_=lambda_, verbose=0, random_state=seed,
        optimizer=tf.keras.optimizers.RMSprop(),
    )
    fit_kwargs = {"epochs": epochs, "batch_size": batch_size}
    if early_stopping_patience > 0 and X_val_s is not None:
        fit_kwargs["validation_data"] = (X_val_s, y_val.astype("float32"))
        fit_kwargs["callbacks"] = [EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience,
            restore_best_weights=True, min_delta=1e-4,
        )]
    try:
        model.fit(X_tr_s, y_tr.astype("float32"), **fit_kwargs)
        hist = {k: [float(v) for v in vals]
                for k, vals in getattr(model, "history_", {}).items()
                if isinstance(vals, (list, np.ndarray))}
        return model, None, hist
    except Exception as exc:
        return None, str(exc), {}


def fit_cdan(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    lambda_: float = _CDAN_LAMBDA,
    epochs: int = _DANN_EPOCHS,
    batch_size: int = _DANN_BATCH,
    seed: int = 0,
    # IWCV grid-search parameters (None → legacy single-fit behaviour)
    lambda_grid: list | None = None,
    arch_grid: list | None = None,
    use_iwcv: bool = True,
    epochs_search: int = _DANN_EPOCHS_SEARCH,
    epochs_final: int = _DANN_EPOCHS_FINAL,
) -> dict:
    """
    CDAN: Conditional Domain-Adversarial Network.

    Extends DANN by conditioning the domain discriminator on the task-network
    prediction via a multilinear map.  With encoder dim=32 and task dim=1 the
    product dimension is 32, so max_features=4096 (default) is never hit.

    IWCV grid-search follows the same protocol as fit_dann().
    """
    import tensorflow as tf

    X_all, y_all = _pool_sources(source_datasets)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all,
    )

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_tgt_s = scaler.transform(target_dataset.x).astype("float32")

    run_iwcv = use_iwcv or (lambda_grid is not None) or (arch_grid is not None)
    if run_iwcv:
        lam_grid  = lambda_grid if lambda_grid is not None else _DEFAULT_LAMBDA_GRID
        a_grid    = arch_grid   if arch_grid   is not None else _DEFAULT_ARCH_GRID
        iw        = _importance_weights(X_tr_s, X_tgt_s)
        best_score, best_cfg = -1.0, (lam_grid[0], a_grid[0])
        hp_rows = []
        for lam in lam_grid:
            for arch in a_grid:
                m, err, _ = _fit_cdan_single(
                    X_tr_s, y_tr, X_tgt_s, lam, arch, epochs_search, batch_size, seed,
                    X_val_s=X_val_s, y_val=y_val, early_stopping_patience=5,
                )
                if m is None:
                    score = float("nan")
                else:
                    p_val = _squeeze_proba(m, X_val_s)
                    score = _weighted_auroc(y_val, p_val, iw)
                hp_rows.append({"lambda": lam, "arch": arch, "iwcv_auroc": score})
                if not np.isnan(score) and score > best_score:
                    best_score, best_cfg = score, (lam, arch)
        best_lam, best_arch = best_cfg
        model, err, loss_history = _fit_cdan_single(
            X_tr_s, y_tr, X_tgt_s, best_lam, best_arch, epochs_final, batch_size, seed,
            X_val_s=X_val_s, y_val=y_val, early_stopping_patience=10,
        )
        chosen_lambda, chosen_arch = best_lam, best_arch
    else:
        hp_rows = []
        model, err, loss_history = _fit_cdan_single(
            X_tr_s, y_tr, X_tgt_s, lambda_, _ENC_UNITS, epochs, batch_size, seed,
        )
        chosen_lambda, chosen_arch = lambda_, _ENC_UNITS

    if model is None:
        warnings.warn(f"CDAN training failed (seed={seed}): {err}")
        return {
            "feature_set":    "cdan",
            "error":          err,
            "source_domains": {},
            "source_pooled":  _nan_score(len(y_val)),
            "target":         _nan_score(target_dataset.n),
            "hp":             {"lambda": chosen_lambda, "arch": chosen_arch, "hp_rows": hp_rows},
            "loss_history":   {},
        }

    probs_val = _squeeze_proba(model, X_val_s)
    probs_tgt = _squeeze_proba(model, X_tgt_s)

    source_scores = {}
    for ds in source_datasets:
        X_d_s = scaler.transform(ds.x).astype("float32")
        source_scores[f"source_d{ds.domain_id}"] = _score(ds.y, _squeeze_proba(model, X_d_s), ds.n)

    return {
        "feature_set":    "cdan",
        "source_domains": source_scores,
        "source_pooled":  _score(y_val, probs_val, len(y_val)),
        "target":         _score(target_dataset.y, probs_tgt, target_dataset.n),
        "model":          model,
        "scaler":         scaler,
        "hp":             {"lambda": chosen_lambda, "arch": chosen_arch, "hp_rows": hp_rows},
        "loss_history":   loss_history,
    }


# ---------------------------------------------------------------------------
# IRM — Invariant Risk Minimization (Arjovsky et al. 2019)
# ---------------------------------------------------------------------------

def fit_irm(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    lam: float = 1e2,
    n_iter: int = 300,
    seed: int = 0,
) -> dict:
    """
    IRM: Invariant Risk Minimization (Arjovsky et al. 2019).

    Finds a linear classifier w such that w is simultaneously optimal in every
    source environment via a gradient penalty:

        min_w  Σ_e CE(X_e w, y_e)  +  λ · Σ_e ||X_e^T(σ(X_e w)-y_e)/n_e||²

    Fixed λ=100 (standard in the IRM literature for tabular data).
    No inner CV — λ grid search is prohibitively slow at benchmark scale.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    scaler = StandardScaler()
    X_all, y_all = _pool_sources(source_datasets)
    scaler.fit(X_all)

    envs    = [(scaler.transform(ds.x), ds.y.astype(float))
               for ds in source_datasets]
    X_tgt_s = scaler.transform(target_dataset.x)
    y_src_all = np.concatenate([y for _, y in envs])
    p = envs[0][0].shape[1]

    def _obj_and_grad(w):
        loss = 0.0
        grad = np.zeros(p)
        for X, y in envs:
            probs = expit(X @ w)
            r     = probs - y
            # CE loss
            loss += float(np.mean(np.log1p(np.exp(-(2*y-1) * (X @ w)))))
            grad += X.T @ r / len(y)
            # IRM penalty: ||X^T r / n||²  and its gradient
            g_env  = X.T @ r / len(y)
            loss  += lam * float(g_env @ g_env)
            # d/dw lam*||g||² = 2*lam * J^T g  where J_{kj} = dg_k/dw_j
            # g_k = Σ_i X_ik*(σ_i-y_i)/n  →  dg_k/dw_j = Σ_i X_ik*X_ij*σ_i(1-σ_i)/n
            diag_H = probs * (1 - probs)          # (n,)
            JtG    = (X.T * diag_H) @ (X @ g_env) / len(y)
            grad  += 2.0 * lam * JtG
        return loss / len(envs), grad / len(envs)

    rng = np.random.default_rng(seed)
    w0  = rng.standard_normal(p) * 0.01
    res = minimize(_obj_and_grad, w0, jac=True, method="L-BFGS-B",
                   options={"maxiter": n_iter, "ftol": 1e-8, "gtol": 1e-5})
    w_opt = res.x

    # Intercept from pooled source prevalence
    b_opt = float(np.log(y_src_all.mean() / (1 - y_src_all.mean() + 1e-8)))

    def _predict(X):
        return expit(X @ w_opt + b_opt)

    source_scores = {}
    for ds in source_datasets:
        X_d = scaler.transform(ds.x)
        source_scores[f"source_d{ds.domain_id}"] = _score(ds.y, _predict(X_d), ds.n)

    X_src_pool = np.vstack([X for X, _ in envs])

    return {
        "feature_set":    "irm",
        "source_domains": source_scores,
        "source_pooled":  _score(y_src_all, _predict(X_src_pool), len(y_src_all)),
        "target":         _score(target_dataset.y, _predict(X_tgt_s), target_dataset.n),
        "model":          {"w": w_opt, "b": b_opt, "lam": lam},
        "scaler":         scaler,
    }


# ---------------------------------------------------------------------------
# Convenience: run all DA methods together
# ---------------------------------------------------------------------------

def evaluate_all_adaptation(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    seed: int = 0,
    run_cdan: bool = True,
) -> list[dict]:
    """Run CORAL, DANN, and optionally CDAN; return list of result dicts."""
    results = [
        fit_coral(source_datasets, target_dataset, seed=seed),
        fit_dann(source_datasets, target_dataset, seed=seed),
    ]
    if run_cdan:
        results.append(fit_cdan(source_datasets, target_dataset, seed=seed))
    return results
