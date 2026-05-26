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
from utils.metrics import auroc, auprc, brier, ece, prevalence

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pool_sources(datasets: list[DomainDataset]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([ds.x for ds in datasets])
    y = np.concatenate([ds.y for ds in datasets])
    return X, y


def _score(y_true: np.ndarray, probs: np.ndarray, n: int) -> dict:
    return {
        "auroc":      round(auroc(y_true, probs), 4),
        "auprc":      round(auprc(y_true, probs), 4),
        "brier":      round(brier(y_true, probs), 4),
        "ece":        round(ece(y_true, probs), 4),
        "prevalence": round(prevalence(y_true), 4),
        "n": n,
    }


def _nan_score(n: int = 0) -> dict:
    d = {k: float("nan") for k in ["auroc", "auprc", "brier", "ece", "prevalence"]}
    d["n"] = n
    return d


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

def _build_networks(input_dim: int, seed: int = 0, task_softmax: bool = False):
    """
    Build fresh Keras encoder, task, and discriminator Sequential models.

    Called once per (method, seed) to avoid state leakage across runs.

    Architecture
    ~~~~~~~~~~~~
    encoder      : Dense(64, relu) → Dense(32, relu)  [32-dim embedding]
    task (DANN)  : Dense(1, sigmoid)                   [binary — ADAPT reshapes to match y shape]
    task (CDAN)  : Dense(2, softmax)                   [multiclass — required by CDAN multilinear map]
    discriminator: Dense(32, relu) → Dense(1, sigmoid) [domain classifier]
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    tf.random.set_seed(seed)

    encoder = Sequential([
        Dense(_ENC_UNITS[0], activation="relu", input_shape=(input_dim,)),
        Dense(_ENC_UNITS[1], activation="relu"),
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

def fit_dann(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    lambda_: float = _DANN_LAMBDA,
    epochs: int = _DANN_EPOCHS,
    batch_size: int = _DANN_BATCH,
    seed: int = 0,
) -> dict:
    """
    DANN: Domain-Adversarial Neural Network.

    The encoder jointly minimises task loss and maximises domain discriminator
    loss (via gradient reversal).  Only source labels are used; target data is
    passed as unlabeled (yt=None, the ADAPT default).

    On training failure the function catches the exception and returns NaN
    scores so multi-seed runs are not aborted.
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

    tf.random.set_seed(seed)
    encoder, task, disc = _build_networks(X_tr_s.shape[1], seed=seed)

    model = _DANN(
        encoder=encoder,
        task=task,
        discriminator=disc,
        Xt=X_tgt_s,
        lambda_=lambda_,
        verbose=0,
        random_state=seed,
        optimizer=tf.keras.optimizers.legacy.RMSprop(),
    )

    try:
        model.fit(X_tr_s, y_tr.astype("float32"), epochs=epochs, batch_size=batch_size)
    except Exception as exc:
        warnings.warn(f"DANN training failed (seed={seed}): {exc}")
        return {
            "feature_set":    "dann",
            "error":          str(exc),
            "source_domains": {},
            "source_pooled":  _nan_score(len(y_val)),
            "target":         _nan_score(target_dataset.n),
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
    }


# ---------------------------------------------------------------------------
# CDAN
# ---------------------------------------------------------------------------

def fit_cdan(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    lambda_: float = _CDAN_LAMBDA,
    epochs: int = _DANN_EPOCHS,
    batch_size: int = _DANN_BATCH,
    seed: int = 0,
) -> dict:
    """
    CDAN: Conditional Domain-Adversarial Network.

    Extends DANN by conditioning the domain discriminator on the task-network
    prediction via a multilinear map (outer product of encoder output and task
    output).  With encoder dim=32 and task dim=1 the product dim is 32, so
    max_features=4096 (default) is never hit.

    Same training protocol and failure handling as DANN.
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

    tf.random.set_seed(seed)
    encoder, task, disc = _build_networks(X_tr_s.shape[1], seed=seed, task_softmax=False)

    model = _CDAN(
        encoder=encoder,
        task=task,
        discriminator=disc,
        Xt=X_tgt_s,
        lambda_=lambda_,
        verbose=0,
        random_state=seed,
        optimizer=tf.keras.optimizers.legacy.RMSprop(),
    )

    try:
        model.fit(X_tr_s, y_tr.astype("float32"), epochs=epochs, batch_size=batch_size)
    except Exception as exc:
        warnings.warn(f"CDAN training failed (seed={seed}): {exc}")
        return {
            "feature_set":    "cdan",
            "error":          str(exc),
            "source_domains": {},
            "source_pooled":  _nan_score(len(y_val)),
            "target":         _nan_score(target_dataset.n),
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
