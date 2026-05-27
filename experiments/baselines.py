"""
Logistic regression baselines for the sanity-check phase.

Three feature sets, following the paper's oracle-diagnostic framing:
  - invariant_only  : use u_inv (the true invariant latent factors — oracle)
  - spurious_only   : use u_spu (the true spurious latent factors — oracle)
  - all_observed    : use x (the observed covariates as a downstream model would)

Since we are in the synthetic setting, the oracle views (u_inv, u_spu) are
available and serve as upper / lower bounds for what is achievable by a method
that perfectly isolates invariant vs spurious structure.

Return format
-------------
Each `fit_*` function returns a dict:
  {
    "model"      : fitted LogisticRegression,
    "feature_set": str,
    "source_auroc": float,
    "target_auroc": float,
    ...
  }

`evaluate_all_baselines` returns a list of such dicts, one per feature set.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from generator.base_generator import DomainDataset
from utils.metrics import auroc, auprc, brier, ece, prevalence


_LR_KWARGS = dict(max_iter=500, solver="lbfgs", C=1.0, random_state=0)
_MLP_KWARGS = dict(
    hidden_layer_sizes=(32, 16),   # halved from (64,32); reduces ~3 400 → ~900 params
    activation="relu",
    solver="lbfgs",
    alpha=0.01,                    # L2 regularisation — critical for n≈1 200 samples
    max_iter=1000,
    tol=1e-4,
    random_state=0,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pool_sources(datasets: list[DomainDataset], attr: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate feature attr and labels across source domains."""
    X = np.vstack([getattr(ds, attr) for ds in datasets])
    y = np.concatenate([ds.y for ds in datasets])
    return X, y


def _fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    feature_attr: str,
    feature_set: str,
    model,
    val_size: float = 0.2,
) -> dict:
    """
    Fit a logistic regression on 80% of the pooled source data and evaluate on:
      - a 20% held-out source validation set  (reported as source_pooled)
      - each source domain individually        (in-sample, for diagnostics only)
      - the target                             (never seen during training)

    Using a held-out validation set for source_pooled ensures the source AUROC
    is an unbiased estimate, making the source → target gap a clean measure of
    transfer degradation rather than a mix of overfitting and shift.
    """
    # Split pooled source into train and validation before fitting anything
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=0,
        stratify=y_train,
    )

    # Handle degenerate case
    if len(np.unique(y_tr)) < 2:
        return {"feature_set": feature_set, "error": "single class in training set"}

    # Scaler fitted on train split only — applied consistently to val and target
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_tr_scaled, y_tr)

    def _score_ds(ds: DomainDataset) -> dict:
        X_ds = scaler.transform(getattr(ds, feature_attr))
        probs = model.predict_proba(X_ds)[:, 1]
        return {
            "auroc": round(auroc(ds.y, probs), 4),
            "auprc": round(auprc(ds.y, probs), 4),
            "brier": round(brier(ds.y, probs), 4),
            "ece": round(ece(ds.y, probs), 4),
            "prevalence": round(prevalence(ds.y), 4),
            "n": ds.n,
        }

    # Per-domain scores are in-sample (full domain data) — kept for diagnostics
    source_scores = {f"source_d{ds.domain_id}": _score_ds(ds) for ds in source_datasets}

    # Held-out validation score — unbiased estimate of source performance
    probs_val = model.predict_proba(scaler.transform(X_val))[:, 1]
    source_pooled = {
        "auroc": round(auroc(y_val, probs_val), 4),
        "auprc": round(auprc(y_val, probs_val), 4),
        "brier": round(brier(y_val, probs_val), 4),
        "ece": round(ece(y_val, probs_val), 4),
        "prevalence": round(prevalence(y_val), 4),
        "n": len(y_val),
    }

    target_score = _score_ds(target_dataset)

    return {
        "feature_set": feature_set,
        "source_domains": source_scores,
        "source_pooled": source_pooled,
        "target": target_score,
        "model": model,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_invariant_oracle(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> dict:
    """LR oracle on true invariant latent factors u_inv."""
    X_train, y_train = _pool_sources(source_datasets, "u_inv")
    return _fit_and_score(X_train, y_train, source_datasets, target_dataset,
                          feature_attr="u_inv", feature_set="invariant_oracle",
                          model=LogisticRegression(**_LR_KWARGS))


def fit_spurious_oracle(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> dict:
    """LR oracle on true spurious latent factors u_spu."""
    X_train, y_train = _pool_sources(source_datasets, "u_spu")
    return _fit_and_score(X_train, y_train, source_datasets, target_dataset,
                          feature_attr="u_spu", feature_set="spurious_oracle",
                          model=LogisticRegression(**_LR_KWARGS))


def fit_all_observed(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> dict:
    """LR on all observed covariates x — standard ERM baseline."""
    X_train, y_train = _pool_sources(source_datasets, "x")
    return _fit_and_score(X_train, y_train, source_datasets, target_dataset,
                          feature_attr="x", feature_set="all_observed",
                          model=LogisticRegression(**_LR_KWARGS))


def fit_prior_recalib(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    n_em_iter: int = 50,
) -> dict:
    """
    Prior-shift recalibration (Saerens, Latinne & Decaestecker 2002).

    Fits an ERM LR on source, then estimates the target class prevalence π_tgt
    from unlabeled target data via EM and rescales predictions accordingly.
    No target labels are used.

    EM update
    ---------
    E-step:  p̃(y=1|xᵢ) = f(xᵢ)·(π_tgt/π_src) / [f(xᵢ)·(π_tgt/π_src) + (1-f(xᵢ))·((1-π_tgt)/(1-π_src))]
    M-step:  π_tgt ← mean(p̃(y=1|xᵢ))  over unlabeled target

    Appropriate for: baseline_hazard_shift (stable P(Y|X), shifted class prior).
    No benefit for: concept_shift (P(Y|X) changes), covariate_shift without prior shift.
    AUROC: unchanged (rank-invariant to prior rescaling).
    ECE/Brier: substantially improved when the prior shifted.
    """
    X_all, y_all = _pool_sources(source_datasets, "x")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all,
    )

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_tgt_s = scaler.transform(target_dataset.x)

    model = LogisticRegression(**_LR_KWARGS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_tr_s, y_tr)

    pi_src    = float(y_tr.mean())
    p_raw     = model.predict_proba(X_tgt_s)[:, 1]   # P_src(Y=1|x) on target x

    # EM: estimate π_tgt from unlabeled target
    pi_tgt = float(p_raw.mean())
    for _ in range(n_em_iter):
        num   = p_raw * (pi_tgt / pi_src)
        den   = num + (1.0 - p_raw) * ((1.0 - pi_tgt) / (1.0 - pi_src))
        p_adj = num / np.clip(den, 1e-12, None)
        pi_new = float(p_adj.mean())
        if abs(pi_new - pi_tgt) < 1e-7:
            break
        pi_tgt = pi_new

    def _sc(y, p):
        return {
            "auroc":      round(auroc(y, p), 4),
            "auprc":      round(auprc(y, p), 4),
            "brier":      round(brier(y, p), 4),
            "ece":        round(ece(y, p), 4),
            "prevalence": round(prevalence(y), 4),
            "n":          len(y),
        }

    source_scores = {}
    for ds in source_datasets:
        X_d = scaler.transform(ds.x)
        source_scores[f"source_d{ds.domain_id}"] = _sc(ds.y, model.predict_proba(X_d)[:, 1])

    return {
        "feature_set":      "prior_recalib",
        "source_domains":   source_scores,
        "source_pooled":    _sc(y_val, model.predict_proba(X_val_s)[:, 1]),
        "target":           _sc(target_dataset.y, p_adj),
        "model":            model,
        "scaler":           scaler,
        "pi_src":           pi_src,
        "pi_tgt_estimated": pi_tgt,
    }


def fit_invariant_oracle_mlp(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> dict:
    """MLP oracle on true invariant latent factors u_inv."""
    X_train, y_train = _pool_sources(source_datasets, "u_inv")
    return _fit_and_score(X_train, y_train, source_datasets, target_dataset,
                          feature_attr="u_inv", feature_set="invariant_oracle_mlp",
                          model=MLPClassifier(**_MLP_KWARGS))


def fit_spurious_oracle_mlp(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> dict:
    """MLP oracle on true spurious latent factors u_spu."""
    X_train, y_train = _pool_sources(source_datasets, "u_spu")
    return _fit_and_score(X_train, y_train, source_datasets, target_dataset,
                          feature_attr="u_spu", feature_set="spurious_oracle_mlp",
                          model=MLPClassifier(**_MLP_KWARGS))


def fit_all_observed_mlp(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> dict:
    """MLP on all observed covariates x — non-linear ERM baseline."""
    X_train, y_train = _pool_sources(source_datasets, "x")
    return _fit_and_score(X_train, y_train, source_datasets, target_dataset,
                          feature_attr="x", feature_set="all_observed_mlp",
                          model=MLPClassifier(**_MLP_KWARGS))


def evaluate_all_baselines(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
) -> list[dict]:
    """Run all six baselines (LR + MLP × three feature sets) and return results list."""
    return [
        fit_invariant_oracle(source_datasets, target_dataset),
        fit_spurious_oracle(source_datasets, target_dataset),
        fit_all_observed(source_datasets, target_dataset),
        fit_invariant_oracle_mlp(source_datasets, target_dataset),
        fit_spurious_oracle_mlp(source_datasets, target_dataset),
        fit_all_observed_mlp(source_datasets, target_dataset),
    ]


def baselines_to_flat_rows(results: list[dict]) -> list[dict]:
    """
    Flatten baseline results to a list of rows suitable for a summary CSV.
    Each row: feature_set, split, domain, auroc, auprc, brier, prevalence, n
    """
    rows = []
    for r in results:
        fs = r.get("feature_set", "?")
        if "error" in r:
            rows.append({"feature_set": fs, "error": r["error"]})
            continue

        # Source domains individually
        for d_key, scores in r["source_domains"].items():
            rows.append({"feature_set": fs, "split": "source", "domain": d_key, **scores})

        # Pooled source
        rows.append({"feature_set": fs, "split": "source_pooled", "domain": "all", **r["source_pooled"]})

        # Target
        rows.append({"feature_set": fs, "split": "target", "domain": "target", **r["target"]})

    return rows
