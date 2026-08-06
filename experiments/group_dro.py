"""
Group Distributionally Robust Optimization (GroupDRO) baseline.

Sagawa, Koh, Hashimoto & Liang, "Distributionally Robust Neural Networks for
Group Shifts" (ICLR 2020).

Why it is here
--------------
CORAL/DANN/CDAN are *domain-adaptation* methods that use the unlabeled target.
A reviewer's sharpest objection to a "DA does not beat ERM" claim is "you did not
try a strong *domain-generalization* method that trains for worst-case
robustness across the source centers."  GroupDRO is the canonical such method:
it minimizes the worst-group (worst source center) risk rather than the pooled
average, so it is the natural strong baseline to include.  Unlike CORAL/DANN it
uses *no* target data — it defends against shift purely by being robust across
the labeled source domains, which is exactly the multi-source regime SHIFT-SURV
is built for.  It is only meaningful with >=2 source centers; with one source it
reduces to ERM (we warn).

Objective
---------
    min_theta  max_{q in simplex}  sum_g q_g * L_g(theta)

where L_g is the average logistic loss in source group (center) g.  We use the
online algorithm of Sagawa et al. (their Alg. 1), full-batch here since the
per-domain sample sizes are small:

  each step:
    L_g          = mean logistic loss of group g at the current theta
    q_g         <- q_g * exp(eta_q * L_g);  q <- q / sum(q)     # upweight worst
    grad         = sum_g q_g * grad L_g  +  l2 * theta
    theta       <- theta - lr * grad

The model is a linear logistic classifier on the observed covariates x, so its
ERM counterpart is exactly the LR ERM baseline (experiments.baselines.
fit_all_observed) — a fair, capacity-matched comparison.

This module is intentionally free of TensorFlow / the adapt library so it can be
imported and unit-tested without the Keras-2 stack.
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from generator.base_generator import DomainDataset
from utils.metrics import auroc, auprc, ece, prevalence, f1, sensitivity, specificity
from experiments import hparams


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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _group_losses_and_grads(
    theta: np.ndarray, b: float, groups: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Per-group mean logistic loss and its gradient wrt (theta, b)."""
    losses, g_thetas, g_bs = [], [], []
    for Xg, yg in groups:
        z = Xg @ theta + b
        s = _sigmoid(z)
        # mean logistic loss = mean( softplus(z) - y*z ), softplus via logaddexp
        loss = float(np.mean(np.logaddexp(0.0, z) - yg * z))
        resid = s - yg
        losses.append(loss)
        g_thetas.append(Xg.T @ resid / len(yg))
        g_bs.append(float(np.mean(resid)))
    return np.asarray(losses), g_thetas, g_bs


def train_group_dro(
    groups: list[tuple[np.ndarray, np.ndarray]],
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lr: float = 0.05,
    q_lr: float = 0.02,
    l2: float = 1e-3,
    steps: int = 3000,
    eval_every: int = 25,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Full-batch GroupDRO for a linear logistic model, with early stopping on a
    pooled source-validation loss.

    The full-batch, deterministic exponentiated-gradient q update can, on some
    real-data folds, over-concentrate all weight on a single hard source center
    and drive the logits to diverge (a collapsed classifier).  The stochastic
    minibatch GroupDRO of Sagawa et al. is self-regularized by gradient noise;
    the full-batch version is not, so we instead keep the (theta, b) with the
    lowest pooled source-validation logistic loss over training.  The validation
    set is source-only — no target labels are used, so model selection stays
    within the unsupervised-DA protocol.  Returns (theta, b, q).
    """
    p = groups[0][0].shape[1]
    theta = np.zeros(p)
    b = 0.0
    G = len(groups)
    q = np.full(G, 1.0 / G)

    def _val_loss(th: np.ndarray, bb: float) -> float:
        z = X_val @ th + bb
        return float(np.mean(np.logaddexp(0.0, z) - y_val * z))

    best_theta, best_b, best_q = theta.copy(), b, q.copy()
    best_val = _val_loss(theta, b) if X_val is not None else np.inf

    for t in range(steps):
        losses, g_thetas, g_bs = _group_losses_and_grads(theta, b, groups)
        # Exponentiated-gradient ascent on q: upweight the worst (highest-loss) group.
        q = q * np.exp(q_lr * losses)
        q = q / q.sum()
        # Descent on the q-weighted risk (+ L2 on theta only).
        grad_theta = sum(q[g] * g_thetas[g] for g in range(G)) + l2 * theta
        grad_b = sum(q[g] * g_bs[g] for g in range(G))
        theta = theta - lr * grad_theta
        b = b - lr * grad_b
        if X_val is not None and (t % eval_every == 0 or t == steps - 1):
            vl = _val_loss(theta, b)
            if vl < best_val:
                best_val, best_theta, best_b, best_q = vl, theta.copy(), b, q.copy()

    if X_val is None:
        return theta, b, q
    return best_theta, best_b, best_q


def fit_group_dro(
    source_datasets: list[DomainDataset],
    target_dataset: DomainDataset,
    seed: int = 0,
    lr: float = 0.05,
    q_lr: float | None = None,
    l2: float = 1e-3,
    steps: int = 3000,
) -> dict:
    """
    GroupDRO on observed covariates x, treating each source center as a group.

    Protocol mirrors the other methods: pool the source domains, 80/20 stratified
    split per domain, StandardScaler fitted on the pooled train split only, target
    labels never used.  With a single source domain this reduces to ERM.

    Returns the standard SHIFT-SURV result dict (feature_set, source_domains,
    source_pooled, target, target_probs, model, scaler).
    """
    q_lr = hparams.get("group_dro", q_lr)
    if len(source_datasets) < 2:
        warnings.warn(
            "fit_group_dro called with <2 source domains: GroupDRO reduces to ERM. "
            "Run with n_source_domains>=2 for a meaningful domain-generalization baseline."
        )

    # Per-domain 80/20 split, keeping domain identity as the DRO group label.
    # random_state=0, not `seed`: every other method splits its source with
    # random_state=0, and the paired per-seed tests (max_statistic_test) assume
    # all methods saw the same data within a seed. Using `seed` here meant
    # GroupDRO trained on a different 80% than the ERM it is differenced
    # against, putting split noise into the paired difference.
    tr_parts, val_parts, groups_idx = [], [], []
    for gi, ds in enumerate(source_datasets):
        X_tr, X_val, y_tr, y_val = train_test_split(
            ds.x, ds.y, test_size=0.2, random_state=0, stratify=ds.y,
        )
        tr_parts.append((X_tr, y_tr, gi))
        val_parts.append((X_val, y_val))

    X_tr_all = np.vstack([p[0] for p in tr_parts])
    scaler = StandardScaler().fit(X_tr_all)

    groups = [(scaler.transform(X).astype(float), y.astype(float))
              for X, y, _ in tr_parts]

    # Pooled source-validation set for early stopping (scaled; no target leak).
    X_val_all = np.vstack([p[0] for p in val_parts])
    y_val_all = np.concatenate([p[1] for p in val_parts])
    X_val_s = scaler.transform(X_val_all).astype(float)

    theta, b, q = train_group_dro(
        groups, X_val=X_val_s, y_val=y_val_all.astype(float),
        lr=lr, q_lr=q_lr, l2=l2, steps=steps,
    )

    def _predict(X_raw: np.ndarray) -> np.ndarray:
        return _sigmoid(scaler.transform(X_raw) @ theta + b)

    # Source-pooled = held-out validation across all domains (unbiased).
    probs_val = _predict(X_val_all)

    source_scores = {}
    for ds in source_datasets:
        source_scores[f"source_d{ds.domain_id}"] = _score(ds.y, _predict(ds.x), ds.n)

    probs_tgt = _predict(target_dataset.x)

    return {
        "feature_set":    "group_dro",
        "source_domains": source_scores,
        "source_pooled":  _score(y_val_all, probs_val, len(y_val_all)),
        "target":         _score(target_dataset.y, probs_tgt, target_dataset.n),
        "target_probs":   probs_tgt,
        "model":          {"theta": theta, "b": b, "q": q},
        "scaler":         scaler,
    }
