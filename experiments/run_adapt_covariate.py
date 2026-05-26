"""
Single-shot DANN / CDAN on covariate_shift scenario using ADAPT docs signatures.

Uses large n_per_domain to give the neural methods enough data.
CORAL included as linear DA baseline.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from adapt.feature_based import CORAL as _CORAL, DANN as _DANN, CDAN as _CDAN

from generator.scenarios import generate_scenario

# ── config ────────────────────────────────────────────────────────────────────

N_PER_DOMAIN  = 5000       # source samples per domain; 3 domains → 15 000 source total
N_SOURCE_DOM  = 3
SEED          = 0
EPOCHS_DANN   = 300
EPOCHS_CDAN   = 500        # CDAN needs more epochs for the conditional map to stabilise
BATCH         = 64
LAMBDA_DANN   = 0.3


# ── data ─────────────────────────────────────────────────────────────────────

print("Generating covariate_shift data...")
data = generate_scenario(
    "covariate_shift",
    n_source_domains=N_SOURCE_DOM,
    n_per_domain=N_PER_DOMAIN,
    seed=SEED,
)
source_datasets = data["source"]
target_dataset  = data["target"]

X_src = np.vstack([ds.x for ds in source_datasets])
y_src = np.concatenate([ds.y for ds in source_datasets])
X_tgt = target_dataset.x
y_tgt = target_dataset.y

print(f"  source: {X_src.shape}  prevalence={y_src.mean():.3f}")
print(f"  target: {X_tgt.shape}  prevalence={y_tgt.mean():.3f}")

# 80/20 stratified split on pooled source; target is held out entirely
X_tr, X_val, y_tr, y_val = train_test_split(
    X_src, y_src, test_size=0.2, stratify=y_src, random_state=SEED,
)

sc = StandardScaler().fit(X_tr)
X_tr_s  = sc.transform(X_tr).astype("float32")
X_val_s = sc.transform(X_val).astype("float32")
X_tgt_s = sc.transform(X_tgt).astype("float32")

input_dim = X_tr_s.shape[1]


# ── helpers ──────────────────────────────────────────────────────────────────

def build_encoder(input_dim):
    return Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
    ], name="encoder")

def build_task(units=1, activation="sigmoid"):
    return Sequential([Dense(units, activation=activation)], name="task")

def build_discriminator():
    return Sequential([
        Dense(32, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ], name="discriminator")


def auroc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def report(name, fn_src, fn_tgt=None):
    if fn_tgt is None:
        fn_tgt = fn_src
    tr = auroc(y_tr,  fn_src(X_tr_s))
    va = auroc(y_val, fn_src(X_val_s))
    ho = auroc(y_tgt, fn_tgt(X_tgt_s))
    print(f"{name:8s}  src_train={tr:.3f}  src_val={va:.3f}  target={ho:.3f}"
          f"  gap(val-tgt)={va-ho:+.3f}")
    return tr, va, ho


# ── 1. LR ERM ────────────────────────────────────────────────────────────────

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(X_tr_s, y_tr)

report("LR ERM", lambda X: lr.predict_proba(X)[:, 1])


# ── 2. CORAL ─────────────────────────────────────────────────────────────────

coral = _CORAL(
    estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
    Xt=X_tgt_s, lambda_=1.0, verbose=1, random_state=SEED,
)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    coral.fit(X_tr_s, y_tr)

report(
    "CORAL",
    fn_src=lambda X: coral.estimator_.predict_proba(coral.transform(X, domain="src"))[:, 1],
    fn_tgt=lambda X: coral.estimator_.predict_proba(X)[:, 1],
)


# ── 3. DANN ──────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)

dann = _DANN(
    encoder=build_encoder(input_dim),
    task=build_task(),
    discriminator=build_discriminator(),
    loss="bce",
    optimizer=Adam(0.001, beta_1=0.5),
    lambda_=LAMBDA_DANN,
    metrics=["acc"],
    random_state=SEED,
    verbose=0,
)
print(f"\nTraining DANN ({EPOCHS_DANN} epochs, batch={BATCH}, λ={LAMBDA_DANN})...")
dann.fit(
    X_tr_s, y_tr.astype("float32"),
    X_tgt_s, y_tgt.astype("float32"),
    epochs=EPOCHS_DANN, batch_size=BATCH, verbose=0,
)

def dann_proba(X):
    return np.squeeze(dann.predict(X, verbose=0)).astype(float)

report("DANN (neural)", dann_proba)

# LR on DANN-aligned encoder representations
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr_dann = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(
        dann.transform(X_tr_s), y_tr)
report(
    "DANN+LR",
    fn_src=lambda X: lr_dann.predict_proba(dann.transform(X))[:, 1],
    fn_tgt=lambda X: lr_dann.predict_proba(dann.transform(X))[:, 1],
)


# ── 4. CDAN ──────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)

cdan = _CDAN(
    encoder=build_encoder(input_dim),
    task=build_task(),
    discriminator=build_discriminator(),
    loss="bce",
    optimizer=Adam(0.001, beta_1=0.5),
    lambda_=LAMBDA_DANN,
    random_state=SEED,
    verbose=0,
)
print(f"\nTraining CDAN ({EPOCHS_CDAN} epochs, batch={BATCH})...")
cdan.fit(
    X_tr_s, y_tr.astype("float32"),
    X_tgt_s, y_tgt.astype("float32"),
    epochs=EPOCHS_CDAN, batch_size=BATCH, verbose=0,
)

def cdan_proba(X):
    out = np.squeeze(cdan.predict(X, verbose=0)).astype(float)
    if out.ndim == 2:
        return out[:, 1]
    return out

report("CDAN (neural)", cdan_proba)

# LR on CDAN-aligned encoder representations
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr_cdan = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(
        cdan.transform(X_tr_s), y_tr)
report(
    "CDAN+LR",
    fn_src=lambda X: lr_cdan.predict_proba(cdan.transform(X))[:, 1],
    fn_tgt=lambda X: lr_cdan.predict_proba(cdan.transform(X))[:, 1],
)
