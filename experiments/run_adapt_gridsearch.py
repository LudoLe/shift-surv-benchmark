"""
Hyperparameter grid search for DANN / CDAN on covariate_shift.

Grid:
  lambda  ∈ {0.1, 0.3, 1.0}
  arch    ∈ {small(10→10), medium(32→16), large(64→32)}

Selection criterion: src_val AUROC (no target labels used).
Two-stage:
  1. Grid search at 100 epochs to rank configs.
  2. Retrain best config per method at 300 epochs for final evaluation.
"""
from __future__ import annotations

import os, sys, warnings
from itertools import product
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam

from adapt.feature_based import DANN as _DANN, CDAN as _CDAN
from adapt.feature_based import CORAL as _CORAL

from generator.scenarios import generate_scenario

# ── config ────────────────────────────────────────────────────────────────────

N_PER_DOMAIN   = 5000
N_SOURCE_DOM   = 3
SEED           = 0
BATCH          = 64
SEARCH_EPOCHS  = 100
FINAL_EPOCHS   = 300

LAMBDAS = [0.1, 0.3, 1.0]
ARCHS   = {
    "small":  (10, 10),
    "medium": (32, 16),
    "large":  (64, 32),
}


# ── data ─────────────────────────────────────────────────────────────────────

print("Generating covariate_shift data...")
data = generate_scenario(
    "covariate_shift",
    n_source_domains=N_SOURCE_DOM,
    n_per_domain=N_PER_DOMAIN,
    seed=SEED,
)
X_src = np.vstack([ds.x for ds in data["source"]])
y_src = np.concatenate([ds.y for ds in data["source"]])
X_tgt = data["target"].x
y_tgt = data["target"].y

X_tr, X_val, y_tr, y_val = train_test_split(
    X_src, y_src, test_size=0.2, stratify=y_src, random_state=SEED,
)
sc = StandardScaler().fit(X_tr)
X_tr_s  = sc.transform(X_tr).astype("float32")
X_val_s = sc.transform(X_val).astype("float32")
X_tgt_s = sc.transform(X_tgt).astype("float32")
input_dim = X_tr_s.shape[1]

print(f"  source train: {X_tr_s.shape}  val: {X_val_s.shape}  target: {X_tgt_s.shape}")
print(f"  prevalence  : train={y_tr.mean():.3f}  val={y_val.mean():.3f}  target={y_tgt.mean():.3f}\n")


# ── helpers ──────────────────────────────────────────────────────────────────

def auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

def build_nets(h1, h2):
    enc  = Sequential([Dense(h1, "relu", input_shape=(input_dim,)), Dense(h2, "relu")], name="enc")
    task = Sequential([Dense(1, "sigmoid")], name="task")
    disc = Sequential([Dense(h2, "relu"), Dense(1, "sigmoid")], name="disc")
    return enc, task, disc

def fit_lr_on_encoder(model, X_tr, y_tr):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        lr.fit(model.transform(X_tr), y_tr)
    return lr

def eval_dann_lr(model, lr, X, y):
    return auroc(y, lr.predict_proba(model.transform(X))[:, 1])


# ── LR ERM baseline ──────────────────────────────────────────────────────────

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr_erm = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(X_tr_s, y_tr)

lr_val = auroc(y_val, lr_erm.predict_proba(X_val_s)[:, 1])
lr_tgt = auroc(y_tgt, lr_erm.predict_proba(X_tgt_s)[:, 1])
print(f"LR ERM          src_val={lr_val:.3f}  target={lr_tgt:.3f}\n")


# ── CORAL baseline ────────────────────────────────────────────────────────────

coral = _CORAL(
    estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
    Xt=X_tgt_s, lambda_=1.0, verbose=0, random_state=SEED,
)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    coral.fit(X_tr_s, y_tr)

coral_val = auroc(y_val, coral.estimator_.predict_proba(coral.transform(X_val_s, domain="src"))[:, 1])
coral_tgt = auroc(y_tgt, coral.estimator_.predict_proba(X_tgt_s)[:, 1])
print(f"CORAL           src_val={coral_val:.3f}  target={coral_tgt:.3f}\n")


# ── grid search ───────────────────────────────────────────────────────────────

print("=" * 65)
print(f"GRID SEARCH  ({SEARCH_EPOCHS} epochs each, select by src_val AUROC)")
print("=" * 65)

dann_grid, cdan_grid = [], []

for (arch_name, (h1, h2)), lam in product(ARCHS.items(), LAMBDAS):
    tf.random.set_seed(SEED)
    enc, task, disc = build_nets(h1, h2)

    dann = _DANN(
        encoder=enc, task=task, discriminator=disc,
        loss="bce", optimizer=Adam(0.001, beta_1=0.5),
        lambda_=lam, random_state=SEED, verbose=0,
    )
    dann.fit(X_tr_s, y_tr.astype("float32"), X_tgt_s, y_tgt.astype("float32"),
             epochs=SEARCH_EPOCHS, batch_size=BATCH, verbose=0)

    lr_d = fit_lr_on_encoder(dann, X_tr_s, y_tr)
    val_a = eval_dann_lr(dann, lr_d, X_val_s, y_val)
    tgt_a = eval_dann_lr(dann, lr_d, X_tgt_s, y_tgt)
    dann_grid.append({"arch": arch_name, "h1": h1, "h2": h2, "lam": lam,
                       "val": val_a, "tgt": tgt_a, "model": dann, "lr": lr_d})
    print(f"  DANN  arch={arch_name:6s} λ={lam:.1f}  src_val={val_a:.3f}  target={tgt_a:.3f}")

print()

for (arch_name, (h1, h2)), lam in product(ARCHS.items(), LAMBDAS):
    tf.random.set_seed(SEED)
    enc, task, disc = build_nets(h1, h2)

    cdan = _CDAN(
        encoder=enc, task=task, discriminator=disc,
        loss="bce", optimizer=Adam(0.001, beta_1=0.5),
        lambda_=lam, random_state=SEED, verbose=0,
    )
    cdan.fit(X_tr_s, y_tr.astype("float32"), X_tgt_s, y_tgt.astype("float32"),
             epochs=SEARCH_EPOCHS, batch_size=BATCH, verbose=0)

    lr_c = fit_lr_on_encoder(cdan, X_tr_s, y_tr)
    val_a = eval_dann_lr(cdan, lr_c, X_val_s, y_val)
    tgt_a = eval_dann_lr(cdan, lr_c, X_tgt_s, y_tgt)
    cdan_grid.append({"arch": arch_name, "h1": h1, "h2": h2, "lam": lam,
                       "val": val_a, "tgt": tgt_a, "model": cdan, "lr": lr_c})
    print(f"  CDAN  arch={arch_name:6s} λ={lam:.1f}  src_val={val_a:.3f}  target={tgt_a:.3f}")


# ── select best and retrain ───────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"FINAL TRAINING  (best config × {FINAL_EPOCHS} epochs)")
print("=" * 65)

for method_name, grid, cls in [("DANN", dann_grid, _DANN), ("CDAN", cdan_grid, _CDAN)]:
    best = max(grid, key=lambda r: r["val"])
    print(f"\n  Best {method_name}: arch={best['arch']}  λ={best['lam']}  "
          f"src_val={best['val']:.3f}  (grid target={best['tgt']:.3f})")

    tf.random.set_seed(SEED)
    enc, task, disc = build_nets(best["h1"], best["h2"])
    model = cls(
        encoder=enc, task=task, discriminator=disc,
        loss="bce", optimizer=Adam(0.001, beta_1=0.5),
        lambda_=best["lam"], random_state=SEED, verbose=0,
    )
    model.fit(X_tr_s, y_tr.astype("float32"), X_tgt_s, y_tgt.astype("float32"),
              epochs=FINAL_EPOCHS, batch_size=BATCH, verbose=0)

    lr_final = fit_lr_on_encoder(model, X_tr_s, y_tr)
    val_f = eval_dann_lr(model, lr_final, X_val_s, y_val)
    tgt_f = eval_dann_lr(model, lr_final, X_tgt_s, y_tgt)
    print(f"  {method_name}+LR (final)  src_val={val_f:.3f}  target={tgt_f:.3f}"
          f"  gap={val_f - tgt_f:+.3f}")

print(f"\n{'─'*65}")
print(f"LR ERM (reference)      src_val={lr_val:.3f}  target={lr_tgt:.3f}")
print(f"CORAL  (reference)      src_val={coral_val:.3f}  target={coral_tgt:.3f}")
