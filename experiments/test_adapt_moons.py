"""
Sanity check for ADAPT methods on the make_moons DA scenario from the ADAPT docs.
Tests CORAL, DANN, CDAN on a 2D toy problem where the target is a rotated version
of the source — ground truth is known and DA should clearly help.

Reports src_train / src_val / target AUROC and plots convergence curves.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from adapt.feature_based import CORAL as _CORAL, DANN as _DANN, CDAN as _CDAN
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

OUT = "/Users/ludole/Desktop/SHIFT-SURV/outputs/figures"
EPOCHS = 200
BATCH  = 34
SEED   = 0


# ── data ─────────────────────────────────────────────────────────────────────

def make_moons_da(n_samples=2000, rotation=30, noise=0.05, random_state=0):
    Xs, ys = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    Xs[:, 0] -= 0.5
    theta = np.radians(-rotation)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Xt = Xs.dot(R)
    return Xs, ys, Xt, ys.copy()


Xs_raw, ys, Xt_raw, yt = make_moons_da()

# 80/20 stratified split on source; target is held out entirely
Xstr, Xsval, ystr, ysval = train_test_split(
    Xs_raw, ys, test_size=0.2, stratify=ys, random_state=SEED)

sc = StandardScaler().fit(Xstr)
Xstr_s  = sc.transform(Xstr).astype("float32")
Xsval_s = sc.transform(Xsval).astype("float32")
Xt_s    = sc.transform(Xt_raw).astype("float32")


# ── grid for decision boundary ───────────────────────────────────────────────

x_min, y_min = np.min([Xstr_s.min(0), Xt_s.min(0)], 0) - 0.3
x_max, y_max = np.max([Xstr_s.max(0), Xt_s.max(0)], 0) + 0.3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
X_grid = np.stack([xx.ravel(), yy.ravel()], -1).astype("float32")


# ── helpers ──────────────────────────────────────────────────────────────────

def get_task(activation="sigmoid", units=1):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(units, activation=activation))
    return model


def auroc(proba_fn, X, y):
    p = proba_fn(X)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def scores(name, proba_fn):
    tr = auroc(proba_fn, Xstr_s, ystr)
    va = auroc(proba_fn, Xsval_s, ysval)
    ho = auroc(proba_fn, Xt_s, yt)
    print(f"{name:8s}  src_train={tr:.3f}  src_val={va:.3f}  target={ho:.3f}"
          f"  gap(val-tgt)={va-ho:+.3f}")
    return tr, va, ho


def plot_boundary(ax, proba_fn, title):
    Z = proba_fn(X_grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.4, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(Xstr_s[ystr==0,0], Xstr_s[ystr==0,1], c="tomato",    edgecolors="k", s=15, lw=0.4, label="src y=0")
    ax.scatter(Xstr_s[ystr==1,0], Xstr_s[ystr==1,1], c="steelblue", edgecolors="k", s=15, lw=0.4, label="src y=1")
    ax.scatter(Xt_s[:,0],         Xt_s[:,1],          c="k",         edgecolors="w", s=15, lw=0.4, marker="^", label="target")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


histories = {}   # name → dict of loss arrays for convergence plot


# ── 1. LR ERM (source only) ──────────────────────────────────────────────────

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xstr_s, ystr)

lr_tr, lr_va, lr_ho = scores("LR ERM", lambda X: lr.predict_proba(X)[:, 1])


# ── 2. CORAL ─────────────────────────────────────────────────────────────────

coral = _CORAL(
    estimator=LogisticRegression(max_iter=1000, C=1.0),
    Xt=Xt_s, lambda_=1.0, verbose=1, random_state=SEED,
)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    coral.fit(Xstr_s, ystr)

# source needs whitening+recoloring; target is already in target space — no transform
_coral_src_fn = lambda X: coral.estimator_.predict_proba(coral.transform(X, domain="src"))[:, 1]
_coral_tgt_fn = lambda X: coral.estimator_.predict_proba(X)[:, 1]
coral_tr = auroc(_coral_src_fn, Xstr_s, ystr)
coral_va = auroc(_coral_src_fn, Xsval_s, ysval)
coral_ho = auroc(_coral_tgt_fn, Xt_s, yt)
print(f"{'CORAL':8s}  src_train={coral_tr:.3f}  src_val={coral_va:.3f}  target={coral_ho:.3f}"
      f"  gap(val-tgt)={coral_va-coral_ho:+.3f}")


# ── 3. DANN ──────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)

dann = _DANN(
    task=get_task(),
    loss="bce",
    optimizer=Adam(0.001, beta_1=0.5),
    copy=True,
    lambda_=0.3,
    metrics=["acc"],
    random_state=SEED,
    verbose=0,
)
hist_dann = dann.fit(
    Xstr_s, ystr.astype("float32"),
    Xt_s,   yt.astype("float32"),
    epochs=EPOCHS, batch_size=BATCH, verbose=0,
)
histories["DANN"] = hist_dann.history.history

def dann_proba(X):
    return np.squeeze(dann.predict(X, verbose=0)).astype(float)

dann_tr, dann_va, dann_ho = scores("DANN", dann_proba)


# ── 4. CDAN ──────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)

cdan = _CDAN(
    task=get_task(units=2, activation="softmax"),
    entropy=True,
    loss="categorical_crossentropy",
    optimizer=Adam(0.001, beta_1=0.5),
    copy=True,
    random_state=SEED,
    verbose=0,
)
ys_cat = tf.keras.utils.to_categorical(ystr, num_classes=2).astype("float32")
yt_cat = tf.keras.utils.to_categorical(yt,   num_classes=2).astype("float32")
hist_cdan = cdan.fit(
    Xstr_s, ys_cat,
    Xt_s,   yt_cat,
    epochs=EPOCHS, batch_size=BATCH, verbose=0,
)
histories["CDAN"] = hist_cdan.history.history

def cdan_proba(X):
    out = np.squeeze(cdan.predict(X, verbose=0)).astype(float)
    if out.ndim == 2:
        return out[:, 1]
    return out

cdan_tr, cdan_va, cdan_ho = scores("CDAN", cdan_proba)


# ── figures ──────────────────────────────────────────────────────────────────

# decision boundaries
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
plot_boundary(axes[0], lambda X: lr.predict_proba(X)[:, 1],
              f"LR ERM\nval={lr_va:.3f}  tgt={lr_ho:.3f}")
plot_boundary(axes[1], _coral_tgt_fn,
              f"CORAL\nval={coral_va:.3f}  tgt={coral_ho:.3f}")
plot_boundary(axes[2], dann_proba,
              f"DANN (λ=0.3)\nval={dann_va:.3f}  tgt={dann_ho:.3f}")
plot_boundary(axes[3], cdan_proba,
              f"CDAN\nval={cdan_va:.3f}  tgt={cdan_ho:.3f}")
axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Make Moons DA — decision boundaries (200 epochs)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/moons_da_methods.png", dpi=150, bbox_inches="tight")

# convergence curves
fig2, axes2 = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))
if len(histories) == 1:
    axes2 = [axes2]
for ax, (name, h) in zip(axes2, histories.items()):
    for key, vals in h.items():
        if "loss" in key.lower() or "acc" in key.lower():
            ax.plot(vals, label=key)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("epoch"); ax.set_ylabel("metric")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig2.suptitle("Training convergence", fontsize=10)
fig2.tight_layout()
fig2.savefig(f"{OUT}/moons_da_convergence.png", dpi=150, bbox_inches="tight")

print(f"\nSaved -> {OUT}/moons_da_methods.png")
print(f"Saved -> {OUT}/moons_da_convergence.png")
