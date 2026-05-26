"""
MSK enrichment experiments (LR + MLP, binary label at τ=12m).

Three experiments
-----------------
1. MSK internal CV
   5-fold CV on MSK alone. Baseline discriminative signal at MSK.

2. i3lung → MSK transfer
   Train on all 6 i3lung centres (7 common features), evaluate on MSK.
   Does European signal transfer across the Atlantic?

3. LOOCO enrichment: i3lung vs i3lung + MSK
   For each held-out i3lung centre:
     - source_only  : pool the 5 remaining i3lung centres
     - source + MSK : add MSK to the source pool
   Train LR + MLP on both. Compare AUROC@τ on the held-out centre.

Feature space: 7 common features (SEX, ECOG PS, NLR, SMOKING x3, PDL1 CATEGORY)
Tau: 12 months  (fixed-horizon binary label)

Usage
-----
  cd /Users/ludole/Desktop/SHIFT-SURV
  python -m experiments.run_msk_enrichment
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data import registry_i3lung_vanguri as reg
from data.preprocess import impute_median

TAU     = 12.0
OUT_DIR = _ROOT / "outputs" / "tables" / "i3lung_vanguri"

_LR_KW = dict(C=1.0, max_iter=500, solver="lbfgs",
              class_weight="balanced", random_state=0)
_MLP_KW = dict(hidden_layer_sizes=(32, 16), activation="relu",
               solver="lbfgs", alpha=0.01, max_iter=1000, random_state=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_label(t, delta, tau=TAU):
    return ((t <= tau) & (delta == 1)).astype(int)


def _auroc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _fit_predict(X_tr, y_tr, X_te, model_cls, kw):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m = model_cls(**kw)
        m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def _impute_scale(X_src_pool, X_targets: list[np.ndarray]):
    """Impute with source medians, then standardise. Returns scaled arrays."""
    _, medians = impute_median(X_src_pool)
    imp = lambda X: np.where(np.isnan(X), medians, X)   # noqa: E731
    X_src_imp = imp(X_src_pool)
    scaler = StandardScaler().fit(X_src_imp)
    return [scaler.transform(imp(X)) for X in [X_src_pool] + X_targets]


# ---------------------------------------------------------------------------
# Experiment 1: MSK internal CV
# ---------------------------------------------------------------------------

def run_msk_cv(msk: dict, n_splits: int = 5, verbose: bool = True) -> pd.DataFrame:
    y = _binary_label(msk["t"], msk["delta"])

    # median-impute within MSK (no source — self-imputation)
    X_imp, _ = impute_median(msk["X"])
    scaler = StandardScaler().fit(X_imp)
    X_s = scaler.transform(X_imp)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X_s, y)):
        for name, cls, kw in [("lr", LogisticRegression, _LR_KW),
                               ("mlp", MLPClassifier, _MLP_KW)]:
            proba = _fit_predict(X_s[tr], y[tr], X_s[te], cls, kw)
            rows.append({
                "experiment": "msk_cv",
                "model": name,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "auroc": _auroc(y[te], proba),
                "prev_test": float(y[te].mean()),
            })

    df = pd.DataFrame(rows)
    if verbose:
        print("=== Experiment 1: MSK internal CV ===")
        for name, sub in df.groupby("model"):
            print(f"  {name}: AUROC {sub['auroc'].mean():.4f} ± {sub['auroc'].std():.4f}  "
                  f"(prev={sub['prev_test'].mean():.3f})")
        print()
    return df


# ---------------------------------------------------------------------------
# Experiment 2: i3lung → MSK transfer
# ---------------------------------------------------------------------------

def run_transfer(i3lung_centers: list[dict], msk: dict, verbose: bool = True) -> pd.DataFrame:
    X_src_pool = np.vstack([c["X"] for c in i3lung_centers])
    t_src      = np.concatenate([c["t"]     for c in i3lung_centers])
    delta_src  = np.concatenate([c["delta"] for c in i3lung_centers])
    y_src      = _binary_label(t_src, delta_src)
    y_tgt      = _binary_label(msk["t"], msk["delta"])

    scaled = _impute_scale(X_src_pool, [msk["X"]])
    X_src_s, X_msk_s = scaled[0], scaled[1]

    rows = []
    for name, cls, kw in [("lr", LogisticRegression, _LR_KW),
                           ("mlp", MLPClassifier, _MLP_KW)]:
        proba_src = _fit_predict(X_src_s, y_src, X_src_s, cls, kw)
        proba_tgt = _fit_predict(X_src_s, y_src, X_msk_s, cls, kw)
        rows.append({
            "experiment": "i3lung_to_msk",
            "model": name,
            "auroc_src": _auroc(y_src, proba_src),
            "auroc_msk": _auroc(y_tgt, proba_tgt),
            "gap": _auroc(y_src, proba_src) - _auroc(y_tgt, proba_tgt),
            "prev_msk": float(y_tgt.mean()),
        })

    df = pd.DataFrame(rows)
    if verbose:
        print("=== Experiment 2: i3lung → MSK transfer ===")
        for _, row in df.iterrows():
            print(f"  {row['model']}: src={row['auroc_src']:.4f}  "
                  f"MSK={row['auroc_msk']:.4f}  gap={row['gap']:+.4f}")
        print()
    return df


# ---------------------------------------------------------------------------
# Experiment 3: LOOCO enrichment
# ---------------------------------------------------------------------------

def run_enrichment(i3lung_centers: list[dict], msk: dict,
                   verbose: bool = True) -> pd.DataFrame:
    D    = len(i3lung_centers)
    rows = []

    if verbose:
        print("=== Experiment 3: LOOCO enrichment (i3lung vs i3lung + MSK) ===")
        hdr = f"  {'center':>6}  {'model':>4}  {'src_only':>9}  {'src+MSK':>9}  {'Δ':>7}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

    for held_idx in range(D):
        held       = i3lung_centers[held_idx]
        src_cs     = [c for i, c in enumerate(i3lung_centers) if i != held_idx]
        hname      = reg.CENTER_NAMES[held["center_id"]]

        y_tgt = _binary_label(held["t"], held["delta"])
        if len(np.unique(y_tgt)) < 2:
            continue

        # ── source only ───────────────────────────────────────────────────
        X_src = np.vstack([c["X"] for c in src_cs])
        t_src = np.concatenate([c["t"] for c in src_cs])
        d_src = np.concatenate([c["delta"] for c in src_cs])
        y_src = _binary_label(t_src, d_src)

        scaled_so = _impute_scale(X_src, [held["X"]])
        X_src_s, X_tgt_s = scaled_so[0], scaled_so[1]

        # ── source + MSK ──────────────────────────────────────────────────
        X_aug = np.vstack([X_src, msk["X"]])
        t_aug = np.concatenate([t_src, msk["t"]])
        d_aug = np.concatenate([d_src, msk["delta"]])
        y_aug = _binary_label(t_aug, d_aug)

        scaled_aug = _impute_scale(X_aug, [held["X"]])
        X_aug_s, X_tgt_aug_s = scaled_aug[0], scaled_aug[1]

        for name, cls, kw in [("lr", LogisticRegression, _LR_KW),
                               ("mlp", MLPClassifier, _MLP_KW)]:
            p_so  = _fit_predict(X_src_s,   y_src, X_tgt_s,     cls, kw)
            p_aug = _fit_predict(X_aug_s,   y_aug, X_tgt_aug_s, cls, kw)
            a_so  = _auroc(y_tgt, p_so)
            a_aug = _auroc(y_tgt, p_aug)
            delta = a_aug - a_so

            rows.append({
                "experiment": "enrichment",
                "center": hname,
                "model": name,
                "n_src": len(y_src),
                "n_aug": len(y_aug),
                "n_tgt": held["n"],
                "auroc_src_only": round(a_so,  4),
                "auroc_src_msk":  round(a_aug, 4),
                "delta_auroc":    round(delta, 4),
                "prev_tgt": round(float(y_tgt.mean()), 4),
            })

            if verbose:
                sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "·")
                print(f"  {hname:>6}  {name:>4}  {a_so:>9.4f}  {a_aug:>9.4f}  "
                      f"{delta:>+7.4f} {sign}")

        if verbose:
            print()

    df = pd.DataFrame(rows)
    if verbose:
        print("  Mean Δ AUROC (i3lung + MSK vs i3lung only):")
        for name, sub in df.groupby("model"):
            print(f"    {name}: {sub['delta_auroc'].mean():+.4f}  "
                  f"(min={sub['delta_auroc'].min():+.4f}  "
                  f"max={sub['delta_auroc'].max():+.4f})")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    centers, _ = reg.load_all_centers()
    i3lung_cs  = [c for c in centers if c["center_id"] != 6]
    msk        = next(c for c in centers if c["center_id"] == 6)

    df1 = run_msk_cv(msk,                        verbose=verbose)
    df2 = run_transfer(i3lung_cs, msk,            verbose=verbose)
    df3 = run_enrichment(i3lung_cs, msk,          verbose=verbose)

    all_df = pd.concat([df1, df2, df3], ignore_index=True)
    all_df.to_csv(OUT_DIR / "msk_enrichment.csv", index=False)
    if verbose:
        print(f"Saved → {OUT_DIR}/msk_enrichment.csv")

    return {"msk_cv": df1, "transfer": df2, "enrichment": df3}


if __name__ == "__main__":
    run_all(verbose=True)
