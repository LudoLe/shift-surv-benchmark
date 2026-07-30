"""
Real-data experiment: I3LUNG clinical features (CB), INT → GHD transfer.

Source: INT  (n≈668, after dropping missing OS_6)
Target: GHD  (n=430)
Features: 11 CB features from features.json (clinical + blood)
Label: OS_6 (6-month overall survival, binary)

Protocol mirrors the synthetic benchmark:
  - 10 random seeds, 80/20 stratified source train/val split
  - Median imputation fitted on source train only
  - StandardScaler fitted on source train only
  - Target labels never used during training or HP selection
  - Same methods: lr_erm, coral, prior_recalib, dann, cdan

Usage
-----
  python -m experiments.run_real_data --data_dir /path/to/I3LUNG_DATA
  python -m experiments.run_real_data --data_dir /path/to/I3LUNG_DATA --out_prefix i3lung_cb
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.metrics import auroc, auprc, ece, f1, sensitivity, specificity, prevalence
from utils.io import save_table_csv
from utils.shift_diag import rbf_mmd, rbf_mmd_permutation
from utils.calibration import fit_platt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TABLES  = _ROOT / "outputs" / "tables"
_FIGURES = _ROOT / "outputs" / "figures"
_LR_KWARGS = dict(max_iter=500, solver="lbfgs", C=1.0, random_state=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_i3lung(
    data_dir: str | Path,
    source_center: str = "INT",
    target_center: str = "GHD",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load and merge cb.csv + outcomes.csv; return (source_df, target_df, feature_names).
    Rows with missing OS_6 are dropped.
    """
    data_dir = Path(data_dir)
    cb       = pd.read_csv(data_dir / "cb.csv")
    outcomes = pd.read_csv(data_dir / "outcomes.csv")

    with open(data_dir / "features.json") as f:
        feats = json.load(f)
    cb_feats = feats["CB"]

    df = cb.merge(outcomes[["Subject", "OS_6"]], on="Subject")
    df = df.dropna(subset=["OS_6"])

    src = df[df["CENTER"] == source_center][cb_feats + ["OS_6"]].reset_index(drop=True)
    tgt = df[df["CENTER"] == target_center][cb_feats + ["OS_6"]].reset_index(drop=True)

    return src, tgt, cb_feats


class _CenterDataset:
    """Minimal DomainDataset-compatible holder (x, y, domain_id, n) so the
    synthetic-benchmark method functions (fit_all_observed, fit_coral, fit_irm,
    fit_group_dro, ...) can be reused verbatim on real per-center cohorts."""

    def __init__(self, x, y, domain_id):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y).astype(float)
        self.domain_id = domain_id
        self.n = len(self.y)
        # A few method paths reference latent factors; on real data we alias x.
        self.u_inv = self.x
        self.u_spu = self.x[:, :2] if self.x.shape[1] >= 2 else self.x


def load_i3lung_all(
    data_dir: str | Path,
    min_n: int = 50,
) -> tuple[dict, list[str]]:
    """Load every I3LUNG center as a {center: (X_df, y)} dict for the
    leave-one-center-out (LOCO) multi-source domain-generalization experiment.
    Rows with missing OS_6 are dropped; centers with < min_n patients are
    skipped (too small to be a useful source or held-out target)."""
    data_dir = Path(data_dir)
    cb       = pd.read_csv(data_dir / "cb.csv")
    outcomes = pd.read_csv(data_dir / "outcomes.csv")
    with open(data_dir / "features.json") as f:
        cb_feats = json.load(f)["CB"]
    df = cb.merge(outcomes[["Subject", "OS_6"]], on="Subject").dropna(subset=["OS_6"])
    centers: dict = {}
    for c in sorted(df["CENTER"].unique()):
        sub = df[df["CENTER"] == c]
        if len(sub) < min_n:
            continue
        X = sub[cb_feats].reset_index(drop=True)
        y = sub["OS_6"].astype(int).reset_index(drop=True).values
        centers[c] = (X, y)
    return centers, cb_feats


# Public multicentric registries selectable from the CLI.
_PUBLIC_REGISTRIES = {
    "rotterdam_gbsg": "data.registry_rotterdam_gbsg",
    "seer":           "data.registry_seer",
}


def _center_to_df(
    center: dict,
    feature_names: list[str],
    tau: float,
    label_col: str,
) -> pd.DataFrame:
    """Binarise one survival centre to a fixed-horizon label at ``tau`` months.

    Mirrors the I3LUNG ``OS_6`` protocol:
      y=1  if event observed before tau        (t < tau and delta == 1)
      y=0  if event-free through tau            (t >= tau)
      drop if censored before tau               (t < tau and delta == 0)
    Rows censored before the horizon have an unknown label and are dropped,
    exactly as ``load_i3lung`` drops rows with missing OS_6.
    """
    t = np.asarray(center["t"], dtype=float)
    d = np.asarray(center["delta"], dtype=int)
    y = np.where(t < tau, d, 0).astype(int)          # event-before-tau else 0
    keep = ~((t < tau) & (d == 0))                    # drop censored-before-tau
    df = pd.DataFrame(center["X"][keep], columns=feature_names)
    df[label_col] = y[keep]
    return df.reset_index(drop=True)


def load_public_transfer(
    registry_name: str,
    source_center: str,
    target_center: str,
    tau: float,
    label_col: str = "event",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load a public multicentric survival registry and build a source→target
    fixed-horizon binary transfer, matching the (src_df, tgt_df, features) shape
    consumed by ``_run_seed``.

    ``source_center``/``target_center`` may be centre names (values of the
    registry's ``CENTER_NAMES``) or integer center ids.
    """
    import importlib

    if registry_name not in _PUBLIC_REGISTRIES:
        raise ValueError(
            f"Unknown registry '{registry_name}'. Available: {list(_PUBLIC_REGISTRIES)}"
        )
    reg = importlib.import_module(_PUBLIC_REGISTRIES[registry_name])
    centers, feature_names = reg.load_all_centers()

    name_to_id = {v: k for k, v in reg.CENTER_NAMES.items()}

    def _resolve(sel: str) -> int:
        if sel in name_to_id:
            return name_to_id[sel]
        try:
            return int(sel)
        except (TypeError, ValueError):
            raise ValueError(
                f"Center '{sel}' not found. Available: {list(reg.CENTER_NAMES.values())}"
            )

    by_id = {c["center_id"]: c for c in centers}
    src_c = by_id[_resolve(source_center)]
    tgt_c = by_id[_resolve(target_center)]

    src_df = _center_to_df(src_c, feature_names, tau, label_col)
    tgt_df = _center_to_df(tgt_c, feature_names, tau, label_col)
    return src_df, tgt_df, feature_names


# ---------------------------------------------------------------------------
# Score helper
# ---------------------------------------------------------------------------

def _score(y_true: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "auroc":       round(auroc(y_true, probs), 4),
        "auprc":       round(auprc(y_true, probs), 4),
        "ece":         round(ece(y_true, probs), 4),
        "f1":          round(f1(y_true, probs), 4),
        "sensitivity": round(sensitivity(y_true, probs), 4),
        "specificity": round(specificity(y_true, probs), 4),
        "prevalence":  round(prevalence(y_true), 4),
        "n":           len(y_true),
    }


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def _run_seed(
    src: pd.DataFrame,
    tgt: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    use_iwcv: bool = True,
    label_col: str = "OS_6",
) -> tuple[list[dict], np.ndarray]:
    """Fit all methods for one seed.

    Returns (results, y_tgt) where each result dict additionally carries
    ``tgt_probs`` (target-domain positive-class probabilities, aligned to the
    fixed target row order) so callers can run a patient-level bootstrap.
    """
    from experiments.adaptation_methods import fit_coral, fit_dann, fit_cdan
    from experiments.baselines import fit_prior_recalib

    X_src = src[feature_cols].values
    y_src = src[label_col].values.astype(int)
    X_tgt = tgt[feature_cols].values
    y_tgt = tgt[label_col].values.astype(int)

    # 80/20 stratified split on source
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_src, y_src, test_size=0.2, random_state=seed, stratify=y_src,
    )

    # Imputer fitted on source train
    imputer = SimpleImputer(strategy="median")
    X_tr  = imputer.fit_transform(X_tr)
    X_val = imputer.transform(X_val)
    X_tgt_imp = imputer.transform(X_tgt)

    # Scaler fitted on source train
    scaler = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_val_s  = scaler.transform(X_val)
    X_tgt_s  = scaler.transform(X_tgt_imp)

    results = []

    # --- LR ERM ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(**_LR_KWARGS)
        lr.fit(X_tr_s, y_tr)
    p_val = lr.predict_proba(X_val_s)[:, 1]
    p_tgt = lr.predict_proba(X_tgt_s)[:, 1]
    results.append({
        "method": "lr_erm",
        "source_pooled": _score(y_val, p_val),
        "target": _score(y_tgt, p_tgt),
        "tgt_probs": np.asarray(p_tgt),
    })

    # --- IW-ERM (importance-weighted ERM) ---
    # Reweight the source training risk by w(x)=P(tgt|x)/P(src|x) (domain
    # classifier), the consistent correction under covariate shift.  Weights
    # estimated on the training points themselves (the risk being reweighted).
    from experiments.adaptation_methods import _importance_weights
    sw = _importance_weights(X_tr_s, X_tgt_s, X_eval=X_tr_s)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr_iw = LogisticRegression(**_LR_KWARGS)
        lr_iw.fit(X_tr_s, y_tr, sample_weight=sw)
    p_val_iw = lr_iw.predict_proba(X_val_s)[:, 1]
    p_tgt_iw = lr_iw.predict_proba(X_tgt_s)[:, 1]
    results.append({
        "method": "iw_erm",
        "source_pooled": _score(y_val, p_val_iw),
        "target": _score(y_tgt, p_tgt_iw),
        "tgt_probs": np.asarray(p_tgt_iw),
    })

    # --- Prior recalibration ---
    pi_src = float(y_tr.mean())
    p_raw  = lr.predict_proba(X_tgt_s)[:, 1]
    pi_tgt = float(p_raw.mean())
    for _ in range(50):
        num   = p_raw * (pi_tgt / pi_src)
        den   = num + (1 - p_raw) * ((1 - pi_tgt) / (1 - pi_src))
        p_adj = num / np.clip(den, 1e-12, None)
        pi_new = float(p_adj.mean())
        if abs(pi_new - pi_tgt) < 1e-7:
            break
        pi_tgt = pi_new
    results.append({
        "method": "prior_recalib",
        "source_pooled": _score(y_val, p_val),
        "target": _score(y_tgt, p_adj),
        "tgt_probs": np.asarray(p_adj),
    })

    # --- Platt scaling (sigmoid calibration) ---
    # Shared with experiments.baselines._fit_posthoc_calib via utils.calibration,
    # so the synthetic and real-data Platt baselines are the same method by
    # construction rather than by a comment. See fit_platt for why the fitted
    # map is checked for monotonicity.
    p_val_raw = lr.predict_proba(X_val_s)[:, 1]
    p_tgt_raw = lr.predict_proba(X_tgt_s)[:, 1]
    p_val_platt, p_tgt_platt, _, _ = fit_platt(p_val_raw, y_val, p_tgt_raw)
    results.append({
        "method": "platt_scaling",
        "source_pooled": _score(y_val, p_val_platt),
        "target": _score(y_tgt, p_tgt_platt),
        "tgt_probs": np.asarray(p_tgt_platt),
    })

    # --- CORAL ---
    from adapt.feature_based import CORAL as _CORAL
    coral = _CORAL(
        estimator=LogisticRegression(**_LR_KWARGS),
        Xt=X_tgt_s, lambda_=1.0, verbose=0, random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        coral.fit(X_tr_s, y_tr)
    p_val_c = coral.estimator_.predict_proba(coral.transform(X_val_s, domain="src"))[:, 1]
    p_tgt_c = coral.estimator_.predict_proba(X_tgt_s)[:, 1]
    results.append({
        "method": "coral",
        "source_pooled": _score(y_val, p_val_c),
        "target": _score(y_tgt, p_tgt_c),
        "tgt_probs": np.asarray(p_tgt_c),
    })

    # Build DomainDataset-compatible objects for DANN/CDAN
    # We create minimal objects with .x, .y, .domain_id, .n attributes
    class _DS:
        def __init__(self, x, y, domain_id):
            self.x = x
            self.y = y.astype(float)
            self.domain_id = domain_id
            self.n = len(y)
            # DANN/CDAN also need u_inv, u_spu — not used here
            self.u_inv = x
            self.u_spu = x[:, :2]

    # Pass already-imputed (but not yet scaled) data — fit_dann/fit_cdan
    # will do their own scaling internally. We pass raw imputed arrays.
    src_ds = [_DS(X_tr, y_tr, domain_id=0)]   # 80% source train
    tgt_ds  = _DS(X_tgt_imp, y_tgt, domain_id=1)

    # --- DANN ---
    try:
        r_dann = fit_dann(src_ds, tgt_ds, seed=seed, use_iwcv=use_iwcv)
        results.append({"method": "dann",
                        "source_pooled": r_dann.get("source_pooled", {}),
                        "target": r_dann.get("target", {}),
                        "tgt_probs": np.asarray(r_dann.get("target_probs"))
                        if r_dann.get("target_probs") is not None else None})
    except Exception as e:
        warnings.warn(f"DANN failed seed={seed}: {e}")
        results.append({"method": "dann", "source_pooled": {}, "target": {}, "tgt_probs": None})

    # --- CDAN ---
    try:
        r_cdan = fit_cdan(src_ds, tgt_ds, seed=seed, use_iwcv=use_iwcv)
        results.append({"method": "cdan",
                        "source_pooled": r_cdan.get("source_pooled", {}),
                        "target": r_cdan.get("target", {}),
                        "tgt_probs": np.asarray(r_cdan.get("target_probs"))
                        if r_cdan.get("target_probs") is not None else None})
    except Exception as e:
        warnings.warn(f"CDAN failed seed={seed}: {e}")
        results.append({"method": "cdan", "source_pooled": {}, "target": {}, "tgt_probs": None})

    return results, y_tgt


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["src_auroc", "tgt_auroc", "tgt_ece", "tgt_f1",
                "tgt_sensitivity", "tgt_specificity", "gap"]
_METHODS     = ["lr_erm", "iw_erm", "prior_recalib", "platt_scaling", "coral",
                "irm", "group_dro", "dann", "cdan"]


def _flatten_real(r: dict, seed: int) -> dict:
    sp  = r.get("source_pooled", {})
    tgt = r.get("target", {})
    sa  = sp.get("auroc",  float("nan"))
    ta  = tgt.get("auroc", float("nan"))
    gap = round(sa - ta, 4) if not (np.isnan(sa) or np.isnan(ta)) else float("nan")
    return {
        "seed":            seed,
        "method":          r["method"],
        "src_auroc":       sa,
        "tgt_auroc":       ta,
        "tgt_ece":         tgt.get("ece",         float("nan")),
        "tgt_f1":          tgt.get("f1",           float("nan")),
        "tgt_sensitivity": tgt.get("sensitivity",  float("nan")),
        "tgt_specificity": tgt.get("specificity",  float("nan")),
        "gap":             gap,
    }


def _aggregate_real(raw: list[dict]) -> list[dict]:
    agg = []
    for m in _METHODS:
        subset = [r for r in raw if r["method"] == m]
        if not subset:
            continue
        row = {"method": m, "n_seeds": len(subset)}
        for key in _METRIC_KEYS:
            vals = np.array([r[key] for r in subset], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_ci95"] = float("nan")
                continue
            mean = float(vals.mean())
            std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            se   = std / np.sqrt(len(vals))
            tc   = sp_stats.t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
            row[f"{key}_mean"] = round(mean, 4)
            row[f"{key}_ci95"] = round(tc * se, 4)
        agg.append(row)
    return agg


# ---------------------------------------------------------------------------
# Patient-level bootstrap (target-cohort sampling uncertainty)
# ---------------------------------------------------------------------------

def _bootstrap_real_gaps(
    prob_by_method: dict[str, np.ndarray],
    y_tgt: np.ndarray,
    reference: str = "lr_erm",
    n_boot: int = 2000,
    seed: int = 0,
    n_bins: int = 10,
    stratified: bool = True,
    replicate_sink: list | None = None,
) -> list[dict]:
    """Patient-level paired bootstrap of AUROC/ECE and gaps vs a reference method.

    Why this exists
    ---------------
    The per-seed CIs in ``_aggregate_real`` vary only the source train/val split
    and the network initialisation; the source and target *cohorts* are fixed, so
    those CIs say nothing about how the result would change on a different sample
    of target patients.  Here we instead resample target patients with replacement
    to obtain CIs (and a p-value for each method-vs-reference gap) that reflect
    target-cohort sampling uncertainty -- the quantity a clinical reviewer cares
    about when asking "would this transfer to the next 430 patients?".

    Protocol
    --------
    * ``prob_by_method[m]`` is an (n_seeds, n_tgt) array of target probabilities.
      The target cohort is identical and identically ordered across seeds, so the
      seeds are averaged into one ensemble probability per patient before
      resampling (this removes split/init noise and isolates cohort sampling).
    * For each of ``n_boot`` resamples, the SAME patient indices are used for
      every method (paired bootstrap), so the gap distribution accounts for the
      positive correlation between methods evaluated on the same patients.
    * Gap sign convention: AUROC gap = method - reference (higher = method better);
      ECE gap = reference - method (higher = method better, i.e. lower ECE).
      The two-sided bootstrap p-value is 2*min(P(gap>0), P(gap<0)).
    * By default the resample is *stratified by outcome*: positives and negatives
      are resampled separately to their observed sizes, so every resample
      preserves the target prevalence and always contains both classes (AUROC is
      always defined). This removes the sampling noise from prevalence
      fluctuation -- which only inflates the CIs -- and matters most for small or
      low-prevalence target centers. Set stratified=False for an ordinary
      (prevalence-varying) bootstrap. n_pos / n_neg are reported so the effective
      power of each transfer is visible.
    * ``replicate_sink``: if a list is passed, the per-replicate gaps are appended
      to it as one row per (replicate, method). The summary statistics below
      collapse the joint distribution across methods, but a max-statistic test
      needs that joint distribution: the same resample is shared by all methods,
      so their gaps are correlated, and the null distribution of the *maximum*
      cannot be reconstructed from per-method margins alone. Only bootstrap
      indices that survive the single-class check are recorded, and every method
      shares the same ``b`` index, which is what makes the rows pairable.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_tgt).astype(int)
    n = len(y)

    ens = {m: np.asarray(P).mean(axis=0) for m, P in prob_by_method.items()
           if P is not None and len(P) > 0}
    if reference not in ens:
        return []

    # Class indices for stratified resampling and effective-power reporting.
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # Bootstrap indices shared across methods (paired). Stratified by default:
    # resample each class to its observed size so every resample preserves the
    # target prevalence and yields a defined AUROC. Falls back to an ordinary
    # bootstrap if a class is empty (AUROC is undefined there regardless).
    if stratified and n_pos > 0 and n_neg > 0:
        pos_boot = pos_idx[rng.integers(0, n_pos, size=(n_boot, n_pos))]
        neg_boot = neg_idx[rng.integers(0, n_neg, size=(n_boot, n_neg))]
        idx = np.concatenate([pos_boot, neg_boot], axis=1)   # (n_boot, n)
    else:
        idx = rng.integers(0, n, size=(n_boot, n))

    def _auroc_ece(yb, pb):
        return auroc(yb, pb), ece(yb, pb, n_bins=n_bins)

    boot = {m: {"auroc": [], "ece": [], "d_auroc": [], "d_ece": []} for m in ens}
    for b in range(n_boot):
        ib = idx[b]
        yb = y[ib]
        if len(np.unique(yb)) < 2:
            continue  # AUROC undefined on a single-class resample
        ref_au, ref_ec = _auroc_ece(yb, ens[reference][ib])
        for m, p in ens.items():
            au, ec = _auroc_ece(yb, p[ib])
            boot[m]["auroc"].append(au)
            boot[m]["ece"].append(ec)
            boot[m]["d_auroc"].append(au - ref_au)        # method - ref
            boot[m]["d_ece"].append(ref_ec - ec)          # ref - method
            if replicate_sink is not None:
                replicate_sink.append({
                    "b": b, "method": m,
                    "d_auroc": round(au - ref_au, 6),
                    "d_ece": round(ref_ec - ec, 6),
                })

    def _ci(v):
        v = np.asarray(v, dtype=float)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def _p_two_sided(v):
        # Inclusive comparisons so probability mass exactly at 0 counts on both
        # sides: a gap distribution centred at 0 (e.g. a method identical to the
        # reference) yields p ≈ 1, while one cleanly shifted off 0 yields p ≈ 0.
        v = np.asarray(v, dtype=float)
        if len(v) == 0:
            return float("nan")
        frac_ge = float(np.mean(v >= 0))
        frac_le = float(np.mean(v <= 0))
        return float(min(1.0, 2.0 * min(frac_ge, frac_le)))

    rows = []
    for m, d in ens.items():
        au_lo, au_hi = _ci(boot[m]["auroc"])
        ec_lo, ec_hi = _ci(boot[m]["ece"])
        dau_lo, dau_hi = _ci(boot[m]["d_auroc"])
        dec_lo, dec_hi = _ci(boot[m]["d_ece"])
        rows.append({
            "method":          m,
            "auroc":           round(float(auroc(y, ens[m])), 4),
            "auroc_lo":        round(au_lo, 4),
            "auroc_hi":        round(au_hi, 4),
            "ece":             round(float(ece(y, ens[m], n_bins=n_bins)), 4),
            "ece_lo":          round(ec_lo, 4),
            "ece_hi":          round(ec_hi, 4),
            "auroc_gap_vs_ref":   round(float(np.mean(boot[m]["d_auroc"])), 4),
            "auroc_gap_lo":       round(dau_lo, 4),
            "auroc_gap_hi":       round(dau_hi, 4),
            "auroc_gap_p":        round(_p_two_sided(boot[m]["d_auroc"]), 4),
            "ece_gap_vs_ref":     round(float(np.mean(boot[m]["d_ece"])), 4),
            "ece_gap_lo":         round(dec_lo, 4),
            "ece_gap_hi":         round(dec_hi, 4),
            "ece_gap_p":          round(_p_two_sided(boot[m]["d_ece"]), 4),
            "reference":          reference,
            "n_boot":             n_boot,
            "n_tgt":              n,
            "n_pos":              int(n_pos),
            "n_neg":              int(n_neg),
            "stratified":         bool(stratified),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-seed patient-level bootstrap (total uncertainty: training + cohort)
# ---------------------------------------------------------------------------

def _fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie-averaged ranks.

    Numerically identical to ``utils.metrics.auroc`` (sklearn) but without the
    per-call validation overhead, which matters because the per-seed bootstrap
    evaluates it n_boot * n_seeds * n_methods times.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = sp_stats.rankdata(p)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE, vectorised.

    Algebraically identical to ``utils.metrics.ece``: that function computes
    sum_b (n_b/n)*|mean(y_b) - mean(p_b)|, and since mean(.)=sum(.)/n_b the n_b
    cancels, leaving sum_b |sum(y_b) - sum(p_b)| / n.

    Binning must use the *same* edges, not floor(p*n_bins): the latter disagrees
    on values that land on an edge, because the edge from linspace and the
    product p*n_bins round differently (e.g. 0.3*10 = 3.0000000000000004 while
    linspace(0,1,11)[3] = 0.30000000000000004, so 0.3 falls in bin 3 one way and
    bin 2 the other). searchsorted against the linspace edges reproduces the
    half-open [lo, hi) convention exactly, with the last bin closed at 1.0.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    sy = np.bincount(b, weights=y, minlength=n_bins)
    sp = np.bincount(b, weights=p, minlength=n_bins)
    return float(np.abs(sy - sp).sum() / len(y))


def _bootstrap_real_gaps_per_seed(
    prob_by_method: dict[str, np.ndarray],
    y_tgt: np.ndarray,
    reference: str = "lr_erm",
    n_boot: int = 2000,
    seed: int = 0,
    n_bins: int = 10,
    stratified: bool = True,
    replicate_sink: list | None = None,
    seed_summary_sink: list | None = None,
) -> list[dict]:
    """Patient-level bootstrap run *separately for every training seed*.

    Why this exists (and why it replaces the ensemble version as the headline)
    ------------------------------------------------------------------------
    The synthetic benchmark regenerates source and target from the generator at
    every seed, so its 10-seed spread contains two sources of variability at
    once: the training run (split, initialisation, HP selection) and the finite
    sample drawn from the population. On real data the cohort is fixed, so the
    two have to be reintroduced separately, and the previous two analyses each
    captured only one of them:

    * ``_aggregate_real`` -- t-CI across seeds. Cohort fixed, so it answers
      "how much does the result move if I retrain?", not "would it hold on the
      next 430 patients?".
    * ``_bootstrap_real_gaps`` -- averages the seeds into one ensemble
      probability per patient, then resamples patients. This *deliberately*
      removes training variability, and it also changes the estimand: a
      10-member ensemble of logistic regressions is a lower-variance predictor
      than the single model whose mean performance the seed table reports, so
      its AUROC is not the number in the summary table.

    This function keeps the seed axis. For every bootstrap replicate ``b`` and
    every seed ``s`` it evaluates all methods on the resampled patients using
    seed ``s``'s own predictions, and pools the resulting (b, s) values. The
    pooled spread therefore carries cohort sampling *and* training variability,
    which is the quantity that is comparable to the synthetic 10-seed interval.

    The point estimate is the across-seed mean of the observed (un-resampled)
    metric, so it now agrees with ``tgt_auroc_mean`` in the seed summary table.

    Pairing
    -------
    The same patient indices are used for every method *and* every seed, so
    method-vs-reference gaps are formed within a single (b, s) cell before any
    pooling. That preserves the positive correlation between methods, which the
    max-statistic test depends on.

    Variance decomposition
    ----------------------
    Each row reports, for AUROC and for the AUROC gap:
      * ``*_sd_within``  -- mean over seeds of the SD across replicates
                            (cohort sampling at fixed training run)
      * ``*_sd_between`` -- SD over seeds of the per-seed mean
                            (training variability at fixed cohort)
    If ``sd_between`` is small relative to ``sd_within``, the ensemble shortcut
    was harmless and the old numbers stand; if it is not, it was hiding real
    variability. Either way the paper can state which dominates instead of
    assuming.

    ``replicate_sink`` keeps the flat (replicate, method) schema that
    ``max_statistic_bootstrap.py`` consumes: ``b`` is a composite index over
    (bootstrap replicate, seed) cells, and ``boot_b`` / ``seed_idx`` are added
    for transparency. Every method shares each composite index, which is what
    makes the rows pairable.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_tgt).astype(int)
    n = len(y)

    P = {m: np.atleast_2d(np.asarray(v, dtype=float))
         for m, v in prob_by_method.items() if v is not None and np.size(v) > 0}
    if reference not in P:
        return []
    n_seeds = P[reference].shape[0]
    if any(v.shape != (n_seeds, n) for v in P.values()):
        raise ValueError(
            "per-seed bootstrap needs every method to have the same "
            f"(n_seeds, n_tgt) = ({n_seeds}, {n}) probability matrix; got "
            + ", ".join(f"{m}:{v.shape}" for m, v in P.items())
        )

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if stratified and n_pos > 0 and n_neg > 0:
        pos_boot = pos_idx[rng.integers(0, n_pos, size=(n_boot, n_pos))]
        neg_boot = neg_idx[rng.integers(0, n_neg, size=(n_boot, n_neg))]
        idx = np.concatenate([pos_boot, neg_boot], axis=1)
    else:
        idx = rng.integers(0, n, size=(n_boot, n))

    methods = list(P.keys())
    # (n_seeds, n_boot) per method, so the seed axis survives to the summary.
    au = {m: np.full((n_seeds, n_boot), np.nan) for m in methods}
    ec = {m: np.full((n_seeds, n_boot), np.nan) for m in methods}
    d_au = {m: np.full((n_seeds, n_boot), np.nan) for m in methods}
    d_ec = {m: np.full((n_seeds, n_boot), np.nan) for m in methods}

    rep_id = 0
    for b in range(n_boot):
        ib = idx[b]
        yb = y[ib]
        if len(np.unique(yb)) < 2:
            continue
        for s in range(n_seeds):
            ref_au = _fast_auroc(yb, P[reference][s][ib])
            ref_ec = _fast_ece(yb, P[reference][s][ib], n_bins=n_bins)
            for m in methods:
                pb = P[m][s][ib]
                a = _fast_auroc(yb, pb)
                e = _fast_ece(yb, pb, n_bins=n_bins)
                au[m][s, b] = a
                ec[m][s, b] = e
                d_au[m][s, b] = a - ref_au        # method - ref
                d_ec[m][s, b] = ref_ec - e        # ref - method
                if replicate_sink is not None:
                    replicate_sink.append({
                        "b": rep_id, "boot_b": b, "seed_idx": s, "method": m,
                        "d_auroc": round(a - ref_au, 6),
                        "d_ece": round(ref_ec - e, 6),
                    })
            rep_id += 1

    def _pooled(M):
        v = np.asarray(M, dtype=float).ravel()
        return v[~np.isnan(v)]

    def _ci(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
            if len(v) else (float("nan"), float("nan"))

    def _p_two_sided(v):
        if len(v) == 0:
            return float("nan")
        return float(min(1.0, 2.0 * min(float(np.mean(v >= 0)), float(np.mean(v <= 0)))))

    def _decomp(M):
        """(sd_within, sd_between) over the (n_seeds, n_boot) grid."""
        M = np.asarray(M, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            per_seed_sd = np.nanstd(M, axis=1, ddof=1)
            per_seed_mean = np.nanmean(M, axis=1)
            within = float(np.nanmean(per_seed_sd))
            between = float(np.nanstd(per_seed_mean, ddof=1)) if M.shape[0] > 1 else 0.0
        return within, between

    # Observed (un-resampled) per-seed metrics: the point estimates.
    obs_au = {m: np.array([_fast_auroc(y, P[m][s]) for s in range(n_seeds)]) for m in methods}
    obs_ec = {m: np.array([_fast_ece(y, P[m][s], n_bins=n_bins) for s in range(n_seeds)])
              for m in methods}

    if seed_summary_sink is not None:
        for m in methods:
            for s in range(n_seeds):
                seed_summary_sink.append({
                    "method": m, "seed_idx": s,
                    "auroc_obs": round(float(obs_au[m][s]), 4),
                    "ece_obs": round(float(obs_ec[m][s]), 4),
                    "auroc_boot_mean": round(float(np.nanmean(au[m][s])), 4),
                    "auroc_boot_sd": round(float(np.nanstd(au[m][s], ddof=1)), 4),
                    "auroc_gap_boot_mean": round(float(np.nanmean(d_au[m][s])), 4),
                    "ece_gap_boot_mean": round(float(np.nanmean(d_ec[m][s])), 4),
                })

    rows = []
    for m in methods:
        pau, pec = _pooled(au[m]), _pooled(ec[m])
        pdau, pdec = _pooled(d_au[m]), _pooled(d_ec[m])
        au_lo, au_hi = _ci(pau)
        ec_lo, ec_hi = _ci(pec)
        dau_lo, dau_hi = _ci(pdau)
        dec_lo, dec_hi = _ci(pdec)
        au_w, au_b = _decomp(au[m])
        dau_w, dau_b = _decomp(d_au[m])
        rows.append({
            "method":            m,
            # Point estimates: across-seed mean of the observed metric, so these
            # match tgt_auroc_mean / tgt_ece_mean in the seed summary table.
            "auroc":             round(float(np.nanmean(obs_au[m])), 4),
            "auroc_lo":          round(au_lo, 4),
            "auroc_hi":          round(au_hi, 4),
            "ece":               round(float(np.nanmean(obs_ec[m])), 4),
            "ece_lo":            round(ec_lo, 4),
            "ece_hi":            round(ec_hi, 4),
            "auroc_gap_vs_ref":  round(float(np.nanmean(pdau)), 4),
            "auroc_gap_lo":      round(dau_lo, 4),
            "auroc_gap_hi":      round(dau_hi, 4),
            "auroc_gap_p":       round(_p_two_sided(pdau), 4),
            "ece_gap_vs_ref":    round(float(np.nanmean(pdec)), 4),
            "ece_gap_lo":        round(dec_lo, 4),
            "ece_gap_hi":        round(dec_hi, 4),
            "ece_gap_p":         round(_p_two_sided(pdec), 4),
            # Variance decomposition: which axis dominates.
            "auroc_sd_within":     round(au_w, 4),
            "auroc_sd_between":    round(au_b, 4),
            "auroc_gap_sd_within": round(dau_w, 4),
            "auroc_gap_sd_between": round(dau_b, 4),
            "reference":         reference,
            "n_boot":            n_boot,
            "n_seeds":           n_seeds,
            "n_cells":           int(len(pdau)),
            "n_tgt":             n,
            "n_pos":             int(n_pos),
            "n_neg":             int(n_neg),
            "stratified":        bool(stratified),
            "uncertainty":       "cohort+training",
        })
    return rows


def _print_bootstrap(rows: list[dict], source: str, target: str) -> None:
    if not rows:
        return
    r0 = rows[0]
    ref = r0["reference"]
    per_seed = "n_seeds" in r0
    if per_seed:
        head = (f"\n  PER-SEED PATIENT-LEVEL BOOTSTRAP — {source} → {target}  "
                f"(n_boot={r0['n_boot']} × {r0['n_seeds']} seeds = {r0['n_cells']} cells, "
                f"gaps vs {ref}, 95% percentile CI)")
        note = ("    uncertainty = target-cohort sampling + training variability "
                "(comparable to the synthetic 10-seed interval)")
    else:
        head = (f"\n  PATIENT-LEVEL BOOTSTRAP (seed-ensemble) — {source} → {target}  "
                f"(n_boot={r0['n_boot']}, gaps vs {ref}, 95% percentile CI)")
        note = "    uncertainty = target-cohort sampling only, seeds averaged out"
    print(head)
    print(note)
    print(f"  {'method':16s}{'AUROC [95% CI]':26s}{'ΔAUROC vs ref (p)':26s}{'ECE [95% CI]':24s}{'ΔECE vs ref (p)':22s}")
    for r in rows:
        au  = f"{r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]"
        dau = f"{r['auroc_gap_vs_ref']:+.3f} [{r['auroc_gap_lo']:+.3f},{r['auroc_gap_hi']:+.3f}] p={r['auroc_gap_p']:.3f}"
        ec  = f"{r['ece']:.3f} [{r['ece_lo']:.3f},{r['ece_hi']:.3f}]"
        dec = f"{r['ece_gap_vs_ref']:+.3f} p={r['ece_gap_p']:.3f}"
        print(f"  {r['method']:16s}{au:26s}{dau:26s}{ec:24s}{dec:22s}")

    if per_seed:
        print(f"\n  VARIANCE DECOMPOSITION (SD of the AUROC gap vs {ref})")
        print(f"  {'method':16s}{'within-seed (cohort)':>22s}{'between-seed (training)':>26s}{'ratio':>10s}")
        for r in rows:
            if r["method"] == ref:
                continue
            w, b = r["auroc_gap_sd_within"], r["auroc_gap_sd_between"]
            ratio = f"{b / w:.2f}" if w > 1e-9 else "n/a"
            print(f"  {r['method']:16s}{w:>22.4f}{b:>26.4f}{ratio:>10s}")
        print("    ratio > 1 means retraining moves the gap more than resampling patients;")
        print("    the seed-ensemble bootstrap discards exactly the between-seed part.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_real_results(agg: list[dict], n_seeds: int, source: str, target: str) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  I3LUNG REAL DATA — {source} → {target}  (mean ± 95% CI, {n_seeds} seeds)")
    print(sep)
    col = 18
    labels = _METHODS
    metrics = [("tgt_auroc", "Target AUROC ↑"),
               ("gap",       "AUROC gap (src−tgt) ↓"),
               ("tgt_ece",   "Target ECE ↓"),
               ("tgt_f1",    "Target F1 ↑")]
    idx = {r["method"]: r for r in agg}
    for metric, title in metrics:
        print(f"\n  {title}")
        print(f"  {'method':20s}" + "".join(f"  {m:>{col}s}" for m in labels))
        print("  " + "-" * (22 + (col + 2) * len(labels)))
        line = f"  {'':20s}"
        for m in labels:
            r = idx.get(m, {})
            mean = r.get(f"{metric}_mean", float("nan"))
            ci   = r.get(f"{metric}_ci95", float("nan"))
            s = f"{mean:.3f}±{ci:.3f}" if not np.isnan(mean) else "—"
            line += f"  {s:>{col}s}"
        print(line)


# ---------------------------------------------------------------------------
# Covariate-shift diagnostic
# ---------------------------------------------------------------------------

def _shift_diagnostic(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    source_center: str,
    target_center: str,
    tau: float | None = None,
) -> list[dict]:
    """RBF-MMD between source/target features (median-imputed + source-fit
    standardised) with a permutation p-value, plus the label-prevalence gap —
    one row, saved to *_shiftdiag.csv."""
    Xs = src_df[feature_cols].values.astype(float)
    Xt = tgt_df[feature_cols].values.astype(float)

    imputer = SimpleImputer(strategy="median").fit(Xs)
    scaler  = StandardScaler().fit(imputer.transform(Xs))
    Xs_s = scaler.transform(imputer.transform(Xs))
    Xt_s = scaler.transform(imputer.transform(Xt))

    mmd, mmd_p = rbf_mmd_permutation(Xs_s, Xt_s, n_perm=1000, seed=0)
    prev_gap = abs(float(src_df[label_col].mean()) - float(tgt_df[label_col].mean()))
    return [{
        "source":          source_center,
        "target":          target_center,
        "tau":             tau if tau is not None else "",
        "rbf_mmd":         round(mmd, 4),
        "rbf_mmd_p":       round(mmd_p, 4),
        "prevalence_gap":  round(prev_gap, 4),
        "n_src":           int(len(src_df)),
        "n_tgt":           int(len(tgt_df)),
        "n_features":      len(feature_cols),
    }]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _run_transfer(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    source_center: str,
    target_center: str,
    seeds: list[int],
    use_iwcv: bool,
    out_prefix: str,
    banner: str,
    verbose: bool,
    tau: float | None = None,
) -> dict:
    """Dataset-agnostic core: run all methods over seeds, aggregate, bootstrap,
    save tables. Shared by the I3LUNG and public-registry entry points."""
    if verbose:
        print(f"\n{banner}: {source_center} (n={len(src_df)}) → {target_center} (n={len(tgt_df)})")
        print(f"Features ({len(feature_cols)}): {feature_cols}")
        print(f"{label_col} prevalence — {source_center}: {src_df[label_col].mean():.3f}  "
              f"{target_center}: {tgt_df[label_col].mean():.3f}")

    all_rows: list[dict] = []
    prob_accum: dict[str, list[np.ndarray]] = {}
    y_tgt_ref: np.ndarray | None = None
    for seed in seeds:
        if verbose:
            print(f"  [Seed {seed}]", flush=True)
        results, y_tgt = _run_seed(
            src_df, tgt_df, feature_cols, seed=seed, use_iwcv=use_iwcv, label_col=label_col,
        )
        y_tgt_ref = y_tgt
        for r in results:
            all_rows.append(_flatten_real(r, seed))
            probs = r.get("tgt_probs")
            if probs is not None and np.asarray(probs).size == len(y_tgt):
                prob_accum.setdefault(r["method"], []).append(np.asarray(probs))

    agg = _aggregate_real(all_rows)

    # Patient-level bootstrap: stack per-seed probabilities (target cohort is
    # fixed and identically ordered across seeds) and resample patients.
    #
    # Headline analysis is the per-seed bootstrap: it keeps the seed axis, so
    # the reported interval carries cohort sampling AND training variability
    # together, which is what makes it comparable to the synthetic 10-seed
    # interval. The ensemble version is still computed and saved alongside so
    # the two can be diffed -- it answers a different question (an ensemble
    # predictor's cohort uncertainty) and its AUROC is not the seed-mean AUROC.
    boot_rows: list[dict] = []
    boot_reps: list[dict] = []
    boot_by_seed: list[dict] = []
    boot_ens_rows: list[dict] = []
    if y_tgt_ref is not None:
        prob_by_method = {m: np.vstack(P) for m, P in prob_accum.items() if P}
        boot_rows = _bootstrap_real_gaps_per_seed(
            prob_by_method, y_tgt_ref, reference="lr_erm",
            replicate_sink=boot_reps, seed_summary_sink=boot_by_seed,
        )
        boot_ens_rows = _bootstrap_real_gaps(prob_by_method, y_tgt_ref,
                                             reference="lr_erm")

    # Covariate-shift diagnostic: how far apart are source/target features?
    diag_rows = _shift_diagnostic(src_df, tgt_df, feature_cols, label_col,
                                  source_center, target_center, tau=tau)

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(all_rows, _TABLES / f"{out_prefix}_raw.csv")
    save_table_csv(agg,      _TABLES / f"{out_prefix}_summary.csv")
    if boot_reps:
        save_table_csv(boot_reps, _TABLES / f"{out_prefix}_bootreps.csv")
    if boot_rows:
        save_table_csv(boot_rows, _TABLES / f"{out_prefix}_bootstrap.csv")
    if boot_by_seed:
        save_table_csv(boot_by_seed, _TABLES / f"{out_prefix}_bootstrap_byseed.csv")
    if boot_ens_rows:
        save_table_csv(boot_ens_rows, _TABLES / f"{out_prefix}_bootstrap_ensemble.csv")
    if diag_rows:
        save_table_csv(diag_rows, _TABLES / f"{out_prefix}_shiftdiag.csv")

    if verbose:
        _print_real_results(agg, len(seeds), source_center, target_center)
        _print_bootstrap(boot_rows, source_center, target_center)
        if diag_rows:
            d = diag_rows[0]
            print(f"\n  COVARIATE SHIFT DIAGNOSTIC — {source_center} → {target_center}")
            print(f"    RBF-MMD (standardised features): {d['rbf_mmd']:.4f}  (perm p={d['rbf_mmd_p']:.4f})")
            print(f"    label prevalence gap: {d['prevalence_gap']:.4f}")
        print(f"\n  Saved: {_TABLES}/{out_prefix}_raw.csv")
        print(f"  Saved: {_TABLES}/{out_prefix}_summary.csv")
        if boot_rows:
            print(f"  Saved: {_TABLES}/{out_prefix}_bootstrap.csv")
        if diag_rows:
            print(f"  Saved: {_TABLES}/{out_prefix}_shiftdiag.csv")

    return {"raw": all_rows, "aggregated": agg, "bootstrap": boot_rows,
            "bootstrap_byseed": boot_by_seed, "bootstrap_ensemble": boot_ens_rows,
            "shiftdiag": diag_rows}


def run_real_data(
    data_dir: str | Path,
    source_center: str = "INT",
    target_center: str = "GHD",
    seeds: list[int] | None = None,
    use_iwcv: bool = True,
    out_prefix: str = "i3lung_cb",
    verbose: bool = True,
) -> dict:
    if seeds is None:
        seeds = list(range(10))
    src_df, tgt_df, feature_cols = load_i3lung(data_dir, source_center, target_center)
    return _run_transfer(
        src_df, tgt_df, feature_cols, label_col="OS_6",
        source_center=source_center, target_center=target_center,
        seeds=seeds, use_iwcv=use_iwcv, out_prefix=out_prefix,
        banner="I3LUNG real data", verbose=verbose,
    )


def run_public_data(
    registry_name: str,
    source_center: str,
    target_center: str,
    tau: float,
    seeds: list[int] | None = None,
    use_iwcv: bool = True,
    out_prefix: str | None = None,
    verbose: bool = True,
) -> dict:
    """Same protocol as ``run_real_data`` on a public multicentric survival
    registry, with the outcome binarised at a fixed horizon ``tau`` (months)."""
    if seeds is None:
        seeds = list(range(10))
    if out_prefix is None:
        out_prefix = f"public_{registry_name}_{source_center}_{target_center}"
    src_df, tgt_df, feature_cols = load_public_transfer(
        registry_name, source_center, target_center, tau,
    )
    return _run_transfer(
        src_df, tgt_df, feature_cols, label_col="event",
        source_center=source_center, target_center=target_center,
        seeds=seeds, use_iwcv=use_iwcv, out_prefix=out_prefix,
        banner=f"Public data [{registry_name}, tau={tau:g}m]", verbose=verbose,
        tau=tau,
    )


# ---------------------------------------------------------------------------
# Leave-one-center-out multi-source domain generalization (I3LUNG)
# ---------------------------------------------------------------------------

def _run_loco_seed(
    centers: dict,
    feature_cols: list[str],
    target_center: str,
    seed: int,
    use_iwcv: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """One LOCO fold/seed: pool every non-target center as a labeled source
    (each a DRO group / IRM environment), evaluate on the held-out target.

    The median imputer is fitted on the pooled source features only (no target
    leakage); each method function then does its own standardization internally.
    Reuses the synthetic-benchmark method functions unchanged via _CenterDataset.
    """
    from experiments.baselines import fit_all_observed, fit_prior_recalib, fit_platt_scaling
    from experiments.adaptation_methods import fit_iw_erm, fit_coral, fit_dann, fit_cdan, fit_irm
    from experiments.group_dro import fit_group_dro

    source_names = [c for c in centers if c != target_center]
    X_pool = np.vstack([centers[c][0].values for c in source_names])
    imputer = SimpleImputer(strategy="median").fit(X_pool)

    src_ds = [_CenterDataset(imputer.transform(centers[c][0].values), centers[c][1], i)
              for i, c in enumerate(source_names)]
    X_tgt, y_tgt_raw = centers[target_center]
    tgt_ds = _CenterDataset(imputer.transform(X_tgt.values), y_tgt_raw, domain_id=len(source_names))
    y_tgt = tgt_ds.y.astype(int)

    # ERM + target-using DA + multi-source DG (IRM, GroupDRO), all reusing the
    # same per-center source list (DA methods pool it; DG methods group by it).
    method_fns = [
        ("lr_erm",        lambda: fit_all_observed(src_ds, tgt_ds)),
        ("iw_erm",        lambda: fit_iw_erm(src_ds, tgt_ds, seed=seed)),
        ("prior_recalib", lambda: fit_prior_recalib(src_ds, tgt_ds)),
        ("platt_scaling", lambda: fit_platt_scaling(src_ds, tgt_ds)),
        ("coral",         lambda: fit_coral(src_ds, tgt_ds, seed=seed)),
        ("irm",           lambda: fit_irm(src_ds, tgt_ds, seed=seed)),
        ("group_dro",     lambda: fit_group_dro(src_ds, tgt_ds, seed=seed)),
        ("dann",          lambda: fit_dann(src_ds, tgt_ds, seed=seed, use_iwcv=use_iwcv)),
        ("cdan",          lambda: fit_cdan(src_ds, tgt_ds, seed=seed, use_iwcv=use_iwcv)),
    ]
    results = []
    for name, fn in method_fns:
        try:
            r = fn()
            probs = r.get("target_probs")
            results.append({
                "method":        name,
                "source_pooled": r.get("source_pooled", {}),
                "target":        r.get("target", {}),
                "tgt_probs":     np.asarray(probs) if probs is not None else None,
            })
        except Exception as e:
            warnings.warn(f"{name} failed (target={target_center}, seed={seed}): {e}")
            results.append({"method": name, "source_pooled": {}, "target": {}, "tgt_probs": None})
    return results, y_tgt


def run_i3lung_loco(
    data_dir: str | Path,
    seeds: list[int] | None = None,
    use_iwcv: bool = True,
    out_prefix: str = "i3lung_loco",
    verbose: bool = True,
) -> None:
    """Leave-one-center-out multi-source domain generalization on I3LUNG.

    For each held-out target center the remaining centers are pooled as labeled
    sources (each a DRO group / IRM environment).  Runs ERM, the target-using DA
    methods, and the multi-source DG methods (IRM, GroupDRO), with a patient-
    level bootstrap vs ERM per fold.  Writes <out_prefix>_<TARGET>_bootstrap.csv,
    which experiments.meta_analysis --prefix i3lung_loco then pools across folds.
    """
    if seeds is None:
        seeds = list(range(10))
    centers, feature_cols = load_i3lung_all(data_dir)
    names = list(centers.keys())
    if verbose:
        print(f"\nI3LUNG LOCO — {len(names)} centers: "
              + ", ".join(f"{c}(n={len(centers[c][1])})" for c in names))

    for target in names:
        all_rows: list[dict] = []
        prob_accum: dict[str, list[np.ndarray]] = {}
        y_ref: np.ndarray | None = None
        if verbose:
            print(f"\n=== target = {target}  (sources = {[c for c in names if c != target]}) ===")
        for seed in seeds:
            if verbose:
                print(f"  [seed {seed}]", flush=True)
            results, y_tgt = _run_loco_seed(centers, feature_cols, target, seed, use_iwcv)
            y_ref = y_tgt
            for r in results:
                all_rows.append(_flatten_real(r, seed))
                probs = r.get("tgt_probs")
                if probs is not None and np.asarray(probs).size == len(y_tgt):
                    prob_accum.setdefault(r["method"], []).append(np.asarray(probs))

        agg = _aggregate_real(all_rows)
        # Same headline/comparison pair as _run_transfer: the per-seed bootstrap
        # is the reported one, the seed-ensemble kept alongside for diffing.
        boot: list[dict] = []
        boot_reps: list[dict] = []
        boot_by_seed: list[dict] = []
        boot_ens: list[dict] = []
        if y_ref is not None:
            prob_by_method = {m: np.vstack(P) for m, P in prob_accum.items() if P}
            boot = _bootstrap_real_gaps_per_seed(
                prob_by_method, y_ref, reference="lr_erm",
                replicate_sink=boot_reps, seed_summary_sink=boot_by_seed)
            boot_ens = _bootstrap_real_gaps(prob_by_method, y_ref, reference="lr_erm")

        _TABLES.mkdir(parents=True, exist_ok=True)
        pfx = f"{out_prefix}_{target}"
        save_table_csv(all_rows, _TABLES / f"{pfx}_raw.csv")
        save_table_csv(agg,      _TABLES / f"{pfx}_summary.csv")
        if boot_reps:
            save_table_csv(boot_reps, _TABLES / f"{pfx}_bootreps.csv")
        if boot:
            save_table_csv(boot, _TABLES / f"{pfx}_bootstrap.csv")
        if boot_by_seed:
            save_table_csv(boot_by_seed, _TABLES / f"{pfx}_bootstrap_byseed.csv")
        if boot_ens:
            save_table_csv(boot_ens, _TABLES / f"{pfx}_bootstrap_ensemble.csv")
        if verbose:
            _print_real_results(agg, len(seeds), f"{len(names)-1}-source pool", target)
            _print_bootstrap(boot, "pooled-sources", target)
            print(f"  Saved: {_TABLES}/{pfx}_{{raw,summary,bootstrap,"
                  f"bootstrap_byseed,bootstrap_ensemble}}.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHIFT-SURV real-data experiment — I3LUNG or public multicentric registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="i3lung",
                        choices=["i3lung"] + list(_PUBLIC_REGISTRIES),
                        help="i3lung (private CSVs, needs --data_dir) or a public registry.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to I3LUNG_DATA directory (required for --dataset i3lung).")
    parser.add_argument("--source", type=str, default=None,
                        help="Source center name (default: INT for i3lung, Rotterdam for public).")
    parser.add_argument("--target", type=str, default=None,
                        help="Target center name (default: GHD for i3lung, GBSG2 for public).")
    parser.add_argument("--tau", type=float, default=None,
                        help="Fixed horizon in months for public survival binarisation "
                             "(default: the registry's TAU).")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--no_iwcv", dest="use_iwcv", action="store_false", default=True)
    parser.add_argument("--loco", action="store_true",
                        help="I3LUNG leave-one-center-out multi-source domain generalization "
                             "(adds IRM + GroupDRO; ignores --source/--target).")
    parser.add_argument("--list_centers", action="store_true",
                        help="For a public registry: print the available centers (name, n) and exit.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if args.dataset == "i3lung":
        if args.list_centers:
            parser.error("--list_centers is only for public registries")
        if not args.data_dir:
            parser.error("--data_dir is required for --dataset i3lung")
        if args.loco:
            run_i3lung_loco(
                data_dir=args.data_dir,
                seeds=seeds,
                use_iwcv=args.use_iwcv,
                out_prefix=args.out_prefix or "i3lung_loco",
                verbose=not args.quiet,
            )
        else:
            run_real_data(
                data_dir=args.data_dir,
                source_center=args.source or "INT",
                target_center=args.target or "GHD",
                seeds=seeds,
                use_iwcv=args.use_iwcv,
                out_prefix=args.out_prefix or "i3lung_cb",
                verbose=not args.quiet,
            )
    else:
        import importlib
        reg = importlib.import_module(_PUBLIC_REGISTRIES[args.dataset])

        if args.list_centers:
            centers, feats = reg.load_all_centers()
            print(f"\n{args.dataset}: {len(centers)} centers  (features: {feats})")
            for c in sorted(centers, key=lambda c: -c["n"]):
                print(f"  center_id={c['center_id']:<3d} n={c['n']:<7d} {reg.CENTER_NAMES[c['center_id']]}")
            sys.exit(0)

        # rotterdam_gbsg has sensible defaults; other registries need explicit pairs.
        default_src = "Rotterdam" if args.dataset == "rotterdam_gbsg" else None
        default_tgt = "GBSG2"     if args.dataset == "rotterdam_gbsg" else None
        source = args.source or default_src
        target = args.target or default_tgt
        if source is None or target is None:
            parser.error(
                f"--source and --target are required for --dataset {args.dataset}. "
                f"Run with --list_centers first to see the available center names."
            )
        tau = args.tau if args.tau is not None else float(reg.TAU)
        run_public_data(
            registry_name=args.dataset,
            source_center=source,
            target_center=target,
            tau=tau,
            seeds=seeds,
            use_iwcv=args.use_iwcv,
            out_prefix=args.out_prefix,
            verbose=not args.quiet,
        )
