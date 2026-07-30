"""
Synthetic twins of the real transfers: does the benchmark PREDICT what
adaptation recovers on real data, from measurable quantities alone?

The paper currently claims consistency between the synthetic benchmark and the
real cohorts. This is a stronger test. For a given real transfer we measure the
only four things that can be measured on real data without target labels --

    source and target sample size
    source and target event prevalence
    the RBF-MMD between standardised feature sets
    the number of covariates

-- then build a synthetic cohort matched to all four, by combining the two
mechanisms those quantities correspond to: a covariate shift whose magnitude is
tuned by bisection until the synthetic MMD lands on the observed one, and a
label-prior shift imposed by class-conditional resampling to the observed
prevalences. Every other mechanism is left absent.

If the twin reproduces the observed gains, then the two measurable mechanisms
are sufficient to explain what adaptation achieves on that cohort, and no
appeal to unmeasured measurement or concept shift is needed. If it does not,
the residual is evidence that something unmeasured is doing the work -- which is
also worth reporting.

Both the real transfer and its twin are evaluated with the same method set and
the same per-seed patient-level bootstrap, so the comparison is like for like:
one cohort held fixed, n_seeds training runs on it, patients resampled with
replacement, and the (replicate, seed) cells pooled. Keep --n_seeds equal to
the real transfer's or the two intervals stop being comparable.

Usage
-----
  # breast-cancer registry, both directions, one horizon
  python -m experiments.synthetic_twin --dataset rotterdam_gbsg --tau 36 \
      --source GBSG2 --target Rotterdam

  # an I3LUNG transfer
  python -m experiments.synthetic_twin --data_dir /path/to/I3LUNG_DATA \
      --source INT --target GHD

Writes outputs/tables/twin_<source>_<target>[_tau<NN>].csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from generator.base_generator import DomainDataset
from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.io import save_table_csv
from utils.shift_diag import rbf_mmd

_TABLES = _ROOT / "outputs" / "tables"
ERM_KEY = "all_observed"
ORACLE_KEY = "invariant_oracle"          # a reference, never a competitor
# Monotone transformations of the ERM scores: their AUROC gap is exactly zero by
# construction, so they are compared on calibration only.
RANK_PRESERVING = {"prior_recalib", "platt_scaling", "isotonic"}


# ---------------------------------------------------------------------------
# Measuring the real transfer
# ---------------------------------------------------------------------------

def measure(src_df, tgt_df, feature_cols, label_col):
    """The four quantities a clinician could compute without target labels."""
    Xs = src_df[feature_cols].to_numpy(float)
    Xt = tgt_df[feature_cols].to_numpy(float)
    imp = SimpleImputer(strategy="median").fit(Xs)
    sc = StandardScaler().fit(imp.transform(Xs))
    Xs_s, Xt_s = sc.transform(imp.transform(Xs)), sc.transform(imp.transform(Xt))
    return dict(
        n_src=len(src_df), n_tgt=len(tgt_df),
        prev_src=float(src_df[label_col].mean()),
        prev_tgt=float(tgt_df[label_col].mean()),
        mmd=float(rbf_mmd(Xs_s, Xt_s)),
        n_features=len(feature_cols),
    )


# ---------------------------------------------------------------------------
# Building the twin
# ---------------------------------------------------------------------------

def build_twin(spec, mu_scale, seed):
    """Covariate shift of the given magnitude, then class-conditional resampling
    to the observed prevalences. Returns two DomainDataset-shaped objects."""
    base = BaseParams(p=spec["n_features"], seed=seed)
    pool = max(spec["n_src"], spec["n_tgt"]) * 3
    d = generate_scenario("covariate_shift", n_source_domains=1, n_per_domain=pool,
                          seed=seed, base=base,
                          mu_inv_shift_scale=mu_scale, mu_spu_shift_scale=mu_scale)
    rng = np.random.default_rng(seed + 1)

    def resample(ds, n, prev, domain_id):
        pos, neg = np.where(ds.y == 1)[0], np.where(ds.y == 0)[0]
        n_pos = int(round(n * prev))
        idx = np.r_[rng.choice(pos, n_pos, replace=len(pos) < n_pos),
                    rng.choice(neg, n - n_pos, replace=len(neg) < n - n_pos)]
        rng.shuffle(idx)
        return DomainDataset(domain_id=domain_id, u_inv=ds.u_inv[idx],
                             u_spu=ds.u_spu[idx], x=ds.x[idx], eta=ds.eta[idx],
                             T=ds.T[idx], y=ds.y[idx], params=ds.params)

    S = resample(d["source"][0], spec["n_src"], spec["prev_src"], 0)
    T = resample(d["target"], spec["n_tgt"], spec["prev_tgt"], 1)
    return S, T


def twin_mmd(spec, mu_scale, seeds=5):
    vals = []
    for s in range(seeds):
        S, T = build_twin(spec, mu_scale, s)
        sc = StandardScaler().fit(S.x)
        vals.append(rbf_mmd(sc.transform(S.x), sc.transform(T.x)))
    return float(np.mean(vals))


def tune_scale(spec, iters=12):
    """Bisection on the covariate-shift magnitude until the twin's MMD matches."""
    lo, hi = 0.0, 4.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if twin_mmd(spec, mid) < spec["mmd"]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Evaluation, identical for the twin and the real transfer
# ---------------------------------------------------------------------------

def _name(r):
    """The baselines label themselves with "feature_set", the adaptation
    methods with "method"."""
    return r.get("method") or r.get("feature_set")


def run_methods(S, T, seed=0):
    """Full method set when the adapt/TF stack is importable, linear subset
    otherwise. Returns {method: target probabilities}."""
    from experiments.baselines import (fit_all_observed, fit_invariant_oracle,
                                       fit_prior_recalib, fit_platt_scaling)
    out = {}

    def _call(fn, takes_seed):
        """The baselines take (sources, target); the adaptation methods also
        take a seed. Try the richer signature first, fall back if rejected."""
        try:
            r = fn([S], T, seed=seed) if takes_seed else fn([S], T)
        except TypeError:
            r = fn([S], T)
        return r

    for fn in (fit_all_observed, fit_invariant_oracle, fit_prior_recalib,
               fit_platt_scaling):
        try:
            r = _call(fn, takes_seed=False)
            if r.get("target_probs") is not None:
                out[_name(r)] = np.asarray(r["target_probs"])
        except Exception as e:                                   # noqa: BLE001
            print(f"    ! {fn.__name__}: {type(e).__name__}: {e}")
    try:
        from experiments.adaptation_methods import fit_coral, fit_dann, fit_cdan, fit_iw_erm
        for fn in (fit_coral, fit_iw_erm, fit_dann, fit_cdan):
            try:
                r = _call(fn, takes_seed=True)
                if r.get("target_probs") is not None:
                    out[_name(r)] = np.asarray(r["target_probs"])
            except Exception as e:                               # noqa: BLE001
                print(f"    ! {fn.__name__}: {type(e).__name__}: {e}")
    except ImportError:
        print("    (adapt/TensorFlow assente: CORAL, DANN, CDAN, IW-ERM saltati)")
    return out


def bootstrap_gaps(probs, y, n_boot=2000, seed=0):
    """Per-seed patient-level stratified bootstrap of each method's gap vs ERM.

    ``probs[m]`` is an (n_seeds, n_tgt) array: one row per training seed, all
    rows evaluated on the same fixed twin cohort.

    Why per-seed, and why the cohort is fixed
    -----------------------------------------
    This has to be the *same estimator* the real transfer uses, or the
    twin-vs-real comparison stops being like for like -- which is the whole
    point of the twin. On the real side the cohort cannot be redrawn, so the
    interval is built by varying the training seed and resampling patients with
    replacement. The twin therefore does exactly that too: one cohort, drawn
    once, then n_seeds training runs and a shared patient bootstrap over both.

    It would be tempting to regenerate the twin cohort at every seed, since
    unlike the real data we can. That would be wrong here: the twin's interval
    would then carry genuine cohort resampling while the real one carries the
    bootstrap approximation of it, and the two would differ for a reason that
    has nothing to do with the transfer being modelled.

    The (b, s) cells are pooled into a single gap distribution per method, so
    the returned D matrices are (n_methods, n_seeds * n_boot) and ``max_stat``
    consumes them unchanged. Gaps are formed inside a cell -- same patients,
    same seed -- so methods stay paired.

    ``_fast_auroc`` / ``_fast_ece`` are imported from ``run_real_data`` rather
    than reimplemented, so the metric definitions and the stratified index
    scheme are literally shared with the real-data path.
    """
    from experiments.run_real_data import _fast_auroc, _fast_ece

    methods = [m for m in probs if m not in (ERM_KEY, ORACLE_KEY)]
    P = {m: np.atleast_2d(np.asarray(v, dtype=float)) for m, v in probs.items()}
    n_seeds = P[ERM_KEY].shape[0]
    y = np.asarray(y).astype(int)

    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]

    # Indices shared by every method AND every seed, so the pairing holds at
    # both levels.
    idx_all = [np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
               for _ in range(n_boot)]

    n_cells = n_boot * n_seeds
    D_auc = np.full((len(methods), n_cells), np.nan)
    D_ece = np.full((len(methods), n_cells), np.nan)

    c = 0
    for idx in idx_all:
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            c += n_seeds
            continue
        for s in range(n_seeds):
            ref_a = _fast_auroc(yb, P[ERM_KEY][s][idx])
            ref_e = _fast_ece(yb, P[ERM_KEY][s][idx])
            for i, m in enumerate(methods):
                pb = P[m][s][idx]
                D_auc[i, c] = _fast_auroc(yb, pb) - ref_a
                D_ece[i, c] = ref_e - _fast_ece(yb, pb)
            c += 1

    keep = ~np.isnan(D_auc[0]) if len(methods) else np.zeros(n_cells, bool)
    return methods, D_auc[:, keep], D_ece[:, keep]


def max_stat(D, names):
    mean, sd = D.mean(1), D.std(1, ddof=1)
    sd = np.where(sd < 1e-12, np.inf, sd)
    t = mean / sd
    k = int(t.argmax())
    null = ((D - mean[:, None]) / sd[:, None]).max(0)
    return names[k], float(mean[k]), float((null >= t[k]).mean())


# ---------------------------------------------------------------------------

def load_real(args):
    from experiments.run_real_data import load_i3lung, load_public_transfer
    if args.data_dir:
        src, tgt, feats = load_i3lung(args.data_dir, args.source, args.target)
        return src, tgt, feats, "OS_6"
    src, tgt, feats = load_public_transfer(args.dataset, args.source, args.target,
                                           tau=args.tau)
    return src, tgt, feats, "event"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="I3LUNG_DATA (i3lung)")
    ap.add_argument("--dataset", default="rotterdam_gbsg", help="registro pubblico")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=10,
                    help="training seeds on the fixed twin cohort; must match "
                         "the real transfer's --n_seeds for the comparison to "
                         "be like for like")
    args = ap.parse_args()

    src_df, tgt_df, feats, label = load_real(args)
    spec = measure(src_df, tgt_df, feats, label)
    tag = f"{args.source}_{args.target}" + (f"_tau{int(args.tau)}" if args.tau else "")

    print(f"\nTRASFERIMENTO REALE  {args.source} -> {args.target}"
          + (f"  (tau={args.tau:.0f})" if args.tau else ""))
    print(f"  n {spec['n_src']}/{spec['n_tgt']}   prevalenza "
          f"{spec['prev_src']:.3f}/{spec['prev_tgt']:.3f}   "
          f"MMD {spec['mmd']:.4f}   {spec['n_features']} covariate")

    print("\n  taratura del gemello...")
    scale = tune_scale(spec)
    got = twin_mmd(spec, scale, seeds=10)
    print(f"  scala covariate shift {scale:.3f}  ->  MMD gemello {got:.4f} "
          f"(bersaglio {spec['mmd']:.4f})")

    # One twin cohort, held fixed across seeds -- see bootstrap_gaps for why.
    S, T = build_twin(spec, scale, args.seed)
    print(f"  prevalenze gemello: {S.y.mean():.3f} / {T.y.mean():.3f}\n")

    print(f"  metodi sul gemello ({args.n_seeds} seed, coorte fissa):")
    per_seed = []
    for s in range(args.seed, args.seed + args.n_seeds):
        print(f"    [seed {s}]", flush=True)
        per_seed.append(run_methods(S, T, seed=s))

    # Keep only methods that every seed produced, so the probability matrices
    # are rectangular and the (b, s) cells stay comparable across methods.
    common = set(per_seed[0])
    for p in per_seed[1:]:
        common &= set(p)
    dropped = (set(per_seed[0]) | set().union(*(set(p) for p in per_seed))) - common
    if dropped:
        print(f"    ! metodi scartati (non prodotti da tutti i seed): {sorted(dropped)}")
    if ERM_KEY not in common:
        raise SystemExit("ERM non disponibile, impossibile calcolare i gap")

    probs = {m: np.vstack([p[m] for p in per_seed]) for m in common}
    names, D_auc, D_ece = bootstrap_gaps(probs, T.y, args.n_boot, args.seed)

    rows = []
    print(f"\n  {'metodo':<18}{'gap AUROC [95% CI]':>30}{'gap ECE [95% CI]':>30}")
    for i, m in enumerate(names):
        a_lo, a_hi = np.percentile(D_auc[i], [2.5, 97.5])
        e_lo, e_hi = np.percentile(D_ece[i], [2.5, 97.5])
        print(f"  {m:<18}"
              f"{f'{D_auc[i].mean():+.4f} [{a_lo:+.4f},{a_hi:+.4f}]':>30}"
              f"{f'{D_ece[i].mean():+.4f} [{e_lo:+.4f},{e_hi:+.4f}]':>30}")
        rows.append(dict(transfer=tag, method=m,
                         twin_auroc_gap=round(float(D_auc[i].mean()), 5),
                         twin_auroc_gap_lo=round(float(a_lo), 5),
                         twin_auroc_gap_hi=round(float(a_hi), 5),
                         twin_ece_gap=round(float(D_ece[i].mean()), 5),
                         twin_ece_gap_lo=round(float(e_lo), 5),
                         twin_ece_gap_hi=round(float(e_hi), 5),
                         n_seeds=args.n_seeds, n_boot=args.n_boot,
                         n_cells=int(D_auc.shape[1]),
                         **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in spec.items()}))
    for lbl, D, drop in (("AUROC", D_auc, RANK_PRESERVING), ("ECE", D_ece, set())):
        keep = [i for i, m in enumerate(names) if m not in drop]
        if not keep:
            continue
        b, g, p = max_stat(D[keep], [names[i] for i in keep])
        print(f"  -> {lbl}: best {b}, gap {g:+.4f}, p = {p:.4f}")

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / f"twin_{tag}.csv")
    print(f"\n  Salvato: outputs/tables/twin_{tag}.csv")


if __name__ == "__main__":
    main()
