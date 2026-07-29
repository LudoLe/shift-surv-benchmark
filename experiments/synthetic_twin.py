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
the same patient-level bootstrap, so the comparison is like for like.

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from generator.base_generator import DomainDataset
from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.io import save_table_csv
from utils.shift_diag import rbf_mmd
from utils.metrics import ece

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
    """Patient-level stratified bootstrap of each method's gap vs ERM."""
    methods = [m for m in probs if m not in (ERM_KEY, ORACLE_KEY)]
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    D_auc = np.zeros((len(methods), n_boot))
    D_ece = np.zeros((len(methods), n_boot))
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        yb = y[idx]
        ref_a = roc_auc_score(yb, probs[ERM_KEY][idx])
        ref_e = ece(yb, probs[ERM_KEY][idx])
        for i, m in enumerate(methods):
            D_auc[i, b] = roc_auc_score(yb, probs[m][idx]) - ref_a
            D_ece[i, b] = ref_e - ece(yb, probs[m][idx])
    return methods, D_auc, D_ece


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

    S, T = build_twin(spec, scale, args.seed)
    print(f"  prevalenze gemello: {S.y.mean():.3f} / {T.y.mean():.3f}\n")

    print("  metodi sul gemello:")
    probs = run_methods(S, T, seed=args.seed)
    if ERM_KEY not in probs:
        raise SystemExit("ERM non disponibile, impossibile calcolare i gap")
    names, D_auc, D_ece = bootstrap_gaps(probs, T.y, args.n_boot, args.seed)

    rows = []
    print(f"\n  {'metodo':<18}{'gap AUROC':>12}{'gap ECE':>12}")
    for i, m in enumerate(names):
        print(f"  {m:<18}{D_auc[i].mean():>+12.4f}{D_ece[i].mean():>+12.4f}")
        rows.append(dict(transfer=tag, method=m,
                         twin_auroc_gap=round(float(D_auc[i].mean()), 5),
                         twin_ece_gap=round(float(D_ece[i].mean()), 5),
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
