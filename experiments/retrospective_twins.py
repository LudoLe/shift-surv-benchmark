"""Retrospective twins: which shift component reproduces the real method behaviour?

Builds three twins per transfer, each calibrated to a different part of the
measured discrepancy, and scores how well each reproduces the real transfer's
*method-effect vector* -- the per-method gaps against ERM.

  mean-only   latent means displaced until the standardised mean distance
              matches the observed d_mean; covariance left alone
  cov-only    target latent covariance anisotropically stretched until the
              covariance distance matches the observed d_cov; means left alone
  mean+cov    both, each matched to its own observed value

The three differ only in which component is present, and each component is
calibrated by bisection against a quantity measured on the real transfer from
*unlabelled covariates alone*. No mechanism is ever tuned against the real
method results: the calibration targets are d_mean and d_cov, and the method
results are only read afterwards, to score the twins. Tuning on them would make
the exercise circular -- of course some mixture reproduces the answer if you fit
to the answer.

Prevalence is matched in all three, by class-conditional resampling, because it
is directly observable and unambiguous. That is the one component the twin has
always identified rather than assumed.

Scoring, over the method-effect vectors (ERM as reference, so the compared
methods are IW-ERM and CORAL first, then any others that ran):
  sign_agreement  fraction of methods whose gap has the same sign
  spearman        rank correlation between synthetic and real gaps
  norm_error      ||synth - real|| / ||real||, so 0 is exact and 1 is as wrong
                  as predicting no effect at all
  auroc/ece_mae   mean absolute difference, in AUROC and ECE units

Usage
-----
  python -m experiments.retrospective_twins --audit outputs/tables/shift_audit_transfers.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler

from generator.parameters import (BaseParams, default_source_params,
                                  make_shared_loading_matrices)
from generator.base_generator import SyntheticGenerator, DomainDataset
from generator.scenarios import _covariate_shift
from experiments.shift_audit import _standardise, geometry
from utils.io import save_table_csv
from utils.metrics import auroc, ece

_TABLES = _ROOT / "outputs" / "tables"

# ERM is the reference; these are the compared methods, in the order the spec asks.
COMPARED = ["iw_erm", "coral", "dann", "cdan"]
ERM_KEY = "all_observed"
VARIANTS = ["mean_only", "cov_only", "mean_plus_cov"]


# ---------------------------------------------------------------------------
# Twin construction
# ---------------------------------------------------------------------------

def build_twin(spec: dict, mu_scale: float, cov_scale: float, seed: int
               ) -> tuple[DomainDataset, DomainDataset]:
    """Twin with an independently controllable mean and covariance component.

    The covariance component is anisotropic with unit determinant: one invariant
    direction is stretched by cov_scale and another compressed by 1/cov_scale.
    Scaling the whole block instead would raise the total variance of the
    risk-carrying factors and make the target *easier* -- which is a change in
    discriminability, not in the shape of the discrepancy, and would confound
    the comparison.
    """
    base = BaseParams(p=spec["n_features"], seed=seed)
    pool = max(spec["n_src"], spec["n_tgt"]) * 3
    sA_inv, sA_spu, sB_nl = make_shared_loading_matrices(base, seed=seed)
    gen = SyntheticGenerator(base, shared_A_inv=sA_inv, shared_A_spu=sA_spu,
                             shared_B_nl=sB_nl)
    sp = default_source_params(base, sA_inv, sA_spu, n_domains=1, n_per_domain=pool)
    src_p, tgt_p = _covariate_shift(sp, 1, pool, base, seed,
                                    mu_inv_shift_scale=mu_scale,
                                    mu_spu_shift_scale=mu_scale)
    if cov_scale != 1.0 and base.k_inv >= 2:
        k = base.k_inv + base.k_spu
        C = np.eye(k) if tgt_p.latent_cov is None else np.array(tgt_p.latent_cov, float)
        C[0, 0] *= cov_scale
        C[1, 1] /= cov_scale
        tgt_p.latent_cov = C

    src_ds = gen.generate_domains(src_p)
    tgt_ds = gen.generate_domain(tgt_p, domain_id=1)

    rng = np.random.default_rng(seed + 1)

    def resample(ds, n, prev, did):
        pos, neg = np.where(ds.y == 1)[0], np.where(ds.y == 0)[0]
        n_pos = int(round(n * prev))
        if len(pos) == 0 or len(neg) == 0:
            idx = rng.integers(0, len(ds.y), n)
        else:
            idx = np.r_[rng.choice(pos, n_pos, replace=len(pos) < n_pos),
                        rng.choice(neg, n - n_pos, replace=len(neg) < n - n_pos)]
        rng.shuffle(idx)
        return DomainDataset(domain_id=did, u_inv=ds.u_inv[idx], u_spu=ds.u_spu[idx],
                             x=ds.x[idx], eta=ds.eta[idx], T=ds.T[idx], y=ds.y[idx],
                             params=ds.params)

    return (resample(src_ds[0], spec["n_src"], spec["prev_src"], 0),
            resample(tgt_ds, spec["n_tgt"], spec["prev_tgt"], 1))


def twin_geometry(spec, mu_scale, cov_scale, seeds=4) -> dict:
    """d_mean / d_cov of the twin, measured exactly as on the real transfer."""
    g = []
    for s in range(seeds):
        S, T = build_twin(spec, mu_scale, cov_scale, s)
        a, c = _standardise(S.x, T.x)
        g.append(geometry(a, c))
    return {k: float(np.mean([x[k] for x in g])) for k in g[0]}


def _bisect(fn, target: float, lo: float, hi: float, iters: int = 12) -> float:
    """Smallest knob value whose diagnostic reaches `target`. Monotone in both
    knobs over these ranges, so plain bisection is enough."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate(spec: dict, variant: str) -> tuple[float, float]:
    """Knobs for one variant, calibrated ONLY against d_mean / d_cov."""
    if variant == "mean_only":
        mu = _bisect(lambda m: twin_geometry(spec, m, 1.0)["d_mean"],
                     spec["d_mean"], 0.0, 6.0)
        return mu, 1.0
    if variant == "cov_only":
        cs = _bisect(lambda c: twin_geometry(spec, 0.0, c)["d_cov"],
                     spec["d_cov"], 1.0, 8.0)
        return 0.0, cs
    mu = _bisect(lambda m: twin_geometry(spec, m, 1.0)["d_mean"],
                 spec["d_mean"], 0.0, 6.0)
    cs = _bisect(lambda c: twin_geometry(spec, mu, c)["d_cov"],
                 spec["d_cov"], 1.0, 8.0)
    return mu, cs


# ---------------------------------------------------------------------------
# Method effects and scoring
# ---------------------------------------------------------------------------

def effect_vector(S, T, seeds: list[int]) -> dict:
    """Per-method (AUROC gap, ECE gap) against ERM, averaged over seeds."""
    from experiments.synthetic_twin import run_methods
    au, ec = {}, {}
    for s in seeds:
        probs = run_methods(S, T, seed=s)
        if ERM_KEY not in probs:
            continue
        a0, e0 = auroc(T.y, probs[ERM_KEY]), ece(T.y, probs[ERM_KEY])
        for m, p in probs.items():
            # Oracles are not comparable methods: on real data u_inv is aliased
            # to x, so the latent oracle degenerates to ERM and reports 0.
            if m == ERM_KEY or "oracle" in m:
                continue
            au.setdefault(m, []).append(auroc(T.y, p) - a0)
            ec.setdefault(m, []).append(e0 - ece(T.y, p))       # positivo = meglio
    return {"auroc": {m: float(np.mean(v)) for m, v in au.items()},
            "ece": {m: float(np.mean(v)) for m, v in ec.items()}}


def score(synth: dict, real: dict, methods: list[str]) -> dict:
    out = {}
    for metric in ("auroc", "ece"):
        s = np.array([synth[metric].get(m, np.nan) for m in methods], float)
        r = np.array([real[metric].get(m, np.nan) for m in methods], float)
        ok = ~(np.isnan(s) | np.isnan(r))
        s, r = s[ok], r[ok]
        if len(s) < 2:
            continue
        out[f"{metric}_sign_agreement"] = round(float(np.mean(np.sign(s) == np.sign(r))), 4)
        out[f"{metric}_mae"] = round(float(np.mean(np.abs(s - r))), 5)
        denom = np.linalg.norm(r)
        out[f"{metric}_norm_error"] = (round(float(np.linalg.norm(s - r) / denom), 4)
                                       if denom > 1e-12 else float("nan"))
        if len(s) >= 3 and np.std(s) > 1e-12 and np.std(r) > 1e-12:
            out[f"{metric}_spearman"] = round(float(sp_stats.spearmanr(s, r).statistic), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(_TABLES / "shift_audit_transfers.csv"))
    ap.add_argument("--transfers", nargs="*", default=None,
                    help="sottoinsieme di transfer (default: tutti quelli nell'audit)")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tau", type=float, default=36.0)
    ap.add_argument("--out_prefix", default="retro_twins")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    # stdout is block-buffered when it is not a terminal, so a run piped to tee
    # shows nothing for hours while TensorFlow's stderr warnings scroll past,
    # which looks exactly like a hang. Line buffering costs nothing here.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:                                    # Python < 3.7
        pass

    audit = pd.read_csv(args.audit)
    if args.transfers:
        audit = audit[audit["transfer"].isin(args.transfers)]
    if audit.empty:
        raise SystemExit(f"nessun transfer in {args.audit}")

    from experiments.run_real_data import load_public_transfer, load_i3lung
    seeds = list(range(args.n_seeds))
    rows = []

    for _, a in audit.iterrows():
        name = a["transfer"]
        parts = name.split("_")
        if name.startswith("registry_"):
            src, tgt, feats = load_public_transfer("rotterdam_gbsg", parts[1], parts[2],
                                                   tau=args.tau)
            label = "event"
        else:
            if not args.data_dir:
                print(f"  salto {name}: serve --data_dir")
                continue
            src, tgt, feats = load_i3lung(args.data_dir, parts[1], parts[2])
            label = "OS_6"

        spec = {"n_src": int(a["n_src"]), "n_tgt": int(a["n_tgt"]),
                "n_features": int(a["n_features"]),
                "prev_src": float(a["prev_src_retro"]), "prev_tgt": float(a["prev_tgt_retro"]),
                "d_mean": float(a["d_mean"]), "d_cov": float(a["d_cov"])}

        print(f"\n=== {name}   d_mean {spec['d_mean']:.3f}  d_cov {spec['d_cov']:.3f}  "
              f"cov_share {a['cov_share']:.3f}")

        # Real effect vector, from the same code path as the twins.
        real_S, real_T = real_datasets(src, tgt, feats, label)
        real = effect_vector(real_S, real_T, seeds)
        print("  reale     " + "  ".join(
            f"{m} {real['auroc'].get(m, float('nan')):+.4f}" for m in COMPARED))

        for variant in VARIANTS:
            mu, cs = calibrate(spec, variant)
            got = twin_geometry(spec, mu, cs)
            S, T = build_twin(spec, mu, cs, 0)
            synth = effect_vector(S, T, seeds)
            row = {"transfer": name, "variant": variant,
                   "mu_scale": round(mu, 4), "cov_scale": round(cs, 4),
                   "d_mean_target": spec["d_mean"], "d_mean_got": round(got["d_mean"], 4),
                   "d_cov_target": spec["d_cov"], "d_cov_got": round(got["d_cov"], 4)}
            for m in COMPARED:
                row[f"auroc_gap_{m}"] = round(synth["auroc"].get(m, float("nan")), 5)
                row[f"real_auroc_gap_{m}"] = round(real["auroc"].get(m, float("nan")), 5)
            row.update(score(synth, real, COMPARED))
            rows.append(row)
            print(f"  {variant:<14}" + "  ".join(
                f"{m} {synth['auroc'].get(m, float('nan')):+.4f}" for m in COMPARED) +
                f"   sign {row.get('auroc_sign_agreement', float('nan')):.2f}"
                f"  nerr {row.get('auroc_norm_error', float('nan')):.2f}")

    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / f"{args.out_prefix}.csv")
    df = pd.DataFrame(rows)
    print("\n=== media per variante (su tutti i transfer)")
    for v in VARIANTS:
        d = df[df["variant"] == v]
        if d.empty:
            continue
        print(f"  {v:<14} sign {d['auroc_sign_agreement'].mean():.2f}   "
              f"norm_err {d['auroc_norm_error'].mean():.2f}   "
              f"mae {d['auroc_mae'].mean():.4f}")
    print(f"\n  Salvato: {_TABLES}/{args.out_prefix}.csv")


def real_datasets(src_df, tgt_df, feats, label) -> tuple[DomainDataset, DomainDataset]:
    """Wrap a real transfer so effect_vector uses the same code path as the twins.

    Preprocessing is fitted on the SOURCE and applied to both. Scaling each
    cohort by its own statistics would centre and rescale the target
    independently, which removes most of the very shift being studied: an
    earlier version did that and reported CORAL at -0.0008 on a transfer where
    run_real_data measures +0.007.

    u_inv/u_spu are aliased to x because no latent factors exist on real data.
    The oracles are therefore meaningless here and effect_vector drops them.
    """
    Xs = np.asarray(src_df[feats].to_numpy(float))
    Xt = np.asarray(tgt_df[feats].to_numpy(float))
    Xs_s, Xt_s = _standardise(Xs, Xt)

    def _wrap(X, y, did):
        n = len(y)
        return DomainDataset(domain_id=did, u_inv=X, u_spu=X[:, :2], x=X,
                             eta=np.zeros(n), T=np.zeros(n), y=y, params=None)

    return (_wrap(Xs_s, src_df[label].to_numpy().astype(float), 0),
            _wrap(Xt_s, tgt_df[label].to_numpy().astype(float), 1))


if __name__ == "__main__":
    main()
