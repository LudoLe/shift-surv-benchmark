"""
A synthetic twin of the breast-cancer registry: does the benchmark predict what
CORAL actually recovers there?

The registry transfer GBSG2 -> Rotterdam at a 36-month horizon is characterised
by four measured quantities:

    source n = 555, prevalence 0.404      (GBSG2)
    target n = 2894, prevalence 0.301     (Rotterdam)
    RBF-MMD between standardised features = 0.0996 (permutation p = 0.001)
    8 shared covariates

and on it CORAL gains +0.0056 AUROC (bootstrap max-statistic p = 0.008).

This script builds a synthetic cohort matched to all four, by combining the two
mechanisms that are measurable on real data -- a covariate shift, with its
magnitude tuned so the synthetic RBF-MMD lands on the observed 0.0996, and a
label-prior shift, imposed by class-conditional resampling of the target to the
observed prevalence -- then measures what CORAL recovers under the same
patient-level bootstrap used on the real cohort.

If the synthetic twin reproduces the observed gain, the benchmark predicts the
real result from measurable quantities alone, which is a stronger claim than
consistency: it means the two shift mechanisms we can see on real data are
sufficient to explain what adaptation achieves there.

Methods are the linear ones, implemented locally so this runs without the
TensorFlow stack. The registry result being probed is CORAL's.

Run from repo root:  python -m experiments.registry_synthetic_twin
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.shift_diag import rbf_mmd
from experiments.registry_power_check import fit_methods, bootstrap_maxstat, ERM

# ---- measured registry characteristics, GBSG2 -> Rotterdam, tau = 36 --------
N_SRC, N_TGT = 555, 2894
PREV_SRC, PREV_TGT = 0.4036, 0.3013
MMD_OBS = 0.0996
N_FEATURES = 8
CORAL_OBSERVED = 0.0056        # AUROC gain measured on the real transfer


def standardise(Xs, Xt):
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    return (Xs - mu) / sd, (Xt - mu) / sd


def build(mu_scale, seed, n_src=N_SRC, n_tgt=N_TGT):
    """Covariate shift of the given magnitude, then resample the target to the
    observed prevalence so both measurable mechanisms are present at once."""
    base = BaseParams(p=N_FEATURES, seed=seed)
    d = generate_scenario("covariate_shift", n_source_domains=1,
                          n_per_domain=max(n_src, n_tgt) * 3, seed=seed, base=base,
                          mu_inv_shift_scale=mu_scale, mu_spu_shift_scale=mu_scale)
    S, T = d["source"][0], d["target"]
    rng = np.random.default_rng(seed + 1)

    def resample(ds, n, prev):
        pos, neg = np.where(ds.y == 1)[0], np.where(ds.y == 0)[0]
        n_pos = int(round(n * prev))
        idx = np.r_[rng.choice(pos, n_pos, len(pos) < n_pos),
                    rng.choice(neg, n - n_pos, len(neg) < n - n_pos)]
        rng.shuffle(idx)
        return ds.x[idx], ds.y[idx], ds.u_inv[idx]

    Xs, ys, us = resample(S, n_src, PREV_SRC)
    Xt, yt, ut = resample(T, n_tgt, PREV_TGT)
    return Xs, ys, us, Xt, yt, ut


def measured_mmd(mu_scale, seeds=5):
    vals = []
    for s in range(seeds):
        Xs, _, _, Xt, _, _ = build(mu_scale, s)
        a, b = standardise(Xs, Xt)
        vals.append(rbf_mmd(a, b))
    return float(np.mean(vals))


def tune():
    """Bisection on the covariate-shift magnitude to hit the observed MMD."""
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if measured_mmd(mid) < MMD_OBS:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate(mu_scale, n_boot=2000, seed=0):
    Xs, ys, us, Xt, yt, ut = build(mu_scale, seed)
    p = fit_methods(Xs, ys, Xt, us, ut)
    names = ["CORAL", "IW-ERM"]
    rng = np.random.default_rng(seed)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    D = np.zeros((len(names), n_boot))
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        yb = yt[idx]
        ref = roc_auc_score(yb, p[ERM][idx])
        for i, m in enumerate(names):
            D[i, b] = roc_auc_score(yb, p[m][idx]) - ref
    a = {m: roc_auc_score(yt, v) for m, v in p.items()}
    return bootstrap_maxstat(D, names), a, {m: D[i].mean() for i, m in enumerate(names)}


if __name__ == "__main__":
    print("GEMELLO SINTETICO DEL REGISTRY  (GBSG2 -> Rotterdam, tau = 36)\n")
    print(f"  bersagli misurati: n {N_SRC}/{N_TGT}, prevalenza {PREV_SRC:.3f}/{PREV_TGT:.3f}, "
          f"MMD {MMD_OBS:.4f}, {N_FEATURES} covariate")

    scale = tune()
    got = measured_mmd(scale, seeds=10)
    print(f"\n  taratura covariate shift: scala {scale:.3f}  ->  MMD sintetica {got:.4f} "
          f"(bersaglio {MMD_OBS:.4f})")

    (best, gap, p), aurocs, means = evaluate(scale)
    Xs, ys, _, Xt, yt, _ = build(scale, 0)
    print(f"  prevalenze ottenute: source {ys.mean():.3f}  target {yt.mean():.3f}\n")
    print("  AUROC sul target:")
    for m in ("oracle", ERM, "CORAL", "IW-ERM"):
        print(f"     {m:<8s} {aurocs[m]:.4f}")
    print(f"\n  guadagni rispetto a ERM (bootstrap pazienti, 2000 repliche):")
    for m, v in means.items():
        print(f"     {m:<8s} {v:+.4f}")
    print(f"     -> best {best}, gap {gap:+.4f}, p = {p:.4f}")
    print(f"\n  CORAL osservato sul registry vero: {CORAL_OBSERVED:+.4f}")
    print(f"  CORAL sul gemello sintetico:       {means['CORAL']:+.4f}")
