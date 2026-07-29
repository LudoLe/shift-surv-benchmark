"""
Is the synthetic/registry discrepancy an effect-size difference or a power one?

On synthetic covariate shift CORAL gains +0.005 AUROC at p=0.062 (ten seeds,
500 patients per centre). On the registry it gains +0.010 at p<0.001. The two
studies differ in two ways at once -- the data, and how uncertainty is measured
-- so the comparison of p-values alone cannot say which is responsible.

This script holds the data-generating process fixed and swaps only the
measurement design, running the SAME synthetic covariate shift under:

  A. the benchmark design    500 source / 500 target, uncertainty across 10 seeds,
                             tested by exact sign-flip max-statistic
  B. the registry design     ~600 source / ~2900 target (GBSG2 -> Rotterdam),
                             uncertainty from a patient-level bootstrap on the
                             target, tested by the bootstrap max-statistic

If B turns the same effect significant, the discrepancy is power, not biology.

Methods are the linear ones, implemented here directly so the script runs
without TensorFlow: ERM, IW-ERM, CORAL, plus the latent oracle for reference.
DANN and CDAN are omitted -- they need the adapt/TF stack, and the registry
result being probed is CORAL's.

Run from repo root:  python -m experiments.registry_power_check
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from generator.scenarios import generate_scenario

LR = dict(max_iter=1000, solver="lbfgs", C=1.0)
ERM = "ERM"


def _sqrtm_reg(C, eps=1e-6):
    C = C + eps * np.eye(C.shape[0])
    return linalg.sqrtm(C).real


def fit_methods(Xs, ys, Xt, us_inv, ut_inv):
    """Return {method: target probabilities}. Source-only fitting throughout."""
    sc = StandardScaler().fit(Xs)
    Xs_s, Xt_s = sc.transform(Xs), sc.transform(Xt)
    out = {}

    out[ERM] = LogisticRegression(**LR).fit(Xs_s, ys).predict_proba(Xt_s)[:, 1]

    # CORAL: recolour the source to the target covariance, then predict target.
    Cs, Ct = np.cov(Xs_s, rowvar=False), np.cov(Xt_s, rowvar=False)
    A = np.linalg.pinv(_sqrtm_reg(Cs)) @ _sqrtm_reg(Ct)
    Xs_coral = (Xs_s - Xs_s.mean(0)) @ A + Xt_s.mean(0)
    out["CORAL"] = LogisticRegression(**LR).fit(Xs_coral, ys).predict_proba(Xt_s)[:, 1]

    # IW-ERM: density ratio from a source-vs-target domain classifier.
    Xd = np.vstack([Xs_s, Xt_s])
    yd = np.r_[np.zeros(len(Xs_s)), np.ones(len(Xt_s))]
    pd_ = LogisticRegression(**LR).fit(Xd, yd).predict_proba(Xs_s)[:, 1]
    w = np.clip(pd_ / np.clip(1 - pd_, 1e-6, None), 0, 20)
    out["IW-ERM"] = LogisticRegression(**LR).fit(
        Xs_s, ys, sample_weight=w).predict_proba(Xt_s)[:, 1]

    # Latent oracle: same protocol, but reads the invariant factors.
    out["oracle"] = LogisticRegression(**LR).fit(
        us_inv, ys).predict_proba(ut_inv)[:, 1]
    return out


def studentised(d):
    sd = d.std(ddof=1)
    return 0.0 if sd < 1e-12 else float(d.mean() / (sd / np.sqrt(len(d))))


def signflip_maxstat(diffs):
    """Design A: exact enumeration over seeds."""
    names = list(diffs)
    D = np.vstack([diffs[m] for m in names])
    obs = np.array([studentised(d) for d in D])
    k = int(obs.argmax())
    S = D.shape[1]
    null = []
    for eps in itertools.product([1.0, -1.0], repeat=S):
        F = D * np.array(eps)
        sd = F.std(axis=1, ddof=1)
        t = np.where(sd < 1e-12, 0.0, F.mean(axis=1) / (sd / np.sqrt(S)))
        null.append(t.max())
    return names[k], float(D[k].mean()), float((np.array(null) >= obs[k]).mean())


def bootstrap_maxstat(D, names):
    """Design B: centre each method's bootstrap gaps, keep the joint structure."""
    mean, sd = D.mean(1), D.std(1, ddof=1)
    sd = np.where(sd < 1e-12, np.inf, sd)
    t_obs = mean / sd
    k = int(t_obs.argmax())
    null_max = ((D - mean[:, None]) / sd[:, None]).max(0)
    return names[k], float(mean[k]), float((null_max >= t_obs[k]).mean())


def design_A(scenario, n=500, seeds=10):
    diffs = {m: [] for m in ("CORAL", "IW-ERM")}
    for s in range(seeds):
        d = generate_scenario(scenario, n_source_domains=1, n_per_domain=n, seed=s)
        S, T = d["source"][0], d["target"]
        p = fit_methods(S.x, S.y, T.x, S.u_inv, T.u_inv)
        a = {m: roc_auc_score(T.y, v) for m, v in p.items()}
        for m in diffs:
            diffs[m].append(a[m] - a[ERM])
    diffs = {m: np.array(v) for m, v in diffs.items()}
    return signflip_maxstat(diffs), {m: v.mean() for m, v in diffs.items()}


def design_B(scenario, n_src=600, n_tgt=2900, n_boot=2000, seed=0):
    d = generate_scenario(scenario, n_source_domains=1, n_per_domain=max(n_src, n_tgt),
                          seed=seed)
    S, T = d["source"][0], d["target"]
    Xs, ys, us = S.x[:n_src], S.y[:n_src], S.u_inv[:n_src]
    Xt, yt, ut = T.x[:n_tgt], T.y[:n_tgt], T.u_inv[:n_tgt]
    p = fit_methods(Xs, ys, Xt, us, ut)

    rng = np.random.default_rng(seed)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    names = ["CORAL", "IW-ERM"]
    D = np.zeros((len(names), n_boot))
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        yb = yt[idx]
        ref = roc_auc_score(yb, p[ERM][idx])
        for i, m in enumerate(names):
            D[i, b] = roc_auc_score(yb, p[m][idx]) - ref
    return bootstrap_maxstat(D, names), {m: D[i].mean() for i, m in enumerate(names)}


if __name__ == "__main__":
    sc = "covariate_shift"
    print(f"scenario: {sc}\n")

    (bA, gA, pA), meansA = design_A(sc)
    print("A. DISEGNO BENCHMARK   500/500, incertezza sui 10 seed, sign-flip")
    for m, v in meansA.items():
        print(f"     {m:<8s} gap medio {v:+.4f}")
    print(f"     -> best {bA}, gap {gA:+.4f}, p = {pA:.4f}\n")

    (bB, gB, pB), meansB = design_B(sc)
    print("B. DISEGNO REGISTRY    600/2900, incertezza dal bootstrap pazienti")
    for m, v in meansB.items():
        print(f"     {m:<8s} gap medio {v:+.4f}")
    print(f"     -> best {bB}, gap {gB:+.4f}, p = {pB:.4f}")
