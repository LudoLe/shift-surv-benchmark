"""Two controls that must run before any twin result on real data is interpreted.

Recovery (--test recovery)
--------------------------
The retrospective twins failed to reproduce the method-effect vector of the real
registry transfers. That is only interpretable if the twin machinery works when
the answer is known. So: build synthetic worlds whose mechanism we chose, hide
the fact from the twin, and ask whether the variant matching the true mechanism
scores best.

Worlds: an identity control (source == target, where every twin must report
approximately nothing), mean-only, covariance-only, mean-plus-covariance, and a
concept-shift world whose mechanism the twin cannot see at all. Each is treated
exactly as a real transfer would be -- its d_mean and d_cov are
measured from unlabelled covariates, the three twins are calibrated against
those, and the effect vectors are compared.

If the matching variant wins, a failure on real data is a statement about the
real data. If it does not, the twins are exploratory and nothing about the real
transfers follows from them.

Both the fitted knobs and the effect-vector error are reported, because a twin
can land on the right effects from the wrong configuration; only the knobs say
whether the mechanism itself was recovered.

Identifiability (--test identifiability)
----------------------------------------
A twin can only ever use what is observable without target labels. This asks
whether that is enough in principle, by exhibiting two worlds whose target
covariates are identical by construction and whose transfer outcomes are not.

concept_shift changes only the risk vector w. It enters the event times through
eta and therefore changes P(Y|X), but it never touches u_inv, u_spu or the
observation map, so the observed target covariates are bit-identical to those
of no_shift. Any diagnostic computed on unlabelled target data -- MMD, a domain
classifier, the moments the shift audit measures -- returns exactly the same
value in both, while the transfer outcomes differ.

That is a demonstration by construction rather than an empirical observation:
some transport-relevant mechanisms are not identifiable from unlabelled target
covariates even though their consequences for adaptation differ.

Usage
-----
  python -m experiments.twin_recovery_check --test identifiability
  python -m experiments.twin_recovery_check --test recovery --n_seeds 5
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from experiments.baselines import fit_all_observed
from experiments.shift_audit import _standardise, geometry
from experiments.retrospective_twins import (build_twin, calibrate, effect_vector,
                                             score, twin_geometry, COMPARED, VARIANTS)
from utils.io import save_table_csv
from utils.shift_diag import rbf_mmd

_TABLES = _ROOT / "outputs" / "tables"

# The worlds the recovery test builds, and which twin variant should win on each.
WORLDS = {
    "identity":      dict(mu=0.0, cov=1.0, expect="(nessuno: nessun effetto)"),
    "mean_only":     dict(mu=1.2, cov=1.0, expect="mean_only"),
    "cov_only":      dict(mu=0.0, cov=3.0, expect="cov_only"),
    "mean_plus_cov": dict(mu=1.2, cov=3.0, expect="mean_plus_cov"),
    # Not buildable by the twin: concept shift lives entirely in P(Y|X) and
    # leaves the target covariates untouched, so every label-free diagnostic the
    # twin calibrates against reads zero. Included precisely because no variant
    # can succeed here -- it marks the boundary of what the approach can address
    # rather than testing it, and a twin that appeared to succeed would be
    # reading something it cannot have seen.
    "concept":       dict(mu=None, cov=None, expect="(nessuno: non identificabile)"),
}


# ---------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------

def run_identifiability(n: int, seeds: list[int]) -> list[dict]:
    print("Due mondi con covariate target identiche e meccanismo d'esito diverso\n")
    per_world, X_ref = {}, {}
    for name in ("no_shift", "concept_shift"):
        erm, prev, eff = [], [], []
        for s in seeds:
            d = generate_scenario(name, n_source_domains=1, n_per_domain=n,
                                  seed=s, base=BaseParams(seed=s))
            S, T = d["source"][0], d["target"]
            erm.append(fit_all_observed([S], T)["target"]["auroc"])
            prev.append(float(T.y.mean()))
            if s == seeds[0]:
                X_ref[name] = T.x.copy()
        per_world[name] = (float(np.mean(erm)), float(np.mean(prev)))

    XA, XB = X_ref["no_shift"], X_ref["concept_shift"]
    identical = bool(np.allclose(XA, XB))
    sc = StandardScaler().fit(XA)
    mmd = float(rbf_mmd(sc.transform(XA), sc.transform(XB)))
    Xd = np.vstack([sc.transform(XA), sc.transform(XB)])
    yd = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = float(np.mean(cross_val_score(
            LogisticRegression(max_iter=1000), Xd, yd,
            cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")))

    for name in ("no_shift", "concept_shift"):
        a, p = per_world[name]
        print(f"  {name:<16} ERM target AUROC {a:.3f}   prevalenza {p:.3f}")
    gap = per_world["no_shift"][0] - per_world["concept_shift"][0]
    print(f"\n  covariate target bit-identiche : {identical}")
    print(f"  MMD fra le due x target        : {mmd:.6f}")
    print(f"  classificatore di dominio AUROC: {auc:.4f}   (0.5 = indistinguibili)")
    print(f"  divario di AUROC di ERM        : {gap:+.3f}")
    print("\n  Nessuna quantita' calcolabile senza etichette del target distingue")
    print("  i due mondi, ma uno ha segnale recuperabile e l'altro no.")

    return [{
        "test": "identifiability", "n_per_domain": n, "n_seeds": len(seeds),
        "covariates_bit_identical": identical,
        "mmd_between_targets": round(mmd, 8),
        "domain_classifier_auroc": round(auc, 4),
        "erm_auroc_no_shift": round(per_world["no_shift"][0], 4),
        "erm_auroc_concept_shift": round(per_world["concept_shift"][0], 4),
        "erm_auroc_gap": round(gap, 4),
        "prevalence_no_shift": round(per_world["no_shift"][1], 4),
        "prevalence_concept_shift": round(per_world["concept_shift"][1], 4),
    }]


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _concept_world(seed: int, n: int):
    """Ground-truth transfer whose mechanism is invisible to the twin."""
    d = generate_scenario("concept_shift", n_source_domains=1, n_per_domain=n,
                          seed=seed, base=BaseParams(seed=seed))
    return d["source"][0], d["target"]


def _world(spec: dict, seed: int, n_src: int, n_tgt: int, p: int):
    """A synthetic 'ground truth' transfer, built with the twin's own generator so
    that a variant matching the true mechanism could in principle reproduce it
    exactly. Making the world harder than the twin can express would test the
    generator's expressiveness, not the twin's recovery."""
    s = dict(n_src=n_src, n_tgt=n_tgt, n_features=p, prev_src=0.35, prev_tgt=0.35)
    return build_twin(s, spec["mu"], spec["cov"], seed), s


def run_recovery(n_seeds: int, n_src: int, n_tgt: int, p: int) -> list[dict]:
    seeds = list(range(n_seeds))
    rows = []
    print(f"Recupero del meccanismo su mondi noti  (n {n_src}/{n_tgt}, p {p}, "
          f"{n_seeds} seed)\n")
    for wname, wspec in WORLDS.items():
        if wname == "concept":
            S, T = _concept_world(0, max(n_src, n_tgt))
            base_spec = dict(n_src=S.n, n_tgt=T.n, n_features=S.x.shape[1],
                             prev_src=float(S.y.mean()), prev_tgt=float(T.y.mean()))
        else:
            (S, T), base_spec = _world(wspec, 0, n_src, n_tgt, p)
        a, c = _standardise(S.x, T.x)
        g = geometry(a, c)
        truth = effect_vector(S, T, seeds)
        print(f"=== mondo {wname}   d_mean {g['d_mean']:.3f}  d_cov {g['d_cov']:.3f}"
              f"  (atteso vincitore: {wspec['expect']})", flush=True)
        print("  verita'   " + "  ".join(
            f"{m} {truth['auroc'].get(m, float('nan')):+.4f}" for m in COMPARED), flush=True)

        # The twin sees only the label-free diagnostics of this world.
        spec = dict(base_spec, d_mean=g["d_mean"], d_cov=g["d_cov"])
        best, best_err = None, float("inf")
        for variant in VARIANTS:
            mu, cs = calibrate(spec, variant)
            St, Tt = build_twin(spec, mu, cs, 0)
            sc = score(effect_vector(St, Tt, seeds), truth, COMPARED)
            nerr = sc.get("auroc_norm_error", float("nan"))
            if nerr == nerr and nerr < best_err:
                best, best_err = variant, nerr
            # Parameter recovery, reported separately from effect-vector error:
            # a twin can land on the right effects from the wrong configuration,
            # and only the knobs say whether the mechanism itself was recovered.
            got = twin_geometry(spec, mu, cs, seeds=3)
            rows.append(dict(test="recovery", world=wname, variant=variant,
                             expected=wspec["expect"],
                             mu_true=wspec["mu"], cov_true=wspec["cov"],
                             mu_fitted=round(mu, 4), cov_fitted=round(cs, 4),
                             d_mean_world=round(g["d_mean"], 4),
                             d_cov_world=round(g["d_cov"], 4),
                             d_mean_twin=round(got["d_mean"], 4),
                             d_cov_twin=round(got["d_cov"], 4), **sc))
            print(f"  {variant:<14} sign {sc.get('auroc_sign_agreement', float('nan')):.2f}"
                  f"  nerr {nerr:.2f}", flush=True)
        ok = "OK " if best == wspec["expect"] else "NO "
        print(f"  -> vincitore {best}   {ok}(atteso {wspec['expect']})\n", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["identifiability", "recovery", "both"],
                    default="both")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=2000, help="n per dominio (identifiability)")
    ap.add_argument("--n_src", type=int, default=500)
    ap.add_argument("--n_tgt", type=int, default=1500)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--out_prefix", default="twin_checks")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    rows = []
    if args.test in ("identifiability", "both"):
        rows += run_identifiability(args.n, list(range(args.n_seeds)))
        print()
    if args.test in ("recovery", "both"):
        rows += run_recovery(args.n_seeds, args.n_src, args.n_tgt, args.p)

    _TABLES.mkdir(parents=True, exist_ok=True)
    out = _TABLES / f"{args.out_prefix}_{args.test}.csv"
    save_table_csv(rows, out)
    print(f"  Salvato: {out}")


if __name__ == "__main__":
    main()
