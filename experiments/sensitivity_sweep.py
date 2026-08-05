"""Sensitivity of the benchmark's conclusion to the generator's free parameters.

The question
------------
The setup section fixes about a dozen constants. Some are not free: tau=6 is the
endpoint I3LUNG's oncologists use, n=500 is the middle of the real centres' range
(191-668), and rho=0.7 was fixed by the pre-specified pilot in
experiments/pilot_spurious_rho.py before any benchmark run. Others are
calibrated to land on an interpretable quantity: lambda0 gives a no-shift
prevalence of ~0.33, the target lambda0 and the label-prior prevalence are
matched at ~0.6 so the two prevalence-moving regimes are comparable, and
w=(1, a, -a) with a=0.5 makes the concept flip give a cosine of exactly 1/3.

What is left is genuinely conventional: k_inv, k_spu, p, sigma, nu, the size of
the subordinate risk coefficients, and each regime's shift amplitude. This
module asks whether the paper's conclusion survives changing them.

What "exhaustive" can mean here
-------------------------------
A full factorial is not an option and would not be worth it: eight axes at three
or four values each is over five thousand configurations, times six regimes and
ten seeds, which is a day of compute producing a table nobody can read. Two
weaker designs answer two different questions, and this module runs both.

  oat     One axis moved at a time around the benchmark's configuration.
          Answers "does this particular choice drive the result?" per parameter,
          and produces a table a reader can scan. Blind to interactions: it can
          only find a parameter that matters on its own.

  worlds  Every axis drawn at once from its plausible range, repeatedly. Each
          draw is one "world" -- a different benchmark that is equally defensible
          a priori -- and all regimes are evaluated inside it. This is the design
          that can catch an interaction, and it is what supports a statement
          about the parameter space rather than about one axis at a time.

Neither can prove robustness. A sweep can only fail to find a counterexample in
the region searched, so the defensible claim is "across the configurations we
varied, none reversed the conclusion", never "the result is robust". Both modes
report the breaking configurations explicitly so that a failure is something to
investigate rather than something to average away.

What is measured
----------------
Per (configuration, regime), as paired per-seed differences against ERM on the
same generated data:

  adapt_gain  = best adaptation method - ERM   (target AUROC)
                The claim under test: it should stay below the pre-specified
                practical margin of 0.02.

  headroom    = observed-data oracle - ERM     (target AUROC)
                Not a second claim. It is what makes a configuration
                informative: where the headroom is indistinguishable from zero,
                the target covariates support no more than ERM already achieves,
                so a null gain is a tautology rather than evidence. Under
                covariate shift this is the expected state, because a mean
                displacement orthogonal to w translates the covariate cloud
                rigidly, offsets every score by the same constant, and therefore
                leaves the ranking -- and the AUROC -- exactly unchanged.

Verdicts, all pre-specified and keyed on the interval rather than the estimate:

  supported        gain 95% upper limit < margin
  inconclusive     the interval spans the margin -- a power problem, not an effect
  candidate_break  point estimate above the margin, lower limit not yet above it;
                   must be re-run on seeds disjoint from the screening set
  BREAK            gain 95% lower limit > margin
  uninformative    headroom below HEADROOM_MIN, so nothing was recoverable and a
                   null gain is a tautology rather than evidence

How the free parameters are reached
-----------------------------------
k_inv, k_spu and p are BaseParams fields. sigma and the shift amplitudes are
scenario keyword arguments. lambda0, nu and w live in the ``domain_defaults``
block of the config, which has no keyword argument -- so those are varied
through the documented ``SHIFT_SURV_CONFIG`` mechanism, by writing a derived
YAML per configuration. Nothing in the generator is modified.

Usage
-----
  python -m experiments.sensitivity_sweep                      # oat, all regimes
  python -m experiments.sensitivity_sweep --mode worlds --n_worlds 100
  python -m experiments.sensitivity_sweep --n_source_domains 3 # adds IRM/GroupDRO
  python -m experiments.sensitivity_sweep --with_nets          # adds DANN/CDAN (slow)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy import stats as sp_stats

from generator.parameters import BaseParams
from generator.scenarios import generate_scenario
from utils.io import save_table_csv

_TABLES = _ROOT / "outputs" / "tables"
_CONFIG = _ROOT / "configs" / "generator.yaml"

# Pre-specified practical margin (configs/generator.yaml, experiment block).
MARGIN = 0.02
# Pre-specified in configs/parameter_space.yaml: below this a configuration
# cannot test the claim, because ERM already sits at what x supports.
HEADROOM_MIN = 0.02

REGIMES = ["covariate_shift", "observational_shift", "spurious_shift",
           "baseline_hazard_shift", "prior_shift", "concept_shift"]

# ---------------------------------------------------------------------------
# The free parameters
# ---------------------------------------------------------------------------
# Each entry: grid for the one-at-a-time mode, the benchmark's value, and the
# range the random mode draws from.
#
# Why w is split into a norm and a shape
# --------------------------------------
# A single "size of the subordinate coefficients" knob, w = (1, a, -a), was the
# first attempt and it is too weak to be worth reporting: over a in [0.2, 1] the
# norm ||w|| only moves between 1.08 and 1.73, so it varies the *shape* of w
# while holding its *strength* almost fixed. Strength is the part that matters
# for the outcome, since eta = w . u_inv has variance ||w||^2 and therefore sets
# how discriminable the label is at all -- the achievable AUROC. The family also
# excludes every asymmetric or same-sign w by construction.
#
# So the two are separated. ``w_norm`` scales ||w|| across a real range, moving
# the ceiling AUROC; ``w_shape`` changes the direction on the unit sphere at
# fixed norm, so shape is isolated from strength. The benchmark's
# w = (1, 0.5, -0.5) is the shape "benchmark" at norm sqrt(1.5) = 1.2247.
#
# In the worlds mode neither grid is used: the direction is drawn uniformly on
# the unit sphere in k_inv dimensions and the norm log-uniformly, so no
# parametric family restricts it at all. That is the honest answer to "why only
# a handful of values for w".
#
# Note the coupling: w also fixes the concept-shift cosine, since flipping the
# coefficients in J gives cos = 1 - 2*sum_{j in J} w_j^2 / ||w||^2. Varying the
# shape therefore varies the concept severity as well. That is a property of the
# benchmark, not an artefact of the sweep, and the resulting cosine is reported
# as a column so the two can be read together.
W_BENCH = np.array([1.0, 0.5, -0.5])
W_NORM_BENCH = float(np.linalg.norm(W_BENCH))          # sqrt(1.5) = 1.2247

W_SHAPES = {
    "benchmark":    [1.0, 0.5, -0.5],
    "single":       [1.0, 0.0, 0.0],     # one factor carries all the prognosis
    "equal_signed": [1.0, 1.0, -1.0],    # equal magnitude, mixed sign
    "equal_same":   [1.0, 1.0, 1.0],     # equal magnitude, one sign
    "asymmetric":   [1.0, 0.8, -0.2],    # no symmetry between the subordinates
}

AXES = {
    "k_inv":  dict(grid=[2, 3, 4, 5],            default=3,    draw=("int", 2, 5)),
    "k_spu":  dict(grid=[1, 2, 3, 4],            default=2,    draw=("int", 1, 4)),
    "p":      dict(grid=[10, 20, 40, 80],        default=20,   draw=("logint", 10, 80)),
    "sigma":  dict(grid=[0.25, 0.5, 1.0, 2.0],   default=0.5,  draw=("log", 0.2, 2.0)),
    "lambda0": dict(grid=[0.01, 0.02, 0.035, 0.05], default=0.02, draw=("log", 0.01, 0.05)),
    "nu":     dict(grid=[0.7, 1.0, 1.5, 2.5],    default=1.5,  draw=("lin", 0.7, 2.5)),
    "w_norm": dict(grid=[0.5 * W_NORM_BENCH, W_NORM_BENCH,
                         2.0 * W_NORM_BENCH, 3.0 * W_NORM_BENCH],
                   default=W_NORM_BENCH,         draw=("log", 0.4, 4.0)),
    "w_shape": dict(grid=list(W_SHAPES),         default="benchmark",
                    draw=("sphere", None, None)),
}


def _w_vector(shape, norm: float, k_inv: int) -> list[float]:
    """Build w of length k_inv with the requested shape and norm.

    ``shape`` is either a name from W_SHAPES or an explicit vector (the worlds
    mode passes a point drawn on the sphere). It is truncated or zero-padded to
    k_inv and then rescaled, so the norm is exactly ``norm`` whatever k_inv is --
    the generator's own truncation would otherwise change the norm silently when
    k_inv < 3.
    """
    v = np.asarray(W_SHAPES[shape] if isinstance(shape, str) else shape, dtype=float)
    if len(v) < k_inv:
        v = np.pad(v, (0, k_inv - len(v)))
    v = v[:k_inv]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:                      # truncation emptied it; fall back to e_0
        v = np.zeros(k_inv); v[0] = 1.0; nv = 1.0
    return [float(x) for x in v / nv * norm]

# Each regime's own severity knob, with the kwarg it maps to.
AMPLITUDE = {
    "covariate_shift":      dict(kwarg="mu_inv_shift_scale",
                                 grid=[0.5, 1.0, 1.5, 2.5], default=1.5,
                                 draw=("lin", 0.5, 3.0)),
    "observational_shift":  dict(kwarg="a_inv_rotation_angle_deg",
                                 grid=[30, 45, 60, 75], default=60,
                                 draw=("lin", 20, 85)),
    "spurious_shift":       dict(kwarg="rho_source",
                                 grid=[0.5, 0.7, 0.9], default=0.7,
                                 draw=("lin", 0.4, 0.95)),
    "baseline_hazard_shift": dict(kwarg="lambda0_target",
                                 grid=[0.03, 0.06, 0.12], default=0.06,
                                 draw=("log", 0.03, 0.15)),
    "prior_shift":          dict(kwarg="prevalence_target",
                                 grid=[0.45, 0.60, 0.75], default=0.60,
                                 draw=("lin", 0.40, 0.80)),
    "concept_shift":        dict(kwarg="w_flip_indices",
                                 grid=[(2,), (1, 2), (0,), (0, 1, 2)], default=(1, 2),
                                 draw=("choice", None, None)),
}


def _draw(spec, rng):
    kind, lo, hi = spec
    if kind == "int":
        return int(rng.integers(lo, hi + 1))
    if kind == "logint":
        return int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "lin":
        return float(rng.uniform(lo, hi))
    if kind == "choice":
        return [(2,), (1, 2), (0,), (0, 1, 2)][int(rng.integers(0, 4))]
    if kind == "sphere":
        # Direction drawn uniformly on the unit sphere: a standard normal vector
        # normalised. Length 5 covers the largest k_inv in the grid; _w_vector
        # truncates it and rescales, so the direction stays uniform on the
        # sphere of whatever dimension the world actually uses.
        v = rng.standard_normal(5)
        return list(v / np.linalg.norm(v))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Reaching lambda0 / nu / w through the documented config mechanism
# ---------------------------------------------------------------------------

_CFG_DIR = Path(tempfile.mkdtemp(prefix="shift_surv_sweep_"))


def _config_for(lambda0: float, nu: float, w: list[float]) -> str | None:
    """Write (once per distinct configuration) a config with these domain defaults.

    Returns the path, or None when everything is at the benchmark's own values,
    in which case the repo config is used unchanged and the sweep's default row
    is bit-identical to the benchmark.
    """
    is_bench = (lambda0 == AXES["lambda0"]["default"]
                and nu == AXES["nu"]["default"]
                and len(w) == len(W_BENCH)
                and np.allclose(w, W_BENCH))
    if is_bench:
        return None
    import yaml
    tag = f"{lambda0:.6g}_{nu:.6g}_" + "_".join(f"{x:.4f}" for x in w)
    path = _CFG_DIR / f"cfg_{tag}.yaml"
    if not path.exists():
        cfg = yaml.safe_load(_CONFIG.read_text())
        cfg["domain_defaults"].update(
            {"lambda0": float(lambda0), "nu": float(nu),
             "w": [float(x) for x in w],
             # v must match k_spu; the generator pads/truncates, but writing it
             # explicitly keeps the emitted config self-consistent and readable.
             "v": [0.0] * len(cfg["domain_defaults"].get("v", [0.0, 0.0]))})
        path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _concept_cosine(w: list[float], flip) -> float:
    """Source-target cosine of w under a concept flip of the indices in ``flip``."""
    w = np.asarray(w, dtype=float)
    j = [i for i in flip if i < len(w)]
    return float(1.0 - 2.0 * float(np.sum(w[j] ** 2)) / float(np.dot(w, w)))


def _kwargs_for(regime: str, cfgv: dict) -> tuple[dict, dict]:
    """Split one configuration into BaseParams fields and scenario kwargs."""
    base_kw = {k: int(cfgv[k]) for k in ("k_inv", "k_spu", "p")}
    scen_kw: dict = {"source_sigma_base": float(cfgv["sigma"])}

    # Regimes that declare their own lambda0/nu ignore domain_defaults for those,
    # so the corresponding axis has to be pushed into their kwargs too, or the
    # sweep would silently not move anything for that regime.
    if regime == "baseline_hazard_shift":
        scen_kw["source_lambda0_base"] = float(cfgv["lambda0"])
        scen_kw["source_nu"] = float(cfgv["nu"])
        scen_kw["nu_target"] = float(cfgv["nu"])
    # spurious_shift and prior_shift take no sigma kwarg.
    if regime in ("spurious_shift", "prior_shift", "baseline_hazard_shift", "concept_shift"):
        scen_kw.pop("source_sigma_base", None)

    amp = cfgv.get("amplitude")
    if amp is not None:
        spec = AMPLITUDE[regime]
        key = spec["kwarg"]
        if key.endswith("_deg"):
            scen_kw["a_inv_rotation_angle"] = float(np.radians(amp))
        elif key == "w_flip_indices":
            scen_kw[key] = tuple(amp)
        else:
            scen_kw[key] = amp
            if regime == "covariate_shift":
                scen_kw["mu_spu_shift_scale"] = amp
    return base_kw, scen_kw


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def _evaluate(regime: str, cfgv: dict, seeds: list[int], n: int,
              n_src: int, with_nets: bool) -> dict:
    from experiments.baselines import (
        fit_all_observed, fit_invariant_oracle, fit_observed_target_oracle)
    from experiments.adaptation_methods import fit_coral, fit_iw_erm

    base_kw, scen_kw = _kwargs_for(regime, cfgv)
    w_vec = _w_vector(cfgv["w_shape"], float(cfgv["w_norm"]), int(cfgv["k_inv"]))
    cfg_path = _config_for(cfgv["lambda0"], cfgv["nu"], w_vec)

    family = ["coral", "iw_erm"]
    if n_src >= 2:
        family += ["irm", "group_dro"]
    if with_nets:
        family += ["dann", "cdan"]
    acc: dict[str, list[float]] = {k: [] for k in
                                   ["erm", "latent_oracle", "obs_oracle"] + family}

    prev_env = os.environ.get("SHIFT_SURV_CONFIG")
    try:
        if cfg_path:
            os.environ["SHIFT_SURV_CONFIG"] = cfg_path
        else:
            os.environ.pop("SHIFT_SURV_CONFIG", None)
        for s in seeds:
            data = generate_scenario(
                regime, n_source_domains=n_src, n_per_domain=n, seed=s,
                base=BaseParams(seed=s, **base_kw), **scen_kw)
            src, tgt = data["source"], data["target"]
            if len(np.unique(tgt.y)) < 2:
                continue
            acc["erm"].append(fit_all_observed(src, tgt)["target"]["auroc"])
            acc["latent_oracle"].append(fit_invariant_oracle(src, tgt)["target"]["auroc"])
            acc["obs_oracle"].append(
                fit_observed_target_oracle(src, tgt, seed=s)["target"]["auroc"])
            acc["coral"].append(fit_coral(src, tgt, seed=s)["target"]["auroc"])
            acc["iw_erm"].append(fit_iw_erm(src, tgt, seed=s)["target"]["auroc"])
            if n_src >= 2:
                from experiments.adaptation_methods import fit_irm
                from experiments.group_dro import fit_group_dro
                acc["irm"].append(fit_irm(src, tgt, seed=s)["target"]["auroc"])
                acc["group_dro"].append(fit_group_dro(src, tgt, seed=s)["target"]["auroc"])
            if with_nets:
                from experiments.adaptation_methods import fit_dann, fit_cdan
                acc["dann"].append(fit_dann(src, tgt, seed=s, use_iwcv=False)["target"]["auroc"])
                acc["cdan"].append(fit_cdan(src, tgt, seed=s, use_iwcv=False)["target"]["auroc"])
    finally:
        if prev_env is None:
            os.environ.pop("SHIFT_SURV_CONFIG", None)
        else:
            os.environ["SHIFT_SURV_CONFIG"] = prev_env

    A = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    if len(A["erm"]) < 2:
        return {"regime": regime, "verdict": "degenerate", **_flat(cfgv)}

    means = {m: float(np.nanmean(A[m])) for m in family}
    best = max(means, key=means.get)
    gain, headroom = A[best] - A["erm"], A["obs_oracle"] - A["erm"]

    def _mci(v):
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return float("nan"), float("nan")
        m = float(v.mean())
        if len(v) < 2:
            return m, 0.0
        return m, float(sp_stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))

    g_m, g_ci = _mci(gain)
    h_m, h_ci = _mci(headroom)
    return {
        "regime": regime, **_flat(cfgv),
        "n_seeds": len(A["erm"]), "family": "+".join(family),
        "erm_auroc": round(float(np.nanmean(A["erm"])), 4),
        "latent_oracle_auroc": round(float(np.nanmean(A["latent_oracle"])), 4),
        "obs_oracle_auroc": round(float(np.nanmean(A["obs_oracle"])), 4),
        "best_adapt": best, "best_adapt_auroc": round(means[best], 4),
        "adapt_gain": round(g_m, 4), "adapt_gain_ci95": round(g_ci, 4),
        "adapt_gain_lo": round(g_m - g_ci, 4),
        "adapt_gain_hi": round(g_m + g_ci, 4),
        "headroom": round(h_m, 4), "headroom_lo": round(h_m - h_ci, 4),
        "verdict": _verdict(g_m, g_ci, h_m),
    }


def _verdict(gain: float, gain_ci: float, headroom: float) -> str:
    """Five pre-specified states, keyed on the interval rather than the estimate.

    Declaring a break because the point estimate exceeds the margin is the same
    error as declaring significance from a point estimate. At ten seeds the gain
    carries a standard error of roughly 0.005, comparable to the effects being
    measured, so isolated configurations will cross 0.02 by chance alone. A break
    therefore has to be an interval lying above the margin; an estimate that
    merely crosses it is a *candidate*, and is re-run on seeds disjoint from the
    screening set before it is called anything at all.

    ``uninformative`` keys on a pre-specified threshold on the headroom point
    estimate rather than on a test, so the classification does not depend on how
    precisely the headroom happened to be measured in that configuration.
    """
    lo, hi = gain - gain_ci, gain + gain_ci
    if headroom < HEADROOM_MIN:
        return "uninformative"
    if lo > MARGIN:
        return "BREAK"
    if gain > MARGIN:
        return "candidate_break"
    if hi >= MARGIN:
        return "inconclusive"
    return "supported"


def _flat(cfgv: dict) -> dict:
    out = {k: (cfgv[k] if k != "w_shape" else
               (cfgv[k] if isinstance(cfgv[k], str) else "sphere")) for k in AXES}
    out["amplitude"] = str(cfgv.get("amplitude"))
    w = _w_vector(cfgv["w_shape"], float(cfgv["w_norm"]), int(cfgv["k_inv"]))
    out["w"] = "[" + ",".join(f"{x:.3f}" for x in w) + "]"
    amp = cfgv.get("amplitude")
    out["concept_cos"] = (round(_concept_cosine(w, amp), 4)
                          if isinstance(amp, (tuple, list)) else "")
    return out


def _defaults(regime: str) -> dict:
    d = {k: v["default"] for k, v in AXES.items()}
    d["amplitude"] = AMPLITUDE[regime]["default"]
    return d


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def oat(regimes, seeds, n, n_src, with_nets) -> list[dict]:
    rows, t0 = [], time.time()
    jobs = []
    for r in regimes:
        for ax in list(AXES) + ["amplitude"]:
            grid = AXES[ax]["grid"] if ax in AXES else AMPLITUDE[r]["grid"]
            for v in grid:
                jobs.append((r, ax, v))
    print(f"  one-at-a-time: {len(jobs)} configurazioni x {len(seeds)} seed\n")
    for i, (r, ax, v) in enumerate(jobs, 1):
        cfgv = _defaults(r)
        cfgv[ax] = v
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = _evaluate(r, cfgv, seeds, n, n_src, with_nets)
        row.update(mode="oat", axis=ax, value=str(v),
                   is_default=(v == (_defaults(r)[ax])))
        rows.append(row)
        print(f"  [{i:>3}/{len(jobs)}] {r:<22}{ax:<10}={str(v):<10}"
              f"gain={row.get('adapt_gain', float('nan')):+.4f}"
              f"  headroom={row.get('headroom', float('nan')):+.4f}"
              f"  {row['verdict']:<14}({time.time()-t0:.0f}s)", flush=True)
    return rows


def worlds(regimes, seeds, n, n_src, with_nets, n_worlds, seed0) -> list[dict]:
    rng = np.random.default_rng(seed0)
    rows, t0 = [], time.time()
    print(f"  worlds: {n_worlds} mondi x {len(regimes)} regimi x {len(seeds)} seed\n")
    for wi in range(n_worlds):
        cfgv = {k: _draw(v["draw"], rng) for k, v in AXES.items()}
        for r in regimes:
            c = dict(cfgv)
            c["amplitude"] = _draw(AMPLITUDE[r]["draw"], rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = _evaluate(r, c, seeds, n, n_src, with_nets)
            row.update(mode="worlds", world=wi, axis="", value="", is_default=False)
            rows.append(row)
        done = [x for x in rows if x.get("world") == wi]
        worst = max((x for x in done if "adapt_gain" in x),
                    key=lambda x: x["adapt_gain"], default=None)
        print(f"  [{wi+1:>3}/{n_worlds}] k=({cfgv['k_inv']},{cfgv['k_spu']}) p={cfgv['p']:<3}"
              f" sig={cfgv['sigma']:.2f} lam={cfgv['lambda0']:.3f} nu={cfgv['nu']:.2f}"
              f" |w|={cfgv['w_norm']:.2f}   peggior gain="
              f"{(worst['adapt_gain'] if worst else float('nan')):+.4f}"
              f" ({worst['regime'] if worst else '-'})   ({time.time()-t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(rows: list[dict]) -> None:
    good = [r for r in rows if r["verdict"] != "degenerate"]
    STATES = ("supported", "inconclusive", "candidate_break", "BREAK", "uninformative")
    n = {k: sum(r["verdict"] == k for r in good) for k in STATES}
    print(f"\n{'='*96}\n  ESITO SU {len(good)} CONFIGURAZIONI VALUTATE"
          f"  ({len(rows)-len(good)} degeneri, scartate)\n{'='*96}")
    print("  " + "   ".join(f"{k} {n[k]}" for k in STATES))

    print(f"\n  Per regime:")
    print(f"  {'regime':<24}{'n':>4}{'sup':>5}{'inc':>5}{'cand':>6}{'BRK':>5}{'n.i.':>6}"
          f"{'gain max':>11}{'hi max':>10}{'headroom med':>14}")
    for r in sorted({x['regime'] for x in good}):
        sub = [x for x in good if x['regime'] == r]
        gm = max(x['adapt_gain'] for x in sub)
        hm = max(x['adapt_gain_hi'] for x in sub)
        hd = float(np.median([x['headroom'] for x in sub]))
        print(f"  {r:<24}{len(sub):>4}"
              f"{sum(x['verdict']=='supported' for x in sub):>5}"
              f"{sum(x['verdict']=='inconclusive' for x in sub):>5}"
              f"{sum(x['verdict']=='candidate_break' for x in sub):>6}"
              f"{sum(x['verdict']=='BREAK' for x in sub):>5}"
              f"{sum(x['verdict']=='uninformative' for x in sub):>6}"
              f"{gm:>+11.4f}{hm:>+10.4f}{hd:>+14.4f}")

    brk = [r for r in good if r["verdict"] in ("BREAK", "candidate_break")]
    if brk:
        print(f"\n  {len(brk)} DA RICONFERMARE SU SEED DISGIUNTI (BREAK o candidate_break):")
        for r in sorted(brk, key=lambda x: -x["adapt_gain"])[:20]:
            cfg = " ".join(f"{k}={r[k]}" for k in AXES)
            print(f"    {r['regime']:<22}gain={r['adapt_gain']:+.4f} ({r['best_adapt']})"
                  f"  amp={r['amplitude']}  {cfg}")
    else:
        print("\n  Nessuna configurazione in cui la stima puntuale supera il margine.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sensitivity of the conclusion to the generator's free parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", default="oat", choices=["oat", "worlds"])
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n_source_domains", type=int, default=1)
    ap.add_argument("--n_worlds", type=int, default=100)
    ap.add_argument("--world_seed", type=int, default=12345)
    ap.add_argument("--with_nets", action="store_true")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    seeds = list(range(args.n_seeds))
    t0 = time.time()
    if args.mode == "oat":
        rows = oat(args.regimes, seeds, args.n, args.n_source_domains, args.with_nets)
    else:
        rows = worlds(args.regimes, seeds, args.n, args.n_source_domains,
                      args.with_nets, args.n_worlds, args.world_seed)
    report(rows)

    prefix = args.out_prefix or f"sensitivity_{args.mode}_{args.n_source_domains}src"
    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / f"{prefix}.csv")
    print(f"\n  {time.time()-t0:.0f}s   Saved: {_TABLES}/{prefix}.csv")
