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
import yaml
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


# ---------------------------------------------------------------------------
# The frozen space, loaded rather than hardcoded
# ---------------------------------------------------------------------------
# Grids are no longer written here. They come from configs/parameter_space.yaml,
# which declares each continuous parameter by the quantity it controls, and from
# configs/parameter_space_calibrated.yaml, which holds the raw value that
# bisection found for each declared target on the fixed super-population. Any
# grid written in this file would be a second, undeclared design decision, and
# the whole point of freezing the space is that there is only one.
#
# The worlds mode samples inside the frozen space rather than from a separate
# range: each axis is drawn between the extremes of its own calibrated grid.

_SPACE_PATH = _ROOT / "configs" / "parameter_space.yaml"
_CALIB_PATH = _ROOT / "configs" / "parameter_space_calibrated.yaml"


# The generator's own values: what the paper actually ran.
_BENCH_RAW = {}


def _bench_raw() -> dict:
    g = yaml.safe_load((_ROOT / "configs" / "generator.yaml").read_text())["domain_defaults"]
    return {"lambda0": float(g["lambda0"]),
            "sigma": float(g["sigma_noise"]),
            "w_norm": float(np.linalg.norm(np.asarray(g["w"], dtype=float)))}


def _require_default(name: str, levels: list[dict]):
    """The level marked as the benchmark's own, or a hard failure.

    The one-at-a-time design perturbs around a centre, and that centre has to be
    the configuration the paper actually ran. Guessing it (the middle of the
    list, say) produces a table whose "default" row is some other benchmark.
    """
    hit = [lv for lv in levels if "default" in (lv.get("note") or "")]
    if len(hit) != 1:
        raise SystemExit(
            f"parameter_space.yaml: '{name}' ha {len(hit)} livelli marcati "
            "'benchmark default', ne serve esattamente uno")
    return hit[0].get("target", hit[0].get("value"))


def _load_space() -> tuple[dict, dict, dict]:
    global _BENCH_RAW
    _BENCH_RAW = _bench_raw()
    """Return (AXES, AMPLITUDE, meta) built from the frozen + calibrated files."""
    if not _CALIB_PATH.exists():
        raise SystemExit(
            f"manca {_CALIB_PATH.name}.\n"
            "Lancia prima:  python -m experiments.calibrate")
    space = yaml.safe_load(_SPACE_PATH.read_text())
    calib = yaml.safe_load(_CALIB_PATH.read_text())["calibrated"]

    axes: dict = {}

    # Calibrated axes: the grid is the set of raw values bisection produced, and
    # each carries the target it was calibrated to, so a row can be reported in
    # semantic terms rather than as a raw number.
    for name, rows in calib.items():
        good = [r for r in rows if r.get("bracketed", True)
                and not np.isnan(r.get("value", float("nan")))]
        if not good:
            raise SystemExit(f"nessun livello calibrato per {name}: ricalibra")
        grid = [float(r["value"]) for r in good]
        # Swap the calibrated default for the benchmark's raw value, so the grid
        # and the centre agree and `is_default` matches on identity.
        _raw = _bench_raw().get(name)
        if _raw is not None:
            _i = min(range(len(good)),
                     key=lambda k: abs(good[k]["target"] - _require_default(name, good)))
            grid[_i] = _raw
        axes[name] = dict(
            grid=grid,
            # The centre of the one-at-a-time design is the benchmark's own raw
            # value, not the calibrated one. They differ slightly -- calibration
            # lands sigma at 0.468 for the same observed-oracle target the
            # benchmark's 0.5 achieves -- and a centre that is merely near the
            # benchmark answers a question nobody asked. Off-centre levels stay
            # calibrated, so a row differs from the benchmark in exactly one
            # semantic quantity; the benchmark's own achieved semantics are
            # measured and reported rather than assumed.
            default=float(_BENCH_RAW[name]),
            draw=("log", min(grid), max(grid)),
            semantic={float(r["value"]): (r["target"], r["achieved"], r["tag"]) for r in good},
            controls=space["calibrated_parameters"][name]["controls"],
        )

    # Direct axes: values used as declared, no calibration needed.
    for name, spec in space["direct_parameters"].items():
        vals = [lv["value"] for lv in spec["levels"]]
        dflt = next((lv["value"] for lv in spec["levels"]
                     if "default" in (lv.get("note") or "")), None)
        if dflt is None:
            # Falling back to the middle of the list would silently centre the
            # one-at-a-time design on a configuration that is not the benchmark,
            # so every "default" row would describe a different paper.
            raise SystemExit(
                f"parameter_space.yaml: nessun livello di '{name}' e' marcato "
                "'benchmark default' nella nota; il centro dell'OAT sarebbe arbitrario")
        # The sampling range defaults to the grid's extremes, but an axis may
        # declare its own: a level can be worth one one-at-a-time cell and not
        # worth paying for in every sampled world. n_per_domain is the case --
        # its 10000 level is the only one reaching the registry's p/n, and a
        # log-uniform draw up to there would cost five times as much per world.
        _dr = spec.get("draw_range")
        _lo, _hi = (_dr if _dr else (min(vals), max(vals)))
        if name == "w_shape":
            draw = ("sphere", None, None)
        elif all(isinstance(v, int) for v in vals):
            draw = ("logint", _lo, _hi) if _hi / max(_lo, 1) >= 4 else ("int", _lo, _hi)
        else:
            draw = ("lin", _lo, _hi)
        axes[name] = dict(grid=vals, default=dflt, draw=draw,
                          semantic={v: (None, None, lv["tag"])
                                    for v, lv in zip(vals, spec["levels"])},
                          controls=spec["controls"])

    amplitude: dict = {}
    for regime, spec in space["shift_amplitude"].items():
        vals = [lv["value"] for lv in spec["levels"]]
        dflt = next((lv["value"] for lv in spec["levels"]
                     if "default" in (lv.get("note") or "")), vals[len(vals) // 2])
        draw = ("lin", min(vals), max(vals))
        amplitude[regime] = dict(kwarg=spec["kwarg"], grid=vals, default=dflt, draw=draw,
                                 semantic={v: (None, None, lv["tag"])
                                           for v, lv in zip(vals, spec["levels"])})
    return axes, amplitude, space


AXES, AMPLITUDE, SPACE = _load_space()
MARGIN = float(SPACE["meta"]["margin_auroc"])
HEADROOM_MIN = float(SPACE["meta"]["headroom_threshold"])
PROTOCOL = SPACE["protocol"]
MAX_K_OVER_P = float(SPACE.get("constraints", {}).get("max_k_over_p", 0.5))
# Concept severity is a cosine now, so it is continuous and the worlds mode
# draws it on a range instead of picking from a list. The indices realising it
# are solved per configuration.
_CONCEPT_COSINES = [float(lv["value"]) for lv
                    in SPACE["shift_amplitude"]["concept_shift"]["levels"]]


def _w_vector(shape, norm: float, k_inv: int) -> list[float]:
    """Build w of length k_inv with the requested shape and norm.

    ``shape`` is either a name from W_SHAPES or an explicit vector (the worlds
    mode passes a point drawn on the sphere). It is truncated or zero-padded to
    k_inv and then rescaled, so the norm is exactly ``norm`` whatever k_inv is --
    the generator's own truncation would otherwise change the norm silently when
    k_inv < 3.
    """
    if isinstance(shape, str):
        if shape not in W_SHAPES:
            raise KeyError(
                f"w_shape {shape!r} non e' una forma dichiarata. In modalita' worlds "
                "la direzione e' un punto sulla sfera e va ricostruita dalla colonna "
                "'w' del CSV, non dal nome: vedi confirm().")
        shape = W_SHAPES[shape]
    v = np.asarray(shape, dtype=float)
    if len(v) < k_inv:
        v = np.pad(v, (0, k_inv - len(v)))
    v = v[:k_inv]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:                      # truncation emptied it; fall back to e_0
        v = np.zeros(k_inv); v[0] = 1.0; nv = 1.0
    return [float(x) for x in v / nv * norm]


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
        return float(rng.uniform(min(_CONCEPT_COSINES), max(_CONCEPT_COSINES)))
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


def _config_for(lambda0: float, nu: float, w: list[float],
                sigma: float) -> str | None:
    """Write (once per distinct configuration) a config with these domain defaults.

    Returns the path, or None when everything is at the benchmark's own values,
    in which case the repo config is used unchanged and the sweep's default row
    is bit-identical to the benchmark.
    """
    is_bench = (lambda0 == AXES["lambda0"]["default"]
                and nu == AXES["nu"]["default"]
                and sigma == AXES["sigma"]["default"]
                and len(w) == len(W_BENCH)
                and np.allclose(w, W_BENCH))
    if is_bench:
        return None
    import yaml
    tag = f"{lambda0:.6g}_{nu:.6g}_{sigma:.6g}_" + "_".join(f"{x:.4f}" for x in w)
    path = _CFG_DIR / f"cfg_{tag}.yaml"
    if not path.exists():
        cfg = yaml.safe_load(_CONFIG.read_text())
        cfg["domain_defaults"].update(
            {"lambda0": float(lambda0), "nu": float(nu),
             "w": [float(x) for x in w],
             # sigma has to travel through domain_defaults, not only through the
             # source_sigma_base keyword: four of the six scenario functions do
             # not take that keyword, so the axis was inert for them and their
             # four sigma rows came out identical -- reading as "sigma does not
             # matter" when sigma had never moved.
             "sigma_noise": float(sigma),
             # v must match k_spu; the generator pads/truncates, but writing it
             # explicitly keeps the emitted config self-consistent and readable.
             "v": [0.0] * len(cfg["domain_defaults"].get("v", [0.0, 0.0]))})
        path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _flip_for_cosine(w: list[float], target_cos: float) -> tuple:
    """The index subset whose sign flip lands closest to ``target_cos``.

    Flipping the coefficients in J gives cos = 1 - 2*sum_{j in J} w_j^2/||w||^2,
    so a fixed index list is a fixed severity only for a fixed w. It is not: at
    k_inv=2 the benchmark shape becomes (1.095, 0.548) and flipping (1,2) yields
    0.60 rather than 1/3. Declaring the cosine and solving for the indices makes
    the level mean the same thing in every configuration, which is what a
    severity level has to do.

    k_inv is at most 5 here, so all 2^k subsets are enumerated exactly. The empty
    subset is excluded: it is not a concept shift.
    """
    import itertools
    w = np.asarray(w, dtype=float)
    n2 = float(np.dot(w, w))
    best, best_err = (0,), float("inf")
    for r in range(1, len(w) + 1):
        for J in itertools.combinations(range(len(w)), r):
            c = 1.0 - 2.0 * float(np.sum(w[list(J)] ** 2)) / n2
            if abs(c - target_cos) < best_err:
                best, best_err = J, abs(c - target_cos)
    return best


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
    # Four scenario functions take no sigma keyword; for them sigma arrives
    # through domain_defaults instead (see _config_for).
    if regime in ("spurious_shift", "prior_shift", "baseline_hazard_shift", "concept_shift"):
        scen_kw.pop("source_sigma_base", None)

    amp = cfgv.get("amplitude")
    if amp is not None:
        spec = AMPLITUDE[regime]
        key = spec["kwarg"]
        if key.endswith("_deg"):
            scen_kw["a_inv_rotation_angle"] = float(np.radians(amp))
        elif key == "w_flip_indices":
            # The level is a target cosine; the indices that realise it depend on
            # the w of this particular configuration.
            w_here = _w_vector(cfgv["w_shape"], float(cfgv["w_norm"]), int(cfgv["k_inv"]))
            scen_kw[key] = _flip_for_cosine(w_here, float(amp))
        elif key == "lambda0_multiplier":
            # Relative to this configuration's own source rate, so "three times
            # the source" means the same thing whatever nu, ||w|| and tau are.
            scen_kw["source_lambda0_base"] = float(cfgv["lambda0"])
            scen_kw["lambda0_target"] = float(cfgv["lambda0"]) * float(amp)
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
    # n is an axis now, so the configuration wins over the runner default; the
    # --n flag stays as the centre of that axis.
    n = int(cfgv.get("n_per_domain", n))
    from experiments.baselines import (
        fit_all_observed, fit_invariant_oracle, fit_observed_target_oracle,
        fit_prior_recalib, fit_platt_scaling)
    from experiments.adaptation_methods import fit_coral, fit_iw_erm

    cfgv = dict(cfgv, _is_concept=(regime == "concept_shift"))
    base_kw, scen_kw = _kwargs_for(regime, cfgv)
    w_vec = _w_vector(cfgv["w_shape"], float(cfgv["w_norm"]), int(cfgv["k_inv"]))
    cfg_path = _config_for(cfgv["lambda0"], cfgv["nu"], w_vec,
                           float(cfgv["sigma"]))

    family = ["coral", "iw_erm"]
    if n_src >= 2:
        family += ["irm", "group_dro"]
    if with_nets:
        family += ["dann", "cdan"]
    acc: dict[str, list[float]] = {k: [] for k in
                                   ["erm", "latent_oracle", "obs_oracle"] + family}
    # Calibration is tracked for ERM and the two post-hoc recalibrations only.
    # The pre-specified directional prediction concerns prior adjustment, which
    # is justified under label shift and is expected to degrade ECE elsewhere;
    # Platt scaling assumes nothing about the shift and is carried as its
    # mechanism-agnostic comparator, with no directional verdict attached.
    #
    # Both are rescalings of the ERM predictions and never touch the adversarial
    # fits, so their delta is identical whether or not --no_nets is used. That is
    # what makes it legitimate to run the calibration sweep without the nets and
    # read the discrimination verdicts from the run that had them.
    acc_ece: dict[str, list[float]] = {k: [] for k in
                                       ["erm", "prior_recalib", "platt_scaling"]}
    # Achieved, not requested: in the worlds mode the raw lambda0 was drawn, not
    # calibrated to a prevalence, so the realised minority fraction is the only
    # honest description of what the configuration actually was.
    minority: list[float] = []
    # Seeds dropped by the guards or by a failing fit. Reported per row so a
    # configuration whose verdict rests on three seeds instead of ten is visible
    # in the output rather than hidden behind n_seeds.
    n_skipped = 0
    skip_reason = ""

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
                n_skipped += 1
                skip_reason = skip_reason or "target single-class"
                continue
            # The source must also be non-degenerate. At extreme lambda0 the
            # pooled source can come out single-class, and every fit_* helper
            # then returns {"feature_set", "error"} with no "target" key, so the
            # seed used to raise KeyError and kill the whole shard. Stratified
            # splitting additionally needs two members of each class. A seed
            # failing either test carries no information about adaptation, which
            # is exactly what the target guard above already assumes.
            _ys = np.concatenate([np.asarray(d.y) for d in src]).astype(int)
            if len(np.unique(_ys)) < 2 or np.bincount(_ys).min() < 2:
                n_skipped += 1
                skip_reason = skip_reason or "source single-class or <2 in a class"
                continue
            # The seed is committed only if every fit succeeds. The two guards
            # above catch the source and target being single-class, but at a few
            # events per domain other fits are fragile in their own way -- an
            # internal split, a cross-validation fold -- and chasing each one
            # with its own invented threshold would be guessing. A seed on which
            # some fit is numerically impossible carries no information about
            # adaptation, which is what the guards already assume. Accumulating
            # into a per-seed dict first matters: appending directly would leave
            # the method lists at different lengths when a fit halfway down the
            # list raises, and every mean after that would compare different
            # subsets of seeds.
            _y = np.asarray(tgt.y)
            try:
                _res = {
                    "erm":  fit_all_observed(src, tgt),
                    "latent_oracle": fit_invariant_oracle(src, tgt),
                    "obs_oracle": fit_observed_target_oracle(src, tgt, seed=s),
                    "coral": fit_coral(src, tgt, seed=s),
                    "iw_erm": fit_iw_erm(src, tgt, seed=s),
                }
                vals = {k: v["target"]["auroc"] for k, v in _res.items()}
                # Inside the same try: a seed on which any fit is numerically
                # impossible is dropped whole, so the AUROC and ECE means never
                # end up averaging different subsets of seeds.
                _cal = {"prior_recalib": fit_prior_recalib(src, tgt),
                        "platt_scaling": fit_platt_scaling(src, tgt)}
                vals_ece = {"erm": _res["erm"]["target"]["ece"],
                            **{k: v["target"]["ece"] for k, v in _cal.items()}}
                if n_src >= 2:
                    from experiments.adaptation_methods import fit_irm
                    from experiments.group_dro import fit_group_dro
                    vals["irm"] = fit_irm(src, tgt, seed=s)["target"]["auroc"]
                    vals["group_dro"] = fit_group_dro(src, tgt, seed=s)["target"]["auroc"]
                if with_nets:
                    from experiments.adaptation_methods import fit_dann, fit_cdan
                    vals["dann"] = fit_dann(src, tgt, seed=s, use_iwcv=False)["target"]["auroc"]
                    vals["cdan"] = fit_cdan(src, tgt, seed=s, use_iwcv=False)["target"]["auroc"]
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                n_skipped += 1
                skip_reason = f"{type(exc).__name__}: {exc}"
                continue
            minority.append(float(min(_y.mean(), 1.0 - _y.mean())))
            for _k, _v in vals.items():
                acc[_k].append(_v)
            for _k, _v in vals_ece.items():
                acc_ece[_k].append(_v)
    finally:
        if prev_env is None:
            os.environ.pop("SHIFT_SURV_CONFIG", None)
        else:
            os.environ["SHIFT_SURV_CONFIG"] = prev_env

    A = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    if len(A["erm"]) < 2:
        return {"regime": regime, "verdict": "degenerate", **_flat(cfgv),
                "n_seeds": len(A["erm"]), "n_seeds_skipped": n_skipped,
                "skip_reason": skip_reason}

    means = {m: float(np.nanmean(A[m])) for m in family}
    # max() keeps its first element and replaces it only when f(x) > f(best);
    # every comparison against NaN is False, so a family whose first member
    # scored NaN returned that member as the "best". The gain and headroom were
    # then NaN too, and _verdict fell through all four of its tests -- each also
    # False on NaN -- onto its final "supported". A configuration where
    # adaptation produced no usable number was being counted as evidence FOR the
    # paper's claim, which is the one direction an error here must never take.
    finite = {m: v for m, v in means.items() if np.isfinite(v)}
    if not finite:
        return {"regime": regime, "verdict": "degenerate", **_flat(cfgv),
                "n_seeds": len(A["erm"]), "n_seeds_skipped": n_skipped,
                "skip_reason": skip_reason or "nessun metodo ha prodotto un AUROC finito"}
    best = max(finite, key=finite.get)
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

    # Paired over seeds, same convention as the AUROC gain above. The sign is
    # chosen so that positive means the recalibration REDUCED the error, i.e.
    # improved calibration, matching Delta_ECE in the paper.
    E = {k: np.asarray(v, dtype=float) for k, v in acc_ece.items()}
    if len(E["erm"]) >= 1:
        dp_m, dp_ci = _mci(E["erm"] - E["prior_recalib"])
        dl_m, dl_ci = _mci(E["erm"] - E["platt_scaling"])
        ece_erm = float(np.nanmean(E["erm"]))
    else:
        dp_m = dp_ci = dl_m = dl_ci = ece_erm = float("nan")
    return {
        "regime": regime, **_flat(cfgv),
        "n_seeds": len(A["erm"]), "n_seeds_skipped": n_skipped,
        "skip_reason": skip_reason, "family": "+".join(family),
        "erm_auroc": round(float(np.nanmean(A["erm"])), 4),
        "latent_oracle_auroc": round(float(np.nanmean(A["latent_oracle"])), 4),
        "obs_oracle_auroc": round(float(np.nanmean(A["obs_oracle"])), 4),
        "minority_fraction": round(float(np.mean(minority)), 4) if minority else float("nan"),
        "best_adapt": best, "best_adapt_auroc": round(means[best], 4),
        "adapt_gain": round(g_m, 4), "adapt_gain_ci95": round(g_ci, 4),
        "adapt_gain_lo": round(g_m - g_ci, 4),
        "adapt_gain_hi": round(g_m + g_ci, 4),
        "ece_erm": round(ece_erm, 4),
        "d_prior_ece": round(dp_m, 4),
        "d_prior_lo": round(dp_m - dp_ci, 4),
        "d_prior_hi": round(dp_m + dp_ci, 4),
        "d_platt_ece": round(dl_m, 4),
        "d_platt_lo": round(dl_m - dl_ci, 4),
        "d_platt_hi": round(dl_m + dl_ci, 4),
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

    A non-finite gain or headroom is ``degenerate``, never one of the five. Each
    of the comparisons below is False on NaN, so without this guard such a
    configuration reached the final ``supported`` -- counting a cell that
    measured nothing as evidence for the claim under test.
    """
    if not (np.isfinite(gain) and np.isfinite(gain_ci) and np.isfinite(headroom)):
        return "degenerate"
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
    # Recorded per row: a world whose k had to be reduced to stay inside the
    # overcompleteness assumption is not quite the world the hypercube drew, and
    # a reader comparing rows should be able to see which ones were adjusted.
    out["k_clamped"] = cfgv.get("k_clamped", 0)
    out["amplitude"] = str(cfgv.get("amplitude"))
    w = _w_vector(cfgv["w_shape"], float(cfgv["w_norm"]), int(cfgv["k_inv"]))
    out["w"] = "[" + ",".join(f"{x:.3f}" for x in w) + "]"
    # The achieved cosine, not the requested one: the subset that realises a
    # target exactly may not exist for a given w, and the row has to say what it
    # actually got.
    amp = cfgv.get("amplitude")
    if isinstance(amp, (int, float)) and cfgv.get("_is_concept"):
        out["concept_cos_target"] = round(float(amp), 4)
        out["concept_cos"] = round(_concept_cosine(w, _flip_for_cosine(w, float(amp))), 4)
    else:
        out["concept_cos_target"] = ""
        out["concept_cos"] = ""
    return out


def _defaults(regime: str) -> dict:
    d = {k: v["default"] for k, v in AXES.items()}
    d["amplitude"] = AMPLITUDE[regime]["default"]
    return d


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def oat(regimes, seeds, n, n_src, with_nets, axes=None) -> list[dict]:
    rows, t0 = [], time.time()
    jobs = []
    for r in regimes:
        for ax in (axes or (list(AXES) + ["amplitude"])):
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
        vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"  [{i:>3}/{len(jobs)}] {r:<22}{ax:<8}={vs:<14}"
              f"gain={row.get('adapt_gain', float('nan')):+.4f}"
              f"  headroom={row.get('headroom', float('nan')):+.4f}"
              f"  {row['verdict']:<14}({time.time()-t0:.0f}s)", flush=True)
    return rows


def _from_unit(spec, u: float):
    """Map u in [0,1] to a parameter value, so a Latin hypercube can drive every
    axis through the same interface the random draws use."""
    kind, lo, hi = spec
    if kind == "int":
        return int(min(hi, lo + int(u * (hi - lo + 1))))
    if kind == "logint":
        return int(round(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))))
    if kind == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    if kind == "lin":
        return float(lo + u * (hi - lo))
    if kind == "choice":
        _lo, _hi = min(_CONCEPT_COSINES), max(_CONCEPT_COSINES)
        return float(_lo + u * (_hi - _lo))
    raise ValueError(kind)


def worlds(regimes, seeds, n, n_src, with_nets, n_worlds, seed0,
           world_slice: tuple[int, int] = (0, 1)) -> list[dict]:
    """Space-filling sample of joint configurations.

    A Latin hypercube rather than independent uniform draws: with a hundred
    points in eight dimensions, independent sampling leaves large parts of each
    marginal empty by chance, and the whole reason for this mode is coverage. The
    hypercube stratifies every axis, so each of the hundred worlds sits in its
    own slice of every marginal.

    The direction of w is the exception. It is a point on a sphere, not a scalar,
    so it keeps its own uniform draw; stratifying a direction through one
    hypercube column would not be uniform on the sphere.

    One extra column per regime carries that regime's amplitude, so severity is
    stratified alongside the generator parameters rather than resampled
    independently inside the loop.
    """
    from scipy.stats import qmc

    # The hypercube is always built over ALL regimes, never over the subset being
    # run. Its dimension is 8 + len(REGIMES), so sizing it to the requested
    # regimes would give a different design -- and therefore different worlds --
    # every time the run is split. Splitting by regime across machines is exactly
    # how this gets parallelised, and a world has to mean the same thing in every
    # shard or "all regimes inside one world" is not what happened.
    scalar = [k for k in AXES if AXES[k]["draw"][0] != "sphere"]
    d = len(scalar) + len(REGIMES)
    sample = qmc.LatinHypercube(d=d, seed=seed0).random(n_worlds)
    rng = np.random.default_rng(seed0 + 1)          # sphere directions only

    # Splitting across machines takes a contiguous slice of the SAME hypercube,
    # never a fresh one with a different seed: a different seed is a different
    # design, and two blocks would then be exploring two different spaces while
    # claiming to be one run. The rows are generated identically everywhere and
    # each block simply evaluates its own range of them.
    k, nk = world_slice
    idx = list(range(n_worlds))[k * n_worlds // nk:(k + 1) * n_worlds // nk]

    rows, t0 = [], time.time()
    print(f"  worlds: {n_worlds} mondi (Latin hypercube, {d} dimensioni), "
          f"blocco {k+1}/{nk} = mondi {idx[0] if idx else '-'}-{idx[-1] if idx else '-'}"
          f"  x {len(regimes)} regimi x {len(seeds)} seed\n")
    for wi in idx:
        u = sample[wi]
        cfgv = {k: _from_unit(AXES[k]["draw"], u[i]) for i, k in enumerate(scalar)}
        cfgv["w_shape"] = _draw(("sphere", None, None), rng)
        # The benchmark assumes an overcomplete observation, p >> k_inv + k_spu.
        # A joint draw can leave that region -- k=(5,4) with p=10 gives a 10x9
        # loading matrix -- and such a world is not a harder version of this
        # benchmark but a different one. Clamp k rather than drop the world, so
        # the hypercube keeps all its rows and only the offending coordinate
        # moves; the adjustment is recorded per row.
        cfgv["k_clamped"] = 0
        while cfgv["k_inv"] + cfgv["k_spu"] > MAX_K_OVER_P * cfgv["p"]:
            if cfgv["k_spu"] > 1:
                cfgv["k_spu"] -= 1
            elif cfgv["k_inv"] > 2:
                cfgv["k_inv"] -= 1
            else:
                break
            cfgv["k_clamped"] = 1
        for r in regimes:
            c = dict(cfgv)
            # Fixed column per regime, by its position in REGIMES rather than in
            # the requested subset, for the same reason.
            c["amplitude"] = _from_unit(AMPLITUDE[r]["draw"],
                                        u[len(scalar) + REGIMES.index(r)])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row = _evaluate(r, c, seeds, n, n_src, with_nets)
            except Exception as exc:
                # One pathological world must not cost the other ninety-nine.
                # worlds() only returns its rows at the end, so an exception used
                # to discard every world already computed in the shard. The row is
                # kept with an explicit verdict so a failure stays visible in the
                # output instead of silently shrinking the design.
                print(f"      mondo {wi+1} / {r}: {type(exc).__name__}: {exc}",
                      flush=True)
                row = {"regime": r, "verdict": "error",
                       "error": f"{type(exc).__name__}: {exc}", **_flat(c)}
            row.update(mode="worlds", world=wi, axis="", value="", is_default=False)
            rows.append(row)
        done = [x for x in rows if x.get("world") == wi and "adapt_gain" in x]
        worst = max(done, key=lambda x: x["adapt_gain"], default=None)
        print(f"  [{wi+1:>3}/{n_worlds}] k=({cfgv['k_inv']},{cfgv['k_spu']}) "
              f"p={cfgv['p']:<3} sig={cfgv['sigma']:.2f} lam={cfgv['lambda0']:.4f} "
              f"nu={cfgv['nu']:.2f} |w|={cfgv['w_norm']:.2f}   peggior gain="
              f"{(worst['adapt_gain'] if worst else float('nan')):+.4f}"
              f" ({worst['regime'] if worst else '-'})   ({time.time()-t0:.0f}s)",
              flush=True)
    return rows


def confirm(csv_paths, seeds, n, n_src, with_nets, n_confirm,
            min_erm_auroc: float | None = None,
            only_breaks: bool = False) -> list[dict]:
    """Re-run the largest observed gains on seeds disjoint from the screening set.

    A sweep is a search for candidate counterexamples, not an estimate of the
    maximum gain. The largest value found after looking at hundreds of
    configurations is selected, and a selected maximum is biased upwards by
    exactly the amount that makes it look interesting. The only way to know
    whether a candidate is real is to measure it again on data that had no part
    in selecting it -- which is what the disjoint confirmation seeds in the
    frozen protocol are for.

    Uninformative configurations are excluded from the ranking: their gain is
    near zero by construction, so they can neither produce nor confirm a break.

    ``min_erm_auroc`` restricts the candidate pool to configurations whose
    screening ERM reached that AUROC
    ------------------------------------------------------------------------
    Every break the sweep finds sits under severe concept shift, where ERM is
    inverted -- it ranks patients backwards and lands below chance -- so the
    confirmed gain is the distance between a model that is badly wrong and one
    that is slightly less badly wrong. The question that leaves open is whether
    any break survives among configurations where the deployed model was worth
    deploying at all, and that is what this filter answers.

    It selects which configurations are *carried into* confirmation. It does not
    touch ``_verdict``: the decision rule stays exactly as pre-specified, the
    unfiltered confirmation is still the headline, and this runs beside it as a
    declared subgroup rather than in place of it. Amending the rule after seeing
    which cells it flags would void the pre-specification, which is the only
    reason the sensitivity claim carries any weight.

    The filter reads the screening ERM, not the confirmation one, because it has
    to be applied before the confirmation is run.

    ``only_breaks`` restricts the pool to configurations the screening already
    flagged
    -------------------------------------------------------------------------
    The default pool is every informative configuration, ranked by gain, because
    a confirmation stage should be able to catch a real effect the screening
    verdict happened to miss. Combined with ``min_erm_auroc`` that is the wrong
    pool for one specific question -- "of the configurations that broke, do any
    survive once the inverted ones are set aside?" -- which is about the breaks
    themselves and not about the largest gains in a subgroup. Setting both
    reproduces exactly that analysis.
    """
    import ast, csv as _csv

    pool: list[dict] = []
    n_inverted = n_not_break = 0
    for path in csv_paths:
        with open(path) as fh:
            for r in _csv.DictReader(fh):
                if r.get("verdict") in ("uninformative", "degenerate", "error"):
                    continue
                try:
                    r["_gain"] = float(r["adapt_gain"])
                except (KeyError, ValueError):
                    continue
                if only_breaks and r.get("verdict") not in ("BREAK", "candidate_break"):
                    n_not_break += 1
                    continue
                if min_erm_auroc is not None:
                    try:
                        erm = float(r["erm_auroc"])
                    except (KeyError, ValueError):
                        continue
                    if not (erm >= min_erm_auroc):
                        n_inverted += 1
                        continue
                r["_src"] = Path(path).name
                pool.append(r)
    if not pool:
        # An empty pool is the answer, not a failure: with only_breaks and a
        # non-inverted filter it means no break survived the filter at all, which
        # is exactly what the analysis set out to measure. Say so rather than
        # exiting with a message that reads like a misconfiguration.
        detail = []
        if only_breaks:
            detail.append(f"{n_not_break} non erano break/candidate")
        if min_erm_auroc is not None:
            detail.append(f"{n_inverted} avevano ERM sotto {min_erm_auroc}")
        raise SystemExit(
            "nessuna configurazione da confermare"
            + (f" ({', '.join(detail)})" if detail else " nei CSV indicati"))

    pool.sort(key=lambda r: -r["_gain"])
    top = pool[:n_confirm]
    if only_breaks:
        print(f"  solo break: {n_not_break} configurazioni non flaggate escluse")
    if min_erm_auroc is not None:
        print(f"  sottogruppo non invertito: erm_auroc >= {min_erm_auroc}; "
              f"{n_inverted} configurazioni escluse, {len(pool)} restano candidabili")
    print(f"  conferma: {len(top)} configurazioni (su {len(pool)}) "
          f"su {len(seeds)} seed disgiunti {seeds[0]}-{seeds[-1]}\n")

    def _parse(v: str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v

    rows, t0 = [], time.time()
    for i, r in enumerate(top, 1):
        cfgv = {k: _parse(r[k]) for k in AXES}
        cfgv["amplitude"] = _parse(r["amplitude"])
        # A sampled world records its w as coordinates, not as a shape name: the
        # name "sphere" is a tag, not something _w_vector can resolve. Rebuilding
        # the direction from the recorded coordinates is what makes a world
        # reproducible at all, and a confirmation stage that cannot reproduce the
        # configuration it is confirming would be measuring something else.
        if cfgv.get("w_shape") == "sphere":
            w_rec = _parse(r["w"])
            if not isinstance(w_rec, (list, tuple)) or not len(w_rec):
                raise SystemExit(
                    f"mondo non riproducibile: w_shape='sphere' ma colonna w={r.get('w')!r}")
            cfgv["w_shape"] = [float(x) for x in w_rec]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = _evaluate(r["regime"], cfgv, seeds, n, n_src, with_nets)
        row.update(mode="confirm", axis=r.get("axis", ""), value=r.get("value", ""),
                   is_default=False, screen_gain=r["_gain"],
                   screen_verdict=r.get("verdict", ""), screen_source=r["_src"])
        rows.append(row)
        held = row["verdict"] in ("BREAK", "candidate_break")
        print(f"  [{i:>2}/{len(top)}] {r['regime']:<22}"
              f"screening={r['_gain']:+.4f} ({r.get('verdict','')})  ->  "
              f"conferma={row.get('adapt_gain', float('nan')):+.4f} "
              f"[{row.get('adapt_gain_lo', float('nan')):+.4f}, "
              f"{row.get('adapt_gain_hi', float('nan')):+.4f}]  {row['verdict']}"
              f"{'' if held else '   -> non confermata'}   ({time.time()-t0:.0f}s)",
              flush=True)
    return rows


def report(rows: list[dict]) -> None:
    n_err = sum(r.get("verdict") == "error" for r in rows)
    good = [r for r in rows if r["verdict"] not in ("degenerate", "error")]
    STATES = ("supported", "inconclusive", "candidate_break", "BREAK", "uninformative")
    n = {k: sum(r["verdict"] == k for r in good) for k in STATES}
    if n_err:
        print(f"\n  ATTENZIONE: {n_err} configurazioni terminate con errore ed "
              f"escluse dai conteggi. Cercale con verdict=error nel CSV.")
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
        _report_break_levels(brk)
        print(f"\n  {len(brk)} DA RICONFERMARE SU SEED DISGIUNTI (BREAK o candidate_break):")
        for r in sorted(brk, key=lambda x: -x["adapt_gain"])[:20]:
            cfg = " ".join(f"{k}={r[k]}" for k in AXES)
            print(f"    {r['regime']:<22}gain={r['adapt_gain']:+.4f} ({r['best_adapt']})"
                  f"  amp={r['amplitude']}  {cfg}")
    else:
        print("\n  Nessuna configurazione in cui la stima puntuale supera il margine.")


# The margin is a rule about a *difference*. It says nothing about where the two
# models being differenced actually sit, and under severe concept shift they sit
# below chance: the source relationship is inverted, ERM ranks patients backwards,
# and an adaptation method that makes it less backwards earns a gain above the
# margin while remaining unusable. Those configurations are genuine breaks under
# the pre-specified rule and are reported as such -- the rule is not amended after
# seeing which cells it flags, which would void the pre-specification that makes
# the sensitivity claim worth anything. What follows is the level the rule does
# not look at, so the two can be read together.
_CHANCE = 0.5


def _report_break_levels(brk: list[dict]) -> None:
    """Absolute AUROC of the winner in each break, alongside the gap that flagged it."""
    def _f(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    win = np.array([_f(r, "best_adapt_auroc") for r in brk])
    erm = np.array([_f(r, "erm_auroc") for r in brk])
    head = np.array([_f(r, "headroom") for r in brk])
    gain = np.array([_f(r, "adapt_gain") for r in brk])

    n_win_ok = int(np.nansum(win > _CHANCE))
    n_erm_inv = int(np.nansum(erm < _CHANCE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ok = np.isfinite(gain) & np.isfinite(head) & (head > 0)
        frac = float(np.median(gain[ok] / head[ok])) if ok.any() else float("nan")

    print(f"\n  LIVELLO ASSOLUTO NEI {len(brk)} BREAK/candidate "
          f"(la regola guarda il divario, non il livello):")
    print(f"    vincitore sopra {_CHANCE} AUROC        {n_win_ok:>4} / {len(brk)}")
    print(f"    ERM invertito (sotto {_CHANCE})        {n_erm_inv:>4} / {len(brk)}")
    print(f"    AUROC vincitore, mediana        {np.nanmedian(win):>9.4f}")
    print(f"    AUROC ERM, mediana              {np.nanmedian(erm):>9.4f}")
    print(f"    quota di headroom recuperata    {frac:>9.4f}  (mediana)")
    if n_win_ok == 0 and len(brk):
        print(f"    -> in nessun break il vincitore raggiunge il caso: il divario e' "
              f"reale sotto la regola fissata, il modello resta inutilizzabile.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sensitivity of the conclusion to the generator's free parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", default="oat", choices=["oat", "worlds", "confirm"])
    ap.add_argument("--confirm_from", nargs="+", default=None, metavar="CSV",
                    help="CSV di screening da cui prendere i top-gain (modo confirm)")
    ap.add_argument("--n_confirm", type=int, default=None)
    ap.add_argument("--confirm_min_erm", type=float, default=None, metavar="AUROC",
                    help="modo confirm: limita i candidati alle configurazioni il cui "
                         "ERM di screening raggiunge questa AUROC (usa 0.5 per il "
                         "sottogruppo non invertito). Non tocca la regola _verdict; "
                         "va lanciato ACCANTO alla conferma senza filtro, non al suo posto")
    ap.add_argument("--confirm_only_breaks", action="store_true",
                    help="modo confirm: limita i candidati alle configurazioni gia' "
                         "flaggate BREAK/candidate_break dallo screening. Con "
                         "--confirm_min_erm 0.5 risponde a 'dei break, quanti "
                         "sopravvivono escludendo quelli con ERM invertito'")
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n_source_domains", type=int, default=1)
    ap.add_argument("--n_worlds", type=int, default=100)
    ap.add_argument("--world_seed", type=int, default=12345)
    ap.add_argument("--world_slice", default="0/1", metavar="K/N",
                    help="valuta solo il blocco K di N dello stesso ipercubo")
    # One-at-a-time perturbs each axis independently around the same centre, so a
    # subset of axes is the same rows the full run would have produced. Unlike
    # the worlds mode, splitting here needs no care: there is no shared design to
    # keep consistent.
    ap.add_argument("--axes", nargs="+", default=None, metavar="AXIS",
                    help="sottoinsieme di assi per il modo oat (default: tutti)")
    # DANN and CDAN cost 3.2x, not the 30x first estimated: 129s vs 41s per
    # configuration at ten seeds. That makes them affordable where it matters and
    # unaffordable only in the mode with the most cells. Excluding them from
    # every mode would narrow the claim from "standard unsupervised adaptation"
    # to "CORAL and importance weighting", which is a different paper.
    #
    # They are now included in every mode. Excluding them from the coverage
    # search would have forced the text to say that global robustness was checked
    # only for the cheap subset, which is a weaker claim than the one the paper
    # makes. --no_nets restores the reduced family where the budget requires it.
    ap.add_argument("--with_nets", dest="with_nets", action="store_true", default=None)
    ap.add_argument("--no_nets", dest="with_nets", action="store_false")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    seeds = list(range(args.n_seeds))
    if args.with_nets is None:
        args.with_nets = True
    if args.mode != "confirm" and (args.confirm_min_erm is not None
                                   or args.confirm_only_breaks):
        ap.error("--confirm_min_erm e --confirm_only_breaks valgono solo con --mode confirm")
    print(f"\n  modalita' {args.mode}, {args.n_source_domains} sorgente/i, "
          f"famiglia {'con' if args.with_nets else 'senza'} DANN/CDAN")
    t0 = time.time()
    if args.mode == "confirm":
        srcs = args.confirm_from or [str(_TABLES / f"sensitivity_{m}_{args.n_source_domains}src.csv")
                                     for m in ("oat", "worlds")]
        srcs = [x for x in srcs if Path(x).exists()]
        if not srcs:
            raise SystemExit("nessun CSV di screening trovato; passa --confirm_from")
        rows = confirm(srcs, PROTOCOL["confirmation_seeds"], args.n,
                       args.n_source_domains, args.with_nets,
                       args.n_confirm or PROTOCOL["n_confirm"],
                       min_erm_auroc=args.confirm_min_erm,
                       only_breaks=args.confirm_only_breaks)
    elif args.mode == "oat":
        rows = oat(args.regimes, seeds, args.n, args.n_source_domains,
                   args.with_nets, axes=args.axes)
    else:
        _k, _n = (int(x) for x in args.world_slice.split("/"))
        rows = worlds(args.regimes, seeds, args.n, args.n_source_domains,
                      args.with_nets, args.n_worlds, args.world_seed,
                      world_slice=(_k, _n))
    report(rows)

    # The filtered confirmation is an additional analysis, so it gets its own file:
    # writing it over sensitivity_confirm_<k>src.csv would replace the headline
    # result with the subgroup and leave no trace that the filter was applied.
    prefix = args.out_prefix or f"sensitivity_{args.mode}_{args.n_source_domains}src"
    if args.confirm_min_erm is not None and not args.out_prefix:
        prefix += "_noninverted"
    _TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, _TABLES / f"{prefix}.csv")
    print(f"\n  {time.time()-t0:.0f}s   Saved: {_TABLES}/{prefix}.csv")
