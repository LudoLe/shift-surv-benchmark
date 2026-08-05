"""Calibrate generator parameters to interpretable targets by bisection.

Why calibrate instead of picking values
---------------------------------------
``lambda0 = 0.02`` says nothing to a reader; "the value that makes the minority
class 33% of the cohort" does. The same holds for ``w`` (which really means "how
discriminable is the outcome at all") and ``sigma`` (which really means "how much
of that signal survives the observation process"). Choosing the raw numbers by
doubling and halving is a convention with no defence; choosing them so that an
interpretable quantity hits a stated target is a design decision that can be
stated, checked, and argued with.

Protocol
--------
Calibration runs on a fixed super-population -- one large sample, generated once
from a dedicated seed -- and never on the finite samples the benchmark itself
uses. That separation is what keeps the calibrated value from absorbing the
sampling noise of the replicates it will later parameterise. Common random
numbers are used across bisection steps, so the objective is a smooth function
of the parameter rather than a noisy one, which is what makes bisection valid
here at all.

Order matters, because the three targets are not independent: ``w`` sets the
ceiling, ``sigma`` decides how much of it reaches ``x``, and ``lambda0`` moves
the prevalence (which also shifts discrimination slightly). They are calibrated
in that order, in a single pass, and the ACHIEVED value of every target is
recorded next to the requested one. A residual mismatch is therefore visible
rather than hidden -- which is the point, since a second pass would buy little
and would make the procedure harder to state.

Usage
-----
  python -m experiments.calibrate                    # calibrate the frozen space
  python -m experiments.calibrate --out calibrated_space.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

_SPACE = _ROOT / "configs" / "parameter_space.yaml"
_GENCFG = _ROOT / "configs" / "generator.yaml"


# ---------------------------------------------------------------------------
# Bisection
# ---------------------------------------------------------------------------

def calibrate_parameter(
    name: str,
    lower: float,
    upper: float,
    target: float,
    evaluate: Callable[[float], float],
    tolerance: float = 0.005,
    max_iter: int = 40,
    verbose: bool = True,
) -> dict:
    """Bisect ``evaluate`` on [lower, upper] until it is within ``tolerance``.

    ``evaluate`` must be (weakly) monotone in the parameter over the bracket;
    the direction is detected from the endpoints rather than assumed, so the
    same routine works for a target that increases with the parameter
    (prevalence in lambda0) and for one that decreases (observed AUROC in sigma).

    Returns the achieved value as well as the parameter, and never raises on a
    bracket failure: it reports it. A silent clamp to an endpoint would be a
    calibration that looks successful and is not.
    """
    f_lo, f_hi = evaluate(lower), evaluate(upper)
    increasing = f_hi >= f_lo
    lo_v, hi_v = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if not (lo_v - tolerance <= target <= hi_v + tolerance):
        return {"parameter": name, "target": target, "value": float("nan"),
                "achieved": float("nan"), "n_iter": 0, "bracketed": False,
                "endpoints": [float(f_lo), float(f_hi)],
                "note": f"target {target} outside [{min(f_lo,f_hi):.4f}, "
                        f"{max(f_lo,f_hi):.4f}] on [{lower}, {upper}]"}

    a, b = float(lower), float(upper)
    mid, f_mid = a, f_lo
    for i in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = evaluate(mid)
        if abs(f_mid - target) <= tolerance:
            break
        if (f_mid < target) == increasing:
            a = mid
        else:
            b = mid
    if verbose:
        print(f"    {name:<10} target={target:.3f}  ->  {mid:.6g}  "
              f"(achieved {f_mid:.4f}, {i+1} iter)")
    return {"parameter": name, "target": float(target), "value": float(mid),
            "achieved": float(f_mid), "n_iter": i + 1, "bracketed": True,
            "endpoints": [float(f_lo), float(f_hi)], "note": ""}


# ---------------------------------------------------------------------------
# The three objectives, all on the fixed super-population
# ---------------------------------------------------------------------------

class SuperPopulation:
    """One large no-shift sample, regenerated per parameter setting but always
    from the same seed, so bisection sees a smooth objective (common random
    numbers) rather than fresh sampling noise at every step."""

    def __init__(self, n: int = 20000, seed: int = 999, p: int = 20,
                 k_inv: int = 3, k_spu: int = 2):
        self.n, self.seed, self.p = n, seed, p
        self.k_inv, self.k_spu = k_inv, k_spu

    def _generate(self, lambda0: float, nu: float, w: list[float], sigma: float):
        import os, yaml
        import generator.config as gc
        from generator.parameters import BaseParams
        from generator.scenarios import generate_scenario

        cfg = yaml.safe_load(_GENCFG.read_text())
        cfg["domain_defaults"].update(
            {"lambda0": float(lambda0), "nu": float(nu),
             "w": [float(x) for x in w], "sigma_noise": float(sigma)})
        tmp = Path(gc.ROOT) / "configs" / "_calib_tmp.yaml"
        tmp.write_text(yaml.safe_dump(cfg))
        gc._CACHE.pop(str(tmp), None)
        prev = os.environ.get("SHIFT_SURV_CONFIG")
        os.environ["SHIFT_SURV_CONFIG"] = str(tmp)
        try:
            data = generate_scenario(
                "no_shift", n_source_domains=1, n_per_domain=self.n,
                seed=self.seed, base=BaseParams(seed=self.seed, p=self.p,
                                                k_inv=self.k_inv, k_spu=self.k_spu))
        finally:
            if prev is None:
                os.environ.pop("SHIFT_SURV_CONFIG", None)
            else:
                os.environ["SHIFT_SURV_CONFIG"] = prev
            tmp.unlink(missing_ok=True)
        return data

    def event_prevalence(self, lambda0, nu, w, sigma) -> float:
        """P(event by tau), which is monotone increasing in lambda0.

        The frozen space declares its targets as the *minority class fraction*,
        for polarity invariance against I3LUNG's OS_6 (where 1 means alive). The
        two coincide whenever the target is at or below 0.5, which every declared
        level is, so calibrating on the prevalence is both equivalent and
        bisectable. Calibrating on min(prev, 1-prev) directly is not: that
        function rises to 0.5 and falls again, so it is non-monotone and no
        bracket contains the target on a single branch.
        """
        d = self._generate(lambda0, nu, w, sigma)
        return float(d["target"].y.mean())

    def _auroc_of(self, X, y) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan")
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=500, C=1.0))
        oof = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, oof))

    def latent_oracle_auroc(self, lambda0, nu, w, sigma) -> float:
        d = self._generate(lambda0, nu, w, sigma)
        return self._auroc_of(d["target"].u_inv, d["target"].y)

    def observed_oracle_auroc(self, lambda0, nu, w, sigma) -> float:
        d = self._generate(lambda0, nu, w, sigma)
        return self._auroc_of(d["target"].x, d["target"].y)


def _w_of_norm(norm: float, shape=(1.0, 0.5, -0.5), k_inv: int = 3) -> list[float]:
    v = np.asarray(shape, dtype=float)
    if len(v) < k_inv:
        v = np.pad(v, (0, k_inv - len(v)))
    v = v[:k_inv]
    return list(v / np.linalg.norm(v) * norm)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(space_path: Path, out_path: Path, verbose: bool = True) -> dict:
    import yaml
    space = yaml.safe_load(space_path.read_text())
    proto = space["protocol"]
    pop = SuperPopulation(n=proto["superpopulation_n"],
                          seed=proto["superpopulation_seed"])

    # Benchmark defaults, used as the fixed background while one target moves.
    gen = yaml.safe_load(_GENCFG.read_text())["domain_defaults"]
    d_lam, d_nu = float(gen["lambda0"]), float(gen["nu"])
    d_w = [float(x) for x in gen["w"]]
    d_sigma = float(gen["sigma_noise"])
    d_wnorm = float(np.linalg.norm(d_w))

    results: dict = {"lambda0": [], "w_norm": [], "sigma": []}

    # 1. w_norm -> latent-oracle AUROC (the ceiling), at default lambda0/sigma.
    if verbose:
        print("\n  [1/3] w_norm  ->  AUROC oracolo latente")
    spec = space["calibrated_parameters"]["w_norm"]
    lo, hi = spec["search_bracket"]
    for lvl in spec["levels"]:
        r = calibrate_parameter(
            "w_norm", lo, hi, lvl["target"],
            lambda v: pop.latent_oracle_auroc(d_lam, d_nu, _w_of_norm(v), d_sigma),
            tolerance=spec["tolerance"], verbose=verbose)
        r.update(tag=lvl["tag"], note=lvl.get("note", ""))
        results["w_norm"].append(r)

    # 2. sigma -> observed-oracle AUROC, at the *default* w so the two axes stay
    #    interpretable independently (sigma is "what observation costs", not
    #    "what observation costs given this particular signal strength").
    if verbose:
        print("\n  [2/3] sigma  ->  AUROC oracolo su dati osservati")
    spec = space["calibrated_parameters"]["sigma"]
    lo, hi = spec["search_bracket"]
    for lvl in spec["levels"]:
        r = calibrate_parameter(
            "sigma", lo, hi, lvl["target"],
            lambda v: pop.observed_oracle_auroc(d_lam, d_nu, d_w, v),
            tolerance=spec["tolerance"], verbose=verbose)
        r.update(tag=lvl["tag"], note=lvl.get("note", ""))
        results["sigma"].append(r)

    # 3. lambda0 -> minority-class fraction.
    if verbose:
        print("\n  [3/3] lambda0  ->  prevalenza evento (= frazione minoritaria per target <= 0.5)")
    spec = space["calibrated_parameters"]["lambda0"]
    lo, hi = spec["search_bracket"]
    for lvl in spec["levels"]:
        assert lvl["target"] <= 0.5, (
            "a minority-class target above 0.5 is not a minority; declare it as "
            "1 - target instead")
        r = calibrate_parameter(
            "lambda0", lo, hi, lvl["target"],
            lambda v: pop.event_prevalence(v, d_nu, d_w, d_sigma),
            tolerance=spec["tolerance"], verbose=verbose)
        r.update(tag=lvl["tag"], note=lvl.get("note", ""))
        results["lambda0"].append(r)

    out = {
        "calibrated_on": {
            "superpopulation_n": pop.n, "superpopulation_seed": pop.seed,
            "background": {"lambda0": d_lam, "nu": d_nu, "w": d_w,
                           "sigma": d_sigma, "w_norm": d_wnorm},
        },
        "calibrated": results,
    }
    out_path.write_text(yaml.safe_dump(out, sort_keys=False))

    if verbose:
        print(f"\n{'='*78}\n  RIEPILOGO — richiesto vs raggiunto\n{'='*78}")
        print(f"  {'parametro':<10}{'target':>9}{'raggiunto':>11}{'valore':>13}"
              f"{'tag':>10}   ok")
        for par, rows in results.items():
            for r in rows:
                ok = ("si" if r["bracketed"] and
                      abs(r["achieved"] - r["target"]) <= 0.01 else "NO")
                print(f"  {par:<10}{r['target']:>9.3f}{r['achieved']:>11.4f}"
                      f"{r['value']:>13.6g}{r['tag']:>10}   {ok}")
        bad = [r for rows in results.values() for r in rows if not r["bracketed"]]
        if bad:
            print("\n  NON RAGGIUNTI (bracket troppo stretto, allargarlo nello YAML):")
            for r in bad:
                print(f"    {r['parameter']} target={r['target']}: {r['note']}")
        print(f"\n  Scritto: {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bisection calibration of the frozen space.")
    ap.add_argument("--space", default=str(_SPACE))
    ap.add_argument("--out", default=str(_ROOT / "configs" / "parameter_space_calibrated.yaml"))
    args = ap.parse_args()
    run(Path(args.space), Path(args.out))
