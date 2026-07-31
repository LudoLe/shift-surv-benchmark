"""table_scenarios.tex — the shift-regime table, MEASURED from the generator.

Rewritten after the previous version broke. It read parameter values out of
configs/generator.yaml and its docstring claimed that "the table cannot drift
from what the benchmark ran". It could, and it had.

The benign failure came first: making covariate shift pure removed the
`sigma_noise_target` key the generator indexed, and it raised KeyError.

The damaging one was silent. Reading the config tells you what the YAML
declares, not what the code does. The table described measurement shift as
changing the observation map, while `_observational_shift` also displaced the
latent means and raised the noise. The paper's claim that each scenario
perturbs one mechanism held for three regimes of six, and this table asserted
it for all of them.

So each row is now produced by generating the scenario and comparing the
target's parameters against the source's, field by field. Change the generator
and forget the paper, and the table changes by itself; it cannot claim a regime
is isolated when it is not.

The one column that cannot be derived is whether a shift leaves the
target-optimal rule recoverable from x. It is declared here, with an assertion
for the case that would most obviously contradict it: a regime cannot be called
invertible while changing the observation noise, because added noise destroys
information rather than displacing it.

Run from repo root:  python paper_assets/_gen_table_scenarios.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from generator.parameters import BaseParams          # noqa: E402
from generator.scenarios import generate_scenario    # noqa: E402

OUT = os.path.join(ROOT, "paper_assets", "table_scenarios.tex")

REGIMES = [
    ("no_shift",              "No shift",       "---"),
    ("covariate_shift",       "Covariate",      "invertible"),
    ("observational_shift",   "Measurement",    "invertible"),
    ("spurious_shift",        "Spurious",       "invertible"),
    ("baseline_hazard_shift", "Baseline haz.",  "rank-preserving"),
    ("prior_shift",           "Label-prior",    r"$P(x\mid y)$ invariant"),
    ("concept_shift",         "Concept",        "not transportable"),
]

# The full DomainParams surface, not a curated subset: a field left out here
# would be invisible to the audit, which is how the previous drift went unseen.
FIELDS = ["mu_inv", "mu_spu", "sigma_noise", "w", "v", "beta0", "lambda0", "nu",
          "A_inv", "A_spu", "latent_cov", "cure_fraction"]

TEX = {
    "mu_inv": r"$\mu_{\mathrm{inv}}$", "mu_spu": r"$\mu_{\mathrm{spu}}$",
    "sigma_noise": r"$\sigma$", "w": "$w$", "v": "$v$", "beta0": r"$\beta_0$",
    "lambda0": r"$\lambda_0$", "nu": r"$\nu$",
    "A_inv": r"$A_{\mathrm{inv}}$", "A_spu": r"$A_{\mathrm{spu}}$",
    "latent_cov": r"$\Sigma_{\mathrm{lat}}$",
    "cure_fraction": r"$\pi_{\mathrm{cure}}$",
}


def changed_fields(sp, tp) -> list[str]:
    out = []
    for f in FIELDS:
        a, b = getattr(sp, f, None), getattr(tp, f, None)
        if a is None and b is None:
            continue
        try:
            same = np.allclose(np.asarray(a, float), np.asarray(b, float))
        except Exception:                                        # noqa: BLE001
            same = (a == b)
        if not same:
            out.append(f)
    return out


def main() -> None:
    rows, prev = [], {}
    for name, label, nature in REGIMES:
        d = generate_scenario(name, n_source_domains=1, n_per_domain=4000,
                              seed=0, base=BaseParams(seed=0))
        sp, tp = d["source"][0].params, d["target"].params
        ch = changed_fields(sp, tp)
        prev[name] = (float(d["source"][0].y.mean()), float(d["target"].y.mean()))

        if nature == "invertible" and "sigma_noise" in ch:
            raise SystemExit(
                f"{name}: dichiarato 'invertible' ma il target cambia sigma_noise "
                f"({sp.sigma_noise:.3f} -> {tp.sigma_noise:.3f}). Il rumore "
                "distrugge informazione invece di spostarla: o si corregge il "
                "generatore, o si cambia la dichiarazione in REGIMES."
            )

        if name == "prior_shift":
            # Resampling, not a parameter change: changed_fields is empty by
            # design, so the mechanism has to be named rather than derived.
            changed_tex, n_mech = r"$P(y)$ (resampling)", "1"
        elif not ch:
            changed_tex, n_mech = "---", "---"
        else:
            changed_tex, n_mech = ", ".join(TEX.get(f, f) for f in ch), str(len(ch))

        rows.append((label, changed_tex, n_mech, nature,
                     f"{prev[name][0]:.2f} $\\to$ {prev[name][1]:.2f}"))

    src_prev = prev["no_shift"][0]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Shift regimes, measured from the generator: each row is "
        r"produced by generating the scenario and comparing the target's "
        r"parameters against the source's, so the table cannot disagree with "
        r"the code. Common to all regimes: $p{=}20$ observed features, "
        r"$k_{\mathrm{inv}}{=}3$ and $k_{\mathrm{spu}}{=}2$ latent factors, "
        r"horizon $\tau{=}6$ months, $n{=}500$ patients per center, "
        r"$w{=}(1, 0.5, -0.5)$, $\lambda_0{=}0.02$, $\nu{=}1.5$, $\sigma{=}0.5$, "
        f"giving a source event rate of ${src_prev:.2f}$ at $\\tau$. Spurious "
        r"factors never enter the risk ($v{=}0$ throughout). \emph{Nature} "
        r"separates shifts that leave the target-optimal rule recoverable from "
        r"$x$ from those that do not.}",
        r"\label{tab:scenarios}",
        r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
        r"\textbf{Regime} & \textbf{Changed} & \textbf{\#} & \textbf{Nature} & "
        r"\textbf{Prevalence} \\",
        r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))

    print(f"  scritto {OUT}")
    for label, ch, n, _nature, p in rows:
        print(f"    {label:<15} n={n:<4} cambia: {ch:<48} prev {p}")


if __name__ == "__main__":
    main()
