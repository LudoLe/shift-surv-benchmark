"""
Generate table_scenarios.tex — the shift table for the Setup section.

Replaces the "Effect" column of the previous version with the actual shift
*parameters*, which the prose above the table already describes in words. Every
number is read from configs/generator.yaml, so the table cannot drift from what
the benchmark ran (tests/test_config.py in turn pins the YAML to the Python
defaults).

Single-column `table` to match the existing draft and the 10-page budget.
Notation follows the paper: subscript inv/spu, \\Theta, \\lambda_0, \\nu.

Run from repo root:  python paper_assets/_gen_table_scenarios.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generator.config import load_config  # noqa: E402

OUT = os.path.join(ROOT, "paper_assets", "table_scenarios.tex")

# Measured, not configured: the no-shift source event rate at tau under the
# domain_defaults below. Quoted so the prevalence shifts read as source->target.
SOURCE_PREVALENCE = 0.33


def _g(x: float) -> str:
    """Trim trailing zeros so 1.5 -> 1.5 and 0.60 -> 0.6."""
    return f"{float(x):g}"


def rows(cfg: dict) -> list[tuple[str, str, str, str]]:
    """(scenario label, changed, held fixed, parameters) — parameters from cfg."""
    s = cfg["scenarios"]
    d = cfg["domain_defaults"]
    sigma_src = d["sigma_noise"]
    lam_src = d["lambda0"]

    cov, meas = s["covariate_shift"], s["observational_shift"]
    spu, haz = s["spurious_shift"], s["baseline_hazard_shift"]
    prior, con = s["prior_shift"], s["concept_shift"]

    mu_cov = (rf"$\lVert\Delta\mu_{{\mathrm{{inv}}}}\rVert{{=}}{_g(cov['mu_inv_shift_scale'])}$, "
              rf"$\sigma{{:}}\,{_g(sigma_src)}{{\to}}{_g(cov['sigma_noise_target'])}$")
    mu_meas = (rf"{mu_cov}, $R{{=}}{_g(meas['a_inv_rotation_angle_deg'])}^\circ$")
    flip = ", ".join(rf"w_{int(i) + 1}" for i in con["w_flip_indices"])

    return [
        ("No shift", "---", r"all $\Theta$", "---"),
        ("Covariate",
         r"$\mu_{\mathrm{inv}},\mu_{\mathrm{spu}},\Sigma$", r"$A,w,\lambda_0,\nu$",
         mu_cov),
        ("Measurement",
         r"$A_{\mathrm{inv}}\!\to\!A_{\mathrm{inv}}R$", r"$w,\lambda_0,\nu$",
         mu_meas),
        ("Spurious",
         r"$\Sigma_{\rho},\mu_{\mathrm{spu}}$", r"$w,A,\lambda_0,\sigma$",
         rf"$\rho{{=}}{_g(spu['rho_source'])}$, "
         rf"$\mu_{{\mathrm{{spu}}}}{{\to}}{_g(spu['spu_mean_target'])}$"),
        ("Baseline haz.",
         r"$\lambda_0$", r"$w,\nu,\beta_0,A,\mu$",
         rf"$\lambda_0{{:}}\,{_g(lam_src)}{{\to}}{_g(haz['lambda0_target'])}$"),
        ("Label-prior",
         r"$P(y)$ (resampling)", r"$P(x\mid y)$",
         rf"$\pi{{:}}\,{_g(SOURCE_PREVALENCE)}{{\to}}{_g(prior['prevalence_target'])}$"),
        ("Concept",
         r"$w$ (sign flip)", r"$A,\lambda_0,\nu,\mu$",
         rf"flip ${flip}$"),
    ]


def main() -> None:
    cfg = load_config()
    if not cfg:
        raise SystemExit("configs/generator.yaml not found — cannot build the table")

    base, dom = cfg["base"], cfg["domain_defaults"]

    caption = (
        r"Shift scenarios. Each perturbs one mechanism and holds the rest of "
        r"$\Theta$ fixed; the last column gives the value used. Common to all "
        rf"scenarios: $p{{=}}{base['p']}$ observed features, "
        rf"$k_{{\mathrm{{inv}}}}{{=}}{base['k_inv']}$ and "
        rf"$k_{{\mathrm{{spu}}}}{{=}}{base['k_spu']}$ latent factors, horizon "
        rf"$\tau{{=}}{_g(base['tau'])}$ months, $n{{=}}{dom['n']}$ patients per "
        rf"center, $w{{=}}({', '.join(_g(v) for v in dom['w'])})$, "
        rf"$\lambda_0{{=}}{_g(dom['lambda0'])}$, $\nu{{=}}{_g(dom['nu'])}$, "
        rf"giving a source event rate of ${_g(SOURCE_PREVALENCE)}$ at $\tau$. "
        r"Spurious factors never enter the risk ($v{=}0$ throughout). Values are "
        r"those in the configuration file released with the code."
    )

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        r"\label{tab:scenarios}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Changed} & \textbf{Held fixed} & \textbf{Parameters} \\",
        r"\midrule",
    ]
    for label, changed, fixed, params in rows(cfg):
        lines.append(f"{label} & {changed} & {fixed} & {params} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
