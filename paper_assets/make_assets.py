"""
Generate all paper artifacts (LaTeX tables + figures) from the result CSVs.

Run from repo root:  python paper_assets/make_assets.py

Outputs (paper_assets/):
  table3_auroc.tex   — synthetic target AUROC, scenario x method, mean±CI
  table4_ece.tex     — synthetic target ECE,   scenario x method, mean±CI
  table5_real.tex    — 7 I3LUNG transfers, ERM/bestDA/ECE
  fig1_divergence.pdf/.png — AUROC gap vs ECE gap per scenario
  fig2_headroom.pdf/.png   — ERM vs best-DA vs oracle AUROC per scenario
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(ROOT, "outputs", "tables")
OUT  = os.path.join(ROOT, "paper_assets")

SCEN = ["no_shift", "covariate_shift", "observational_shift", "spurious_shift",
        "baseline_hazard_shift", "concept_shift", "shape_shift"]
SCEN_LBL = {
    "no_shift": "No shift", "covariate_shift": "Covariate",
    "observational_shift": "Measurement", "spurious_shift": "Spurious",
    "baseline_hazard_shift": "Baseline hazard", "concept_shift": "Concept",
    "shape_shift": "Shape",
}
METH = ["invariant_oracle", "all_observed", "coral", "prior_recalib",
        "platt_scaling", "dann", "cdan"]
METH_LBL = {
    "invariant_oracle": "Oracle", "all_observed": "ERM", "coral": "CORAL",
    "prior_recalib": "Prior-EM", "platt_scaling": "Platt",
    "dann": "DANN", "cdan": "CDAN",
}


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r.get("scenario", ""), r["method"])
            d[key] = r
    return d


def fnum(r, k):
    return float(r[k]) if r and r.get(k, "") not in ("", "nan") else float("nan")


def _cell(r, base):
    """Format mean±ci for a metric base name, e.g. 'tgt_auroc'."""
    m = fnum(r, base + "_mean"); c = fnum(r, base + "_ci95")
    if np.isnan(m):
        return "--"
    return f"{m:.3f}\\,$\\pm$\\,{c:.3f}"


def make_synth_table(A, metric, caption, label, bold_min=False):
    """scenario x method table for a synthetic metric (tgt_auroc / tgt_ece)."""
    lines = []
    lines.append("\\begin{table*}[t]\\centering\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(METH) + "}")
    lines.append("\\toprule")
    lines.append("Scenario & " + " & ".join(METH_LBL[m] for m in METH) + " \\\\")
    lines.append("\\midrule")
    for sc in SCEN:
        cells = []
        # bold: highest AUROC (bold_min=False) or lowest ECE (bold_min=True),
        # excluding oracle and any NaN entries.
        vals = {m: fnum(A.get((sc, m)), metric + "_mean")
                for m in METH if m != "invariant_oracle"}
        finite = {m: v for m, v in vals.items() if not np.isnan(v)}
        if finite:
            best = min(finite, key=finite.__getitem__) if bold_min \
                   else max(finite, key=finite.__getitem__)
        else:
            best = None
        for m in METH:
            txt = _cell(A.get((sc, m)), metric)
            if best and m == best:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(SCEN_LBL[sc] + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    A = load(os.path.join(TAB, "benchmark_1s1t_iwcv_summary.csv"))

    t3 = make_synth_table(
        A, "tgt_auroc",
        "Target AUROC (mean\\,$\\pm$\\,95\\% CI over 10 seeds) on the synthetic benchmark. "
        "Best non-oracle method per row in \\textbf{bold}. No adaptation method exceeds ERM "
        "on any shift, despite large oracle headroom.",
        "tab:auroc")
    open(os.path.join(OUT, "table3_auroc.tex"), "w").write(t3 + "\n")

    t4 = make_synth_table(
        A, "tgt_ece",
        "Target ECE (mean\\,$\\pm$\\,95\\% CI over 10 seeds; lower is better). "
        "Best non-oracle method per row in \\textbf{bold}. Calibration degrades independently "
        "of discrimination: baseline-hazard shift leaves AUROC near-intact yet yields the worst ECE.",
        "tab:ece", bold_min=True)
    open(os.path.join(OUT, "table4_ece.tex"), "w").write(t4 + "\n")

    print("wrote table3_auroc.tex, table4_ece.tex")

    # ---- Table 5: real I3LUNG transfers ----
    files = sorted(f for f in glob.glob(os.path.join(TAB, "i3lung_*_*_summary.csv"))
                   if "i3lung_cb" not in os.path.basename(f))
    rl = []
    rl.append("\\begin{table}[t]\\centering\\small")
    rl.append("\\setlength{\\tabcolsep}{4pt}")
    rl.append("\\begin{tabular}{lccccc}")
    rl.append("\\toprule")
    rl.append("Transfer & ERM & CORAL & DANN & CDAN & ERM ECE \\\\")
    rl.append("\\midrule")
    best_deltas = []
    for f in files:
        name = os.path.basename(f).replace("i3lung_", "").replace("_summary.csv", "")
        src, tgt = name.split("_")
        R = {r["method"]: r for r in csv.DictReader(open(f))}
        erm = fnum(R.get("lr_erm"), "tgt_auroc_mean")
        das = [fnum(R.get(m), "tgt_auroc_mean") for m in ("coral", "dann", "cdan")]
        best_deltas.append(float(np.nanmax(das)) - erm)
        row = f"{src}$\\to${tgt} & {erm:.3f}"
        for v in das:
            row += f" & {v:.3f}"
        row += f" & {fnum(R.get('lr_erm'), 'tgt_ece_mean'):.3f} \\\\"
        rl.append(row)
    deltas = best_deltas
    rl.append("\\bottomrule")
    rl.append("\\end{tabular}")
    rl.append("\\caption{Real multi-center transfers on I3LUNG (6 centers, mean over 10 seeds). "
              "Across all seven source$\\to$target pairs, even the best adaptation method per "
              f"transfer does not exceed ERM target AUROC (best-DA\\,$-$\\,ERM: mean {np.nanmean(deltas):+.3f}, "
              f"range [{np.nanmin(deltas):+.3f}, {np.nanmax(deltas):+.3f}]), reproducing the synthetic finding.}}")
    rl.append("\\label{tab:real}")
    rl.append("\\end{table}")
    open(os.path.join(OUT, "table5_real.tex"), "w").write("\n".join(rl) + "\n")
    print("wrote table5_real.tex")

    # ---- Fig 1: discrimination-calibration divergence ----
    # x = AUROC gap (ERM source-pooled vs target), y = ECE gap (target - oracle)
    scn = [s for s in SCEN if s != "no_shift"]
    auroc_gap, tgt_ece, labels = [], [], []
    for sc in scn:
        erm = A[(sc, "all_observed")]
        auroc_gap.append(fnum(erm, "gap_mean"))      # source-target AUROC drop (ERM)
        tgt_ece.append(fnum(erm, "tgt_ece_mean"))    # absolute target calibration error
        labels.append(SCEN_LBL[sc])
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(auroc_gap, tgt_ece, s=70, color="steelblue", zorder=3)
    for x, y, l in zip(auroc_gap, tgt_ece, labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("AUROC gap (source $-$ target)")
    ax.set_ylabel("Target ECE")
    ax.set_title("Discrimination and calibration degrade independently")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_divergence.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_divergence.png"), dpi=150)
    plt.close(fig)
    print("wrote fig1_divergence.{pdf,png}")

    # ---- Fig 2: oracle headroom (ERM vs best-DA vs oracle) ----
    erm_v, bestda_v, orc_v = [], [], []
    for sc in SCEN:
        erm_v.append(fnum(A[(sc, "all_observed")], "tgt_auroc_mean"))
        orc_v.append(fnum(A[(sc, "invariant_oracle")], "tgt_auroc_mean"))
        bestda_v.append(max(fnum(A[(sc, m)], "tgt_auroc_mean") for m in ("coral", "dann", "cdan")))
    x = np.arange(len(SCEN)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(x - w, erm_v,    w, label="ERM",      color="steelblue")
    ax.bar(x,     bestda_v, w, label="best DA",  color="seagreen")
    ax.bar(x + w, orc_v,    w, label="Oracle",   color="black")
    ax.set_xticks(x); ax.set_xticklabels([SCEN_LBL[s] for s in SCEN], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Target AUROC"); ax.set_ylim(0.5, 0.88)
    ax.set_title("Recoverable headroom (Oracle) is not captured by any adaptation method")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_headroom.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_headroom.png"), dpi=150)
    plt.close(fig)
    print("wrote fig2_headroom.{pdf,png}")



