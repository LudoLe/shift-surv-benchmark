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
# Table 3 (AUROC): Prior-EM and Platt omitted — they are rank-preserving
# transformations of ERM scores, so AUROC is mathematically identical to ERM.
METH_AUROC = ["invariant_oracle", "all_observed", "coral", "dann", "cdan"]
# Table 4 (ECE): all methods included — calibration baselines are informative here.
METH_ECE   = ["invariant_oracle", "all_observed", "coral", "prior_recalib",
              "platt_scaling", "dann", "cdan"]
METH_LBL = {
    "invariant_oracle": "Oracle", "all_observed": "ERM", "coral": "CORAL",
    "prior_recalib": "Prior-EM", "platt_scaling": "Platt",
    "dann": "DANN", "cdan": "CDAN",
}


from scipy import stats as _stats


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r.get("scenario", ""), r["method"])
            d[key] = r
    return d


def load_raw_synth(path):
    """Returns {(scenario, method): [per-seed metric values]}  for tgt_auroc and tgt_ece."""
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            sc, m = r.get("scenario", ""), r["method"]
            for metric in ("tgt_auroc", "tgt_ece"):
                key = (sc, m, metric)
                d.setdefault(key, [])
                try:
                    d[key].append(float(r[metric]))
                except (ValueError, KeyError):
                    pass
    return d


def _pstars(v_erm, v_da):
    """Paired t-test (across seeds) vs ERM; empty string if not significant."""
    n = min(len(v_erm), len(v_da))
    if n < 2:
        return ""
    _, p = _stats.ttest_rel(v_erm[:n], v_da[:n])
    if p <= 0.001: return "$^{***}$"
    if p <= 0.01:  return "$^{**}$"
    if p <= 0.05:  return "$^{*}$"
    return ""


def fnum(r, k):
    return float(r[k]) if r and r.get(k, "") not in ("", "nan") else float("nan")


def _cell(r, base):
    """Format mean±ci for a metric base name, e.g. 'tgt_auroc'."""
    m = fnum(r, base + "_mean"); c = fnum(r, base + "_ci95")
    if np.isnan(m):
        return "--"
    return f"{m:.3f}\\,$\\pm$\\,{c:.3f}"


def make_synth_table(A, R, metric, caption, label, bold_min=False, meth=None):
    """scenario x method table for a synthetic metric (tgt_auroc / tgt_ece).
    R: raw per-seed dict from load_raw_synth; None to skip significance stars.
    """
    if meth is None:
        meth = METH_ECE
    lines = []
    lines.append("\\begin{table*}[t]\\centering\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(meth) + "}")
    lines.append("\\toprule")
    lines.append("Scenario & " + " & ".join(METH_LBL[m] for m in meth) + " \\\\")
    lines.append("\\midrule")
    for sc in SCEN:
        cells = []
        vals = {m: fnum(A.get((sc, m)), metric + "_mean")
                for m in meth if m != "invariant_oracle"}
        finite = {m: v for m, v in vals.items() if not np.isnan(v)}
        if finite:
            best = min(finite, key=finite.__getitem__) if bold_min \
                   else max(finite, key=finite.__getitem__)
        else:
            best = None
        erm_seeds = R.get((sc, "all_observed", metric), []) if R else []
        for m in meth:
            txt = _cell(A.get((sc, m)), metric)
            if R and m not in ("invariant_oracle", "all_observed"):
                da_seeds = R.get((sc, m, metric), [])
                txt += _pstars(erm_seeds, da_seeds)
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
    R = load_raw_synth(os.path.join(TAB, "benchmark_1s1t_iwcv_raw.csv"))

    STAR_NOTE = (" Superscripts on non-ERM cells: $^{*}p{<}0.05$,"
                 " $^{**}p{<}0.01$, $^{***}p{<}0.001$"
                 " (paired $t$-test vs ERM across seeds).")

    # Table 3: AUROC — Prior-EM and Platt excluded (rank-preserving → AUROC ≡ ERM)
    t3 = make_synth_table(
        A, R, "tgt_auroc",
        "Target AUROC (mean\\,$\\pm$\\,95\\% CI over 10 seeds) on the synthetic benchmark. "
        "Prior-EM and Platt scaling are omitted: as rank-preserving transformations of ERM "
        "scores their AUROC is identical to ERM by construction. "
        "Best non-oracle per row in \\textbf{bold}." + STAR_NOTE,
        "tab:auroc", meth=METH_AUROC)
    open(os.path.join(OUT, "table3_auroc.tex"), "w").write(t3 + "\n")

    # Table 4: ECE — all methods included (calibration baselines are informative here)
    t4 = make_synth_table(
        A, R, "tgt_ece",
        "Target ECE (mean\\,$\\pm$\\,95\\% CI over 10 seeds; lower is better). "
        "Best non-oracle per row in \\textbf{bold}. Calibration degrades independently "
        "of discrimination: baseline-hazard shift leaves AUROC near-intact yet yields the worst ECE."
        + STAR_NOTE,
        "tab:ece", bold_min=True, meth=METH_ECE)
    open(os.path.join(OUT, "table4_ece.tex"), "w").write(t4 + "\n")

    print("wrote table3_auroc.tex, table4_ece.tex")

    # ---- Tables 5a/5b: real I3LUNG transfers split into AUROC and ECE ----

    def _ci(vals):
        """Return (mean, ci95_half) from a list of per-seed values."""
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2:
            return (vals[0] if vals else np.nan), 0.0
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
        t  = float(_stats.t.ppf(0.975, df=len(vals) - 1))
        return m, t * se

    def _fmt(m, ci):
        if np.isnan(m):
            return "--"
        return f"{m:.3f}\\,{{\\small$\\pm${ci:.3f}}}"

    def _pstars(v_erm, v_da):
        """Paired t-test stars vs ERM; empty string if not significant."""
        v_erm = [v for v in v_erm if not np.isnan(v)]
        v_da  = [v for v in v_da  if not np.isnan(v)]
        n = min(len(v_erm), len(v_da))
        if n < 2:
            return ""
        _, p = _stats.ttest_rel(v_erm[:n], v_da[:n])
        if p <= 0.001: return "$^{***}$"
        if p <= 0.01:  return "$^{**}$"
        if p <= 0.05:  return "$^{*}$"
        return ""

    raw_files = sorted(f for f in glob.glob(os.path.join(TAB, "i3lung_*_*_raw.csv"))
                       if "i3lung_cb" not in os.path.basename(f))

    METHS    = ["lr_erm", "coral", "dann", "cdan"]
    METH_HDR = ["ERM", "CORAL", "DANN", "CDAN"]

    # parse all transfer data once
    transfers = []
    for f in raw_files:
        name = os.path.basename(f).replace("i3lung_", "").replace("_raw.csv", "")
        src, tgt = name.split("_")
        vals = {}
        for r in csv.DictReader(open(f)):
            m = r["method"]
            vals.setdefault(m, {"tgt_auroc": [], "tgt_ece": []})
            for k in ("tgt_auroc", "tgt_ece"):
                try:
                    vals[m][k].append(float(r[k]))
                except (ValueError, KeyError):
                    pass
        transfers.append((src, tgt, vals))

    # Bootstrap-based real tables (cohort-level CI + p-values), aligned with
    # fig4. Reads i3lung_*_*_bootstrap.csv: point estimate + 2.5/97.5 percentile
    # CI over target-patient resampling, and the bootstrap p-value of the gap vs
    # ERM. Supersedes the per-seed t-test, which reflected only the train/val
    # split rather than target-cohort sampling.
    boot_files = sorted(f for f in glob.glob(os.path.join(TAB, "i3lung_*_*_bootstrap.csv"))
                        if "i3lung_cb" not in os.path.basename(f))
    boot_transfers = []
    for f in boot_files:
        name = os.path.basename(f).replace("i3lung_", "").replace("_bootstrap.csv", "")
        src, tgt = name.split("_")
        bvals = {}
        for r in csv.DictReader(open(f)):
            def g(k):
                try:
                    return float(r[k])
                except (ValueError, KeyError):
                    return np.nan
            bvals[r["method"]] = {
                "auroc": g("auroc"), "auroc_lo": g("auroc_lo"), "auroc_hi": g("auroc_hi"),
                "auroc_p": g("auroc_gap_p"),
                "ece": g("ece"), "ece_lo": g("ece_lo"), "ece_hi": g("ece_hi"),
                "ece_p": g("ece_gap_p"),
            }
        boot_transfers.append((src, tgt, bvals))

    def _bstars(p):
        if np.isnan(p): return ""
        if p <= 0.001:  return "$^{***}$"
        if p <= 0.01:   return "$^{**}$"
        if p <= 0.05:   return "$^{*}$"
        return ""

    def _bfmt(m, lo, hi):
        if np.isnan(m):
            return "--"
        return f"{m:.3f}\\,{{\\small[{lo:.3f},\\,{hi:.3f}]}}"

    def build_real_table(metric, bold_min, caption, label, fname):
        """metric in {'auroc','ece'}; uses bootstrap point estimate + CI + gap p."""
        rl = []
        rl.append("\\begin{table*}[t]\\centering\\footnotesize")
        rl.append("\\setlength{\\tabcolsep}{4pt}")
        rl.append("\\begin{tabular}{l " + "c" * len(METHS) + "}")
        rl.append("\\toprule")
        rl.append("Transfer & " + " & ".join(METH_HDR) + " \\\\")
        rl.append("\\midrule")

        best_deltas = []
        for src, tgt, bvals in boot_transfers:
            means = {m: bvals.get(m, {}).get(metric, np.nan) for m in METHS}
            if bold_min:
                best_m = min(means, key=lambda k: means[k]
                             if not np.isnan(means[k]) else np.inf)
            else:
                best_m = max(means, key=lambda k: means[k]
                             if not np.isnan(means[k]) else -np.inf)

            cells = []
            for m in METHS:
                b = bvals.get(m, {})
                txt = _bfmt(b.get(metric, np.nan),
                            b.get(f"{metric}_lo", np.nan), b.get(f"{metric}_hi", np.nan))
                if m != "lr_erm":
                    txt += _bstars(b.get(f"{metric}_p", np.nan))
                if m == best_m:
                    txt = "\\textbf{" + txt + "}"
                cells.append(txt)
            rl.append(f"{src}$\\to${tgt} & " + " & ".join(cells) + " \\\\")

            if not bold_min:
                da_means = [means[m] for m in ("coral", "dann", "cdan")]
                best_deltas.append(float(np.nanmax(da_means)) - means["lr_erm"])

        rl.append("\\bottomrule")
        rl.append("\\end{tabular}")
        if best_deltas:
            extra = (f" Best-DA$-$ERM: mean {np.nanmean(best_deltas):+.3f}, "
                     f"range [{np.nanmin(best_deltas):+.3f}, {np.nanmax(best_deltas):+.3f}].")
        else:
            extra = ""
        rl.append(f"\\caption{{{caption}{extra}"
                  " Superscripts on non-ERM cells: $^{{*}}p{<}0.05$,"
                  " $^{{**}}p{<}0.01$, $^{{***}}p{<}0.001$ (two-sided bootstrap,"
                  " gap vs ERM over target-patient resampling).}}")
        rl.append(f"\\label{{{label}}}")
        rl.append("\\end{table*}")
        open(os.path.join(OUT, fname), "w").write("\n".join(rl) + "\n")
        print(f"wrote {fname}")

    build_real_table(
        "auroc", bold_min=False,
        caption=("Real multi-center transfers on I3LUNG, target AUROC $\\uparrow$"
                 " (point estimate, 95\\% bootstrap CI [2.5, 97.5] over target patients;"
                 " best per row in \\textbf{bold})."),
        label="tab:real_auroc",
        fname="table5a_real_auroc.tex")

    build_real_table(
        "ece", bold_min=True,
        caption=("Real multi-center transfers on I3LUNG, target ECE $\\downarrow$"
                 " (point estimate, 95\\% bootstrap CI [2.5, 97.5] over target patients;"
                 " best per row in \\textbf{bold})."),
        label="tab:real_ece",
        fname="table5b_real_ece.tex")

    # keep legacy combined table for backward compatibility
    rl = []
    rl.append("\\begin{table*}[t]\\centering\\footnotesize")
    rl.append("\\setlength{\\tabcolsep}{3pt}")
    rl.append("\\begin{tabular}{l cccc cccc}")
    rl.append("\\toprule")
    rl.append("& \\multicolumn{4}{c}{Target AUROC $\\uparrow$}"
              " & \\multicolumn{4}{c}{Target ECE $\\downarrow$} \\\\")
    rl.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
    rl.append("Transfer & " + " & ".join(METH_HDR) +
              " & " + " & ".join(METH_HDR) + " \\\\")
    rl.append("\\midrule")
    best_deltas = []
    for src, tgt, vals in transfers:
        get_ci = lambda m, metric: _ci(vals.get(m, {}).get(metric, []))
        auroc_means = {m: get_ci(m, "tgt_auroc")[0] for m in METHS}
        ece_means   = {m: get_ci(m, "tgt_ece")[0]   for m in METHS}
        best_auroc  = max(auroc_means, key=lambda k: auroc_means[k]
                          if not np.isnan(auroc_means[k]) else -np.inf)
        best_ece    = min(ece_means,   key=lambda k: ece_means[k]
                          if not np.isnan(ece_means[k])   else np.inf)
        def cell(m, metric, best_m):
            mv, mci = get_ci(m, metric)
            txt = _fmt(mv, mci)
            if m == best_m:
                txt = "\\textbf{" + txt + "}"
            return txt
        auroc_cells = " & ".join(cell(m, "tgt_auroc", best_auroc) for m in METHS)
        ece_cells   = " & ".join(cell(m, "tgt_ece",   best_ece)   for m in METHS)
        rl.append(f"{src}$\\to${tgt} & {auroc_cells} & {ece_cells} \\\\")
        da_means = [auroc_means[m] for m in ("coral", "dann", "cdan")]
        best_deltas.append(float(np.nanmax(da_means)) - auroc_means["lr_erm"])
    rl.append("\\bottomrule")
    rl.append("\\end{tabular}")
    rl.append("\\caption{Real multi-center transfers on I3LUNG (6 centers, "
              "mean\\,$\\pm$\\,95\\% CI over 10 seeds; best per row in "
              "\\textbf{bold}). "
              "No adaptation method exceeds ERM in target AUROC across all "
              f"seven transfers (best-DA$-$ERM: mean {np.nanmean(best_deltas):+.3f}, "
              f"range [{np.nanmin(best_deltas):+.3f}, {np.nanmax(best_deltas):+.3f}]). "
              "DANN and CDAN achieve lower ECE than ERM on several transfers, "
              "consistent with the synthetic finding that adversarial alignment "
              "can improve calibration even when discrimination is not recovered.}}")
    rl.append("\\label{tab:real}")
    rl.append("\\end{table*}")
    open(os.path.join(OUT, "table5_real.tex"), "w").write("\n".join(rl) + "\n")
    print("wrote table5_real.tex (with CI)")


if False:  # dead code — figures are generated by make_figures.py
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



