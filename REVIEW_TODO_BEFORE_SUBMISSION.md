# Review TODO — before submission

**Venue:** IEEE BigData 2026, Special Session on Healthcare Data
**Deadline:** 2026-08-29 (special session — NOT the 2026-08-21 main track; select the special-session track in CyberChair)
**Format:** IEEE 2-column (IEEEtran), **10 pages including references**
**Code reviewed:** 2026-07-16, commit `ac2d834` (branch `feat/public-multicentric-recovery`), local == origin verified.

---

## Bottom line of the code review

**No bugs found that invalidate any headline claim.** The pipeline is clean on the things that matter most — in particular there is **no train/test leakage**: target labels never enter any fit (IWCV uses domain labels + source-val; CORAL/DANN/CDAN use unlabeled target features only; calibration fits on source-val; oracle on true `u_inv` from source). See "Verified correct" at the bottom — those are citeable in a rebuttal.

The items below are refinements and disclosures, not correctness fixes. Safe to draft the paper now and clear these before submitting.

---

## A0. Supervisor's review list — status as of 2026-07-28

Drafts live in `paper_assets/snippet_*.tex`, ready to paste into Overleaf. They were
written against the 2026-07-20 local `.tex` snapshot, so **merge, don't overwrite** —
the Overleaf draft moved on until 2026-07-23 (Results, intro, abstract, figures).

- [x] **Config file for hardcoded parameters.** `configs/generator.yaml` is now actually
      wired in (`generator/config.py`); `tests/test_config.py` proves generation is
      bit-identical to before, so no result changes. Also fixed the `tau=6.0` hardcode
      in `_shape_shift` (was item C below).
- [x] **Model complexity.** → `paper_assets/snippet_complexity.tex` (end of Methods).
- [x] **Shift parameters in the setup.** → `paper_assets/table_scenarios.tex`, generated
      from the YAML by `_gen_table_scenarios.py`. Single-column; replaces the current
      `tab:scenarios`, swapping the redundant "Effect" column for the actual values.
- [x] **Metrics were never defined in the paper.** AUROC + ECE (10 equal-width bins) +
      the pre-specified 0.02 AUROC margin → `paper_assets/snippet_setup.tex`.
- [x] **Invariant oracle as a *learning* oracle** (same model class, same finite sample,
      conservative rather than information-theoretic) → same snippet.
- [x] **Clinically sound / bridge the gap** and **multimodal FM future work** →
      `paper_assets/snippet_discussion.tex`.
- [ ] **Figure 1**: check multiplication signs, replace concrete numbers with formal
      symbols. Blocked — the script that generated `fig1_generator.png` is not in the
      repo or on the Desktop.
- [ ] **Figures a/b**: letters below in parentheses, no per-panel titles, then single
      figures with their own captions. Needs the current `.tex` + the plotting scripts.
- [ ] **Shift described conceptually in Methods and operationally in Experiments.**
      Needs the current `.tex`.

## A1. Bibliography — five of six `[Verify]` entries were mis-cited (2026-07-28)

All six describe **real** papers, none fabricated, but the titles/venues/years were wrong.
Corrected entries in `paper_assets/snippet_bibliography_fixed.tex`; keys unchanged so no
`\cite` breaks. Worst offenders: `shaker2022multisource` (actually *Multi-Source Survival
Domain Adaptation*, AAAI 2023) and `li2023transfer` (no paper by the cited title —
confirm the intended work is Li, Shen & Ning, JASA 2023). Two page ranges still marked
`<CHECK>`; fill from the publisher, don't guess.

## A. Decisions already taken (this session)

- [x] **Cut the "few labelled target" analysis entirely.** The `ece_labeltgt` ceiling in `ece_recoverability.py` fits on 50% of the target (`train_test_split(test_size=0.5)`), which does not support the "a handful of labels" framing. Decision: remove the claim/analysis rather than soften or sweep it. → Remove from abstract, talk, and paper; the `ece_labeltgt` / `labeltgt_gain` columns can stay in the CSV but are not reported.
- [~] **`prior_shift` promoted to a full scenario.** Runs launched on the server (benchmark + n-sweep, 4 jobs). When the CSVs land: verify numbers reproduce, add `prior_shift` to fig1/2/3 scenario lists, regenerate figures.

---

## B. Code / analysis — clear before submission

- [ ] **Disclose meta-analysis transfer dependence.** DerSimonian-Laird treats transfers as independent, but i3lung shares INT as source in 5/7 transfers and Rotterdam pools correlated τ × directions, so pooled CIs are mildly too tight. Effect sizes are so small the "no clinically meaningful gain" conclusion is unaffected — but state the dependence (one primary τ + others as sensitivity, or a cluster/direction-level note). *(meta_analysis.py; also in project notes.)*
- [ ] **Fix figure/slide captions: synthetic = paired t-test over seeds, real = patient-level bootstrap.** Both are correct in code; make each caption name the right test for its panel. The old "t-test" caption on the real-data panel is stale.
- [ ] **(Optional, strengthens honesty) across-scenario multiplicity.** fig3 Bonferroni-corrects the best-ECE-method selection *within* a scenario, but not across the 5–6 scenarios. Correcting across scenarios only makes "significant only under covariate shift" stricter, so the claim is safe either way — mention the correction family explicitly.

## C. Cosmetic / low priority (won't block, fix if touched)

- [ ] `_shape_shift` hardcodes `tau = 6.0` (scenarios.py:417) for the λ₀ prevalence match instead of `base.tau`. **Not in the paper** (no figure/table uses shape_shift), so cosmetic — fix only if shape_shift is ever surfaced or run at another τ.
- [ ] Stale docstring: `_build_networks` says CDAN task = `Dense(2, softmax)`; code uses `task_softmax=False` (sigmoid 1-D). Code is right; fix the doc.
- [ ] Framing precision: `covariate_shift` also bumps `sigma_noise` 0.5→0.8 (invariance holds at the latent `P(Y|u)` level, not textbook `P(Y|x)`); `measurement`/`spurious` each change two parameters (disclosed in the poster's "held fixed" column). Keep the paper text at latent-level framing to stay accurate.

## D. Paper text / figures — from earlier review (project notes)

- [ ] **Broken LaTeX refs**: "Figure ??" (×4 in Results), "CDAN [??]", one "?" citation in intro, "Table ??" for real data. Figures fig1–4 exist in `paper_assets/` but are not embedded in the compiled draft — embed them and fix the refs.
- [ ] **Confirm Overleaf is IEEEtran 2-column.** The Downloads draft was single-column NeurIPS-style — not IEEE. Convert before writing too much (changes how much fits in 10 pages).
- [ ] **K=3 vs 1-source contradiction**: §3.3 says "K=3 source centers", contradicting §4 ("one labeled source center"). Actual design is **1 source + 1 target, n=500, 10 seeds**. Fix §3.3.
- [ ] **10-page limit includes references** — budget accordingly (≈9 pages of text with the current ~8 refs).
- [ ] Related work pass (Undermind).

---

## Verified correct (citeable in rebuttal)

- **No target-label leakage** anywhere; IWCV is honest (domain-label classifier + importance-weighted source-val AUROC) — pre-empts "undertuned/leaky".
- **CORAL applied in the correct direction** (estimator trained on source recolored to target covariance; target predicted directly) → "CORAL doesn't recover discrimination" is real, not an implementation artifact.
- **Oracle is a fair upper bound** (same train/eval protocol as ERM, true `u_inv`); the concept-shift oracle collapse to ~0.61 is a genuine internal-validity check (cosine(w_src, w_tgt)=1/3).
- **Real-data τ binarization correct**: censored-before-τ patients are dropped, not mislabelled (`keep = ~((t<tau)&(d==0))`).
- **IRMv1 gradient correct** (penalty `g_e²`, verified by hand — the earlier naive `‖∇R‖²` bug is gone; supports "IRM ≤ ERM" citing Gulrajani & Lopez-Paz).
- **Weibull inverse-CDF sampling correct**; `y(τ)=1[T≤τ]` reduces cleanly under `censoring_lambda=0`.
- **DerSimonian-Laird correct**; **paired stratified bootstrap correct**; **paired comparison valid** (scenario generated once per seed, shared across all methods).
- **GroupDRO correct and hardened** (source-val early stopping, no target labels).

## Not reviewed (scope)

`generator/fitting.py` (Paper 2 / few-surv, not this paper), data registries/loaders, `run_survival.py`, `utils/plotting.py`, `decision_geometry.py` beyond the angle/covdist core (which is correct — `β_tgt*` uses target labels only as a diagnostic reference, not to fit any compared method). Caveat: reviewed the local repo (== origin); if the server has uncommitted `.py` changes, re-check those.
