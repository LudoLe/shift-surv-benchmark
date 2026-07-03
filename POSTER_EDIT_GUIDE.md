# Poster edit guide — ESAC-26 (box-by-box, for BioRender)

Goal: align the poster with the current paper (discrimination/calibration split + real-data
I3LUNG validation) and swap in the four polished figures from `paper_assets/`.
Ready-made PNGs: `paper_assets/fig1_divergence.png`, `fig2_headroom.png`,
`fig3_ece_headroom.png`, `fig4_realdata.png` (vector `.pdf` versions also exist — better for
large-format print if BioRender accepts them).

Proposed layout: LEFT column = Introduction · Methods · Conclusion; RIGHT column = Results
(four figures in a 2×2 grid). Header and footer unchanged except the title.

> NOTE — IRM has been dropped. The current paper results table
> (`benchmark_1s1t_iwcv_compact.csv`) contains only: inv_oracle, lr_erm, coral, prior_recalib,
> platt, dann, cdan. IRM (and iw_erm / isotonic) are **not** in the paper figures. Remove IRM
> everywhere on the poster (text + figure legends).

---

## 0. HEADER (title)
REPLACE the title with:
> **SHIFT-SURV: A Synthetic Multi-Centre Generator for Evaluating Unsupervised Domain
> Adaptation in Fixed-Horizon Cancer Survival Risk**

Authors / affiliations: unchanged.

---

## 1. INTRODUCTION (left column, box 1) — rewrite the bullets
- Risk models trained at one site **degrade** when transferred elsewhere (case mix,
  measurement protocols, care pathways, baseline risk).¹˒²
- SHIFT-SURV is a synthetic **Weibull proportional-hazards** multi-centre generator that
  separates **invariant, spurious, and noise** components.
- **Question:** does unsupervised domain adaptation actually recover transportable risk under
  multi-centre shift — and how would we know?
- Test-bed for ERM, CORAL, DANN, CDAN **and post-hoc calibration** (prior recalibration,
  Platt scaling).³⁻⁶

Key edits: "Cox–Weibull" → "Weibull proportional-hazards"; **removed IRM**; added post-hoc
calibration; added the research question.

---

## 2. METHODS (left column, box 2)

### 2a. Synthetic Generator — equations stay (they are correct). Keep:
u_inv ~ N(μ_inv, I), u_spu ~ N(μ_spu, I) · x = A_inv u_inv + A_spu u_spu + ε ·
η = β0 + wᵀu_inv + vᵀu_spu (v=0 in base regime) · T = (−log U /(λ0 e^η))^(1/ν) · y(τ)=1[T≤τ].
ADD one line with dimensions and horizon:
> k_inv = 3, k_spu = 2, p = 20 observed; **τ = 6 months**; 1 source + 1 target, 10 seeds.

### 2b. Domain Adaptation Methods — REPLACE the list (drop IRM, add calibration)
- **Invariant oracle** — LR on true u_inv (upper bound)
- **LR-ERM** — pooled-source logistic regression
- **CORAL** — covariance alignment
- **DANN / CDAN** — (conditional) domain-adversarial networks, IWCV HP tuning
- **Prior recalibration** — Saerens EM (prior-shift, no target labels)
- **Platt scaling** — post-hoc sigmoid calibration

### 2c. REPLACE the three shift schematics with the 7-SCENARIO TABLE
(covers every shift in little space; the current covariate/hazard/concept schematics only
show three):

| Scenario | What shifts | Expected effect |
|---|---|---|
| No shift | nothing | baseline, gap ≈ 0 |
| Covariate | latent means P(X) | AUROC stable |
| Measurement | A_inv rotated 60° | AUROC drops; DA partially recovers |
| Spurious | latent cross-covariance | spurious features unreliable |
| Baseline hazard | λ0 | AUROC stable, ECE degrades |
| Concept | partial sign-flip of w | AUROC drops, no recovery |
| Shape | Weibull shape ν | calibration degrades, ranking preserved |

---

## 3. RESULTS (right column) — replace bullets + all five current figures

### 3a. Lead-in (3 bullets above the figures), REPLACE the current ones:
- **Discrimination and calibration diverge**: each shift induces a different mix of AUROC loss
  and recoverable calibration error (Fig. 1).
- **No practical AUROC recovery**: best-DA ≈ ERM everywhere (paired t-test ns), worse on
  Spurious; large headroom to the oracle left uncaptured (Fig. 2).
- **Calibration partially/selectively recoverable**: after multiplicity correction it improves
  significantly only on **Covariate (**)**; the earlier Measurement effect does not survive
  correction (Fig. 3).

### 3b. Figures — REMOVE current Fig.1–5; INSERT in a 2×2 grid:

**Fig. 1 ← `paper_assets/fig1_divergence.png`**
Caption: *Discrimination vs calibration (ERM only). Each point = one scenario; axes = AUROC gap
(source−target) and ECE gap (ERM−oracle), 95% CI over 10 seeds. The two axes degrade
independently.*

**Fig. 2 ← `paper_assets/fig2_headroom.png`**
Caption: *Target AUROC per scenario: oracle (grey = ceiling), ERM, best-DA (always CORAL);
stars = paired t-test vs ERM, Bonferroni-corrected for best-DA selection. Best-DA ≈ ERM
everywhere (ns); only Spurious is significant (*), where DA is slightly worse than ERM.*

> NOTE — all significance stars in Fig. 2/3 (and Fig. 4) are now Bonferroni-corrected for the
> post-hoc "best-DA" selection (×3 for AUROC/real, ×5 for synthetic ECE). Net effect: AUROC
> headline unchanged; synthetic Spurious **→*; synthetic calibration recovery now Covariate-only
> (Measurement drops out); real-data VHIO→INT ECE *** is robust. Update talk slide 10 ("Covariate
> **, Measurement *") accordingly.

**Fig. 3 ← `paper_assets/fig3_ece_headroom.png`**
Caption: *Target ECE per scenario (↓ better): oracle (floor), ERM, best-calibration-DA; stars =
paired t-test vs ERM, Bonferroni-corrected for best-DA selection. Calibration recovers
significantly only on Covariate (**); other shifts are ns after correction.*

**Fig. 4 ← `paper_assets/fig4_realdata.png`** (most important — currently missing)
Caption (CORRECTED — the figure already reflects the patient-level bootstrap; the old
"worse on 4/7 / improves on several" wording came from the superseded per-seed t-test):
*Real-data validation, I3LUNG, 7 transfers. Point = best-DA vs ERM (patient-level bootstrap,
95% CI; stars = bootstrap p vs ERM; line at 0 = no change). (a) AUROC: best-DA is statistically
indistinguishable from ERM on all 7 transfers (no significant gain or loss). (b) ECE: DA
significantly improves calibration only on VHIO→INT (***); elsewhere ns. The direction matches
the synthetic split (no discrimination gain; occasional calibration help), but cohort-level
uncertainty leaves almost all gaps non-significant.*

> ⚠️ Also fix talk slide 11 and Conclusion accordingly: replace "significantly worse on 4/7"
> → "no significant difference on any of 7"; "improves on several transfers" → "improves
> significantly on 1/7 (VHIO→INT)".

(Optional "Robustness" corner: `outputs/figures/n_sweep_auroc_vs_n.png` — best-DA−ERM from
n=250 to 2000.)

---

## 4. CONCLUSION (left column, box 3) — replace entirely
- In the small-sample, multi-centre regime, unsupervised DA **does not recover AUROC** over
  ERM, despite large oracle headroom.
- Discrimination and calibration **degrade independently**: a stable AUROC can **mask
  substantial calibration drift**.
- On **7 real I3LUNG transfers** the direction is consistent: DA is **indistinguishable from
  ERM in discrimination** (no transfer significant) and improves calibration significantly on
  **only 1/7** (VHIO→INT).
- SHIFT-SURV is useful as a **controlled benchmark for attributing transport failure to
  specific shift mechanisms**, not as evidence that any single DA method "wins".

---

## 5. FOOTER (references)
- **Remove** the IRM reference (Arjovsky et al., 2019) — IRM is no longer a method.
- **Add** the I3LUNG citation (real data now appears on the poster):
  > Prelaj A, et al. The EU-funded I3LUNG Project… Clinical Lung Cancer, 2023.
  > ClinicalTrials.gov: NCT05537922.

---

## Quick checklist
- [ ] Title updated
- [ ] "Cox–Weibull" → "Weibull proportional-hazards"
- [ ] IRM removed (Introduction, Methods, figure legends, references)
- [ ] Methods: oracle, LR-ERM, CORAL, DANN, CDAN, prior recalib, Platt; 7-scenario table
- [ ] Old Fig.1–5 removed (including the "12 months" one)
- [ ] fig1–4 from paper_assets inserted in a 2×2 grid with captions
- [ ] Conclusion rewritten (split + I3LUNG)
- [ ] I3LUNG reference added, IRM reference removed
