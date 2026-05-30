# SHIFT-SURV: A Synthetic Benchmark for Distribution Shift in Survival Analysis

Synthetic benchmark for studying how covariate shift, concept shift, baseline hazard shift,
and spurious correlations affect survival models transported across clinical sites.

## Setup

```bash
python -m venv shift-surv-env
source shift-surv-env/bin/activate
pip install -r requirements.txt
```

## Run the benchmark

```bash
# Main benchmark (7 core scenarios × 8 methods × 10 seeds)
python -m experiments.run_adaptation

# Include appendix scenarios (strong_covariate_shift, cure_shift)
python -m experiments.run_adaptation --include_appendix --out_prefix benchmark_1s1t_full

# IWCV ablation — covariate_shift without importance-weighted HP tuning
python -m experiments.run_adaptation --scenarios covariate_shift --no_iwcv \
    --out_prefix ablation_no_iwcv
```

Outputs are written to `outputs/tables/<prefix>_raw/summary/compact/hp_sweep.csv`.

## Scenarios

| Scenario | What shifts | Expected effect |
|---|---|---|
| `no_shift` | nothing | baseline, gap ≈ 0 |
| `covariate_shift` | P(X): latent means μ only | AUROC stable (P(Y\|u_inv) fixed) |
| `observational_shift` | P(X) + A_inv rotated 60° | AUROC drops, DA methods recover partially |
| `spurious_shift` | latent cross-covariance | spurious features become unreliable |
| `baseline_hazard_shift` | λ₀ only | AUROC stable, ECE degrades; recalibration helps |
| `concept_shift` | w_d partial sign flip | AUROC drops, no method recovers |
| `shape_shift` | Weibull shape ν | calibration degrades; ranking preserved |
| `strong_covariate_shift` | P(X): A_spu rotated 90° + noise | positive control, large gap (appendix) |
| `cure_shift` | cure fraction | non-PH, appendix only |

## Methods

| Method | Key | Description |
|---|---|---|
| Invariant oracle | `inv_oracle` | LR on true latent u_inv — upper bound |
| ERM LR | `lr_erm` | Pooled source logistic regression |
| IW-ERM | `iw_erm` | Importance-weighted ERM (covariate-shift correction) |
| CORAL | `coral` | Covariance alignment + LR |
| IRM | `irm` | Invariant Risk Minimization (IRMv1; needs ≥2 source domains) |
| Prior recalibration | `prior_recalib` | Saerens EM prior-shift correction (no target labels) |
| Platt scaling | `platt_scaling` | Sigmoid post-hoc calibration on source val |
| Isotonic | `isotonic` | Isotonic post-hoc calibration on source val |
| DANN | `dann` | Domain-adversarial network, IWCV HP tuning |
| CDAN | `cdan` | Conditional DANN, IWCV HP tuning |

## Generator

The data-generating process is a Weibull proportional hazards model:

- Event time: T = (−log U / (λ₀ exp(η)))^(1/ν)
- Binary label: y = 1[T ≤ τ],  τ = 6 months
- Latent factors: u_inv (k=3, invariant) + u_spu (k=2, spurious)
- Observed covariates: x = A_inv u_inv + A_spu u_spu  (p=20, linear)
- Risk: η = w_d · u_inv + w_spu · u_spu

Each shift scenario modifies exactly one component of the DGP.

## Protocol

- 1 source domain + 1 target domain, 500 samples each
- 80/20 stratified source train/val split
- StandardScaler fitted on source train only
- Target labels never used during training or HP selection
- DANN/CDAN HP tuning via importance-weighted cross-validation (IWCV)
- 10 random seeds, results reported as mean ± 95% CI
