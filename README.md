# SHIFT-SURV: A Synthetic Benchmark for Distribution Shift in Fixed-Horizon Survival Risk Prediction

Synthetic benchmark for studying how covariate shift, concept shift, baseline hazard shift,
and spurious correlations affect **fixed-horizon survival-risk models** transported across
clinical sites.

## Scope

The data-generating process is a Weibull proportional-hazards survival model (see
[Generator](#generator)), but the prediction task evaluated here is **fixed-horizon binary
classification**: the label is `y = 1[T ≤ τ]` (event by τ = 6 months) and every method is a
classifier of that label. The benchmark therefore measures transportability of fixed-horizon
*risk* under multi-center shift, not estimation of the full survival curve. The main benchmark
is **uncensored** (every event time is observed before τ); the generator additionally supports
independent censoring and a mixture-cure model, used only by the appendix `cure_shift` scenario.
We do not fit time-to-event models (Cox, RSF, DeepSurv) — extending the protocol to censored,
curve-level evaluation is left as future work.

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

# Multi-source benchmark with the IRM + GroupDRO domain-generalization baselines
# (both need >=2 source centers; they degenerate to ERM with one source):
python -m experiments.run_adaptation --n_source_domains 3 --out_prefix benchmark_3s1t

# Real-data counterpart: leave-one-center-out multi-source DG on I3LUNG
# (pools the other centers as sources; adds IRM + GroupDRO; feeds meta_analysis):
python -m experiments.run_real_data --dataset i3lung --data_dir /path/to/I3LUNG_DATA --loco
python -m experiments.meta_analysis --prefix i3lung_loco

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
| IRM | `irm` | Invariant Risk Minimization (IRMv1, linear; only added when `--n_source_domains >= 2`) |
| Prior recalibration | `prior_recalib` | Saerens EM prior-shift correction (no target labels) |
| Platt scaling | `platt_scaling` | Sigmoid post-hoc calibration on source val |
| Isotonic | `isotonic` | Isotonic post-hoc calibration on source val |
| DANN | `dann` | Domain-adversarial network, IWCV HP tuning |
| CDAN | `cdan` | Conditional DANN, IWCV HP tuning |
| GroupDRO | `group_dro` | Worst-source-center robust optimization (domain generalization, no target data); only added when `--n_source_domains >= 2` |

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
