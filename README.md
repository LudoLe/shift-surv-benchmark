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

## Reproducing the paper

The result tables the paper is built from are tracked in this repository, so the
figures can be rebuilt without rerunning anything:

```bash
jupyter notebook reproduce_figures.ipynb
```

The sections below rebuild those tables from scratch. Each stage writes to
`outputs/tables/` and is independent of the others, so they can be run in any
order or skipped.

### 0. Environment

`adapt==0.4.5` targets Keras 2, and TensorFlow 2.16 and later ship Keras 3, which
silently degrades DANN and CDAN. The pins therefore need Python 3.10 or 3.11.

```bash
bash setup_env.sh          # creates the conda environment and installs the pins
conda activate shift310
python -c "import adapt, tensorflow; print(adapt.__version__, tensorflow.__version__)"
```

### 1. Synthetic benchmark

Produces the discrimination and calibration results per shift regime, under both
the adaptation protocol and the domain-generalization protocol.

```bash
bash experiments/run_all_synthetic.sh
```

Writes `benchmark_1s1t_iwcv_*`, `benchmark_1s1t_prior_*`, `benchmark_3s1t_*`,
`benchmark_1s1t_full_*` and `ablation_no_iwcv_*`. Around two hours on a machine
with a few free cores; the individual commands are listed further down for
running one at a time.

### 2. I3LUNG

Needs the cohort, which is not public. The directory must contain `cb.csv`,
`outcomes.csv` and `features.json`.

```bash
export I3LUNG_DATA=/path/to/I3LUNG_DATA
bash experiments/run_all_realdata.sh $I3LUNG_DATA        # 30 directed transfers
python -m experiments.run_real_data --dataset i3lung \
       --data_dir $I3LUNG_DATA --loco                    # leave-one-centre-out
python -m experiments.meta_analysis --prefix i3lung --by_target
python -m experiments.meta_analysis --prefix i3lung --exclude loco \
       --out_prefix i3lung_meta_pairs                    # thirty-pair control
python -m experiments.meta_analysis --prefix i3lung_loco
python -m experiments.prior_estimability --data_dir $I3LUNG_DATA
python -m experiments.center_directions --data_dir $I3LUNG_DATA
```

The two meta-analyses answer different questions. `--by_target` pools the five
transfers that share a target cohort and carries six disjoint units, which is the
level at which the independence assumption holds; it is the analysis the paper
quotes. The pairwise version treats all thirty transfers as units and is reported
alongside as a control.

### 3. Public registry

Rotterdam and GBSG2 download themselves, so this stage needs no local data.

```bash
bash experiments/run_all_public.sh rotterdam_gbsg
python -m experiments.meta_analysis --prefix public_rotterdam_gbsg
```

### 4. Sensitivity analysis

Repeats the benchmark across the generator's free parameters, sampled with a
Latin hypercube. Every method that has a hyperparameter is selected per run by
the importance-weighted criterion, which is the protocol the main experiments
use.

```bash
bash experiments/run_sensitivity_parallel.sh
```

This is the expensive stage, measured in machine-days rather than hours. It
splits by regime and by source count, and each shard writes its own CSV, so the
work survives an interruption and rerunning the script resumes from what is
missing.

### 5. Figures

```bash
jupyter notebook reproduce_figures.ipynb
```

or, one at a time:

```bash
python paper_assets/_gen_fig_disc.py
python paper_assets/_gen_fig_cal.py
python paper_assets/_gen_fig_realmain.py
python paper_assets/_gen_fig_predictability.py
```

## Individual commands

```bash
# Adaptation protocol, one source centre, seven core scenarios
python -m experiments.run_adaptation

# Domain-generalization protocol: three source centres, adds IRM and GroupDRO
python -m experiments.run_adaptation --n_source_domains 3 --out_prefix benchmark_3s1t

# Appendix scenarios (strong_covariate_shift, cure_shift)
python -m experiments.run_adaptation --include_appendix --out_prefix benchmark_1s1t_full

# Ablation: covariate shift without importance-weighted hyperparameter selection
python -m experiments.run_adaptation --scenarios covariate_shift --no_iwcv \
    --out_prefix ablation_no_iwcv

# One I3LUNG transfer
python -m experiments.run_real_data --dataset i3lung --data_dir $I3LUNG_DATA \
    --source INT --target GHD

# Sensitivity, one regime at a time
python -m experiments.sensitivity_sweep --mode worlds --n_source_domains 1 \
    --regimes concept_shift --n_seeds 10 --n_worlds 100 --with_nets --tune_all
```

Outputs are written to `outputs/tables/<prefix>_{raw,summary,compact,maxstat}.csv`.

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
| `prior_shift` | P(Y): class-conditional resample, P(X\|Y) fixed | pure label shift; Prior-EM recovers ECE (calibration positive control, appendix) |
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

## Configuration

Every structural constant, per-domain default and per-scenario shift parameter is declared in
[`configs/generator.yaml`](configs/generator.yaml) — the single place to read or change what the
benchmark ran with. Point `SHIFT_SURV_CONFIG` at another file to use a different one:

```bash
SHIFT_SURV_CONFIG=configs/my_variant.yaml python -m experiments.run_adaptation
```

The Python defaults mirror the config exactly, so the code also runs without it (and without
`pyyaml`). `tests/test_config.py` enforces both halves of that contract: the YAML values must
equal the Python defaults, and generation must be bit-identical with and without the config.
Explicit keyword arguments still override the config, which is how the sweeps and ablations vary
one parameter at a time.

## Protocol

- 1 source domain + 1 target domain, 500 samples each
- 80/20 stratified source train/val split
- StandardScaler fitted on source train only
- Target labels never used during training or HP selection
- DANN/CDAN HP tuning via importance-weighted cross-validation (IWCV)
- 10 random seeds, results reported as mean ± 95% CI
