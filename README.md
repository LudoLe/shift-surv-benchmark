# SHIFT-SURV

Benchmark and evaluation framework for **fixed-horizon survival-risk prediction under
multi-center distribution shift**. Companion code for the paper *"SHIFT-SURV: Diagnosing Why
Domain Adaptation Fails in Multi-Center Survival Risk Prediction"* (under review).

A latent-factor Weibull proportional-hazards generator produces multi-center cohorts and
perturbs one shift mechanism at a time (covariate, measurement, spurious association,
baseline hazard, label prior, concept). Domain adaptation (IW-ERM, CORAL, DANN, CDAN),
domain generalization (IRM, GroupDRO) and recalibration (prior adjustment, Platt scaling,
isotonic regression) are evaluated against two reference models: a **latent-feature
reference** fitted on the true generative factors, and an **observed-data reference**
fitted to the target center's own covariates and labels out of fold. The same pipeline
runs on two real cohorts: the six-center I3LUNG lung-cancer cohort (private) and the
public Rotterdam–GBSG2 breast-cancer registry.

## Repository map

| Path | Content |
|---|---|
| `generator/` | data-generating process and the six shift scenarios |
| `experiments/` | training, evaluation, per-run hyperparameter selection, meta-analysis, sensitivity sweep |
| `data/` | cohort loaders (I3LUNG, Rotterdam–GBSG2) |
| `configs/` | generator parameters (`generator.yaml`) |
| `outputs/tables/` | **the tracked result tables every paper number comes from** |
| `outputs/figures/paper/` | the three figures the paper includes |
| `paper_assets/` | figure generators (`_gen_fig_disc.py`, `_gen_fig_cal.py`, `_gen_fig_realmain.py`; `_gen_fig_predictability.py` prints the Spearman quoted in the discussion) |
| `reproduce_figures.ipynb` | rebuilds the paper figures from the tracked tables |
| `tests/` | generator and pipeline invariants (43 tests) |

## Environment

`adapt==0.4.5` targets Keras 2; TensorFlow ≥ 2.16 ships Keras 3, which silently degrades
DANN and CDAN. The pins therefore require Python 3.10/3.11, `tensorflow<2.16`, `numpy<2`
(full rationale in `requirements.txt`).

```bash
bash setup_env.sh              # creates conda env "shift310" and installs the pins
conda activate shift310
python -c "import keras, adapt; print(keras.__version__)"   # must print 2.x
```

## Data

- **Synthetic benchmark**: no data needed.
- **I3LUNG**: private clinical cohort, not distributed (see LICENSE). Every I3LUNG stage
  takes a data directory containing `cb.csv`, `outcomes.csv`, `features.json`.
- **Rotterdam–GBSG2**: public. GBSG2 loads through `scikit-survival`; Rotterdam comes from
  the Rdatasets mirror of the R `survival` package:

```bash
mkdir -p data/public
curl -L -o data/public/rotterdam.csv \
  https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/rotterdam.csv
```

## Reproducing the paper

### From the tracked tables (no data access, minutes)

The tables in `outputs/tables/` are the exact final-run outputs behind the paper. From them:

```bash
bash paper_assets/make_paper_figures.sh        # or: jupyter notebook reproduce_figures.ipynb
python -m experiments.meta_analysis --prefix i3lung --exclude loco --by_target --out_prefix i3lung
python -m experiments.meta_analysis --prefix i3lung --exclude loco --out_prefix i3lung_meta_pairs
python -m experiments.meta_analysis --prefix i3lung_loco --out_prefix i3lung_loco
python -m experiments.meta_analysis --prefix public_rotterdam_gbsg
```

The pairs, generalization, and registry metas read only tracked per-transfer tables and
reproduce byte for byte on a fresh clone. The by-target meta additionally needs the
patient-level bootstrap replicate matrices (`*_bootreps.csv`, large and untracked) for its
joint standard errors; stage 4 of the full re-run regenerates them, after which it too
reproduces byte for byte.

### Full re-run

Stages are independent unless a later one reads an earlier one's tables. `$I3LUNG_DATA` is
the private cohort directory.

| # | Stage | Command | Writes | Time |
|---|---|---|---|---|
| 1 | Synthetic benchmark + sample-size sweep | `bash experiments/run_all_synthetic.sh` | `benchmark_1s1t_iwcv_*`, `benchmark_1s1t_prior_*`, `benchmark_3s1t_*`, `n_sweep_iwcv_*`, `survival_cox_*` | ~2 h, multi-core |
| 2 | Nonlinear-observation ablation | `python -m experiments.run_adaptation --nl_strength 0.5 --n_sweep 250,500,1000,2000 --out_prefix nl_nsweep` and the same with `--scenarios prior_shift --out_prefix nl_nsweep_prior` | `nl_nsweep_*` | ~1 h |
| 3 | Sensitivity sweep (1692 configurations) | `bash run_tuneall.sh` and `bash run_oat_tuneall.sh` (self-resuming shards), then the confirmation runs (`python -m experiments.sensitivity_sweep --mode confirm …`, flags in the script header), then `python -m experiments.collect_sweep` | `sens_tuneall_*` | days |
| 4 | I3LUNG: 30 transfers + 6 LOCO folds | `bash experiments/run_all_parallel.sh $I3LUNG_DATA` (or `run_all_realdata.sh` for the transfers alone) | `i3lung_*_{summary,bootstrap,shiftdiag}.csv` | hours |
| 5 | Registry | `bash experiments/run_all_public.sh rotterdam_gbsg` | `public_rotterdam_gbsg_*` | ~1 h |
| 6 | References and diagnostics | `python -m experiments.real_target_oracle --data_dir $I3LUNG_DATA` (and `--dataset rotterdam_gbsg --tau 36`), `python -m experiments.headroom_ci --data_dir $I3LUNG_DATA`, `python -m experiments.prior_estimability --data_dir $I3LUNG_DATA` | `*_target_oracle.csv`, `i3lung_headroom_ci.csv`, `i3lung_prior_estimability.csv` | ~1 h |
| 7 | Censored-survival check on I3LUNG | `python -m experiments.run_survival_real --data_dir $I3LUNG_DATA` | `survival_real_transfers.csv` | minutes |
| 8 | Meta-analysis | the four `meta_analysis` invocations above | `*_meta.csv`, selection bounds | minutes |

### Where each paper number lives

| Paper element | Tracked tables |
|---|---|
| Synthetic discrimination / calibration figures | `benchmark_1s1t_iwcv_*`, `benchmark_1s1t_prior_*` (+ `_maxstat` for the tests) |
| Domain-generalization table | `benchmark_3s1t_*`, `benchmark_3s1t_prior_*` |
| Observed-data headroom table | `i3lung_headroom_ci.csv` |
| Real-cohort forest plot and pooled effects | `i3lung_meta.csv`, `i3lung_meta_pairs_meta.csv`, `i3lung_loco_meta.csv`, `public_rotterdam_gbsg_*_bootstrap.csv` |
| Per-transfer interval counts | `i3lung_<SRC>_<TGT>_bootstrap.csv` (30 files) |
| Selection upper bounds | `best_per_transfer` rows of the metas, `i3lung_loco_selection_bound.csv`, registry bootstraps |
| Prior-adjustment failure mechanism | `i3lung_prior_estimability.csv` |
| Center heterogeneity (RBF-MMD, prevalence) | `i3lung_*_shiftdiag.csv`, `public_rotterdam_gbsg_*_shiftdiag.csv` |
| Sensitivity analysis | `sens_tuneall_{worlds,oat,confirm}_*` |
| Sample-size paragraph | `n_sweep_iwcv_*` |
| Nonlinear-observation paragraph | `nl_nsweep_*` |
| Fixed-horizon / Cox paragraph | `survival_cox_*`, `survival_real_transfers.csv` |
| Discussion Spearman (covariate discrepancy vs loss) | `i3lung_*_shiftdiag.csv` + `i3lung_*_summary.csv`, computed by `paper_assets/_gen_fig_predictability.py` |
| Observed-data reference on real cohorts | `i3lung_target_oracle.csv`, `public_rotterdam_gbsg_target_oracle.csv` |

## Determinism

All scikit-learn-based methods and the generator are byte-reproducible from their seeds.
DANN and CDAN train on TensorFlow, whose op-level nondeterminism means they reproduce in
distribution across machines; every DANN/CDAN number in the paper is a mean over 10 seeds
with its confidence interval.

## Tests

```bash
python -m pytest    # 43 passed
```

## License and citation

MIT (see LICENSE; the I3LUNG clinical data are not part of this repository).
Citation metadata in `CITATION.cff`.
