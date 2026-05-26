# SHIFT-SURV — Generator Validation (Phase 1)

## Overview

This phase implements and validates the synthetic multi-center generator described in the paper.
The goal is **not** adaptation benchmarking yet — it is verifying that the five shift regimes
are realised in a clean and interpretable way before any adaptation methods are plugged in.

## Quick start

```bash
cd /Users/ludole/Desktop/SHIFT-SURV
pip install numpy scikit-learn matplotlib

# Run all five scenarios
python -m experiments.run_sanity_checks

# Run a single scenario
python -m experiments.run_sanity_checks --scenario spurious_shift

# Larger sample size
python -m experiments.run_sanity_checks --n 1000
```

Outputs land in `outputs/figures/<scenario>/` and `outputs/tables/<scenario>/`.

---

## Generator design

### Notation (Section 4 of the paper)

Each individual *i* in domain *d* is generated as follows:

```
u_inv ~ N(μ_d^inv, I_{k_inv})          # invariant latent factors
u_spu ~ N(μ_d^spu, I_{k_spu})          # spurious latent factors
x     = A_d [u_inv; u_spu] + ε         # observed covariates, ε ~ N(0, σ² I_p)

η     = β_{0,d} + w_d^T u_inv + v_d^T u_spu    # latent log-risk
U     ~ Uniform(0,1)
T     = (- log U / (λ_{0,d} exp(η)))^{1/ν_d}   # Weibull PH event time

y     = 1[T ≤ τ],   τ = 6 months               # fixed-horizon label
```

The full domain parameter tuple is:

```
Θ_d = (μ_d^inv, μ_d^spu, A_d, Σ_d, β_{0,d}, w_d, v_d, λ_{0,d}, ν_d)
```

### Key structural choices

- **`k_inv = 3`** invariant latent factors, **`k_spu = 2`** spurious factors, **`p = 20`** observed features.
- In the base regime, `v_d = 0`: spurious factors enter the covariates but not the risk.
- The loading matrix `A` is shared across domains by default; scenarios may override it.
- Labels are derived from the latent event time **without censoring** — censoring is a later-phase extension.

---

## Scenario parameterisation

Each scenario changes a different subset of `Θ_d` (Table 1 of the paper):

| Scenario | What varies | What is fixed | Expected effect |
|---|---|---|---|
| `no_shift` | nothing | all Θ_d identical | Control: source and target look the same |
| `covariate_shift` | μ^inv, μ^spu, Σ | β₀, w, v, λ₀, ν | p(x) shifts; outcome mechanism invariant |
| `spurious_shift` | μ^spu, Σ | β₀, w, λ₀, ν; v_d=0 always | Shortcut features flip direction at target |
| `baseline_hazard_shift` | β₀, λ₀, ν | μ^inv, μ^spu, A, Σ, w, v | Prevalence shifts; ranking preserved |
| `concept_shift` | w_d (sign flip) | everything else | Invariant mechanism changes; no shared optimal predictor |

---

## Sanity checks per scenario

### No shift
- [ ] Prevalence difference |src − tgt| < 0.06
- [ ] Covariate shift score < 0.30
- [ ] AUROC gap (all_observed, src_pooled − target) < 0.05

### Covariate shift
- [ ] Covariate shift score > 0.30 (PCA clouds separated)
- [ ] Invariant oracle AUROC gap < all_observed AUROC gap

### Spurious shift
- [ ] Spurious oracle AUROC degrades strongly on target (gap > 0.15)
- [ ] Invariant oracle gap < spurious oracle gap
- [ ] Source spurious AUROC is high; target spurious AUROC degrades toward chance

### Baseline hazard shift
- [ ] Prevalence shift > 0.10 (target prevalence substantially higher)
- [ ] Covariate shift score < 0.30 (p(x) unchanged)
- [ ] Invariant oracle AUROC gap < 0.10 (ranking preserved; discrimination stable)

### Concept shift
- [ ] Invariant oracle gap > 0.20
- [ ] Target AUROC for invariant oracle ≤ 0.60 (near-chance or inverted)

---

## Project structure

```
SHIFT-SURV/
├── generator/
│   ├── base_generator.py   # SyntheticGenerator + DomainDataset
│   ├── scenarios.py        # generate_scenario("name", ...)
│   └── parameters.py       # BaseParams, DomainParams, defaults
├── experiments/
│   ├── baselines.py        # LogReg: invariant_oracle, spurious_oracle, all_observed
│   ├── diagnostics.py      # Per-scenario diagnostic suite
│   └── run_sanity_checks.py  # Top-level runner (entry point)
├── utils/
│   ├── metrics.py          # AUROC, AUPRC, Brier
│   ├── plotting.py         # PCA, latent dists, prevalence bars, AUROC comparison
│   └── io.py               # Save CSV/JSON/figures
├── outputs/
│   ├── figures/<scenario>/ # Diagnostic plots
│   └── tables/<scenario>/  # CSV + JSON summaries
└── README_generator_validation.md
```

---

## How to plug in adaptation methods later (Phase 2)

The `generate_scenario` function returns a dict with keys `"source"` (list of
`DomainDataset`) and `"target"` (`DomainDataset`).  Each `DomainDataset` exposes:

- `.x` — observed covariates `(n, p)`
- `.y` — fixed-horizon labels `(n,)`
- `.domain_id` — integer domain index

To integrate the [ADAPT library](https://github.com/adapt-python/adapt):

```python
from generator import generate_scenario
import numpy as np

result = generate_scenario("covariate_shift")
X_src = np.vstack([ds.x for ds in result["source"]])
y_src = np.concatenate([ds.y for ds in result["source"]])
X_tgt = result["target"].x   # unlabeled target

# ADAPT expects (X_src, y_src, X_tgt) — plug directly here
# from adapt.feature_based import DANN
# model = DANN(...)
# model.fit(X_src, y_src, X_tgt)
```

No adaptation-specific code needs to change in the generator or diagnostics modules.
