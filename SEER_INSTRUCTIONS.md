# SEER multicentric benchmark — how to obtain and run

SEER geographic **registries** are used as the centers, so a source→target
transfer is a genuine cross-site generalisation (unlike SUPPORT2's disease-class
pseudo-domains). SEER is access-controlled, so you download it once and drop a
cleaned CSV at `data/public/seer.csv`; the code (`data/registry_seer.py`) does the rest.

## 1. Get access (free, ~1–2 business days)
1. Go to <https://seer.cancer.gov/data/access.html> and submit the **SEER Research Data** request (institutional email, sign the data-use agreement).
2. Once approved, download **SEER*Stat** (<https://seer.cancer.gov/seerstat/>) and sign in with your credentials.

## 2. Build a Case Listing export in SEER*Stat
1. **File → New → Case Listing Session.**
2. **Data tab:** pick a database, e.g. *Incidence – SEER Research Data, 17 Registries*.
3. **Selection tab:** restrict to one cancer site to keep features comparable — e.g. lung (to mirror I3LUNG): `Site and Morphology → Site recode ICD-O-3/WHO 2008 = "Lung and Bronchus"`. Optionally restrict to known survival (`Survival months` available).
4. **Table tab → select these variables** (drag to "Column"):
   - `SEER registry`
   - `Survival months`
   - `Vital status recode (study cutoff used)`
   - `Age recode` (or `Age at diagnosis`)
   - `Sex`
   - `Year of diagnosis`
   - `Combined Summary Stage` (or `SEER historic stage A`)
   - `Grade` (or `Grade Recode`)
   - `CS tumor size` / `Tumor Size Summary`
5. **Execute**, then **Matrix → Export → Results as text/CSV**.

## 3. Clean to the expected schema
The registry expects these exact column names in `data/public/seer.csv`:

| column | meaning | encoding |
|---|---|---|
| `registry` | SEER registry (the center) | string |
| `survival_months` | overall survival | numeric |
| `vital_status` | event | `1`=dead / `0`=alive (or the strings "Dead"/"Alive") |
| `age` | age at diagnosis (years) | numeric |
| `sex` | sex | `0`=male, `1`=female |
| `year_dx` | year of diagnosis | numeric |
| `stage` | ordinal stage | e.g. `0`=localized, `1`=regional, `2`=distant |
| `grade` | tumour grade | `1..4` |
| `tumor_size` | tumour size (mm) | numeric (unknown → blank/NaN) |

A minimal cleaning template (adapt the raw SEER*Stat column names on the left):

```python
import pandas as pd, numpy as np
raw = pd.read_csv("seer_export.csv")          # your SEER*Stat export
df = pd.DataFrame({
    "registry":        raw["SEER registry"],
    "survival_months": pd.to_numeric(raw["Survival months"], errors="coerce"),
    "vital_status":    raw["Vital status recode (study cutoff used)"].str.startswith("Dead").astype(int),
    "age":             pd.to_numeric(raw["Age at diagnosis"], errors="coerce"),
    "sex":             (raw["Sex"] == "Female").astype(int),
    "year_dx":         pd.to_numeric(raw["Year of diagnosis"], errors="coerce"),
    "stage":           raw["Combined Summary Stage"].map(
                          {"Localized": 0, "Regional": 1, "Distant": 2}),
    "grade":           pd.to_numeric(raw["Grade"].str[0], errors="coerce"),  # "3" from "3: poorly diff"
    "tumor_size":      pd.to_numeric(raw["CS tumor size"], errors="coerce"),
})
df.to_csv("data/public/seer.csv", index=False)
```
Coded-missing values (blanks, 999, "Unknown") that don't parse become NaN and are
median-imputed on the source center downstream — same as the I3LUNG pipeline.
(For a non-lung site, edit `FEATURE_COLS` / `TAU` in `data/registry_seer.py`.)

## 4. Run (on the server, in the Keras-2 env `shift310`)
```bash
conda activate shift310

# 1) discover the registries (name + n); pick the transfers
python -m experiments.run_real_data --dataset seer --list_centers

# 2) run a transfer both directions (put the LARGEST registry as --target for
#    tight CIs, since the calibration recovery is power-driven). Repeat per tau.
python -m experiments.run_real_data --dataset seer \
    --source "<REGISTRY_A>" --target "<REGISTRY_B>" --tau 12 \
    --out_prefix public_seer_A_B_tau12
python -m experiments.run_real_data --dataset seer \
    --source "<REGISTRY_B>" --target "<REGISTRY_A>" --tau 12 \
    --out_prefix public_seer_B_A_tau12
# (also --tau 6 / 24 for the robustness sweep, as with rotterdam_gbsg)

# 3) verdict (Holm-corrected, cherry-pick-proof)
python -m experiments.summarize_recovery --glob "public_seer_*_bootstrap.csv"
```

The success criterion mirrors move-1: on the large-target direction, does a method
significantly beat ERM on **ECE** (robust across τ) after Holm — confirming the
calibration-recovery finding on a second, genuinely multi-site real dataset.
`data/public/seer.csv` is git-ignored (data-use agreement) — keep it local to the server.
