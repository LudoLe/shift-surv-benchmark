"""
Download TCGA-LUAD and TCGA-LUSC clinical data from the GDC REST API.

No authentication required (open-access clinical data).

Saves
-----
  data/public/tcga_luad_clinical.csv
  data/public/tcga_lusc_clinical.csv

Fields extracted
----------------
  patient_id    : TCGA barcode (e.g. TCGA-05-4244)
  tss_code      : tissue source site = chars 5-6 of barcode (institution proxy)
  project       : TCGA-LUAD or TCGA-LUSC
  vital_status  : 1=dead  0=alive
  os_days       : days to death (dead) or last follow-up (alive)
  age_years     : age at diagnosis (days_to_diagnosis / 365.25)
  sex           : 1=male  0=female
  stage         : pathologic stage ordinal (I=1 II=2 III=3 IV=4  NaN if missing)
  pack_years    : smoking pack-years (NaN if missing)
  smoking_cat   : 1=current  2=former  3=never  NaN if unknown

Usage
-----
  python -m data.download_tcga_nsclc
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_PUBLIC  = Path(__file__).parent / "public"
_GDC_URL = "https://api.gdc.cancer.gov/cases"

_FIELDS = [
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.gender",
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.tumor_stage",
    "diagnoses.pack_years_smoked",
    "diagnoses.cigarettes_per_day",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
]


def _fetch_project(project_id: str, page_size: int = 500) -> list[dict]:
    filt = json.dumps({
        "op": "=",
        "content": {"field": "project.project_id", "value": project_id}
    })

    records, from_idx = [], 0
    while True:
        resp = requests.get(_GDC_URL, params={
            "filters": filt,
            "fields":  ",".join(_FIELDS),
            "format":  "JSON",
            "size":    page_size,
            "from":    from_idx,
        }, timeout=60)
        resp.raise_for_status()
        hits  = resp.json()["data"]["hits"]
        total = resp.json()["data"]["pagination"]["total"]
        records.extend(hits)
        print(f"  {project_id}: fetched {len(records)}/{total}")
        if len(records) >= total:
            break
        from_idx += page_size
        time.sleep(0.3)

    return records


def _parse_stage(stage_str: str | None) -> float:
    if not stage_str:
        return float("nan")
    s = str(stage_str).lower().replace(" ", "")
    for roman, val in [("iv", 4), ("iii", 3), ("ii", 2), ("i", 1)]:
        if roman in s:
            return float(val)
    return float("nan")


def _parse_smoking(cpd) -> float:
    """cigarettes_per_day → smoking category proxy (NaN = unknown)."""
    if cpd is None:
        return float("nan")
    try:
        v = float(cpd)
        return 1.0 if v > 0 else 3.0   # >0 → current/former; 0 → never
    except (TypeError, ValueError):
        return float("nan")


def _records_to_df(records: list[dict], project_id: str) -> pd.DataFrame:
    rows = []
    for r in records:
        pid = r.get("submitter_id", "")
        # TSS code = chars 5-6 of TCGA-XX-NNNN
        tss = pid[5:7] if len(pid) >= 7 else ""

        dem   = r.get("demographic") or {}
        diags = r.get("diagnoses") or [{}]
        diag  = diags[0] if diags else {}

        vital  = 1 if str(dem.get("vital_status", "")).lower() == "dead" else 0
        dtd    = dem.get("days_to_death")
        dtlf   = diag.get("days_to_last_follow_up")

        if vital == 1 and dtd is not None:
            os_days = float(dtd)
        elif dtlf is not None:
            os_days = float(dtlf)
        else:
            os_days = float("nan")

        age_d  = diag.get("age_at_diagnosis")
        age_yr = float(age_d) / 365.25 if age_d is not None else float("nan")

        gender = str(dem.get("gender", "")).lower()
        sex    = 1.0 if gender == "male" else (0.0 if gender == "female" else float("nan"))

        stage  = _parse_stage(diag.get("tumor_stage"))
        pkyrs  = diag.get("pack_years_smoked")
        pack_y = float(pkyrs) if pkyrs is not None else float("nan")

        rows.append({
            "patient_id":   pid,
            "tss_code":     tss,
            "project":      project_id,
            "vital_status": vital,
            "os_days":      os_days,
            "age_years":    age_yr,
            "sex":          sex,
            "stage":        stage,
            "pack_years":   pack_y,
        })

    return pd.DataFrame(rows)


def download_tcga_nsclc(force: bool = False) -> dict[str, pd.DataFrame]:
    _PUBLIC.mkdir(parents=True, exist_ok=True)
    results = {}

    for project in ["TCGA-LUAD", "TCGA-LUSC"]:
        fname = _PUBLIC / f"tcga_{project.lower().replace('-', '_')}_clinical.csv"
        if fname.exists() and not force:
            print(f"  {project}: already exists at {fname}, skipping (use force=True to re-download)")
            results[project] = pd.read_csv(fname)
            continue

        print(f"Downloading {project}...")
        records = _fetch_project(project)
        df = _records_to_df(records, project)
        df.to_csv(fname, index=False)
        print(f"  Saved {len(df)} patients → {fname}")
        results[project] = df

    return results


if __name__ == "__main__":
    dfs = download_tcga_nsclc()
    for proj, df in dfs.items():
        print(f"\n=== {proj} ===")
        print(f"  n={len(df)}  OS missing={df['os_days'].isnull().sum()}")
        print(f"  TSS codes ({df['tss_code'].nunique()}): {sorted(df['tss_code'].unique())}")
        print(f"  vital_status: {df['vital_status'].value_counts().to_dict()}")
        print(f"  stage: {df['stage'].value_counts().sort_index().to_dict()}")
        print(f"  sex: {df['sex'].value_counts().to_dict()}")
        print(f"  pack_years missing: {df['pack_years'].isnull().sum()}/{len(df)}")
        print(f"  os_days: min={df['os_days'].min():.0f}  max={df['os_days'].max():.0f}  "
              f"median={df['os_days'].median():.0f}")
