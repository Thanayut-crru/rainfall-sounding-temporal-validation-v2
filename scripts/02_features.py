# -*- coding: utf-8 -*-
"""
02_features.py
==============
Aligns sounding indices with daily rainfall observations, computes
seasonal harmonics and wind-profile features, applies an unsupervised
correlation filter (|r| >= 0.90 on training data only), and assembles
the four feature sets (F1-F4) used in model training.

Inputs
------
    outputs/sounding_indices_raw.csv   (from 01_preprocess.py)
    data/raw/rainfall_tmd.csv          (TMD rain-gauge data — see data/README.md)
      OR
    data/sample/sample_dataset.csv     (demo only — set USE_SAMPLE=True below)

Outputs
-------
    outputs/dataset_final.csv          merged dataset with all features + labels
    outputs/feature_sets.json          feature-set membership lists F1-F4
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
CORR_THRESH   = 0.90          # |r| >= this on training data → drop from F1
TRAIN_YEARS   = list(range(2011, 2020))
RAIN_MM_THRESH = 0.1          # Study rain-occurrence threshold, mm

BASE = Path(os.environ.get('RAINFALL_PROJECT_DIR', Path(__file__).resolve().parent.parent))
OUT = Path(os.environ.get('RAINFALL_OUTPUT_DIR', BASE / 'outputs'))
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Assemble rainfall feature candidates.")
parser.add_argument(
    "--mode", choices=("real", "demo"), default="real",
    help="Use real TMD rainfall by default; demo explicitly uses synthetic labels.",
)
args = parser.parse_args()
USE_SAMPLE = args.mode == "demo"

# Indices recomputed from local IGRA profiles; names retain archive conventions.
F1_CANDIDATES = [
    "SHOW","LIFT","KINX","CTOT","VTOT","TTOT",
    "CAPE","CINS","EQLV","LFCT","LCLT","LCLP","PWAT",
]
WIND_FEATURES = ["WS850","WDsin850","WDcos850","WS700","WS200","VWS","VWS_lo"]
OROG_FEATURES = ["OROG850_phuket", "OROG850_krabi", "OROG850_phangnga"]

# ── Load sounding indices ──────────────────────────────────────────────────────
if USE_SAMPLE:
    src = BASE / "data" / "sample" / "sample_dataset.csv"
    print(f"[DEMO] Loading sample dataset: {src}")
    df = pd.read_csv(src, parse_dates=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Sample dataset already has labels and seasonal features; just write outputs
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Build feature sets from available columns
    f1_avail = [c for c in F1_CANDIDATES if c in df.columns]
    train_df = df[df["year"].isin(TRAIN_YEARS)]
    # Do not screen here: screening before CV exposes each validation fold to
    # information from the other years. 03_train.py fits the correlation
    # screen independently inside every training fold.
    f1_final = f1_avail
    to_drop = []
    print(f"  F1 candidates: {len(f1_avail)} (screened inside each training fold)")

    wind_avail = [c for c in WIND_FEATURES if c in df.columns]
    feat_sets  = {
        "F1": f1_final,
        "F2": f1_final + ["sin_month", "cos_month"],
        "F3": f1_final + wind_avail,
        "F4": f1_final + ["sin_month", "cos_month"] + wind_avail,
        "DROPPED": to_drop,
        "_meta": {
            "F1_description": f"{len(f1_final)} candidate stability indices; |r|>={CORR_THRESH} screening is fold-local",
            "F2_description": "F1 + cyclical season encoding (sin/cos month)",
            "F3_description": f"F1 + {len(wind_avail)} wind features",
            "F4_description": "F1 + season + wind = all features",
        },
    }
    df.to_csv(OUT / "dataset_final.csv", index=False)
    with open(OUT / "feature_sets.json", "w") as f:
        json.dump(feat_sets, f, indent=2)
    print(f"  Saved {len(df)} rows -> outputs/dataset_final.csv")
    print(f"  Saved feature_sets.json  (F1: {len(f1_final)} features)")
    raise SystemExit(0)

# ── REAL DATA MODE ─────────────────────────────────────────────────────────────
snd_path  = OUT / "sounding_indices_raw.csv"
rain_path = BASE / "data" / "raw" / "rainfall_tmd.csv"
if (OUT / 'rainfall_daily_aligned.csv').exists():
    rain_path = OUT / 'rainfall_daily_aligned.csv'
alignment_path = OUT / 'temporal_alignment_status.json'
alignment = json.loads(alignment_path.read_text(encoding='utf-8')) if alignment_path.exists() else {'status':'pending'}

print(f"Loading sounding indices: {snd_path}")
snd = pd.read_csv(snd_path, parse_dates=["date"])
if "qc_thermo_usable" in snd.columns:
    n_before = len(snd)
    usable = snd['qc_thermo_usable'].astype(str).str.lower().map({'true':True, 'false':False, '1':True, '0':False})
    if usable.isna().any():
        raise ValueError('Unrecognized sounding QC flag; refusing truthy-string coercion')
    snd = snd[usable].copy()
    print(f"Sounding QC: retained {len(snd)}/{n_before} profiles under documented revision-stage QC")

print(f"Loading TMD rainfall: {rain_path}")
rain = pd.read_csv(rain_path, parse_dates=["date"])
# Expected columns: date, RF_phuket, RF_krabi, RF_phangnga (daily totals, mm)

df = pd.merge(snd, rain, on="date", how="inner", validate='one_to_one')
orog_path = OUT / "orographic_forcing_daily.csv"
if orog_path.exists():
    orog = pd.read_csv(orog_path, parse_dates=["date"])
    df = pd.merge(df, orog, on="date", how="left", validate='one_to_one')
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

# Seasonal harmonics
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

# Missing rainfall is an unknown label, never a dry day.
for stn in ["phuket", "krabi", "phangnga"]:
    col = f"RF_{stn}"
    if col in df.columns:
        df[f"y_{stn}"] = (df[col] >= RAIN_MM_THRESH).astype('Int64').mask(df[col].isna())

# Candidate pool only. Correlation screening is fitted within each CV training
# fold in 03_train.py, and on all 2011-2019 training years for final holdout fits.
f1_avail = [c for c in F1_CANDIDATES if c in df.columns]
f1_final = f1_avail
to_drop = []
print(f"F1 candidates: {len(f1_avail)} (screened inside each training fold)")

wind_avail = [c for c in WIND_FEATURES if c in df.columns]
feat_sets  = {
    "F1": f1_final,
    "F2": f1_final + ["sin_month", "cos_month"],
    "F3": f1_final + wind_avail,
    "F4": f1_final + ["sin_month", "cos_month"] + wind_avail,
    "F5": f1_final + ["sin_month", "cos_month"] + wind_avail
          + [c for c in OROG_FEATURES if c in df.columns],
    "DROPPED": to_drop,
    "_meta": {
        "F1_description": f"{len(f1_final)} candidate stability indices; |r|>={CORR_THRESH} screening is fold-local",
        "F2_description": "F1 + cyclical season encoding (sin/cos month)",
        "F3_description": f"F1 + {len(wind_avail)} wind features",
        "F4_description": "F1 + season + wind = all features",
        "F5_description": "F4 + station-specific 850-hPa orographic forcing u dot grad(h); sensitivity analysis",
        "temporal_alignment_status": alignment.get('status','pending'),
        "rainfall_input": str(rain_path),
        "warning": None if alignment.get('status')=='confirmed' else 'PROVISIONAL: rainfall interval/date metadata remains unconfirmed; do not claim future-rainfall skill.',
    },
}

df.to_csv(OUT / "dataset_final.csv", index=False)
with open(OUT / "feature_sets.json", "w") as f:
    json.dump(feat_sets, f, indent=2)

print(f"Saved {len(df)} rows -> outputs/dataset_final.csv")
print(f"Saved feature_sets.json  (F1: {len(f1_final)} features)")
df.groupby('year').agg(n_profiles=('date','size')).to_csv(OUT / 'dataset_year_counts.csv')
