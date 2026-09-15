# Pipeline — Run Order

Run scripts in numbered order. Each script saves outputs to `outputs/` and
reads inputs from the previous step(s).

## Prerequisites

```bash
conda env create -f environment.yml
conda activate rainfall-env
```

## Step 0 — Rainfall Import and Temporal Alignment

**`00_import_rainfall.py`** — Converts the 3-hour TMD workbook (Phuket, Krabi,
Takua Pa / Phang-nga) to daily totals.

```bash
python scripts/00_import_rainfall.py --input path/to/rain3hr.xlsx
```

**Output:** `data/raw/rainfall_tmd.csv`, `outputs/rainfall_import_qc.csv`,
`outputs/rainfall_import_metadata.json`.

**`00b_import_daily.py`** — Converts the horizontal TMD daily-rainfall
worksheet ("ปริมาณฝนรายวัน", one station-month per row) to dated records for
the additional eastern gauge, Nakhon Si Thammarat.

```bash
python scripts/00b_import_daily.py --input path/to/rain1day.xlsx
```

**Output:** `data/raw/rainfall_tmd_daily_east.csv`,
`outputs/rainfall_daily_east_qc.csv`, `outputs/rainfall_daily_east_metadata.json`.

**`00c_temporal_alignment.py`** — Validates and applies the provider-confirmed
rainfall accumulation semantics. The primary daily label spans after 22:00 on
day D−1 through 22:00 on day D; the diurnal script (Step 6) additionally uses
this alignment to build non-overlapping 07–13 / 13–19 / 19–01 six-hour targets.
Never infers accumulation timing from downstream model skill; writes a
`PENDING` status if the convention is not explicitly confirmed.

```bash
python scripts/00c_temporal_alignment.py
```

**Output:** `outputs/rainfall_daily_aligned.csv`,
`outputs/rainfall_daily_east_aligned.csv`, `outputs/rainfall_intervals_aligned.csv`,
`outputs/temporal_alignment_status.json`.

Trace (`T`) is treated as 0 mm for the 0.1-mm rain-day threshold. A day is
recorded as missing if any of its eight 3-hour observations is absent or `-`.

Raw TMD rainfall observations are not redistributed in this repository —
see `data/README.md`.

---

## Step 1 — Sounding Preprocessing and Terrain

**`01_preprocess.py`** — Parses WMO 48565 (station THM00048565) radiosonde
profiles from the NOAA NCEI Integrated Global Radiosonde Archive (IGRA v2.2),
computes thermodynamic indices locally with MetPy, and applies the
revision-stage physical-coverage QC (minimum paired pressure–temperature–
dew-point levels, lowest usable pressure ≥ 950 hPa, coverage to 500 hPa, no
gap exceeding 200 hPa).

**Output:** `outputs/sounding_indices_raw.csv`,
`outputs/sounding_preprocess_metadata.json`, `outputs/sounding_qc_by_year.csv`.

**`01b_terrain.py`** — Extracts a 21×21-pixel Copernicus GLO-30 DEM
neighbourhood around each gauge, computes the median eastward/northward
terrain gradient, and derives the dynamic orographic-forcing proxy
`OROG850 = U850·∂h/∂x + V850·∂h/∂y` (units m/s) used by feature set F5.

**Input:** `data/external/dem/*.tif`
**Output:** `outputs/station_terrain.csv`, `outputs/orographic_forcing_daily.csv`.

---

## Step 2 — Feature Engineering (`02_features.py`)

Aligns sounding indices with daily rainfall observations, converts and
interpolates wind components in log-pressure coordinates, assembles the F1–F5
nested feature sets, and applies correlation screening (|r| ≥ 0.90, training
data only, sequential by declared column order).

**Input:** `outputs/sounding_indices_raw.csv`, aligned rainfall, `outputs/orographic_forcing_daily.csv`
**Output:** `outputs/dataset_final.csv`, `outputs/feature_sets.json`

---

## Step 3 — Model Training & Validation (`03_train.py`)

Runs year-grouped 5-fold GroupKFold CV (2011–2019; 2017 excluded — no
profiles passed QC) and chronological holdout evaluation (2020–2021) for LR,
SVM, RF, XGB and MLP against a monthly-climatology baseline, for every
feature set.

**Input:** `outputs/dataset_final.csv`, `outputs/feature_sets.json`
**Output:** `outputs/binary_cv_results.csv`, `outputs/binary_test_results.csv`,
`outputs/data_split_summary.csv`, `outputs/holdout_predictions.csv`,
`outputs/cv_selected_configurations.csv`

---

## Step 4 — Annual Robustness & Significance (`04_validate.py`)

Runs the Leave-One-Year-Out (LOYO) analysis (common F4 series for all five
classifiers plus the CV-selected configuration), calibration metrics (Brier
score, PR-AUC, descriptive bootstrap intervals), and the DeLong / Wilcoxon
paired significance tests with Benjamini–Hochberg correction.

**Input:** `outputs/dataset_final.csv`, `outputs/binary_test_results.csv`
**Output:** `outputs/loyo_results.csv`, `outputs/loyo_selected_results.csv`,
`outputs/loyo_year_status.csv`, `outputs/calibration_results.csv`,
`outputs/delong_results.csv`, `outputs/wilcoxon_results.csv`,
figures in `outputs/figures/` (`fig_loyo.png`, `fig_cv_holdout.png`,
`fig_holdout_models.png`)

---

## Step 5 — SHAP Diagnostics (`05_shap.py`)

Computes TreeSHAP values for XGB+F4 (the fixed nonlinear diagnostic) on the
chronological holdout set for every primary station, and generates the
beeswarm and bar-plot figures.

**Input:** `outputs/dataset_final.csv`, `outputs/binary_test_results.csv`
**Output:** `outputs/shap_feature_rank.csv`,
`outputs/figures/fig_shap_barplot.png`, `outputs/figures/fig_shap_beeswarm.png`

---

## Step 6 — Sensitivity Analyses

**`06_diurnal.py`** — Fits separate F4 models for three non-overlapping
six-hour post-sounding rain-occurrence targets (morning 07–13, afternoon
13–19, evening 19–01 next day), using the same fixed model settings as the
primary analysis.

**Output:** `outputs/diurnal_results.csv`, `outputs/diurnal_targets.csv`,
`outputs/figures/fig_diurnal.png`

**`06b_cross_mountain.py`** — Trains station-specific F4/F5 models for the
additional eastern gauge, Nakhon Si Thammarat, using that station's own
rainfall labels together with the Phuket sounding predictors (not a
frozen west-to-east model-transfer experiment).

**Output:** `outputs/cross_mountain_results.csv`,
`outputs/cross_mountain_loyo.csv`, `outputs/cross_mountain_shap.csv`,
`outputs/figures/fig_cross_mountain.png`

---

## Supporting Modules

- `model_common.py` — shared fixed model configurations and the common
  `fit_predict` / provenance-writing routine used by all training scripts.
- `station_metadata.py` / `refresh_distance_metadata.py` — station
  coordinates and great-circle sounding-to-gauge distances.
- `figure1_station_map.py` — generates the Figure 1 station map.
- `test_models.py`, `test_physics.py`, `test_station_metadata.py`,
  `test_temporal_alignment.py` — unit tests for the corresponding modules.

## Expected Runtime

| Script | Approx. time |
|---|---|
| 01_preprocess.py | 30–90 min (network dependent; IGRA download) |
| 01b_terrain.py | < 5 min |
| 02_features.py | < 1 min |
| 03_train.py | 5–15 min |
| 04_validate.py | 5–10 min |
| 05_shap.py | 2–5 min |
| 06_diurnal.py | 5–10 min |
| 06b_cross_mountain.py | 2–5 min |
