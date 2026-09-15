# rainfall-sounding-temporal-validation

A reproducible, temporally validated modelling framework for binary rainfall
occurrence prediction from radiosonde-derived atmospheric instability indices.



---

## Overview

This repository provides the complete pipeline to reproduce the results of the
associated paper, including:

- Radiosonde ingestion from the NOAA NCEI Integrated Global Radiosonde
  Archive (IGRA v2.2, station THM00048565), with all thermodynamic indices
  recomputed locally with MetPy
- Feature extraction: 13 thermodynamic candidate indices screened for
  collinearity, nested into four primary feature sets (F1–F4) plus a
  terrain-sensitivity set (F5) using a Copernicus GLO-30 DEM-derived
  orographic-forcing proxy (OROG850)
- Temporal validation: year-grouped 5-fold CV (2011–2019), chronological
  holdout (2020–2021), and Leave-One-Year-Out (LOYO) robustness
- Model comparison: LR, SVM, RF, XGB, MLP vs. a monthly-climatology baseline
- Paired significance testing: DeLong (holdout) and Wilcoxon (LOYO) with
  Benjamini–Hochberg correction
- SHAP (TreeSHAP) physical-plausibility diagnostics for XGB+F4
- Sensitivity analyses: an additional eastern gauge (Nakhon Si Thammarat,
  beyond the central mountain range) and post-sounding six-hour
  morning/afternoon/evening rain-occurrence targets

---

## Quick Start

```bash
git clone https://github.com/Thanayut-crru/rainfall-sounding-temporal-validation.git
cd rainfall-sounding-temporal-validation
conda env create -f environment.yml
conda activate rainfall-env
```

Then run scripts in order (see PIPELINE.md):

```bash
python scripts/00_import_rainfall.py --input path/to/rain3hr.xlsx
python scripts/00b_import_daily.py --input path/to/rain1day.xlsx
python scripts/00c_temporal_alignment.py
python scripts/01_preprocess.py
python scripts/01b_terrain.py
python scripts/02_features.py
python scripts/03_train.py
python scripts/04_validate.py
python scripts/05_shap.py
python scripts/06_diurnal.py
python scripts/06b_cross_mountain.py
```

Demo notebook (no raw TMD data required):

```bash
jupyter notebook notebooks/demo_workflow.ipynb
```


## Repository Structure

```
rainfall-sounding-temporal-validation/
├── README.md
├── LICENSE                        MIT licence
├── requirements.txt               pip-compatible dependencies
├── environment.yml                conda environment spec
├── CITATION.cff                   citation metadata
├── PIPELINE.md                    step-by-step run order
│
├── data/
│   ├── README.md                  data access instructions
│   ├── external/                  DEM/IGRA provenance and format docs (small files only)
│   └── sample/
│       └── sample_dataset.csv     sounding predictors + synthetic labels (demo)
│
├── scripts/
│   ├── 00_import_rainfall.py      3-hour TMD workbook → daily totals
│   ├── 00b_import_daily.py        horizontal daily TMD sheet (eastern gauge)
│   ├── 00c_temporal_alignment.py  confirmed accumulation-window alignment
│   ├── 01_preprocess.py           IGRA sounding download + QC
│   ├── 01b_terrain.py             DEM terrain gradient + OROG850 proxy
│   ├── 02_features.py             F1–F5 feature construction + screening
│   ├── 03_train.py                year-grouped CV + chronological holdout
│   ├── 04_validate.py             LOYO + calibration + DeLong/Wilcoxon
│   ├── 05_shap.py                 TreeSHAP (XGB+F4) + figures
│   ├── 06_diurnal.py              post-sounding 6-hour time-of-day targets
│   ├── 06b_cross_mountain.py      additional eastern gauge (Nakhon Si Thammarat)
│   └── model_common.py, station_metadata.py, ...  shared/support modules
│
├── notebooks/
│   └── demo_workflow.ipynb        end-to-end demo with sample data
│
└── outputs/
    ├── README.md                  expected output files
    ├── figures/                   Figures 1–9 as used in the manuscript
    └── Supplementary_Tables/      Tables S1–S9 (aggregate statistics only)
```
---

## Data Access

**Sounding data:** Publicly available from the NOAA NCEI Integrated Global
Radiosonde Archive (IGRA v2.2). Script `01_preprocess.py` downloads and
parses it automatically; indices are recomputed locally with MetPy rather
than taken from any pre-processed third-party product.

**Terrain data:** Publicly available from the Copernicus DEM (GLO-30).
Provenance recorded in `data/external/DATA_DOWNLOAD_PROVENANCE.md`; the DEM
tiles themselves are not committed to this repository (see that file for the
download source) and are read by `01b_terrain.py`.

**Rainfall data:** Obtained from the Thai Meteorological Department (TMD) under
a data-sharing agreement. Raw data cannot be redistributed. Contact TMD
(www.tmd.go.th) for access. A sample processed dataset (no raw TMD values)
is provided in `data/sample/`.

---

## Station

| Parameter | Value |
|---|---|
| Sounding station | Phuket Airport (WMO 48565, station THM00048565) |
| Radiosonde | InterMet iMet-2 (2011–2021; project/TMD-provided metadata) |
| Observation time | 00 UTC (07:00 LT, UTC+7) |
| Primary period | 2011–2021 (1,487 of 2,631 available profiles passed QC) |
| Primary rain-gauge stations | Phuket (30.6 km), Krabi (72.0 km), Takua Pa / Phang-nga (60.3 km) |
| Additional eastern gauge | Nakhon Si Thammarat (183.3 km, Gulf of Thailand side) |

---

## Requirements

Python 3.11. See `environment.yml` or `requirements.txt`.

Key packages: scikit-learn 1.4, xgboost 2.0, shap 0.46, pandas 2.1,
numpy 1.26, matplotlib 3.8, python-docx 1.2, metpy 1.6, rasterio 1.3,
openpyxl 3.1, requests, tqdm.
