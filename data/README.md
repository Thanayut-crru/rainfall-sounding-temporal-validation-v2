# Data

## Sounding Data

Publicly available from the University of Wyoming Upper Air Sounding Archive:
https://weather.uwyo.edu/upperair/

Script `01_preprocess.py` downloads data automatically for station WMO 48565.

The radiosonde used during the 2011–2021 study period was identified as an
InterMet iMet-2 from project/TMD-provided instrument information. The ground
receiving system and any processing details not contained in the University of
Wyoming processed products must be reported only after separate confirmation.

## Rainfall Data

Obtained from the Thai Meteorological Department (TMD) under a data-sharing
agreement. Raw data **cannot be redistributed**.

To replicate the full pipeline from raw inputs, contact TMD:
https://www.tmd.go.th

The analysis-ready daily file is produced from the original TMD 3-hour XLSX
workbook by `scripts/00_import_rainfall.py`. The station mapping used in the
paper is Phuket (564201), Krabi (566202), and Takua Pa in Phang-nga Province
(561201). See `PIPELINE.md` for the import command and missing-data rules.

## Sample Data (this folder)

`sample/sample_dataset.csv` — sample processed dataset containing sounding-derived
predictors (real WMO 48565 indices) and synthetic rainfall labels. These data allow
the demo workflow (`notebooks/demo_workflow.ipynb`) to run end-to-end without raw
TMD rainfall observations.

### sample_dataset.csv columns

| Column | Description |
|---|---|
| date | YYYY-MM-DD |
| SHOW | Showalter index |
| LIFT | Lifted index |
| SWET | SWEAT index |
| KINX | K-index |
| VTOT | Vertical totals index |
| CAPE | CAPE (J/kg) |
| CINS | Convective inhibition (J/kg) |
| PWAT | Precipitable water (mm) |
| WDcos850 | cos(850 hPa wind direction) |
| WDsin850 | sin(850 hPa wind direction) |
| WS850 | 850 hPa wind speed (kt) |
| RF_phuket | Synthetic daily rainfall label — Phuket (mm) |
| RF_krabi | Synthetic daily rainfall label — Krabi (mm) |
| RF_phangnga | Synthetic daily rainfall label — Phang-nga (mm) |
| ... | (see scripts/02_features.py for full column list) |

> **Note:** RF_* columns contain **synthetic** rainfall amounts derived from PWAT
> and seasonal signals. They replicate the statistical structure of the real TMD
> data but are not real observations. Use for demonstration only.
