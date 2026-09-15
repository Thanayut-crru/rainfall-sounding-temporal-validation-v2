# Scripts

Numbered analysis scripts used to reproduce the modelling workflow described
in `PIPELINE.md`.

The expected run order is:

1. `00_import_rainfall.py`, `00b_import_daily.py`, `00c_temporal_alignment.py`
2. `01_preprocess.py`, `01b_terrain.py`
3. `02_features.py`
4. `03_train.py`
5. `04_validate.py`
6. `05_shap.py`
7. `06_diurnal.py`, `06b_cross_mountain.py`

Supporting modules: `model_common.py`, `station_metadata.py`,
`refresh_distance_metadata.py`, `figure1_station_map.py`.

Unit tests: `test_models.py`, `test_physics.py`, `test_station_metadata.py`,
`test_temporal_alignment.py`.

Raw TMD rainfall observations are not redistributed in this repository.
