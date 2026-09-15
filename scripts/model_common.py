"""One model/preprocessing definition for every revision experiment.

Model settings reproduce the original main-training script (03_train.py).
The MLP's internal random 10% stopping subset is drawn only from the current
training partition; it is not a temporally independent validation result.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

BASE = Path(os.environ.get("RAINFALL_PROJECT_DIR", Path(__file__).resolve().parent.parent))
OUT = Path(os.environ.get("RAINFALL_OUTPUT_DIR", BASE / "outputs"))
SEED = 42
TRAIN_YEARS = list(range(2011, 2020))
TEST_YEARS = [2020, 2021]
RAIN_THRESH = 0.1
MODELS = ["LR", "SVM", "RF", "XGB", "MLP"]
SCALE_MODELS = {"LR", "SVM", "MLP"}
MODEL_CONFIG_VERSION = "2026-09-07-shared-original03-v1"


def make_model(name, y):
    """Return an estimator with identical settings in all experiments."""
    y = np.asarray(y)
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    if name == "SVM":
        return SVC(probability=True, kernel="rbf", random_state=SEED, class_weight="balanced")
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=SEED,
                                      class_weight="balanced", n_jobs=-1)
    if name == "XGB":
        n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
        return xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=SEED, verbosity=0,
                                 scale_pos_weight=float(n_neg / n_pos) if n_pos else 1.0)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=SEED, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=20)
    raise ValueError(f"Unknown model: {name}")


def prepare_matrices(tr, te, columns, thermo, corr=0.9):
    """Return Xtrain, Xeval, kept columns; fit every operation on train only.

    Nonfinite values are missing. Correlation screening applies only to
    thermodynamic candidates, in declared column order, followed by removal of
    all-missing training columns and training-mean imputation.
    """
    columns = list(dict.fromkeys(c for c in columns if c in tr.columns))
    missing_eval = [c for c in columns if c not in te.columns]
    if missing_eval:
        raise ValueError(f"Evaluation frame lacks columns: {missing_eval}")
    a = tr[columns].replace([np.inf, -np.inf], np.nan)
    b = te[columns].replace([np.inf, -np.inf], np.nan)
    therm = [c for c in columns if c in thermo]
    matrix = a[therm].corr().abs()
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    drop = {c for c in upper if (upper[c] >= corr).any()}
    keep = [c for c in columns if c not in drop]
    means = a[keep].mean()
    keep = [c for c in keep if pd.notna(means[c])]
    if not keep:
        raise ValueError("No usable predictors in the current training partition")
    means = means[keep]
    xtr, xte = a[keep].fillna(means).to_numpy(dtype=float), b[keep].fillna(means).to_numpy(dtype=float)
    if not (np.isfinite(xtr).all() and np.isfinite(xte).all()):
        raise ValueError("Nonfinite predictors remained after training-only imputation")
    return xtr, xte, keep


def fit_predict(name, tr, te, columns, thermo, y, corr=0.9):
    """Return probabilities, fitted estimator, transformed Xtrain/Xeval, keep."""
    y = np.asarray(y)
    if np.unique(y).size != 2:
        raise ValueError("A classifier training partition must contain both classes")
    a, b, keep = prepare_matrices(tr, te, columns, thermo, corr)
    if name in SCALE_MODELS:
        scaler = StandardScaler()
        a, b = scaler.fit_transform(a), scaler.transform(b)
    model = make_model(name, y)
    model.fit(a, y)
    return model.predict_proba(b)[:, 1], model, a, b, keep


def climate_probability(tr, te, col):
    frequencies = tr.groupby("month")[col].apply(lambda x: (x >= RAIN_THRESH).mean())
    return te["month"].map(frequencies).fillna((tr[col] >= RAIN_THRESH).mean()).to_numpy()


def select_cv_configurations(results):
    """Select from F1-F4 on mean year-blocked CV AUC; never use holdout AUC."""
    eligible = results[results.Feature_Set.isin(["F1", "F2", "F3", "F4"])
                       & results.Model.isin(MODELS)].copy()
    eligible = eligible.dropna(subset=["CV_AUC_mean"])
    # Explicit deterministic tie breaking, independent of holdout values.
    eligible = eligible.sort_values(["station", "CV_AUC_mean", "Feature_Set", "Model"],
                                     ascending=[True, False, True, True])
    return eligible.groupby("station", sort=False).head(1).reset_index(drop=True)


def assert_main_predictions(out, station, model, feature_set, dates, y, probability):
    """Fail if a refitted diagnostic disagrees with the main experiment."""
    reference = pd.read_csv(Path(out) / "holdout_predictions_main.csv", parse_dates=["date"])
    reference = reference[(reference.station == station) & (reference.model == model)
                          & (reference.feature_set == feature_set)].copy()
    observed = pd.DataFrame({"date": pd.to_datetime(np.asarray(dates)), "y": y,
                             "probability_check": probability})
    if reference.date.duplicated().any() or observed.date.duplicated().any():
        raise AssertionError("Holdout dates must be unique within model/station/feature set")
    joined = reference.merge(observed, on="date", how="outer", validate="one_to_one", suffixes=("", "_check"))
    if len(joined) != len(observed) or joined[["y", "y_check", "probability", "probability_check"]].isna().any().any():
        raise AssertionError(f"Holdout rows differ from main experiment: {station}/{model}/{feature_set}")
    np.testing.assert_array_equal(joined.y, joined.y_check)
    np.testing.assert_allclose(joined.probability, joined.probability_check, rtol=0, atol=1e-10,
                               err_msg=f"Model mismatch: {station}/{model}/{feature_set}")
    from sklearn.metrics import roc_auc_score
    reported = pd.read_csv(Path(out) / "binary_test_results.csv")
    row = reported[(reported.station == station) & (reported.Model == model)
                   & (reported.Feature_Set == feature_set)]
    if len(row) != 1:
        raise AssertionError("Expected one main-result row")
    np.testing.assert_allclose(roc_auc_score(y, probability), row.iloc[0].Test_AUC, atol=1e-10, rtol=0)


def write_provenance(out, script_name, extra=None):
    """Save software, model settings and input hashes for each analysis stage."""
    out = Path(out)
    packages = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "statsmodels"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    hashes = {}
    for filename in ["dataset_final.csv", "feature_sets.json"]:
        p = out / filename
        if p.exists():
            hashes[filename] = hashlib.sha256(p.read_bytes()).hexdigest()
    payload = {"model_config_version": MODEL_CONFIG_VERSION, "script": script_name,
               "python": platform.python_version(), "packages": packages,
               "seed": SEED, "nominal_train_years": TRAIN_YEARS, "test_years": TEST_YEARS,
               "parameters": {name: make_model(name, [0, 1]).get_params() for name in MODELS},
               "xgb_weight_note": "scale_pos_weight recomputed from each training partition",
               "mlp_stopping_note": "random internal stopping split within training only; not year-blocked",
               "screening": "thermodynamic absolute Pearson correlation >=0.90, train only",
               "imputation": "training-column mean after removing all-missing training columns",
               "input_sha256": hashes,
               "model_common_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if extra:
        payload.update(extra)
    path = out / f"{Path(script_name).stem}_provenance.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
