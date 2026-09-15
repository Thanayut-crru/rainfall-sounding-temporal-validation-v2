"""Binary rainfall classification: year-blocked CV and chronological holdout.

F1-F4 are the primary comparison; F5 is a reviewer-requested terrain sensitivity.
No multiclass results are generated or claimed by this script.
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from model_common import (OUT, MODELS, TRAIN_YEARS, TEST_YEARS, RAIN_THRESH,
                          climate_probability, fit_predict, select_cv_configurations,
                          write_provenance)
from station_metadata import station_distance_km

N_FOLDS = 5
STATIONS = {name: {"col": f"RF_{name}", "dist_km": station_distance_km(name)}
            for name in ("phuket", "krabi", "phangnga")}


def metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"Test_AUC": roc_auc_score(y, probability) if np.unique(y).size > 1 else np.nan,
            "Test_F1": f1_score(y, prediction, average="macro", zero_division=0),
            "Test_Recall": recall_score(y, prediction, zero_division=0),
            "Test_Precision": precision_score(y, prediction, zero_division=0),
            "Test_Specificity": tn / (tn + fp) if tn + fp else np.nan}


def run_cv(train, columns, thermo, col, name):
    y = (train[col] >= RAIN_THRESH).astype(int).to_numpy()
    n_folds = min(N_FOLDS, train.year.nunique())
    if n_folds < 2:
        raise ValueError("At least two observed training years are required")
    scores, details = [], []
    for fold, (ti, vi) in enumerate(GroupKFold(n_folds).split(train, y, groups=train.year)):
        tr, va = train.iloc[ti], train.iloc[vi]
        if set(tr.year) & set(va.year):
            raise AssertionError("Year-blocked CV partitions overlap")
        if name == "Climatological":
            p = climate_probability(tr, va, col)
            keep = []
        else:
            p, _, _, _, keep = fit_predict(name, tr, va, columns, thermo, y[ti])
        auc = roc_auc_score(y[vi], p) if np.unique(y[vi]).size == 2 else np.nan
        scores.append(auc)
        details.append({"fold": fold, "CV_AUC": auc, "n_train": len(tr), "n_validation": len(va),
                        "train_years": ";".join(map(str, sorted(tr.year.unique()))),
                        "validation_years": ";".join(map(str, sorted(va.year.unique()))),
                        "retained_features": ";".join(keep)})
    return scores, details


def run_holdout(train, test, columns, thermo, col, name):
    ytr = (train[col] >= RAIN_THRESH).astype(int).to_numpy()
    yte = (test[col] >= RAIN_THRESH).astype(int).to_numpy()
    if name == "Climatological":
        p = climate_probability(train, test, col)
        prediction, keep = (p >= 0.5).astype(int), []
    else:
        p, model, _, xte, keep = fit_predict(name, train, test, columns, thermo, ytr)
        prediction = model.predict(xte)
    result = metrics(yte, prediction, p)
    result.update(N_Features=len(keep), Features_Dropped=";".join(c for c in columns if c not in keep),
                  Features_Retained=";".join(keep))
    if name == "Climatological":
        result["Features_Dropped"] = ""
    return result, p, prediction


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "dataset_final.csv", parse_dates=["date"])
    if df.date.duplicated().any():
        raise ValueError("dataset_final.csv must contain exactly one record per date")
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    fs = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    feature_sets = {k: v for k, v in fs.items() if k in ["F1", "F2", "F3", "F4", "F5"]}
    thermo = set(fs["F1"])
    cv_rows, test_rows, split_rows, predictions = [], [], [], []
    for station, info in STATIONS.items():
        col, distance = info["col"], info["dist_km"]
        if col not in df:
            raise ValueError(f"Missing main-station rainfall column: {col}")
        station_df = df[df[col].notna()].copy()
        train = station_df[station_df.year.isin(TRAIN_YEARS)].reset_index(drop=True)
        test = station_df[station_df.year.isin(TEST_YEARS)].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"Empty training or holdout partition: {station}")
        ytr = (train[col] >= RAIN_THRESH).astype(int).to_numpy()
        yte = (test[col] >= RAIN_THRESH).astype(int).to_numpy()
        rr_tr, rr_te = ytr.mean() * 100, yte.mean() * 100
        split_rows.append({"station": station, "dist_km": distance, "n_train": len(train), "n_test": len(test),
                           "rain_rate_train": rr_tr, "rain_rate_test": rr_te,
                           "n_observed_train_years": train.year.nunique(),
                           "observed_train_years": ";".join(map(str, sorted(train.year.unique()))),
                           "missing_nominal_train_years": ";".join(map(str, sorted(set(TRAIN_YEARS) - set(train.year))))})
        print(f"{station}: train={len(train)}, test={len(test)}, training years={sorted(train.year.unique())}", flush=True)
        for feature_set, raw_columns in feature_sets.items():
            columns = [c for c in raw_columns if c in station_df and
                       (not c.startswith("OROG850_") or c == f"OROG850_{station}")]
            if not columns:
                raise ValueError(f"No columns for {station}/{feature_set}")
            for name in ["Climatological"] + MODELS:
                start = time.time()
                scores, details = run_cv(train, columns, thermo, col, name)
                holdout, probability, prediction = run_holdout(train, test, columns, thermo, col, name)
                common = {"experiment": "terrain_sensitivity" if feature_set == "F5" else "main",
                          "station": station, "dist_km": distance, "Feature_Set": feature_set, "Model": name}
                cv_rows.extend(dict(common, **detail) for detail in details)
                test_rows.append(dict(common, n_train=len(train), n_test=len(test), rain_rate_train=rr_tr,
                                      rain_rate_test=rr_te, CV_AUC_mean=np.nanmean(scores),
                                      CV_AUC_std=np.nanstd(scores), **holdout, Time_s=time.time() - start))
                predictions.extend({"date": date, "station": station, "model": name, "feature_set": feature_set,
                                    "y": int(y), "probability": float(p), "prediction": int(pred)}
                                   for date, y, p, pred in zip(test.date, yte, probability, prediction))
                print(f"  {feature_set}/{name}: CV={np.nanmean(scores):.4f}, holdout={holdout['Test_AUC']:.4f}", flush=True)
    results = pd.DataFrame(test_rows)
    pd.DataFrame(cv_rows).to_csv(OUT / "binary_cv_results.csv", index=False)
    results.to_csv(OUT / "binary_test_results.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "data_split_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "holdout_predictions_main.csv", index=False)
    selected = select_cv_configurations(results)
    selected["selection_rule"] = "maximum_mean_year_blocked_CV_AUC_F1_F4; ties feature_set_then_model; no_holdout_selection"
    selected.to_csv(OUT / "cv_selected_configurations.csv", index=False)
    write_provenance(OUT, __file__, {"cv_folds_requested": N_FOLDS,
                                    "selection_rule": selected.selection_rule.iloc[0],
                                    "actual_station_splits": split_rows})
    print("Saved main experiments, exact holdout predictions and CV-selected configurations.", flush=True)


if __name__ == "__main__":
    main()
