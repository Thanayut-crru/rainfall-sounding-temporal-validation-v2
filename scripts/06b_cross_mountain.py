"""Station-specific remote-sounding test at Nakhon Si Thammarat.

Each model is trained on Nakhon's historical daily rainfall labels. This is NOT
a frozen west-to-east model-transfer experiment and does not isolate terrain.
"""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from model_common import (BASE, OUT, MODELS, TRAIN_YEARS, TEST_YEARS, RAIN_THRESH,
                          fit_predict, write_provenance)
from station_metadata import station_distance_km

STATION = "nakhon_si_thammarat"
COL = "RF_nakhon_si_thammarat"
EXPERIMENT = "station_specific_remote_sounding_test"


def fit(name, train, test, columns, thermo):
    y = (train[COL] >= RAIN_THRESH).astype(int).to_numpy()
    return fit_predict(name, train, test, columns, thermo, y)


def main():
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fs = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    df = pd.read_csv(OUT / "dataset_final.csv", parse_dates=["date"])
    east_path = OUT / 'rainfall_daily_east_aligned.csv'
    if not east_path.exists():
        east_path = BASE / 'data/raw/rainfall_tmd_daily_east.csv'
    east = pd.read_csv(east_path, parse_dates=["date"])[["date", COL]]
    orog_col = "OROG850_nakhon_si_thammarat"
    forcing = pd.read_csv(OUT / "orographic_forcing_daily.csv", parse_dates=["date"])[["date", orog_col]]
    # Explicit authoritative sources avoid duplicate-column suffixes on reruns.
    df = df.drop(columns=[c for c in [COL, orog_col] if c in df])
    df = df.merge(east, on="date", how="left", validate="one_to_one")
    df = df.merge(forcing, on="date", how="left", validate="one_to_one")
    df = df.dropna(subset=[COL]).reset_index(drop=True)
    df["year"] = df.date.dt.year
    train, test = df[df.year.isin(TRAIN_YEARS)], df[df.year.isin(TEST_YEARS)]
    yte = (test[COL] >= RAIN_THRESH).astype(int).to_numpy()
    if train.year.nunique() < 2 or np.unique(yte).size != 2:
        raise ValueError("Insufficient Nakhon training years or holdout rain classes")
    sets = {"F4": fs["F4"], "F5": fs["F4"] + [orog_col]}
    rows, predictions, fold_rows = [], [], []
    for feature_set, columns in sets.items():
        for name in MODELS:
            scores = []
            for fold, (ti, vi) in enumerate(GroupKFold(min(5, train.year.nunique())).split(train, groups=train.year)):
                tr, va = train.iloc[ti], train.iloc[vi]
                y = (va[COL] >= RAIN_THRESH).astype(int).to_numpy()
                p, *_ = fit(name, tr, va, columns, set(fs["F1"]))
                score = roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
                scores.append(score)
                fold_rows.append({"station": STATION, "experiment_type": EXPERIMENT,
                                  "feature_set": feature_set, "model": name, "fold": fold, "cv_auc": score,
                                  "validation_years": ";".join(map(str, sorted(va.year.unique())))})
            p, model, _, xte, keep = fit(name, train, test, columns, set(fs["F1"]))
            prediction = model.predict(xte)
            rows.append({"station": STATION, "experiment_type": EXPERIMENT, "frozen_transfer": False,
                         "training_label_station": STATION, "distance_km": station_distance_km(STATION),
                         "feature_set": feature_set, "model": name, "n_train": len(train), "n_test": len(test),
                         "observed_train_years": ";".join(map(str, sorted(train.year.unique()))),
                         "rain_rate_train": (train[COL] >= RAIN_THRESH).mean(), "rain_rate_test": yte.mean(),
                         "cv_auc_mean": np.nanmean(scores), "cv_auc_std": np.nanstd(scores),
                         "test_auc": roc_auc_score(yte, p), "test_pr_auc": average_precision_score(yte, p),
                         "test_brier": brier_score_loss(yte, p),
                         "test_macro_f1": f1_score(yte, prediction, average="macro", zero_division=0),
                         "retained_features": ";".join(keep)})
            predictions.extend({"station": STATION, "experiment_type": EXPERIMENT, "date": date,
                                "feature_set": feature_set, "model": name, "y": int(y),
                                "probability": float(prob), "prediction": int(pred)}
                               for date, y, prob, pred in zip(test.date, yte, p, prediction))
            print(f"Nakhon {name}+{feature_set}: CV={np.nanmean(scores):.4f}, holdout={roc_auc_score(yte,p):.4f}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "cross_mountain_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "cross_mountain_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "cross_mountain_cv.csv", index=False)
    loyo, status = [], []
    for year in TRAIN_YEARS:
        tr, va = train[train.year != year], train[train.year == year]
        y = (va[COL] >= RAIN_THRESH).astype(int).to_numpy()
        reason = "evaluated" if len(va) >= 5 and np.unique(y).size == 2 else "insufficient_records_or_rain_classes"
        if len(va) == 0:
            reason = "no_station_matched_records_after_upstream_filters"
        status.append({"station": STATION, "hold_year": year, "n": len(y), "status": reason})
        if reason != "evaluated":
            continue
        for name in MODELS:
            p, *_ = fit(name, tr, va, fs["F4"], set(fs["F1"]))
            loyo.append({"station": STATION, "experiment_type": EXPERIMENT, "feature_set": "F4",
                         "hold_year": year, "model": name, "auc": roc_auc_score(y, p), "n": len(y)})
    pd.DataFrame(loyo).to_csv(OUT / "cross_mountain_loyo.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "cross_mountain_loyo_year_status.csv", index=False)
    probability, model, _, xte, keep = fit("XGB", train, test, fs["F4"], set(fs["F1"]))
    reference = pd.DataFrame(predictions)
    reference = reference[(reference.model == "XGB") & (reference.feature_set == "F4")]
    np.testing.assert_allclose(reference.probability.to_numpy(), probability, rtol=0, atol=1e-10)
    explainer = shap.TreeExplainer(model, model_output="raw")
    values = np.asarray(explainer.shap_values(xte))
    np.testing.assert_allclose(values.sum(axis=1) + float(np.asarray(explainer.expected_value).ravel()[0]),
                               model.predict(xte, output_margin=True), atol=1e-4, rtol=1e-4)
    pd.DataFrame({"station": STATION, "experiment_type": EXPERIMENT, "model": "XGB", "feature_set": "F4",
                  "feature": keep, "mean_abs_shap": np.abs(values).mean(axis=0),
                  "mean_signed_shap": values.mean(axis=0), "shap_units": "log_odds"}).to_csv(
                      OUT / "cross_mountain_shap.csv", index=False)
    west = pd.read_csv(OUT / "binary_test_results.csv")
    west = west[(west.Feature_Set == "F4") & west.Model.isin(MODELS)][["station", "Model", "Test_AUC"]]
    west = west.rename(columns={"Model": "model", "Test_AUC": "auc"})
    east_plot = results[results.feature_set == "F4"][["station", "model", "test_auc"]].rename(columns={"test_auc": "auc"})
    plot = pd.concat([west, east_plot], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, q in plot.groupby("station", sort=False):
        q = q.set_index("model").reindex(MODELS)
        ax.plot(MODELS, q.auc, "o-", label=name.replace("_", " "))
    ax.set(ylabel="2020-2021 ROC-AUC", title="F4 models trained separately for each rainfall station\nNot frozen-model spatial transfer")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_cross_mountain.png", dpi=300)
    plt.close(fig)
    write_provenance(OUT, __file__, {"experiment_type": EXPERIMENT, "frozen_transfer": False,
                                    "training_label_station": STATION, "sounding_station": "Phuket Airport",
                                    "n_train": len(train), "n_test": len(test),
                                    "observed_train_years": sorted(train.year.unique().tolist()),
                                    "rainfall_source":str(east_path),
                                    "comparison_caveat": "Different distance, exposure, rainfall source and matched samples; no causal terrain isolation"})


if __name__ == "__main__":
    main()
