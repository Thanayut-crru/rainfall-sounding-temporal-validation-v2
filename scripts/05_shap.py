"""Held-out TreeSHAP for the shared XGB+F4 nonlinear comparator.

Attributions describe model associations, not verified causal mechanisms.
"""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from model_common import (OUT, TRAIN_YEARS, TEST_YEARS, RAIN_THRESH,
                          assert_main_predictions, fit_predict, write_provenance)

STATIONS = {"phuket": ("RF_phuket", "Phuket"), "krabi": ("RF_krabi", "Krabi"),
            "phangnga": ("RF_phangnga", "Takua Pa (Phang-nga)")}


def main():
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "dataset_final.csv", parse_dates=["date"])
    df["year"] = df.date.dt.year
    fs = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    rows, station_features, phuket = [], {}, None
    for station, (col, label) in STATIONS.items():
        s = df[df[col].notna()]
        train, test = s[s.year.isin(TRAIN_YEARS)], s[s.year.isin(TEST_YEARS)]
        ytr = (train[col] >= RAIN_THRESH).astype(int).to_numpy()
        yte = (test[col] >= RAIN_THRESH).astype(int).to_numpy()
        probability, model, _, xte, keep = fit_predict("XGB", train, test, fs["F4"], set(fs["F1"]), ytr)
        assert_main_predictions(OUT, station, "XGB", "F4", test.date, yte, probability)
        explainer = shap.TreeExplainer(model, model_output="raw")
        values = np.asarray(explainer.shap_values(xte))
        if values.shape != xte.shape:
            raise ValueError(f"Unexpected binary TreeSHAP shape: {values.shape}")
        margin = model.predict(xte, output_margin=True)
        np.testing.assert_allclose(values.sum(axis=1) + float(np.asarray(explainer.expected_value).ravel()[0]),
                                   margin, atol=1e-4, rtol=1e-4)
        importance = np.abs(values).mean(axis=0)
        station_features[station] = keep
        for i, feature in enumerate(keep):
            rows.append({"station": station, "model": "XGB", "feature_set": "F4", "feature": feature,
                         "mean_abs_shap": float(importance[i]), "mean_signed_shap": float(values[:, i].mean()),
                         "n_train": len(train), "n_test": len(test), "shap_units": "log_odds",
                         "main_predictions_verified": True})
        if station == "phuket":
            phuket = values, xte, keep
        print(f"{label}: verified XGB+F4; leading attribution {keep[int(np.argmax(importance))]}", flush=True)
    ranks = pd.DataFrame(rows)
    ranks.to_csv(OUT / "shap_feature_rank.csv", index=False)
    # Union of retained features prevents hiding features retained only outside Phuket.
    order = ranks.groupby("feature").mean_abs_shap.mean().sort_values(ascending=False).index.tolist()
    x, width = np.arange(len(order)), 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (station, (_, label)) in enumerate(STATIONS.items()):
        subset = ranks[ranks.station == station].set_index("feature")
        values = [subset.loc[feature, "mean_abs_shap"] if feature in subset.index else np.nan for feature in order]
        ax.bar(x + i * width, values, width, label=label, color=["#0072B2", "#009E73", "#D55E00"][i])
    ax.set_xticks(x + width, order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean absolute SHAP value (log-odds)")
    ax.set_title("XGB+F4, 2020-2021 holdout\nAbsent bars indicate predictors removed by station-specific training screening")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_shap_barplot.png", dpi=300)
    plt.close(fig)
    if phuket is not None:
        values, xte, keep = phuket
        plt.figure(figsize=(9, 7))
        shap.summary_plot(values, xte, feature_names=keep, show=False, plot_type="dot")
        plt.title("XGB+F4, Phuket holdout 2020-2021", fontsize=11)
        plt.xlabel("SHAP value (change in model log-odds)")
        plt.tight_layout()
        plt.savefig(figure_dir / "fig_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
    write_provenance(OUT, __file__, {"model": "XGB", "feature_set": "F4", "shap_units": "log_odds",
                                    "retained_features": station_features,
                                    "all_holdout_probabilities_verified_against_main": True,
                                    "interpretation_scope": "model associations; not causal proof or independent terrain attribution"})


if __name__ == "__main__":
    main()
