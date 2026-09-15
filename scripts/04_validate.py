"""Consistent F4 diagnostics, CV-selected LOYO, calibration and paired tests."""
from __future__ import annotations

import itertools
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.stats.multitest import multipletests
from model_common import (OUT, MODELS, SEED, TRAIN_YEARS, TEST_YEARS, RAIN_THRESH,
                          assert_main_predictions, climate_probability, fit_predict,
                          select_cv_configurations, write_provenance)

STATIONS = {"phuket": "RF_phuket", "krabi": "RF_krabi", "phangnga": "RF_phangnga"}
LABELS = {"phuket": "Phuket", "krabi": "Krabi", "phangnga": "Takua Pa (Phang-nga)"}


def exact_signed_rank(difference):
    """Two-sided Wilcoxon sign enumeration; exact with small n, ties and zeros.

    Differences rounded at1e-12 only to suppress floating subtraction noise.
    The symmetry/exchangeable-sign assumption still applies.
    """
    d=np.round(np.asarray(difference,dtype=float),12)
    d=d[d!=0]
    if not len(d):return 0.0,1.0
    if len(d)>20:raise ValueError('Exact enumeration restricted to at most20 nonzero annual pairs')
    ranks=stats.rankdata(np.abs(d),method='average')
    total=float(ranks.sum());positive=float(ranks[d>0].sum())
    observed=min(positive,total-positive)
    masks=((np.arange(2**len(d))[:,None] >> np.arange(len(d))) & 1)
    sums=masks@ranks
    probability=float(np.mean(np.minimum(sums,total-sums)<=observed+1e-12))
    return observed,probability


def predict(name, train, evaluation, col, columns, thermo):
    y = (train[col] >= RAIN_THRESH).astype(int).to_numpy()
    return fit_predict(name, train, evaluation, columns, thermo, y)[0]


def bootstrap_ci(y, p, n=2000):
    """Paired observation bootstrap; descriptive CI, not a temporal-block CI."""
    rng, estimates = np.random.default_rng(SEED), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            estimates.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(estimates, [2.5, 97.5]) if estimates else [np.nan, np.nan]


def midrank(x):
    order = np.argsort(x)
    sorted_x, ranks = np.asarray(x)[order], np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    answer = np.empty(len(x))
    answer[order] = ranks
    return answer


def delong(y, a, b):
    """Correlated ROC-AUC comparison using tied-score midranks."""
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("DeLong requires binary labels")
    order = np.argsort(-y)
    probabilities = np.vstack([a, b])[:, order]
    m, n = int(y.sum()), len(y) - int(y.sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong requires at least two positive and two negative observations")
    tx = np.array([midrank(x) for x in probabilities[:, :m]])
    ty = np.array([midrank(x) for x in probabilities[:, m:]])
    tz = np.array([midrank(x) for x in probabilities])
    auc = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    np.testing.assert_allclose(auc, [roc_auc_score(y, a), roc_auc_score(y, b)], atol=1e-12)
    v01, v10 = (tz[:, :m] - tx) / n, 1 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    contrast = np.array([1, -1])
    variance = float(contrast @ covariance @ contrast)
    if variance <= 1e-15:
        if abs(auc[0] - auc[1]) <= 1e-15:
            return auc, 0.0, 1.0
        return auc, np.copysign(np.inf, auc[0] - auc[1]), 0.0
    z = (auc[0] - auc[1]) / np.sqrt(variance)
    return auc, z, float(2 * stats.norm.sf(abs(z)))


def apply_bh(frame):
    frame = frame.copy()
    valid = frame.p_raw.notna()
    frame["p_bh"] = np.nan
    if valid.any():
        frame.loc[valid, "p_bh"] = multipletests(frame.loc[valid, "p_raw"], method="fdr_bh")[1]
    frame["reject_bh_0.05"] = frame.p_bh < 0.05
    frame["family_size_planned"] = len(frame)
    frame["family_size_computable"] = int(valid.sum())
    return frame


def loyo_analysis(df, fs, selected):
    common_rows, selected_rows, status_rows = [], [], []
    for station, col in STATIONS.items():
        s = df[df[col].notna()].copy()
        choice = selected[selected.station == station].iloc[0]
        for year in TRAIN_YEARS:
            train = s[s.year.isin(TRAIN_YEARS) & (s.year != year)]
            validation = s[s.year == year]
            y = (validation[col] >= RAIN_THRESH).astype(int).to_numpy()
            reason = "evaluated"
            if len(validation) == 0:
                reason = "no_station_matched_records_after_upstream_filters"
            elif len(validation) < 5:
                reason = "fewer_than_five_validation_records"
            elif np.unique(y).size != 2:
                reason = "validation_year_has_only_one_rainfall_class"
            elif (train[col] >= RAIN_THRESH).nunique() != 2:
                reason = "training_partition_has_only_one_rainfall_class"
            status_rows.append({"station": station, "hold_year": year, "n": len(y),
                                "n_positive": int(y.sum()), "n_negative": len(y) - int(y.sum()),
                                "status": reason})
            if reason != "evaluated":
                continue
            by_model = {}
            for name in MODELS:
                p = predict(name, train, validation, col, fs["F4"], set(fs["F1"]))
                row = {"station": station, "model": name, "feature_set": "F4",
                       "hold_year": year, "auc": roc_auc_score(y, p), "n": len(y)}
                common_rows.append(row)
                by_model[name] = row
            if choice.Feature_Set == "F4":
                row = dict(by_model[choice.Model])
            else:
                p = predict(choice.Model, train, validation, col, fs[choice.Feature_Set], set(fs["F1"]))
                row = {"station": station, "model": choice.Model, "feature_set": choice.Feature_Set,
                       "hold_year": year, "auc": roc_auc_score(y, p), "n": len(y)}
            row["selection_basis"] = "full_training_year_blocked_CV; post_selection_year_sensitivity_not_nested_selection_estimate"
            selected_rows.append(row)
            print(f"LOYO {station}/{year}: F4 models and {choice.Model}+{choice.Feature_Set}", flush=True)
    return pd.DataFrame(common_rows), pd.DataFrame(selected_rows), pd.DataFrame(status_rows)


def draw_figures(loyo, selected_loyo, predictions, figure_dir):
    colors = dict(zip(MODELS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))
    markers = dict(zip(MODELS, ["o", "s", "^", "D", "v"]))
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
    for ax, station in zip(axes, STATIONS):
        for name in MODELS:
            q = loyo[(loyo.station == station) & (loyo.model == name)].set_index("hold_year").reindex(TRAIN_YEARS)
            ax.plot(q.index, q.auc, color=colors[name], marker=markers[name], label=f"{name}+F4", lw=1.4, ms=4)
        q = selected_loyo[selected_loyo.station == station].sort_values("hold_year")
        if not q.empty:
            choice = q.iloc[0]
            if choice.feature_set != "F4":
                q = q.set_index("hold_year").reindex(TRAIN_YEARS)
                ax.plot(q.index, q.auc, "k*--", lw=1.8, ms=8, label=f"CV-selected {choice.model}+{choice.feature_set}")
            else:
                ax.text(0.02, 0.02, f"CV-selected: {choice.model}+F4", transform=ax.transAxes, fontsize=9)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(LABELS[station])
        ax.set_xlabel("Held-out year")
        ax.set_xticks(TRAIN_YEARS[::2])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5,1.02))
    axes[0].set_ylabel("LOYO ROC-AUC")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_loyo.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, station in zip(axes, STATIONS):
        for name in ["LR", "SVM", "XGB"]:
            q = predictions[(predictions.station == station) & (predictions.model == name)]
            fraction, mean = calibration_curve(q.y, q.probability, n_bins=8, strategy="quantile")
            ax.plot(mean, fraction, "o-", label=f"{name}+F4")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title(LABELS[station])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed rain frequency")
    axes[1].set_xlabel("Predicted rain probability")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_reliability.png", dpi=300)
    plt.close(fig)


def main():
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "dataset_final.csv", parse_dates=["date"])
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    fs = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    selected = select_cv_configurations(pd.read_csv(OUT / "binary_test_results.csv"))
    loyo, selected_loyo, year_status = loyo_analysis(df, fs, selected)
    loyo.to_csv(OUT / "loyo_results.csv", index=False)
    selected_loyo.to_csv(OUT / "loyo_selected_results.csv", index=False)
    year_status.to_csv(OUT / "loyo_year_status.csv", index=False)
    prediction_rows, calibration, seasonal = [], [], []
    for station, col in STATIONS.items():
        s = df[df[col].notna()]
        train, test = s[s.year.isin(TRAIN_YEARS)], s[s.year.isin(TEST_YEARS)]
        y = (test[col] >= RAIN_THRESH).astype(int).to_numpy()
        probabilities = {"Climatological": climate_probability(train, test, col)}
        for name in MODELS:
            probabilities[name] = predict(name, train, test, col, fs["F4"], set(fs["F1"]))
        for name, p in probabilities.items():
            assert_main_predictions(OUT, station, name, "F4", test.date, y, p)
            feature_set = "baseline" if name == "Climatological" else "F4"
            lo, hi = bootstrap_ci(y, p)
            calibration.append({"station": station, "model": name, "feature_set": feature_set, "n": len(y),
                                "roc_auc": roc_auc_score(y, p), "ci_lo": lo, "ci_hi": hi,
                                "ci_method": "paired_observation_bootstrap_not_temporal_block",
                                "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
                                "main_predictions_verified": True})
            prediction_rows.extend({"date": date, "station": station, "y": int(yy), "model": name,
                                    "feature_set": feature_set, "probability": float(pp)}
                                   for date, yy, pp in zip(test.date, y, p))
            for season, months in [("SW", [5, 6, 7, 8, 9, 10]), ("NE", [11, 12, 1, 2, 3, 4])]:
                use = test.month.isin(months).to_numpy()
                seasonal.append({"station": station, "model": name, "feature_set": feature_set,
                                 "season": season, "n": int(use.sum()), "n_years": test[use].year.nunique(),
                                 "auc": roc_auc_score(y[use], p[use]) if np.unique(y[use]).size == 2 else np.nan})
        print(f"Verified all F4 holdout predictions against03: {station}", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.DataFrame(calibration).to_csv(OUT / "calibration_results.csv", index=False)
    pd.DataFrame(seasonal).to_csv(OUT / "seasonal_results.csv", index=False)
    dl_rows, wx_rows = [], []
    for station in STATIONS:
        z = predictions[predictions.station == station]
        wide = z.pivot(index=["date", "y"], columns="model", values="probability")
        if wide[MODELS].isna().any().any():
            raise AssertionError("Paired significance tests require identical holdout rows")
        y = wide.index.get_level_values("y").to_numpy()
        years = loyo[loyo.station == station].pivot(index="hold_year", columns="model", values="auc")
        for a, b in itertools.combinations(MODELS, 2):
            auc, statistic, p = delong(y, wide[a].to_numpy(), wide[b].to_numpy())
            dl_rows.append({"station": station, "feature_set": "F4", "model_a": a, "model_b": b,
                            "n_observations": len(y), "auc_a": auc[0], "auc_b": auc[1], "z": statistic, "p_raw": p})
            q = years[[a, b]].dropna()
            difference = q[a] - q[b]
            if len(q) < 5:
                statistic, p, status = np.nan, np.nan, "insufficient_paired_years"
            elif np.all(difference == 0):
                statistic, p, status = 0.0, 1.0, "all_paired_differences_zero"
            else:
                statistic, p = exact_signed_rank(difference)
                status = "computed"
            wx_rows.append({"station": station, "feature_set": "F4", "model_a": a, "model_b": b,
                            "n_years": len(q), "paired_years": ";".join(map(str, q.index)),
                            "median_diff": difference.median(), "statistic": statistic, "p_raw": p, "test_status": status,
                            "method":"exact_two_sided_signed_rank_sign_enumeration_zeros_discarded"})
    dl, wx = apply_bh(pd.DataFrame(dl_rows)), apply_bh(pd.DataFrame(wx_rows))
    if len(dl) != 30 or len(wx) != 30:
        raise AssertionError("Expected10 F4 model pairs at each of3 stations in each test family")
    dl.to_csv(OUT / "delong_results.csv", index=False)
    wx.to_csv(OUT / "wilcoxon_results.csv", index=False)
    pd.concat([dl.assign(test="DeLong"), wx.assign(test="Wilcoxon")], ignore_index=True).to_csv(
        OUT / "significance_complete.csv", index=False)
    summary = {"delong_family_size": len(dl), "delong_significant_bh": int(dl["reject_bh_0.05"].sum()),
               "wilcoxon_family_size": len(wx), "wilcoxon_computable": int(wx.p_raw.notna().sum()),
               "wilcoxon_significant_bh": int(wx["reject_bh_0.05"].sum()),
               "common_f4_loyo_rows": len(loyo), "selected_configuration_loyo_rows": len(selected_loyo),
               "all_holdout_probabilities_verified_against_main": True,
               "actual_loyo_years": {s: sorted(loyo[loyo.station == s].hold_year.unique().tolist()) for s in STATIONS},
               "significant_delong_pairs": dl[dl["reject_bh_0.05"]].to_dict("records"),
               "selected_loyo_caveat": "post-selection sensitivity; not nested evaluation of the selection process",
               "inference_caveat": "DeLong and observation bootstrap do not adjust for serial dependence; LOYO has few, overlapping-training annual fits"}
    (OUT / "validation_verified_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_provenance(OUT, __file__, summary)
    draw_figures(loyo, selected_loyo, predictions, figure_dir)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
