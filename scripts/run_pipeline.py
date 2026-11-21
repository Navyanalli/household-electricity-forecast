"""Run the full household electricity forecasting pipeline end to end.

Steps: download + clean data -> train/test split -> benchmarks -> SARIMA ->
SARIMAX (weekend flag; measured variables as a conditional forecast) ->
XGBoost -> Chronos-Bolt -> metrics table and final forecast plot in results/.

Usage (from the repository root):
    python scripts/run_pipeline.py                     # loads the saved SARIMA grid search
    python scripts/run_pipeline.py --run-grid-search   # refits all 147 SARIMA models (~40 min)
    python scripts/run_pipeline.py --skip-chronos      # skip the foundation model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (FIGURE_DIR, FORECAST_DIR, GRID_D, GRID_P, GRID_Q, MEASURED,
                        METRICS_DIR, RESULTS_DIR, SARIMA_SEASONAL, SARIMA_WEEKS, TARGET,
                        TEST_DAYS)
from src.data_prep import load_hourly_data
from src.evaluation import evaluate_models, train_test_split
from src.models import (FEATURE_GROUPS, build_exog, chronos_forecast, fit_sarima,
                        make_features, rolling_benchmarks, rolling_forecast,
                        sarima_grid_search, train_xgb)


def main(run_grid_search=False, skip_chronos=False):
    for folder in [RESULTS_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Data and split
    df = load_hourly_data()
    train, test, split_point = train_test_split(df)
    series = df[TARGET]
    test_days = pd.date_range(split_point, periods=TEST_DAYS, freq="D")
    print(f"[data] {len(df):,} hourly rows | train {len(train):,} h | test {len(test)} h")

    forecasts = pd.DataFrame({"Actual": test[TARGET]})

    # 2. Benchmarks
    bench = rolling_benchmarks(series, test_days)
    forecasts = forecasts.join(bench)
    print("[benchmarks] done")

    # 3. SARIMA: AIC grid search (or load saved results), then rolling forecasts
    y_window = train[TARGET].iloc[-24 * 7 * SARIMA_WEEKS:].asfreq("h")
    grid_path = RESULTS_DIR / "sarima_grid_search.csv"
    if run_grid_search or not grid_path.exists():
        grid = sarima_grid_search(y_window, SARIMA_SEASONAL, grid_path, GRID_P, GRID_D, GRID_Q)
    else:
        grid = pd.read_csv(grid_path).sort_values("aic").reset_index(drop=True)
    order = tuple(int(v) for v in grid.loc[0, ["p", "d", "q"]])
    sarima_fit = fit_sarima(y_window, order, SARIMA_SEASONAL)
    forecasts["SARIMA"] = rolling_forecast(sarima_fit, series, test_days)["forecast"]
    print(f"[sarima] SARIMA{order}{SARIMA_SEASONAL} AIC={sarima_fit.aic:.1f}")

    # 4. SARIMAX: A = weekend flag (true forecast); B = measured variables (conditional)
    exog = build_exog(df, y_window.index)
    X_window = exog.loc[y_window.index]
    fit_a = fit_sarima(y_window, order, SARIMA_SEASONAL, exog=X_window[["is_weekend"]])
    forecasts["SARIMAX A (weekend flag)"] = rolling_forecast(
        fit_a, series, test_days, exog, ["is_weekend"])["forecast"]
    fit_b = fit_sarima(y_window, order, SARIMA_SEASONAL, exog=X_window[MEASURED])
    forecasts["SARIMAX B (measured, conditional)"] = rolling_forecast(
        fit_b, series, test_days, exog, MEASURED)["forecast"]
    print("[sarimax] done")

    # 5. XGBoost: add feature groups one at a time
    X_all = make_features(df)
    X_train = X_all.loc[train.index].dropna()
    y_train = series.loc[X_train.index]
    X_test = X_all.loc[test.index]
    used = []
    for name, cols in FEATURE_GROUPS.items():
        used += cols
        label = "XGB: " + " + ".join(list(FEATURE_GROUPS)[:list(FEATURE_GROUPS).index(name) + 1])
        forecasts[label] = train_xgb(X_train[used], y_train).predict(X_test[used])
    print("[xgboost] done")

    # 6. Chronos-Bolt (zero-shot foundation model)
    if not skip_chronos:
        forecasts["Chronos-Bolt (zero-shot)"] = chronos_forecast(series, test_days)["forecast"]
        print("[chronos] done")

    # 7. Evaluation and outputs
    metrics = evaluate_models(forecasts, train_series=train[TARGET])
    metrics.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")
    plot_forecasts(forecasts, metrics)
    print("\n" + metrics.to_string(index=False))
    print("\nSaved outputs/forecasts/all_forecasts.csv, outputs/metrics/model_comparison.csv "
          "and outputs/figures/forecast_comparison.png")


def plot_forecasts(forecasts, metrics):
    """One panel per key model across the test period, saved as a PNG."""
    models = [m for m in ["Seasonal naive (weekly)", "SARIMA", "SARIMAX A (weekend flag)",
                          "XGB: calendar + lags + rolling", "Chronos-Bolt (zero-shot)",
                          "SARIMAX B (measured, conditional)"] if m in forecasts]
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 2.7 * len(models)),
                             sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        score = metrics.loc[metrics.Model == model, "RMSE"].item()
        ax.plot(forecasts.index, forecasts["Actual"], color="black", linewidth=0.9, label="Actual")
        ax.plot(forecasts.index, forecasts[model], color="tab:orange", linewidth=0.9,
                label="Forecast")
        ax.set_title(f"{model}  (RMSE = {score:.3f} kW)", loc="left")
        ax.set_ylabel("Power (kW)")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Date")
    fig.suptitle("24-hour-ahead forecasts vs actual demand (test period)", y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "forecast_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Household electricity forecasting pipeline")
    parser.add_argument("--run-grid-search", action="store_true",
                        help="refit all SARIMA grid-search models instead of loading results")
    parser.add_argument("--skip-chronos", action="store_true",
                        help="skip the Chronos foundation model")
    args = parser.parse_args()
    main(run_grid_search=args.run_grid_search, skip_chronos=args.skip_chronos)
