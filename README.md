# Household Power Consumption Forecasting

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting household electricity consumption over the next 24 hours.

The project uses the **UCI Individual Household Electric Power Consumption** dataset: one-minute measurements of electric power consumption from a single household in Sceaux, France, from December 2006 to November 2010.

Simple benchmark models, SARIMA/SARIMAX, a feature-based machine-learning model (XGBoost) and a time-series foundation model (Chronos-Bolt) are compared on the same test period.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Navyanalli/household-electricity-forecast/blob/main/household_electricity_forecast.ipynb)

## Project aim

To forecast short-term household electricity demand and evaluate whether more complex models improve on simple benchmark methods. The analysis addresses:

1. How well do simple benchmark models forecast household electricity demand?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do additional measured variables and time-based features improve forecast accuracy?
4. Does a feature-based machine-learning model improve performance?
5. Does a time-series foundation model (Chronos) provide additional benefit?
6. Which model would be most suitable for practical household energy forecasting?

## Dataset

The data contain one-minute measurements. All calendar timestamps are present, but 25,979 minutes (1.25%) have every measurement missing (meter outages, the longest about 5 days).

Original variables:

```text
Date, Time, Global_active_power, Global_reactive_power, Voltage,
Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
```

**Target variable:** `Global_active_power` (household global active power, kW).

### Preprocessing

1. `Date` and `Time` are combined into a datetime index.
2. Minute data are averaged to **hourly** values (short gaps disappear in the hourly mean).
3. Hours left empty by long outages (421 hours, 1.22%) are filled with the value from the **same hour one week earlier**, preserving daily and weekly patterns; time interpolation is a final fallback.
4. The partial first and last days are removed, giving **1,440 complete days (34,560 hours)** from 2006-12-17 to 2010-11-25.

The sub-metering variables are not used as covariates because they are components of the target.

## Forecasting task

```text
Target:   Global_active_power (hourly mean, kW)
Horizon:  24 hours
Test set: final 14 days (2010-11-12 to 2010-11-25, 336 hours)
Train:    all earlier data (2006-12-17 to 2010-11-11, 34,224 hours)
```

Every model makes **14 separate 24-hour forecasts**, one from midnight of each test day, using only data available before that day (rolling origin). The 14-day test period follows the assignment brief (Part 6).

## Models

| Model | Details |
|---|---|
| Benchmarks | Mean, naive, daily seasonal naive (lag 24), weekly seasonal naive (lag 168), drift |
| SARIMA | SARIMA(6,0,6)(0,1,1)<sub>24</sub>. Non-seasonal orders chosen by AIC over p = 0–6, d = 0–2, q = 0–6 (147 models); seasonal order chosen from the ACF/PACF of the seasonally differenced series and confirmed by AIC. Fitted on the last 4 weeks of training data to keep the grid search feasible. |
| SARIMAX A | SARIMA + weekend flag (known in advance → **true forecast**) |
| SARIMAX B | SARIMA + Global_reactive_power, Voltage, Global_intensity using their **realised test-period values** → **conditional forecast**, reported for comparison only |
| XGBoost | Feature groups added one at a time: calendar → lags → rolling windows → measured variables (lag 24). Number of trees chosen by early stopping on the last 14 days of training data. |
| Chronos-Bolt (small) | Zero-shot, target-only (no covariates, no fine-tuning). Context: previous 2,048 hours. Median used as point forecast; 10%/90% quantiles give an 80% interval. |

## Features (XGBoost)

All features are **known at the forecast origin** (midnight before the forecast day):

```text
Calendar:  hour, dayofweek, is_weekend, month
Lags:      lag_24, lag_48, lag_72, lag_168
Rolling:   roll_mean_24, roll_std_24, roll_mean_168  (windows end 24 h before the target hour)
Measured:  Global_reactive_power_lag24, Voltage_lag24, Global_intensity_lag24
```

Lags shorter than 24 hours (e.g. `lag_1`) are deliberately **not** used: for hours later in the forecast day, the previous hour would not yet have been observed at the forecast origin.

## Repository structure

```text
household-electricity-forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── household_electricity_forecast.ipynb   # main notebook: full analysis with explanations
├── data/                                  # raw data (downloaded automatically, not committed)
├── src/
│   ├── config.py           # settings (target, horizon, test period, model orders)
│   ├── download_data.py    # downloads and unzips the UCI dataset
│   ├── data_prep.py        # loading, cleaning, hourly resampling
│   ├── evaluation.py       # train/test split, metrics, stationarity tests
│   └── models.py           # benchmarks, SARIMA/SARIMAX, XGBoost, Chronos
├── scripts/
│   └── run_pipeline.py     # runs the full pipeline from the command line
├── tests/
│   └── test_pipeline.py    # pytest tests
├── results/                # outputs saved by the notebook (incl. SARIMA grid search)
└── outputs/                # outputs saved by the pipeline script
    ├── forecasts/all_forecasts.csv
    ├── metrics/model_comparison.csv
    └── figures/forecast_comparison.png
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the analysis

**Notebook (recommended for marking):** open `household_electricity_forecast.ipynb` in Google Colab using the badge above and select **Runtime → Run all**. The notebook clones this repository and downloads the data itself.

**Pipeline script** (from the repository root):

```bash
python scripts/run_pipeline.py                     # full pipeline (~2-3 minutes)
python scripts/run_pipeline.py --skip-chronos      # without the foundation model
python scripts/run_pipeline.py --run-grid-search   # also refit all 147 SARIMA models (~40 minutes)
```

The SARIMA grid search takes about 40 minutes, so its results are saved in `results/sarima_grid_search.csv` and loaded by default. Use `--run-grid-search` (or set `RUN_GRID_SEARCH = True` in the notebook) to recompute them.

The pipeline: downloads the data → parses timestamps → handles missing values → resamples to hourly → splits train/test → fits benchmarks, SARIMA, SARIMAX, XGBoost and Chronos → evaluates all forecasts → saves forecasts, metrics and a comparison figure to `outputs/`.

## Evaluation metrics

All models are evaluated on the same 336 test hours using **RMSE, MAE, MASE, Bias** (and MAPE). MASE is scaled by the in-sample daily seasonal naive error. Every model is compared against the strongest benchmark (weekly seasonal naive).

| Model | RMSE | MAE | MASE | Bias | True forecast? |
|---|---|---|---|---|---|
| SARIMAX B (measured, conditional) | 0.028 | 0.016 | 0.026 | 0.001 | No |
| SARIMA | 0.632 | 0.445 | 0.697 | −0.012 | Yes |
| XGB: calendar + lags + rolling | 0.635 | 0.460 | 0.719 | 0.071 | Yes |
| SARIMAX A (weekend flag) | 0.638 | 0.459 | 0.718 | −0.013 | Yes |
| Chronos-Bolt (zero-shot) | 0.638 | 0.422 | 0.660 | −0.059 | Yes |
| Seasonal naive (weekly) | 0.832 | 0.573 | 0.897 | 0.063 | Yes |

Full results: `outputs/metrics/model_comparison.csv` and `results/final_metrics_all_models.csv`.

## Data leakage

- Lag and rolling features use only values from 24 or more hours before the target hour.
- Measured variables are standardised using training-window statistics only.
- XGBoost hyperparameters (number of trees) are chosen on a validation period inside the training data, never on the test set.
- Future time-of-day and day-of-week variables are known in advance. Future measured electrical variables are not; SARIMAX B uses their realised test values and is therefore a **conditional forecast**, not a true forecast.

## Tests

```bash
pytest
```

Tests check that metrics are zero for a perfect forecast, benchmark forecasts have the correct length, lag features do not use future target values, and the processed hourly data has no missing target values and contains only complete days.

## Reproducibility

- Random seed fixed (`random_state=42`) for XGBoost.
- Raw data are not committed; they are downloaded by `src/download_data.py`.
- The pipeline runs from a fresh clone using the instructions above.
