"""Forecasting models: benchmarks, SARIMA/SARIMAX, XGBoost and Chronos.

Every model produces one 24-hour forecast from midnight of each test day,
using only data available before that day (rolling origin).
"""
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.config import (CHRONOS_CONTEXT, CHRONOS_MODEL, HORIZON, MEASURED, RANDOM_STATE,
                        TARGET)


# ---------------------------------------------------------------- benchmarks
def benchmark_forecasts(history, horizon=HORIZON):
    """Mean, naive, daily/weekly seasonal naive and drift forecasts from a history series."""
    y = history.values
    index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return pd.DataFrame({
        "Mean": np.full(horizon, y.mean()),
        "Naive": np.full(horizon, y[-1]),
        "Seasonal naive (daily)": y[-24:][:horizon],
        "Seasonal naive (weekly)": y[-168:-168 + horizon],
        "Drift": y[-1] + slope * np.arange(1, horizon + 1),
    }, index=index)


def rolling_benchmarks(series, test_days):
    """Benchmark forecasts for every test day (history ends the hour before each day)."""
    return pd.concat([benchmark_forecasts(series.loc[:day - pd.Timedelta(hours=1)])
                      for day in test_days])


# ---------------------------------------------------------------- SARIMA / SARIMAX
def fit_sarima(y, order, seasonal_order, exog=None):
    """Fit a SARIMA/SARIMAX model (warnings hidden)."""
    model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=50)


def sarima_grid_search(y, seasonal_order, save_path, p_values, d_values, q_values):
    """AIC grid search over (p, d, q); progress is saved after each model so it can resume."""
    rows = pd.read_csv(save_path).to_dict("records") if os.path.exists(save_path) else []
    done = {(r["p"], r["d"], r["q"]) for r in rows}
    for p, d, q in itertools.product(p_values, d_values, q_values):
        if (p, d, q) in done:
            continue
        start = time.time()
        try:
            aic = fit_sarima(y, (p, d, q), seasonal_order).aic
        except Exception:
            aic = np.nan
        rows.append({"p": p, "d": d, "q": q, "aic": aic, "seconds": round(time.time() - start, 1)})
        pd.DataFrame(rows).to_csv(save_path, index=False)
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def rolling_forecast(fit, series, test_days, exog_all=None, exog_cols=None, alpha=0.05):
    """24-hour forecasts per test day; each day's actuals are appended without refitting.

    Exog values for the forecast day are taken from the data, which is only a true
    forecast if those variables are known in advance (e.g. a calendar flag).
    """
    days, current = [], fit
    for day in test_days:
        idx = pd.date_range(day, periods=HORIZON, freq="h")
        x_day = exog_all.loc[idx, exog_cols] if exog_cols else None
        fc = current.get_forecast(steps=HORIZON, exog=x_day)
        ci = fc.conf_int(alpha=alpha)
        days.append(pd.DataFrame({"forecast": fc.predicted_mean.values,
                                  "lower": ci.iloc[:, 0].values,
                                  "upper": ci.iloc[:, 1].values}, index=idx))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current = current.append(series.loc[idx].asfreq("h"), exog=x_day, refit=False)
    return pd.concat(days)


def build_exog(df, window_index):
    """Weekend flag plus measured variables standardised with the training-window statistics."""
    exog = pd.DataFrame(index=df.index)
    exog["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    for col in MEASURED:
        w = df.loc[window_index, col]
        exog[col] = (df[col] - w.mean()) / w.std()
    return exog


# ---------------------------------------------------------------- XGBoost
FEATURE_GROUPS = {
    "calendar": ["hour", "dayofweek", "is_weekend", "month"],
    "lags": ["lag_24", "lag_48", "lag_72", "lag_168"],
    "rolling": ["roll_mean_24", "roll_std_24", "roll_mean_168"],
    "measured (lag 24)": [f"{c}_lag24" for c in MEASURED],
}


def make_features(df):
    """Features known at the forecast origin: calendar, lags >= 24 h, lagged rolling stats
    and measured variables lagged 24 h."""
    y = df[TARGET]
    X = pd.DataFrame(index=df.index)
    X["hour"] = df.index.hour
    X["dayofweek"] = df.index.dayofweek
    X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
    X["month"] = df.index.month
    for lag in [24, 48, 72, 168]:
        X[f"lag_{lag}"] = y.shift(lag)
    past = y.shift(24)
    X["roll_mean_24"] = past.rolling(24).mean()
    X["roll_std_24"] = past.rolling(24).std()
    X["roll_mean_168"] = past.rolling(168).mean()
    for col in MEASURED:
        X[f"{col}_lag24"] = df[col].shift(24)
    return X


def train_xgb(X_tr, y_tr, val_days=14, seed=RANDOM_STATE):
    """Choose the number of trees by early stopping on the last `val_days` of training data,
    then refit on all training data (the test period is never used for tuning)."""
    split = X_tr.index.max() - pd.Timedelta(days=val_days)
    after = split + pd.Timedelta(hours=1)
    params = dict(learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, random_state=seed)
    finder = XGBRegressor(n_estimators=2000, early_stopping_rounds=50, **params)
    finder.fit(X_tr.loc[:split], y_tr.loc[:split],
               eval_set=[(X_tr.loc[after:], y_tr.loc[after:])], verbose=False)
    model = XGBRegressor(n_estimators=finder.best_iteration + 1, **params)
    model.fit(X_tr, y_tr)
    return model


# ---------------------------------------------------------------- Chronos
def chronos_forecast(series, test_days, model_name=CHRONOS_MODEL, context=CHRONOS_CONTEXT):
    """Zero-shot Chronos-Bolt forecasts: median plus 10% and 90% quantiles (80% interval)."""
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map="cpu",
                                                   torch_dtype=torch.float32)
    contexts = [torch.tensor(series.loc[:day - pd.Timedelta(hours=1)].values[-context:],
                             dtype=torch.float32) for day in test_days]
    quantiles, _ = pipeline.predict_quantiles(contexts, prediction_length=HORIZON,
                                              quantile_levels=[0.1, 0.5, 0.9])
    q = np.asarray(quantiles).reshape(-1, 3)
    index = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(day, periods=HORIZON, freq="h") for day in test_days]))
    return pd.DataFrame({"forecast": q[:, 1], "lower": q[:, 0], "upper": q[:, 2]}, index=index)
