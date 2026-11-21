"""Train/test split, accuracy metrics and stationarity tests."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import SEASON, TEST_DAYS


def train_test_split(df, test_days=TEST_DAYS):
    """Hold out the last `test_days` complete days as the test set."""
    split_point = df.index.max() - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    return df.loc[:split_point - pd.Timedelta(hours=1)], df.loc[split_point:], split_point


def rmse(actual, forecast):
    """Root mean squared error - penalises large errors more."""
    return np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2))


def mae(actual, forecast):
    """Mean absolute error in kW."""
    return np.mean(np.abs(np.asarray(actual) - np.asarray(forecast)))


def mape(actual, forecast):
    """Mean absolute percentage error (%)."""
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return np.mean(np.abs((actual - forecast) / actual)) * 100


def mase(actual, forecast, train_series, season=SEASON):
    """MAE scaled by the in-sample daily seasonal naive MAE (< 1 beats seasonal naive)."""
    y = np.asarray(train_series)
    return mae(actual, forecast) / np.mean(np.abs(y[season:] - y[:-season]))


def bias(actual, forecast):
    """Mean error (forecast - actual): positive = over-prediction on average."""
    return np.mean(np.asarray(forecast) - np.asarray(actual))


def evaluate_models(forecast_df, train_series, actual_col="Actual"):
    """RMSE, MAE, MAPE, MASE and Bias for every forecast column, sorted by RMSE."""
    actual = forecast_df[actual_col]
    rows = [{"Model": m,
             "RMSE": rmse(actual, forecast_df[m]),
             "MAE": mae(actual, forecast_df[m]),
             "MAPE (%)": mape(actual, forecast_df[m]),
             "MASE": mase(actual, forecast_df[m], train_series),
             "Bias": bias(actual, forecast_df[m])}
            for m in forecast_df.columns.drop(actual_col)]
    return pd.DataFrame(rows).sort_values("RMSE").round(3).reset_index(drop=True)


def stationarity_tests(series, name="series"):
    """ADF (H0: unit root) and KPSS (H0: stationary) tests in one table."""
    series = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p = adfuller(series, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[:2]
    return pd.DataFrame({
        "series": [name, name],
        "test": ["ADF", "KPSS"],
        "statistic": [round(adf_stat, 3), round(kpss_stat, 3)],
        "p_value": [round(adf_p, 4), round(kpss_p, 4)],
        "conclusion": ["stationary" if adf_p < 0.05 else "non-stationary",
                       "non-stationary" if kpss_p < 0.05 else "stationary"],
    })
