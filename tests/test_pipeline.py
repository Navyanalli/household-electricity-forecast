"""Simple tests for key pipeline functions (run with: pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import HORIZON, TARGET
from src.data_prep import to_hourly
from src.evaluation import bias, mae, mase, rmse
from src.models import benchmark_forecasts, make_features


def make_hourly(days=30, seed=0):
    """Synthetic hourly data with a daily cycle, in the same format as the real data."""
    idx = pd.date_range("2010-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    y = 1 + 0.5 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, len(idx))
    return pd.DataFrame({TARGET: y, "Global_reactive_power": 0.1,
                         "Voltage": 240.0, "Global_intensity": y * 4}, index=idx)


def test_metrics_are_zero_for_perfect_forecast():
    y = make_hourly()[TARGET]
    assert rmse(y, y) == 0 and mae(y, y) == 0 and bias(y, y) == 0
    assert mase(y.iloc[-24:], y.iloc[-24:], y) == 0


def test_benchmark_forecast_length_matches_horizon():
    fc = benchmark_forecasts(make_hourly()[TARGET])
    assert len(fc) == HORIZON
    assert fc.notna().all().all()


def test_lag_features_do_not_use_future_values():
    """Changing the target in the last 24 hours must not change features for those hours."""
    df = make_hourly()
    original = make_features(df)
    changed = df.copy()
    changed.iloc[-24:, changed.columns.get_loc(TARGET)] = 999.0
    assert make_features(changed).iloc[-24:].equals(original.iloc[-24:])


def test_hourly_data_has_no_missing_target_and_full_days():
    idx = pd.date_range("2010-01-01 17:24", periods=60 * 24 * 10, freq="min")
    minute = pd.DataFrame({TARGET: 1.0, "Voltage": 240.0}, index=idx)
    minute.iloc[5000:5200] = np.nan                         # simulate an outage
    hourly = to_hourly(minute)
    assert hourly[TARGET].isna().sum() == 0
    assert hourly.index[0].hour == 0 and hourly.index[-1].hour == 23
    assert len(hourly) % 24 == 0
