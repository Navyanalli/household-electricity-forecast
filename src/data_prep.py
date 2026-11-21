"""Loading and cleaning the UCI Individual Household Electric Power Consumption data."""
import pandas as pd

from src.config import RAW_FILE
from src.download_data import download_data


def load_raw(path=RAW_FILE):
    """Read the raw minute-level file ('?' marks missing readings) with a datetime index."""
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    return df.set_index("Datetime").drop(columns=["Date", "Time"])


def to_hourly(df_minute):
    """Average to hourly values, fill empty hours and trim to complete days.

    Short gaps (1-3 minutes) disappear in the hourly mean. Hours left completely
    empty by long outages are filled with the value from the same hour one week
    earlier (preserving daily and weekly patterns), with time interpolation as a
    final fallback. The partial first and last days are removed.
    """
    hourly = df_minute.resample("h").mean()
    filled = hourly.copy()
    for _ in range(3):
        filled = filled.fillna(filled.shift(168))
    filled = filled.interpolate(method="time")

    start = filled.index[0].normalize()
    if filled.index[0] != start:
        start += pd.Timedelta(days=1)
    end = filled.index[-1].normalize() - pd.Timedelta(hours=1)
    if filled.index[-1].hour == 23:
        end = filled.index[-1]
    return filled.loc[start:end]


def load_hourly_data():
    """Download the data if needed and return the clean hourly DataFrame."""
    download_data()
    return to_hourly(load_raw())
