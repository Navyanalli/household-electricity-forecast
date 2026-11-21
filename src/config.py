"""Shared settings for the household electricity forecasting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"          # saved SARIMA grid search (input)
OUTPUT_DIR = ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
RAW_FILE = DATA_DIR / "household_power_consumption.txt"

TARGET = "Global_active_power"            # hourly mean demand (kW)
MEASURED = ["Global_reactive_power", "Voltage", "Global_intensity"]

HORIZON = 24          # forecast 24 hours ahead
TEST_DAYS = 14        # last 14 days are the test period
SEASON = 24           # daily seasonal period (hours)

SARIMA_WEEKS = 4                      # SARIMA is fitted on the last 4 weeks of training data
SARIMA_SEASONAL = (0, 1, 1, 24)       # chosen from ACF/PACF and AIC comparison
GRID_P, GRID_D, GRID_Q = range(0, 7), range(0, 3), range(0, 7)

CHRONOS_MODEL = "amazon/chronos-bolt-small"
CHRONOS_CONTEXT = 2048                # hours of history given to Chronos
RANDOM_STATE = 42
