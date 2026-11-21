# Household Electricity Demand Forecasting

Forecasting the next 24 hours of household electricity demand (`Global_active_power`)
using the UCI Individual Household Electric Power Consumption dataset.

Models: benchmark forecasts (mean, naive, seasonal naive, drift), SARIMAX,
a feature-based machine-learning model, and a time-series foundation model (Chronos).

## Repository structure
```
data/        raw data (downloaded by script, not stored on GitHub)
notebooks/   step-by-step analysis notebooks
results/     metrics tables and figures
src/         reusable Python code
```

## Quick start
```bash
pip install -r requirements.txt
python src/download_data.py
```

*(This README will be completed following the module's README example.)*
