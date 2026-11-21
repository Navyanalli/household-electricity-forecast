"""Download and unzip the UCI Individual Household Electric Power Consumption dataset.

Usage (from the repository root):
    python src/download_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/235/"
       "individual+household+electric+power+consumption.zip")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TXT_FILE = DATA_DIR / "household_power_consumption.txt"


def download_data() -> Path:
    """Download the dataset into data/ if it is not already there."""
    DATA_DIR.mkdir(exist_ok=True)
    if TXT_FILE.exists():
        print(f"Data already present: {TXT_FILE}")
        return TXT_FILE

    zip_path = DATA_DIR / "power.zip"
    print("Downloading dataset from UCI ...")
    urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    print(f"Saved to {TXT_FILE}")
    return TXT_FILE


if __name__ == "__main__":
    download_data()
