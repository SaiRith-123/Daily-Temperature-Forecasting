from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "daily_climate.csv"


def load_temperature_series(path: str | Path | None = None) -> pd.Series:
    """Load and validate the daily mean-temperature series from a CSV file."""
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.is_file():
        raise FileNotFoundError(f"Temperature dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    required_columns = {"date", "meantemp"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    temperatures = pd.to_numeric(frame["meantemp"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Dataset contains invalid or missing dates.")
    if temperatures.isna().any():
        raise ValueError("Dataset contains invalid or missing meantemp values.")

    series = pd.Series(
        temperatures.to_numpy(),
        index=pd.DatetimeIndex(dates, name="date"),
        name="meantemp",
    ).sort_index()
    if series.index.has_duplicates:
        raise ValueError("Dataset contains duplicate dates.")

    frequency = pd.infer_freq(series.index) if len(series) > 2 else None
    if frequency is not None:
        series.index = pd.DatetimeIndex(series.index, name="date", freq=frequency)

    return series
