import pandas as pd
import pytest

from src.data.load_data import load_temperature_series


def test_load_temperature_series_sorts_dates(tmp_path):
    csv_path = tmp_path / "daily_climate.csv"
    pd.DataFrame(
        {"date": ["2020-01-02", "2020-01-01"], "meantemp": [11.0, 10.0]}
    ).to_csv(csv_path, index=False)

    result = load_temperature_series(csv_path)

    assert result.index.tolist() == list(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert result.tolist() == [10.0, 11.0]


def test_load_temperature_series_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "daily_climate.csv"
    pd.DataFrame({"date": ["2020-01-01"], "humidity": [50]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="meantemp"):
        load_temperature_series(csv_path)
