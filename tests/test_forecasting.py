import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import calculate_forecast_metrics, evaluate_arima
from src.models.arima import DEFAULT_ARIMA_ORDER, forecast_temperatures


@pytest.fixture
def temperature_series():
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    values = 15 + np.random.default_rng(42).normal(0, 1, size=80)
    return pd.Series(values, index=dates, name="meantemp")


def test_forecast_returns_requested_daily_horizon(temperature_series):
    assert DEFAULT_ARIMA_ORDER == (1, 1, 0)
    result = forecast_temperatures(temperature_series, steps=5)

    assert len(result) == 5
    assert result.index[0] == temperature_series.index[-1] + pd.Timedelta(days=1)
    assert result.index.freqstr == "D"
    assert np.isfinite(result.to_numpy()).all()


def test_forecast_rejects_zero_horizon(temperature_series):
    with pytest.raises(ValueError, match="at least 1"):
        forecast_temperatures(temperature_series, steps=0)


def test_evaluate_arima_returns_finite_metrics(temperature_series):
    result = evaluate_arima(temperature_series, test_size=10)

    assert set(result) == {"mae", "rmse", "mape_percent"}
    assert all(np.isfinite(value) for value in result.values())
    assert result["mape_percent"] >= 0


def test_forecast_metrics_reject_actual_zero_for_mape():
    with pytest.raises(ValueError, match="MAPE is undefined"):
        calculate_forecast_metrics(np.array([0.0, 2.0]), np.array([1.0, 2.5]))


def test_forecast_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same non-zero length"):
        calculate_forecast_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_evaluate_rejects_invalid_test_size(temperature_series):
    with pytest.raises(ValueError, match="test_size"):
        evaluate_arima(temperature_series, test_size=len(temperature_series))
