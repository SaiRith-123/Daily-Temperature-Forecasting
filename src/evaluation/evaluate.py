import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.models.arima import DEFAULT_ARIMA_ORDER, ArimaOrder


def calculate_forecast_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE (as a percentage) for aligned values."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")
    if actual_values.size == 0 or actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same non-zero length.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Actual and predicted values must contain only finite numbers.")
    if np.any(actual_values == 0):
        raise ValueError("MAPE is undefined when actual values contain zero.")

    errors = actual_values - predicted_values
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape_percent": float(np.mean(np.abs(errors / actual_values)) * 100),
    }


def evaluate_arima(
    series: pd.Series,
    test_size: int = 30,
    order: ArimaOrder = DEFAULT_ARIMA_ORDER,
) -> dict[str, float]:
    """Evaluate ARIMA on the final observations using a chronological split."""
    if test_size < 1 or test_size >= len(series):
        raise ValueError("test_size must be between 1 and the number of observations - 1.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Temperature series must use a DatetimeIndex.")
    if series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Temperature series must contain only finite values.")

    train = series.iloc[:-test_size].astype(float)
    actual = series.iloc[-test_size:].astype(float)
    fitted_model = ARIMA(train, order=order).fit()
    predicted = pd.Series(
        np.asarray(fitted_model.forecast(steps=test_size), dtype=float),
        index=actual.index,
    )
    return calculate_forecast_metrics(actual, predicted)
