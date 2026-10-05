from typing import TypeAlias

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ArimaOrder: TypeAlias = tuple[int, int, int]
DEFAULT_ARIMA_ORDER: ArimaOrder = (1, 1, 0)


def forecast_temperatures(
    series: pd.Series,
    steps: int = 7,
    order: ArimaOrder = DEFAULT_ARIMA_ORDER,
) -> pd.Series:
    """Fit an ARIMA model and forecast the requested number of days."""
    if series.empty:
        raise ValueError("Cannot forecast from an empty temperature series.")
    if steps < 1:
        raise ValueError("Forecast steps must be at least 1.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Temperature series must use a DatetimeIndex.")
    if series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Temperature series must contain only finite values.")

    fitted_model = ARIMA(series.astype(float), order=order).fit()
    values = np.asarray(fitted_model.forecast(steps=steps), dtype=float)
    forecast_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
        name=series.index.name or "date",
    )
    return pd.Series(values, index=forecast_dates, name="forecast_meantemp")
