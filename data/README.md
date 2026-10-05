# Dataset

Keep `daily_climate.csv` in this directory for the notebook, API, and default
data-loading function. The CSV must contain:

- `date`: one parseable date per observation
- `meantemp`: numeric daily mean temperature
- `humidity`, `wind_speed`, `meanpressure`: optional numeric daily measurements

The loader sorts observations by date and rejects missing values, invalid values,
and duplicate dates. When observations form a regular sequence, the loader also
preserves that frequency for the ARIMA model. The exploration notebook applies
additional full-dataset checks; its cleaning actions are in-memory and do not
overwrite this source file.
