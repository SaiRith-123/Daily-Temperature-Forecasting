import argparse
import json
import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import ADFullerResult, adfuller

from src.data.load_data import DEFAULT_DATA_PATH, load_temperature_series
from src.evaluation.evaluate import calculate_forecast_metrics
from src.evaluation.mlflow_tracking import log_arima_run
from src.models.arima import ArimaOrder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "delhi_arima.pkl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "models" / "evaluation.json"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT_NAME = "Delhi Temperature Forecasting"
DEFAULT_REGISTERED_MODEL_NAME = "DelhiTemperatureARIMA"


def _log_candidate_run(
    *,
    order: ArimaOrder,
    aic: float,
    bic: float,
    metrics: dict[str, float],
    train_end: str,
    validation_start: str,
    validation_end: str,
    convergence_warning: bool,
) -> None:
    with mlflow.start_run(run_name=f"pipeline-validation-ARIMA{order}"):
        mlflow.log_params(
            {
                "model": "ARIMA",
                "p": order[0],
                "d": order[1],
                "q": order[2],
            }
        )
        mlflow.log_metrics(
            {
                "aic": aic,
                "bic": bic,
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_mape_percent": metrics["mape_percent"],
            }
        )
        mlflow.set_tags(
            {
                "pipeline_stage": "validation",
                "model_order": str(order),
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "convergence_warning": str(convergence_warning).lower(),
            }
        )


def _save_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def run_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    test_size: int = 30,
    validation_fraction: float = 0.2,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    candidate_orders: Sequence[ArimaOrder] | None = None,
) -> dict[str, Any]:
    """Validate data, select and evaluate ARIMA, then persist its artifacts."""
    data_file = Path(data_path)
    artifact_file = Path(model_path)
    report_file = Path(report_path)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if test_size < 1:
        raise ValueError("test_size must be at least 1.")
    if artifact_file.resolve() == report_file.resolve():
        raise ValueError("Model artifact and report must use different paths.")

    # Data loading parses the target and date column, sorts observations, and
    # rejects invalid dates, missing targets, and duplicate dates.
    series = load_temperature_series(data_file)
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite meantemp values.")
    if not series.between(-50, 60).all():
        raise ValueError("Dataset contains meantemp values outside the -50..60 C screening range.")

    expected_dates = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="D",
        name="date",
    )
    missing_dates = expected_dates.difference(pd.DatetimeIndex(series.index))
    if not missing_dates.empty:
        raise ValueError(
            "Dataset is missing calendar dates: "
            + ", ".join(date.date().isoformat() for date in missing_dates)
        )

    # Preprocess to an explicit daily frequency. Keep valid temperature
    # observations, including statistical outliers, rather than imputing them.
    temperatures = series.asfreq("D")
    if len(temperatures) <= test_size + 2:
        raise ValueError("Not enough daily observations for the requested test split.")

    test = temperatures.iloc[-test_size:]
    development = temperatures.iloc[:-test_size]
    validation_size = max(1, math.ceil(len(development) * validation_fraction))
    if validation_size >= len(development):
        raise ValueError("Not enough development data for a chronological validation split.")
    selection_train = development.iloc[:-validation_size]
    validation = development.iloc[-validation_size:]

    if not np.isfinite(validation.to_numpy(dtype=float)).all():
        raise ValueError("Validation target contains non-finite values.")
    if not np.isfinite(test.to_numpy(dtype=float)).all():
        raise ValueError("Test target contains non-finite values.")

    adf_result = cast(
        ADFullerResult,
        adfuller(selection_train, result_object=True),
    )
    differencing_order = int(adf_result.pvalue >= 0.05)
    if candidate_orders is None:
        orders = [
            (p, differencing_order, q)
            for p in range(4)
            for q in range(4)
        ]
        benchmark_order = (5, differencing_order, 0)
        if benchmark_order not in orders:
            orders.append(benchmark_order)
    else:
        orders = list(candidate_orders)
        if not orders:
            raise ValueError("At least one candidate ARIMA order is required.")
        if any(
            len(order) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in order)
            for order in orders
        ):
            raise ValueError("Candidate ARIMA orders must contain non-negative integer (p, d, q) values.")
        if any(order[1] != differencing_order for order in orders):
            raise ValueError(
                f"Candidate orders must use d={differencing_order}, as selected by the ADF test."
            )
        benchmark_order = (5, differencing_order, 0)
        if benchmark_order not in orders:
            orders.append(benchmark_order)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)

    candidate_results: list[dict[str, Any]] = []
    for order in orders:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted = ARIMA(selection_train, order=order).fit()
            predicted = np.asarray(fitted.forecast(steps=len(validation)), dtype=float)
        validation_metrics = calculate_forecast_metrics(validation, predicted)
        convergence_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in fit_warnings
        )
        _log_candidate_run(
            order=order,
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            metrics=validation_metrics,
            train_end=selection_train.index.max().date().isoformat(),
            validation_start=validation.index.min().date().isoformat(),
            validation_end=validation.index.max().date().isoformat(),
            convergence_warning=convergence_warning,
        )
        candidate_results.append(
            {
                "order": order,
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_mape_percent": validation_metrics["mape_percent"],
                "convergence_warning": convergence_warning,
            }
        )

    candidate_results.sort(key=lambda result: result["validation_rmse"])
    selected = candidate_results[0]
    selected_order = selected["order"]

    train = temperatures.iloc[:-test_size].astype(float)
    test = temperatures.iloc[-test_size:].astype(float)
    with warnings.catch_warnings(record=True) as final_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model = ARIMA(train, order=selected_order).fit()
        forecast = np.asarray(final_model.forecast(steps=len(test)), dtype=float)
    test_metrics = calculate_forecast_metrics(test, forecast)

    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact_file.with_name(f".{artifact_file.name}.tmp")
    final_model.save(str(temporary_artifact))
    os.replace(temporary_artifact, artifact_file)

    model_version = log_arima_run(
        run_name=f"pipeline-final-ARIMA{selected_order}",
        model=final_model,
        order=selected_order,
        metrics={"aic": float(final_model.aic), "bic": float(final_model.bic), **test_metrics},
        registered_model_name=registered_model_name,
        tags={
            "pipeline_stage": "final_evaluation",
            "selected_by": "validation_rmse",
            "experiment_id": experiment.experiment_id,
            "train_start": train.index.min().date().isoformat(),
            "train_end": train.index.max().date().isoformat(),
            "test_start": test.index.min().date().isoformat(),
            "test_end": test.index.max().date().isoformat(),
        },
    )

    report: dict[str, Any] = {
        "data": {
            "path": str(data_file.resolve()),
            "observations": len(temperatures),
            "start_date": temperatures.index.min().date().isoformat(),
            "end_date": temperatures.index.max().date().isoformat(),
            "duplicate_dates": 0,
            "missing_calendar_dates": 0,
        },
        "preprocessing": {
            "target": "meantemp",
            "frequency": "D",
            "operations": [
                "Parse dates and numeric mean temperatures",
                "Sort by date and enforce daily frequency",
                "Retain valid temperature observations without outlier removal or imputation",
            ],
        },
        "stationarity": {
            "adf_statistic": float(adf_result.statistic),
            "p_value": float(adf_result.pvalue),
            "differencing_order": differencing_order,
            "evaluated_through": selection_train.index.max().date().isoformat(),
        },
        "split": {
            "train_start": train.index.min().date().isoformat(),
            "train_end": train.index.max().date().isoformat(),
            "train_observations": len(train),
            "test_start": test.index.min().date().isoformat(),
            "test_end": test.index.max().date().isoformat(),
            "test_observations": len(test),
            "selection_train_end": selection_train.index.max().date().isoformat(),
            "validation_start": validation.index.min().date().isoformat(),
            "validation_end": validation.index.max().date().isoformat(),
            "validation_observations": len(validation),
            "random_shuffle": False,
        },
        "model": {
            "family": "ARIMA",
            "order": list(selected_order),
            "aic": float(final_model.aic),
            "bic": float(final_model.bic),
            "validation_metrics": {
                "mae": selected["validation_mae"],
                "rmse": selected["validation_rmse"],
                "mape_percent": selected["validation_mape_percent"],
            },
            "test_metrics": test_metrics,
            "convergence_warning": any(
                issubclass(item.category, ConvergenceWarning) for item in final_warnings
            ),
            "candidate_results": [
                {**result, "order": list(result["order"])}
                for result in candidate_results
            ],
        },
        "artifacts": {
            "model_path": str(artifact_file.resolve()),
            "report_path": str(report_file.resolve()),
            "registered_model_name": registered_model_name,
            "model_version": model_version,
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_name": experiment_name,
            "mlflow_experiment_id": experiment.experiment_id,
        },
    }
    _save_json_atomic(report_file, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Delhi temperature validation, preprocessing, ARIMA training, and evaluation pipeline."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    args = parser.parse_args()

    report = run_pipeline(
        data_path=args.data,
        model_path=args.model,
        report_path=args.report,
        test_size=args.test_size,
        validation_fraction=args.validation_fraction,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
