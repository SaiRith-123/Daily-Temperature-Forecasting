import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMAResults

from src.pipeline import run_pipeline


def test_run_pipeline_creates_model_report_and_mlflow_version(tmp_path):
    data_path = tmp_path / "daily_climate.csv"
    model_path = tmp_path / "artifacts" / "temperature.pkl"
    report_path = tmp_path / "artifacts" / "evaluation.json"
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    values = 18 + np.random.default_rng(42).normal(0, 1, size=len(dates))
    pd.DataFrame({"date": dates, "meantemp": values}).to_csv(data_path, index=False)

    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    try:
        report = run_pipeline(
            data_path=data_path,
            model_path=model_path,
            report_path=report_path,
            test_size=10,
            validation_fraction=0.2,
            tracking_uri=tracking_uri,
            experiment_name="Pipeline integration test",
            registered_model_name="PipelineIntegrationTestARIMA",
            candidate_orders=[(0, 0, 0), (1, 0, 0)],
        )

        assert model_path.is_file()
        assert report_path.is_file()
        assert json.loads(report_path.read_text(encoding="utf-8")) == report
        assert report["data"]["observations"] == 60
        assert report["split"]["train_end"] < report["split"]["test_start"]
        assert report["split"]["train_observations"] == 50
        assert report["split"]["test_observations"] == 10
        assert report["split"]["random_shuffle"] is False
        assert (
            report["stationarity"]["evaluated_through"]
            == report["split"]["selection_train_end"]
        )
        assert report["stationarity"]["evaluated_through"] < report["split"]["test_start"]
        assert report["artifacts"]["model_version"] == 1
        loaded_model = ARIMAResults.load(model_path)
        loaded_forecast = np.asarray(loaded_model.forecast(steps=2), dtype=float)
        assert loaded_forecast.shape == (2,)
        assert np.isfinite(loaded_forecast).all()
        assert set(report["model"]["test_metrics"]) == {
            "mae",
            "rmse",
            "mape_percent",
        }

        client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment = client.get_experiment_by_name("Pipeline integration test")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == len(report["model"]["candidate_results"]) + 1
        assert any(run.data.tags.get("pipeline_stage") == "final_evaluation" for run in runs)
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


def test_run_pipeline_rejects_missing_calendar_date(tmp_path):
    data_path = tmp_path / "daily_climate.csv"
    dates = pd.date_range("2020-01-01", periods=40, freq="D").delete(10)
    pd.DataFrame(
        {"date": dates, "meantemp": np.linspace(10, 20, len(dates))}
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="missing calendar dates"):
        run_pipeline(
            data_path=data_path,
            model_path=tmp_path / "model.pkl",
            report_path=tmp_path / "report.json",
            test_size=5,
            tracking_uri=f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}",
            candidate_orders=[(0, 0, 0)],
        )
