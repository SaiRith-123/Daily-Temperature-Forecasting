from collections.abc import Mapping

import mlflow
from mlflow.tracking import MlflowClient

from src.models.arima import ArimaOrder


def log_arima_run(
    *,
    run_name: str,
    model: object,
    order: ArimaOrder,
    metrics: Mapping[str, float],
    registered_model_name: str,
    tags: Mapping[str, str] | None = None,
) -> int:
    """Log an ARIMA run and register its fitted model, returning its version."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "ARIMA",
                "p": order[0],
                "d": order[1],
                "q": order[2],
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tags({"model_order": str(order), **(dict(tags) if tags else {})})

        model_info = mlflow.statsmodels.log_model(model, name="model")
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registered_model_name,
            await_registration_for=300,
        )
        version = int(model_version.version)
        MlflowClient().set_model_version_tag(
            name=registered_model_name,
            version=str(version),
            key="model_version",
            value=str(version),
        )
        mlflow.set_tag("model_version", str(version))
        mlflow.set_tag("registered_model_name", registered_model_name)

    return version
