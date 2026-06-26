import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


def setup_mlflow(config: DictConfig) -> tuple[MlflowClient, str | None]:
    """Initializes the MLflow client and extracts the target run ID."""
    mlflow_run_id = config.get("mlflow_run_id")
    if mlflow_run_id is None:
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow_run_id = active_run.info.run_id
    return MlflowClient(), mlflow_run_id
