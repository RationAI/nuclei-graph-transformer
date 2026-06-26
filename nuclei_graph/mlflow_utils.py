from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


def tag_parent_run(run_id: str, parent_run_id: str | None) -> None:
    """Tags an MLflow run as a child of an existing run, for UI nesting."""
    if parent_run_id:
        MlflowClient().set_tag(run_id, MLFLOW_PARENT_RUN_ID, parent_run_id)
