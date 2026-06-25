from lightning import Callback, LightningModule, Trainer
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from rationai.mlkit.lightning.loggers import MLFlowLogger


class ParentRunTagCallback(Callback):
    """Tags the active MLflow run as a child of an existing run, for UI nesting."""

    def __init__(self, parent_run_id: str | None = None) -> None:
        super().__init__()
        self.parent_run_id = parent_run_id

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not self.parent_run_id:
            return

        assert (
            isinstance(trainer.logger, MLFlowLogger)
            and trainer.logger.run_id is not None
        )
        MlflowClient().set_tag(
            trainer.logger.run_id, MLFLOW_PARENT_RUN_ID, self.parent_run_id
        )
