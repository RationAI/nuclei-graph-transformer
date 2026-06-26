from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run


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
        tag_parent_run(trainer.logger.run_id, self.parent_run_id)
