import torch
from lightning import Callback, LightningModule, Trainer
from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import NestedMetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryNegativePredictiveValue,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


class WSLSlidePredictionMetricsCallback(Callback):
    """Computes nuclei-level metrics per slide.

    The metrics are also saved as a JSON table in MLflow.
    """

    def __init__(self, threshold: float) -> None:
        self.slide_nuclei_metrics = NestedMetricCollection(
            metrics={
                "accuracy": BinaryAccuracy(threshold),
                "precision": BinaryPrecision(threshold),
                "recall": BinaryRecall(threshold),
                "specificity": BinarySpecificity(threshold),
                "negative_predictive_value": BinaryNegativePredictiveValue(threshold),
            }
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.slide_nuclei_metrics = self.slide_nuclei_metrics.to(pl_module.device)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        slide = batch["slides"]

        targets_sup = slide["y"]["nuclei"]
        if targets_sup.numel() == 0:
            return

        logits_sup = outputs["nuclei"][slide["sup_mask"]].squeeze(-1)
        assert targets_sup.shape == logits_sup.shape

        preds_sup = torch.sigmoid(logits_sup)
        metadata = batch["metadata"][0]  # batch size is 1
        keys = [metadata["slide_id"]] * len(preds_sup)
        self.slide_nuclei_metrics.update(preds_sup, targets_sup.long(), keys)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        trainer.logger.log_table(
            self.slide_nuclei_metrics.compute(), "slide_nuclei_metrics.json"
        )
        self.slide_nuclei_metrics.reset()
