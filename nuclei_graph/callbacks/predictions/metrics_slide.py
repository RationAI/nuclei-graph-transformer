import torch
from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import NestedMetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryNegativePredictiveValue,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch


class WSLSlidePredictionMetricsCallback(Callback):
    def __init__(self, threshold: float) -> None:
        self.slide_nuclei_metrics = NestedMetricCollection(
            metrics={
                "AUPRC": BinaryAveragePrecision(),
                "AUROC": BinaryAUROC(),
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
        slide = batch["slides"]  # batch size is 1
        metadata = batch["metadata"][0]

        targets_sup = slide["y"]["nuclei"]
        if targets_sup.numel() == 0:
            return

        logits_sup = outputs["nuclei"][slide["sup_mask"]].squeeze(-1)
        assert targets_sup.shape == logits_sup.shape

        preds_sup = torch.sigmoid(logits_sup)
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


class MILSlidePredictionMetricsCallback(Callback):
    def __init__(self, threshold_nuclei: float, threshold_graph: float) -> None:
        self.slide_nuclei_metrics = NestedMetricCollection(
            metrics={
                "AUPRC": BinaryAveragePrecision(),
                "AUROC": BinaryAUROC(),
                "accuracy": BinaryAccuracy(threshold_nuclei),
                "precision": BinaryPrecision(threshold_nuclei),
                "recall": BinaryRecall(threshold_nuclei),
                "specificity": BinarySpecificity(threshold_nuclei),
                "negative_predictive_value": BinaryNegativePredictiveValue(
                    threshold_nuclei
                ),
            }
        )
        self.slide_graph_metrics = NestedMetricCollection(
            metrics={
                "AUPRC": BinaryAveragePrecision(),
                "AUROC": BinaryAUROC(),
                "accuracy": BinaryAccuracy(threshold_graph),
                "precision": BinaryPrecision(threshold_graph),
                "recall": BinaryRecall(threshold_graph),
                "specificity": BinarySpecificity(threshold_graph),
                "negative_predictive_value": BinaryNegativePredictiveValue(
                    threshold_graph
                ),
            }
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.slide_nuclei_metrics = self.slide_nuclei_metrics.to(pl_module.device)
        self.slide_graph_metrics = self.slide_graph_metrics.to(pl_module.device)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        slide = batch["slides"]  # batch size is 1
        metadata = batch["metadata"][0]

        targets_graph = slide["y"]["graph"]
        if targets_graph is not None:
            logits_graph = outputs["graph"].view(-1)
            targets_graph = targets_graph.view(-1)

            preds_graph = torch.sigmoid(logits_graph)
            keys = [metadata["slide_id"]] * len(preds_graph)
            self.slide_graph_metrics.update(preds_graph, targets_graph.long(), keys)

        targets_sup = slide["y"]["nuclei"]
        if targets_sup is not None and targets_sup.numel() > 0:
            logits_sup = outputs["nuclei"][slide["sup_mask"]].squeeze(-1)
            assert targets_sup.shape == logits_sup.shape

            preds_sup = torch.sigmoid(logits_sup)
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

        trainer.logger.log_table(
            self.slide_graph_metrics.compute(), "slide_graph_metrics.json"
        )
        self.slide_graph_metrics.reset()
