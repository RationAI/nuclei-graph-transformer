import matplotlib.pyplot as plt
import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


class WSLDatasetPredictionMetricsCallback(Callback):
    """Computes nuclei-level global metrics across the entire dataset and logs them to MLflow."""

    def __init__(self, threshold: float) -> None:
        self.dataset_nuclei_metrics = MetricCollection(
            metrics={
                "AUPRC": BinaryAveragePrecision(),
                "AUROC": BinaryAUROC(),
                "precision": BinaryPrecision(threshold),
                "recall": BinaryRecall(threshold),
                "accuracy": BinaryAccuracy(threshold),
                "specificity": BinarySpecificity(threshold),
            },
            prefix="prediction/",
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.dataset_nuclei_metrics = self.dataset_nuclei_metrics.to(pl_module.device)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        computed_metrics = self.dataset_nuclei_metrics.compute()

        for key, value in computed_metrics.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(f"prediction/{metric_name}", float(value))

        self.dataset_nuclei_metrics.reset()

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
        targets_sup = slide["y"]["nuclei"]

        if targets_sup.numel() == 0:
            return

        logits_sup = outputs["nuclei"][slide["sup_mask"]].squeeze(-1)
        assert targets_sup.shape == logits_sup.shape

        preds_sup = torch.sigmoid(logits_sup)
        self.dataset_nuclei_metrics.update(preds_sup, targets_sup.long())


class MILDatasetPredictionMetricsCallback(Callback):
    """Computes slide-level metrics across the entire dataset and logs them to MLflow."""

    def __init__(self, threshold: float) -> None:
        self.dataset_graph_metrics = MetricCollection(
            metrics={
                "AUPRC_graph": BinaryAveragePrecision(),
                "AURO_graphC": BinaryAUROC(),
                "precision_graph": BinaryPrecision(threshold),
                "recall_graph": BinaryRecall(threshold),
                "accuracy_graph": BinaryAccuracy(threshold),
                "specificity_graph": BinarySpecificity(threshold),
                "confusion_matrix_graph": BinaryConfusionMatrix(threshold),
            },
            prefix="prediction/",
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.dataset_graph_metrics = self.dataset_graph_metrics.to(pl_module.device)

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

        targets_graph = slide["y"]["graph"]
        if targets_graph is not None:
            logits_graph = outputs["graph"].view(-1)
            preds_graph = torch.sigmoid(logits_graph)

            self.dataset_graph_metrics.update(
                preds_graph, targets_graph.view(-1).long()
            )

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        computed_metrics = self.dataset_graph_metrics.compute()
        for key, value in computed_metrics.items():
            metric_name = key.split("/")[-1]

            if "confusion_matrix" in metric_name:
                cm_array = value.cpu().numpy()
                disp_cf = ConfusionMatrixDisplay(
                    confusion_matrix=cm_array, display_labels=["Negative", "Positive"]
                )

                fig, ax = plt.subplots(figsize=(6, 5))
                disp_cf.plot(ax=ax, cmap="Blues", colorbar=False)
                plt.title("Slide-Level Confusion Matrix")
                plt.tight_layout()

                mlflow.log_figure(fig, "confusion_matrix.png")

                plt.close(fig)
            else:
                mlflow.log_metric(f"prediction/{metric_name}", float(value))

        self.dataset_graph_metrics.reset()
