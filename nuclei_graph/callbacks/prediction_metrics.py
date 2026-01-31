from typing import cast

import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchmetrics import MetricCollection

from nuclei_graph.nuclei_graph_typing import PredictBatch


class PredictionMetricsCallback(Callback):
    """Computes global metrics across the entire dataset."""

    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics_module.reset()

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics = metrics_module.compute()

        for key, value in metrics.items():
            metric_name = key.split("/")[-1]
            value = float(value.item()) if isinstance(value, Tensor) else float(value)
            mlflow.log_metric(f"prediction/{metric_name}", value)
        metrics_module.reset()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        sup_mask = batch["batch"]["wsl_masks"]["sup_mask"]
        targets_sup = batch["batch"]["y"]
        logits_sup = outputs.squeeze(-1)[sup_mask]
        assert targets_sup.shape == logits_sup.shape

        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics_module.update(torch.sigmoid(logits_sup), targets_sup.long())


class PredictionSlideMetricsCallback(Callback):
    """Computes metrics per slide (prediction batch size is assumed to be 1)."""

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"][0]  # batch size is 1
        sup_mask = batch["batch"]["wsl_masks"]["sup_mask"]
        targets_sup = batch["batch"]["y"]
        logits_sup = outputs.squeeze(-1)[sup_mask]
        assert targets_sup.shape == logits_sup.shape

        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        slide_metrics = metrics_module.clone()
        slide_metrics.reset()
        slide_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

        metrics = slide_metrics.compute()
        for key, value in metrics.items():
            metric_name = key.split("/")[-1]
            value = float(value.item()) if isinstance(value, Tensor) else float(value)
            mlflow.log_metric(
                f"prediction/{metadata['slide_id']}/{metric_name}",
                value,
                step=pl_module.global_step,
            )
        slide_metrics.reset()
