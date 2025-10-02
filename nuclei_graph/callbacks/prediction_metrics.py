"""Lightning callbacks to save nuclei per-slide and per-dataset metrics."""

from typing import cast

import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from torchmetrics import MetricCollection

from nuclei_graph.typing import Outputs, PredictInput


class PredictionMetricsCallback(Callback):
    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics = metrics_module.compute()
        for key, value in metrics.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(f"prediction/{metric_name}", float(value))
        metrics_module.reset()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        sample, _ = batch
        targets_masked = sample["y"]
        logits_masked = outputs[sample["annot_mask"]]
        assert targets_masked.shape == logits_masked.shape

        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics_module.update(torch.sigmoid(logits_masked), targets_masked.long())


class PredictionMetricsBatchCallback(Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        sample, metadata_list = batch
        metadata = metadata_list[0]
        targets_masked = sample["y"]
        logits_masked = outputs[sample["annot_mask"]]
        assert targets_masked.shape == logits_masked.shape

        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        slide_metrics = metrics_module.clone()
        slide_metrics.update(torch.sigmoid(logits_masked), targets_masked.long())

        slide_id = metadata["slide_id"]
        metrics = slide_metrics.compute()
        for key, value in metrics.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(
                f"prediction/{slide_id}/{metric_name}",
                float(value),
                step=pl_module.global_step,
            )
        slide_metrics.reset()
