from typing import cast

import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchmetrics import MetricCollection

from nuclei_graph.nuclei_graph_typing import PredictBatch


class PredictionMetricsCallback(Callback):
    """Computes global metrics across the entire dataset."""

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        metrics = cast("MetricCollection", pl_module.predict_metrics_global)
        computed_metrics = metrics.compute()

        for key, value in computed_metrics.items():
            metric_name = key.split("/")[-1]
            value = float(value)
            mlflow.log_metric(f"prediction/{metric_name}", value)

        metrics.reset()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        slide = batch["slides"]  # batch size is 1
        targets_sup = slide["y"]

        if targets_sup.numel() == 0:
            return

        logits_sup = outputs.squeeze(-1)[slide["sup_mask"]]
        assert targets_sup.shape == logits_sup.shape

        preds_sup = torch.sigmoid(logits_sup)
        metrics = cast("MetricCollection", pl_module.predict_metrics_global)
        metrics.update(preds_sup, targets_sup.long())


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
        slide = batch["slides"]  # batch size is 1
        metadata = batch["metadata"][0]

        targets_sup = slide["y"]
        logits_sup = outputs.squeeze(-1)[slide["sup_mask"]]
        assert targets_sup.shape == logits_sup.shape

        metrics = cast("MetricCollection", pl_module.predict_metrics_slide)
        metrics.update(torch.sigmoid(logits_sup), targets_sup.long())
        computed_metrics = metrics.compute()

        for key, value in computed_metrics.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(
                f"prediction/{metadata['slide_id']}/{metric_name}",
                float(value),
                step=pl_module.global_step,
            )

        metrics.reset()
