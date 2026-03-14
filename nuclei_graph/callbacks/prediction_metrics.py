from typing import cast

import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch
from torchmetrics import MetricCollection


class WSLPredictionMetricsCallback(Callback):
    """Computes nuclei-level global metrics across the entire dataset."""

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        metrics = cast("MetricCollection", pl_module.predict_metrics)
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
        metrics = cast("MetricCollection", pl_module.predict_metrics)
        metrics.update(preds_sup, targets_sup.long())


class MILPredictionMetricsCallback(Callback):
    """Computes slide-level and nucleus-level metrics across the entire dataset."""

    def __init__(self) -> None:
        super().__init__()

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
            targets_graph = targets_graph.view(-1)

            preds_graph = torch.sigmoid(logits_graph)
            graph_metrics = cast("MetricCollection", pl_module.predict_graph_metrics)
            graph_metrics.update(preds_graph, targets_graph.long())

        targets_sup = slide["y"]["nuclei"]
        if targets_sup is not None and targets_sup.numel() > 0:
            logits_sup = outputs["nuclei"][slide["sup_mask"]].squeeze(-1)
            assert targets_sup.shape == logits_sup.shape

            preds_sup = torch.sigmoid(logits_sup)
            nuclei_metrics = cast("MetricCollection", pl_module.predict_nuclei_metrics)
            nuclei_metrics.update(preds_sup, targets_sup.long())

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        graph_metrics = cast("MetricCollection", pl_module.predict_graph_metrics)
        computed_graph = graph_metrics.compute()
        for key, value in computed_graph.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(f"prediction/graph/{metric_name}", float(value))

        graph_metrics.reset()

        nuclei_metrics = cast("MetricCollection", pl_module.predict_nuclei_metrics)
        computed_nuclei = nuclei_metrics.compute()
        for key, value in computed_nuclei.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(f"prediction/nuclei/{metric_name}", float(value))

        nuclei_metrics.reset()
