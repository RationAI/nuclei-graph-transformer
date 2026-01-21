from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

from nuclei_graph.nuclei_graph_typing import PredictInput, Sample


class NucleiWSMetaArch(LightningModule):
    def __init__(self, lr: float, warmup_epochs: int, net: nn.Module):
        super().__init__()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.net = net
        self.bce = nn.BCEWithLogitsLoss()

        metrics: dict[str, Metric | MetricCollection] = {
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "AUROC": BinaryAUROC(),
            "AUPRC": BinaryAveragePrecision(),
        }

        self.val_metrics = MetricCollection(metrics, prefix="validation/")
        self.test_metrics = MetricCollection(metrics, prefix="test/")
        self.predict_metrics = MetricCollection(metrics, prefix="prediction/")

    def forward(self, batch: Sample) -> Tensor:
        return self.net(
            batch["x"], batch["pos"], batch["block_mask"], batch["num_points"]
        )

    def training_step(self, batch: Sample) -> Tensor:
        targets_masked = batch["y"]
        logits = self(batch)
        logits_masked = logits[batch["sup_mask"]]
        assert targets_masked.shape == logits_masked.shape

        masked_size = targets_masked.numel()
        self.log("train/masked_batch_size", float(masked_size), on_step=True)
        assert masked_size > 0, "There are no annotated targets to compute loss from"

        loss = self.bce(logits_masked, targets_masked)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=masked_size,
        )
        return loss

    def validation_step(self, batch: Sample) -> None:
        targets_masked = batch["y"]
        logits = self(batch)
        logits_masked = logits[batch["sup_mask"]]
        assert targets_masked.shape == logits_masked.shape

        masked_size = targets_masked.numel()
        if masked_size == 0:  # there are no annotated targets to compute loss from
            return None  # skip this batch

        loss = self.bce(logits_masked, targets_masked)
        self.log(
            "validation/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=masked_size,
        )
        self.val_metrics.update(torch.sigmoid(logits_masked), targets_masked.long())

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: Sample) -> None:
        targets_masked = batch["y"]
        logits = self(batch)
        logits_masked = logits[batch["sup_mask"]]
        assert targets_masked.shape == logits_masked.shape

        self.test_metrics.update(torch.sigmoid(logits_masked), targets_masked.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()

    def predict_step(self, batch: PredictInput) -> Tensor:
        sample, _ = batch
        logits = self(sample)
        return logits

    def _get_optimizer_params(self) -> list[dict[str, Any]]:
        decay_params = []
        no_decay_params = []

        for param in self.net.parameters():
            if not param.requires_grad:
                continue
            if getattr(param, "_no_weight_decay", False) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self._get_optimizer_params(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        assert self.trainer.max_epochs is not None  # set in config
        steps_per_epoch = total_steps // self.trainer.max_epochs
        warmup_steps = max(1, self.warmup_epochs * steps_per_epoch)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1e-2,
                    end_factor=1.0,
                    total_iters=int(warmup_steps),
                ),
                CosineAnnealingLR(
                    optimizer, T_max=int(total_steps - warmup_steps), eta_min=1e-6
                ),  # type: ignore[arg-type]
            ],
            milestones=[int(warmup_steps)],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
