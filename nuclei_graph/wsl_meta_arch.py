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


class WSLMetaArch(LightningModule):
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
        logits = self(batch)
        probs = torch.sigmoid(logits)

        sup_mask = batch["sup_mask"]
        logits_sup = logits[sup_mask]
        targets_sup = batch["y"]

        assert logits_sup.numel() == targets_sup.numel()
        self.log("train/sup_batch_size", float(logits_sup.numel()), on_step=True)

        # it is assumed training batches do not contain padding
        loss_sup = (
            self.bce(logits_sup, targets_sup)
            if logits_sup.numel() > 0
            else torch.tensor(0.0, device=self.device, requires_grad=True)
        )

        probs_unsup = probs[~sup_mask]
        entropy = -(
            probs_unsup * torch.log(probs_unsup + 1e-8)
            + (1 - probs_unsup) * torch.log(1 - probs_unsup + 1e-8)
        )
        loss_entropy = (
            -entropy.mean()
            if probs_unsup.numel() > 0
            else torch.tensor(0.0, device=self.device, requires_grad=True)
        )
        total_loss = loss_sup + 0.1 * loss_entropy

        self.log("train/loss_sup", loss_sup, on_step=True, prog_bar=True)
        self.log("train/loss_ent", loss_entropy, on_step=True, prog_bar=True)
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss

    def validation_step(self, batch: Sample) -> None:
        targets_sup = batch["y"]
        logits = self(batch)
        logits_sup = logits[batch["sup_mask"]]
        assert targets_sup.shape == logits_sup.shape

        sup_size = targets_sup.numel()
        if sup_size == 0:
            return None

        loss = self.bce(logits_sup, targets_sup)
        self.log(
            "validation/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.val_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: Sample) -> None:
        targets_sup = batch["y"]
        logits = self(batch)
        logits_sup = logits[batch["sup_mask"]]
        assert targets_sup.shape == logits_sup.shape

        self.test_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

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
