from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from warmup_scheduler import GradualWarmupScheduler

from nuclei_graph.typing import Outputs, PredictInput, Sample


THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]


class NucleiGraphTransformer(LightningModule):
    def __init__(self, lr: float, net: nn.Module):
        super().__init__()
        self.lr = lr
        self.net = net
        self.criterion = nn.BCEWithLogitsLoss()

        thresholded_metrics: dict[str, Metric | MetricCollection] = {}
        for t in THRESHOLDS:
            t_str = str(t).replace(".", "_")
            thresholded_metrics[f"precision_{t_str}"] = BinaryPrecision(threshold=t)
            thresholded_metrics[f"recall_{t_str}"] = BinaryRecall(threshold=t)
        thresholded_metrics["AUROC"] = BinaryAUROC()
        thresholded_metrics["AUPRC"] = BinaryAveragePrecision()

        metrics: dict[str, Metric | MetricCollection] = {
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "AUROC": BinaryAUROC(),
            "AUPRC": BinaryAveragePrecision(),
        }

        self.val_metrics = MetricCollection(thresholded_metrics, prefix="validation/")
        self.test_metrics = MetricCollection(metrics, prefix="test/")
        self.predict_metrics = MetricCollection(metrics, prefix="prediction/")

    def forward(self, batch: Sample) -> Outputs:
        return self.net(batch["x"], batch["pos"], batch["block_mask"])

    def training_step(self, batch: Sample) -> Tensor:
        targets_masked = batch["y"]
        logits = self(batch)
        logits_masked = logits[batch["annot_mask"]]
        assert targets_masked.shape == logits_masked.shape

        masked_size = targets_masked.numel()
        assert masked_size > 0, "There are no annotated targets to compute loss from"

        loss = self.criterion(logits_masked, targets_masked)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Sample) -> None:
        targets_masked = batch["y"]
        logits = self(batch)
        logits_masked = logits[batch["annot_mask"]]
        assert targets_masked.shape == logits_masked.shape

        masked_size = targets_masked.numel()
        if masked_size == 0:  # there are no annotated targets to compute loss from
            return None  # skip this batch

        loss = self.criterion(logits_masked, targets_masked)
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
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: Sample) -> None:
        targets_masked = batch["y"]
        logits = self(batch)
        logits_masked = logits[batch["annot_mask"]]
        assert targets_masked.shape == logits_masked.shape

        self.test_metrics.update(torch.sigmoid(logits_masked), targets_masked.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: PredictInput) -> Outputs:
        sample, _ = batch
        logits = self(sample)
        return logits

    def on_train_start(self) -> None:
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if isinstance(scheduler, GradualWarmupScheduler):
            cosine = scheduler.after_scheduler
            assert cosine is not None
            total_epochs = (
                self.trainer.estimated_stepping_batches
                / self.trainer.num_training_batches
            )
            remaining_epochs = max(1, int(total_epochs - scheduler.total_epoch))
            cosine.T_max = remaining_epochs
            print(f"Set CosineAnnealingLR T_max to {cosine.T_max}")

    def _get_optimizer_params(self) -> list[dict[str, Any]]:
        """Groups model parameters into those with weight decay and those without.

        Excludes weight decay for all tagged parameters (layer scale params, RoPE freqs and P matrix),
        biases and norms.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue

            if (
                hasattr(param, "_no_weight_decay")
                or name.endswith(".bias")
                or "norm" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_grouped_parameters = self._get_optimizer_params()
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=1,  # placeholder, overwritten in `on_train_start`
            eta_min=1.0e-06,
        )
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "validation/loss",
            },
        }
