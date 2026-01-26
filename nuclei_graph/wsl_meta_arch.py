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

from nuclei_graph.data.augmentations.geom_augs import apply_augmentations
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    CriterionInput,
    PredictBatch,
)


class WSLMetaArch(LightningModule):
    def __init__(
        self,
        lr: float,
        warmup_epochs: int,
        net: nn.Module,
        criterion: nn.Module,
        use_augmentations: bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.net = net
        self.criterion = criterion
        self.use_augmentations = use_augmentations
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

    def forward(self, batch: Batch) -> Tensor:
        return self.net(
            batch["x"], batch["pos"], batch["block_mask"], batch["num_points"]
        )

    def training_step(self, batch: Batch) -> Tensor:
        rampup_epochs = 10
        current_weight_factor = min(1.0, self.current_epoch / rampup_epochs)

        logits_aug = (
            self(apply_augmentations(batch)) if self.use_augmentations else None
        )
        loss, logs = self.criterion(
            criterion_input=CriterionInput(logits=self(batch), logits_aug=logits_aug),
            targets_sup=batch["y"],
            masks=batch["masks"],
            weight_factor=current_weight_factor,
        )
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=logs["sup_size"],
        )
        return loss

    def validation_step(self, batch: Batch) -> None:
        targets_sup = batch["y"]
        sup_mask = batch["masks"]["sup_mask"]
        logits_sup = self(batch)[sup_mask]

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
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: Batch) -> None:
        targets_sup = batch["y"]
        sup_mask = batch["masks"]["sup_mask"]
        logits_sup = self(batch)[sup_mask]

        sup_size = targets_sup.numel()
        if sup_size == 0:
            return None

        self.test_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: PredictBatch) -> Tensor:
        return self(batch["items"])

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
            {"params": decay_params, "weight_decay": 1e-4},
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
