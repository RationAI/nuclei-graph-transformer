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

from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PredictBatch,
)


class WSLMetaArch(LightningModule):
    def __init__(self, lr: float, warmup_epochs: int, net: nn.Module) -> None:
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

        self.best_val_loss = float("inf")
        self.best_val_metrics: dict[str, Tensor] = {}

    def forward(self, batch: Batch) -> Tensor:
        return self.net(batch["x"], batch["pos"], batch["block_mask"])

    def training_step(self, batch: Batch) -> Tensor:
        logits = self(batch).squeeze(-1)
        logits_sup = logits[batch["sup_mask"]]
        targets_sup = batch["y"]
        sup_size = targets_sup.numel()

        loss_sup = (
            self.bce(logits_sup, targets_sup) if sup_size > 0 else logits.sum() * 0.0
        )
        self.log_dict({"train/sup_size": sup_size}, on_step=True)
        self.log(
            "train/loss",
            loss_sup,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=sup_size,
        )
        return loss_sup

    def validation_step(self, batch: Batch) -> None:
        logits = self(batch).squeeze(-1)
        sup_mask = batch["sup_mask"]
        logits_sup = logits[sup_mask]

        targets_sup = batch["y"]

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

        val_loss_t = self.trainer.callback_metrics.get("validation/loss")
        if val_loss_t is None:
            return
        val_loss = float(val_loss_t.detach().item())

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_metrics: dict[str, Tensor] = {
                "best/validation/loss": torch.tensor(val_loss, dtype=torch.float32),
                "best/epoch": torch.tensor(self.current_epoch, dtype=torch.int64),
            }
            for k, v in metrics.items():
                best_metrics[f"best/validation/{k}"] = v

            self.best_val_metrics = best_metrics
            self.log_dict(best_metrics, prog_bar=False)

    def test_step(self, batch: Batch) -> None:
        logits = self(batch).squeeze(-1)
        sup_mask = batch["sup_mask"]
        logits_sup = logits[sup_mask]

        targets_sup = batch["y"]

        sup_size = targets_sup.numel()
        if sup_size == 0:
            return None

        self.test_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: PredictBatch) -> Tensor:
        return self(batch["slides"])

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
                ),
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
