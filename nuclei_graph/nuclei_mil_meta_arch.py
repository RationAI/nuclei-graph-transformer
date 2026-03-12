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

from nuclei_graph.data.block_mask import mask_mixed_blocks
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    Outputs,
    PredictBatch,
)


class NucleiMILMetaArch(LightningModule):
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
        self.val_step_losses: list[Tensor] = []
        self.val_step_sizes: list[int] = []

    def forward(self, batch: Batch) -> Outputs:
        block_mask = batch["block_mask"]

        # in case of validation/test/pediction stage we have to handle mixed blocks
        if not self.training:
            block_mask = mask_mixed_blocks(block_mask, batch["seq_len"])

        return self.net(batch["x"], batch["pos"], block_mask, batch["seq_len"])

    def training_step(self, batch: Batch) -> Tensor:
        targets = batch["y"]["graph"]
        assert targets is not None
        targets = targets.view(-1)

        logits = self(batch)["graph"].view(-1)

        loss = self.bce(logits, targets)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )
        return loss

    def validation_step(self, batch: Batch) -> None:
        targets = batch["y"]["graph"]
        assert targets is not None
        targets = targets.view(-1)

        logits = self(batch)["graph"].view(-1)

        loss = self.bce(logits, targets)
        self.log(
            "validation/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )
        self.val_metrics.update(torch.sigmoid(logits), targets.long())

        batch_size = targets.size(0)
        self.val_step_losses.append(loss.detach() * batch_size)
        self.val_step_sizes.append(batch_size)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

        if not self.val_step_losses:
            return

        total_loss = torch.stack(self.val_step_losses).sum()
        total_size = sum(self.val_step_sizes)
        val_loss = (total_loss / total_size).item()

        self.val_step_losses.clear()
        self.val_step_sizes.clear()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_metrics: dict[str, Tensor] = {
                "best/validation/loss": torch.tensor(val_loss, dtype=torch.float32),
                "best/epoch": torch.tensor(self.current_epoch, dtype=torch.int64),
            }
            for k, v in metrics.items():
                best_metrics[f"best/{k}"] = v

            self.best_val_metrics = best_metrics
            self.log_dict(best_metrics, prog_bar=False)

    def test_step(self, batch: Batch) -> None:
        targets = batch["y"]["graph"]
        assert targets is not None
        targets = targets.view(-1)

        logits = self(batch)["graph"].view(-1)

        loss = self.bce(logits, targets)
        self.log(
            "test/loss", loss, on_epoch=True, prog_bar=True, batch_size=targets.size(0)
        )
        self.test_metrics.update(torch.sigmoid(logits), targets.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: PredictBatch) -> Outputs:
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
