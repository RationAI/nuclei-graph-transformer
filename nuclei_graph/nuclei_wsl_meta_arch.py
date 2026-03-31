from typing import Any

import torch
import torch.nn.functional as F
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
from nuclei_graph.nuclei_graph_typing import Batch, Outputs, PredictBatch


class NucleiWSLMetaArch(LightningModule):
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
        targets_sup = batch["y"]["nuclei"]
        assert targets_sup is not None

        logits = self(batch)["nuclei"]
        logits_sup = logits[batch["sup_mask"]].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:  # empty supervision batch
            return logits.sum() * 0.0

        # compute weights s.t. sum(positive weights) == sum(negative weights)
        n_pos = (targets_sup == 1).sum().float()
        # n_neg = (targets_sup == 0).sum().float()
        # num_classes = (n_pos > 0).float() + (n_neg > 0).float()

        # weight_pos = float(sup_size) / (num_classes * n_pos.clamp(min=1.0))
        # weight_neg = float(sup_size) / (num_classes * n_neg.clamp(min=1.0))
        # weights = torch.where(targets_sup == 1, weight_pos, weight_neg)

        pos_ratio = n_pos / sup_size if sup_size > 0 else 0.0
        self.log("train/pos_ratio", pos_ratio, on_step=True, prog_bar=True)

        loss_sup = F.binary_cross_entropy_with_logits(
            logits_sup,
            targets_sup,
            # weight=weights,
        )
        self.log(
            "train/loss",
            loss_sup,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(sup_size),
        )
        return loss_sup

    def validation_step(self, batch: Batch) -> None:
        targets_sup = batch["y"]["nuclei"]
        assert targets_sup is not None

        logits = self(batch)["nuclei"]
        logits_sup = logits[batch["sup_mask"]].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:  # empty supervision batch
            return None

        loss_sup = self.bce(logits_sup, targets_sup)
        self.log(
            "validation/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.val_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())
        self.val_step_losses.append(loss_sup.detach() * sup_size)
        self.val_step_sizes.append(sup_size)

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
                "best/epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
            }
            for k, v in metrics.items():
                best_metrics[f"best/{k}"] = v

            self.best_val_metrics = best_metrics
            self.log_dict(best_metrics, prog_bar=False)

    def test_step(self, batch: Batch) -> None:
        targets_sup = batch["y"]["nuclei"]
        assert targets_sup is not None

        logits = self(batch)["nuclei"]
        logits_sup = logits[batch["sup_mask"]].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:  # empty supervision batch
            return None
        loss_sup = self.bce(logits_sup, targets_sup)
        self.log(
            "test/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.test_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: PredictBatch) -> Outputs:
        return self(batch["slides"])

    def _get_optimizer_params(self) -> list[dict[str, Any]]:
        decay_params = []
        no_decay_params = []

        for n, w in self.net.named_parameters():
            if not w.requires_grad:
                continue
            if w.ndim <= 1 or ".rope." in n:
                no_decay_params.append(w)
            else:
                decay_params.append(w)

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
