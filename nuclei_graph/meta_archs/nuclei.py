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

from nuclei_graph.data.datamodules.collator import ensure_block_mask
from nuclei_graph.nuclei_graph_typing import Batch, Outputs


TRAIN_LOSSES = ("bce", "focus_weighted", "crop_balanced")


def focus_weighted_bce(
    logits: Tensor, targets: Tensor, focus_nuclei: Tensor, power: float
) -> Tensor:
    """BCE with positive nuclei weighted by (focus size)^(-power), negatives by 1.

    The positive weights are renormalized to a mean of 1 over the positives of the batch;
    the loss is the weighted mean over all nuclei.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos = targets > 0.5
    weights = torch.ones_like(bce)
    if pos.any():
        sizes = focus_nuclei[pos].to(bce.dtype)
        assert bool((sizes > 0).all()), "positive nucleus without a focus size"
        w = sizes.pow(-power)
        weights[pos] = w / w.mean()
    return (weights * bce).mean()


def crop_balanced_bce(
    logits: Tensor, targets: Tensor, crop_ids: Tensor, n_crops: int
) -> Tensor:
    """Per crop, mean BCE of its positives + mean BCE of its negatives; mean over crops.

    A class absent from a crop contributes 0; crops without nuclei are left out.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos = targets > 0.5
    per_crop = bce.new_zeros(n_crops)
    for cls in (pos, ~pos):
        total = bce.new_zeros(n_crops).index_add(0, crop_ids[cls], bce[cls])
        count = torch.bincount(crop_ids[cls], minlength=n_crops)
        per_crop = per_crop + torch.where(count > 0, total / count.clamp(min=1), 0.0)
    has_nuclei = torch.bincount(crop_ids, minlength=n_crops) > 0
    return per_crop[has_nuclei].mean()


class NucleiModelMetaArch(LightningModule):
    """Nucleus-level classification.

    Args:
        train_loss: Training loss over the supervised nuclei: "bce" (mean BCE), "focus_weighted"
            (`focus_weighted_bce`; needs the `focus_nuclei` labels of the dataset's `foci`) or
            "crop_balanced" (`crop_balanced_bce`). Validation and test always use the mean BCE.
        focus_weight_power: The exponent of "focus_weighted".
    """

    def __init__(
        self,
        lr: float,
        warmup_epochs: int,
        net: nn.Module,
        train_loss: str = "bce",
        focus_weight_power: float = 0.5,
    ) -> None:
        super().__init__()
        assert train_loss in TRAIN_LOSSES, f"Invalid train_loss: {train_loss}"
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.net = net
        self.train_loss = train_loss
        self.focus_weight_power = focus_weight_power
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

    def forward(self, inputs: Batch) -> Outputs:
        ensure_block_mask(inputs)
        return self.net(
            x=inputs["features"],
            pos=inputs["pos"],
            block_mask=inputs["block_mask"],
            seq_lens=inputs["seq_lens"],
            neighbor_idx=inputs["global_neighbor_idx"],
            bboxes=inputs.get("bboxes"),
        )

    def training_step(self, batch: Batch) -> Tensor:
        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        assert batch["labels"]["nuclei"] is not None
        targets_sup = batch["labels"]["nuclei"][:seq_len][sup_mask]

        logits = self(batch)
        logits_sup = logits["nuclei"][sup_mask].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:
            return logits_sup.sum() * 0.0

        # log the ratio of positive and negative nuclei in the batch
        n_pos = (targets_sup == 1).sum().float()
        pos_ratio = n_pos / sup_size if sup_size > 0 else 0.0
        self.log("train/pos_ratio", pos_ratio, on_step=True, prog_bar=True)

        loss_sup = self.training_loss(batch, logits_sup, targets_sup, sup_mask)
        self.log(
            "train/loss",
            loss_sup,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(sup_size),
        )
        return loss_sup

    def training_loss(
        self, batch: Batch, logits_sup: Tensor, targets_sup: Tensor, sup_mask: Tensor
    ) -> Tensor:
        """The `train_loss` of the supervised nuclei (`sup_mask` over the real nuclei)."""
        if self.train_loss == "bce":
            return self.bce(logits_sup, targets_sup)
        if self.train_loss == "focus_weighted":
            focus_nuclei = batch["labels"].get("focus_nuclei")
            assert focus_nuclei is not None, "focus_weighted needs the dataset's foci"
            return focus_weighted_bce(
                logits_sup,
                targets_sup,
                focus_nuclei[: len(sup_mask)][sup_mask],
                self.focus_weight_power,
            )
        seq_lens = batch["seq_lens"]
        crop_ids = torch.repeat_interleave(
            torch.arange(len(seq_lens), device=seq_lens.device), seq_lens.long()
        )
        return crop_balanced_bce(
            logits_sup, targets_sup, crop_ids[sup_mask], len(seq_lens)
        )

    def validation_step(self, batch: Batch) -> Outputs | None:
        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        assert batch["labels"]["nuclei"] is not None
        targets_sup = batch["labels"]["nuclei"][:seq_len][sup_mask]

        logits = self(batch)
        logits_nuclei_sup = logits["nuclei"][sup_mask].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:
            return None

        loss_sup = self.bce(logits_nuclei_sup, targets_sup)
        self.log(
            "validation/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.val_metrics.update(torch.sigmoid(logits_nuclei_sup), targets_sup.long())
        self.val_step_losses.append(loss_sup.detach() * sup_size)
        self.val_step_sizes.append(sup_size)
        return logits

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
        seq_len = int(batch["seq_lens"].sum().item())

        sup_mask = batch["sup_mask"][:seq_len]

        assert batch["labels"]["nuclei"] is not None
        targets_sup = batch["labels"]["nuclei"][:seq_len][sup_mask]

        logits = self(batch)
        logits_nuclei_sup = logits["nuclei"][sup_mask].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:
            return

        loss_sup = self.bce(logits_nuclei_sup, targets_sup)
        self.log(
            "test/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.test_metrics.update(torch.sigmoid(logits_nuclei_sup), targets_sup.long())

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: Batch) -> Outputs:
        return self(batch)

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
