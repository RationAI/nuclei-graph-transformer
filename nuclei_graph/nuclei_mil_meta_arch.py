from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
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
    def _create_metrics(self, prefix: str) -> MetricCollection:
        return MetricCollection(
            {
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "AUROC": BinaryAUROC(),
                "AUPRC": BinaryAveragePrecision(),
            },
            prefix=prefix,
        )

    def __init__(self, lr: float, warmup_epochs: int, net: nn.Module) -> None:
        super().__init__()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.net = net
        self.bce = nn.BCEWithLogitsLoss()

        self.val_graph_metrics = self._create_metrics("validation/graph/")
        self.test_graph_metrics = self._create_metrics("test/graph/")
        self.predict_graph_metrics = self._create_metrics("prediction/graph/")

        self.val_nuclei_metrics = self._create_metrics("validation/nuclei/")
        self.test_nuclei_metrics = self._create_metrics("test/nuclei/")
        self.predict_nuclei_metrics = self._create_metrics("prediction/nuclei/")

        self.best_val_graph_loss = float("inf")
        self.best_val_graph_metrics: dict[str, Tensor] = {}
        self.val_step_graph_losses: list[Tensor] = []
        self.val_step_graph_sizes: list[int] = []

    def forward(self, batch: Batch) -> Outputs:
        block_mask = batch["block_mask"]

        # in case of validation/test/pediction stage we have to handle mixed blocks
        if not self.training:
            block_mask = mask_mixed_blocks(block_mask, batch["seq_len"])

        return self.net(batch["x"], batch["pos"], block_mask, batch["seq_len"])

    def training_step(self, batch: Batch) -> Tensor:
        targets_graph = batch["y"]["graph"]
        assert targets_graph is not None
        targets_graph = targets_graph.view(-1)

        logits_graph = self(batch)["graph"].view(-1)

        loss_graph = self.bce(logits_graph, targets_graph)

        self.log(
            "train/graph/loss",
            loss_graph,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets_graph.size(0),
        )
        return loss_graph

    def validation_step(self, batch: Batch) -> None:
        logits = self(batch)

        # graph-level metrics
        targets_graph = batch["y"]["graph"]
        assert targets_graph is not None
        targets_graph = targets_graph.view(-1)

        logits_graph = logits["graph"].view(-1)

        loss_graph = self.bce(logits_graph, targets_graph)
        self.log(
            "validation/graph/loss",
            loss_graph,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets_graph.size(0),
        )
        self.val_graph_metrics.update(torch.sigmoid(logits_graph), targets_graph.long())

        batch_size = targets_graph.size(0)
        self.val_step_graph_losses.append(loss_graph.detach() * batch_size)
        self.val_step_graph_sizes.append(batch_size)

        # nuclei-level metrics
        targets_sup = batch["y"]["nuclei"]
        assert targets_sup is not None

        logits_sup = logits["nuclei"][batch["sup_mask"]].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:  # empty supervision batch
            return None

        loss_sup = self.bce(logits_sup, targets_sup)
        self.log(
            "validation/nuclei/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.val_nuclei_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

    def on_validation_epoch_end(self) -> None:
        # compute and reset nuclei-level metrics
        nuclei_metrics = self.val_nuclei_metrics.compute()
        self.log_dict(nuclei_metrics, on_epoch=True, prog_bar=True)
        self.val_nuclei_metrics.reset()

        # compute and reset graph-level metrics
        graph_metrics = self.val_graph_metrics.compute()
        self.log_dict(graph_metrics, on_epoch=True, prog_bar=True)
        self.val_graph_metrics.reset()

        if not self.val_step_graph_losses:
            return

        # compute the best validation graph loss
        total_loss = torch.stack(self.val_step_graph_losses).sum()
        total_size = sum(self.val_step_graph_sizes)
        val_loss = (total_loss / total_size).item()

        self.val_step_graph_losses.clear()
        self.val_step_graph_sizes.clear()

        if val_loss < self.best_val_graph_loss:
            self.best_val_graph_loss = val_loss
            val_loss_name = "best/validation/graph/loss"
            best_metrics: dict[str, Tensor] = {
                val_loss_name: torch.tensor(val_loss, dtype=torch.float32),
                "best/graph/epoch": torch.tensor(
                    self.current_epoch, dtype=torch.float32
                ),
            }
            for k, v in graph_metrics.items():
                clean_key = k.replace("validation/", "best/")
                best_metrics[clean_key] = v

            self.best_val_graph_metrics = best_metrics
            self.log_dict(best_metrics, prog_bar=False)

    def test_step(self, batch: Batch) -> None:
        logits = self(batch)

        # graph-level metrics
        targets_graph = batch["y"]["graph"]
        assert targets_graph is not None
        targets_graph = targets_graph.view(-1)

        logits_graph = logits["graph"].view(-1)

        loss_graph = self.bce(logits_graph, targets_graph)
        self.log(
            "test/graph/loss",
            loss_graph,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets_graph.size(0),
        )
        self.test_graph_metrics.update(
            torch.sigmoid(logits_graph), targets_graph.long()
        )

        # nuclei-level metrics
        targets_sup = batch["y"]["nuclei"]
        assert targets_sup is not None

        logits_nuclei = logits["nuclei"]
        logits_sup = logits_nuclei[batch["sup_mask"]].squeeze(-1)

        sup_size = targets_sup.numel()
        if sup_size == 0:  # empty supervision batch
            return None
        loss_sup = self.bce(logits_sup, targets_sup)
        self.log(
            "test/nuclei/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=sup_size,
        )
        self.test_nuclei_metrics.update(torch.sigmoid(logits_sup), targets_sup.long())

    def on_test_epoch_end(self) -> None:
        # compute and reset graph-level metrics
        graph_metrics = self.test_graph_metrics.compute()
        self.log_dict(graph_metrics, on_epoch=True, prog_bar=True)
        self.test_graph_metrics.reset()

        # compute and reset nuclei-level metrics
        nuclei_metrics = self.test_nuclei_metrics.compute()
        self.log_dict(nuclei_metrics, on_epoch=True, prog_bar=True)
        self.test_nuclei_metrics.reset()

    def predict_step(self, batch: PredictBatch) -> Outputs:
        return self(batch["slides"])

    def _get_optimizer_params(self) -> list[dict[str, Any]]:
        no_decay_params = [
            w
            for n, w in self.net.named_parameters()
            if w.requires_grad and (w.ndim <= 1 or ".rope." in n)
        ]
        decay_params = list(set(self.net.parameters()).difference(no_decay_params))
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
