from typing import Any

import torch
import torch.nn.functional as F
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

from nuclei_graph.nuclei_graph_typing import Batch, Outputs


class CropModelMetaArch(LightningModule):
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

    def __init__(
        self,
        lr: float,
        warmup_epochs: int,
        net: nn.Module,
        neg_instance_weight: float = 0.0,
        pseudo_pos_weight: float = 0.0,
        selection_metric: str = "loss",
    ) -> None:
        super().__init__()
        assert selection_metric in {"loss", "AUROC"}, selection_metric
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.net = net
        # λ: for slides labelled 0, λ * the mean BCE of all their nucleus logits
        # against 0 is added to the slide's loss (no nucleus labels are used)
        self.neg_instance_weight = neg_instance_weight
        # μ: μ * the mean BCE of a slide's pseudo-positive nucleus logits against 1 is
        # added to its loss (pseudo-labels from a trained model, see postprocessing)
        self.pseudo_pos_weight = pseudo_pos_weight
        # validation graph metric ("loss" or "AUROC") whose best epoch is logged as best/
        self.selection_metric = selection_metric
        self.bce = nn.BCEWithLogitsLoss()

        self.val_graph_metrics = self._create_metrics("validation/graph/")
        self.test_graph_metrics = self._create_metrics("test/graph/")

        self.val_nuclei_metrics = self._create_metrics("validation/nuclei/")
        self.test_nuclei_metrics = self._create_metrics("test/nuclei/")

        self.best_val_graph_loss = float("inf")
        self.best_val_graph_auroc = float("-inf")
        self.best_val_graph_metrics: dict[str, Tensor] = {}
        self.val_step_graph_losses: list[Tensor] = []
        self.val_step_graph_sizes: list[int] = []

    def forward(self, inputs: Batch) -> Outputs:
        return self.net(
            x=inputs["features"],
            pos=inputs["pos"],
            block_mask=inputs["block_mask"],
            seq_lens=inputs["seq_lens"],
            neighbor_idx=inputs["global_neighbor_idx"],
            bboxes=inputs.get("bboxes"),
        )

    def training_step(self, batch: Batch) -> Tensor:
        targets_graph = batch["labels"]["graph"]
        assert targets_graph is not None
        targets_graph = targets_graph.view(-1)

        outputs = self(batch)
        logits_graph = outputs["graph"].view(-1)
        loss_graph = self.bce(logits_graph, targets_graph)

        self.log(
            "train/graph/loss",
            loss_graph,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets_graph.size(0),
        )
        if getattr(self.net, "pooling", None) == "lse":
            self.log("train/lse_r", self.net.lse_r, on_step=True, on_epoch=False)

        loss = loss_graph
        seq_lens = batch["seq_lens"].tolist()
        logits_nuclei = outputs["nuclei"].view(-1)

        if self.neg_instance_weight > 0.0:
            loss_neg = self.neg_instance_loss(logits_nuclei, seq_lens, targets_graph)
            self.log(
                "train/nuclei/neg_instance_loss",
                loss_neg,
                on_step=True,
                on_epoch=True,
                batch_size=targets_graph.size(0),
            )
            loss = loss + self.neg_instance_weight * loss_neg

        if self.pseudo_pos_weight > 0.0:
            pseudo = batch["labels"].get("pseudo")
            assert pseudo is not None, "pseudo_pos_weight > 0 requires pseudo-labels."
            loss_pseudo = self.pseudo_pos_loss(logits_nuclei, seq_lens, pseudo)
            self.log(
                "train/nuclei/pseudo_pos_loss",
                loss_pseudo,
                on_step=True,
                on_epoch=True,
                batch_size=targets_graph.size(0),
            )
            loss = loss + self.pseudo_pos_weight * loss_pseudo

        return loss

    def neg_instance_loss(
        self, logits_nuclei: Tensor, seq_lens: list[int], targets_graph: Tensor
    ) -> Tensor:
        """Per-slide mean BCE of nucleus logits against 0 on negative slides, batch-averaged."""
        losses = [
            F.softplus(_x).mean() if y == 0 else _x.new_zeros(())
            for _x, y in zip(
                torch.split(logits_nuclei, seq_lens), targets_graph, strict=True
            )
        ]
        return torch.stack(losses).mean()

    def pseudo_pos_loss(
        self, logits_nuclei: Tensor, seq_lens: list[int], pseudo: Tensor
    ) -> Tensor:
        """Per-slide mean BCE of pseudo-positive nucleus logits against 1, batch-averaged."""
        pseudo = pseudo[: logits_nuclei.shape[0]].bool()
        losses = [
            F.softplus(-_x[_p]).mean() if _p.any() else _x.new_zeros(())
            for _x, _p in zip(
                torch.split(logits_nuclei, seq_lens),
                torch.split(pseudo, seq_lens),
                strict=True,
            )
        ]
        return torch.stack(losses).mean()

    def validation_step(self, batch: Batch) -> Outputs | None:
        logits = self(batch)

        # graph-level metrics
        targets_graph = batch["labels"]["graph"]
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
        if batch["sup_mask"].any():
            self.val_graph_metrics.update(
                torch.sigmoid(logits_graph), targets_graph.long()
            )

        batch_size = targets_graph.size(0)
        self.val_step_graph_losses.append(loss_graph.detach() * batch_size)
        self.val_step_graph_sizes.append(batch_size)

        # nuclei-level metrics
        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        targets_sup = batch["labels"]["nuclei"]
        assert targets_sup is not None
        targets_sup = targets_sup[:seq_len][sup_mask]

        logits_sup = logits["nuclei"][sup_mask].squeeze(-1)

        loss_sup = self.bce(logits_sup, targets_sup)
        self.log(
            "validation/nuclei/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets_sup.numel(),
        )
        if logits_sup.numel() > 0:
            self.val_nuclei_metrics.update(
                torch.sigmoid(logits_sup), targets_sup.long()
            )
        return logits

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

        # the sanity check sees only a few slides
        if self.trainer.sanity_checking:
            return

        val_auroc = float(graph_metrics["validation/graph/AUROC"])
        improved = (
            val_loss < self.best_val_graph_loss
            if self.selection_metric == "loss"
            else val_auroc > self.best_val_graph_auroc
        )
        self.best_val_graph_loss = min(self.best_val_graph_loss, val_loss)
        self.best_val_graph_auroc = max(self.best_val_graph_auroc, val_auroc)

        if improved:
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
        targets_graph = batch["labels"]["graph"]
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
        if batch["sup_mask"].any():
            self.test_graph_metrics.update(
                torch.sigmoid(logits_graph), targets_graph.long()
            )

        # nuclei-level metrics
        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        targets_sup = batch["labels"]["nuclei"]
        assert targets_sup is not None
        targets_sup = targets_sup[:seq_len][sup_mask]

        logits_nuclei = logits["nuclei"]
        logits_sup = logits_nuclei[sup_mask].squeeze(-1)

        loss_sup = self.bce(logits_sup, targets_sup)
        self.log(
            "test/nuclei/loss",
            loss_sup,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets_sup.numel(),
        )
        if logits_sup.numel() > 0:
            self.test_nuclei_metrics.update(
                torch.sigmoid(logits_sup), targets_sup.long()
            )

    def on_test_epoch_end(self) -> None:
        # compute and reset graph-level metrics
        graph_metrics = self.test_graph_metrics.compute()
        self.log_dict(graph_metrics, on_epoch=True, prog_bar=True)
        self.test_graph_metrics.reset()

        # compute and reset nuclei-level metrics
        nuclei_metrics = self.test_nuclei_metrics.compute()
        self.log_dict(nuclei_metrics, on_epoch=True, prog_bar=True)
        self.test_nuclei_metrics.reset()

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
