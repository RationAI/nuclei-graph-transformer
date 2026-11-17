from collections.abc import Mapping
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import SequentialSampler
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from nuclei_graph.data.datamodules.base import PROVIDER_COL
from nuclei_graph.nuclei_graph_typing import Batch


class ProviderGraphMetricsCallback(Callback):
    """Validation slide AUROC and AUPRC of each provider of a multi-provider dataset.

    Logs `validation/graph/<provider>/{AUROC,AUPRC}` for every provider of the
    validation slides (the `provider` column of the datamodule's validation table).
    The validation loader yields one slide per batch, in table order, so the batch
    index is the slide's row. Every validation slide is counted (as with
    `graph_metrics_all_slides`).
    """

    def __init__(self) -> None:
        self.scores: dict[str, list[torch.Tensor]] = {}
        self.targets: dict[str, list[torch.Tensor]] = {}

    def _providers(self, trainer: Trainer) -> list[str] | None:
        dataset = getattr(trainer.datamodule, "validation_dataset", None)  # type: ignore[attr-defined]
        if dataset is None or PROVIDER_COL not in dataset.metadata:
            return None
        return dataset.metadata[PROVIDER_COL].astype(str).tolist()

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.scores, self.targets = {}, {}
        loader = trainer.val_dataloaders
        assert loader is None or isinstance(loader.sampler, SequentialSampler)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        providers = self._providers(trainer)
        if providers is None or not isinstance(outputs, Mapping):
            return
        assert len(batch["seq_lens"]) == 1, "one slide per validation batch"
        provider = providers[batch_idx]
        target = batch["labels"]["graph"]
        assert target is not None
        self.scores.setdefault(provider, []).append(
            torch.sigmoid(outputs["graph"].detach().view(-1).float()).cpu()
        )
        self.targets.setdefault(provider, []).append(target.view(-1).long().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics: dict[str, Any] = {}
        for provider, scores in self.scores.items():
            s = torch.cat(scores)
            t = torch.cat(self.targets[provider])
            prefix = f"validation/graph/{provider}"
            metrics[f"{prefix}/n_slides"] = float(len(t))
            if t.min() == t.max():  # one class only (e.g. a truncated dry run)
                continue
            metrics[f"{prefix}/AUROC"] = binary_auroc(s, t)
            metrics[f"{prefix}/AUPRC"] = binary_average_precision(s, t)
        if metrics:
            pl_module.log_dict(metrics, on_epoch=True)
