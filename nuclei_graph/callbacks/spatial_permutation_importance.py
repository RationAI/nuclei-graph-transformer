import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from tqdm import tqdm

from nuclei_graph.nuclei_graph_typing import Batch


class SpatialPermutationImportanceCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.cached_batches: list[Batch] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return

        self.cached_batches.append(batch)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not self.cached_batches:
            return
        device = pl_module.device

        metric_collection = MetricCollection({
            "AUROC": BinaryAUROC(),
            "AUPRC": BinaryAveragePrecision(),
            "Accuracy": BinaryAccuracy(),
            "Precision": BinaryPrecision(),
            "Recall": BinaryRecall(),
            "Specificity": BinarySpecificity(),
        }).to(device)

        baseline_metrics = metric_collection.clone()
        perm_metrics = metric_collection.clone()

        for batch in tqdm(self.cached_batches, desc="Baseline Metrics"):
            seq_len = int(batch["seq_lens"].sum().item())
            sup_mask = batch["sup_mask"][:seq_len]

            nuclei_labels = batch["labels"]["nuclei"]
            if nuclei_labels is None:
                continue

            targets_sup = nuclei_labels[:seq_len][sup_mask]
            if targets_sup.numel() == 0:
                continue

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = pl_module(batch)["nuclei"]

            logits_sup = logits[sup_mask].squeeze(-1)
            baseline_metrics.update(logits_sup, targets_sup.long())

        base_scores = baseline_metrics.compute()

        for batch in tqdm(
            self.cached_batches, desc="Permuting Spatial Positions", leave=False
        ):
            seq_len = int(batch["seq_lens"].sum().item())
            sup_mask = batch["sup_mask"][:seq_len]

            nuclei_labels = batch["labels"]["nuclei"]
            if nuclei_labels is None:
                continue

            targets_sup = nuclei_labels[:seq_len][sup_mask]
            if targets_sup.numel() == 0:
                continue

            pos = batch["pos"]
            pos_perm = pos.clone()

            idx = torch.randperm(seq_len, device=device)
            pos_perm[:seq_len] = pos_perm[idx]

            permuted_batch = dict(batch)
            permuted_batch["pos"] = pos_perm

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = pl_module(permuted_batch)["nuclei"]

            logits_sup = logits[sup_mask].squeeze(-1)
            perm_metrics.update(logits_sup, targets_sup.long())

        perm_scores = perm_metrics.compute()

        metric_drops = {}
        for metric_name in base_scores.keys():
            drop = base_scores[metric_name].item() - perm_scores[metric_name].item()
            metric_drops[metric_name] = drop

        self._plot_and_log_importances(metric_drops)
        self.cached_batches.clear()

    def _plot_and_log_importances(self, metric_drops: dict[str, float]) -> None:
        names = list(metric_drops.keys())
        drops = list(metric_drops.values())

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(names, drops, color="royalblue", edgecolor="black")
        ax.set_title("Spatial Permutation Test")
        ax.set_ylabel("Metric Decrease")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=15))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "spatial_permutation_importance.png"
            fig.savefig(output_path, dpi=1200)

            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.log_artifact(str(output_path), run_id=active_run.info.run_id)

        plt.close(fig)