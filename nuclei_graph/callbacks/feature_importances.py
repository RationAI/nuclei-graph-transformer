import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import torch
from lightning import Callback, LightningModule, Trainer
from mlflow.tracking import MlflowClient
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from nuclei_graph.nuclei_graph_typing import Batch


class PermutationImportanceCallback(Callback):
    """Computes permutation feature importance on the test set.

    At the end of the test epoch, each feature group is shuffled, the AUROC is recomputed,
    and the drop relative to the baseline AUROC is used as the feature importance score.

    Args:
        efd_order: Number of EFD orders in the feature vector.
        feature_group_size: Number of EFD orders per feature group.
        run_id: Optional MLflow run ID for artifact logging.
    """

    def __init__(
        self,
        efd_order: int,
        feature_group_size: int = 3,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.efd_order = efd_order
        self.mlflow_run_id = mlflow_run_id
        self.cached_batches: list[Batch] = []
        self.feature_groups = self._build_feature_groups(efd_order, feature_group_size)

    def _build_feature_groups(
        self, efd_order: int, feature_group_size: int
    ) -> dict[str, slice]:
        groups = {}

        for start_order in range(0, efd_order, feature_group_size):
            end_order = min(start_order + feature_group_size - 1, efd_order - 1)

            start_idx = start_order * 4
            end_idx = (end_order + 1) * 4

            if start_order == end_order:
                label = f"EFD Order {start_order}"
            else:
                label = f"EFD Orders {start_order}-{end_order}"

            groups[label] = slice(start_idx, end_idx)

        groups["Scale"] = slice(efd_order * 4, efd_order * 4 + 1)
        groups["Rotation"] = slice(efd_order * 4 + 1, efd_order * 4 + 3)

        return groups

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

        baseline_auroc = BinaryAUROC().to(device)
        for batch in tqdm(self.cached_batches, desc="Baseline AUROC"):
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
            baseline_auroc.update(logits_sup, targets_sup.long())

        base_score = baseline_auroc.compute().item()

        importances = {}
        for name, f_slice in tqdm(self.feature_groups.items(), desc="Feature groups"):
            perm_auroc = BinaryAUROC().to(device)

            for batch in tqdm(
                self.cached_batches, desc=f"Permuting {name}", leave=False
            ):
                seq_len = int(batch["seq_lens"].sum().item())
                sup_mask = batch["sup_mask"][:seq_len]

                nuclei_labels = batch["labels"]["nuclei"]
                if nuclei_labels is None:
                    continue

                targets_sup = nuclei_labels[:seq_len][sup_mask]
                if targets_sup.numel() == 0:
                    continue

                x = batch["features"]
                x_perm = x.clone()

                idx = torch.randperm(seq_len, device=device)
                x_perm[:seq_len, f_slice] = x_perm[idx, f_slice]

                permuted_batch = dict(batch)
                permuted_batch["features"] = x_perm

                with torch.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    logits = pl_module(permuted_batch)["nuclei"]

                logits_sup = logits[sup_mask].squeeze(-1)
                perm_auroc.update(logits_sup, targets_sup.long())

            drop = base_score - perm_auroc.compute().item()
            importances[name] = drop

        self._plot_and_log_importances(importances)
        self.cached_batches.clear()

    def _plot_and_log_importances(self, importances: dict[str, float]) -> None:
        names = list(importances.keys())
        drops = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, drops, color="coral", edgecolor="black")
        ax.set_title("Permutation Feature Importance (Drop in AUROC)")
        ax.set_ylabel("AUROC Decrease")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "feature_importances.png"
            fig.savefig(output_path, dpi=1200)

            target_run_id = self.mlflow_run_id

            if target_run_id is None:
                active_run = mlflow.active_run()
                if active_run is not None:
                    target_run_id = active_run.info.run_id

            if target_run_id is not None:
                client = MlflowClient()
                client.log_artifact(run_id=target_run_id, local_path=str(output_path))

        plt.close(fig)
