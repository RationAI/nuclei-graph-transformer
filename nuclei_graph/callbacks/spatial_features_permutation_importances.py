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


# Spatial feature vector layout (must match get_spatial_features exactly):
#
#   Index  Name
#   ─────  ────────────────────────────────────────────────────────
#   0      kNN dist k=1  (µm)
#   1      kNN dist k=3  (µm)
#   2      kNN dist k=5  (µm)
#   3      mean_dist     (µm)   ─┐ local spacing group
#   4      std_dist      (µm)   ─┘
#   5      angular_dispersion   (unitless [0, 1])
#   6      count_r20            (absolute count within 20 µm)
#   7      count_r50            (absolute count within 50 µm)
#   8      density_ratio        (count_r20 / count_r50)
#   9      degree               (Delaunay degree)


_SPATIAL_FEATURE_GROUPS: dict[str, slice | list[int]] = {
    # Individual kNN distances — tests whether each distance scale matters
    "kNN dist k=1": slice(0, 1),
    "kNN dist k=3": slice(1, 2),
    "kNN dist k=5": slice(2, 3),
    # Mean + std together — captures overall spacing distribution
    "Local spacing (mean+std)": slice(3, 5),
    # All five spacing features together — full local-spacing group
    "All kNN spacing": slice(0, 5),
    # Directional organisation of the neighbourhood
    "Angular dispersion": slice(5, 6),
    # Absolute density at two radii
    "Density @ 20 µm": slice(6, 7),
    "Density @ 50 µm": slice(7, 8),
    # Density profile shape (ratio, radius-invariant to absolute count)
    "Density ratio (20/50)": slice(8, 9),
    # Graph topology
    "Delaunay degree": slice(9, 10),
    # Compound groups — useful for understanding which scale matters most
    "All density features": slice(6, 9),  # count_r20, count_r50, ratio
    "All spatial features": slice(0, 10),  # sanity check: should reproduce full drop
}


class SpatialPermutationImportanceCallback(Callback):
    """Computes permutation feature importance for spatial (arrangement) features.

    Mirrors PermutationImportanceCallback but operates on the spatial feature
    vector produced by get_spatial_features() rather than EFD coefficients.

    At the end of the test epoch each named feature group is independently
    shuffled across nuclei, AUROC is recomputed, and the drop relative to the
    baseline is recorded as the importance score.

    Args:
        mlflow_run_id: Optional MLflow run ID for artifact logging. Falls back
                       to the currently active MLflow run when None.
    """

    def __init__(self, mlflow_run_id: str | None = None) -> None:
        super().__init__()
        self.mlflow_run_id = mlflow_run_id
        self.cached_batches: list[Batch] = []
        self.feature_groups = _SPATIAL_FEATURE_GROUPS

    # ------------------------------------------------------------------ #
    # Lightning hooks                                                       #
    # ------------------------------------------------------------------ #

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

        # ── baseline AUROC ────────────────────────────────────────────── #
        baseline_auroc = BinaryAUROC().to(device)
        for batch in tqdm(self.cached_batches, desc="Spatial PI — baseline"):
            logits_sup, targets_sup = self._forward(pl_module, batch, device)
            if logits_sup is None:
                continue
            baseline_auroc.update(logits_sup, targets_sup.long())

        base_score = baseline_auroc.compute().item()

        # ── per-group permutation ─────────────────────────────────────── #
        importances: dict[str, float] = {}
        for name, f_slice in tqdm(
            self.feature_groups.items(), desc="Spatial PI — groups"
        ):
            perm_auroc = BinaryAUROC().to(device)

            for batch in self.cached_batches:
                logits_sup, targets_sup = self._forward(
                    pl_module, batch, device, permute=f_slice
                )
                if logits_sup is None:
                    continue
                perm_auroc.update(logits_sup, targets_sup.long())

            drop = base_score - perm_auroc.compute().item()
            importances[name] = drop

        self._plot_and_log(importances, base_score)
        self.cached_batches.clear()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _forward(
        self,
        pl_module: LightningModule,
        batch: Batch,
        device: torch.device,
        permute: slice | list[int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run a forward pass, optionally permuting a feature slice."""
        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        nuclei_labels = batch["labels"]["nuclei"]
        if nuclei_labels is None:
            return None, None

        targets_sup = nuclei_labels[:seq_len][sup_mask]
        if targets_sup.numel() == 0:
            return None, None

        if permute is not None:
            x = batch["features"]
            if x is None:
                return None, None
            x_perm = x.clone()
            # Shuffle the chosen feature columns across the real (non-padded) nuclei
            idx = torch.randperm(seq_len, device=device)
            x_perm[:seq_len, permute] = x_perm[idx, permute]
            batch = dict(batch)
            batch["features"] = x_perm

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = pl_module(batch)["nuclei"]

        logits_sup = logits[sup_mask].squeeze(-1)
        return logits_sup, targets_sup

    def _plot_and_log(self, importances: dict[str, float], base_score: float) -> None:
        names = list(importances.keys())
        drops = list(importances.values())

        # Colour: positive drop = red (feature helps), negative = blue (noise helps)
        colours = ["coral" if d >= 0 else "steelblue" for d in drops]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(names, drops, color=colours, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"Spatial Feature Permutation Importance\n"
            f"(baseline AUROC = {base_score:.4f}; bars show AUROC drop)"
        )
        ax.set_ylabel("AUROC Decrease")
        ax.set_xlabel("Feature group")
        plt.xticks(rotation=45, ha="right")

        # Annotate each bar with its numeric value
        for bar, drop in zip(bars, drops, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{drop:+.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.tight_layout()

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "spatial_feature_importances.png"
            fig.savefig(output_path, dpi=300)

            target_run_id = self.mlflow_run_id
            if target_run_id is None:
                active_run = mlflow.active_run()
                if active_run is not None:
                    target_run_id = active_run.info.run_id

            if target_run_id is not None:
                client = MlflowClient()
                client.log_artifact(run_id=target_run_id, local_path=str(output_path))

        plt.close(fig)
