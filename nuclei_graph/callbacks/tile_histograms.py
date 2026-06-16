import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.nuclei_graph_typing import Batch


class TileHistogramsCallback(Callback):
    def __init__(self) -> None:
        """This callback creates prediction histograms for negative and positive tiles."""
        super().__init__()
        self.all_preds: list[np.ndarray] = []
        self.all_labels: list[np.ndarray] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = outputs["graph"] if isinstance(outputs, dict) else outputs
        
        targets = batch["labels"]["graph"] if isinstance(batch, dict) else batch[1]

        if logits is None or targets is None:
            return

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        labels = targets.detach().cpu().numpy().flatten()

        self.all_preds.append(probs)
        self.all_labels.append(labels)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.all_preds:
            return
            
        assert isinstance(trainer.logger, MLFlowLogger)

        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)

        pos_preds = preds[labels == 1]
        neg_preds = preds[labels == 0]

        with tempfile.TemporaryDirectory() as output_dir:
            out_path = Path(output_dir)

            _, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 6))

            ax_pos.hist(pos_preds, bins=20, range=(0, 1), color="green", alpha=0.7)
            ax_pos.set_title("Positive Tiles")
            ax_pos.set_xlabel("Predicted Probability")
            ax_pos.set_ylabel("Count")

            ax_neg.hist(neg_preds, bins=20, range=(0, 1), color="red", alpha=0.7)
            ax_neg.set_title("Negative Tiles")
            ax_neg.set_xlabel("Predicted Probability")
            ax_neg.set_ylabel("Count")

            plt.suptitle("Predicted Probability Histograms by Tile Class")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            
            plot_file = out_path / "tile_histograms.png" 
            plt.savefig(plot_file, dpi=300)

            if active_run := mlflow.active_run():
                mlflow.log_artifact(
                    str(out_path / "tile_histograms.png"), run_id=active_run.info.run_id
                )

            plt.close("all")

        self.all_preds.clear()
        self.all_labels.clear()