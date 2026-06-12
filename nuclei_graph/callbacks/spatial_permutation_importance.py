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
    BinaryPrecisionRecallCurve,
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

        metric_collection = MetricCollection(
            {
                "AUROC": BinaryAUROC(),
                "AUPRC": BinaryAveragePrecision(),
                "Accuracy": BinaryAccuracy(),
                "Precision": BinaryPrecision(),
                "Recall": BinaryRecall(),
                "Specificity": BinarySpecificity(),
            }
        ).to(device)

        pr_curve = BinaryPrecisionRecallCurve().to(device)

        base_logits, base_targets = [], []
        perm_logits, perm_targets = [], []

        baseline_metrics = metric_collection.clone()
        perm_metrics = metric_collection.clone()

        for batch in tqdm(self.cached_batches, desc="Baseline Metrics"):
            seq_len = int(batch["seq_lens"].sum().item())
            sup_mask = batch["sup_mask"][:seq_len]
            targets = batch["labels"]["nuclei"][:seq_len][sup_mask].long()

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = pl_module(batch)["nuclei"][sup_mask].squeeze(-1)

            baseline_metrics.update(logits, targets)
            base_logits.append(logits.sigmoid())
            base_targets.append(targets)

        base_scores = baseline_metrics.compute()
        base_logits_t, base_targets_t = torch.cat(base_logits), torch.cat(base_targets)

        for batch in tqdm(
            self.cached_batches, desc="Permuting Spatial Positions", leave=False
        ):
            seq_len = int(batch["seq_lens"].sum().item())
            sup_mask = batch["sup_mask"][:seq_len]
            targets = batch["labels"]["nuclei"][:seq_len][sup_mask].long()

            pos_perm = batch["pos"].clone()
            idx = torch.randperm(seq_len, device=device)
            pos_perm[:seq_len] = pos_perm[idx]

            permuted_batch = dict(batch)
            permuted_batch["pos"] = pos_perm

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = pl_module(permuted_batch)["nuclei"][sup_mask].squeeze(-1)

            perm_metrics.update(logits, targets)
            perm_logits.append(logits.sigmoid())
            perm_targets.append(targets)

        perm_scores = perm_metrics.compute()
        perm_logits_t, perm_targets_t = torch.cat(perm_logits), torch.cat(perm_targets)

        base_prec, base_rec, _ = pr_curve(base_logits_t, base_targets_t)
        perm_prec, perm_rec, _ = pr_curve(perm_logits_t, perm_targets_t)

        metric_drops = {
            k: base_scores[k].item() - perm_scores[k].item() for k in base_scores
        }

        self._plot_and_log_results(
            metric_drops,
            base_prec.cpu(),
            base_rec.cpu(),
            perm_prec.cpu(),
            perm_rec.cpu(),
        )

        self.cached_batches.clear()

    def _plot_and_log_results(
        self,
        metric_drops: dict[str, float],
        b_p: torch.Tensor,
        b_r: torch.Tensor,
        p_p: torch.Tensor,
        p_r: torch.Tensor,
    ) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            out_path = Path(output_dir)

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.bar(
                list(metric_drops.keys()),
                list(metric_drops.values()),
                color="royalblue",
                edgecolor="black",
            )
            ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax1.set_title("Spatial Permutation Importance")
            ax1.set_ylabel("Metric Decrease")
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=15))
            ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig1.savefig(out_path / "importance.png", dpi=300)

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.plot(b_r, b_p, label="Baseline (w/ RoPE)", color="blue")
            ax2.plot(p_r, p_p, label="Permuted (No RoPE)", color="red", linestyle="--")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Precision-Recall Curve Comparison")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            fig2.savefig(out_path / "pr_curve.png", dpi=300)

            if active_run := mlflow.active_run():
                mlflow.log_artifact(
                    str(out_path / "importance.png"), run_id=active_run.info.run_id
                )
                mlflow.log_artifact(
                    str(out_path / "pr_curve.png"), run_id=active_run.info.run_id
                )

        plt.close("all")
