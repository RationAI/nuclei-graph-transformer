import tempfile
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from mlflow.tracking import MlflowClient
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from nuclei_graph.nuclei_graph_typing import Batch


def plot_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    plot_label: str | None,
    to_pinpoint: list[tuple[float, float]],
    point_labels: list[str],
    point_colors: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    loc: str,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, label=plot_label)

    for i in range(len(to_pinpoint)):
        x, y = to_pinpoint[i]
        ax.scatter(x, y, color=point_colors[i], label=point_labels[i], zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=loc)
    ax.grid(True)
    fig.tight_layout()
    return fig


MAX_CURVE_POINTS = 2000


def _downsample_curve(*arrays: np.ndarray) -> list[np.ndarray]:
    """Downsamples curve arrays (via uniform index subsampling) for fast, light plotting."""
    n = len(arrays[0])
    if n <= MAX_CURVE_POINTS:
        return list(arrays)
    idx = np.unique(
        np.linspace(0, n - 1, MAX_CURVE_POINTS, dtype=np.int64, endpoint=True)
    )
    return [a[idx] for a in arrays]


def perform_roc(
    y_true: np.ndarray, y_pred: np.ndarray, title: str
) -> tuple[matplotlib.figure.Figure, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    idx = np.where(np.isclose(tpr, 1.0))[0]
    if len(idx) > 0:
        tpr_idx = idx[np.argmin(fpr[idx])]
        tpr_threshold = float(thresholds[tpr_idx])
        tpr_label = f"TPR Thresh = {tpr_threshold:.3f}"
    else:
        tpr_idx = 0
        tpr_threshold = np.nan
        tpr_label = "TPR Thresh = N/A"

    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    j_threshold = thresholds[j_idx]

    fpr_plot, tpr_plot = _downsample_curve(fpr, tpr)
    fig = plot_curve(
        fpr_plot,
        tpr_plot,
        f"AUC = {roc_auc:.3f}",
        [(fpr[tpr_idx], tpr[tpr_idx]), (fpr[j_idx], tpr[j_idx])],
        [tpr_label, f"J Thresh = {j_threshold:.3f}"],
        ["red", "green"],
        "False Positive Rate",
        "True Positive Rate",
        title,
        "lower right",
    )
    return fig, tpr_threshold, j_threshold


def perform_pr(
    y_true: np.ndarray, y_pred: np.ndarray, title: str
) -> tuple[matplotlib.figure.Figure, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    p = precision[:-1]
    r = recall[:-1]

    f1 = 2 * (p * r) / (p + r + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]

    recall_plot, precision_plot = _downsample_curve(recall, precision)
    fig = plot_curve(
        recall_plot,
        precision_plot,
        None,
        [(recall[best_idx], precision[best_idx])],
        [f"F1 Thresh = {best_threshold:.3f}"],
        ["red"],
        "Recall",
        "Precision",
        title,
        "lower left",
    )
    return fig, best_threshold


class BaseCurvesCallback(Callback):
    def __init__(self, mlflow_run_id: str | None = None) -> None:
        super().__init__()
        self.mlflow_run_id = mlflow_run_id

    def _log_and_clear_curves(
        self,
        preds_list: list[torch.Tensor],
        targets_list: list[torch.Tensor],
        level_name: str,
    ) -> None:
        """Computes, plots, and logs ROC and PR curves."""
        if not preds_list:
            return

        y_pred = torch.cat(preds_list).numpy()
        y_true = torch.cat(targets_list).numpy()

        title_prefix = "Graph-Level" if level_name == "graph" else "Nuclei-Level"

        fig_roc, roc_t, j_t = perform_roc(y_true, y_pred, f"{title_prefix} ROC")
        fig_pr, pr_t = perform_pr(y_true, y_pred, f"{title_prefix} PR Curve")

        mlflow_run_id = self.mlflow_run_id

        if mlflow_run_id is None:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow_run_id = active_run.info.run_id

        if mlflow_run_id is not None:
            client = MlflowClient()

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                roc_path = tmp_path / f"{level_name}_roc.png"
                pr_path = tmp_path / f"{level_name}_precision_recall.png"

                fig_roc.savefig(roc_path, dpi=200)
                fig_pr.savefig(pr_path, dpi=200)

                client.log_artifact(mlflow_run_id, str(roc_path), "curves")
                client.log_artifact(mlflow_run_id, str(pr_path), "curves")

            client.log_metric(
                mlflow_run_id, f"thresholds/{level_name}_tpr", float(roc_t)
            )
            client.log_metric(mlflow_run_id, f"thresholds/{level_name}_j", float(j_t))
            client.log_metric(mlflow_run_id, f"thresholds/{level_name}_f1", float(pr_t))

        plt.close(fig_roc)
        plt.close(fig_pr)

        preds_list.clear()
        targets_list.clear()


class CropCurvesCallback(BaseCurvesCallback):
    """Generates ROC and Precision-Recall curves for graph and nuclei-level validation set."""

    def __init__(self, mlflow_run_id: str | None = None) -> None:
        super().__init__(mlflow_run_id=mlflow_run_id)
        self.graph_preds: list[torch.Tensor] = []
        self.graph_targets: list[torch.Tensor] = []
        self.nuclei_preds: list[torch.Tensor] = []
        self.nuclei_targets: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking or outputs is None:
            return

        targets_graph = batch["labels"]["graph"]
        if targets_graph is not None:
            graph_outputs = outputs["graph"].view(-1)
            self.graph_preds.append(torch.sigmoid(graph_outputs).detach().cpu())
            self.graph_targets.append(targets_graph.view(-1).detach().cpu())

        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        targets_sup = batch["labels"]["nuclei"]
        if targets_sup is not None:
            targets_sup = targets_sup[:seq_len]
            if targets_sup.numel() > 0:
                nuclei_outputs = outputs["nuclei"][sup_mask].squeeze(-1)
                self.nuclei_preds.append(torch.sigmoid(nuclei_outputs).detach().cpu())
                self.nuclei_targets.append(targets_sup[sup_mask].detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        self._log_and_clear_curves(self.graph_preds, self.graph_targets, "val_graph")
        self._log_and_clear_curves(self.nuclei_preds, self.nuclei_targets, "val_nuclei")


class NucleiCurvesCallback(BaseCurvesCallback):
    """Generates ROC and Precision-Recall curves for nuclei-level validation set."""

    def __init__(self, mlflow_run_id: str | None = None) -> None:
        super().__init__(mlflow_run_id=mlflow_run_id)
        self.nuclei_preds: list[torch.Tensor] = []
        self.nuclei_targets: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking or outputs is None:
            return

        seq_len = int(batch["seq_lens"].sum().item())
        sup_mask = batch["sup_mask"][:seq_len]

        targets_sup = batch["labels"]["nuclei"]
        if targets_sup is not None:
            targets_sup = targets_sup[:seq_len]
            if targets_sup.numel() > 0:
                nuclei_outputs = outputs["nuclei"][sup_mask].squeeze(-1)
                self.nuclei_preds.append(torch.sigmoid(nuclei_outputs).detach().cpu())
                self.nuclei_targets.append(targets_sup[sup_mask].detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        self._log_and_clear_curves(self.nuclei_preds, self.nuclei_targets, "val_nuclei")
