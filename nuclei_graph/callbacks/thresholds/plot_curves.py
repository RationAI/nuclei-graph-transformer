import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from nuclei_graph.nuclei_graph_typing import Batch, Outputs


class BaseCurvesCallback(Callback):
    def _log_and_clear_curves(
        self, preds_list: list, targets_list: list, level_name: str
    ) -> None:
        """Computes, plots, logs, and clears ROC and PR curves for a given prediction level."""
        if not preds_list:
            return

        y_pred = torch.cat(preds_list).numpy()
        y_true = torch.cat(targets_list).numpy()

        title_prefix = "Slide-Level" if level_name == "graph" else "Nuclei-Level"

        fig_roc, roc_t, j_t = self._perform_roc(y_true, y_pred, f"{title_prefix} ROC")
        fig_pr, pr_t = self._perform_pr(y_true, y_pred, f"{title_prefix} PR Curve")

        mlflow.log_figure(fig_roc, f"plots/{level_name}_roc.png")
        mlflow.log_figure(fig_pr, f"plots/{level_name}_precision_recall.png")

        mlflow.log_metric(f"thresholds/{level_name}_tpr_threshold", float(roc_t))
        mlflow.log_metric(f"thresholds/{level_name}_j_threshold", float(j_t))
        mlflow.log_metric(f"thresholds/{level_name}_f1_threshold", float(pr_t))

        plt.close(fig_roc)
        plt.close(fig_pr)

        preds_list.clear()
        targets_list.clear()

    def _plot_curve(
        self,
        xs,
        ys,
        plot_label,
        to_pinpoint,
        point_labels,
        point_colors,
        xlabel,
        ylabel,
        title,
        loc,
    ):
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

    def _perform_roc(self, y_true, y_pred, title):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        idx = np.where(np.isclose(tpr, 1.0))[0]
        if len(idx) > 0:
            tpr_idx = idx[np.argmin(fpr[idx])]
            tpr_threshold = thresholds[tpr_idx]
            tpr_label = f"TPR Thresh = {tpr_threshold:.3f}"
        else:
            tpr_idx = 0
            tpr_threshold = np.nan
            tpr_label = "TPR Thresh = N/A"

        j_scores = tpr - fpr
        j_idx = np.argmax(j_scores)
        j_threshold = thresholds[j_idx]

        fig = self._plot_curve(
            fpr,
            tpr,
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

    def _perform_pr(self, y_true, y_pred, title):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

        p = precision[:-1]
        r = recall[:-1]

        f1 = 2 * (p * r) / (p + r + 1e-8)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]

        fig = self._plot_curve(
            recall,
            precision,
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


class MILCurvesCallback(BaseCurvesCallback):
    """Generates ROC and Precision-Recall curves for graph and nuclei-level validation set."""

    def __init__(self) -> None:
        super().__init__()
        self.graph_preds, self.graph_targets = [], []
        self.nuclei_preds, self.nuclei_targets = [], []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs | None,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return

        targets_graph = batch["y"]["graph"]
        if targets_graph is not None:
            graph_outputs = outputs["graph"].view(-1)
            self.graph_preds.append(torch.sigmoid(graph_outputs).detach().cpu())
            self.graph_targets.append(targets_graph.view(-1).detach().cpu())

        targets_sup = batch["y"]["nuclei"]
        if targets_sup is not None and targets_sup.numel() > 0:
            nuclei_outputs = outputs["nuclei"][batch["sup_mask"]].squeeze(-1)
            self.nuclei_preds.append(torch.sigmoid(nuclei_outputs).detach().cpu())
            self.nuclei_targets.append(targets_sup.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        if trainer.state.fn == "validate":
            self._log_and_clear_curves(
                self.graph_preds, self.graph_targets, "val_graph"
            )
            self._log_and_clear_curves(
                self.nuclei_preds, self.nuclei_targets, "val_nuclei"
            )
        else:
            self.graph_preds.clear()
            self.graph_targets.clear()
            self.nuclei_preds.clear()
            self.nuclei_targets.clear()


class WSLCurvesCallback(BaseCurvesCallback):
    """Generates ROC and Precision-Recall curves for nuclei-level validation set WSL model."""

    def __init__(self) -> None:
        super().__init__()
        self.nuclei_preds, self.nuclei_targets = [], []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs | None,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking or outputs is None:
            return

        targets_sup = batch["y"]["nuclei"]

        if targets_sup is not None and targets_sup.numel() > 0:
            nuclei_outputs = outputs["nuclei"][batch["sup_mask"]].squeeze(-1)
            self.nuclei_preds.append(torch.sigmoid(nuclei_outputs).detach().cpu())
            self.nuclei_targets.append(targets_sup.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        if trainer.state.fn == "validate":
            self._log_and_clear_curves(
                self.nuclei_preds, self.nuclei_targets, "val_nuclei"
            )
        else:
            self.nuclei_preds.clear()
            self.nuclei_targets.clear()
