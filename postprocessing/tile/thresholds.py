from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.mlflow_utils import setup_mlflow
from postprocessing.tile.data_loading import load_tile_predictions_with_labels


def plot_roc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[plt.Figure, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    idx = np.where(np.isclose(tpr, 1.0))[0]
    if len(idx) > 0:
        tpr_idx = idx[np.argmin(fpr[idx])]
        tpr_threshold = float(thresholds[tpr_idx])
    else:
        tpr_idx = 0
        tpr_threshold = float("nan")

    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    j_threshold = float(thresholds[j_idx])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.scatter(
        fpr[tpr_idx],
        tpr[tpr_idx],
        color="red",
        label=f"TPR Thresh = {tpr_threshold:.3f}",
    )
    ax.scatter(
        fpr[j_idx], tpr[j_idx], color="green", label=f"J Thresh = {j_threshold:.3f}"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Tile-Level ROC")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    return fig, tpr_threshold, j_threshold


def plot_pr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[plt.Figure, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    p, r = precision[:-1], recall[:-1]
    f1 = 2 * (p * r) / (p + r + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = float(thresholds[best_idx])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision)
    ax.scatter(
        recall[best_idx],
        precision[best_idx],
        color="red",
        label=f"F1 Thresh = {best_threshold:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Tile-Level PR Curve")
    ax.legend(loc="lower left")
    ax.grid(True)
    fig.tight_layout()

    return fig, best_threshold


@with_cli_args(["+postprocessing=tile/thresholds"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.get("mlflow_parent_run_id"))

    merged_df = load_tile_predictions_with_labels(config)
    y_true = merged_df[config.label_column].to_numpy()
    y_pred = merged_df["tile_prediction"].to_numpy()

    fig_roc, tpr_t, j_t = plot_roc(y_true, y_pred)
    fig_pr, f1_t = plot_pr(y_true, y_pred)

    client, mlflow_run_id = setup_mlflow(config)
    assert mlflow_run_id is not None

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        roc_path = tmp_path / "tile_roc.png"
        pr_path = tmp_path / "tile_precision_recall.png"
        fig_roc.savefig(roc_path, dpi=1200)
        fig_pr.savefig(pr_path, dpi=1200)
        client.log_artifact(mlflow_run_id, str(roc_path), "curves")
        client.log_artifact(mlflow_run_id, str(pr_path), "curves")

    client.log_metric(mlflow_run_id, "thresholds/tile_tpr", tpr_t)
    client.log_metric(mlflow_run_id, "thresholds/tile_j", j_t)
    client.log_metric(mlflow_run_id, "thresholds/tile_f1", f1_t)

    plt.close(fig_roc)
    plt.close(fig_pr)


if __name__ == "__main__":
    main()
