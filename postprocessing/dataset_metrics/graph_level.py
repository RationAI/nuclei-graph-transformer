"""Script for computing dataset graph-level evaluation metrics from model predictions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


def get_predictions(slide_ids: pd.Series, predictions_dir: Path) -> pd.DataFrame:
    slide_preds = []
    for slide_id in slide_ids:
        parquet_path = predictions_dir / f"{slide_id}.parquet"
        graph_preds = pd.read_parquet(parquet_path, columns=["graph_prediction"])
        graph_pred = float(
            graph_preds.iloc[0]["graph_prediction"]
        )  # all nuclei have the same graph prediction
        slide_preds.append({"slide_id": slide_id, "graph_prediction": graph_pred})

    return pd.DataFrame(slide_preds)


def log_metrics(
    logger: MLFlowLogger, computed_metrics: dict[str, torch.Tensor]
) -> None:
    numerical_metrics = {}
    for key, value in computed_metrics.items():
        metric_name = key.split("/")[-1]
        if "confusion_matrix" in metric_name:
            cm_array = value.numpy()
            disp_cf = ConfusionMatrixDisplay(
                confusion_matrix=cm_array, display_labels=["Negative", "Positive"]
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            disp_cf.plot(ax=ax, cmap="Blues", colorbar=False)
            plt.title("Slide-Level Confusion Matrix")
            plt.tight_layout()

            with TemporaryDirectory() as output_dir:
                fig_path = Path(output_dir) / "confusion_matrix.png"
                fig.savefig(fig_path)
                logger.log_artifact(local_path=str(fig_path))

            plt.close(fig)
        else:
            numerical_metrics[key] = float(value)

    if numerical_metrics:
        logger.log_metrics(numerical_metrics)


@with_cli_args(["+postprocessing/dataset_metrics=graph_level"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    predictions_dir = Path(download_artifacts(config.predictions_uri))
    metadata = pd.read_parquet(download_artifacts(config.metadata_uri))
    preds_df = get_predictions(metadata["slide_id"], predictions_dir)

    merged_df = pd.merge(
        preds_df, metadata[["slide_id", "is_carcinoma"]], on="slide_id", how="inner"
    )

    preds_t = torch.tensor(merged_df["graph_prediction"].values)
    targets_t = torch.tensor(merged_df["is_carcinoma"].values).long()

    metrics = MetricCollection(
        metrics={
            "AUPRC": BinaryAveragePrecision(),
            "AUROC": BinaryAUROC(),
            "precision": BinaryPrecision(config.threshold),
            "recall": BinaryRecall(config.threshold),
            "accuracy": BinaryAccuracy(config.threshold),
            "specificity": BinarySpecificity(config.threshold),
            "confusion_matrix": BinaryConfusionMatrix(config.threshold),
        },
        prefix="prediction/graph/",
    )
    log_metrics(logger, metrics(preds_t, targets_t))


if __name__ == "__main__":
    main()
