"""Computes tile-level classification metrics from pooled predictions.

Loads pooled tile predictions merged with their ground-truth labels and computes:

- Per-slide metrics (accuracy, precision, recall, specificity, negative
  predictive value), logged as a table.
- Global tile-level metrics, logged as scalar metrics plus the confusion matrix

Intended to be run after `thresholds.py` has selected a threshold and
`nuclei_to_tile_pooling.py` has produced the pooled tile predictions.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
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
    BinaryNegativePredictiveValue,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from postprocessing.tile.data_loading import load_tile_predictions


@with_cli_args(["+postprocessing=tile/metrics"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    merged_df = load_tile_predictions(config)
    target_col = config.label_column

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        slide_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(config.threshold),
                "precision": BinaryPrecision(config.threshold),
                "recall": BinaryRecall(config.threshold),
                "specificity": BinarySpecificity(config.threshold),
                "negative_predictive_value": BinaryNegativePredictiveValue(
                    config.threshold
                ),
            }
        )

        slide_results = []
        for slide_id, group in merged_df.groupby("slide_id"):
            preds_t = torch.tensor(group["tile_prediction"].values)
            targets_t = torch.tensor(group[target_col].values).long()

            computed = slide_metrics(preds_t, targets_t)
            res = {"slide_id": slide_id, **{k: float(v) for k, v in computed.items()}}
            slide_results.append(res)
            slide_metrics.reset()

        slide_results_df = pd.DataFrame(slide_results)

        logger.log_table(
            data={
                str(k): v for k, v in slide_results_df.to_dict(orient="list").items()
            },
            artifact_file=config.mlflow_artifact_path,
        )

        global_tile_metrics = MetricCollection(
            {
                "AUPRC": BinaryAveragePrecision(),
                "AUROC": BinaryAUROC(),
                "precision": BinaryPrecision(config.threshold),
                "recall": BinaryRecall(config.threshold),
                "accuracy": BinaryAccuracy(config.threshold),
                "specificity": BinarySpecificity(config.threshold),
                "negative_predictive_value": BinaryNegativePredictiveValue(
                    config.threshold
                ),
                "confusion_matrix": BinaryConfusionMatrix(config.threshold),
            },
            prefix="test_thresholded/",
        )

        preds_t = torch.tensor(merged_df["tile_prediction"].values)
        targets_t = torch.tensor(merged_df[target_col].values).long()
        computed_global_tiles = global_tile_metrics(preds_t, targets_t)

        numerical_metrics = {}
        for key, value in computed_global_tiles.items():
            if "confusion_matrix" in key:
                disp_cf = ConfusionMatrixDisplay(
                    confusion_matrix=value.cpu().numpy(),
                    display_labels=["Negative", "Positive"],
                )
                fig, ax = plt.subplots(figsize=(6, 5))
                disp_cf.plot(ax=ax, cmap="Blues", colorbar=False)
                plt.title("Global Tile-Level Confusion Matrix")
                plt.tight_layout()

                fig_path = tmp_path / "tile_confusion_matrix.png"
                fig.savefig(fig_path, dpi=600)
                logger.log_artifact(local_path=str(fig_path), artifact_path="plots")
                plt.close(fig)
            else:
                numerical_metrics[key] = float(value)

        if numerical_metrics:
            logger.log_metrics(numerical_metrics)


if __name__ == "__main__":
    main()
