"""Script for computing dataset nuclei-level evaluation metrics from model predictions."""

from pathlib import Path

import hydra
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


def get_predictions(slide_ids: pd.Series, predictions_dir: Path) -> pd.DataFrame:
    all_preds = []
    for slide_id in slide_ids.unique():
        parquet_path = predictions_dir / f"{slide_id}.parquet"
        slide_pred_df = pd.read_parquet(parquet_path, columns=["id", "prediction"])
        slide_pred_df["slide_id"] = slide_id
        all_preds.append(slide_pred_df)
    return pd.concat(all_preds, ignore_index=True)


@with_cli_args(["+postprocessing/dataset_metrics=nuclei_level"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    predictions_dir = Path(download_artifacts(config.predictions_uri))
    metadata_df = pd.read_parquet(download_artifacts(config.metadata_uri))
    supervision_df = pd.read_parquet(config.supervision_dir)

    preds_df = get_predictions(metadata_df["slide_id"], predictions_dir)
    merged_df = pd.merge(preds_df, supervision_df, on=["slide_id", "id"], how="left")
    merged_df[config.label_column] = (
        merged_df[config.label_column].fillna(0).astype(int)
    )

    preds_t = torch.tensor(merged_df["prediction"].values)
    targets_t = torch.tensor(merged_df[config.label_column].values).long()

    metrics = MetricCollection(
        metrics={
            "AUPRC": BinaryAveragePrecision(),
            "AUROC": BinaryAUROC(),
            "precision": BinaryPrecision(config.threshold),
            "recall": BinaryRecall(config.threshold),
            "accuracy": BinaryAccuracy(config.threshold),
            "specificity": BinarySpecificity(config.threshold),
        },
        prefix="prediction/",
    )
    computed = metrics(preds_t, targets_t)
    logger.log_metrics({k: float(v) for k, v in computed.items()})


if __name__ == "__main__":
    main()
