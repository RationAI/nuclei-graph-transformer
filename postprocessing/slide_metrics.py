"""Script for computing nuclei-level evaluation metrics per slide from model predictions."""

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
    BinaryNegativePredictiveValue,
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


@with_cli_args(["+postprocessing=slide_metrics"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    predictions_dir = Path(download_artifacts(config.predictions_uri))
    supervision_df = pd.read_parquet(download_artifacts(config.supervision_uri))

    preds_df = get_predictions(supervision_df["slide_id"], predictions_dir)
    merged_df = pd.merge(preds_df, supervision_df, on=["slide_id", "id"], how="inner")

    metrics = MetricCollection(
        metrics={
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
        preds_t = torch.tensor(group["prediction"].values)
        targets_t = torch.tensor(group[config.label_column].values).long()

        computed = metrics(preds_t, targets_t)

        res = {"slide_id": slide_id, **{k: float(v) for k, v in computed.items()}}
        slide_results.append(res)

        metrics.reset()

    results_df = pd.DataFrame(slide_results)

    logger.log_table(
        data={str(k): v for k, v in results_df.to_dict(orient="list").items()},
        artifact_file=config.mlflow_artifact_path,
    )


if __name__ == "__main__":
    main()
