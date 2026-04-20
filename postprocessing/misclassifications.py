"""Script for generating slide-level misclassification csv report based on model predictions and metadata."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


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


@with_cli_args(["+postprocessing=misclassifications"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata = pd.read_parquet(download_artifacts(config.metadata_uri))
    predictions_dir = Path(download_artifacts(config.predictions_uri))
    preds_df = get_predictions(metadata["slide_id"], predictions_dir)

    merged_df = pd.merge(
        preds_df,
        metadata[["slide_id", "is_carcinoma", "slide_path"]],
        on="slide_id",
        how="inner",
    )

    merged_df["predicted_class"] = (
        merged_df["graph_prediction"] >= config.threshold
    ).astype(bool)
    misclassif_df = merged_df[
        merged_df["predicted_class"] != merged_df["is_carcinoma"]
    ].copy()

    with TemporaryDirectory(dir=os.getcwd()) as output_dir:
        csv_path = Path(output_dir) / "misclassifications.csv"
        misclassif_df.to_csv(csv_path, index=False)
        logger.log_artifact(local_path=str(csv_path))


if __name__ == "__main__":
    main()
