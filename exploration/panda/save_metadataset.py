"""This script generates a CSV metadataset file for the `PANDA Challenge` Dataset.

The metadata extracted for each slide relies on the original train.csv file:
    - `slide_id` (str): 32-character hex string identifier.
    - `data_provider` (str): 'radboud' or 'karolinska'.
    - `is_carcinoma` (bool): True if the slide contains carcinoma based on the provided ISUP grade, else False.
    - `annotation` (bool): True if the annotation exists.

The resulting CSV files along with their summaries are logged as artifacts to MLflow.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+exploration=panda/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_dir = Path(config.train_images)
    annots_dir = Path(config.train_label_masks)

    df = pd.read_csv(Path(config.train_csv))
    df.rename(columns={"image_id": "slide_id"}, inplace=True)

    def check_image(slide_id: str) -> bool:
        return (slides_dir / f"{slide_id}.tiff").exists()

    def check_mask(slide_id: str) -> bool:
        return (annots_dir / f"{slide_id}_mask.tiff").exists()

    exists_mask = df["slide_id"].apply(check_image)
    df = df[exists_mask].reset_index(drop=True)

    df["annotation"] = df["slide_id"].apply(check_mask)
    df["is_carcinoma"] = df["isup_grade"] > 0

    with TemporaryDirectory() as output_dir:
        cols = ["slide_id", "data_provider", "is_carcinoma", "annotation"]
        df[cols].to_csv(Path(output_dir, "slides_metadata.csv"), index=False)

        summary_df = (
            df.groupby(["data_provider", "is_carcinoma", "isup_grade"])
            .agg(Total_Slides=("slide_id", "count"), Annotations=("annotation", "sum"))
            .reset_index()
        )
        summary_df.to_csv(Path(output_dir, "summary.csv"), index=False)

        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df[cols], name="panda")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    main()