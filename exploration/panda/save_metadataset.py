"""This script generates a CSV metadataset file for the `PANDA Challenge` Dataset.

The metadata extracted for each slide are:
    - `slide_id` (str): 32-character hex string identifier for each slide.
    - `slide_path` (str): The path to the slide image.
    - `id` (str): ID of the slide in the parquet dataset with segmented nuclei.
    - `data_provider` (str): 'radboud' or 'karolinska'.
    - `is_carcinoma` (bool): True if the slide contains carcinoma based on the provided ISUP grade, else False.
    - `annotation` (bool): True if the annotation exists.
    - `extent_x` (float): The width of the slide.
    - `extent_y` (float): The height of the slide.
    - `mpp_x` (float): Microns per pixel in the x direction.
    - `mpp_y` (float): Microns per pixel in the y direction.

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


def get_dataframes(
    df_path: Path,
    slides_dir: Path,
    annots_dir: Path,
    properties_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(df_path)
    df.rename(columns={"image_id": "slide_id"}, inplace=True)
    df["slide_path"] = df["slide_id"].apply(lambda x: str(slides_dir / f"{x}.tiff"))

    def check_slide(slide_id: str) -> bool:
        return (slides_dir / f"{slide_id}.tiff").exists()

    def check_mask(slide_id: str) -> bool:
        return (annots_dir / f"{slide_id}_mask.tiff").exists()

    exists_mask = df["slide_id"].apply(check_slide)
    df = df[exists_mask].reset_index(drop=True)
    df["annotation"] = df["slide_id"].apply(check_mask)
    df["is_carcinoma"] = df["isup_grade"] > 0

    summary_df = (
        df.groupby(["data_provider", "is_carcinoma", "isup_grade"])
        .agg(Total_Slides=("slide_id", "count"), Annotations=("annotation", "sum"))
        .reset_index()
    )

    df = df[["slide_id", "data_provider", "is_carcinoma", "annotation", "slide_path"]]

    properties_df = pd.read_parquet(
        properties_path,
        columns=["id", "path", "extent_x", "extent_y", "mpp_x", "mpp_y"],
    )
    properties_df["slide_id"] = properties_df["path"].apply(lambda p: Path(p).stem)
    properties_df = properties_df.drop(columns=["path"])

    final_df = df.merge(
        properties_df, left_on="slide_id", right_on="slide_id", how="left"
    )
    return final_df, summary_df


@with_cli_args(["+exploration=panda/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df, summary_df = get_dataframes(
        df_path=Path(config.train_csv),
        slides_dir=Path(config.train_images),
        annots_dir=Path(config.train_label_masks),
        properties_path=Path(config.slides_properties),
    )

    with TemporaryDirectory() as output_dir:
        df.to_csv(Path(output_dir, "slides_metadata.csv"), index=False)
        summary_df.to_csv(Path(output_dir, "summary.csv"), index=False)
        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="panda")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    main()
