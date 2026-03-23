"""This script generates a CSV metadataset file for the PANDA Challenge Dataset."""

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
    nuclei_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(df_path)
    df.rename(columns={"image_id": "slide_id"}, inplace=True)
    df["slide_path"] = df["slide_id"].apply(lambda x: str(slides_dir / f"{x}.tiff"))

    def check_slide(slide_id: str) -> bool:
        return (slides_dir / f"{slide_id}.tiff").exists()

    exists_slide = df["slide_id"].apply(check_slide)
    df = df[exists_slide].reset_index(drop=True)

    def check_mask(slide_id: str) -> bool:
        return (annots_dir / f"{slide_id}_mask.tiff").exists()

    df["annotation"] = df["slide_id"].apply(check_mask)

    df["is_carcinoma"] = df["isup_grade"] > 0

    properties_df = pd.read_parquet(
        properties_path,
        columns=["id", "path", "extent_x", "extent_y", "mpp_x", "mpp_y"],
    )
    properties_df["slide_id"] = properties_df["path"].apply(lambda p: Path(p).stem)
    properties_df = properties_df.drop(columns=["path"])

    final_df = df.merge(
        properties_df, left_on="slide_id", right_on="slide_id", how="left"
    )

    def check_nuclei(id: str) -> bool:
        if pd.isna(id):
            return False
        return (nuclei_dir / f"slide_id={id}").exists()

    exists_nuclei = final_df["id"].apply(check_nuclei)
    final_df = final_df[exists_nuclei].reset_index(drop=True)

    summary_df = (
        final_df.groupby(["data_provider", "is_carcinoma", "isup_grade"])
        .agg(Total_Slides=("slide_id", "count"), Annotations=("annotation", "sum"))
        .reset_index()
    )

    final_cols = [
        "slide_id",  # 32-character hex string identifier for each slide
        "slide_path",
        "id",  # ID of the slide in the parquet dataset with segmented nuclei
        "data_provider",  # 'radboud' or 'karolinska'
        "is_carcinoma",  # True if the slide contains carcinoma based on the ISUP grade
        "annotation",  # True if the annotation exists
        "extent_x",
        "extent_y",
        "mpp_x",
        "mpp_y",
    ]
    final_df = final_df[final_cols]

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
        nuclei_dir=Path(config.nuclei_path),
    )

    with TemporaryDirectory() as output_dir:
        df.to_csv(Path(output_dir, "slides_metadata.csv"), index=False)
        summary_df.to_csv(Path(output_dir, "summary.csv"), index=False)
        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="panda")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    main()
