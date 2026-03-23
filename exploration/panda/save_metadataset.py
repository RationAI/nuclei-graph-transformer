"""This script generates a CSV exploration metadataset for the PANDA Challenge Dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import numpy as np
import pandas as pd
import ray
import tifffile
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


@ray.remote(num_cpus=1)
def validate_sample(slide_id: str, slides_dir: Path, annots_dir: Path) -> dict:
    slide_path = slides_dir / f"{slide_id}.tiff"
    mask_path = annots_dir / f"{slide_id}_mask.tiff"

    is_wsi_valid = False
    try:
        with OpenSlide(str(slide_path)) as wsi:
            _, _ = wsi.dimensions
            thumb = wsi.get_thumbnail(size=(512, 512)).convert("L")
            thumb_array = np.array(thumb)
            tissue_ratio = np.mean(thumb_array < np.percentile(thumb_array, 95))

            is_wsi_valid = tissue_ratio > 0.001

            if not is_wsi_valid:
                print(f"Low tissue ratio for {slide_path}: {tissue_ratio:.4f}")

    except Exception as e:
        print(f"Error for slide {slide_path}: {e}")
        is_wsi_valid = False

    annot_status = "missing"
    if mask_path.exists():
        try:
            mask = tifffile.imread(str(mask_path))
            if mask is not None and mask.size > 0 and np.max(mask) != 0:
                annot_status = "valid"
            else:
                print(f"Empty or invalid mask for {mask_path}")
                annot_status = "corrupted"
        except Exception as e:
            print(f"Error for mask {mask_path}: {e}")
            annot_status = "corrupted"

    return {
        "slide_id": slide_id,
        "is_wsi_valid": is_wsi_valid,
        "annot_status": annot_status,
    }


def get_dataframes(
    df_path: Path,
    slides_dir: Path,
    annots_dir: Path,
    properties_path: Path,
    max_concurrent: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(df_path)
    df.rename(columns={"image_id": "slide_id"}, inplace=True)

    validation_results = process_items(
        items=df["slide_id"],
        process_item=validate_sample,
        fn_kwargs={"slides_dir": slides_dir, "annots_dir": annots_dir},
        max_concurrent=max_concurrent,
    )

    valid_df = pd.DataFrame(validation_results)
    df = df.merge(valid_df, on="slide_id", how="left")

    df["slide_path"] = df["slide_id"].apply(lambda id: str(slides_dir / f"{id}.tiff"))
    df["is_annotation_corrupted"] = df["annot_status"] == "corrupted"
    df["annotation"] = df["annot_status"] != "missing"
    df["is_carcinoma"] = df["isup_grade"] > 0

    properties_df = pd.read_parquet(properties_path)
    properties_df["slide_id"] = properties_df["path"].apply(lambda p: Path(p).stem)

    final_df = df.merge(properties_df.drop(columns=["path"]), on="slide_id", how="left")

    summary_df = (
        final_df[final_df["is_wsi_valid"] & ~final_df["is_annotation_corrupted"]]
        .groupby(["data_provider", "is_carcinoma", "isup_grade"])
        .agg(Total_Slides=("slide_id", "count"), Annotations=("annotation", "sum"))
        .reset_index()
    )

    final_cols = [
        "slide_id",
        "slide_path",
        "id",
        "data_provider",
        "is_carcinoma",
        "annotation",
        "is_annotation_corrupted",
        "is_wsi_valid",
        "extent_x",
        "extent_y",
        "mpp_x",
        "mpp_y",
    ]

    return final_df[final_cols], summary_df


@with_cli_args(["+exploration=panda/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df, summary_df = get_dataframes(
        df_path=Path(config.train_csv),
        slides_dir=Path(config.train_images),
        annots_dir=Path(config.train_label_masks),
        properties_path=Path(config.slides_properties),
        max_concurrent=config.max_concurrent,
    )

    with TemporaryDirectory() as output_dir:
        df.to_csv(Path(output_dir, "slides_metadata.csv"), index=False)
        summary_df.to_csv(Path(output_dir, "summary.csv"), index=False)
        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="panda")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
