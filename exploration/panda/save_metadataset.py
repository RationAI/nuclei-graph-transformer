"""This script generates a CSV exploration metadataset for the PANDA Challenge Dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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


def extract_properties(slide_path: Path) -> dict[str, Any]:
    with OpenSlide(str(slide_path)) as slide:
        width, height = slide.dimensions
        mpp_x, mpp_y = slide.slide_resolution(level=0)
        return {
            "extent_x": width,
            "extent_y": height,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
        }


@ray.remote(num_cpus=1)
def validate_sample(
    slide_id: str, slides_dir: Path, annots_dir: Path
) -> dict[str, Any]:
    slide_path = slides_dir / f"{slide_id}.tiff"
    error_msg = ""

    is_wsi_valid = False
    try:
        with OpenSlide(str(slide_path)) as wsi:
            thumb = wsi.get_thumbnail(size=(512, 512)).convert("L")
            thumb_array = np.array(thumb)
            tissue_ratio = np.mean(thumb_array < np.percentile(thumb_array, 95))
            is_wsi_valid = tissue_ratio > 0.001

            if not is_wsi_valid:
                error_msg = f"SLIDE_EMPTY: {slide_id} (ratio: {tissue_ratio:.4f})"
    except Exception as e:
        error_msg = f"SLIDE_ERROR: {slide_id} - {e!s}"

    mask_path = annots_dir / f"{slide_id}_mask.tiff"

    annot_status = "missing"
    if mask_path.exists():
        try:
            mask = tifffile.imread(str(mask_path))
            annot_status = (
                "valid" if (mask is not None and mask.size > 0) else "corrupted"
            )
        except Exception as e:
            msg = f"MASK_CORRUPTED: {slide_id} - {e!s}"
            error_msg = f"{error_msg} | {msg}" if error_msg else msg
            annot_status = "corrupted"

    return {
        "slide_id": slide_id,
        "is_wsi_valid": is_wsi_valid,
        "annot_status": annot_status,
        "error_msg": error_msg,
    }


def get_dataframes(
    df_path: Path,
    slides_dir: Path,
    annots_dir: Path,
    properties_path: str | None,
    max_concurrent: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(df_path).rename(columns={"image_id": "slide_id"})

    validation_results = process_items(
        items=df["slide_id"].tolist(),
        process_item=validate_sample,  # type: ignore[arg-type, func-returns-value]
        fn_kwargs={"slides_dir": slides_dir, "annots_dir": annots_dir},
        max_concurrent=max_concurrent,
    )
    df = df.merge(pd.DataFrame(validation_results), on="slide_id", how="left")

    if properties_path is not None:
        properties_df = pd.read_parquet(properties_path)
        properties_df["slide_id"] = properties_df["path"].apply(lambda p: Path(p).stem)
        df = df.merge(
            properties_df[["slide_id", "id", "extent_x", "extent_y", "mpp_x", "mpp_y"]],
            on="slide_id",
            how="left",
        )
    else:
        valid_mask = df["is_wsi_valid"]
        df.loc[valid_mask, "properties"] = df.loc[valid_mask, "slide_id"].apply(
            lambda sid: extract_properties(slides_dir / f"{sid}.tiff")
        )
        properties_cols = pd.json_normalize(df["properties"].dropna().tolist())
        properties_cols.index = df[valid_mask].index
        df = pd.concat([df, properties_cols], axis=1).drop(columns=["properties"])
        df["id"] = None

    error_logs = df[df["error_msg"] != ""]["error_msg"].tolist()
    df["slide_path"] = df["slide_id"].apply(lambda sid: str(slides_dir / f"{sid}.tiff"))
    df["is_annotation_corrupted"] = df["annot_status"] == "corrupted"
    df["annotation"] = df["annot_status"] != "missing"
    df["is_carcinoma"] = df["isup_grade"] > 0

    summary_df = (
        df[df["is_wsi_valid"] & ~df["is_annotation_corrupted"]]
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
    return df[final_cols], summary_df, error_logs


@with_cli_args(["+exploration=panda/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df, summary_df, error_logs = get_dataframes(
        df_path=Path(config.train_csv),
        slides_dir=Path(config.train_images),
        annots_dir=Path(config.train_label_masks),
        properties_path=config.slides_properties,
        max_concurrent=config.max_concurrent,
    )

    with TemporaryDirectory() as output_dir:
        df.to_csv(Path(output_dir, "slides_metadata.csv"), index=False)
        summary_df.to_csv(Path(output_dir, "summary.csv"), index=False)

        if error_logs:
            with open(Path(output_dir, "errors.txt"), "w") as f:
                f.write("\n".join(error_logs))

        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="panda")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
