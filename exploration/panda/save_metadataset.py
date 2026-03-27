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
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from tqdm import tqdm


@ray.remote(num_cpus=1)
def validate_sample(
    slide_id: str, slides_dir: Path, annots_dir: Path
) -> dict[str, str | bool | None]:
    slide_path = slides_dir / f"{slide_id}.tiff"
    error_msg = None

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
            error_msg = f"{error_msg}\n{msg}" if error_msg else msg
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
    properties_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(df_path).rename(columns={"image_id": "slide_id"})

    futures = [
        validate_sample.remote(slide_id, slides_dir, annots_dir)
        for slide_id in df["slide_id"].tolist()
    ]

    results = []
    with tqdm(total=len(futures), desc="Validating Slides") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=min(10, len(futures)))
            results.extend(ray.get(done))
            pbar.update(len(done))

    df = df.merge(pd.DataFrame(results), on="slide_id", how="left")

    properties_df = pd.read_parquet(properties_path)
    properties_df["slide_id"] = [Path(p).stem for p in properties_df["path"]]
    df = df.merge(
        properties_df[["slide_id", "id", "extent_x", "extent_y", "mpp_x", "mpp_y"]],
        on="slide_id",
        how="left",
    )

    error_logs = df.loc[df["error_msg"].notna(), "error_msg"].tolist()
    df["slide_path"] = slides_dir.as_posix() + "/" + df["slide_id"] + ".tiff"
    df["is_annotation_corrupted"] = df["annot_status"] == "corrupted"
    df["has_annotation"] = df["annot_status"] != "missing"
    df["has_segmentation"] = df["id"].notna()
    df["is_carcinoma"] = df["isup_grade"] > 0

    summary_df = (
        df[df["is_wsi_valid"] & ~df["is_annotation_corrupted"]]
        .groupby(["data_provider", "is_carcinoma", "isup_grade"])
        .agg(Total_Slides=("slide_id", "count"), Annotations=("has_annotation", "sum"))
        .reset_index()
    )

    final_cols = [
        "slide_id",  # 32-character hex string identifier for each slide
        "slide_path",
        "id",  # ID of the slide in the parquet dataset with segmented nuclei
        "data_provider",  # 'radboud' or 'karolinska'
        "is_carcinoma",  # True if the slide contains carcinoma based on the ISUP grade
        "has_segmentation",  # True if the segmentation file exists
        "has_annotation",  # True if the annotation mask exists
        "is_annotation_corrupted",  # True if the annotation mask is corrupted or unreadable
        "is_wsi_valid",  # True if the whole slide image is valid and contains tissue
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
    ray.init(num_cpus=config.max_concurrent)

    df, summary_df, error_logs = get_dataframes(
        df_path=Path(config.train_csv),
        slides_dir=Path(config.train_images),
        annots_dir=Path(config.train_label_masks),
        properties_path=Path(config.slides_properties),
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

    ray.shutdown()


if __name__ == "__main__":
    main()
