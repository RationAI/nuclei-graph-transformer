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
    slide_id: str,
    slides_dir: Path,
    annots_dir: Path,
    tissue_threshold: float,
    log_file: Path,
) -> bool:
    """Checks validity of a slide and its corresponding annotation mask.

    The slide is considered valid if it contains a sufficient amount of
    tissue (≥`tissue_threshold`). The annotation mask is checked for
    existence and readability.
    """
    slide_path = slides_dir / f"{slide_id}.tiff"

    def log(msg: str) -> None:
        with log_file.open("a") as f:
            f.write(msg + "\n")
            f.flush()

    try:
        with OpenSlide(str(slide_path)) as wsi:
            thumb = wsi.get_thumbnail(size=(512, 512)).convert("L")
            thumb_array = np.array(thumb)
            tissue_ratio = np.mean(thumb_array < np.percentile(thumb_array, 95))
            if tissue_ratio < tissue_threshold:
                log(f"SLIDE_EMPTY: {slide_id} (ratio={tissue_ratio:.4f})")
                return False
    except Exception as e:
        log(f"SLIDE_ERROR: {slide_id} - {e!s}")
        return False

    mask_path = annots_dir / f"{slide_id}_mask.tiff"
    if mask_path.exists():
        try:
            mask = tifffile.imread(str(mask_path))
            if mask is None or mask.size == 0:
                log(f"MASK_EMPTY: {slide_id}")
                return False
        except Exception as e:
            log(f"MASK_CORRUPTED: {slide_id} - {e!s}")
            return False

    return True


def get_dataframes(
    metadata_csv_path: Path,
    slides_dir: Path,
    annots_dir: Path,
    properties_pq_path: Path,
    exclude_slides: list[str],
    tissue_threshold: float,
    log_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metadata_csv_path).rename(columns={"image_id": "slide_id"})

    if exclude_slides:
        df = df[~df["slide_id"].isin(exclude_slides)]

    slide_ids = df["slide_id"].tolist()

    futures = {
        validate_sample.remote(
            slide_id, slides_dir, annots_dir, tissue_threshold, log_file
        ): slide_id
        for slide_id in slide_ids
    }

    results_by_slide = {}
    with tqdm(total=len(futures), desc="Validating Slides and Annotations") as pbar:
        while futures:
            done, _ = ray.wait(list(futures.keys()), num_returns=min(10, len(futures)))
            for ref in done:
                slide_id = futures.pop(ref)
                results_by_slide[slide_id] = ray.get(ref)
            pbar.update(len(done))

    valid = {sid for sid, is_valid in results_by_slide.items() if is_valid}
    df = df[df["slide_id"].isin(valid)].reset_index(drop=True)

    properties_df = pd.read_parquet(properties_pq_path)
    properties_df["slide_id"] = [Path(p).stem for p in properties_df["path"]]
    properties_df = properties_df.rename(columns={"id": "segmentation_id"})

    df = df.merge(
        properties_df[
            ["slide_id", "segmentation_id", "extent_x", "extent_y", "mpp_x", "mpp_y"]
        ],
        on="slide_id",
        how="left",
    )

    df["slide_path"] = df["slide_id"].apply(lambda sid: str(slides_dir / f"{sid}.tiff"))
    df["has_annotation"] = df["slide_id"].apply(
        lambda sid: (annots_dir / f"{sid}_mask.tiff").exists()
    )
    df["has_segmentation"] = df["segmentation_id"].notna()

    summary_df = (
        df.groupby(["data_provider", "isup_grade", "gleason_score"])
        .agg(Total_Slides=("slide_id", "count"), Annotations=("has_annotation", "sum"))
        .reset_index()
    )

    final_cols = [
        "slide_id",  # 32-character hex string identifier for each slide
        "slide_path",
        "segmentation_id",  # ID of the slide in the parquet dataset with segmented nuclei
        "data_provider",  # 'radboud' or 'karolinska'
        "isup_grade",
        "gleason_score",
        "has_segmentation",  # True if the segmentation file exists
        "has_annotation",  # True if the annotation mask exists
        "extent_x",
        "extent_y",
        "mpp_x",
        "mpp_y",
    ]
    return df[final_cols], summary_df


@with_cli_args(["+exploration=panda/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    ray.init(num_cpus=config.max_concurrent)

    exclude_slides = (
        pd.read_csv(Path(config.exclude_slides))["slide_stem"].tolist()
        if config.exclude_slides
        else []
    )

    with TemporaryDirectory() as output_dir:
        df, summary_df = get_dataframes(
            metadata_csv_path=Path(config.metadata_csv),
            slides_dir=Path(config.slides_dir),
            annots_dir=Path(config.label_masks_dir),
            properties_pq_path=Path(config.slides_properties_parquet),
            exclude_slides=exclude_slides,
            tissue_threshold=config.tissue_threshold,
            log_file=Path(output_dir) / "errors.log",
        )

        df.to_csv(Path(output_dir) / "slides_metadata.csv", index=False)
        summary_df.to_csv(Path(output_dir) / "summary.csv", index=False)

        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="panda")
        mlflow.log_input(slide_dataset, context="slides_metadata")

    ray.shutdown()


if __name__ == "__main__":
    main()
