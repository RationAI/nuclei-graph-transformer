"""Script for computing CAM-based labels for segmented nuclei.

Assumes the following structure of input data:
1. Segmented nuclei (`preprocessing/nuclei_segmentation.py`):
<NUCLEI_PATH>/
    <DATASET_NAME>/
        slide_id=<SLIDE_NAME>/
            *.parquet (columns "id" (str), "polygon" (np.ndarray[float]) and "centroid" (np.ndarray[float]))

2. Exploratory Metadataset (`exploration/save_metadataset.py`):
<DATASET_NAME>/
    slides_metadata.csv (column "slide_path" (str) and "is_carcinoma" (bool))

3. CAM masks (`preprocessing/merge_cam_masks.py`):
<CAM_MASKS_URI>/
    <SLIDE_NAME>.tiff (bipolar heatmap of CAM intensities in [0, 255], where `bipolar_zero_offset` is the neutral point)
<MISSING_CAM_MASKS_URI>.csv (column "slide_path" (str))

This script computes:
- `cam_score` (float): The average CAM intensity under the nucleus' vertices and centroid.
- `cam_label` (int):  Each nucleus is assigned label
    - "1" if the fraction of its polygon vertices overlaps (≥ `overlap_threshold`) with thresholded
      positive CAM region (intensity ≥ `positive_threshold`),
    - "0" if the fraction of its polygon vertices overlaps (≥ `overlap_threshold`) with thresholded
      negative CAM region (intensity ≤ `negative_threshold`),
    - "-1" otherwise (uncertain).
Negative slides are not processed as they are implicitly assumed to have all nuclei labeled as negative.

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), "cam_label" (int), and "cam_score" (float))
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
import ray
import tifffile
from einops import rearrange
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def get_cam_values(
    slide_path: Path,
    nuclei: pd.DataFrame,
    cam_mask_path: Path,
) -> NDArray[np.float32]:
    """Computes CAM intensities at nuclei vertices and their centroids."""
    mask: NDArray[np.uint8] = tifffile.imread(cam_mask_path).squeeze()
    mask_extent_y, mask_extent_x = mask.shape  # assumes mask is single-channel

    with OpenSlide(slide_path) as slide:
        wsi_extent_x, wsi_extent_y = slide.dimensions
    scale_x = mask_extent_x / wsi_extent_x
    scale_y = mask_extent_y / wsi_extent_y

    polygons = rearrange(nuclei["polygon"].tolist(), "b (v d) -> b v d", d=2)
    centroids = np.array(nuclei["centroid"].tolist())[:, None, :]  # b 1 2
    points_to_sample = np.concatenate([polygons, centroids], axis=1)

    coords = np.round(points_to_sample * np.array([scale_x, scale_y])).astype(int)
    x_coords = np.clip(coords[..., 0], 0, mask_extent_x - 1)
    y_coords = np.clip(coords[..., 1], 0, mask_extent_y - 1)

    return mask[y_coords, x_coords].astype(np.float32)


@ray.remote(num_cpus=1, memory=(2 * 1024**3))
def run_cam_labeling(
    slide_path: Path,
    nuclei_dir: Path,
    cam_masks_dir: Path,
    output_dir: Path,
    overlap_thr: float,
    positive_thr: float,
    negative_thr: float,
    bipolar_zero_offset: float,
) -> None:
    slide_id = slide_path.stem
    dataset_name = slide_path.parents[0].name
    nuclei_path = nuclei_dir / dataset_name / f"slide_id={slide_id}"
    nuclei = pd.read_parquet(nuclei_path).sort_values("id")
    nuclei["slide_id"] = slide_id
    nuclei["cam_label"] = -1  # default uncertain

    cam_mask_path = cam_masks_dir / f"{slide_id}.tiff"
    cam_values = get_cam_values(slide_path, nuclei, cam_mask_path)
    nuclei["cam_score"] = np.mean(cam_values, axis=1)
    divisor = np.where(
        cam_values > bipolar_zero_offset,
        255.0 - bipolar_zero_offset,
        bipolar_zero_offset,
    )
    cam_values_scaled = (cam_values - bipolar_zero_offset) / divisor

    neg_mask = np.mean(cam_values_scaled <= -negative_thr, axis=1) >= overlap_thr
    nuclei.loc[neg_mask, "cam_label"] = 0
    pos_mask = np.mean(cam_values_scaled >= positive_thr, axis=1) >= overlap_thr
    nuclei.loc[pos_mask, "cam_label"] = 1

    output_path = output_dir / f"{slide_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["slide_id", "id", "cam_label", "cam_score"]
    nuclei[cols].to_parquet(output_path, index=False)


@with_cli_args(["+preprocessing=cam_labels"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    train_slides = pd.read_csv(download_artifacts(config.train_metadata_uri))
    test_slides = pd.read_csv(download_artifacts(config.test_metadata_uri))
    slides = pd.concat([train_slides, test_slides])
    exclude = pd.read_csv(download_artifacts(config.missing_cam_masks_uri))
    valid_slides = slides[
        slides["is_carcinoma"] & ~slides["slide_path"].isin(exclude["slide_path"])
    ]
    cam_masks_dir = Path(download_artifacts(config.cam_masks_uri))

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=valid_slides["slide_path"].map(Path),
            process_item=run_cam_labeling,
            fn_kwargs={
                "nuclei_dir": Path(config.nuclei_path),
                "cam_masks_dir": cam_masks_dir,
                "output_dir": Path(tmp_dir),
                "overlap_thr": config.overlap_threshold,
                "positive_thr": config.positive_threshold,
                "negative_thr": config.negative_threshold,
                "bipolar_zero_offset": config.bipolar_zero_offset,
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(
            local_dir=tmp_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
