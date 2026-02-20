"""Script for generating tissue masks for Whole Slide Images (WSIs).

This pipeline is intended to be used for very lightly stained slides where traditional Otsu thresholding
on the saturation channel fails (the histogram is almost unimodal). Hard threshold is used instead.
The mask is refined my morphological closing (to fill gaps), hole filling, and opening (to remove noise).

Assumes the following structure of input data:
1. Exploratory Metadataset (TODO):
    <DATASET_NAME>/
        slides_metadata.csv (columns "slide_path" (str))

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel mask of detected tissue)
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import cv2
import hydra
import pandas as pd
import pyvips
import ray
from cv2 import (
    GaussianBlur,
    contourArea,
    cvtColor,
    drawContours,
    findContours,
    getStructuringElement,
    morphologyEx,
)
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask"
)


def tissue_mask(
    slide: pyvips.Image,
    mpp: float,
    disk_factor: float,
    max_hole_size: int,
    threshold: int,
) -> pyvips.Image:
    img_np = slide.numpy()
    if img_np.shape[2] == 4:  # drop alpha
        img_np = img_np[:, :, :3]

    # Extract HSV saturation channel
    hsv = cvtColor(img_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    blurred_s = GaussianBlur(saturation, (7, 7), 0)

    # Hard thresholding (Otsu fails on very lightly stained slides due to skewed histogram)
    _, mask_bin = cv2.threshold(blurred_s, threshold, 255, cv2.THRESH_BINARY)

    # Closing (fill gaps)
    disk_size = max(1, int(disk_factor / mpp))
    k_size = disk_size * 2 + 1
    closing_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask_closed = morphologyEx(mask_bin, cv2.MORPH_CLOSE, closing_kernel)

    # Hole Filling
    contours, hierarchy = findContours(
        mask_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )  # RETR_CCOMP creates a 2-level hierarchy: external boundaries and holes
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # holes are contours with a parent
            if hierarchy[0][i][3] != -1 and contourArea(cnt) < max_hole_size:
                drawContours(mask_closed, [cnt], -1, 255, -1)

    # Opening (remove noise)
    opening_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask_opened = morphologyEx(mask_closed, cv2.MORPH_OPEN, opening_kernel)

    return pyvips.Image.new_from_array(mask_opened).copy(interpretation="b-w")


@ray.remote(memory=3 * 1024**3)
def process_slide(
    slide_path: Path,
    output_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    level: int,
    **morph_params: Any,
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level)
    mpp = (mpp_x + mpp_y) / 2

    slide = cast(
        "pyvips.Image", pyvips.Image.new_from_file(str(slide_path), level=level)
    )
    mask = tissue_mask(slide, mpp, **morph_params)

    output_path = output_dir / f"{slide_path.stem}.tiff"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_big_tiff(
        image=mask,
        path=output_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # slides = pd.read_csv(download_artifacts(config.metadata_uri))

    morph_params = {
        "disk_factor": config.disk_factor,
        "max_hole_size": config.max_hole_size,
        "threshold": config.threshold,
    }

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=[Path(SLIDE_PATH)],  # slides["slide_path"].map(Path),
            process_item=process_slide,
            fn_kwargs={
                "output_dir": Path(OUTPUT_DIR),  # Path(tmp_dir),
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                "level": config.level,
                **morph_params,
            },
            max_concurrent=config.max_concurrent,
        )
        # logger.log_artifacts(
        #     local_dir=tmp_dir, artifact_path=config.mlflow_artifact_path
        # )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
