"""Script for eroding tissue masks for Whole Slide Images (WSIs).

This pipeline applies edge erosion to existing tissue masks. The masks are intended to be used for
filtering out edge artifacts such as unwanted staining that can occur at the borders of tissue sections.
Internal holes in the tissue mask are preserved.

Assumes the following structure of input data:
1. Exploratory Metadataset (TODO):
slides_metadata.csv (columns "slide_path" (str))

2. Tissue masks (`preprocessing/tissue_masks.py`):
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel mask of detected tissue)

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel eroded tissue mask)
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import cv2
import hydra
import numpy as np
import pyvips
import ray
from cv2 import (
    drawContours,
    findContours,
    getStructuringElement,
)
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
INPUT_MASK_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask"
)
OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded"
)


def erode_tissue_mask(
    tissue_mask: pyvips.Image,
    iterations: int,
) -> pyvips.Image:
    mask_np = tissue_mask.numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    mask_uint8 = (mask_np > 127).astype(np.uint8) * 255

    mask_solid = np.zeros_like(mask_uint8)

    # Identify external contours and fill them to create a solid mask
    # (we do not want to erode internal holes in the mask, only the edges)
    ext_contours, _ = findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    drawContours(mask_solid, ext_contours, -1, 255, thickness=cv2.FILLED)

    # Erosion
    k_size = iterations * 2 + 1
    kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask_eroded_solid = cv2.erode(mask_solid, kernel)

    # Recover internal structures
    mask_final = cv2.bitwise_and(mask_uint8, mask_eroded_solid)
    return pyvips.Image.new_from_array(mask_final).copy(interpretation="b-w")


@ray.remote(memory=4 * 1024**3)
def process_eroded_mask(
    slide_path: Path,
    tissue_mask_dir: Path,
    output_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    level: int,
    erosion_iterations: int,
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level)

    slide_id = slide_path.stem
    tissue_mask_path = tissue_mask_dir / f"{slide_id}.tiff"
    slide = cast("pyvips.Image", pyvips.Image.new_from_file(str(tissue_mask_path)))

    mask = erode_tissue_mask(slide, iterations=erosion_iterations)

    output_path = output_dir / f"{slide_id}.tiff"
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
    # tissue_masks_dir = Path(download_artifacts(config.tissue_masks_uri))

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=[Path(SLIDE_PATH)],  # slides["slide_path"].map(Path)
            process_item=process_eroded_mask,
            fn_kwargs={
                "tissue_mask_dir": Path(INPUT_MASK_DIR),  # tissue_masks_dir
                "output_dir": Path(OUTPUT_DIR),  # Path(tmp_dir),
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                "level": config.level,
                "erosion_iterations": config.edge_erosion_iters,
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
