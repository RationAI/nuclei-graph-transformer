"""Script to generate tissue masks for the iCAIRD cervical whole slide images (WSIs).

Many iCAIRD slides are lightly stained, and some have a dark scanner artifact in a corner
of the image. Otsu thresholding of the saturation channel (`rationai.masks.tissue_mask`)
fails on both: the artifact has very high saturation and dominates the threshold, and
strongly stained tissue pulls it above the saturation of the pale tissue. The mask here is
therefore made with a fixed threshold instead:
1. Pixels that are nearly black (HSV value up to `dark_threshold`) are set to zero
   saturation, so artifacts are never tissue.
2. The saturation channel is blurred and thresholded at `saturation_threshold` (glass is
   ~3-9, pale tissue ~15-35).
3. The mask is refined with a closing (fills gaps), a filling of small holes and an opening
   (removes noise).
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import cv2
import hydra
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def tissue_mask(
    slide: pyvips.Image,
    mpp: float,
    dark_threshold: int,
    saturation_threshold: int,
    blur_size: int,
    disk_factor: float,
    max_hole_um2: float,
) -> pyvips.Image:
    """Tissue mask from the blurred saturation channel, ignoring near-black artifacts.

    Args:
        slide: whole-slide image (WSI) at the level to be masked.
        mpp: resolution of `slide` in µm/px.
        dark_threshold: pixels whose HSV value (0-255) is not above this are not tissue.
        saturation_threshold: blurred saturation (0-255) above which a pixel is tissue.
        blur_size: odd size in px of the Gaussian kernel the saturation is blurred with.
        disk_factor: closing and opening use a disk of `disk_factor / mpp` px radius.
        max_hole_um2: holes of the closed mask smaller than this are filled.

    Returns:
        The tissue mask (0/255, single band) with the same size as `slide`.
    """
    _, saturation, value, *_ = slide.extract_band(0, n=3).sRGB2HSV().bandsplit()
    saturation = (value > dark_threshold).ifthenelse(saturation, 0).numpy()
    saturation = saturation.reshape(saturation.shape[:2])  # drop a trailing band axis

    blurred = cv2.GaussianBlur(saturation, (blur_size, blur_size), 0)
    _, mask = cv2.threshold(blurred, saturation_threshold, 255, cv2.THRESH_BINARY)

    disk_size = max(1, round(disk_factor / mpp))
    kernel_size = 2 * disk_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Fill small holes
    max_hole_px = max_hole_um2 / mpp**2
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None:
        for contour, (_, _, _, parent) in zip(contours, hierarchy[0], strict=True):
            if parent != -1 and cv2.contourArea(contour) < max_hole_px:
                cv2.drawContours(mask, [contour], -1, 255, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return pyvips.Image.new_from_array(mask).copy(interpretation="b-w")


@ray.remote(num_cpus=1, memory=(6 * 1024**3))
def process_slide(
    slide_path: Path,
    level: int,
    output_path: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    **mask_params: Any,
) -> None:
    pyvips.concurrency_set(1)

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)

    level_arg = (
        {"page": level}
        if slide_path.suffix.lower() in [".tiff", ".tif"]
        else {"level": level}
    )
    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, **level_arg))
    mask = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2, **mask_params)
    mask_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(
        mask,
        path=mask_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


@with_cli_args(["+preprocessing/tissue_masks=icaird_cervix"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.read_csv(download_artifacts(config.metadata_uri))
    slides_path = [Path(path) for path in df["slide_path"]]

    with TemporaryDirectory() as output_dir:
        process_items(
            slides_path,
            process_item=process_slide,
            fn_kwargs={
                "level": config.level,
                "output_path": Path(output_dir),
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                **config.mask_params,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(local_dir=output_dir, artifact_path="tissue_masks")


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
