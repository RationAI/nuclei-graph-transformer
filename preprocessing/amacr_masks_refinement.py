"""Script for AMACR stain-based annotation masks for Whole Slide Images (WSIs).

This pipeline takes a raw AMACR mask and a pre-computed eroded tissue mask, and applies
a tiled dilation with a morphological refinement.

The main steps are:
1. Tissue Restriction: Masking out edge artifacts using the pre-computed eroded tissue mask.
2. Local Noise Removal: Application of morphological opening and contour-area filtering.
3. Tiled Dilation: Expansion of the cleaned mask.
3. Global Artifact Removal: Eliminating isolated artifact regions that survived the tiled cleanup.
4. Global Hole Filling: Filling internal holes up to a defined maximum area threshold.

Assumes the following structure of input data:
1. Exploratory Metadataset (TODO):
slides_metadata.csv (columns "slide_path" (str))

2. Raw AMACR Masks (`amacr_masks.py`):
<RAW_MASKS_URI>/
    <SLIDE_NAME>.tiff (binary single-channel raw AMACR mask)

3. Eroded Tissue Masks (`tissue_masks_erode_edges.py`):
<ERODED_TISSUE_MASKS_URI>/
    <SLIDE_NAME>.tiff (binary single-channel eroded tissue mask)

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel refined AMACR mask)
"""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast

import cv2
import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from skimage.morphology import remove_small_holes


SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
RAW_MASKS_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_raw"
)
ERODED_TISSUE_MASKS_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded"
)
OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_refined"
)


class RefinementParams(TypedDict):
    noise_removal_radius: int
    min_pre_dilation_area: int
    dilation_disk_size: int
    max_hole_size: int
    min_final_object_size: int
    cleanup_scale: int


def extract_upscaled_tissue_mask(
    tissue_mask: pyvips.Image,
    padded_bounds: tuple[int, int, int, int],
    scales: tuple[float, float],
) -> NDArray:
    """Extracts and upscales a low-res tissue mask tile."""
    padded_x, padded_y, padded_w, padded_h = padded_bounds
    scale_x, scale_y = scales

    lowres_w, lowres_h = tissue_mask.width, tissue_mask.height

    lowres_padded_x = int(np.clip(padded_x * scale_x, 0, lowres_w - 1))
    lowres_padded_y = int(np.clip(padded_y * scale_y, 0, lowres_h - 1))
    lowres_padded_w = int(np.clip(padded_w * scale_x, 1, lowres_w - lowres_padded_x))
    lowres_padded_h = int(np.clip(padded_h * scale_y, 1, lowres_h - lowres_padded_y))

    tissue_mask = tissue_mask.extract_area(
        lowres_padded_x, lowres_padded_y, lowres_padded_w, lowres_padded_h
    ).numpy()

    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]
    _, tissue_mask = cv2.threshold(tissue_mask, 127, 255, cv2.THRESH_BINARY)

    tissue_mask_hires = np.zeros((padded_h, padded_w), dtype=np.uint8)
    if tissue_mask.size > 0:
        tissue_mask_hires = cv2.resize(
            tissue_mask,
            (padded_w, padded_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return tissue_mask_hires


def filter_tile_artifacts(
    tile: NDArray,
    noise_kernel: NDArray,
    min_area: int,
) -> NDArray:
    """Applies morphological opening and filters small contours."""
    if not np.any(tile):
        return tile

    # opening (remove noise)
    processed_tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, noise_kernel)

    # contour filtering (removes objects smaller than min area)
    contours, _ = cv2.findContours(
        processed_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_area
    ]

    if valid_contours:
        clean_mask = np.zeros_like(processed_tile)
        cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)
        return clean_mask
    else:
        processed_tile[:] = 0
        return processed_tile


def tiled_refinement(
    mask_slide: pyvips.Image,
    tissue_mask: pyvips.Image,
    refined_mmap: np.memmap,
    tile_size: int,
    params: RefinementParams,
) -> None:
    """Applies morphological noise removal and dilation."""
    # --- Prepare morphological kernels ---
    noise_ksize = params["noise_removal_radius"] * 2 + 1
    noise_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (noise_ksize, noise_ksize)
    )

    dil_ksize = params["dilation_disk_size"] * 2 + 1
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize)
    )

    # --- Calculate scaling and tiling bounds ---
    hires_w, hires_h = mask_slide.width, mask_slide.height
    lowres_w, lowres_h = tissue_mask.width, tissue_mask.height

    scale_x = lowres_w / hires_w
    scale_y = lowres_h / hires_h

    # pad the tile to prevent artifacts at the tile boundaries during dilation
    padding = params["dilation_disk_size"] + 10
    stride = tile_size - (2 * padding)

    # --- Tiled Processing Loop ---
    for y in range(0, hires_h, stride):
        print(f"(Dilation): Row Y={y}/{hires_h}", flush=True)

        for x in range(0, hires_w, stride):
            # "core" region (an actual area we want to write back to the memmap)
            core_w = min(tile_size, hires_w - x)
            core_h = min(tile_size, hires_h - y)

            lowres_core_x = int(np.clip(x * scale_x, 0, lowres_w - 1))
            lowres_core_y = int(np.clip(y * scale_y, 0, lowres_h - 1))
            lowres_core_w = int(np.clip(core_w * scale_x, 1, lowres_w - lowres_core_x))
            lowres_core_h = int(np.clip(core_h * scale_y, 1, lowres_h - lowres_core_y))

            tissue_core = tissue_mask.extract_area(
                lowres_core_x, lowres_core_y, lowres_core_w, lowres_core_h
            ).numpy()

            # if this core region is empty, skip it
            if tissue_core.size == 0 or np.max(tissue_core) == 0:
                continue

            # extract padded high-resolution data
            padded_x = max(0, x - padding)
            padded_y = max(0, y - padding)
            padded_w = min(hires_w - padded_x, core_w + (x - padded_x) + padding)
            padded_h = min(hires_h - padded_y, core_h + (y - padded_y) + padding)

            hires_tile = mask_slide.extract_area(padded_x, padded_y, padded_w, padded_h)
            hires_tile = hires_tile.numpy()

            if hires_tile.ndim == 3:
                hires_tile = hires_tile[:, :, 0]

            if hires_tile.max() == 0:  # empty tile
                continue

            tile_u8 = (hires_tile > 0).astype(np.uint8) * 255

            # --- Tissue Edge Filtering ---
            padded_bounds = (padded_x, padded_y, padded_w, padded_h)
            scales = (scale_x, scale_y)
            tissue_mask_hires = extract_upscaled_tissue_mask(
                tissue_mask, padded_bounds, scales
            )

            tile_u8 &= tissue_mask_hires

            # --- Morphological Cleaning & Dilation ---
            min_area = params["min_pre_dilation_area"]
            tile_u8 = filter_tile_artifacts(tile_u8, noise_kernel, min_area)

            if np.any(tile_u8):
                tile_u8 = cv2.dilate(tile_u8, dilation_kernel)

            # --- Write Back ---
            offset_x = x - padded_x
            offset_y = y - padded_y

            refined_mmap[y : y + core_h, x : x + core_w] = tile_u8[
                offset_y : offset_y + core_h, offset_x : offset_x + core_w
            ]

            del hires_tile, tile_u8, tissue_mask_hires

    refined_mmap.flush()


def create_downsampled_mask(mmap: np.memmap, scale: int, tile_size: int) -> NDArray:
    """Creates a low-resolution view of a (large) memory-mapped array."""
    assert tile_size % scale == 0, "Tile size must be divisible by the scale."

    mmap_height, mmap_width = mmap.shape
    mask_height, mask_width = mmap_height // scale, mmap_width // scale

    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    for y in range(0, mmap_height, tile_size):
        for x in range(0, mmap_width, tile_size):
            mmap_tile_width = min(tile_size, mmap_width - x)
            mmap_tile_height = min(tile_size, mmap_height - y)

            mmap_tile = mmap[y : y + mmap_tile_height, x : x + mmap_tile_width]
            mask_tile = mmap_tile[::scale, ::scale]

            mask_y, mask_x = y // scale, x // scale
            mask_tile_height, mask_tile_width = mask_tile.shape

            if mask_y + mask_tile_height > mask_height:
                mask_tile_height = mask_height - mask_y
            if mask_x + mask_tile_width > mask_width:
                mask_tile_width = mask_width - mask_x

            mask[
                mask_y : mask_y + mask_tile_height, mask_x : mask_x + mask_tile_width
            ] = mask_tile[:mask_tile_height, :mask_tile_width]

    return mask


def apply_global_cleanup(
    mmap: np.memmap, tile_size: int, params: RefinementParams
) -> NDArray:
    """Removes artifacts and fills holes."""
    # --- Downsampling ---
    scale = params["cleanup_scale"]  # downsample factor
    lowres_mask = create_downsampled_mask(mmap, scale, tile_size)

    # --- Connected Component Analysis ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lowres_mask, connectivity=8
    )
    lowres_min_area = params["min_final_object_size"] // (scale**2)

    # create a lookup table: index -> object ID
    label_map = np.zeros(num_labels, dtype=np.uint8)

    # find the IDs of all objects that are larger than the minimum area.
    valid_labels = np.where(stats[:, cv2.CC_STAT_AREA] >= lowres_min_area)[0]
    label_map[valid_labels] = 255
    label_map[0] = 0  # background label

    lowres_cleaned = label_map[labels]

    # --- Hole Filling ---
    lowres_max_hole_size = params["max_hole_size"] // (scale**2)
    lowres_final = remove_small_holes(
        lowres_cleaned > 0, max_size=lowres_max_hole_size
    ).astype(np.uint8)

    lowres_final *= 255
    return lowres_final


def project_lowres_to_highres(
    mask: NDArray, mmap: np.memmap, scale: int, tile_size: int
) -> None:
    """Projects low-resolution mask back into the high-resolution memory map."""
    mmap_height, mmap_width = mmap.shape

    for y in range(0, mmap_height, tile_size):
        for x in range(0, mmap_width, tile_size):
            mmap_tile_width = min(tile_size, mmap_width - x)
            mmap_tile_height = min(tile_size, mmap_height - y)

            mask_y, mask_x = y // scale, x // scale

            mask_tile_width = (mmap_tile_width // scale) + 1
            mask_tile_height = (mmap_tile_height // scale) + 1

            mask_tile = mask[
                mask_y : mask_y + mask_tile_height, mask_x : mask_x + mask_tile_width
            ]

            upscaled_tile = cv2.resize(
                mask_tile,
                (mmap_tile_width, mmap_tile_height),
                interpolation=cv2.INTER_LINEAR,
            )
            upscaled_tile = (upscaled_tile > 127).astype(np.uint8)
            upscaled_tile *= 255

            mmap[y : y + mmap_tile_height, x : x + mmap_tile_width] = upscaled_tile

    mmap.flush()


@ray.remote(memory=8 * 1024**3)
def process_slide(
    slide_path: Path,
    output_dir: Path,
    raw_masks_dir: Path,
    eroded_tissue_masks_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    tile_size: int,
    **params: RefinementParams,
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=0)

    tissue_mask_path = eroded_tissue_masks_dir / f"{slide_path.stem}.tiff"
    tissue_mask = cast(
        "pyvips.Image", pyvips.Image.new_from_file(str(tissue_mask_path))
    )

    raw_mask_path = raw_masks_dir / f"{slide_path.stem}.tiff"
    raw_mask = cast("pyvips.Image", pyvips.Image.new_from_file(str(raw_mask_path)))

    with TemporaryDirectory() as temp_dir:
        temp_filename = Path(temp_dir) / f"{slide_path.stem}_temp.dat"

        refined_mmap = np.memmap(
            temp_filename,
            dtype="uint8",
            mode="w+",
            shape=(raw_mask.height, raw_mask.width),
        )
        refined_mmap[:] = 0

        print("Tiled Refinement...", flush=True)
        tiled_refinement(raw_mask, tissue_mask, refined_mmap, tile_size, params)

        print("Global Cleanup...", flush=True)
        lowres_cleaned_mask = apply_global_cleanup(refined_mmap, tile_size, params)

        print("Projecting final mask to high resolution...", flush=True)
        scale = params["cleanup_scale"]
        project_lowres_to_highres(lowres_cleaned_mask, refined_mmap, scale, tile_size)

        refined_mask = pyvips.Image.new_from_memory(
            refined_mmap.data, refined_mmap.shape[1], refined_mmap.shape[0], 1, "uchar"
        ).copy(interpretation="b-w")

        output_path = output_dir / f"{slide_path.stem}.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_big_tiff(
            image=refined_mask,
            path=output_path,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=mask_tile_width,
            tile_height=mask_tile_height,
        )

        del refined_mask, refined_mmap
        gc.collect()


@with_cli_args(["+preprocessing=amacr_masks_refinement"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # slides = pd.read_csv(download_artifacts(config.metadata_uri))
    # raw_masks_dir = Path(download_artifacts(config.raw_masks_uri))
    # eroded_tissue_masks_dir = Path(download_artifacts(config.eroded_tissue_masks_uri))

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=[Path(SLIDE_PATH)],  # slides["slide_path"].map(Path)
            process_item=process_slide,
            fn_kwargs={
                "output_dir": Path(OUTPUT_DIR),  # Path(tmp_dir),
                "raw_masks_dir": Path(RAW_MASKS_DIR),  # raw_masks_dir,
                "eroded_tissue_masks_dir": Path(
                    ERODED_TISSUE_MASKS_DIR
                ),  # eroded_tissue_masks_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                "tile_size": config.tile_size,
                **config.refinement_params,
            },
            max_concurrent=config.max_concurrent,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
