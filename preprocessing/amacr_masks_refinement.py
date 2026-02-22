"""Script for AMACR stain-based annotation masks for Whole Slide Images (WSIs).

This pipeline takes a raw AMACR mask and a pre-computed eroded tissue mask, and applies
a tiled dilation with a morphological refinement.

The main steps are:
1. Tissue Restriction: Masking out edge artifacts using the pre-computed eroded tissue mask.
2. Local Noise Removal: Application of morphological opening and contour-area filtering.
3. Tiled Dilation: Expansion of the cleaned mask.
4. Downsampling: Processing of the masks to the target scale.
5. Global Artifact Removal: Eliminating isolated artifact regions that survived the tiled cleanup.
6. Global Hole Filling: Filling internal holes up to a defined maximum area threshold.

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
    hires_bounds: tuple[int, int, int, int],
    scales: tuple[float, float],
) -> NDArray:
    """Extracts and upscales a low-resolution tissue mask tile to match high-resolution bounds."""
    hires_x, hires_y, hires_w, hires_h = hires_bounds
    scale_x, scale_y = scales

    full_lr_w, full_lr_h = tissue_mask.width, tissue_mask.height

    lr_x = int(np.clip(hires_x * scale_x, 0, full_lr_w - 1))
    lr_y = int(np.clip(hires_y * scale_y, 0, full_lr_h - 1))
    lr_w = int(np.clip(hires_w * scale_x, 1, full_lr_w - lr_x))
    lr_h = int(np.clip(hires_h * scale_y, 1, full_lr_h - lr_y))

    lr_tile = tissue_mask.extract_area(lr_x, lr_y, lr_w, lr_h).numpy()

    if lr_tile.ndim == 3:
        lr_tile = lr_tile[:, :, 0]
    _, lr_tile = cv2.threshold(lr_tile, 127, 255, cv2.THRESH_BINARY)

    hires_tile = np.zeros((hires_h, hires_w), dtype=np.uint8)
    if lr_tile.size > 0:
        hires_tile = cv2.resize(
            lr_tile,
            (hires_w, hires_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return hires_tile


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
    refined_buffer: NDArray,
    tile_size: int,
    target_scale: int,
    params: RefinementParams,
) -> None:
    """Applies morphological noise removal and dilation, then downsizes tiles to target scale."""
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

    padding = params["dilation_disk_size"] + 10
    stride = tile_size - (2 * padding)

    # --- Tiled Processing Loop ---
    for y in range(0, hires_h, stride):
        print(f"Row Y={y}/{hires_h}", flush=True)

        for x in range(0, hires_w, stride):
            core_w = min(tile_size, hires_w - x)
            core_h = min(tile_size, hires_h - y)

            lowres_core_x = int(np.clip(x * scale_x, 0, lowres_w - 1))
            lowres_core_y = int(np.clip(y * scale_y, 0, lowres_h - 1))
            lowres_core_w = int(np.clip(core_w * scale_x, 1, lowres_w - lowres_core_x))
            lowres_core_h = int(np.clip(core_h * scale_y, 1, lowres_h - lowres_core_y))

            tissue_core = tissue_mask.extract_area(
                lowres_core_x, lowres_core_y, lowres_core_w, lowres_core_h
            ).numpy()

            if tissue_core.size == 0 or np.max(tissue_core) == 0:
                continue

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

            # --- Downsample ---
            offset_x = x - padded_x
            offset_y = y - padded_y
            clean_core = tile_u8[
                offset_y : offset_y + core_h, offset_x : offset_x + core_w
            ]

            out_w_tile = core_w // target_scale
            out_h_tile = core_h // target_scale

            if out_w_tile > 0 and out_h_tile > 0:
                small_tile = cv2.resize(
                    clean_core,
                    (out_w_tile, out_h_tile),
                    interpolation=cv2.INTER_AREA,
                )
                small_tile = (small_tile > 127).astype(np.uint8) * 255

                # write to the downsampled buffer
                out_x = x // target_scale
                out_y = y // target_scale
                refined_buffer[
                    out_y : out_y + out_h_tile, out_x : out_x + out_w_tile
                ] = small_tile

            del hires_tile, tile_u8, tissue_mask_hires


def apply_global_cleanup(
    mask_buffer: NDArray, target_scale: int, params: RefinementParams
) -> NDArray:
    """Removes artifacts and fills holes on the already-downsampled mask."""
    relative_scale = max(1, params["cleanup_scale"] // target_scale)
    effective_scale = target_scale * relative_scale

    if relative_scale > 1:
        small_h, small_w = (
            mask_buffer.shape[0] // relative_scale,
            mask_buffer.shape[1] // relative_scale,
        )
        work_mask = cv2.resize(
            mask_buffer, (small_w, small_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        work_mask = mask_buffer

    # --- Connected Component Analysis ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        work_mask, connectivity=8
    )
    min_area = params["min_final_object_size"] // (effective_scale**2)

    label_map = np.zeros(num_labels, dtype=np.uint8)
    valid_labels = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
    label_map[valid_labels] = 255
    label_map[0] = 0  # background label

    cleaned = label_map[labels]

    # --- Hole Filling ---
    max_hole_size = params["max_hole_size"] // (effective_scale**2)
    final_small = (
        remove_small_holes(cleaned > 0, max_size=max_hole_size).astype(np.uint8) * 255
    )

    if relative_scale > 1:
        final_mask = cv2.resize(
            final_small,
            (mask_buffer.shape[1], mask_buffer.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        final_mask = final_small

    return final_mask


@ray.remote(memory=8 * 1024**3)
def process_slide(
    slide_path: Path,
    output_dir: Path,
    raw_masks_dir: Path,
    eroded_tissue_masks_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    tile_size: int,
    target_scale: int,
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

    # --- initialize a downsampled buffer ---
    lowres_h = raw_mask.height // target_scale
    lowres_w = raw_mask.width // target_scale
    refined_buffer = np.zeros((lowres_h, lowres_w), dtype=np.uint8)

    print("Tiled Refinement...", flush=True)
    tiled_refinement(
        raw_mask, tissue_mask, refined_buffer, tile_size, target_scale, params
    )
    print("Global Cleanup...", flush=True)
    final_mask = apply_global_cleanup(refined_buffer, target_scale, params)

    final_mask_vips = pyvips.Image.new_from_memory(
        final_mask.data, final_mask.shape[1], final_mask.shape[0], 1, "uchar"
    ).copy(interpretation="b-w")

    output_path = output_dir / f"{slide_path.stem}.tiff"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_big_tiff(
        image=final_mask_vips,
        path=output_path,
        mpp_x=mpp_x * target_scale,
        mpp_y=mpp_y * target_scale,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )

    del final_mask_vips, refined_buffer, final_mask
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
                "target_scale": config.target_scale,
                **config.refinement_params,
            },
            max_concurrent=config.max_concurrent,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
