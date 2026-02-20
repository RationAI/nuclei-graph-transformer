"""Script for refining and dilating AMACR stain masks for Whole Slide Images (WSIs).

This pipeline takes a raw AMACR mask and a pre-computed eroded tissue mask, and applies
a tiled dilation with a morphological refinement.

The main steps are:
1. Tissue Restriction: Masking out edge artifacts using the pre-computed eroded tissue mask.
2. Local Noise Removal: Via morphological opening and contour-area filtering.
3. Tiled Dilation: Expansion of the cleaned mask.
3. Global Artifact Removal: Eliminating isolated artifact regions that survived the tiled cleanup.
4. Global Hole Filling: Filling internal holes up to a defined maximum area threshold using the downsampled view.

Assumes the following structure of input data:
1. Exploratory Metadataset (TODO):
slides_metadata.csv (columns "slide_path" (str))

2. Raw AMACR Masks (`amacr_masks.py`):
<RAW_MASK_DIR>/
    <SLIDE_NAME>.tiff (binary single-channel raw AMACR mask)

3. Eroded Tissue Masks (`tissue_masks_erode_edges.py`):
<ERODED_TISSUE_MASK_DIR>/
    <SLIDE_NAME>.tiff (binary single-channel eroded tissue mask)

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel refined AMACR mask)
"""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

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
RAW_MASK_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_raw"
)
ERODED_TISSUE_MASK_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded"
)
OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_refined"
)


def apply_tiled_dilation(
    mask_slide: pyvips.Image,
    tissue_mask_np: NDArray,
    refined_mmap: np.memmap,
    params: dict[str, Any],
) -> None:
    """Applies morphological noise removal and dilation.

    Args:
        mask_slide (pyvips.Image): The raw AMACR mask to process.
        tissue_mask_np (NDArray): Low-resolution binary tissue mask for valid area bounding.
        refined_mmap (np.memmap): Memory-mapped array to write refined patches into.
        params (dict[str, Any]): Dictionary of morphological hyperparameters:
            - "dilation_disk_size" (int): Radius of the structuring element for dilation.
            - "noise_removal_radius" (int): Radius for morphological opening to remove noise.
            - "min_pre_dilation_area" (int): Minimum area for contours to be kept before dilation.
            - "patch_size" (int): Size of the tiles to process at once.
    """
    # --- Prepare morphological kernels ---
    noise_ksize = params["noise_removal_radius"] * 2 + 1
    noise_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (noise_ksize, noise_ksize)
    )
    dil_ksize = params["dilation_disk_size"] * 2 + 1
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize)
    )
    width, height = mask_slide.width, mask_slide.height

    # --- Calculate scaling factors between the tissue mask and the raw AMACR mask ---
    tissue_extent_y, tissue_extent_x = tissue_mask_np.shape
    scale_x = tissue_extent_x / width
    scale_y = tissue_extent_y / height

    patch_size = params["patch_size"]
    padding = params["dilation_disk_size"] + 10
    stride = patch_size - (2 * padding)

    # --- Tiled Dilation ---
    for y in range(0, height, stride):
        print(f"(Dilation): Row Y={y}/{height}", flush=True)

        for x in range(0, width, stride):
            # The core (unpadded) region for this iteration
            core_width = min(patch_size, width - x)
            core_height = min(patch_size, height - y)

            tile_x = int(np.clip(x * scale_x, 0, tissue_extent_x - 1))
            tile_y = int(np.clip(y * scale_y, 0, tissue_extent_y - 1))
            tile_w = int(np.clip(core_width * scale_x, 1, tissue_extent_x - tile_x))
            tile_h = int(np.clip(core_height * scale_y, 1, tissue_extent_y - tile_y))

            mask_tile = tissue_mask_np[
                tile_y : tile_y + tile_h, tile_x : tile_x + tile_w
            ]
            if mask_tile.size == 0 or np.count_nonzero(mask_tile) == 0:
                continue  # skip empy tissue regions

            # Define the padded region to extract from the high-res WSI
            pad_x = max(0, x - padding)
            pad_y = max(0, y - padding)
            pad_w = min(width - pad_x, core_width + (x - pad_x) + padding)
            pad_h = min(height - pad_y, core_height + (y - pad_y) + padding)

            patch_np = mask_slide.extract_area(pad_x, pad_y, pad_w, pad_h).numpy()

            if patch_np.ndim == 3:
                patch_np = patch_np[:, :, 0]

            if patch_np.max() == 0:
                continue

            patch_u8 = (patch_np > 0).astype(np.uint8) * 255

            # Apply the low-res tissue mask to remove edge artifacts
            lr_x, lr_y = int(pad_x * scale_x), int(pad_y * scale_y)
            lr_end_x = int((pad_x + pad_w) * scale_x) + 1
            lr_end_y = int((pad_y + pad_h) * scale_y) + 1

            tissue_edge_filter = tissue_mask_np[lr_y:lr_end_y, lr_x:lr_end_x]

            if tissue_edge_filter.size > 0:
                tissue_edge_filter_hires = cv2.resize(
                    tissue_edge_filter,
                    (pad_w, pad_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                patch_u8 &= tissue_edge_filter_hires
            else:
                patch_u8[:] = 0

            # Morphological Cleaning (opening -> filtering by area)
            if np.any(patch_u8):
                patch_u8 = cv2.morphologyEx(patch_u8, cv2.MORPH_OPEN, noise_kernel)
                contours, _ = cv2.findContours(
                    patch_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                valid_contours = [
                    contour
                    for contour in contours
                    if cv2.contourArea(contour) >= params["min_pre_dilation_area"]
                ]

                if valid_contours:
                    clean_mask = np.zeros_like(patch_u8)
                    cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)

                    # Dilate the cleaned mask
                    patch_u8 = cv2.dilate(clean_mask, dilation_kernel)
                else:
                    patch_u8[:] = 0

            # Write the processed region back to the memory map (stripping the padding)
            offset_x = x - pad_x
            offset_y = y - pad_y

            refined_mmap[y : y + core_height, x : x + core_width] = patch_u8[
                offset_y : offset_y + core_height, offset_x : offset_x + core_width
            ]

    refined_mmap.flush()


def apply_global_cleanup(refined_mmap: np.memmap, params: dict[str, Any]) -> NDArray:
    """Creates a downsampled global view to remove artifacts and fill holes."""
    scale = params["cleanup_scale"]
    height, width = refined_mmap.shape

    lowres_h, lowres_w = height // scale, width // scale

    lowres_mask = cv2.resize(
        refined_mmap[::scale, ::scale],
        (lowres_w, lowres_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Connected Component Analysis to remove small isolated artifacts
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lowres_mask, connectivity=8
    )

    # Scale area down by scale^2
    lowres_min_area = params["min_final_object_size"] // (scale**2)

    lowres_cleaned = np.zeros_like(lowres_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= lowres_min_area:
            lowres_cleaned[labels == i] = 255

    # Fill internal holes within valid objects
    lowres_max_hole = params["max_hole_size"] // (scale**2)
    lowres_final = remove_small_holes(
        lowres_cleaned > 0, area_threshold=lowres_max_hole
    ).astype(np.uint8)

    return lowres_final * 255


def project_lowres_to_highres(
    low_res_mask: NDArray, high_res_mmap: np.memmap, params: dict[str, Any]
) -> None:
    """Projects the cleaned low-resolution mask back into the high-resolution memmap.

    Args:
        low_res_mask (NDArray): The downsampled mask.
        high_res_mmap (np.memmap): Memory-mapped array to overwrite with final results.
        params (dict[str, Any]): Dictionary containing scale and patch parameters:
            - "cleanup_scale" (int): The downsampling factor used for cleanup.
            - "patch_size" (int): The tile size used for processing.
    """
    scale = params["cleanup_scale"]
    tile_size = params["patch_size"]
    full_h, full_w = high_res_mmap.shape

    for hires_y in range(0, full_h, tile_size):
        for hires_x in range(0, full_w, tile_size):
            hires_extent_x = min(tile_size, full_w - hires_x)
            hires_extent_y = min(tile_size, full_h - hires_y)

            lowres_x, lowres_y = hires_x // scale, hires_y // scale
            lowres_extent_x = hires_extent_x // scale + 1
            lowres_extent_y = hires_extent_y // scale + 1

            lowres_crop = low_res_mask[
                lowres_y : lowres_y + lowres_extent_y,
                lowres_x : lowres_x + lowres_extent_x,
            ]

            upscaled_patch = cv2.resize(
                lowres_crop,
                (hires_extent_x, hires_extent_y),
                interpolation=cv2.INTER_LINEAR,
            )

            high_res_mmap[
                hires_y : hires_y + hires_extent_y, hires_x : hires_x + hires_extent_x
            ] = (upscaled_patch > 127).astype(np.uint8) * 255

    high_res_mmap.flush()


def refine_amacr_mask(
    raw_mask: pyvips.Image,
    tissue_mask: NDArray,
    temp_filename: Path,
    params: dict[str, Any],
) -> tuple[pyvips.Image, np.memmap]:
    """AMACR mask refinement pipeline.

    Args:
        raw_mask (pyvips.Image): The raw AMACR mask.
        tissue_mask (NDArray): The eroded tissue mask for restriction.
        temp_filename (Path): Path to the temporary file for memory mapping.
        params (dict[str, Any]): Hyperparameters for morphology.

    Returns:
        tuple[pyvips.Image, np.memmap]: The refined mask wrapped in PyVips and the memmap array.
    """
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]
    tissue_mask = (tissue_mask > 127).astype(np.uint8) * 255

    refined_mmap = np.memmap(
        temp_filename, dtype="uint8", mode="w+", shape=(raw_mask.height, raw_mask.width)
    )
    refined_mmap[:] = 0

    print("Tiled Dilation...", flush=True)
    apply_tiled_dilation(raw_mask, tissue_mask, refined_mmap, params)

    print("Cleanup & Hole Filling...", flush=True)
    cleaned_mask = apply_global_cleanup(refined_mmap, params)

    print("Projecting final mask to high resolution...", flush=True)
    project_lowres_to_highres(cleaned_mask, refined_mmap, params)

    vips_im = pyvips.Image.new_from_array(refined_mmap).copy(interpretation="b-w")
    return vips_im, refined_mmap


@ray.remote(memory=8 * 1024**3)
def process_slide(
    slide_path: Path,
    output_dir: Path,
    raw_mask_dir: Path,
    eroded_tissue_mask_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    **refinement_params: Any,
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=0)

    eroded_tissue_mask_path = eroded_tissue_mask_dir / f"{slide_path.stem}.tiff"
    eroded_tissue_mask = cast(
        "pyvips.Image", pyvips.Image.new_from_file(str(eroded_tissue_mask_path))
    )

    raw_mask_path = raw_mask_dir / f"{slide_path.stem}.tiff"
    raw_mask = cast("pyvips.Image", pyvips.Image.new_from_file(str(raw_mask_path)))

    with TemporaryDirectory() as temp_dir:
        temp_filename = Path(temp_dir) / f"{slide_path.stem}_temp.dat"

        vips_im, mask_memmap = refine_amacr_mask(
            raw_mask=raw_mask,
            tissue_mask=eroded_tissue_mask.numpy(),
            temp_filename=temp_filename,
            params=refinement_params,
        )

        output_path = output_dir / f"{slide_path.stem}.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_big_tiff(
            image=vips_im,
            path=output_path,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=mask_tile_width,
            tile_height=mask_tile_height,
        )

        del vips_im, mask_memmap
        gc.collect()


@with_cli_args(["+preprocessing=amacr_masks_refinement"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # slides = pd.read_csv(download_artifacts(config.metadata_uri))
    # raw_mask_dir = Path(download_artifacts(config.raw_mask_uri))
    # eroded_tissue_mask_dir = Path(download_artifacts(config.eroded_tissue_mask_uri))

    refinement_params = {
        "noise_removal_radius": config["noise_removal_radius"],
        "min_pre_dilation_area": config["min_pre_dilation_area"],
        "dilation_disk_size": config["dilation_disk_size"],
        "max_hole_size": config["max_hole_size"],
        "min_final_object_size": config["min_final_object_size"],
        "cleanup_scale": config["cleanup_scale"],
        "patch_size": config["patch_size"],
    }

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=[Path(SLIDE_PATH)],  # slides["slide_path"].map(Path)
            process_item=process_slide,
            fn_kwargs={
                "output_dir": Path(OUTPUT_DIR),  # Path(tmp_dir),
                "raw_mask_dir": Path(RAW_MASK_DIR),  # raw_mask_dir,
                "eroded_tissue_mask_dir": Path(
                    ERODED_TISSUE_MASK_DIR
                ),  # eroded_tissue_mask_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                **refinement_params,
            },
            max_concurrent=config.max_concurrent,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
