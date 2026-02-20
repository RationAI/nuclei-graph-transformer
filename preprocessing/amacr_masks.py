"""Script for generating AMACR stain masks for Whole Slide Images (WSIs).

This pipeline isolates specific stains using adaptive color deconvolution and applies
hysteresis thresholding alongside morphological cleaning.

Assumes the following structure of input data:
1. Exploratory Metadataset (TODO):
slides_metadata.csv (columns "slide_path" (str))

2. Tissue Masks (`tissue_masks.py`)
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel mask of tissue regions)

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel mask of detected AMACR tissue)
"""

import gc
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import cv2
import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
from histomicstk.preprocessing import color_conversion
from histomicstk.preprocessing.color_deconvolution import (
    color_deconvolution,
    complement_stain_matrix,
    separate_stains_xu_snmf,
    stain_color_map,
)
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from skimage.util import invert


SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_raw"
)
TISSUE_MASKS_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask"
)


def isolate_stain(image: NDArray, matrix: NDArray, i: int, i_o: int = 230) -> NDArray:
    """Isolate specific stain (e.g., DAB or Hematoxylin) as a grayscale image.

    Args:
        image (NDArray): Image to be processed.
        matrix (NDArray): Stain matrix.
        i (int): Index of stain to be isolated in stain matrix
        i_o (int, optional): Background RGB intensities. Defaults to 230.

    Source: Image Registration repository (`imagereg/stain_processing.py`)
    """
    im_deconvolved = color_deconvolution(
        img_as_ubyte(image),
        complement_stain_matrix(matrix),
        i_o,
    )
    r = im_deconvolved.Stains[:, :, i]
    res = rescale_intensity(img_as_float(r), out_range=(0, 1))
    return invert(res)


def compute_stain_matrix(image: NDArray, stains: list[str] | None = None) -> NDArray:
    """Compute stain matrix adaptively.

    Returns:
        NDArray: stain matrix

    Source: Image Registration repository (`imagereg/stain_processing.py`)
    """
    import numpy as np

    if not hasattr(np, "mat"):  # due to histomicstk's dependency on old numpy version
        np.mat = np.asmatrix

    if stains is None:
        stains = ["hematoxylin", "dab", "null"]

    im_input = img_as_ubyte(image)
    stain_matrix = np.array([stain_color_map[st] for st in stains]).T[:, :2]

    sparsity_factor = 0.5
    i_0 = 230
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        im_sda = color_conversion.rgb_to_sda(im_input, i_0)

    return separate_stains_xu_snmf(im_sda, stain_matrix, sparsity_factor)


def compute_adaptive_thresholds(dab_stain: NDArray) -> tuple[float, float]:
    """Calculates adaptive high and low thresholds for hysteresis thresholding.

    Args:
        dab_stain (NDArray): The deconvolved DAB (brown) stain channel,
            normalized to the range [0, 1].

    Returns:
        tuple[float, float]: A tuple containing (thresh_low, thresh_high).
    """
    dab_uint8 = (dab_stain * 255).astype(np.uint8)

    otsu_val, _ = cv2.threshold(dab_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val /= 255.0

    # Clip the Otsu value to prevent extreme failures on patches with no/heavy stain
    thresh_high = np.clip(otsu_val, 0.20, 0.30)
    # Low threshold allows fainter edges connected to strong signal to be kept
    thresh_low = thresh_high * 0.4

    return thresh_low, thresh_high


def create_marker_mask(
    original_rgb: NDArray,
    dab_stain: NDArray,
    he_stain: NDArray,
    null_stain: NDArray,
    params: dict[str, Any],
) -> NDArray:
    """Generates a binary mask for target markers (e.g., AMACR) from a tissue patch.

    This function uses a combination of hysteresis thresholding on the target DAB stain,
    color space heuristics (HSV saturation), and relative channel intensities to isolate
    true positive staining from dark tissue shadows, background, and overlapping hematoxylin.

    Args:
        original_rgb (NDArray): The original RGB image patch (uint8).
        dab_stain (NDArray): The deconvolved DAB (brown) channel, normalized [0, 1].
        he_stain (NDArray): The deconvolved Hematoxylin (blue) channel, normalized [0, 1].
        null_stain (NDArray): The deconvolved residual/shadow channel, normalized [0, 1].
        params (dict[str, Any]): Dictionary with keys:
            - "sat_threshold" (float): minimum saturation to consider a pixel valid stain,
            - "shadow_ratio" (float): minimum ratio of DAB to null stain to exclude shadows,
            - "brown_ratio" (float): minimum ratio of DAB to HE stain to exclude blue nuclei,
            - "mask_min_area" (int): minimum area in pixels for a valid mask region.

    Returns:
        NDArray: A boolean 2D array representing the cleaned cytokeratin/target mask.
    """
    # --- Color Heuristic Pre-filtering ---
    hsv_img = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    is_saturated = (hsv_img[:, :, 1] / 255.0) > params["sat_threshold"]
    # ignore dark tissue folds or imaging shadows
    not_shadow = dab_stain > (null_stain * params["shadow_ratio"])
    # ignore dark blue nuclei
    is_brown = dab_stain > (he_stain * params["brown_ratio"])

    # --- Adaptive Intensity Thresholding ---
    thresh_low, thresh_high = compute_adaptive_thresholds(dab_stain)

    # --- Hysteresis Thresholding on DAB Channel ---
    hysteresis_mask = apply_hysteresis_threshold(dab_stain, thresh_low, thresh_high)

    final_mask = hysteresis_mask & is_saturated & not_shadow & is_brown

    if not np.any(final_mask):  # empty mask
        return final_mask

    # --- Morphological Cleanup ---
    final_mask_u8 = final_mask.astype(np.uint8) * 255
    # opening (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask_u8 = cv2.morphologyEx(final_mask_u8, cv2.MORPH_OPEN, kernel)
    # identify distinct objects
    cnts, _ = cv2.findContours(
        final_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # filter out objects smaller than the minimum area and draw them solid
    mask_cleaned = np.zeros_like(final_mask_u8)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= params["mask_min_area"]]
    cv2.drawContours(mask_cleaned, valid_cnts, -1, 255, thickness=cv2.FILLED)

    return mask_cleaned > 0


def sample_tissue_pixels(
    slide_path: Path | str,
    level: int = 2,
    max_pixels: int = 100_000,
    intensity_threshold: int = 235,
    seed: int = 42,
) -> NDArray:
    """Extracts a random sample of tissue pixels to fit adaptive stain deconvolution matrices.

    Args:
        slide_path (Path | str): Path to the Whole Slide Image.
        level (int): The pyramid level to extract from (higher = lower resolution).
        max_pixels (int): Maximum number of tissue pixels to return.
        intensity_threshold (int): Max average RGB value to be considered tissue.
        seed (int): Random seed for reproducible sampling.

    Returns:
        NDArray: An array of RGB tissue pixels of shape (N, 1, 3).
    """
    level_img = cast(
        "pyvips.Image", pyvips.Image.new_from_file(str(slide_path), level=level)
    ).numpy()

    if level_img.shape[2] == 4:  # drop alpha
        level_img = level_img[:, :, :3]

    # average RGB intensity below threshold
    mask_tissue = np.mean(level_img, axis=2) < intensity_threshold

    # flatten into a list of RGB pixels
    tissue_pixels = level_img[mask_tissue].reshape(-1, 1, 3)

    # randomly subsample if too many pixels
    if len(tissue_pixels) > max_pixels:
        rng = np.random.default_rng(seed)
        tissue_pixels = tissue_pixels[
            rng.choice(len(tissue_pixels), max_pixels, replace=False)
        ]

    return tissue_pixels


def compute_amacr_mask(
    slide_path: Path,
    tissue_mask_path: Path,
    temp_filename: Path,
    params: dict[str, Any],
) -> tuple[pyvips.Image, np.memmap]:
    """Computes the AMACR mask for a WSI and stores it in a memory-mapped file.

    Args:
        slide_path (Path): Path to the original Whole Slide Image.
        tissue_mask_path (Path): Path to the precomputed tissue mask for the slide.
        temp_filename (Path): Path to the temporary file for memory mapping.
        params (dict[str, Any]): Dictionary of thresholds and hyperparameters needed
            for computation of the cytokeratin mask and morphological cleaning.
            Should include keys:
              - "patch_size" (int): size of patches to process at a time,
            and the color heuristics for stain isolation as described in `create_cytokeratin_mask`.

    Returns:
        tuple[pyvips.Image, np.memmap]: The PyVips image wrapping the mask,
            and the memmap array itself (must be kept alive until saving is done).
    """
    # get a representative sample of tissue colors
    tissue_pixels = sample_tissue_pixels(slide_path)
    # compute the global stain matrix for the whole slide
    hdab_matrix = compute_stain_matrix(tissue_pixels)

    slide_vips = cast(
        "pyvips.Image", pyvips.Image.new_from_file(str(slide_path), level=0)
    )
    width, height = slide_vips.width, slide_vips.height

    mask_vips = cast("pyvips.Image", pyvips.Image.new_from_file(str(tissue_mask_path)))
    scale_x = mask_vips.width / width
    scale_y = mask_vips.height / height

    mask_memmap = np.memmap(
        temp_filename, dtype="uint8", mode="w+", shape=(height, width)
    )
    mask_memmap[:] = 0

    patch_size = params["patch_size"]
    total_rows = len(range(0, height, patch_size))
    for row_idx, y in enumerate(range(0, height, patch_size)):
        print(
            f"[{slide_path.stem}] Processing row {row_idx + 1}/{total_rows} (Y={y})",
            flush=True,
        )

        for x in range(0, width, patch_size):
            cw = min(patch_size, width - x)
            ch = min(patch_size, height - y)

            mask_x = int(np.clip(x * scale_x, 0, mask_vips.width - 1))
            mask_y = int(np.clip(y * scale_y, 0, mask_vips.height - 1))
            mask_cw = int(np.clip(cw * scale_x, 1, mask_vips.width - mask_x))
            mask_ch = int(np.clip(ch * scale_y, 1, mask_vips.height - mask_y))

            tissue_patch = mask_vips.extract_area(mask_x, mask_y, mask_cw, mask_ch)
            if not np.any(tissue_patch.numpy() > 0):  # skip empty tissue patches
                continue

            patch = slide_vips.extract_area(x, y, cw, ch).numpy()
            if patch.shape[2] == 4:  # drop alpha channel
                patch = patch[:, :, :3]

            c_dab = isolate_stain(patch, hdab_matrix, 1)
            c_he = isolate_stain(patch, hdab_matrix, 0)
            c_null = isolate_stain(patch, hdab_matrix, 2)

            patch_mask = create_marker_mask(patch, c_dab, c_he, c_null, params)
            mask_memmap[y : y + ch, x : x + cw] = patch_mask.astype(np.uint8) * 255

    print(f"[{slide_path.stem}] Flushing memory map to disk...", flush=True)
    mask_memmap.flush()

    vips_im = pyvips.Image.new_from_array(mask_memmap).copy(interpretation="b-w")

    return vips_im, mask_memmap


@ray.remote(memory=8 * 1024**3)
def process_slide(
    slide_path: Path,
    output_dir: Path,
    tissue_mask_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    **amacr_params: Any,
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=0)

    tissue_mask_path = tissue_mask_dir / f"{slide_path.stem}.tiff"

    with TemporaryDirectory() as temp_dir:
        temp_filename = Path(temp_dir) / f"{slide_path.stem}_temp.dat"

        vips_im, mask_memmap = compute_amacr_mask(
            slide_path, tissue_mask_path, temp_filename, amacr_params
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


@with_cli_args(["+preprocessing=amacr_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # slides = pd.read_csv(download_artifacts(config.metadata_uri))
    # tissue_masks_dir = Path(download_artifacts(config.tissue_masks_uri))

    amacr_params = {
        "mask_min_area": config["mask_min_area"],
        "patch_size": config["patch_size"],
        "shadow_ratio": config["shadow_ratio"],
        "brown_ratio": config["brown_ratio"],
        "sat_threshold": config["sat_threshold"],
    }

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=[Path(SLIDE_PATH)],  # slides["slide_path"].map(Path),
            process_item=process_slide,
            fn_kwargs={
                "output_dir": Path(OUTPUT_DIR),  # Path(tmp_dir),
                "tissue_mask_dir": Path(TISSUE_MASKS_DIR),  # tissue_masks_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                **amacr_params,
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
