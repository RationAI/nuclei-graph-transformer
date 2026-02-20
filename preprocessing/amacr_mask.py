import gc
import logging
import os
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pyvips
from histomicstk.preprocessing import color_conversion
from histomicstk.preprocessing.color_deconvolution import (
    color_deconvolution,
    complement_stain_matrix,
    separate_stains_xu_snmf,
    stain_color_map,
)
from numpy.typing import NDArray
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from skimage.util import invert


# --- CONFIGURATION & HYPERPARAMETERS ---
CONFIG = {
    "mask_min_area": 60,
    "holes_min_area": 50,
    "holes_max_area": 400,
    "patch_size": 3000,
    "shadow_ratio": 1.8,
    "brown_ratio": 2.0,
    "sat_threshold": 0.12,
}

OUT_DIR = Path("/home/jovyan/nuclei-graph-transformer/amacr_mask/mask")
SOURCE_FP = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"


def isolate_stain(
    image: NDArray, matrix: NDArray, stain_index: int, i_o: int = 230
) -> NDArray:
    """Isolate specific stain (e.g., DAB or Hematoxylin) as a grayscale image."""
    im_deconvolved = color_deconvolution(
        img_as_ubyte(image),
        complement_stain_matrix(matrix),
        i_o,
    )
    r = im_deconvolved.Stains[:, :, stain_index]
    res = rescale_intensity(img_as_float(r), out_range=(0, 1))
    return invert(res)


def compute_adaptive_stain_matrix(
    image: NDArray, stains: list[str] | None = None
) -> NDArray:
    """Compute stain matrix adaptively based on image content."""
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


def create_cytokeratin_mask(
    original_rgb: NDArray,
    dab_stain: NDArray,
    he_stain: NDArray,
    null_stain: NDArray,
    params: dict,
) -> NDArray:
    hsv_img = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    is_saturated = (hsv_img[:, :, 1] / 255.0) > params["sat_threshold"]
    not_shadow = dab_stain > (null_stain * params["shadow_ratio"])
    is_brown = dab_stain > (he_stain * params["brown_ratio"])

    try:
        dab_uint8 = (dab_stain * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            dab_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_val /= 255.0
        thresh_high = np.clip(otsu_val, 0.20, 0.30)
        thresh_low = thresh_high * 0.4
    except Exception:
        thresh_high, thresh_low = 0.25, 0.10

    hysteresis_mask = apply_hysteresis_threshold(dab_stain, thresh_low, thresh_high)
    final_mask = hysteresis_mask & is_saturated & not_shadow & is_brown

    if np.any(final_mask):
        final_mask_u8 = final_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask_u8 = cv2.morphologyEx(final_mask_u8, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(
            final_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_cleaned = np.zeros_like(final_mask_u8)
        valid_cnts = [c for c in cnts if cv2.contourArea(c) >= params["mask_min_area"]]
        cv2.drawContours(mask_cleaned, valid_cnts, -1, 255, thickness=cv2.FILLED)

        return mask_cleaned > 0

    return final_mask


def process_wsi(trg_img_fp: str, out_fp: Path, params: dict):
    logging.getLogger("PIL").setLevel(logging.WARNING)
    os.makedirs(out_fp.parent, exist_ok=True)

    slide = pyvips.Image.new_from_file(str(trg_img_fp), level=0)
    full_width, full_height = slide.width, slide.height

    print(f"Analyzing: {Path(trg_img_fp).name}")
    level_2 = pyvips.Image.new_from_file(str(trg_img_fp), level=2).numpy()[:, :, :3]

    mask_tissue = np.mean(level_2, axis=2) < 235
    tissue_pixels = level_2[mask_tissue].reshape(-1, 1, 3)

    if len(tissue_pixels) > 100000:
        rng = np.random.default_rng(42)
        tissue_pixels = tissue_pixels[
            rng.choice(len(tissue_pixels), 100000, replace=False)
        ]

    hdab_matrix = compute_adaptive_stain_matrix(tissue_pixels)

    with TemporaryDirectory() as temp_dir:
        temp_filename = Path(temp_dir) / f"{Path(trg_img_fp).stem}_temp.dat"

        mask_memmap = np.memmap(
            temp_filename, dtype="uint8", mode="w+", shape=(full_height, full_width)
        )
        mask_memmap[:] = 0

        patch_size = params["patch_size"]
        for y in range(0, full_height, patch_size):
            for x in range(0, full_width, patch_size):
                cw, ch = (
                    min(patch_size, full_width - x),
                    min(patch_size, full_height - y),
                )

                try:
                    patch = slide.extract_area(x, y, cw, ch).numpy()[:, :, :3]
                except Exception:
                    continue

                if np.mean(np.mean(patch, axis=2) < 220) < 0.01:
                    continue

                c_dab = isolate_stain(patch, hdab_matrix, 1)
                c_he = isolate_stain(patch, hdab_matrix, 0)
                c_null = isolate_stain(patch, hdab_matrix, 2)

                patch_mask = create_cytokeratin_mask(patch, c_dab, c_he, c_null, params)
                mask_memmap[y : y + ch, x : x + cw] = patch_mask.astype(np.uint8) * 255

            print(f"Progress: Row Y={y}/{full_height}")

        mask_memmap.flush()

        print("Exporting final pyramidal TIFF...")
        vips_im = pyvips.Image.new_from_array(mask_memmap).copy(interpretation="b-w")
        vips_im.tiffsave(
            str(out_fp),
            bigtiff=True,
            compression="deflate",
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

        del mask_memmap, vips_im
        gc.collect()

    print("Temporary files cleared.")


def main():
    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = "4"

    out_fp = (OUT_DIR / f"{Path(SOURCE_FP).stem}").with_suffix(".tiff")
    process_wsi(SOURCE_FP, out_fp, CONFIG)

    t = time.time() - t0
    print(f"Total time: {t // 60:.0f}m {t % 60:.0f}s")


if __name__ == "__main__":
    main()
