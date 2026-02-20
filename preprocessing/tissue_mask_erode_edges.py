import os
from pathlib import Path

import cv2
import numpy as np
import pyvips
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff


# --- CONFIGURATION ---
SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
INPUT_MASK_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask/2025_09852-01-02-05-AMACR.tiff"
OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded"
)

LEVEL = 2
EDGE_EROSION_ITERATIONS = 50


def erode_tissue_mask(vips_mask: pyvips.Image, iterations: int) -> pyvips.Image:
    mask_np = vips_mask.numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # Ensure uint8 0-255
    mask_uint8 = (mask_np > 127).astype(np.uint8) * 255

    # Create solid mask for safe boundary erosion
    mask_solid = np.zeros_like(mask_uint8)
    ext_contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(mask_solid, ext_contours, -1, 255, thickness=cv2.FILLED)

    k_size = iterations * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    mask_eroded_solid = cv2.erode(mask_solid, kernel)
    mask_final = cv2.bitwise_and(mask_uint8, mask_eroded_solid)

    vi_mask = pyvips.Image.new_from_array(mask_final)
    return vi_mask.copy(interpretation="b-w")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata from original slide
    with OpenSlide(SLIDE_PATH) as slide:
        mpp_x, mpp_y = slide_resolution(slide, LEVEL)

    # Process Mask
    input_vips = pyvips.Image.new_from_file(INPUT_MASK_PATH)
    eroded_vips = erode_tissue_mask(input_vips, EDGE_EROSION_ITERATIONS)

    out_path = Path(OUTPUT_DIR) / Path(INPUT_MASK_PATH).name
    write_big_tiff(
        eroded_vips,
        path=out_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=512,
        tile_height=512,
    )


if __name__ == "__main__":
    main()
