import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyvips
from skimage.morphology import (
    disk,
    erosion,
    remove_small_holes,
    remove_small_objects,
)

# --- CONFIGURATION ---
INPUT_MRXS = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
INPUT_MASK = "/home/jovyan/nuclei-graph-transformer/amacr_mask/mask/2025_09852-01-02-05-AMACR.tiff"
OUTPUT_DIR = "/home/jovyan/nuclei-graph-transformer/amacr_mask/mask_refined"
TEMP_DIR = "temp_refined"

EDGE_EROSION_ITERATIONS = 5
DUST_DISK_SIZE = 2
MIN_OBJECT_SIZE = 100
DILATION_DISK_SIZE = 45
MAX_HOLE_SIZE = 80000

# --- PRE-CALCULATE OPENCV KERNELS ---
dust_ksize = DUST_DISK_SIZE * 2 + 1
DUST_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dust_ksize, dust_ksize))

dil_ksize = DILATION_DISK_SIZE * 2 + 1
DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize))
# ---------------------


def get_tissue_mask_hsv(rgb_image, s_thresh=15):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    saturation = cv2.GaussianBlur(saturation, (7, 7), 0)
    mask = saturation > s_thresh
    return mask


def create_safe_zone_mask(mrxs_path, erosion_iter):
    print(f"Generating tissue zone from: {Path(mrxs_path).name}")
    slide = pyvips.Image.new_from_file(mrxs_path, level=2)
    l2_np = slide.numpy()

    if l2_np.shape[2] == 4:
        l2_np = l2_np[:, :, :3]

    print("Applying HSV Saturation Masking...")
    mask_tissue = get_tissue_mask_hsv(l2_np, s_thresh=15)

    if erosion_iter > 0:
        print(f"Eroding Safe Zone by {erosion_iter} pixels...")
        
        k_size = erosion_iter * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        mask_uint8 = mask_tissue.astype(np.uint8)
        safe_mask_uint8 = cv2.erode(mask_uint8, kernel)
        
        safe_mask = safe_mask_uint8.astype(bool)
    else:
        safe_mask = mask_tissue
        
    print(f"Safe Zone Dimensions: {safe_mask.shape}")
    return safe_mask, slide.width, slide.height


def postprocess_mask():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    out_name = Path(INPUT_MASK).stem + ".tiff"
    out_path = Path(OUTPUT_DIR) / out_name

    safe_mask_l2, l2_w, l2_h = create_safe_zone_mask(
        INPUT_MRXS, EDGE_EROSION_ITERATIONS
    )
    if safe_mask_l2 is None:
        return

    print(f"Loading input mask: {INPUT_MASK}")
    mask_slide = pyvips.Image.new_from_file(INPUT_MASK)

    full_width = mask_slide.width
    full_height = mask_slide.height
    print(f"Processing Full Dimensions: {full_width}x{full_height}")

    scale_x = l2_w / full_width
    scale_y = l2_h / full_height

    temp_filename = Path(TEMP_DIR) / f"{Path(INPUT_MASK).stem}_temp.dat"
    refined_memmap = np.memmap(
        temp_filename, dtype="uint8", mode="w+", shape=(full_height, full_width)
    )

    patch_size = 8192
    print(f"Starting refinement loop with patch size: {patch_size}...")

    for y in range(0, full_height, patch_size):
        for x in range(0, full_width, patch_size):
            cw = min(patch_size, full_width - x)
            ch = min(patch_size, full_height - y)

            try:
                patch_vips = mask_slide.extract_area(x, y, cw, ch)
                patch_np = patch_vips.numpy()
            except Exception:
                continue

            if patch_np.max() == 0:
                continue

            patch_uint8 = (patch_np > 0).astype(np.uint8) * 255

            l2_x_start = int(x * scale_x)
            l2_y_start = int(y * scale_y)
            l2_x_end = int((x + cw) * scale_x)
            l2_y_end = int((y + ch) * scale_y)

            y_start = min(l2_y_start, safe_mask_l2.shape[0])
            y_end = min(l2_y_end + 1, safe_mask_l2.shape[0])
            x_start = min(l2_x_start, safe_mask_l2.shape[1])
            x_end = min(l2_x_end + 1, safe_mask_l2.shape[1])

            safe_patch_l2 = safe_mask_l2[y_start:y_end, x_start:x_end]

            if safe_patch_l2.size > 0:
                safe_patch_l2_gray = safe_patch_l2.astype(np.uint8) * 255
                safe_patch_l0_gray = cv2.resize(
                    safe_patch_l2_gray,
                    (cw, ch),
                    interpolation=cv2.INTER_LINEAR,
                )
                patch_uint8 = patch_uint8 & (safe_patch_l0_gray > 127).astype(np.uint8) * 255
            else:
                patch_uint8[:] = 0

            if np.any(patch_uint8):
                patch_uint8 = cv2.morphologyEx(patch_uint8, cv2.MORPH_OPEN, DUST_KERNEL)
                patch_uint8 = cv2.dilate(patch_uint8, DILATION_KERNEL)
                patch_bool = patch_uint8.astype(bool)
                patch_bool = remove_small_holes(
                    patch_bool, area_threshold=MAX_HOLE_SIZE
                )
                
                patch_bool = remove_small_objects(patch_bool, min_size=MIN_OBJECT_SIZE)
                patch_uint8 = patch_bool.astype(np.uint8) * 255

            refined_memmap[y : y + ch, x : x + cw] = patch_uint8
            print(f"  > Refined tile: x={x}, y={y}", end="\r")

        print(f"\nRow Complete: Y={y}")

    print(f"Saving final TIFF: {out_path}")
    refined_memmap.flush()

    vips_im = pyvips.Image.new_from_array(refined_memmap)
    vips_im = vips_im.copy(interpretation="b-w")

    vips_im.tiffsave(
        str(out_path),
        bigtiff=True,
        compression="deflate",
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
    )

    del refined_memmap
    del vips_im
    import gc
    gc.collect()
    
    if os.path.exists(temp_filename):
        try: os.remove(temp_filename)
        except: pass

    t = time.time() - t0
    print(f"Refinement Complete. Time: {t // 60:.0f}m {t % 60:.0f}s")


if __name__ == "__main__":
    postprocess_mask()