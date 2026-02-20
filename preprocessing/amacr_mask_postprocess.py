import gc
import os
import time
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import pyvips
from rationai.masks import slide_resolution, write_big_tiff
from ratiopath.openslide import OpenSlide
from ratiopath.tiling import grid_tiles
from skimage.morphology import remove_small_holes


# --- CONFIGURATION ---
INPUT_MASK = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_raw/2025_09852-01-02-05-AMACR.tiff"
ERODED_TISSUE_MASK_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded/2025_09852-01-02-05-AMACR.tiff"

OUTPUT_DIR = (
    "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_dilated"
)
TEMP_DIR = "temp_refined"

SPECK_REMOVAL_RADIUS = 5
MIN_PRE_DILATION_AREA = 300
DILATION_DISK_SIZE = 135
MAX_HOLE_SIZE = 1500000
MIN_FINAL_OBJECT_SIZE = 90000
CLEANUP_SCALE = 2

PATCH_SIZE = 8192
PADDING = DILATION_DISK_SIZE + 10
STRIDE = PATCH_SIZE - (2 * PADDING)

# --- KERNELS ---
speck_ksize = SPECK_REMOVAL_RADIUS * 2 + 1
SPECK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (speck_ksize, speck_ksize))
dil_ksize = DILATION_DISK_SIZE * 2 + 1
DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize))


def filter_tissue_tiles(x, y, cw, ch, tissue_mask_np, scale_x, scale_y):
    tx, ty = floor(x * scale_x), floor(y * scale_y)
    tw, th = ceil(cw * scale_x), ceil(ch * scale_y)

    mask_tile = tissue_mask_np[ty : ty + th, tx : tx + tw]
    if mask_tile.size == 0:
        return False

    return np.count_nonzero(mask_tile) > 0


def postprocess_mask():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    vips_tissue_mask = pyvips.Image.new_from_file(ERODED_TISSUE_MASK_PATH)
    tissue_mask_np = vips_tissue_mask.numpy()
    if tissue_mask_np.ndim == 3:
        tissue_mask_np = tissue_mask_np[:, :, 0]
    tissue_mask_np = (
        (tissue_mask_np > 127).astype(np.uint8)
        if tissue_mask_np.max() > 1
        else tissue_mask_np
    )

    mask_slide = pyvips.Image.new_from_file(INPUT_MASK)
    full_w, full_h = mask_slide.width, mask_slide.height

    scale_x = vips_tissue_mask.width / full_w
    scale_y = vips_tissue_mask.height / full_h

    temp_path = Path(TEMP_DIR) / f"{Path(INPUT_MASK).stem}_temp.dat"
    refined_mmap = np.memmap(
        temp_path, dtype="uint8", mode="w+", shape=(full_h, full_w)
    )

    # ==========================================
    # TILED DILATION
    # ==========================================
    print("\n--- Phase 1: Standardized Tiled Dilation ---")

    tiles = grid_tiles(
        slide_extent=(full_w, full_h),
        tile_extent=(PATCH_SIZE, PATCH_SIZE),
        stride=(STRIDE, STRIDE),
        last="keep",
    )

    for x, y in tiles:
        cw, ch = min(PATCH_SIZE, full_w - x), min(PATCH_SIZE, full_h - y)

        if not filter_tissue_tiles(x, y, cw, ch, tissue_mask_np, scale_x, scale_y):
            continue

        px, py = max(0, x - PADDING), max(0, y - PADDING)
        pcw = min(full_w - px, cw + (x - px) + PADDING)
        pch = min(full_h - py, ch + (y - py) + PADDING)

        patch_vips = mask_slide.extract_area(px, py, pcw, pch)
        patch_np = patch_vips.numpy()

        if patch_np.max() == 0:
            continue

        patch_u8 = (patch_np > 0).astype(np.uint8) * 255

        l2_x, l2_y = int(px * scale_x), int(py * scale_y)
        l2_xe, l2_ye = int((px + pcw) * scale_x) + 1, int((py + pch) * scale_y) + 1
        safe_p = tissue_mask_np[l2_y:l2_ye, l2_x:l2_xe]

        if safe_p.size > 0:
            safe_p_hr = cv2.resize(
                safe_p.astype(np.uint8) * 255,
                (pcw, pch),
                interpolation=cv2.INTER_NEAREST,
            )
            patch_u8 &= safe_p_hr
        else:
            patch_u8[:] = 0

        if np.any(patch_u8):
            patch_u8 = cv2.morphologyEx(patch_u8, cv2.MORPH_OPEN, SPECK_KERNEL)
            contours, _ = cv2.findContours(
                patch_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid = [c for c in contours if cv2.contourArea(c) >= MIN_PRE_DILATION_AREA]

            if valid:
                mask_f = np.zeros_like(patch_u8)
                cv2.drawContours(mask_f, valid, -1, 255, -1)
                patch_u8 = cv2.dilate(mask_f, DILATION_KERNEL)
            else:
                patch_u8[:] = 0

        ox, oy = x - px, y - py
        refined_mmap[y : y + ch, x : x + cw] = patch_u8[oy : oy + ch, ox : ox + cw]

        del patch_np, patch_u8
        print(f"  > Processing Tile: x={x}, y={y}", end="\r")

    refined_mmap.flush()

    # ==========================================
    # CLEANUP & HOLE FILLING
    # ==========================================
    print("\n\n--- Phase 2: Global Object Cleanup & Hole Filling ---")

    lr_h, lr_w = full_h // CLEANUP_SCALE, full_w // CLEANUP_SCALE
    lr_mask = cv2.resize(
        refined_mmap[::CLEANUP_SCALE, ::CLEANUP_SCALE],
        (lr_w, lr_h),
        interpolation=cv2.INTER_NEAREST,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lr_mask, connectivity=8
    )
    scaled_min_area = MIN_FINAL_OBJECT_SIZE // (CLEANUP_SCALE**2)

    lr_cleaned = np.zeros_like(lr_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= scaled_min_area:
            lr_cleaned[labels == i] = 255

    scaled_max_hole = MAX_HOLE_SIZE // (CLEANUP_SCALE**2)
    lr_final = (
        remove_small_holes(lr_cleaned > 0, area_threshold=scaled_max_hole).astype(
            np.uint8
        )
        * 255
    )

    print("Final edge smoothing and merging...")
    for y in range(0, full_h, PATCH_SIZE):
        for x in range(0, full_w, PATCH_SIZE):
            cw, ch = min(PATCH_SIZE, full_w - x), min(PATCH_SIZE, full_h - y)
            ly, lx = y // CLEANUP_SCALE, x // CLEANUP_SCALE
            lch, lcw = ch // CLEANUP_SCALE + 1, cw // CLEANUP_SCALE + 1

            k_patch = lr_final[ly : ly + lch, lx : lx + lcw]
            k_patch_hr = cv2.resize(k_patch, (cw, ch), interpolation=cv2.INTER_LINEAR)

            refined_mmap[y : y + ch, x : x + cw] = (k_patch_hr > 127).astype(
                np.uint8
            ) * 255

    refined_mmap.flush()

    # ==========================================
    # EXPORT
    # ==========================================
    print("\n--- Phase 3: Final Export ---")

    vips_im = pyvips.Image.new_from_memory(
        refined_mmap.data, full_w, full_h, 1, "uchar"
    )
    vips_im = vips_im.copy(interpretation="b-w")

    with OpenSlide(INPUT_MASK) as slide:
        mpp_x, mpp_y = slide_resolution(slide, 0)

    out_name = Path(INPUT_MASK).stem + ".tiff"
    out_path = Path(OUTPUT_DIR) / out_name

    print(f"Writing BigTIFF to: {out_path}")

    write_big_tiff(
        vips_im,
        path=out_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=512,
        tile_height=512,
    )

    del refined_mmap, vips_im, mask_slide
    gc.collect()
    time.sleep(1)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"Refinement Complete. Time: {(time.time() - t0) // 60:.0f}m")


if __name__ == "__main__":
    postprocess_mask()
