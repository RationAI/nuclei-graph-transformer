import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyvips
from skimage.morphology import remove_small_holes

# --- CONFIGURATION ---
INPUT_MRXS = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
INPUT_MASK = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_raw/2025_09852-01-02-05-AMACR.tiff"
ERODED_TISSUE_MASK_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded/2025_09852-01-02-05-AMACR.tiff"

OUTPUT_DIR = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_dilated"
TEMP_DIR = "temp_refined"

SPECK_REMOVAL_RADIUS = 5
MIN_PRE_DILATION_AREA = 300

DILATION_DISK_SIZE = 140  
MAX_HOLE_SIZE = 1500000
MIN_FINAL_OBJECT_SIZE = 90000

# --- KERNELS ---
speck_ksize = SPECK_REMOVAL_RADIUS * 2 + 1
SPECK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (speck_ksize, speck_ksize))

dil_ksize = DILATION_DISK_SIZE * 2 + 1
DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize))

print(f"Speck Radius: {SPECK_REMOVAL_RADIUS} px")
print(f"Pre-Dilation Filter: > {MIN_PRE_DILATION_AREA} px")
print(f"Dilation Radius: {DILATION_DISK_SIZE} px")
print(f"Final Filter: > {MIN_FINAL_OBJECT_SIZE} px")
# ---------------------

def load_tissue_mask(mask_path):
    print(f"Loading Pre-Eroded Tissue Mask from: {mask_path}")
    try:
        vips_mask = pyvips.Image.new_from_file(mask_path)
        mask_np = vips_mask.numpy()
        if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
        if mask_np.max() > 1:
            mask_uint8 = (mask_np > 127).astype(np.uint8)
        else:
            mask_uint8 = mask_np.astype(np.uint8)
        return mask_uint8, vips_mask.width, vips_mask.height
    except Exception as e:
        print(f"Failed to load tissue mask: {e}")
        return None, 0, 0

def postprocess_mask():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    out_name = Path(INPUT_MASK).stem + ".tiff"
    out_path = Path(OUTPUT_DIR) / out_name

    safe_mask_l2, l2_w, l2_h = load_tissue_mask(ERODED_TISSUE_MASK_PATH)
    if safe_mask_l2 is None: return

    print(f"Loading Input AMACR Mask: {INPUT_MASK}")
    mask_slide = pyvips.Image.new_from_file(INPUT_MASK)

    full_width = mask_slide.width
    full_height = mask_slide.height
    
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

            if patch_np.max() == 0: continue
            
            patch_uint8 = (patch_np > 0).astype(np.uint8) * 255

            l2_x_start = int(x * scale_x)
            l2_y_start = int(y * scale_y)
            l2_x_end = int((x + cw) * scale_x) + 1
            l2_y_end = int((y + ch) * scale_y) + 1
            y_start = min(l2_y_start, safe_mask_l2.shape[0])
            y_end = min(l2_y_end, safe_mask_l2.shape[0])
            x_start = min(l2_x_start, safe_mask_l2.shape[1])
            x_end = min(l2_x_end, safe_mask_l2.shape[1])

            safe_patch_l2 = safe_mask_l2[y_start:y_end, x_start:x_end]

            if safe_patch_l2.size > 0:
                safe_patch_l2_gray = safe_patch_l2.astype(np.uint8) * 255
                safe_patch_l0 = cv2.resize(
                    safe_patch_l2_gray, (cw, ch), interpolation=cv2.INTER_NEAREST
                )
                patch_uint8 = patch_uint8 & safe_patch_l0
            else:
                patch_uint8[:] = 0

            if np.any(patch_uint8):
                patch_uint8 = cv2.morphologyEx(patch_uint8, cv2.MORPH_OPEN, SPECK_KERNEL)

                contours, _ = cv2.findContours(patch_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_PRE_DILATION_AREA]
                
                if not valid_contours:
                    refined_memmap[y : y + ch, x : x + cw] = 0
                    continue
                
                mask_filtered = np.zeros_like(patch_uint8)
                cv2.drawContours(mask_filtered, valid_contours, -1, 255, thickness=cv2.FILLED)
                patch_uint8 = mask_filtered

                patch_uint8 = cv2.dilate(patch_uint8, DILATION_KERNEL)

                patch_bool = patch_uint8.astype(bool)
                patch_bool = remove_small_holes(
                    patch_bool, area_threshold=MAX_HOLE_SIZE
                )
                patch_uint8 = patch_bool.astype(np.uint8) * 255

                contours, _ = cv2.findContours(patch_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_contours = [c for c in contours if cv2.contourArea(c) > MIN_FINAL_OBJECT_SIZE]
                
                if large_contours:
                    mask_clean = np.zeros_like(patch_uint8)
                    cv2.drawContours(mask_clean, large_contours, -1, 255, thickness=cv2.FILLED)
                    patch_uint8 = mask_clean
                else:
                    patch_uint8[:] = 0

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
    if os.path.exists(temp_filename): os.remove(temp_filename)

    t = time.time() - t0
    print(f"Refinement Complete. Time: {t // 60:.0f}m {t % 60:.0f}s")

if __name__ == "__main__":
    postprocess_mask()