import os
import time
from pathlib import Path
import cv2
import numpy as np
import pyvips
from rationai.masks import write_big_tiff, slide_resolution
from openslide import OpenSlide

SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
TISSUE_MASK_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask/2025_09852-01-02-05-AMACR.tiff"
OUTPUT_DIR = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded"

EDGE_EROSION_ITERATIONS = 50

def save_eroded_tissue_mask():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    
    out_name = Path(TISSUE_MASK_PATH).stem + ".tiff"
    out_path = Path(OUTPUT_DIR) / out_name

    print(f"Loading Tissue Mask: {TISSUE_MASK_PATH}")
    try:
        vips_mask = pyvips.Image.new_from_file(TISSUE_MASK_PATH)
        mask_np = vips_mask.numpy()
        
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
            
        if mask_np.max() > 1:
            mask_uint8 = (mask_np > 127).astype(np.uint8) * 255
        else:
            mask_uint8 = mask_np.astype(np.uint8) * 255

        if EDGE_EROSION_ITERATIONS > 0:
            print(f"Applying Outer-Only Erosion: {EDGE_EROSION_ITERATIONS} pixels...")
            
            mask_solid = np.zeros_like(mask_uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_solid, contours, -1, 255, thickness=cv2.FILLED)
            
            k_size = EDGE_EROSION_ITERATIONS * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask_eroded_solid = cv2.erode(mask_solid, kernel)
            
            mask_eroded = cv2.bitwise_and(mask_uint8, mask_eroded_solid)
            
        else:
            mask_eroded = mask_uint8

        print(f"Result shape: {mask_eroded.shape}")

        print(f"Saving to: {out_path}")
        with OpenSlide(SLIDE_PATH) as slide:
            mpp_x, mpp_y = slide_resolution(slide, 2) 

        height, width = mask_eroded.shape
        mask_eroded = np.ascontiguousarray(mask_eroded)
        vi_mask = pyvips.Image.new_from_memory(
            mask_eroded.data, 
            width, 
            height, 
            bands=1, 
            format="uchar"
        )
        vi_mask = vi_mask.copy(interpretation="b-w")

        write_big_tiff(
            vi_mask, 
            path=out_path, 
            mpp_x=mpp_x, 
            mpp_y=mpp_y,
            tile_width=256,
            tile_height=256 
        )

        t = time.time() - t0
        print(f"Done. Time: {t:.2f}s")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    save_eroded_tissue_mask()