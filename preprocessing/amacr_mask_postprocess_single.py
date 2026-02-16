import time
from pathlib import Path
import cv2
import numpy as np
import pyvips
from skimage.morphology import remove_small_holes, remove_small_objects

# --- CONFIGURATION ---
INPUT_MASK_PATH = "/home/jovyan/nuclei-graph-transformer/amacr_mask/input/crop_refine_region.tiff"
OUTPUT_DIR = Path("/home/jovyan/nuclei-graph-transformer/amacr_mask/mask_refined_single")

DUST_DISK_SIZE = 3        
MIN_OBJECT_SIZE = 100     
DILATION_DISK_SIZE = 25   
MAX_HOLE_SIZE = 80000    

dust_ksize = DUST_DISK_SIZE * 2 + 1
DUST_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dust_ksize, dust_ksize))

dil_ksize = DILATION_DISK_SIZE * 2 + 1
DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize))
# ---------------------

def refine_mask_array(mask_array: np.ndarray) -> np.ndarray:
    print(f"Refining mask shape: {mask_array.shape}")
    
    if mask_array.ndim == 3:
        mask_uint8 = mask_array[:, :, 0]
    else:
        mask_uint8 = mask_array

    if mask_uint8.dtype == bool:
        mask_uint8 = mask_uint8.astype(np.uint8) * 255
    elif mask_uint8.max() <= 1:
        mask_uint8 = (mask_uint8 > 0).astype(np.uint8) * 255
    else:
        mask_uint8 = (mask_uint8 > 0).astype(np.uint8) * 255

    print("  > Step 1: Opening...")
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, DUST_KERNEL)

    print("  > Step 2: Dilating...")
    mask_uint8 = cv2.dilate(mask_uint8, DILATION_KERNEL)

    print("  > Step 3: Filling Holes...")
    mask_bool = mask_uint8.astype(bool)
    mask_bool = remove_small_holes(mask_bool, area_threshold=500000) 

    mask_bool = remove_small_objects(mask_bool, min_size=MIN_OBJECT_SIZE)

    return mask_bool.astype(np.uint8) * 255

def save_as_tif(input_im: np.ndarray, output_path: Path) -> None:
    """Helper to save numpy array as TIFF using PyVips."""
    height, width = input_im.shape
    vips_im = pyvips.Image.new_from_memory(
        input_im.tobytes(),
        width,
        height,
        1,
        format="uchar"
    )
    vips_im = vips_im.copy(interpretation="b-w")
    vips_im.tiffsave(
        str(output_path),
        compression="deflate",
        tile=True
    )

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading input mask: {INPUT_MASK_PATH}")
    
    try:
        vips_img = pyvips.Image.new_from_file(INPUT_MASK_PATH)
        raw_mask = vips_img.numpy()
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    refined_mask = refine_mask_array(raw_mask)
    out_name = Path(INPUT_MASK_PATH).stem + "_postprocessed.tiff"
    out_path = OUTPUT_DIR / out_name
    
    print(f"Saving to: {out_path}")
    save_as_tif(refined_mask, out_path)

    t = time.time() - t0
    print(f"Done. Time: {t:.2f}s")

if __name__ == "__main__":
    main()