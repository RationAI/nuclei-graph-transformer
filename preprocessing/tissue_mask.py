from pathlib import Path
import pyvips
from openslide import OpenSlide
from rationai.masks import (
    slide_resolution,
    write_big_tiff,
)
import cv2
import numpy as np

OUT_PATH = Path("/home/jovyan/nuclei-graph-transformer/amacr_mask/tissue_mask")
SLIDE_PATH = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
LEVEL = 2

def tissue_mask(
    slide: pyvips.Image, 
    mpp: float, 
    disk_factor: float = 40, 
    threshold: int = 10,
    max_hole_size: int = 50000 
) -> pyvips.Image:
    
    img_np = slide.numpy()
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(saturation, (7, 7), 0)
    _, mask_uint8 = cv2.threshold(blurred_s, threshold, 255, cv2.THRESH_BINARY)

    disk_size = int(disk_factor / mpp)
    if disk_size < 1: disk_size = 1
    
    print(f"  > Bridging radius: {disk_size} px")
    k_size = disk_size * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    mask_connected = cv2.dilate(mask_uint8, kernel)

    print(f"  > Filling holes smaller than {max_hole_size} px...")
    
    contours, hierarchy = cv2.findContours(mask_connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(cnt)
                if area < max_hole_size:
                    cv2.drawContours(mask_connected, [cnt], -1, 255, -1)

    erode_radius = max(1, disk_size - 5)
    k_size_erode = erode_radius * 2 + 1
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size_erode, k_size_erode))
    
    mask_final = cv2.erode(mask_connected, kernel_erode)

    height, width = mask_final.shape
    vi_mask = pyvips.Image.new_from_memory(
        mask_final.tobytes(), 
        width, 
        height, 
        1, 
        "uchar"
    )
    vi_mask = vi_mask.copy(interpretation="b-w")
    return vi_mask

def process_slide(slide_path: str, level: int, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level)
    
    print(f"Processing: {Path(slide_path).name}")
    
    slide_vips = pyvips.Image.new_from_file(slide_path, level=level)
    
    mask = tissue_mask(
        slide_vips, 
        mpp=(mpp_x + mpp_y) / 2,
        max_hole_size=50000 
    )
    
    mask_path = output_path / Path(slide_path).with_suffix(".tiff").name
    print(f"Saving to: {mask_path}")
    write_big_tiff(
        mask, 
        path=mask_path, 
        mpp_x=mpp_x, 
        mpp_y=mpp_y,
        tile_width=256,
        tile_height=256 
    )

if __name__ == "__main__":
    process_slide(SLIDE_PATH, LEVEL, OUT_PATH)