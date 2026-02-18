import logging
import os
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np

import cv2 

if not hasattr(np, "mat"):
    np.mat = np.asmatrix

import pyvips
import shapely
import skimage
from histomicstk.preprocessing import color_conversion
from histomicstk.preprocessing.color_deconvolution import (
    color_deconvolution,
    complement_stain_matrix,
    separate_stains_xu_snmf,
    stain_color_map,
)
from numpy.typing import NDArray
from PIL import Image
from skimage import img_as_bool, img_as_float, img_as_ubyte
from skimage.draw import polygon as polygon_draw
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold 
from skimage.measure import label, regionprops
from skimage.transform import warp
from skimage.util import invert


def isolate_stain(image: NDArray, matrix: NDArray, i: int, i_o: int = 230) -> NDArray:
    """Isolate stain as grayscale image."""
    im_deconvolved = color_deconvolution(
        img_as_ubyte(image),
        complement_stain_matrix(matrix),
        i_o,
    )
    r = im_deconvolved.Stains[:, :, i]
    res = rescale_intensity(img_as_float(r), out_range=(0, 1))
    return invert(res)


def compute_stain_matrix(image: NDArray, stains: list[str] | None = None) -> NDArray:
    """Compute stain matrix adaptively."""
    if stains is None:
        stains = ["hematoxylin", "dab", "null"]

    im_input = img_as_ubyte(image)
    stain_matrix = np.array([stain_color_map[st] for st in stains]).T[:, :2]

    sparsity_factor = 0.5
    i_0 = 230
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        im_sda = color_conversion.rgb_to_sda(im_input, i_0)
    stain_matrix = separate_stains_xu_snmf(
        im_sda,
        stain_matrix,
        sparsity_factor,
    )
    return stain_matrix


class TMAMaskerMask:
    def __init__(
        self,
        mask_min_area,
        holes_min_area,
        holes_max_area,
        color_seg_kernel_size,
        color_seg_max_dist,
        color_seg_ratio,
        holes_area_threshold,
    ) -> None:
        self.mask_min_area = mask_min_area
        self.holes_min_area = holes_min_area
        self.holes_max_area = holes_max_area
        self.color_seg_kernel_size = color_seg_kernel_size
        self.color_seg_max_dist = color_seg_max_dist
        self.color_seg_ratio = color_seg_ratio
        self.holes_area_threshold = holes_area_threshold

    def log_params(self):
        mlflow.log_param("mask_min_area", self.mask_min_area)
        mlflow.log_param("holes_min_area", self.holes_min_area)
        mlflow.log_param("holes_max_area", self.holes_max_area)
        mlflow.log_param("color_seg_kernel_size", self.color_seg_kernel_size)
        mlflow.log_param("color_seg_max_dist", self.color_seg_max_dist)
        mlflow.log_param("color_seg_ratio", self.color_seg_ratio)
        mlflow.log_param("holes_area_threshold", self.holes_area_threshold)

    def process_whole(self, trg_img_fp, out_fp, temp_dir="temp_masks"):
        logging.getLogger("PIL").setLevel(logging.WARNING)
        os.makedirs(temp_dir, exist_ok=True)

        print(f"Processing (via PyVips): {Path(trg_img_fp).name}")

        slide = pyvips.Image.new_from_file(str(trg_img_fp), level=0)

        full_width = slide.width
        full_height = slide.height
        patch_size = 3000

        print(f"Full Dimensions: {full_width}x{full_height}")

        print("Computing Global Stain Matrix from Level 2 (Filtered)...")
        try:
            level_2_vips = pyvips.Image.new_from_file(str(trg_img_fp), level=2)
            level_2_np = level_2_vips.numpy()
            if level_2_np.shape[2] == 4:
                level_2_np = level_2_np[:, :, :3]

            mask_tissue = np.mean(level_2_np, axis=2) < 235
            tissue_pixels = level_2_np[mask_tissue].reshape(-1, 1, 3)

            print(f"Pixels available: {len(tissue_pixels)}")

            if len(tissue_pixels) > 100000:
                print("Subsampling to 100,000 pixels...")
                rng = np.random.default_rng(42)
                indices = rng.choice(len(tissue_pixels), size=100000, replace=False)
                tissue_pixels = tissue_pixels[indices]

            stains = ["hematoxylin", "dab", "null"]
            hdab_rgb = compute_stain_matrix(tissue_pixels, stains)
            print(f"Global Stain Matrix:\n{hdab_rgb}")

        except Exception as e:
            print(f"CRITICAL: Failed to compute matrix: {e}")
            return
        print(f"Stain Matrix:\n{hdab_rgb}")

        temp_filename = Path(temp_dir) / f"{Path(trg_img_fp).stem}_temp.dat"
        mask_memmap = np.memmap(
            temp_filename, dtype="uint8", mode="w+", shape=(full_height, full_width)
        )
        mask_memmap[:] = 0

        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for y in range(0, full_height, patch_size):
            for x in range(0, full_width, patch_size):
                cw = min(patch_size, full_width - x)
                ch = min(patch_size, full_height - y)

                try:
                    patch_vips = slide.extract_area(x, y, cw, ch)
                    patch_np = patch_vips.numpy()
                    if patch_np.shape[2] == 4:
                        patch_np = patch_np[:, :, :3]
                except Exception:
                    continue

                if np.mean(np.mean(patch_np, axis=2) < 220) < 0.01:
                    continue

                c_dab_stain = isolate_stain(patch_np, hdab_rgb, 1)
                c_he_stain = isolate_stain(patch_np, hdab_rgb, 0)
                c_null_stain = isolate_stain(patch_np, hdab_rgb, 2)

                patch_mask_bool = self.__create_cytokeratin_mask(
                    c_dab_stain, c_he_stain, c_null_stain, patch_np, self.mask_min_area
                )

                mask_memmap[y : y + ch, x : x + cw] = (
                    patch_mask_bool.astype(np.uint8)
                ) * 255

            print(f"Finished row at Y: {y}")

        mask_memmap.flush()
        print(f"Converting temporary map to final TIFF: {out_fp}")

        vips_im = pyvips.Image.new_from_array(mask_memmap)
        vips_im = vips_im.copy(interpretation="b-w")

        vips_im.tiffsave(
            str(out_fp),
            bigtiff=True,
            compression="deflate",
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

        del mask_memmap
        del vips_im
        import gc
        gc.collect()

        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except: pass

        return None

    def __create_cytokeratin_mask(
        self, dab_stain, he_stain, null_stain, original_rgb, mask_min_area
    ) -> NDArray:
        """Enhanced mask creation using CV2 optimization where possible."""
        
        hsv_img = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv_img[:, :, 1] / 255.0
        is_saturated = saturation > 0.12

        not_shadow = dab_stain > (null_stain * 1.8)
        is_brown = dab_stain > (he_stain * 2.0)

        try:
            dab_uint8 = (dab_stain * 255).astype(np.uint8)
            otsu_val, _ = cv2.threshold(dab_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = otsu_val / 255.0 # Convert back to float for comparison

            thresh_high = np.clip(otsu_val, 0.20, 0.30)
            thresh_low = thresh_high * 0.4
        except:
            thresh_high = 0.25
            thresh_low = 0.10

        hysteresis_mask = apply_hysteresis_threshold(dab_stain, thresh_low, thresh_high)

        final_mask = hysteresis_mask & is_saturated & not_shadow & is_brown

        if np.any(final_mask):
            final_mask_u8 = final_mask.astype(np.uint8) * 255
            
            if not hasattr(self, 'morph_kernel'):
                 self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                 
            final_mask_u8 = cv2.morphologyEx(final_mask_u8, cv2.MORPH_OPEN, self.morph_kernel)

            cnts, _ = cv2.findContours(final_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask_cleaned = np.zeros_like(final_mask_u8)
            valid_cnts = [c for c in cnts if cv2.contourArea(c) >= mask_min_area]
            cv2.drawContours(mask_cleaned, valid_cnts, -1, 255, thickness=cv2.FILLED)
            
            final_mask = mask_cleaned > 0

        return final_mask

OUT_DIR = Path("/home/jovyan/nuclei-graph-transformer/amacr_mask/mask")

os.environ["OMP_NUM_THREADS"] = str(4)

mask_masker = TMAMaskerMask(
    mask_min_area=60,
    holes_min_area=50,
    holes_max_area=400,
    color_seg_kernel_size=3,
    color_seg_max_dist=6,
    color_seg_ratio=0.5,
    holes_area_threshold=150,
)


def main():
    t0 = time.time()

    source_fp = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/wsi_data/2025_09852-01-02-05-AMACR.mrxs"
    out_fp = (OUT_DIR / f"{Path(source_fp).stem}").with_suffix(".tiff")
    
    mask_masker.process_whole(source_fp, out_fp)

    t = time.time() - t0
    print(f"Running time: {t // 60:.0f} minutes {t % 60:.0f} seconds")


if __name__ == "__main__":
    main()