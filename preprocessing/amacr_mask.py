import logging
import os
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np


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
from skimage.color import rgb2gray
from skimage.draw import polygon as polygon_draw
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.filters.rank import enhance_contrast
from skimage.measure import label, regionprops
from skimage.morphology import (
    closing,
    disk,
    opening,
    remove_small_holes,
    remove_small_objects,
)
from skimage.transform import warp
from skimage.util import invert


def isolate_stain(image: NDArray, matrix: NDArray, i: int, i_o: int = 230) -> NDArray:
    """Isolate stain as grayscale image.

    Args:
        image (NDArray): Image to be processed.
        matrix (NDArray): Stain matrix.
        i (int): Index of stain to be isolated in stain matrix
        i_o (int, optional): Background RGB intensities. Defaults to 230.
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
    """
    if stains is None:
        stains = ["hematoxylin", "dab", "null"]

    im_input = img_as_ubyte(image)
    stain_matrix = np.array([stain_color_map[st] for st in stains]).T[:, :2]

    # Compute stain matrix adaptively
    sparsity_factor = 0.5
    i_0 = 230
    # Filter warning for division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        im_sda = color_conversion.rgb_to_sda(im_input, i_0)
    stain_matrix = separate_stains_xu_snmf(
        im_sda,
        stain_matrix,
        sparsity_factor,
    )
    return stain_matrix


def save_as_tif(input_im: NDArray, output_path: Path) -> None:
    """Save image in tiff format.

    Args:
        input_im (NDArray): Image to be saved.
        output_path (Path): Path where image will be saved.
    """
    vips_im = pyvips.Image.new_from_memory(
        np.array(Image.fromarray(input_im).convert("RGBA")),
        input_im.shape[1],
        input_im.shape[0],
        4,
        format="uchar",
    )
    vips_im.tiffsave(
        str(output_path),
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
    )


class TMAMaskerMask:
    # tma_transformer: TMATransformer

    def __init__(
        self,
        mask_min_area,
        holes_min_area,
        holes_max_area,
        color_seg_kernel_size,
        color_seg_max_dist,
        color_seg_ratio,
        # tma_transformer,
        holes_area_threshold,
    ) -> None:
        self.mask_min_area = mask_min_area
        self.holes_min_area = holes_min_area
        self.holes_max_area = holes_max_area
        self.color_seg_kernel_size = color_seg_kernel_size
        self.color_seg_max_dist = color_seg_max_dist
        self.color_seg_ratio = color_seg_ratio
        # self.tma_transformer = tma_transformer
        self.holes_area_threshold = holes_area_threshold

    def log_params(self):
        mlflow.log_param("mask_min_area", self.mask_min_area)
        mlflow.log_param("holes_min_area", self.holes_min_area)
        mlflow.log_param("holes_max_area", self.holes_max_area)
        mlflow.log_param("color_seg_kernel_size", self.color_seg_kernel_size)
        mlflow.log_param("color_seg_max_dist", self.color_seg_max_dist)
        mlflow.log_param("color_seg_ratio", self.color_seg_ratio)
        mlflow.log_param("holes_area_threshold", self.holes_area_threshold)
        # self.tma_transformer.log_params()

    def process_pair(self, trg_img_fp):
        logging.getLogger("PIL").setLevel(
            logging.WARNING
        )  # keep this if you dont want terminal flooded with logs
        print(f"Processing: {trg_img_fp}")

        ce_img = np.array(
            (Image.open(trg_img_fp))
        )  # open the slide you want to process, cant be used on huge slides, it will load ~level 7 1700x800

        dab_smaller = Image.open(trg_img_fp)
        dab_smaller.seek(
            2
        )  # open smaller resolution for cheaper computation of stain matrix
        dab_smaller = np.array((dab_smaller))
        print(ce_img.shape)
        print(dab_smaller.shape)

        print("Obtaining stain matrices")
        stains = ["hematoxylin", "dab", "null"]
        hdab_rgb = compute_stain_matrix(
            dab_smaller, stains
        )  # compute stain matrix which helps separating stains from original slide
        c_he_stain = isolate_stain(ce_img, hdab_rgb, 0)
        c_dab_stain = isolate_stain(ce_img, hdab_rgb, 1)
        c_null_stain = isolate_stain(ce_img, hdab_rgb, 2)

        cytokeratin_mask = img_as_float(
            self.__create_cytokeratin_mask(
                c_dab_stain, c_he_stain, c_null_stain, ce_img, self.mask_min_area
            )
        )
        # excluded fill_holes and remove_background

        return cytokeratin_mask

    def process_whole(self, trg_img_fp, out_fp, temp_dir="temp_masks"):
        logging.getLogger("PIL").setLevel(logging.WARNING)
        os.makedirs(temp_dir, exist_ok=True)

        print(f"Processing (via PyVips): {Path(trg_img_fp).name}")

        # 1. Open with PyVips (Level 0 = Full Res)
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

        for y in range(0, full_height, patch_size):
            for x in range(0, full_width, patch_size):
                cw = min(patch_size, full_width - x)
                ch = min(patch_size, full_height - y)

                try:
                    # FIX: Use PyVips extraction, NOT slide.read_region
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
            try:
                os.remove(temp_filename)
            except:
                pass

        return None

    def __create_cytokeratin_mask(
        self, dab_stain, he_stain, null_stain, original_rgb, mask_min_area
    ) -> NDArray:
        """Enhanced mask creation using Spectral Dominance and Saturation Filtering."""
        hsv_img = skimage.color.rgb2hsv(original_rgb)
        saturation = hsv_img[:, :, 1]
        is_saturated = saturation > 0.12

        not_shadow = dab_stain > (null_stain * 1.8)

        is_brown = dab_stain > (he_stain * 2.0)

        # 3. OTSU THRESHOLDING
        try:
            thresh = threshold_otsu(dab_stain)
            thresh = np.clip(thresh, 0.08, 0.25)
        except:
            thresh = 0.1

        final_mask = (dab_stain > thresh) & is_saturated & not_shadow & is_brown

        if np.any(final_mask):
            final_mask = opening(final_mask, disk(1))
            final_mask = remove_small_objects(final_mask, min_size=mask_min_area)

        return final_mask

    @staticmethod
    def crete_exclude_mask(annot: shapely.MultiPolygon, shape) -> NDArray:
        """Draw exclude mask.

        Args:
            annot (shapely.MultiPolygon) : annotations polygons
            shape (NDArray): shape of mask
        Returns:
            NDArray: exclude mask
        """
        mask = np.zeros(shape[:2])
        if annot is not None:
            TMAMaskerMask.draw_multipolygon(annot, mask, 1, 0)
        return mask

    @staticmethod
    def draw_multipolygon(
        multipolygon: shapely.MultiPolygon, image: NDArray, value: int, channel: int
    ) -> None:
        """Draw multipolygon on image."""
        for p in multipolygon.geoms:
            TMAMaskerMask.draw_polygon(p, image, value, channel)

    @staticmethod
    def polygon_to_annotation(polygon: shapely.Polygon) -> NDArray:
        """Converts polygon to numpy array.

        Returns:
            NDArray: numpy array representation of polygon.
        """
        return np.asarray(polygon.exterior.coords)[:-1, :]

    @staticmethod
    def draw_polygon(
        polygon: shapely.Polygon, image: NDArray, value: int, channel: int
    ) -> NDArray:
        """Draw polygon on image.

        Returns:
            NDArray: image
        """
        annotation = TMAMaskerMask.polygon_to_annotation(polygon)
        rr, cc = polygon_draw(annotation[:, 0], annotation[:, 1])
        for a, b in zip(rr, cc, strict=False):
            if a >= image.shape[0] or b >= image.shape[1]:
                continue
            if len(image.shape) == 2:
                image[a, b] = value
            else:
                image[a, b, channel] = value
        return image

    @staticmethod
    def transform_mask_by_shapely_transform(img: NDArray, transform) -> NDArray:
        """Apply shapely transform to the mask.

        Returns:
            NDArray: transformed mask / image
        """
        translate = skimage.transform.SimilarityTransform(
            translation=(-transform.trans_y, -transform.trans_x)
        )
        translated = skimage.transform.warp(img, translate)

        rotated = skimage.transform.rotate(
            translated,
            transform.rotation_angle,
            center=(transform.rotation_origin_y, transform.rotation_origin_x),
        )
        return rotated

    @staticmethod
    def transform_image_pwise(image, transform):
        if len(image.shape) > 2:
            a = []
            for i in range(image.shape[2]):
                a.append(warp(image[:, :, i], transform.inverse, order=0))
            return np.stack(a, axis=2)

        return warp(image, transform.inverse)

    @staticmethod
    def hole_mask(mask: NDArray, holes_max_area) -> NDArray:
        """Creates binary mask of holes in mask.

        mask (NDArray): Binary mask.

        Returns:
            NDArray: Binary mask of holes in mask.
        """
        h_mask = np.zeros(mask.shape, dtype=bool)
        regionprops = TMAMaskerMask.holes_in_mask(mask, holes_max_area)
        for r in regionprops:
            coords = tuple(zip(*r.coords, strict=False))
            h_mask[coords] = 1
        return h_mask

    @staticmethod
    def fill_holes(
        mask: NDArray, hematoxylin_mask: NDArray, holes_max_area, holes_min_area
    ) -> NDArray:
        """Fills holes in mask.

        Fills parts of holes which intersect hematoxylin_mask, then fills remaining small holes.

        Returns:
            NDArray: Mask with filled holes.
        """
        mask = img_as_bool(mask)
        mask = closing(mask, disk(2))
        hole_m = TMAMaskerMask.hole_mask(mask, holes_max_area)
        hole_m = remove_small_holes(hole_m, 20)
        out = np.logical_or(mask, np.logical_and(hole_m, hematoxylin_mask))
        out = closing(out, disk(3))
        out = remove_small_holes(out, holes_min_area, out=out)
        return out

    @staticmethod
    def holes_in_mask(mask: NDArray, max_size: int) -> NDArray:
        """Find holes in mask.

        Args:
            mask (NDArray): Binary mask.
            max_size (int): Maximal size of holes.

        Returns:
          NDArray:  array of regionprops of holes
        """
        inverted = invert(mask)
        lb = label(inverted)
        regions = np.asarray(regionprops(lb))
        return regions[[r.area < max_size for r in regions]]

    @staticmethod
    def create_he_mask(hematoxylin_stain: NDArray) -> NDArray:
        """Creates binary mask of hematoxylin stain using Otsu method.

        Args:
            hematoxylin_stain (NDArray): NxM
        Returns:
            NDArray: Binary mask.
        """
        mask = hematoxylin_stain > threshold_otsu(hematoxylin_stain)
        mask = closing(mask, disk(2))
        mask = remove_small_holes(mask, 20)
        mask = remove_small_objects(mask, 20)
        return mask

    def mask_remove_he_background(
        self,
        cytokeratin_mask: NDArray,
        he_image: NDArray,
        mask_min_area,
        bg_thresh: int = 195,
    ) -> NDArray:
        """Removes regions of a mask which are considered to be a background of HE image.

        Returns:
            NDArray: Binary mask.
        """
        cytokeratin_mask = skimage.util.img_as_ubyte(cytokeratin_mask)
        he_image = skimage.util.img_as_ubyte(he_image)
        gray = img_as_ubyte(rgb2gray(he_image))
        enhanced = enhance_contrast(gray, disk(3))

        he_to_remove = (enhanced > bg_thresh) * 255
        mask = cytokeratin_mask - he_to_remove
        mask = np.clip(mask, a_min=0, a_max=255)

        mask = closing(mask, disk(3))
        mask = remove_small_objects(mask, mask_min_area)
        mask = remove_small_holes(mask, self.holes_area_threshold)
        return mask


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
    # source_fp = (
    #     "/home/jovyan/nuclei-graph-transformer/amacr_mask/input/crop_pyramid.tiff"
    # )
    out_fp = (OUT_DIR / f"{Path(source_fp).stem}").with_suffix(".tif")
    # mask = mask_masker.process_pair(source_fp)
    mask_masker.process_whole(source_fp, out_fp)
    # save_as_tif(img_as_ubyte(mask), out_fp)

    t = time.time() - t0
    print(f"Running time: {t // 60} minutes {t % 60} seconds")


if __name__ == "__main__":
    main()
