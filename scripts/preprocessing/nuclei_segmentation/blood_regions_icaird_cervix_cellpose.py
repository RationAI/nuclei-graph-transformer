"""Blood regions of the iCAIRD cervix Cellpose batch, from the red blood cells Cellpose outlined.

Cellpose-SAM outlines red blood cells as if they were nuclei. An object counts as a red
blood cell (RBC) when its mean colour is red rather than pink or purple and it carries
almost no hematoxylin:
    hue > -38° (0° = red, i.e. above 322° on the usual 0-360° circle) and H OD < 0.012,
measured by stain_features_icaird_cervix_cellpose.py. The rule was calibrated on
IC-CX-01429-01 against LSP-DETR (it flags 96% of the Cellpose objects LSP-DETR does not
find and 0.4% of those it does) and flags at most 0.05% of objects on gland-rich slides.

Blood regions are where RBC objects dominate: RBC and other objects are counted on a
25 µm grid, both counts smoothed (Gaussian, sigma 1.5 cells), and a cell is blood where the
smoothed RBC count is at least MIN_RBC and at least RATIO times the other objects.
Regions under MIN_CELLS cells are dropped and the rest grown by one cell. Inside these
regions Cellpose finds mostly RBCs and misses or garbles the nuclei between them; blood
that Cellpose left empty is not a region (there is nothing to clean there).

The regions are computed from every original Cellpose object (STAIN/objects_original, see
reconstruct_original_icaird_cervix_cellpose.py); clean_icaird_cervix_cellpose.py computes the
same regions for its step 2 and keeps them in its _measures. This script writes them as masks.

Writes to OUT:
    masks/<slide_id>.tiff   the region mask (255 = blood) at 5 µm/px (each 25 µm grid
                            cell is 5 x 5 px), only for slides with blood regions; each
                            mask is read back with OpenSlide and compared, since OpenSlide
                            garbles images that are only a tile or two wide (on 209 of 431
                            masks written on the 25 µm grid itself);
    slides.csv              per slide: objects, rbc, rbc_in_regions, other_in_regions,
                            region_mm2.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.blood_regions_icaird_cervix_cellpose \
        [--workers 14]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyvips
from openslide import OpenSlide
from rationai.masks import write_big_tiff
from scipy.ndimage import binary_dilation, gaussian_filter, label

from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


OUT = Path("/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/iCAIRD/cervix_cellpose_blood_regions")
HUE_MIN, H_MAX = -38.0, 0.012
CELL_PX = 100  # level-0 px, 25 µm at 0.25 µm/px
MASK_PX = 20  # level-0 px per mask px, 5 µm
SIGMA, MIN_RBC, RATIO, MIN_CELLS, GROW = 1.5, 2.0, 2.0, 4, 1


def hue(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hue in degrees of the mean RGB, centred on red: 0 = red, negative towards magenta."""
    rgb = np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8)[None]
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[0, :, 0].astype(np.float32) * 360 / 256
    return (h + 180) % 360 - 180


def red_blood_cells(features: np.lib.npyio.NpzFile) -> np.ndarray:
    """Objects whose colour is that of a red blood cell (see the module docstring)."""
    f = features
    return (f["npx"] > 0) & (hue(f["r"], f["g"], f["b"]) > HUE_MIN) & (f["h"] < H_MAX)


def blood_regions(centroids: np.ndarray, rbc: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    ij = np.floor(centroids / CELL_PX).astype(int)
    ij = np.clip(ij, 0, np.array(grid_shape[::-1]) - 1)
    counts = np.zeros((2, *grid_shape))
    np.add.at(counts[0], (ij[rbc, 1], ij[rbc, 0]), 1)
    np.add.at(counts[1], (ij[~rbc, 1], ij[~rbc, 0]), 1)
    red, other = (gaussian_filter(c, SIGMA) for c in counts)
    regions = (red >= MIN_RBC) & (red >= RATIO * other)
    labels, _ = label(regions)
    big = np.bincount(labels.ravel()) >= MIN_CELLS
    big[0] = False
    regions = big[labels]
    return binary_dilation(regions, iterations=GROW) if regions.any() else regions


def in_regions(centroids: np.ndarray, regions: np.ndarray) -> np.ndarray:
    ij = np.clip(np.floor(centroids / CELL_PX).astype(int), 0, np.array(regions.shape[::-1]) - 1)
    return regions[ij[:, 1], ij[:, 0]]


def write_mask(regions: np.ndarray, path: Path, mpp: float) -> None:
    k = CELL_PX // MASK_PX
    mask = np.repeat(np.repeat(regions.astype(np.uint8) * 255, k, axis=0), k, axis=1)
    write_big_tiff(pyvips.Image.new_from_array(mask), path=path, mpp_x=mpp * MASK_PX, mpp_y=mpp * MASK_PX)
    with OpenSlide(str(path)) as check:
        read = np.asarray(check.read_region((0, 0), 0, check.level_dimensions[0]))[..., 0]
    if not np.array_equal(read, mask):
        raise RuntimeError(f"{path}: OpenSlide reads the mask back wrong")


def process(slide_id: str) -> dict:
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        width, height = slide.level_dimensions[0]
        mpp = float(slide.properties["openslide.mpp-x"])
    features = np.load(STAIN / "objects_original" / f"{slide_id}.npz")
    rbc = red_blood_cells(features)
    grid_shape = (-(-height // CELL_PX), -(-width // CELL_PX))
    regions = blood_regions(features["centroid"], rbc, grid_shape)
    inside = in_regions(features["centroid"], regions)
    if regions.any():
        write_mask(regions, OUT / "masks" / f"{slide_id}.tiff", mpp)
    return {
        "slide_id": slide_id,
        "objects": len(rbc),
        "rbc": int(rbc.sum()),
        "rbc_in_regions": int((rbc & inside).sum()),
        "other_in_regions": int((~rbc & inside).sum()),
        "region_mm2": float(regions.sum() * (CELL_PX * mpp / 1000) ** 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    slides = sorted(p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*"))
    with Pool(args.workers, maxtasksperchild=50) as pool:
        rows = pool.map(process, slides, chunksize=4)
    table = pd.DataFrame(rows).sort_values("slide_id")
    table.to_csv(OUT / "slides.csv", index=False)
    print(
        f"{(table.region_mm2 > 0).sum()} of {len(table)} slides have blood regions, "
        f"{table.region_mm2.sum():.1f} mm² in all; RBC objects {table.rbc.sum():,}, "
        f"of which in regions {table.rbc_in_regions.sum():,}"
    )


if __name__ == "__main__":
    main()
