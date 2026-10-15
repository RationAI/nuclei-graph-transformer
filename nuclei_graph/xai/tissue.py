"""Tissue-mask geometry for nuclei and analysis windows.

The masks are pyramidal 0/255 TIFFs aligned with level 0 of the WSI; a page with a
resolution close to `target_um_per_px` is used.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pyvips
from numpy.typing import NDArray


@dataclass(frozen=True)
class TissueMask:
    mask: NDArray[np.bool_]  # (height, width), True = tissue
    um_per_px: float

    @classmethod
    def read(cls, path: str, mpp: float, target_um_per_px: float) -> "TissueMask":
        """The coarsest pyramid page with a resolution not above `target_um_per_px`.

        Page 0 has the size of level 0 of the WSI, whose resolution is `mpp`.
        """
        level0 = pyvips.Image.new_from_file(path)
        n_pages, level0_width = level0.get_n_pages(), level0.width
        best = None
        for page in range(n_pages):
            image = pyvips.Image.new_from_file(path, page=page)
            um_per_px = mpp * level0_width / image.width
            if um_per_px <= target_um_per_px:
                best = (image, um_per_px)
        assert best is not None, f"No mask page at ≤ {target_um_per_px} µm/px: {path}"
        image, um_per_px = best
        array = image.numpy()
        return cls(array.reshape(array.shape[:2]) > 0, um_per_px)

    def boundary_distance_um(
        self, x_um: NDArray[np.floating], y_um: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Distance of each point to the nearest non-tissue pixel (0 outside tissue).

        The image border counts as non-tissue.
        """
        padded = np.pad(self.mask.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        col = np.clip(
            (x_um / self.um_per_px).astype(np.int64), 0, self.mask.shape[1] - 1
        )
        row = np.clip(
            (y_um / self.um_per_px).astype(np.int64), 0, self.mask.shape[0] - 1
        )
        return (distance[row, col] * self.um_per_px).astype(np.float32)

    def cell_fraction(self, cell_um: float) -> NDArray[np.float64]:
        """Tissue fraction of every cell of a `cell_um` grid, indexed [x cell, y cell].

        Cells start at the slide origin (cell i covers [i · cell_um, (i + 1) · cell_um));
        area outside the image counts as non-tissue.
        """
        rows, cols = np.indices(self.mask.shape)
        cx = ((cols + 0.5) * self.um_per_px // cell_um).astype(np.int64).ravel()
        cy = ((rows + 0.5) * self.um_per_px // cell_um).astype(np.int64).ravel()
        shape = (cx.max() + 1, cy.max() + 1)
        flat = np.ravel_multi_index((cx, cy), shape)
        tissue = np.bincount(flat, weights=self.mask.ravel(), minlength=np.prod(shape))
        pixels = np.bincount(flat, minlength=np.prod(shape))
        # tissue share of the cell's pixels, times the share of the cell's area that
        # lies inside the image (exact overlap of the cell with the image extent)
        height_um, width_um = np.array(self.mask.shape) * self.um_per_px
        inside_x = np.clip(width_um / cell_um - np.arange(shape[0]), 0.0, 1.0)
        inside_y = np.clip(height_um / cell_um - np.arange(shape[1]), 0.0, 1.0)
        share = (tissue / np.maximum(pixels, 1)).reshape(shape)
        return share * np.outer(inside_x, inside_y)
