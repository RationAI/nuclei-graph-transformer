import json
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.nuclei_graph_typing import (
    BBOX_MASK_MARGIN_UM,
    BBOX_ROTATION_DEG,
    MAX_CROP_PATCH_SIDE,
    TARGET_BBOX_CONTEXT_UM,
    Box,
    DecodedRegion,
    SlideSize,
)


# layout of the patches precomputed by `preprocessing/nuclei_patches.py`
PATCHES_FILE = "patches.npy"
ROWS_FILE = "rows.npy"
PARAMS_FILE = "params.json"


def patches_params(patch_size: int) -> dict[str, float]:
    """The patch geometry that the precomputed patches must have been extracted with."""
    return {
        "patch_size": patch_size,
        "target_bbox_context_um": TARGET_BBOX_CONTEXT_UM,
        "bbox_mask_margin_um": BBOX_MASK_MARGIN_UM,
        "bbox_rotation_deg": BBOX_ROTATION_DEG,
    }


class NucleiFeatureExtractor:
    """Shared spatial, geometric, and CNN patch extraction methods for Nuclei Datasets."""

    efd_order: int
    patch_size: int | None
    patches_path: str | None = None

    def check_patches_params(self) -> None:
        """Asserts that the precomputed patches match the current patch geometry."""
        assert self.patches_path is not None and self.patch_size is not None
        params_path = Path(self.patches_path) / PARAMS_FILE
        stored = json.loads(params_path.read_text())
        expected = patches_params(self.patch_size)
        assert stored == expected, (
            f"Patches in {self.patches_path} were extracted with {stored}, "
            f"not {expected}; rerun `preprocessing/nuclei_patches.py`."
        )

    def read_nuclei_bboxes(
        self, slide_id: str, indices: NDArray[np.int64], n_nuclei: int
    ) -> torch.Tensor:
        """Reads the precomputed patches of the nuclei at `indices` (`get_nuclei` order).

        The rows are read in their stored (spatial) order, so a spatially contiguous
        crop reads mostly contiguous parts of the file.
        """
        assert self.patches_path is not None
        slide_dir = Path(self.patches_path) / f"slide_id={slide_id}"
        rows = np.load(slide_dir / ROWS_FILE)
        assert len(rows) == n_nuclei, (
            f"{slide_dir} holds {len(rows)} nuclei, the slide has {n_nuclei}."
        )
        rows = rows[indices]
        order = np.argsort(rows)
        patches = np.load(slide_dir / PATCHES_FILE, mmap_mode="r")
        bboxes = np.empty((len(rows), *patches.shape[1:]), dtype=patches.dtype)
        bboxes[order] = patches[rows[order]]
        return torch.from_numpy(bboxes)

    def get_efd_features(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> NDArray[np.float32]:
        """Computes Elliptic Fourier Descriptors for a set of nuclei polygons."""
        contours = polygons * np.array([mpp_x, mpp_y], dtype=np.float32)

        efds = elliptic_fourier_descriptors(contours.astype(np.float64), self.efd_order)

        efds, angles = normalize_efd_for_rotation(efds)
        cos_angles, sin_angles = np.cos(2.0 * angles), np.sin(2.0 * angles)

        efds, scales = normalize_efd_for_scale(efds)
        log_scales = np.log(scales + 1e-6)

        efds = rearrange(efds, "n order c -> n (order c)")
        features = np.concatenate([efds, log_scales, cos_angles, sin_angles], axis=-1)
        return features.astype(np.float32)

    def get_orientations(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> NDArray[np.float64]:
        """Orientation ψ of each nucleus's first EFD harmonic (as in `get_efd_features`).

        ψ depends only on the first harmonic, so higher orders are not computed.
        """
        contours = polygons * np.array([mpp_x, mpp_y], dtype=np.float32)
        efds = elliptic_fourier_descriptors(contours.astype(np.float64), order=1)
        _, angles = normalize_efd_for_rotation(efds)
        return np.nan_to_num(angles[:, 0])

    def get_orientation_features(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> NDArray[np.float32]:
        """(cos 2ψ, sin 2ψ) of each nucleus, encoded as the angles in `get_efd_features`."""
        psis = self.get_orientations(polygons, mpp_x, mpp_y)[:, None]
        features = np.concatenate([np.cos(2.0 * psis), np.sin(2.0 * psis)], axis=-1)
        return features.astype(np.float32)

    def clip_box(self, box: Box, slide_size: SlideSize) -> Box:
        """Clips `box` to the slide bounds.

        The result has `w`/`h` <= 0 when `box` doesn't overlap the slide at all.
        """
        read_x, read_y = max(0, box.lx), max(0, box.ly)
        read_rx = min(slide_size.w, box.rx)
        read_ry = min(slide_size.h, box.ry)
        return Box(read_x, read_y, read_rx, read_ry)

    def read_region(self, wsi: OpenSlide, box: Box) -> DecodedRegion:
        """Reads and RGB-converts `box`; the array is empty if `box.w`/`box.h` <= 0."""
        if box.w <= 0 or box.h <= 0:
            return DecodedRegion(np.zeros((0, 0, 3), dtype=np.uint8), box.lx, box.ly)
        array = np.array(
            wsi.read_region((box.lx, box.ly), 0, (box.w, box.h)).convert("RGB")
        )
        return DecodedRegion(array, box.lx, box.ly)

    def extract_patch(
        self, source: DecodedRegion, box: Box, slide_size: SlideSize
    ) -> NDArray[np.uint8]:
        """Slices a single nucleus's `box` patch out of `source`."""
        canvas = np.full((box.h, box.w, 3), 255, dtype=np.uint8)
        clipped = self.clip_box(box, slide_size)

        if clipped.w > 0 and clipped.h > 0:
            src_x = clipped.lx - source.origin_x
            src_y = clipped.ly - source.origin_y
            patch = source.array[src_y : src_y + clipped.h, src_x : src_x + clipped.w]
            canvas_x, canvas_y = clipped.lx - box.lx, clipped.ly - box.ly
            canvas[canvas_y : canvas_y + clipped.h, canvas_x : canvas_x + clipped.w] = (
                patch
            )
        return canvas

    def nucleus_patch(
        self,
        window: NDArray[np.uint8],
        box: Box,
        centroid: NDArray[np.float32],
        polygon: NDArray[np.float32],
        psi: float,
        mpp_x: float,
        mpp_y: float,
    ) -> NDArray[np.uint8]:
        """Masks the nucleus in its read `window` and maps it to a `patch_size` patch.

        Pixels farther than `BBOX_MASK_MARGIN_UM` from the nucleus polygon are set to
        white (as outside the slide). One affine warp then rotates the nucleus by
        `BBOX_ROTATION_DEG` - ψ (first EFD harmonic along the patch diagonal),
        centre-crops `TARGET_BBOX_CONTEXT_UM` around the centroid, and resamples to
        `patch_size`, so the image is only interpolated once.
        """
        assert self.patch_size is not None
        origin = np.array([box.lx, box.ly], dtype=np.float32)

        mask = np.zeros(window.shape[:2], dtype=np.uint8)
        vertices = np.round((polygon - origin) * 16).astype(np.int32)  # 4-bit subpixel
        cv2.fillPoly(mask, [vertices], 255, lineType=cv2.LINE_8, shift=4)
        margin_px = max(1, round(BBOX_MASK_MARGIN_UM / min(mpp_x, mpp_y)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        mask = cv2.dilate(mask, kernel)

        # window px -> µm (D) -> rotated by -(ψ - BBOX_ROTATION_DEG) (R) -> patch px,
        # centroid to patch centre
        angle = psi - np.deg2rad(BBOX_ROTATION_DEG)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        linear = (
            (self.patch_size / TARGET_BBOX_CONTEXT_UM)
            * rotation
            @ np.diag([mpp_x, mpp_y])
        )
        centre = np.full(2, (self.patch_size - 1) / 2)
        offset = centre - linear @ (centroid - origin)
        affine = np.hstack([linear, offset[:, None]])

        size = (self.patch_size, self.patch_size)
        patch = cv2.warpAffine(
            window,
            affine,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        mask = cv2.warpAffine(
            mask,
            affine,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        patch[mask < 128] = 255
        return patch

    def get_nuclei_bboxes(
        self,
        centroids_px: NDArray[np.float32],
        polygons_px: NDArray[np.float32],
        slide_path: str,
        mpp_x: float,
        mpp_y: float,
    ) -> torch.Tensor | None:
        """Extracts a masked, orientation-normalized RGB patch of each nucleus.

        `centroids_px` (n, 2) and `polygons_px` (n, v, 2) are in level-0 pixels and must
        match the WSI pixels (i.e. not augmented). See `nucleus_patch`.
        """
        if self.patch_size is None:
            return None

        psis = self.get_orientations(polygons_px, mpp_x, mpp_y)

        # read √2 larger than the target so rotating leaves no empty corners
        read_um = TARGET_BBOX_CONTEXT_UM * np.sqrt(2)
        read_size_px_x = int(np.ceil(read_um / mpp_x)) + 2
        read_size_px_y = int(np.ceil(read_um / mpp_y)) + 2
        half_read_x = read_size_px_x // 2
        half_read_y = read_size_px_y // 2

        lx = centroids_px[:, 0].astype(np.int64) - half_read_x
        ly = centroids_px[:, 1].astype(np.int64) - half_read_y
        rx, ry = lx + read_size_px_x, ly + read_size_px_y

        with OpenSlide(slide_path) as wsi:
            slide_size = SlideSize(*wsi.dimensions)

            union_w = int(rx.max() - lx.min())
            union_h = int(ry.max() - ly.min())

            bboxes: list[NDArray[np.uint8] | None] = [None] * len(centroids_px)

            if max(union_w, union_h) <= MAX_CROP_PATCH_SIDE:
                union_box = self.clip_box(
                    Box(int(lx.min()), int(ly.min()), int(rx.max()), int(ry.max())),
                    slide_size,
                )
                source = self.read_region(wsi, union_box)
                for i in range(len(centroids_px)):
                    box = Box(int(lx[i]), int(ly[i]), int(rx[i]), int(ry[i]))
                    window = self.extract_patch(source, box, slide_size)
                    bboxes[i] = self.nucleus_patch(
                        window,
                        box,
                        centroids_px[i],
                        polygons_px[i],
                        psis[i],
                        mpp_x,
                        mpp_y,
                    )
            else:
                cell_size_x = MAX_CROP_PATCH_SIDE - read_size_px_x
                cell_size_y = MAX_CROP_PATCH_SIDE - read_size_px_y
                cell_x = centroids_px[:, 0].astype(np.int64) // cell_size_x
                cell_y = centroids_px[:, 1].astype(np.int64) // cell_size_y

                cells: dict[tuple[int, int], list[int]] = {}
                for i, (gx, gy) in enumerate(
                    zip(cell_x.tolist(), cell_y.tolist(), strict=True)
                ):
                    cells.setdefault((gx, gy), []).append(i)

                for (gx, gy), indices in cells.items():
                    cell_box = self.clip_box(
                        Box(
                            gx * cell_size_x - half_read_x,
                            gy * cell_size_y - half_read_y,
                            (gx + 1) * cell_size_x + half_read_x,
                            (gy + 1) * cell_size_y + half_read_y,
                        ),
                        slide_size,
                    )
                    source = self.read_region(wsi, cell_box)
                    for i in indices:
                        box = Box(int(lx[i]), int(ly[i]), int(rx[i]), int(ry[i]))
                        window = self.extract_patch(source, box, slide_size)
                        bboxes[i] = self.nucleus_patch(
                            window,
                            box,
                            centroids_px[i],
                            polygons_px[i],
                            psis[i],
                            mpp_x,
                            mpp_y,
                        )

        assert all(b is not None for b in bboxes)
        return torch.from_numpy(np.stack(bboxes)).permute(0, 3, 1, 2)
