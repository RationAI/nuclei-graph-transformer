import math
from random import uniform

import cv2
import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide
from scipy.spatial import Delaunay, KDTree

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.nuclei_graph_typing import (
    MAX_CROP_PATCH_SIDE,
    TARGET_BBOX_CONTEXT_UM,
    Box,
    DecodedRegion,
    SlideSize,
)


class NucleiFeatureExtractor:
    """Shared spatial, geometric, and CNN patch extraction methods for Nuclei Datasets."""

    efd_order: int
    patch_size: int | None

    def get_efd_features(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> NDArray[np.float32]:
        """Computes Elliptic Fourier Descriptors for a set of nuclei polygons."""
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        contours = rearrange(polygons, "b (v d) -> b v d", d=2) * mpps
        efds = elliptic_fourier_descriptors(contours.astype(np.float64), self.efd_order)

        efds, angles = normalize_efd_for_rotation(efds)
        cos_angles, sin_angles = np.cos(2.0 * angles), np.sin(2.0 * angles)

        efds, scales = normalize_efd_for_scale(efds)
        log_scales = np.log(scales + 1e-6)

        efds = rearrange(efds, "n order c -> n (order c)")
        features = np.concatenate([efds, log_scales, cos_angles, sin_angles], axis=-1)
        return features.astype(np.float32)

    def get_spatial_features(self, pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """Computes explicit spatial statistics."""
        n = len(pos)
        tree = KDTree(pos)

        # 1. K-NN Distances
        dists_all, indices = tree.query(pos, k=6)
        d1 = dists_all[:, 1:2]
        d3 = dists_all[:, 3:4]
        d5 = dists_all[:, 5:6]

        d1_d3_ratio = d1 / (d3 + 1e-6)
        d1_d5_ratio = d1 / (d5 + 1e-6)

        mean_dist = np.mean(dists_all[:, 1:], axis=1, keepdims=True)
        std_dist = np.std(dists_all[:, 1:], axis=1, keepdims=True)
        cv_dist = std_dist / (mean_dist + 1e-6)  # Coefficient of Variation

        # 2. Angular Dispersion
        nn_indices = np.asarray(indices)[:, 1:]
        diffs = pos[nn_indices] - pos[:, None, :]
        angles = np.arctan2(diffs[..., 1], diffs[..., 0])
        mean_sin = np.mean(np.sin(angles), axis=1, keepdims=True)
        mean_cos = np.mean(np.cos(angles), axis=1, keepdims=True)
        angular_dispersion = 1.0 - np.sqrt(mean_sin**2 + mean_cos**2)

        # 3. Delaunay Degree
        tri = Delaunay(pos)
        indptr, _ = tri.vertex_neighbor_vertices
        degree = np.array([indptr[i + 1] - indptr[i] for i in range(n)])[..., None]

        return np.column_stack(
            [
                d1_d3_ratio,
                d1_d5_ratio,
                cv_dist,
                angular_dispersion,
                degree,
            ]
        ).astype(np.float32)

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

    def get_nuclei_bboxes(
        self,
        centroids: NDArray[np.float32],
        slide_path: str,
        mpp_x: float,
        mpp_y: float,
    ) -> torch.Tensor | None:
        """Extracts a fixed-size RGB patch from the WSI around each nucleus's centroid."""
        if self.patch_size is None:
            return None

        read_size_px_x = int(TARGET_BBOX_CONTEXT_UM / mpp_x)
        read_size_px_y = int(TARGET_BBOX_CONTEXT_UM / mpp_y)
        half_read_x = read_size_px_x // 2
        half_read_y = read_size_px_y // 2

        # Convert to Pixels
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        centroids_px = centroids / mpps
        lx = centroids_px[:, 0].astype(np.int64) - half_read_x
        ly = centroids_px[:, 1].astype(np.int64) - half_read_y
        rx, ry = lx + read_size_px_x, ly + read_size_px_y

        with OpenSlide(slide_path) as wsi:
            slide_size = SlideSize(*wsi.dimensions)

            union_w = int(rx.max() - lx.min())
            union_h = int(ry.max() - ly.min())

            bboxes: list[NDArray[np.uint8] | None] = [None] * len(centroids)

            if max(union_w, union_h) <= MAX_CROP_PATCH_SIDE:
                union_box = self.clip_box(
                    Box(int(lx.min()), int(ly.min()), int(rx.max()), int(ry.max())),
                    slide_size,
                )
                source = self.read_region(wsi, union_box)
                for i in range(len(centroids)):
                    box = Box(int(lx[i]), int(ly[i]), int(rx[i]), int(ry[i]))
                    raw_patch = self.extract_patch(source, box, slide_size)
                    bboxes[i] = cv2.resize(
                        raw_patch,
                        (self.patch_size, self.patch_size),
                        interpolation=cv2.INTER_LINEAR,
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
                        raw_patch = self.extract_patch(source, box, slide_size)
                        bboxes[i] = cv2.resize(
                            raw_patch,
                            (self.patch_size, self.patch_size),
                            interpolation=cv2.INTER_LINEAR,
                        )

        assert all(b is not None for b in bboxes)
        return torch.from_numpy(np.stack(bboxes)).permute(0, 3, 1, 2)
