import math
from random import uniform

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
        """Computes Elliptic Fourier Descriptors (EFDs) for a set of nuclei polygons."""
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
        """Computes explicit spatial/topological statistics for a spatial embedding."""
        n = len(pos)
        tree = KDTree(pos)

        dists_all, _ = tree.query(pos, k=6)
        dists = dists_all[:, [1, 3, 5]]

        mean_dist = np.mean(dists_all[:, 1:], axis=1, keepdims=True)
        std_dist = np.std(dists_all[:, 1:], axis=1, keepdims=True)

        count_r20 = np.array(
            [len(idx) - 1 for idx in tree.query_ball_point(pos, r=20)]
        )[..., None]
        count_r50 = np.array(
            [len(idx) - 1 for idx in tree.query_ball_point(pos, r=50)]
        )[..., None]

        tri = Delaunay(pos)
        indptr, _ = tri.vertex_neighbor_vertices
        degree = np.array([indptr[i + 1] - indptr[i] for i in range(n)])[..., None]

        features = np.column_stack(
            [dists, mean_dist, std_dist, count_r20, count_r50, degree]
        )
        return features.astype(np.float32)

    def random_rotate_graph(
        self,
        pos: NDArray[np.float32],
        cos_angles: NDArray[np.float32] | None,
        sin_angles: NDArray[np.float32] | None,
    ) -> tuple[
        NDArray[np.float32], NDArray[np.float32] | None, NDArray[np.float32] | None
    ]:
        theta = uniform(0, 2 * math.pi)

        rotation_matrix = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float32,
        )
        rotated_pos = pos @ rotation_matrix.T
        if cos_angles is None or sin_angles is None:
            return rotated_pos, None, None

        c2 = math.cos(2 * theta)
        s2 = math.sin(2 * theta)

        rotated_cos = (cos_angles * c2 - sin_angles * s2).astype(np.float32)
        rotated_sin = (sin_angles * c2 + cos_angles * s2).astype(np.float32)
        return rotated_pos, rotated_cos, rotated_sin

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
        """Slices a single nucleus's `box` patch out of `source`.

        `source` is an already-decoded region expected to fully cover the (slide-clipped)
        `box`. Out-of-slide area is left white.
        """
        assert self.patch_size is not None
        canvas = np.full((self.patch_size, self.patch_size, 3), 255, dtype=np.uint8)
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
        self, raw_centroids: NDArray[np.float32], slide_path: str
    ) -> torch.Tensor | None:
        """Extracts a fixed-size RGB patch from the WSI around each nucleus's centroid."""
        if self.patch_size is None:
            return None

        half_patch = self.patch_size // 2
        lx = raw_centroids[:, 0].astype(np.int64) - half_patch
        ly = raw_centroids[:, 1].astype(np.int64) - half_patch
        rx, ry = lx + self.patch_size, ly + self.patch_size

        with OpenSlide(slide_path) as wsi:
            slide_size = SlideSize(*wsi.dimensions)

            union_w = int(rx.max() - lx.min())
            union_h = int(ry.max() - ly.min())

            bboxes: list[NDArray[np.uint8] | None] = [None] * len(raw_centroids)
            if max(union_w, union_h) <= MAX_CROP_PATCH_SIDE:
                union_box = self.clip_box(
                    Box(int(lx.min()), int(ly.min()), int(rx.max()), int(ry.max())),
                    slide_size,
                )
                source = self.read_region(wsi, union_box)
                for i in range(len(raw_centroids)):
                    box = Box(int(lx[i]), int(ly[i]), int(rx[i]), int(ry[i]))
                    bboxes[i] = self.extract_patch(source, box, slide_size)
            else:
                cell_size = MAX_CROP_PATCH_SIDE - self.patch_size
                cell_x = raw_centroids[:, 0].astype(np.int64) // cell_size
                cell_y = raw_centroids[:, 1].astype(np.int64) // cell_size

                cells: dict[tuple[int, int], list[int]] = {}
                for i, (gx, gy) in enumerate(
                    zip(cell_x.tolist(), cell_y.tolist(), strict=True)
                ):
                    cells.setdefault((gx, gy), []).append(i)

                for (gx, gy), indices in cells.items():
                    cell_box = self.clip_box(
                        Box(
                            gx * cell_size - half_patch,
                            gy * cell_size - half_patch,
                            (gx + 1) * cell_size + half_patch,
                            (gy + 1) * cell_size + half_patch,
                        ),
                        slide_size,
                    )
                    source = self.read_region(wsi, cell_box)
                    for i in indices:
                        box = Box(int(lx[i]), int(ly[i]), int(rx[i]), int(ry[i]))
                        bboxes[i] = self.extract_patch(source, box, slide_size)

            assert all(b is not None for b in bboxes)

        return torch.from_numpy(np.stack(bboxes)).permute(0, 3, 1, 2)
