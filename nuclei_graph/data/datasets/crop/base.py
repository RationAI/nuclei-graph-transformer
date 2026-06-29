import heapq
import math
from abc import ABC, abstractmethod
from random import choice, randint, randrange, uniform

import numpy as np
import pandas as pd
import torch
from degraph import build_spatial_graph
from einops import rearrange
from numpy.typing import NDArray
from pandas import DataFrame
from ratiopath.openslide import OpenSlide
from torch import Tensor
from torch.utils.data import Dataset

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.data.supervision import DatasetSupervision, NucleiSupervision
from nuclei_graph.nuclei_graph_typing import Sample


MAX_CROP_PATCH_SIDE = 8192


type PriorityQueueItem = tuple[float, int]  # (cost, node_idx)
type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]

type Coords = NDArray[np.float32]


class BaseCropDataset(Dataset[Sample], ABC):
    def __init__(
        self,
        metadata: DataFrame,
        supervision: DatasetSupervision | None = None,
        crop_size_min: int | None = None,
        crop_size_max: int | None = None,
        alpha: float = 0.8,
        efd_order: int = 16,
        full_slide: bool = False,
        random_rotate: bool = False,
        patch_size: int | None = None,
    ) -> None:
        self.metadata = metadata
        self.supervision = supervision
        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max
        self.alpha = alpha
        self.efd_order = efd_order
        self.full_slide = full_slide
        self.random_rotate = random_rotate
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.metadata)

    def find_component(
        self,
        seed_idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: Coords,
        allowed_indices: NDArray[np.int64] | None = None,
    ) -> list[int]:
        """Grows a connected component of up to `k` nuclei starting from a seed index."""
        component_indices: list[int] = []
        visited = np.zeros(len(centroids), dtype=bool)
        allowed_set = set(allowed_indices) if allowed_indices is not None else None

        pq: list[PriorityQueueItem] = []
        heapq.heappush(pq, (0.0, seed_idx))
        start_point_coords = centroids[seed_idx]

        while pq and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)
            if visited[current_idx]:
                continue

            visited[current_idx] = True
            component_indices.append(current_idx)

            for n_idx, edge_dist in graph[current_idx]:
                if not visited[n_idx] and (allowed_set is None or n_idx in allowed_set):
                    start_dist = np.linalg.norm(centroids[n_idx] - start_point_coords)
                    cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist
                    heapq.heappush(pq, (cost, n_idx))  # type: ignore[misc]
        return component_indices

    def get_crop_indices(
        self, centroids: Coords, valid_seeds: list[int], target_size: int
    ) -> NDArray[np.int64]:
        """Selects nuclei indices for a crop by growing a connected component on the spatial graph."""
        n = len(centroids)
        if self.full_slide:
            return np.arange(n, dtype=int)

        center_idx = choice(valid_seeds) if valid_seeds else randrange(n)
        center_coords = centroids[center_idx]
        keep_indices = np.arange(len(centroids))

        # heuristically limit the nuclei for graph building
        limit = int(target_size / max(1.0 - self.alpha, 1e-4))
        if n > limit:
            dists = np.linalg.norm(centroids - center_coords, axis=1)
            keep_indices = np.argpartition(dists, limit - 1)[:limit]

        # drop overlapping nuclei to prevent issues with graph construction
        quantized = np.round(centroids[keep_indices] / 1e-4).astype(np.int64)
        _, unique_local_indices = np.unique(quantized, axis=0, return_index=True)
        keep_indices = keep_indices[np.sort(unique_local_indices)]

        # build Delaunay graph
        centroids = centroids[keep_indices]
        graph = build_spatial_graph(centroids)

        # grow a connected component starting from the center nucleus
        seed = int(np.argmin(np.linalg.norm(centroids - center_coords, axis=1)))
        local_crop_indices = self.find_component(seed, target_size, graph, centroids)

        global_crop_indices = keep_indices[local_crop_indices]
        return np.array(global_crop_indices, dtype=np.int64)

    def get_features(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> NDArray[np.float32]:
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        contours = rearrange(polygons, "b (v d) -> b v d", d=2) * mpps
        efds = elliptic_fourier_descriptors(contours.astype(np.float64), self.efd_order)

        efds, angles = normalize_efd_for_rotation(efds)
        cos_angles = np.cos(2.0 * angles)
        sin_angles = np.sin(2.0 * angles)

        efds, scales = normalize_efd_for_scale(efds)
        log_scales = np.log(scales + 1e-6)

        efds = rearrange(efds, "n order c -> n (order c)")
        features = np.concatenate([efds, log_scales, cos_angles, sin_angles], axis=-1)
        return features.astype(np.float32)

    def random_rotate_graph(
        self,
        pos: NDArray[np.float32],
        cos_angles: NDArray[np.float32],
        sin_angles: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        theta = uniform(0, 2 * math.pi)

        rotation_matrix = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float32,
        )
        rotated_pos = pos @ rotation_matrix.T

        # the original angles are doubled
        c2 = math.cos(2 * theta)
        s2 = math.sin(2 * theta)

        rotated_cos = (cos_angles * c2 - sin_angles * s2).astype(np.float32)
        rotated_sin = (sin_angles * c2 + cos_angles * s2).astype(np.float32)

        return rotated_pos, rotated_cos, rotated_sin

    def get_nuclei(self, nuclei_path: str) -> pd.DataFrame:
        nuclei = pd.read_parquet(nuclei_path)
        return nuclei.sort_values("id").reset_index(drop=True)

    def get_centroids(
        self, nuclei: pd.DataFrame, mpp_x: float, mpp_y: float
    ) -> np.ndarray:
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        return np.stack(nuclei["centroid"].tolist()) * mpps

    def clip_box(
        self, lx: int, ly: int, rx: int, ry: int, wsi_w: int, wsi_h: int
    ) -> tuple[int, int, int, int]:
        """Clips a `[lx, rx) x [ly, ry)` box to the slide bounds.

        Returns `(read_x, read_y, read_w, read_h)`; `read_w`/`read_h` are <= 0 when the box
        doesn't overlap the slide at all.
        """
        read_x, read_y = max(0, lx), max(0, ly)
        read_w = min(wsi_w, rx) - read_x
        read_h = min(wsi_h, ry) - read_y
        return read_x, read_y, read_w, read_h

    def get_nuclei_bboxes(
        self, nuclei: pd.DataFrame, slide_path: str, crop_indices: NDArray[np.int64]
    ) -> Tensor | None:
        """Extracts a fixed-size RGB patch from the WSI around each cropped nucleus's centroid.

        Returns None when `patch_size` is not configured (bbox extraction disabled). When the
        cropped nuclei span a small enough area (e.g. a local training/eval crop, where they
        come from a spatially-clustered connected component), every patch is sliced out of a
        single `read_region` call covering their union (the "crop patch"). For more spread-out
        cases (e.g. a full-slide pass) it falls back to one `read_region` call per nucleus,
        since reading the union could otherwise try to decode too large an area at once.
        """
        if self.patch_size is None:
            return None

        raw_centroids = self.get_centroids(nuclei, 1.0, 1.0)[crop_indices]
        half_patch = self.patch_size // 2

        lx = raw_centroids[:, 0].astype(np.int64) - half_patch
        ly = raw_centroids[:, 1].astype(np.int64) - half_patch
        rx, ry = lx + self.patch_size, ly + self.patch_size

        with OpenSlide(slide_path) as wsi:
            wsi_w, wsi_h = wsi.dimensions

            union_w = int(rx.max() - lx.min())
            union_h = int(ry.max() - ly.min())
            use_crop_patch = max(union_w, union_h) <= MAX_CROP_PATCH_SIDE

            crop_patch, crop_patch_x, crop_patch_y = None, 0, 0
            if use_crop_patch:
                crop_patch_x, crop_patch_y, crop_patch_w, crop_patch_h = self.clip_box(
                    int(lx.min()),
                    int(ly.min()),
                    int(rx.max()),
                    int(ry.max()),
                    wsi_w,
                    wsi_h,
                )
                crop_patch = (
                    np.array(
                        wsi.read_region(
                            (crop_patch_x, crop_patch_y),
                            0,
                            (crop_patch_w, crop_patch_h),
                        ).convert("RGB")
                    )
                    if crop_patch_w > 0 and crop_patch_h > 0
                    else np.zeros((0, 0, 3), dtype=np.uint8)
                )

            bboxes = []
            for i in range(len(raw_centroids)):
                canvas = np.full(
                    (self.patch_size, self.patch_size, 3), 255, dtype=np.uint8
                )
                read_x, read_y, read_w, read_h = self.clip_box(
                    int(lx[i]), int(ly[i]), int(rx[i]), int(ry[i]), wsi_w, wsi_h
                )

                if read_w > 0 and read_h > 0:
                    if use_crop_patch:
                        assert crop_patch is not None
                        src_x = read_x - crop_patch_x
                        src_y = read_y - crop_patch_y
                        patch = crop_patch[
                            src_y : src_y + read_h, src_x : src_x + read_w
                        ]
                    else:
                        patch = np.array(
                            wsi.read_region(
                                (read_x, read_y), 0, (read_w, read_h)
                            ).convert("RGB")
                        )

                    canvas_x, canvas_y = read_x - int(lx[i]), read_y - int(ly[i])
                    canvas[
                        canvas_y : canvas_y + read_h, canvas_x : canvas_x + read_w
                    ] = patch

                bboxes.append(canvas)

        return torch.from_numpy(np.stack(bboxes)).permute(0, 3, 1, 2)

    def get_nuclei_sup(self, slide_id: str) -> NucleiSupervision:
        assert self.supervision is not None
        return self.supervision.supervision_map[slide_id].nuclei_supervision

    def get_crop_size(self, n: int) -> int:
        assert self.crop_size_min is not None and self.crop_size_max is not None
        return min(randint(self.crop_size_min, self.crop_size_max), n)

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        raise NotImplementedError
