import heapq
import math
from abc import ABC, abstractmethod
from random import choice, randint, randrange, uniform
from typing import NamedTuple, cast

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
from nuclei_graph.nuclei_graph_typing import EMBEDDING_MODES, Sample


MAX_CROP_PATCH_SIDE = 8192


type PriorityQueueItem = tuple[float, int]  # (cost, node_idx)
type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]

type Coords = NDArray[np.float32]

type GridCell = tuple[int, int]  # (grid_x, grid_y)
type CellMap = dict[GridCell, list[int]]  # grid cell -> list of nucleus indices


class Box(NamedTuple):
    """A `[lx, rx) x [ly, ry)` region in slide-pixel coordinates."""

    lx: int
    ly: int
    rx: int
    ry: int

    @property
    def w(self) -> int:
        return self.rx - self.lx

    @property
    def h(self) -> int:
        return self.ry - self.ly


class SlideSize(NamedTuple):
    w: int
    h: int


class DecodedRegion(NamedTuple):
    """A `read_region` result and the slide-pixel coordinates it was read from."""

    array: NDArray[np.uint8]
    origin_x: int
    origin_y: int


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
        embedding_mode: str = "efd",
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
        self.embedding_mode = embedding_mode

        assert self.embedding_mode in EMBEDDING_MODES, (
            f"Invalid embedding_mode: {self.embedding_mode}"
        )

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

    def get_efd_features(
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
        self, nuclei: pd.DataFrame, slide_path: str, crop_indices: NDArray[np.int64]
    ) -> Tensor | None:
        """Extracts a fixed-size RGB patch from the WSI around each cropped nucleus's centroid."""
        if self.patch_size is None:
            return None

        raw_centroids = self.get_centroids(nuclei, 1.0, 1.0)[crop_indices]
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

                cells: CellMap = {}
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
            filled_bboxes = cast("list[NDArray[np.uint8]]", bboxes)

        return torch.from_numpy(np.stack(filled_bboxes)).permute(0, 3, 1, 2)

    def get_nuclei_sup(self, slide_id: str) -> NucleiSupervision:
        assert self.supervision is not None
        return self.supervision.supervision_map[slide_id].nuclei_supervision

    def get_crop_size(self, n: int) -> int:
        assert self.crop_size_min is not None and self.crop_size_max is not None
        return min(randint(self.crop_size_min, self.crop_size_max), n)

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        raise NotImplementedError
