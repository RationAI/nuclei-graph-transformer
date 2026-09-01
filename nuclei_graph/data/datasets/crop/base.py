import heapq
from abc import ABC, abstractmethod
from collections.abc import Callable
from random import choice, randint, randrange

import numpy as np
import pandas as pd
from degraph import build_spatial_graph
from einops import rearrange
from numpy.typing import NDArray
from pandas import DataFrame
from torch.utils.data import Dataset

from nuclei_graph.data.datasets.nuclei_features import NucleiFeatureExtractor
from nuclei_graph.data.supervision import DatasetSupervision, NucleiSupervision
from nuclei_graph.nuclei_graph_typing import EMBEDDING_MODES, Sample


type PriorityQueueItem = tuple[float, int]  # (cost, node_idx)
type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]

type Coords = NDArray[np.float32]

type GridCell = tuple[int, int]  # (grid_x, grid_y)
type CellMap = dict[GridCell, list[int]]  # grid cell -> list of nucleus indices


class BaseCropDataset(NucleiFeatureExtractor, Dataset[Sample], ABC):
    def __init__(
        self,
        metadata: DataFrame,
        embedding_mode: str,
        supervision: DatasetSupervision | None = None,
        crop_size_min: int | None = None,
        crop_size_max: int | None = None,
        alpha: float = 0.8,
        efd_order: int = 16,
        full_slide: bool = False,
        patch_size: int | None = None,
        augmentations: Callable[..., dict[str, NDArray[np.float32]]] | None = None,
    ) -> None:
        self.metadata = metadata
        self.supervision = supervision
        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max
        self.alpha = alpha
        self.efd_order = efd_order
        self.full_slide = full_slide
        self.patch_size = patch_size
        self.embedding_mode = embedding_mode
        self.augmentations = augmentations

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

    def process_geometry(
        self,
        nuclei: pd.DataFrame,
        crop_indices: NDArray[np.int64],
        centroids: NDArray[np.float32],
        slide: pd.Series,
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.float32], NDArray[np.float32]]:
        """Extracts polygons, applies augmentations, and calculates localized node positions."""
        crop_polygons = None
        if self.augmentations is not None or self.embedding_mode in [
            "efd",
            "efd_spatial",
        ]:
            crop_polygons = np.array(nuclei["polygon"].iloc[crop_indices].tolist())
            crop_polygons = rearrange(crop_polygons, "n (v c) -> n v c", c=2)

        if self.augmentations is not None:
            mpps = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
            crop_polygons = self.augmentations(polygons=crop_polygons)["polygons"]
            crop_pos = crop_polygons.mean(axis=1) * mpps
        else:
            crop_pos = centroids[crop_indices]

        crop_pos_centered = (crop_pos - crop_pos.mean(axis=0)).astype(np.float32)
        return crop_polygons, crop_pos, crop_pos_centered

    def generate_embeddings(
        self,
        polygons: NDArray[np.float32] | None,
        centroids: NDArray[np.float32],
        slide: pd.Series,
        true_centroids: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
        """Generates geometric features or bounding boxes based on the selected embedding mode.

        `centroids` may reflect augmented (e.g. rotated/jittered) geometry and drives the
        `efd`/`spatial` feature branches. `true_centroids` are the real slide positions
        and must be used for `bbox` extraction, since the WSI pixels themselves
        are never transformed by the geometric augmentations.
        """
        geom_features, bboxes = None, None

        if self.embedding_mode == "efd":
            geom_features = self.get_efd_features(polygons, slide.mpp_x, slide.mpp_y)
        elif self.embedding_mode == "efd_spatial":
            efd_feats = self.get_efd_features(polygons, slide.mpp_x, slide.mpp_y)
            spatial_feats = self.get_spatial_features(centroids)

            efd_to_norm = efd_feats[..., :-2]
            angles = efd_feats[..., -2:]

            geom_features = np.concatenate(
                [efd_to_norm, spatial_feats, angles], axis=-1
            )
        elif self.embedding_mode == "spatial":
            geom_features = self.get_spatial_features(centroids)
        elif self.embedding_mode == "blank":
            geom_features = np.zeros((len(centroids), 1), dtype=np.float32)
        elif self.embedding_mode == "bbox":
            bboxes = self.get_nuclei_bboxes(
                true_centroids, slide.slide_path, slide.mpp_x, slide.mpp_y
            )
        assert geom_features is not None or bboxes is not None
        return geom_features, bboxes

    def get_nuclei(self, nuclei_path: str) -> pd.DataFrame:
        nuclei = pd.read_parquet(nuclei_path)
        return nuclei.sort_values("id").reset_index(drop=True)

    def get_centroids(
        self, nuclei: pd.DataFrame, mpp_x: float, mpp_y: float
    ) -> np.ndarray:
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        return np.stack(nuclei["centroid"].tolist()) * mpps

    def get_nuclei_sup(self, slide_id: str) -> NucleiSupervision:
        assert self.supervision is not None
        return self.supervision.supervision_map[slide_id].nuclei_supervision

    def get_crop_size(self, n: int) -> int:
        assert self.crop_size_min is not None and self.crop_size_max is not None
        return min(randint(self.crop_size_min, self.crop_size_max), n)

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        raise NotImplementedError
