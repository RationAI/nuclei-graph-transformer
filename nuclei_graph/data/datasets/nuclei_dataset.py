import heapq
import math
from random import choice, randrange, uniform

import numpy as np
import pandas as pd
import torch
from degraph import build_spatial_graph
from einops import rearrange
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.spatial import KDTree
from torch import Tensor
from torch.utils.data import Dataset

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.data.supervision import DatasetSupervision, NucleiSupervision
from nuclei_graph.nuclei_graph_typing import Crop, PredictSlide, Targets


type PriorityQueueItem = tuple[float, int]  # (cost, node_idx)
type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]

type Coords = NDArray[np.float32]


class NucleiDataset(Dataset[Crop | PredictSlide]):
    """Dataset for nuclei point clouds from whole-slide images."""

    def __init__(
        self,
        slides: DataFrame,
        supervision: DatasetSupervision | None,
        crop_size: int = 4096,
        crop_pos_thr: float | None = 0.75,
        alpha: float = 0.8,
        efd_order: int = 10,
        full_slide: bool = False,
        predict: bool = False,
        mil: bool = False,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: DataFrame with columns: "slide_id" (str), "is_carcinoma" (bool), "slide_nuclei_path" (str), "mpp_x" (float)
                and "mpp_y" (float). If the predict mode is set to `True` then it should also have a column "slide_path" (str).
            supervision: DatasetSupervision dataclass containing slide-level and nucleus-level labels for positive slides.
                It can be set to None if `predict` is True.
            crop_size: Number of nuclei in a crop (sample) during training.
            crop_pos_thr: Minimum ratio of positive nuclei in the crop to consider it as a valid positive sample during training.
            alpha: Weight between graph edge distance and Euclidean distance when selecting neighbors during graph creation.
            efd_order: Order of the elliptic Fourier descriptors used for nuclei shape representation.
            full_slide: Whether the dataset is used for full slide inference.
            predict: Whether to return the metadata needed for prediction along with the data.
            mil: whether to also return slide-level labels for multiple-instance learning along with the nucleus-level labels.
        """
        assert not mil or (mil and crop_pos_thr is not None)
        assert (supervision is not None) or predict, (
            "Supervision can be None only in predict mode."
        )
        self.slides = slides
        self.supervision = supervision
        self.crop_size = crop_size
        self.crop_pos_thr = crop_pos_thr
        self.alpha = alpha
        self.efd_order = efd_order
        self.full_slide = full_slide
        self.predict = predict
        self.mil = mil
        self.pos_slide_indices = (
            np.where(self.slides["is_carcinoma"])[0].tolist() if not predict else []
        )

    def __len__(self) -> int:
        return len(self.slides)

    def find_component(
        self,
        seed_idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: Coords,
        allowed_indices: NDArray[np.int64] | None = None,
    ) -> list[int]:
        """Grows a connected component of up to `k` nuclei starting from a seed index.

        Args:
            seed_idx: seed nucleus index
            k: maximum number of nuclei to include in the component.
            graph: adjacency list of the nuclei graph.
            centroids: array of nuclei coordinates.
            allowed_indices: optional array of allowed nuclei indices

        Returns:
            list[int]: Indices of nuclei in the component.

        Source: Nuclei Foundational Model repository.
        """
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
        self, centroids: Coords, valid_seeds: list[int]
    ) -> NDArray[np.int64]:
        """Selects nuclei indices for a crop by growing a connected component on the spatial graph.

        If `full_slide` is True, returns all nuclei indices. Otherwise, a random valid seed is chosen
        and a component of nuclei is grown based on the spatial graph.

        Args:
            centroids (np.ndarray[float], shape (n, 2)): Nuclei coordinates.
            valid_seeds (list[int]): Indices eligible as seeds for component sampling.

        Returns:
            np.ndarray: Selected nuclei indices for the crop.
        """
        n = len(centroids)
        if self.full_slide:
            return np.arange(n, dtype=int)

        center_idx = choice(valid_seeds) if valid_seeds else randrange(n)
        center_coords = centroids[center_idx]
        keep_indices = np.arange(len(centroids))

        # heuristically limit the nuclei for graph building
        limit = int(self.crop_size / max(1.0 - self.alpha, 1e-4))
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
        local_crop_indices = self.find_component(seed, self.crop_size, graph, centroids)

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

    def sample_positive_crop(
        self,
        valid_seeds: list[int],
        centroids: NDArray[np.float32],
        targets: Tensor,
        max_attempts: int = 10,
        margin: float = 0.15,
    ) -> NDArray[np.int64] | None:
        """Attempts to sample a positive crop of nuclei around a random valid seed.

        Ensures it contains a fraction of positive nuclei above `self.crop_pos_thr`.
        """
        assert self.crop_pos_thr is not None
        tree = KDTree(centroids)

        for _ in range(max_attempts):
            seed_idx = choice(valid_seeds)

            # heuristic via Euclidean circle
            _, neighbor_idx = tree.query(centroids[seed_idx], k=self.crop_size)
            est_tumor_ratio = (targets[neighbor_idx] == 1).sum().item() / self.crop_size
            if est_tumor_ratio < max(0.0, self.crop_pos_thr - margin):
                continue

            crop_indices = self.get_crop_indices(centroids, [seed_idx])
            crop_targets = targets[torch.from_numpy(crop_indices).long()]
            tumor_ratio = (crop_targets == 1).sum().item() / len(crop_indices)

            if tumor_ratio > self.crop_pos_thr:
                return crop_indices

        return None

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

    def __getitem__(self, idx: int) -> Crop | PredictSlide:
        slide = self.slides.iloc[idx]
        nuclei = self.get_nuclei(slide.slide_nuclei_path)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

        if self.predict:  # no supervision
            crop_indices = np.arange(len(nuclei), dtype=int)  # full-slide
            crop_polygons = np.array(nuclei["polygon"].iloc[crop_indices].tolist())
            crop_pos = centroids[crop_indices]
            crop_features = self.get_features(crop_polygons, slide.mpp_x, slide.mpp_y)

            return PredictSlide(
                slide={
                    "features": crop_features,
                    "pos": (crop_pos - crop_pos.mean(axis=0)).astype(np.float32),
                    "labels": {"nuclei": None, "graph": None},
                    "sup_mask": torch.ones(len(crop_indices), dtype=torch.bool),
                    "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                },
                metadata={
                    "slide_id": slide.slide_id,
                    "slide_path": slide.slide_path,
                    "slide_nuclei_path": slide.slide_nuclei_path,
                    "nuclei_ids": nuclei["id"].to_numpy(),
                },
            )

        crop_indices = np.arange(len(nuclei), dtype=int)
        nuclei_sup = self.get_nuclei_sup(slide.slide_id)

        # Generate a crop
        if not self.full_slide and not slide.is_carcinoma:
            crop_indices = self.get_crop_indices(
                centroids, nuclei_sup.get_neg_seeds(len(nuclei))
            )
        if not self.full_slide and slide.is_carcinoma:
            assert self.crop_pos_thr is not None

            # Target crop distribution:
            # 50% positive, 25% negative (from pos slides), 25% negative (from neg slides).
            # Assuming the slide sampler yields 75% positive slides, we select a positive
            # seed 2/3 of the time.
            if not self.mil:
                valid_seeds = (
                    nuclei_sup.get_pos_seeds(len(nuclei))
                    if torch.rand(1).item() < (2.0 / 3.0)
                    else nuclei_sup.get_neg_seeds(len(nuclei))
                )
                crop_indices = self.get_crop_indices(centroids, valid_seeds)

            curr_slide_idx, curr_crop_indices = None, None
            while self.mil:  # ensure crop positivity ≥ `crop_pos_thr`
                if curr_slide_idx is not None:
                    slide = self.slides.iloc[curr_slide_idx]
                    nuclei = self.get_nuclei(slide.slide_nuclei_path)
                    nuclei_sup = self.get_nuclei_sup(slide.slide_id)

                curr_crop_indices = self.sample_positive_crop(
                    valid_seeds=nuclei_sup.get_pos_seeds(len(nuclei)),
                    centroids=self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y),
                    targets=nuclei_sup.get_targets(len(nuclei)),
                )
                if curr_crop_indices is not None:
                    crop_indices = curr_crop_indices
                    break
                curr_slide_idx = choice(self.pos_slide_indices)

        assert crop_indices is not None
        crop_polygons = np.array(nuclei["polygon"].iloc[crop_indices].tolist())
        crop_pos = centroids[crop_indices]

        crop_indices_t = torch.from_numpy(crop_indices).long()
        crop_nuclei_labels = nuclei_sup.get_targets(len(nuclei))[crop_indices_t]
        crop_labels: Targets = {"nuclei": crop_nuclei_labels, "graph": None}

        if self.mil:
            graph_label = float(slide.is_carcinoma)
            crop_labels["graph"] = torch.tensor([graph_label], dtype=torch.float32)

        crop_features = self.get_features(crop_polygons, slide.mpp_x, slide.mpp_y)
        crop_pos_centered = (crop_pos - crop_pos.mean(axis=0)).astype(np.float32)

        if not self.predict and not self.full_slide:
            pos_rot, cos_rot, sin_rot = self.random_rotate_graph(
                crop_pos_centered, crop_features[..., -2], crop_features[..., -1]
            )
            crop_pos_centered = pos_rot
            crop_features[..., -2] = cos_rot
            crop_features[..., -1] = sin_rot

        return Crop(
            {
                "features": crop_features,
                "labels": crop_labels,
                "pos": crop_pos_centered,
                "sup_mask": nuclei_sup.get_sup_mask(len(nuclei))[crop_indices_t],
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
            }
        )
