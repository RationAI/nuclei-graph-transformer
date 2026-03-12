import heapq
from collections.abc import Iterable
from random import choice, randrange

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from degraph import build_spatial_graph
from einops import rearrange
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.spatial import KDTree
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import Dataset

from nuclei_graph.data import create_block_mask_from_kdtree
from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.data.supervision import DatasetSupervision, NucleiSupervision
from nuclei_graph.nuclei_graph_typing import Crop, Metadata, PredictSlide, Targets


type PriorityQueueItem = tuple[float, int]  # (cost, node_idx)
type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]

type Coords = NDArray[np.float32]


class NucleiDataset(Dataset[Crop | PredictSlide]):
    """Dataset for nuclei point clouds from whole-slide images."""

    def __init__(
        self,
        slides: DataFrame,
        supervision: DatasetSupervision,
        crop_size: int = 4096,
        crop_pos_thr: float | None = 0.75,
        alpha: float = 0.8,
        k: int = 64,
        attn_block_size: int = 128,
        efd_order: int = 10,
        symmetric_block_mask: bool = False,
        full_slide: bool = False,
        predict: bool = False,
        mil: bool = False,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: DataFrame with columns: "slide_id" (str), "is_carcinoma" (bool), "slide_nuclei_path" (str), "mpp_x" (float)
                and "mpp_y" (float). If the predict mode is set to `True` then it should also have a column "slide_path" (str).
            supervision: DatasetSupervision dataclass containing slide-level and nucleus-level labels for positive slides.
            crop_size: Number of nuclei in a crop (sample) during training.
            crop_pos_thr: Minimum ratio of positive nuclei in the crop to consider it as a valid positive sample during training.
            alpha: Weight between graph edge distance and Euclidean distance when selecting neighbors during graph creation.
            k: Number of neighbors for sparse attention.
            attn_block_size: Block size for sparse attention. It must hold that `crop_size` mod `attn_block_size` is 0.
            efd_order: Order of the elliptic Fourier descriptors used for nuclei shape representation.
            symmetric_block_mask: Whether to symmetrize the block mask. Defaults to False.
            full_slide: Whether the dataset is used for full slide inference.
            predict: Whether to return the metadata needed for prediction along with the data.
            mil: whether to return slide-level labels for multiple-instance learning. If False, nucleus-level labels are returned.
        """
        assert crop_size % attn_block_size == 0, (
            "`crop_size` must be divisible by `attn_block_size`."
        )
        self.slides = slides
        self.supervision = supervision
        self.crop_size = crop_size
        self.crop_pos_thr = crop_pos_thr
        self.alpha = alpha
        self.k = k
        self.attn_block_size = attn_block_size
        self.efd_order = efd_order
        self.symmetric_block_mask = symmetric_block_mask
        self.full_slide = full_slide
        self.predict = predict
        self.mil = mil
        self.pos_slide_indices = np.where(self.slides["is_carcinoma"])[0].tolist()

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

        Taken from the Nuclei Foundational Model repository.
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

    def pad_to_block_size(self, tensors: Iterable[Tensor]) -> list[Tensor]:
        """Pads tensors along dim-0 so that their size is divisible by `attn_block_size`.

        Padding is applied at the end and preserves all other dimensions.
        """
        tensors = list(tensors)

        t_size = tensors[0].size(0)
        assert all(t.size(0) == t_size for t in tensors)

        remainder = t_size % self.attn_block_size
        if remainder == 0:
            return tensors

        pad_len = self.attn_block_size - remainder
        return [
            F.pad(t, (0, pad_len))
            if t.dim() == 1
            else F.pad(t, (0, 0) * (t.dim() - 1) + (0, pad_len))
            for t in tensors
        ]

    def get_inverse_perm(self, perm: Tensor) -> Tensor:
        assert perm.dim() == 1
        perm_inverse = torch.empty_like(perm)
        perm_inverse[perm] = torch.arange(perm.size(0), dtype=perm.dtype)
        assert torch.equal(
            perm[perm_inverse], torch.arange(perm.size(0), dtype=perm.dtype)
        )
        return perm_inverse

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

        # compute crop indices by growing a connected component starting from the center nucleus
        seed = int(np.argmin(np.linalg.norm(centroids - center_coords, axis=1)))
        local_crop_indices = self.find_component(seed, self.crop_size, graph, centroids)

        global_crop_indices = keep_indices[local_crop_indices]
        return np.array(global_crop_indices, dtype=np.int64)

    def get_crop_features(
        self,
        nuclei: pd.DataFrame,
        centroids: np.ndarray,
        crop_indices: np.ndarray,
        mpp_x: float,
        mpp_y: float,
    ) -> tuple[Tensor, Tensor, BlockMask, np.ndarray]:
        crop_polygons = nuclei["polygon"].iloc[crop_indices].tolist()
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        contours = rearrange(crop_polygons, "b (v d) -> b v d", d=2) * mpps
        efds = elliptic_fourier_descriptors(contours.astype(np.float64), self.efd_order)

        efds, angles = normalize_efd_for_rotation(efds)
        # double so that θ and θ + π map to the same values (nuclei/ellipses are symmetric)
        cos_angles = np.cos(2.0 * angles)
        sin_angles = np.sin(2.0 * angles)

        efds, scales = normalize_efd_for_scale(efds)
        # nuclei scales have approximately log-normal distribution
        log_scales = np.log(scales + 1e-6)

        efds = rearrange(efds, "n order c -> n (order c)")
        x = np.concatenate([efds, log_scales, cos_angles, sin_angles], axis=-1)

        # prepare positions for RoPE
        crop_centroids = centroids[crop_indices]
        crop_pos_np = crop_centroids - crop_centroids.mean(axis=0)

        # optimize data layout for a block-sparse attention
        perm = KDTree(crop_centroids, leafsize=self.attn_block_size).indices
        crop_pos_t = torch.from_numpy(crop_pos_np[perm]).float()
        crop_x = torch.from_numpy(x[perm].astype(np.float32))

        crop_x, crop_pos = self.pad_to_block_size([crop_x, crop_pos_t])

        crop_block_mask = create_block_mask_from_kdtree(
            kdtree=KDTree(crop_pos_np[perm], leafsize=self.attn_block_size),
            points=crop_pos.cpu().numpy(),
            n_points_unpadded=len(crop_indices),
            k=self.k,
            block_size=self.attn_block_size,
            symmetric=self.symmetric_block_mask,
        )
        return crop_x, crop_pos, crop_block_mask, perm

    def sample_positive_crop(
        self,
        valid_seeds: list[int],
        centroids: np.ndarray,
        targets: Tensor,
        tree: KDTree,
    ) -> NDArray[np.int64] | None:
        assert self.crop_pos_thr is not None

        for _ in range(10):
            seed_idx = choice(valid_seeds)

            # heuristic via Euclidean circle
            _, neighbor_idx = tree.query(centroids[seed_idx], k=self.crop_size)
            tumor_ratio = (targets[neighbor_idx] == 1).sum().item() / self.crop_size

            self.heuristic_threshold = max(0.0, self.crop_pos_thr - 0.15)
            if tumor_ratio < self.heuristic_threshold:
                continue

            crop_indices = self.get_crop_indices(centroids, [seed_idx])
            crop_targets = targets[torch.from_numpy(crop_indices).long()]
            tumor_ratio = (crop_targets == 1).sum().item() / len(crop_targets)

            if tumor_ratio >= self.crop_pos_thr:
                return crop_indices

        return None

    def get_targets(
        self,
        crop_label: float,
        nuclei_sup: NucleiSupervision,
        n: int,
        crop_indices: torch.Tensor,
        perm: torch.Tensor,
    ) -> Targets:
        if self.mil:
            return {
                "nuclei": None,
                "graph": torch.tensor([crop_label], dtype=torch.float32),
            }
        targets = nuclei_sup.get_targets(n)
        crop_nuclei_y = targets[crop_indices][perm]
        crop_nuclei_y_padded = self.pad_to_block_size(crop_nuclei_y)[0]
        return {"nuclei": crop_nuclei_y_padded, "graph": None}

    def get_sup_mask(
        self,
        n: int,
        nuclei_sup: NucleiSupervision,
        indices: torch.Tensor,
        perm: torch.Tensor,
    ) -> torch.Tensor:
        if self.mil:
            return torch.ones(1, dtype=torch.bool)  # dummy mask
        sup_mask = nuclei_sup.get_sup_mask(n)[indices]
        return self.pad_to_block_size(sup_mask[perm])[0]

    def get_nuclei(self, slide: pd.Series) -> pd.DataFrame:
        nuclei = pd.read_parquet(slide.slide_nuclei_path)
        nuclei = nuclei.sort_values("id").reset_index(drop=True)
        return nuclei

    def get_centroids(
        self, nuclei: pd.DataFrame, mpp_x: float, mpp_y: float
    ) -> np.ndarray:
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        return np.stack(nuclei["centroid"].tolist()) * mpps

    def get_nuclei_supervision(self, slide_id: str) -> NucleiSupervision:
        return self.supervision.supervision_map[slide_id].nuclei_supervision

    def __getitem__(self, idx: int) -> Crop | PredictSlide:
        slide = self.slides.iloc[idx]

        nuclei = self.get_nuclei(slide)
        n = len(nuclei)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)
        nuclei_sup = self.get_nuclei_supervision(slide.slide_id)

        # get indices eligible as a seed for growing the crop component
        seed_mask = nuclei_sup.get_seed_mask(n, centroids=centroids)
        valid_seeds = torch.nonzero(seed_mask).flatten().tolist()

        # generate a crop
        crop_indices = self.get_crop_indices(centroids, valid_seeds)
        crop_label = float(slide.is_carcinoma)  # default is slide-level label

        if slide.is_carcinoma and self.mil and not (self.full_slide or self.predict):
            assert self.crop_pos_thr is not None

            curr_idx = None
            crop_indices = None
            while True:  # if a positive slide was chosen, ensure the generated crop has enough positive nuclei
                if curr_idx is not None:
                    slide = self.slides.iloc[curr_idx]

                    nuclei = self.get_nuclei(slide)
                    n = len(nuclei)
                    centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

                    nuclei_sup = self.get_nuclei_supervision(slide.slide_id)
                    seed_mask = nuclei_sup.get_seed_mask(n, centroids=centroids)
                    valid_seeds = torch.nonzero(seed_mask).flatten().tolist()

                targets = nuclei_sup.get_targets(n)
                tree = KDTree(centroids)
                crop_indices = self.sample_positive_crop(
                    valid_seeds, centroids, targets, tree
                )

                if crop_indices is not None:
                    break

                curr_idx = choice(self.pos_slide_indices)

        crop_x, crop_pos, crop_block_mask, perm = self.get_crop_features(
            nuclei, centroids, crop_indices, slide.mpp_x, slide.mpp_y
        )

        crop_indices_t = torch.from_numpy(crop_indices).long()
        perm_t = torch.from_numpy(perm).long()

        crop_y: Targets = self.get_targets(
            crop_label=crop_label,
            nuclei_sup=nuclei_sup,
            n=n,
            crop_indices=crop_indices_t,
            perm=perm_t,
        )
        crop_sup_mask = self.get_sup_mask(n, nuclei_sup, crop_indices_t, perm_t)

        crop: Crop = {
            "x": crop_x,  # (n, efd_order * 4 + 3)
            "pos": crop_pos,  # (n, 2)
            "y": crop_y,  # (num_supervised, )
            "sup_mask": crop_sup_mask.bool(),  # (n, )
            "block_mask": crop_block_mask,
            "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
        }
        if self.predict:
            metadata: Metadata = {
                "slide_id": slide.slide_id,
                "slide_path": slide.slide_path,
                "slide_nuclei_path": slide.slide_nuclei_path,
                "perm_inverse": self.get_inverse_perm(perm_t),
                "nuclei_ids": nuclei["id"].to_numpy(),
            }
            return PredictSlide(slide=crop, metadata=metadata)
        return crop
