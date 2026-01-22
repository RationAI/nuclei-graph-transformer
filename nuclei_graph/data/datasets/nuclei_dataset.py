import heapq
from collections.abc import Iterable
from random import choice, randint

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
from torch.utils.data import Dataset

from nuclei_graph.data import create_block_mask_from_kdtree
from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.nuclei_graph_typing import Metadata, PredictSample, Sample


type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]


class NucleiDataset(Dataset[Sample | PredictSample]):
    """Dataset for nuclei point clouds from whole-slide images."""

    def __init__(
        self,
        df_metadata: DataFrame,
        scale_mean: float,
        scale_std: float,
        neighbor_dist_median: float,
        df_labels: DataFrame | None = None,
        df_refinement: DataFrame | None = None,
        crop_size: int = 4096,
        alpha: float = 0.8,
        k: int = 64,
        attn_block_size: int = 128,
        efd_order: int = 10,
        full_slide: bool = False,
        predict: bool = False,
    ) -> None:
        """Initializes the dataset.

        It is assumed that all slides have mpp same as the training data (0.25). If not the case,
        scaling normalization must be implemented for physical distances (nuclei size, neighbor distance).

        Args:
            df_metadata: DataFrame with columns: "slide_id" (str), "is_carcinoma" (bool), and "slide_nuclei_path" (str)
                (if the predict mode is set to `True` then also "slide_path" (str)), where "slide_nuclei_path" points
                to parquet files containing nuclei segmentation data.
            scale_mean: Mean of log nucleus scales estimated from training data for normalization.
            scale_std: Standard deviation of log nucleus scales estimated from training data for normalization.
            neighbor_dist_median: Median distance between neighboring nuclei in pixels for normalization.
            df_labels: Optional DataFrame containing nuclei labels with columns "slide_id" (str), "id" (str) and "label" (int; 0/1).
            df_refinement: Optional DataFrame containing a boolean filter that masks-out nuclei whose label cannot be determined
                confidently enough (e.g., using a CAM thresholding). It is expected to contain columns "slide_id" (str), "id" (str),
                and "refinement_mask" (bool).
            crop_size: Number of nuclei in a crop (sample) during training.
            alpha: Weight between graph edge distance and Euclidean distance when selecting neighbors during graph creation.
            k: Number of neighbors for sparse attention.
            attn_block_size: Block size for sparse attention. It must hold that `crop_size` mod `attn_block_size` is 0.
            efd_order: Order of the elliptic fourier descriptors used for nucleus shape representation.
            full_slide: Whether the dataset is used for full slide inference (no cropping).
            predict: Whether to return the metadata needed for prediction ("slide_path" (str)) along with the data.
        """
        self.df_metadata = df_metadata
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.neighbor_dist_median = neighbor_dist_median
        self.df_labels = self._build_index(df_labels, ["slide_id", "id"])
        self.df_refinement = self._build_index(df_refinement, ["slide_id", "id"])
        self.crop_size = crop_size
        self.alpha = alpha
        self.k = k
        self.attn_block_size = attn_block_size
        assert self.crop_size % self.attn_block_size == 0
        self.efd_order = efd_order
        self.full_slide = full_slide
        self.predict = predict

    def __len__(self) -> int:
        return len(self.df_metadata)

    def _build_index(self, df: DataFrame | None, cols: list[str]) -> DataFrame | None:
        """Pre-build and sort a multi-index for fast lookup."""
        return df.set_index(cols).sort_index() if df is not None else None

    def find_component(
        self,
        idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: NDArray[np.float32],
        indices: NDArray[np.int64] | None = None,
    ) -> list[int]:
        """Grows a connected component of up to `k` nuclei starting from a seed index.

        Args:
            idx: seed nucleus index
            k: maximum number of nuclei to include in the component.
            graph: adjacency list of the nuclei graph.
            centroids: array of nucleus coordinates.
            indices: optional array of allowed nucleus indices

        Returns:
            list[int]: Indices of nuclei in the component.

        Taken from the Nuclei Foundational Model repository.
        """
        component_indices = []
        visited = np.zeros(len(centroids), dtype=bool)

        pq = []
        heapq.heappush(pq, (0, idx))
        start_point_coords = centroids[idx]

        while pq and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)
            if visited[current_idx]:
                continue

            visited[current_idx] = True
            component_indices.append(current_idx)

            for n_idx, edge_dist in graph[current_idx]:
                if not visited[n_idx] and (indices is None or n_idx in indices):
                    start_dist = np.linalg.norm(centroids[n_idx] - start_point_coords)
                    cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist
                    heapq.heappush(pq, (cost, n_idx))
        return component_indices

    def pad_to_block_size(self, tensors: Iterable[Tensor]) -> list[Tensor]:
        """Pads tensors along dim-0 so that their size is divisible by `attn_block_size`.

        Padding is applied at the end and preserves all other dimensions.
        """
        tensors = list(tensors)

        n = tensors[0].size(0)
        assert all(t.size(0) == n for t in tensors)

        remainder = n % self.attn_block_size
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

    def load_targets_and_masks(
        self, slide_id: str, nuclei_ids: pd.Series, slide_is_carcinoma: bool
    ):
        """Load nucleus-level targets and supervision masks for weakly supervised learning.

        For negative slides, all nuclei are treated as confidently negative.
        For positive slides, only nuclei inside annotations (and passing refinement, if any)
        are considered confidently labeled; the rest are marked as uncertain (inside annotations
        and outside the refinement mask) or ignored (outside annotations).

        Returns:
            targets: Float tensor of shape (n,). Values are 1.0 for positively annotated nuclei and 0.0 otherwise.
            sup_mask: Boolean tensor of shape (n,). True for nuclei with confident labels.
            ignore_mask: Boolean tensor of shape (n,). True for nuclei that should be excluded from all losses.
            valid_seeds: List of nuclei indices eligible as seeds for crop/component sampling.
        """
        n = len(nuclei_ids)
        targets = torch.full((n,), 0.0, dtype=torch.float32)  # default: negative
        sup_mask = torch.full((n,), True, dtype=torch.bool)  # default: no refinement
        ignore_mask = torch.full((n,), False, dtype=torch.bool)
        valid_seeds = list(range(n))

        if not slide_is_carcinoma:
            return targets, sup_mask, ignore_mask, valid_seeds

        assert self.df_labels is not None  # must be provided for positive slides
        targets = torch.from_numpy(
            self.df_labels.loc[slide_id].reindex(nuclei_ids)["label"].values
        ).float()

        sup_mask = (
            torch.from_numpy(
                self.df_refinement.loc[slide_id]
                .reindex(nuclei_ids)["refinement_mask"]
                .values
            ).bool()
            if self.df_refinement is not None
            else sup_mask
        )
        ignore_mask = targets == 0.0  # nuclei outside annotations (in positive slides)
        valid_seeds = torch.nonzero(~ignore_mask).squeeze(-1).tolist()
        return targets, sup_mask, ignore_mask, valid_seeds

    def get_crop_indices(self, centroids: np.ndarray, valid_seeds: list[int]) -> Tensor:
        if self.full_slide:
            return torch.arange(len(centroids))
        graph = build_spatial_graph(centroids)
        seed = choice(valid_seeds) if valid_seeds else randint(0, len(centroids) - 1)
        return torch.tensor(
            self.find_component(seed, self.crop_size, graph, centroids),
            dtype=torch.long,
        )

    def drop_eps_neighbors(self, df: pd.DataFrame, eps: float = 1e-4) -> pd.DataFrame:
        """Removes nuclei closer than `eps` to each other.

        One of the close neighbors is removed, the other is kept. This is to prevent Delaunay triangulation
        from failing due to numerical instabilities when nuclei are too close to each other.
        """
        centroids = np.stack(df["centroid"].tolist())
        quantized = np.round(centroids / eps).astype(np.int64)
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)

        if len(df) - len(unique_indices) > 0:
            return df.iloc[np.sort(unique_indices)].reset_index(drop=True)
        return df

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        nuclei_path = self.df_metadata.iloc[idx].slide_nuclei_path
        nuclei = self.drop_eps_neighbors(pd.read_parquet(nuclei_path).sort_values("id"))
        centroids = np.stack(nuclei["centroid"].tolist())
        contours = rearrange(nuclei["polygon"].tolist(), "b (v c) -> b v c", c=2)

        # compute feature vectors
        efd = elliptic_fourier_descriptors(contours, self.efd_order)
        _, angles = normalize_efd_for_rotation(efd)
        efd, scales = normalize_efd_for_scale(efd)
        x = rearrange(efd, "n order c -> n (order c)")
        log_scales = (np.log(scales + 1e-8) - self.scale_mean) / self.scale_std
        x = np.concatenate([x, log_scales], axis=-1)

        targets, sup_mask, ignore_mask, valid_seeds = self.load_targets_and_masks(
            self.df_metadata.iloc[idx].slide_id,
            nuclei["id"],
            self.df_metadata.iloc[idx].is_carcinoma,
        )

        crop_indices = self.get_crop_indices(centroids, valid_seeds)
        # center to crop mean for numerical stability (RoPE) and divide by fixed average nuclei neighbor
        # distance computed from training set to convert distances into neighbor units ("cell hops")
        center = centroids[crop_indices].mean(axis=0, keepdims=True)
        crop_centroids = (centroids[crop_indices] - center) / self.neighbor_dist_median
        # take modulo π to account for the 180° symmetry and stretch to [0, 2π) to ensure closure at 0/π
        rotation = 2.0 * (angles % np.pi)
        crop_pos = np.concatenate([crop_centroids, rotation[crop_indices]], axis=-1)

        # compute spatial permutation to optimize block attention locality for the crop
        tree = KDTree(crop_centroids, leafsize=self.attn_block_size)
        perm = torch.from_numpy(tree.indices).long()
        sorted_tree = KDTree(crop_centroids[perm], leafsize=self.attn_block_size)

        crop_x = torch.from_numpy(x[crop_indices][perm].astype(np.float32))
        crop_pos = torch.from_numpy(crop_pos[perm].astype(np.float32))
        crop_y = targets[crop_indices][perm]
        crop_sup_mask = sup_mask[crop_indices][perm]
        crop_ignore_mask = ignore_mask[crop_indices][perm]

        crop_x, crop_pos, crop_y, crop_sup_mask, crop_ignore_mask = (
            self.pad_to_block_size(
                [crop_x, crop_pos, crop_y, crop_sup_mask, crop_ignore_mask]
            )
        )
        sample: Sample = {
            "x": crop_x,  # (n, efd_order * 4 + 1)
            "pos": crop_pos,  # (n, 3)
            "y": crop_y[crop_sup_mask].unsqueeze(-1),  # (num_supervised, 1)
            "sup_mask": crop_sup_mask,  # (n,)
            "ignore_mask": crop_ignore_mask,  # (n,)
            "num_points": len(crop_indices),
            "block_mask": create_block_mask_from_kdtree(
                kdtree=sorted_tree,
                points=crop_pos[:, :2].numpy(),  # only pass spatial coordinates
                n_points_unpadded=len(crop_indices),
                k=self.k,
                block_size=self.attn_block_size,
            ),
        }
        if self.predict:
            metadata: Metadata = {
                "slide_id": self.df_metadata.iloc[idx].slide_id,
                "slide_nuclei_path": self.df_metadata.iloc[idx].slide_nuclei_path,
                "nuclei_ids": list(map(str, nuclei.iloc[crop_indices.numpy()]["id"])),
                "perm_inverse": self.get_inverse_perm(perm),
            }
            return sample, metadata
        return sample
