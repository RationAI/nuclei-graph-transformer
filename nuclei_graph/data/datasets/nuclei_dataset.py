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
)
from nuclei_graph.nuclei_graph_typing import Metadata, PredictSample, Sample


type PriorityQueueItem = tuple[float, int]  # (cost, node_idx)
type Neighbor = tuple[int, float]  # (node_idx, edge_distance)
type AdjacencyGraph = list[list[Neighbor]]


class NucleiDataset(Dataset[Sample | PredictSample]):
    """Dataset for nuclei point clouds from whole-slide images."""

    def __init__(
        self,
        df_metadata: DataFrame,
        df_annot_labels: DataFrame,
        df_cam_labels: DataFrame,
        scale_mean: float,
        supervision_mode: str = "agreement-strict",
        crop_size: int = 4096,
        alpha: float = 0.8,
        k: int = 64,
        attn_block_size: int = 128,
        efd_order: int = 10,
        full_slide: bool = False,
        predict: bool = False,
    ) -> None:
        """Initializes the dataset.

        Args:
            df_metadata: DataFrame with columns: "slide_id" (str), "is_carcinoma" (bool), and "slide_nuclei_path" (str) (if the predict mode is set
                to `True` then also "slide_path" (str)), where "slide_nuclei_path" points to parquet files containing nuclei segmentation data.
            df_annot_labels: DataFrame containing annotation-based nuclei labels with columns "slide_id" (str), "id" (str), and "annot_label" (int; 0/1).
            df_cam_labels: DataFrame containing CAM-based nuclei labels with columns "slide_id" (str), "id" (str), and "cam_label" (int; 0/1/-1).
            scale_mean: Mean of nuclei scales estimated from training data for normalization.
            supervision_mode: Supervision mode for weakly supervised learning, one of "annotation", "cam", "agreement", "agreement-strict".
            crop_size: Number of nuclei in a crop (sample) during training.
            alpha: Weight between graph edge distance and Euclidean distance when selecting neighbors during graph creation.
            k: Number of neighbors for sparse attention.
            attn_block_size: Block size for sparse attention. It must hold that `crop_size` mod `attn_block_size` is 0.
            efd_order: Order of the elliptic fourier descriptors used for nuclei shape representation.
            full_slide: Whether the dataset is used for full slide inference (no cropping).
            predict: Whether to return the metadata needed for prediction ("slide_path" (str)) along with the data.
        """
        assert crop_size % attn_block_size == 0, (
            "`crop_size` must be divisible by `attn_block_size`."
        )
        self.df_metadata = df_metadata
        self.df_annot_labels = self._build_index(df_annot_labels, ["slide_id", "id"])
        self.df_cam_labels = self._build_index(df_cam_labels, ["slide_id", "id"])
        self.scale_mean = scale_mean
        self.supervision_mode = supervision_mode
        self.crop_size = crop_size
        self.alpha = alpha
        self.k = k
        self.attn_block_size = attn_block_size
        self.efd_order = efd_order
        self.full_slide = full_slide
        self.predict = predict

    def __len__(self) -> int:
        return len(self.df_metadata)

    def _build_index(self, df: DataFrame, cols: list[str]) -> DataFrame:
        """Pre-build and sort a multi-index for fast lookup."""
        return df.set_index(cols).sort_index()

    def find_component(
        self,
        seed_idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: NDArray[np.float32],
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
                if not visited[n_idx] and (
                    allowed_indices is None or n_idx in allowed_indices
                ):
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

    def load_targets_and_masks(
        self, slide_id: str, nuclei_ids: pd.Series, slide_is_carcinoma: bool
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Load nucleus-level targets and a supervision mask for weakly supervised learning.

        Returns:
            targets (tensor[float], shape (n,)): Labels for each nucleus.
            sup_mask (tensor[bool], shape (n,)): True for nuclei with confident labels, False for uncertain.
            valid_seeds (list[int]): Nuclei indices eligible as seeds for crop generation; i.e. supervised positive nuclei.
        """
        n = len(nuclei_ids)
        targets = torch.full((n,), 0.0, dtype=torch.float32)  # default: negative
        sup_mask = torch.full((n,), True, dtype=torch.bool)  # default: all supervised

        if not slide_is_carcinoma:
            return targets, sup_mask, list(range(n))

        def load_df(df: pd.DataFrame, column: str) -> np.ndarray:
            return (
                df.loc[slide_id].reindex(nuclei_ids)[column].to_numpy(dtype=np.float32)
            )

        annot = load_df(self.df_annot_labels, "annot_label")
        cam = load_df(self.df_cam_labels, "cam_label")
        valid_seeds = np.arange(n)

        match self.supervision_mode:
            case "annotation":
                targets = torch.from_numpy(annot).float()
                valid_seeds = np.nonzero(annot == 1)[0]  # nonzero returns (indices,)
            case "cam":
                targets = torch.from_numpy(cam).float()
                sup_mask = torch.from_numpy(cam != -1).bool()  # -1 indicates uncertain
                valid_seeds = np.nonzero(cam == 1)[0]
            case "agreement":
                targets = torch.from_numpy(annot).float()
                sup_mask = torch.from_numpy(annot == cam).bool()
                valid_seeds = np.nonzero((annot == 1) & (cam == 1))[0]
            case "agreement-strict":
                targets = torch.from_numpy(annot).float()
                sup_mask = torch.from_numpy((annot == 1) & (cam == 1)).bool()
                valid_seeds = np.nonzero((annot == 1) & (cam == 1))[0]

        assert torch.all(targets[sup_mask] != -1.0)  # sup. targets cannot be uncertain
        return targets, sup_mask, valid_seeds.tolist()

    def get_crop_indices(
        self, centroids: NDArray[np.float32], valid_seeds: list[int]
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
        if self.full_slide:
            return np.arange(len(centroids), dtype=int)
        graph = build_spatial_graph(centroids)
        seed_idx = (
            choice(valid_seeds) if valid_seeds else randint(0, len(centroids) - 1)
        )
        return np.array(self.find_component(seed_idx, self.crop_size, graph, centroids))

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

        # --- Extract EFD features ---
        contours = rearrange(nuclei["polygon"].tolist(), "b (v c) -> b v c", c=2)
        efd = elliptic_fourier_descriptors(np.asarray(contours), self.efd_order)
        efd, angles = normalize_efd_for_rotation(efd)
        x = rearrange(efd, "n order c -> n (order c)")
        x = x / self.scale_mean

        # --- Load targets and supervision masks ---
        targets, sup_mask, valid_seeds = self.load_targets_and_masks(
            self.df_metadata.iloc[idx].slide_id,
            nuclei["id"],
            self.df_metadata.iloc[idx].is_carcinoma,
        )

        # --- Create a crop ---
        centroids = np.stack(nuclei["centroid"].tolist())
        crop_indices_np = self.get_crop_indices(centroids, valid_seeds)

        # --- Prepare positional encodings ---
        crop_centroids = centroids[crop_indices_np]
        # take modulo π due to 180° symmetry and stretch to [0, 2π) to ensure closure at 0/π
        rotations = 2.0 * (angles % np.pi)
        crop_rotations = rotations[crop_indices_np]
        crop_pos_np = np.concatenate([crop_centroids, crop_rotations], axis=-1)

        # --- Optimize data layout for block-sparse attention ---
        perm_np = KDTree(crop_centroids, leafsize=self.attn_block_size).indices

        perm_t = torch.from_numpy(perm_np).long()
        crop_indices_t = torch.from_numpy(crop_indices_np).long()

        crop_y = targets[crop_indices_t][perm_t]
        crop_sup_mask = sup_mask[crop_indices_t][perm_t]
        crop_pos = torch.from_numpy(crop_pos_np[perm_np]).float()
        crop_x = torch.from_numpy(x[crop_indices_np][perm_np].astype(np.float32))

        crop_x, crop_pos, crop_y, crop_sup_mask = self.pad_to_block_size(
            [crop_x, crop_pos, crop_y, crop_sup_mask]
        )

        sample: Sample = {
            "x": crop_x,  # (n, efd_order * 4)
            "pos": crop_pos,  # (n, 3)
            "y": crop_y[crop_sup_mask],  # (num_supervised, )
            "sup_mask": crop_sup_mask.bool(),  # (n, )
            "block_mask": create_block_mask_from_kdtree(
                kdtree=KDTree(crop_centroids[perm_np], leafsize=self.attn_block_size),
                points=crop_pos[:, :2].cpu().numpy(),  # only pass spatial coordinates
                n_points_unpadded=len(crop_indices_np),
                k=self.k,
                block_size=self.attn_block_size,
            ),
        }
        if self.predict:
            metadata: Metadata = {
                "slide_id": self.df_metadata.iloc[idx].slide_id,
                "slide_nuclei_path": self.df_metadata.iloc[idx].slide_nuclei_path,
                "nuclei_ids": nuclei.iloc[crop_indices_np.tolist()]["id"],
                "perm_inverse": self.get_inverse_perm(perm_t),
            }
            return PredictSample(sample=sample, metadata=metadata)
        return sample
