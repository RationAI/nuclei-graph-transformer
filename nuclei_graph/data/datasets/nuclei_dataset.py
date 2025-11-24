import heapq
import random
from typing import cast

import numpy as np
import pandas as pd
import torch
from degraph import build_spatial_graph
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torch import Tensor
from torch.utils.data import Dataset

from nuclei_graph.data.block_mask import create_block_mask
from nuclei_graph.features import normalize_efd
from nuclei_graph.nuclei_graph_typing import (
    AdjacencyGraph,
    Metadata,
    PointArray,
    PredictSample,
    Sample,
    Transforms,
)


type HeapItem = tuple[float, int]  # (priority, node_idx)


class NucleiDataset(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        df_metadata: pd.DataFrame,
        labels_path: str,
        label_indicators_path: str | None = None,
        transforms: Transforms | None = None,
        alpha: float = 0.8,
        crop_size: int = 4096,
        k: int = 64,
        attn_block_size: int = 128,
        full_slide: bool = False,
        predict: bool = False,
    ) -> None:
        """Dataset for nuclei point clouds from whole-slide images.

        Args:
            df_metadata: Pandas DataFrame containing slide metadata with columns `slide_id`, `slide_mrxs_path`, and `slide_nuclei_path`.
            labels_path: Local path to the nuclei label parquet files with columns `slide_id`, `nucleus_id`, and `label`.
            label_indicators_path: Optional local path to parquet files containing finer annotation with columns
                                   `slide_id`, `nucleus_id` and `cam_label_indicator`.
            transforms: Optional (list of) transforms to apply to the data (e.g. normalization, ...).
            alpha: Weight between graph edge distance and Euclidean distance when selecting neighbors.
            crop_size: Number of nuclei in a crop.
            k: Number of neighbors for sparse attention.
            attn_block_size: Block size for sparse attention, it must hold that `crop_size` is divisible by `attn_block_size`.
            full_slide: Whether the dataset is used for full slide inference.
            predict: Whether to return the metadata needed for prediction along with the data.
        """
        self.df_metadata = df_metadata
        self.labels_path = labels_path
        self.label_indicators_path = label_indicators_path
        self.slides_positivity = self.compute_slides_positivity()
        self.transforms = transforms if isinstance(transforms, list) else [transforms]
        self.alpha = alpha
        self.crop_size = crop_size
        self.k = k
        self.attn_block_size = attn_block_size
        assert self.crop_size % self.attn_block_size == 0
        self.full_slide = full_slide
        self.predict = predict

    def __len__(self) -> int:
        return len(self.df_metadata)

    def compute_slides_positivity(self) -> dict[int, float]:
        """Computes the fraction of annotated positive nuclei per slide.

        Returns:
            dict[int, float]: Mapping from slide index to positivity score.
        """
        slides_positivity: dict[int, float] = {}
        for idx in range(len(self.df_metadata)):
            slide_id = self.df_metadata.iloc[idx].slide_id
            if self.label_indicators_path is None:
                labels_df = pd.read_parquet(self.labels_path)
                slides_positivity[idx] = (
                    labels_df[labels_df["slide_id"] == slide_id].label.mean().item()
                )
                continue
            label_indicators_df = pd.read_parquet(self.label_indicators_path)
            slides_positivity[idx] = (
                label_indicators_df[label_indicators_df["slide_id"] == slide_id]
                .cam_label_indicator.mean()
                .item()
            )
        return slides_positivity

    def find_component(
        self,
        idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: PointArray,
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
        component_indices: list[int] = []
        in_component = np.zeros(len(centroids), dtype=bool)

        pq: list[HeapItem] = []
        heapq.heappush(pq, (0.0, idx))
        start_point_coords = centroids[idx]

        while len(pq) != 0 and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)
            if in_component[current_idx]:
                continue

            in_component[current_idx] = True
            component_indices.append(current_idx)

            for n_idx, edge_dist in graph[current_idx]:
                if (indices is None or n_idx in indices) and not in_component[n_idx]:
                    start_dist = np.linalg.norm(centroids[n_idx] - start_point_coords)
                    hybrid_cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist
                    heapq.heappush(pq, (hybrid_cost, n_idx))  # type: ignore[misc]

        return component_indices

    def get_annot_mask(self, idx: int) -> Tensor:
        slide_id = self.df_metadata.iloc[idx].slide_id
        mask_path = self.annotations.get(slide_id, None)
        assert mask_path is not None, (
            f"Annotation mask not found for slide '{slide_id}'."
        )
        return torch.load(mask_path, weights_only=False, mmap=True).bool().view(-1, 1)

    def get_slide_metadata(
        self, idx: int, nuclei_count: int, perm_inverse: Tensor
    ) -> Metadata:
        return {
            "slide_id": self.df_metadata.iloc[idx].slide_id,
            "slide_mrxs_path": self.df_metadata.iloc[idx].slide_mrxs_path,
            "slide_nuclei_path": self.df_metadata.iloc[idx].slide_nuclei_path,
            "nuclei_count": nuclei_count,
            "perm_inverse": perm_inverse,
        }

    def get_crop_indices(self, centroids: PointArray, annot_mask: Tensor) -> list[int]:
        """Builds a Delaunay graph and creates a crop of size `crop_size` by growing a component starting from a random nucleus.

        Only annotated nuclei are considered as seeds to avoid creating a crop fully in an unlabeled region.
        """
        graph = build_spatial_graph(centroids)
        annotated_indices = (
            torch.nonzero(annot_mask.squeeze(-1), as_tuple=False).squeeze(-1).tolist()
        )
        assert annotated_indices
        seed = random.choice(annotated_indices)
        return self.find_component(seed, self.crop_size, graph, centroids)

    def build_kd_tree(self, points: PointArray) -> tuple[KDTree, Tensor]:
        """Builds a KDTree and computes permutation that clusters points by leaf for sparse attention."""
        tree_tmp = KDTree(points, leafsize=self.attn_block_size)
        perm = tree_tmp.indices
        tree = KDTree(points[perm], leafsize=self.attn_block_size)
        return tree, torch.from_numpy(perm).long()

    def add_rotation(self, positions: Tensor, psi_1: Tensor) -> Tensor:
        """Append the rotation angle as a third positional dimension."""
        angles = (psi_1 % torch.pi).view(-1, 1)
        return torch.cat([positions, angles], dim=1)

    def pad_to_block_size(
        self, x: torch.Tensor, pos: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pad tensors so that their first dimension is divisible by the attention block size.

        Used for validation/testing/prediction.
        """
        n = x.size(0)
        remainder = n % self.attn_block_size

        if not self.full_slide or remainder == 0:
            return x, pos, y, mask

        pad_len = self.attn_block_size - remainder
        pad_x = torch.zeros((pad_len, x.size(1)), dtype=x.dtype, device=x.device)
        pad_pos = torch.zeros(
            (pad_len, pos.size(1)), dtype=pos.dtype, device=pos.device
        )
        pad_y = torch.zeros((pad_len, 1), dtype=y.dtype, device=y.device)
        pad_mask = torch.zeros((pad_len, 1), dtype=mask.dtype, device=mask.device)

        return (
            torch.cat([x, pad_x], dim=0),
            torch.cat([pos, pad_pos], dim=0),
            torch.cat([y, pad_y], dim=0),
            torch.cat([mask, pad_mask], dim=0),
        )

    def get_inverse_perm(self, perm: Tensor) -> Tensor:
        assert perm.dim() == 1, "perm must be 1D"
        perm_inverse = torch.empty_like(perm)
        perm_inverse[perm] = torch.arange(perm.size(0), dtype=perm.dtype)
        assert torch.equal(
            perm[perm_inverse], torch.arange(perm.size(0), dtype=perm.dtype)
        )
        return perm_inverse

    def get_cropped_data(
        self,
        embeddings: Tensor,
        positions: Tensor,
        labels: Tensor,
        masks: Tensor,
        crop_indices: Tensor,
        perm: Tensor,
        angle: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        crop_embeddings = embeddings[crop_indices][perm]
        crop_positions = positions[crop_indices][perm]
        crop_labels = labels[crop_indices][perm]
        crop_annot_mask = masks[crop_indices][perm]

        crop_positions = self.add_rotation(crop_positions, angle[crop_indices][perm])

        return self.pad_to_block_size(
            crop_embeddings, crop_positions, crop_labels, crop_annot_mask
        )

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        slide_id = self.df_metadata.iloc[idx].slide_id
        slide_nuclei_path = cast(
            "pd.Series",
            self.df_metadata.loc[
                self.df_metadata.slide_id == slide_id, "slide_nuclei_path"
            ],
        ).item()
        nuclei_data["annot_mask"] = self.get_annot_mask(idx)

        # save the rotation angle psi_1 before normalization
        _, psi_1, _ = normalize_efd(nuclei_data["x"], return_angles=True)

        for transform in filter(None, self.transforms):
            nuclei_data = transform(nuclei_data)

        embeddings, positions, labels, masks = (
            nuclei_data["x"],
            nuclei_data["pos"],
            nuclei_data["y"],
            nuclei_data["annot_mask"],
        )

        if self.full_slide:
            crop_indices = torch.arange(len(embeddings))
        else:
            crop_indices = torch.tensor(
                self.get_crop_indices(positions.numpy(), masks),
                dtype=torch.long,
            )

        tree, perm = self.build_kd_tree(positions[crop_indices].numpy())

        crop_embeddings, crop_positions, crop_labels, crop_annot_mask = (
            self.get_cropped_data(
                embeddings, positions, labels, masks, crop_indices, perm, psi_1
            )
        )
        crop_block_mask = create_block_mask(
            kdtree=tree,
            points=crop_positions[
                :, :2
            ].numpy(),  # only pass the centroid 2D coordinates
            n_points_unpadded=len(crop_indices),
            k=self.k,
            block_size=self.attn_block_size,
        )

        item: Sample = {
            "x": crop_embeddings,  # (n, d)
            "pos": crop_positions,  # (n, 3)
            "y": crop_labels[crop_annot_mask],  # (num_filtered,)
            "y_indicator_mask": crop_y_indicator_mask,  # (n, 1)
            "block_mask": crop_block_mask,  # BlockMask
        }

        if self.predict:
            perm_inverse = self.get_inverse_perm(perm)
            return item, self.get_slide_metadata(idx, len(crop_indices), perm_inverse)
        return item
