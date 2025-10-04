"""Adjusted from the Nuclei Foundational Model repository."""

import heapq
import random
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from graph_pytorch_ext import build_adjacency_graph
from numpy.typing import NDArray
from scipy.spatial import Delaunay, KDTree
from torch import Tensor
from torch.utils.data import Dataset

from nuclei_graph.typing import Metadata, PredictSample, Sample, Transforms
from nuclei_graph.utils import create_single_block_mask_from_kdtree
from nuclei_graph.utils.torch_efd import normalize_efd


Neighbor: TypeAlias = tuple[int, float]  # (node_idx, edge_weight)
AdjacencyGraph: TypeAlias = list[list[Neighbor]]
PointArray: TypeAlias = NDArray[np.float32]
HeapItem: TypeAlias = tuple[float, int]  # (priority, node_idx)


class NucleiDataset(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        metadata_path: str,
        nuclei_path: str,
        annot_masks_path: str | None = None,
        transforms: Transforms | None = None,
        alpha: float = 0.8,
        crop_size: int = 4096,
        k: int = 64,
        attn_block_size: int = 128,
        full_slide: bool = False,
        predict: bool = False,
    ) -> None:
        """Dataset for nuclei point clouds.

        Args:
            metadata_path: Local path to dataset metadata.
            nuclei_path: Local path to the .pt files of the nuclei data: Data(x=embeddings, pos=centroids, y=labels).
            annot_masks_path: Optional local path for positive slides, includes finer annotation for masked training.
            transforms: Optional (list of) transforms to apply to the data (e.g. normalization, ...).
            alpha: Weight between graph edge distance and Euclidean distance when selecting neighbors.
            crop_size: Number of nuclei in a crop.
            k: Number of neighbors for sparse attention.
            attn_block_size: Block size for sparse attention, it must hold that `crop_size % attn_block_size == 0`.
            full_slide: Whether the dataset is used for full slide inference.
            predict: Whether to return the metadata needed for prediction along with the data.
        """
        self.full_slide = full_slide
        self.crop_size = crop_size
        self.slides = pd.read_parquet(metadata_path)
        self.nuclei = self._load_nuclei(nuclei_path)
        self.annotations = self._load_annotations(annot_masks_path)
        self.slides_positivity = self._compute_slides_positivity()
        self.transforms = transforms if isinstance(transforms, list) else [transforms]
        self.alpha = alpha
        self.k = k
        self.attn_block_size = attn_block_size
        assert self.crop_size % self.attn_block_size == 0
        self.predict = predict

    def __len__(self) -> int:
        return len(self.slides)

    def _load_nuclei(self, nuclei_path: str) -> dict[str, Path]:
        """Loads nuclei slides, skips slides with fewer nuclei than `crop_size`, and aligns the `self.slides` index."""
        slides_nuclei = list(Path(nuclei_path).rglob("*.pt"))

        valid_slides = {}
        skipped_count = 0
        for slide_nuclei in slides_nuclei:
            slide_id = slide_nuclei.stem
            data = torch.load(slide_nuclei, weights_only=False, mmap=True)
            n_nuclei = data["x"].shape[0]

            if not self.full_slide and n_nuclei < self.crop_size:
                print(
                    f"Warning: Slide {slide_id} has only {n_nuclei} nuclei (crop_size={self.crop_size}). "
                    f"Skipping the slide..."
                )
                skipped_count += 1
                continue
            valid_slides[slide_id] = slide_nuclei

        if skipped_count > 0:
            print(f"Total slides skipped: {skipped_count}")

        self.slides = self.slides[self.slides.slide_id.isin(valid_slides)].reset_index(
            drop=True
        )
        return valid_slides

    def _load_annotations(self, annot_masks_path: str | None) -> dict[str, Path]:
        annotations: dict[str, Path] = {}
        if annot_masks_path is None:
            return annotations

        annot_masks = Path(annot_masks_path)
        for annot_mask in annot_masks.rglob("*.pt"):
            if annot_mask.stem in self.slides.slide_id.values:
                annotations[annot_mask.stem] = annot_mask
        return annotations

    def _compute_slides_positivity(self) -> dict[int, float]:
        """Computes the fraction of annotated positive nuclei per slide.

        Returns:
            dict[int, float]: Mapping from slide index to positivity score.
        """
        slides_positivity: dict[int, float] = {}
        for idx in range(len(self.slides)):
            mask_path = self.annotations.get(self.slides.iloc[idx].slide_id, None)
            slides_positivity[idx] = (
                torch.load(mask_path, weights_only=False, mmap=True)
                .float()
                .mean()
                .item()
                if mask_path is not None
                else 0.0
            )
        return slides_positivity

    def _build_graph(self, points: PointArray) -> AdjacencyGraph:
        """Builds an undirected Delaunay-based adjacency graph with edge weights as Euclidean distances."""
        tri = Delaunay(points)
        distances = np.linalg.norm(
            points[tri.simplices[:, [0, 1, 2]]]
            - points[np.roll(tri.simplices[:, [0, 1, 2]], shift=-1, axis=1)],
            axis=2,
        )
        adj_graph = build_adjacency_graph(
            tri.simplices.astype(np.int64),
            distances.astype(np.float32),
            len(points),
        )
        return adj_graph

    def _find_component(
        self,
        idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: PointArray,
        indices: NDArray[np.uint32] | None = None,
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

    def _get_annot_mask(self, idx: int, nuclei_count: int) -> Tensor:
        """Return the annotation mask for a slide.

        For positive slides, the mask is loaded from a saved file.
        For negative slides, all nuclei are considered annotated.
        """
        mask_path = self.annotations.get(self.slides.iloc[idx].slide_id, None)
        if mask_path is not None:
            return (
                torch.load(mask_path, weights_only=False, mmap=True).bool().view(-1, 1)
            )
        return torch.ones((nuclei_count, 1), dtype=torch.bool)

    def _get_slide_metadata(
        self, idx: int, nuclei_count: int, perm_inverse: Tensor
    ) -> Metadata:
        return {
            "slide_id": self.slides.iloc[idx].slide_id,
            "slide_tiff_path": self.slides.iloc[idx].slide_tiff_path,
            "raw_cells_path": self.slides.iloc[idx].raw_cells_path,
            "nuclei_count": nuclei_count,
            "perm_inverse": perm_inverse,
        }

    def _get_crop_indices(self, centroids: PointArray, annot_mask: Tensor) -> list[int]:
        """Build graph and create a crop of size `crop_size` by growing a component starting from a random nucleus.

        Only annotated nuclei are considered as seeds to avoid creating a crop fully in an unlabelled region.
        """
        graph = self._build_graph(centroids)

        annotated_indices = (
            torch.nonzero(annot_mask.squeeze(-1), as_tuple=False).squeeze(-1).tolist()
        )
        assert annotated_indices
        seed = random.choice(annotated_indices)
        return self._find_component(seed, self.crop_size, graph, centroids)

    def _build_kd_tree(self, points: PointArray) -> tuple[KDTree, Tensor]:
        """Builds a KDTree and computes permutation that clusters points by leaf for sparse attention."""
        tree_tmp = KDTree(points, leafsize=self.attn_block_size)
        perm = tree_tmp.indices
        tree = KDTree(points[perm], leafsize=self.attn_block_size)
        return tree, torch.from_numpy(perm).long()

    def _add_rotation(self, positions: Tensor, embeddings: Tensor) -> Tensor:
        """Append the rotation angle as a third positional dimension."""
        _, psi_1, _ = normalize_efd(embeddings, return_angles=True)
        angles = (psi_1 % torch.pi).view(-1, 1)
        return torch.cat([positions, angles], dim=1)

    def _pad_to_block_size(
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

    def _get_inverse_perm(self, perm: Tensor) -> Tensor:
        """Computes the inverse permutation.

        Note: Used for prediction when generating nuclei label masks from the saved data.
        """
        assert perm.dim() == 1, "perm must be 1D"
        perm_inverse = torch.empty_like(perm)
        perm_inverse[perm] = torch.arange(perm.size(0), dtype=perm.dtype)
        assert torch.equal(
            perm[perm_inverse], torch.arange(perm.size(0), dtype=perm.dtype)
        )
        return perm_inverse

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        slide_nuclei_path = self.nuclei[self.slides.iloc[idx].slide_id]
        nuclei_data = torch.load(slide_nuclei_path, weights_only=False, mmap=True)
        nuclei_data["annot_mask"] = self._get_annot_mask(idx, nuclei_data["x"].shape[0])

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
                self._get_crop_indices(positions.numpy(), masks), dtype=torch.long
            )

        crop_embeddings = embeddings[crop_indices]
        crop_positions = positions[crop_indices]
        crop_labels = labels[crop_indices]
        crop_annot_mask = masks[crop_indices]

        tree, perm = self._build_kd_tree(crop_positions.numpy())

        perm_embeddings = crop_embeddings[perm]
        perm_positions = self._add_rotation(crop_positions[perm], crop_embeddings[perm])
        perm_labels = crop_labels[perm]
        perm_annot_mask = crop_annot_mask[perm]

        crop_embeddings, crop_positions, crop_labels, crop_annot_mask = (
            self._pad_to_block_size(
                perm_embeddings, perm_positions, perm_labels, perm_annot_mask
            )
        )

        crop_block_mask = create_single_block_mask_from_kdtree(
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
            "annot_mask": crop_annot_mask,  # (n, 1)
            "block_mask": crop_block_mask,
        }
        if self.predict:
            perm_inverse = self._get_inverse_perm(perm)
            return item, self._get_slide_metadata(idx, len(crop_indices), perm_inverse)
        return item
