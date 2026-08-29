from collections.abc import Callable
from random import choice

import numpy as np
import torch
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.spatial import KDTree
from torch import Tensor

from nuclei_graph.data.datasets.crop.base import BaseCropDataset
from nuclei_graph.data.supervision import DatasetSupervision
from nuclei_graph.nuclei_graph_typing import Sample


class GraphClassificationDataset(BaseCropDataset):
    def __init__(
        self,
        metadata: DataFrame,
        supervision: DatasetSupervision | None,
        crop_size_min: int,
        crop_size_max: int,
        embedding_mode: str,
        crop_pos_thr: float = 0.75,
        alpha: float = 0.8,
        efd_order: int = 16,
        full_slide: bool = False,
        patch_size: int | None = None,
        augmentations: Callable[..., dict[str, NDArray[np.float32]]] | None = None,
    ) -> None:
        super().__init__(
            metadata=metadata,
            supervision=supervision,
            crop_size_min=crop_size_min,
            crop_size_max=crop_size_max,
            alpha=alpha,
            efd_order=efd_order,
            full_slide=full_slide,
            patch_size=patch_size,
            embedding_mode=embedding_mode,
            augmentations=augmentations,
        )
        self.crop_pos_thr = crop_pos_thr
        self.pos_slide_indices = np.where(self.metadata["is_carcinoma"])[0].tolist()

    def sample_positive_crop(
        self,
        valid_seeds: list[int],
        centroids: NDArray[np.float32],
        targets: Tensor,
        target_size: int,
        max_attempts: int = 10,
        margin: float = 0.15,
    ) -> NDArray[np.int64] | None:
        """Attempts to sample a positive crop of nuclei around a random valid seed."""
        tree = KDTree(centroids)

        for _ in range(max_attempts):
            seed_idx = choice(valid_seeds)

            # heuristic via Euclidean circle
            _, neighbor_idx = tree.query(centroids[seed_idx], k=target_size)
            est_tumor_ratio = (targets[neighbor_idx] == 1).sum().item() / target_size
            if est_tumor_ratio < max(0.0, self.crop_pos_thr - margin):
                continue

            crop_indices = self.get_crop_indices(centroids, [seed_idx], target_size)
            crop_targets = targets[torch.from_numpy(crop_indices).long()]
            tumor_ratio = (crop_targets == 1).sum().item() / len(crop_indices)

            if tumor_ratio > self.crop_pos_thr:
                return crop_indices

        return None

    def __getitem__(self, idx: int) -> Sample:
        slide = self.metadata.iloc[idx]
        nuclei = self.get_nuclei(slide.slide_nuclei_path)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

        crop_indices = np.arange(len(nuclei), dtype=int)
        nuclei_sup = self.get_nuclei_sup(slide.slide_id)

        # Crop Generation
        if not self.full_slide:
            target_size = self.get_crop_size(len(nuclei))

            if not slide.is_carcinoma:
                crop_indices = self.get_crop_indices(
                    centroids, nuclei_sup.get_neg_seeds(len(nuclei)), target_size
                )
            else:
                curr_slide_idx, curr_crop_indices = None, None
                while True:  # ensure crop positivity ≥ `crop_pos_thr`
                    if curr_slide_idx is not None:
                        slide = self.metadata.iloc[curr_slide_idx]
                        nuclei = self.get_nuclei(slide.slide_nuclei_path)
                        nuclei_sup = self.get_nuclei_sup(slide.slide_id)
                        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

                    curr_crop_indices = self.sample_positive_crop(
                        valid_seeds=nuclei_sup.get_pos_seeds(len(nuclei)),
                        centroids=centroids,
                        targets=nuclei_sup.get_targets(len(nuclei)),
                        target_size=self.get_crop_size(len(nuclei)),
                    )
                    if curr_crop_indices is not None:
                        crop_indices = curr_crop_indices
                        break
                    curr_slide_idx = choice(self.pos_slide_indices)
        assert crop_indices is not None

        # Supervision
        crop_indices_t = torch.from_numpy(crop_indices).long()
        crop_nuclei_labels = nuclei_sup.get_targets(len(nuclei))[crop_indices_t]
        crop_graph_label = torch.tensor(
            [float(slide.is_carcinoma)], dtype=torch.float32
        )

        # Geometry & Augmentations
        crop_polygons, crop_pos, crop_pos_centered = self.process_geometry(
            nuclei, crop_indices, centroids, slide
        )

        # Embeddings
        crop_geom_features, crop_bboxes = self.generate_embeddings(
            crop_polygons, crop_pos, slide, centroids[crop_indices]
        )

        return Sample(
            {
                "features": torch.as_tensor(crop_geom_features, dtype=torch.float32)
                if crop_geom_features is not None
                else None,
                "bboxes": crop_bboxes,
                "labels": {"nuclei": crop_nuclei_labels, "graph": crop_graph_label},
                "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
                "sup_mask": nuclei_sup.get_sup_mask(len(nuclei))[crop_indices_t],
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                "metadata": None,
            }
        )
