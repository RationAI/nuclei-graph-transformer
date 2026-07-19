import numpy as np
import torch
from pandas import DataFrame

from nuclei_graph.data.datasets.crop.base import BaseCropDataset
from nuclei_graph.nuclei_graph_typing import Sample


class PredictionDataset(BaseCropDataset):
    def __init__(
        self,
        metadata: DataFrame,
        embedding_mode: str,
        alpha: float = 0.8,
        efd_order: int = 16,
        patch_size: int | None = None,
    ) -> None:
        super().__init__(
            metadata=metadata,
            alpha=alpha,
            efd_order=efd_order,
            full_slide=True,
            patch_size=patch_size,
            embedding_mode=embedding_mode,
        )

    def __getitem__(self, idx: int) -> Sample:
        slide = self.metadata.iloc[idx]
        nuclei = self.get_nuclei(slide.slide_nuclei_path)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)
        crop_indices = np.arange(len(nuclei), dtype=int)  # full-slide

        # Positions
        crop_pos = centroids[crop_indices]
        crop_pos_centered = (crop_pos - crop_pos.mean(axis=0)).astype(np.float32)

        # Embeddings
        crop_features, crop_bboxes = None, None
        if self.embedding_mode == "efd":
            crop_polygons = np.array(nuclei["polygon"].iloc[crop_indices].tolist())
            crop_features = self.get_efd_features(
                crop_polygons, slide.mpp_x, slide.mpp_y
            )
        elif self.embedding_mode == "spatial":
            crop_features = self.get_spatial_features(centroids[crop_indices])
        elif self.embedding_mode == "efd_spatial":
            crop_polygons = np.array(nuclei["polygon"].iloc[crop_indices].tolist())
            efd_feats = self.get_efd_features(crop_polygons, slide.mpp_x, slide.mpp_y)
            spatial_feats = self.get_spatial_features(centroids[crop_indices])

            efd_to_norm = efd_feats[..., :-2]
            angles = efd_feats[..., -2:]

            crop_features = np.concatenate(
                [efd_to_norm, spatial_feats, angles], axis=-1
            )
        elif self.embedding_mode == "bbox":
            raw_centroids = self.get_centroids(nuclei, 1.0, 1.0)[crop_indices]
            crop_bboxes = self.get_nuclei_bboxes(raw_centroids, slide.slide_path)

        assert (
            crop_features is not None
            or crop_bboxes is not None
            or self.embedding_mode == "pos_only"
        )
        return Sample(
            {
                "features": torch.as_tensor(crop_features, dtype=torch.float32)
                if crop_features is not None
                else None,
                "bboxes": crop_bboxes,
                "labels": {"nuclei": None, "graph": None},
                "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
                "sup_mask": torch.ones(len(crop_indices), dtype=torch.bool),
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                "metadata": {
                    "slide_id": slide.slide_id,
                    "slide_path": slide.slide_path,
                    "slide_nuclei_path": slide.slide_nuclei_path,
                    "nuclei_ids": nuclei["id"].to_numpy(),
                },
            }
        )
