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
        crop_geom_features, crop_bboxes = self.generate_embeddings(
            nuclei, crop_indices, centroids, slide
        )

        return Sample(
            {
                "features": torch.as_tensor(crop_geom_features, dtype=torch.float32)
                if crop_geom_features is not None
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
