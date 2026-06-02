import numpy as np
import torch
from pandas import DataFrame

from nuclei_graph.data.datasets.base import BaseNucleiDataset
from nuclei_graph.nuclei_graph_typing import PredictSlide


class PredictionDataset(BaseNucleiDataset):
    def __init__(
        self, metadata: DataFrame, alpha: float = 0.8, efd_order: int = 10
    ) -> None:
        super().__init__(
            metadata=metadata,
            supervision=None,
            alpha=alpha,
            efd_order=efd_order,
            full_slide=True,
            random_rotate=False,
        )

    def __getitem__(self, idx: int) -> PredictSlide:
        slide = self.metadata.iloc[idx]
        nuclei = self.get_nuclei(slide.slide_nuclei_path)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

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
