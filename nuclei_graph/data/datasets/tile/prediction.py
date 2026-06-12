from collections.abc import Iterable

import numpy as np
import torch
from pandas import DataFrame

from nuclei_graph.data.datasets.tile.base import (
    BaseTileDataset,
    get_slide_data,
)
from nuclei_graph.nuclei_graph_typing import Sample


class TilePredictionDataset(BaseTileDataset):
    def __init__(
        self,
        metadata: DataFrame,
        uris: Iterable[str],
        thresholds: dict[str, float],
        tile_size: int,
        margin: float | None = None,
        efd_order: int = 16,
        carcinoma_filter: bool = False,
    ) -> None:
        super().__init__(
            metadata, uris, thresholds, efd_order, carcinoma_filter, tile_size, margin
        )

    def __getitem__(self, idx: int) -> Sample:
        tile = self.tiles.iloc[idx]
        props = self.slide_props[tile["stem"]]
        scaled_props = self.get_scaled_props(tile)

        # Tile-Crop Generation
        nuclei_path = self.slide_props[tile["stem"]]["slide_nuclei_path"]
        polygons, centroids, centroid_tree = get_slide_data(nuclei_path)
        crop_indices = self.get_crop_indices(scaled_props, centroids, centroid_tree)

        # EFD Computation
        if len(crop_indices) == 0:
            crop_features = np.zeros((0, self.efd_order * 4 + 3), dtype=np.float32)
            crop_pos_centered = np.zeros((0, 2), dtype=np.float32)
        else:
            crop_polygons = polygons[crop_indices]
            crop_features = self.get_features(
                crop_polygons, props["mpp_x"], props["mpp_y"]
            )
            scaled_centroids = centroids * np.array(
                [props["mpp_x"], props["mpp_y"]], dtype=np.float32
            )
            crop_pos = scaled_centroids[crop_indices]
            crop_pos_centered = crop_pos - crop_pos.mean(axis=0)

        return Sample(
            {
                "features": torch.as_tensor(crop_features, dtype=torch.float32),
                "labels": {"nuclei": None, "graph": None},
                "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
                "sup_mask": torch.ones(len(crop_indices), dtype=torch.bool),
                "roi_mask": torch.from_numpy(
                    self.get_roi_mask(scaled_props, centroids[crop_indices])
                ),
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                "metadata": {
                    "slide_id": tile["stem"],
                    "x": int(tile["x"]),
                    "y": int(tile["y"]),
                },
            }
        )
