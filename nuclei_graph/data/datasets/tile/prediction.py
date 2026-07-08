from collections.abc import Iterable

import numpy as np
import torch
from pandas import DataFrame

from nuclei_graph.data.datasets.tile.base import (
    BaseTileDataset,
    get_slide_data,
)
from nuclei_graph.nuclei_graph_typing import Sample, Targets


class TilePredictionDataset(BaseTileDataset):
    def __init__(
        self,
        metadata: DataFrame,
        uris: Iterable[str],
        thresholds: dict[str, float],
        tile_size: int,
        margin: float | None = None,
        efd_order: int = 16,
    ) -> None:
        super().__init__(
            metadata, uris, thresholds, efd_order, False, tile_size, margin
        )

    def __getitem__(self, idx: int) -> Sample:
        tile = self.tiles.iloc[idx]
        props = self.slide_props[tile["stem"]]
        scaled_props = self.get_scaled_props(tile)

        # Tile-Crop Generation
        nuclei_path = self.slide_props[tile["stem"]]["slide_nuclei_path"]
        polygons, centroids, centroid_tree, nuclei_ids = get_slide_data(nuclei_path)
        tile_indices = self.get_tile_indices(scaled_props, centroids, centroid_tree)
        tile_labels: Targets = {"nuclei": None, "graph": None}

        # EFD Computation
        if len(tile_indices) == 0:  # empty tile, no nuclei
            tile_features = np.zeros((1, self.efd_order * 4 + 3), dtype=np.float32)
            tile_pos_centered = np.zeros((1, 2), dtype=np.float32)
            tile_sup_mask = torch.tensor([False], dtype=torch.bool)
            tile_nuclei_ids = np.full(1, -1, dtype=nuclei_ids.dtype)
            seq_len = 1
        else:
            tile_polygons = polygons[tile_indices]
            tile_features = self.get_efd_features(
                tile_polygons, props["mpp_x"], props["mpp_y"]
            )
            scaled_centroids = centroids * np.array(
                [props["mpp_x"], props["mpp_y"]], dtype=np.float32
            )
            tile_pos = scaled_centroids[tile_indices]
            tile_pos_centered = tile_pos - tile_pos.mean(axis=0)

            tile_sup_mask = torch.ones(len(tile_indices), dtype=torch.bool)
            tile_nuclei_ids = nuclei_ids[tile_indices]
            seq_len = len(tile_indices)

        return Sample(
            {
                "features": torch.as_tensor(tile_features, dtype=torch.float32),
                "bboxes": None,
                "labels": tile_labels,
                "pos": torch.as_tensor(tile_pos_centered, dtype=torch.float32),
                "sup_mask": tile_sup_mask,
                "seq_len": torch.tensor(seq_len, dtype=torch.int32),
                "metadata": {
                    "slide_id": str(tile["stem"]),
                    "x": int(tile["x"]),
                    "y": int(tile["y"]),
                    "nuclei_ids": tile_nuclei_ids,
                },
            }
        )
