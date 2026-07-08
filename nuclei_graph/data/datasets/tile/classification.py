import math
from collections.abc import Iterable
from random import uniform

import numpy as np
import torch
from pandas import DataFrame

from nuclei_graph.data.datasets.tile.base import (
    BaseTileDataset,
    get_slide_data,
)
from nuclei_graph.data.supervision import DatasetSupervision
from nuclei_graph.nuclei_graph_typing import Sample, Targets


class TileClassificationDataset(BaseTileDataset):
    def __init__(
        self,
        metadata: DataFrame,
        uris: Iterable[str],
        thresholds: dict[str, float],
        supervision: DatasetSupervision,
        tile_size: int,
        random_rotate: bool | None = False,
        margin: float | None = None,
        efd_order: int = 16,
        carcinoma_filter: bool = True,
    ) -> None:
        super().__init__(
            metadata, uris, thresholds, efd_order, carcinoma_filter, tile_size, margin
        )
        self.supervision = supervision
        self.random_rotate = random_rotate

    def random_rotate_graph(self, pos, cos_angles, sin_angles):
        theta = uniform(0, 2 * math.pi)
        rotation_matrix = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float32,
        )
        rotated_pos = pos @ rotation_matrix.T
        c2, s2 = math.cos(2 * theta), math.sin(2 * theta)
        rotated_cos = (cos_angles * c2 - sin_angles * s2).astype(np.float32)
        rotated_sin = (sin_angles * c2 + cos_angles * s2).astype(np.float32)
        return rotated_pos, rotated_cos, rotated_sin

    def __getitem__(self, idx: int) -> Sample:
        tile = self.tiles.iloc[idx]
        props = self.slide_props[tile["stem"]]
        scaled_props = self.get_scaled_props(tile)

        nuclei_path = self.slide_props[tile["stem"]]["slide_nuclei_path"]
        polygons, centroids, centroid_tree, _ = get_slide_data(nuclei_path)
        tile_indices = self.get_tile_indices(scaled_props, centroids, centroid_tree)

        nuclei_sup = self.supervision.supervision_map[tile["stem"]].nuclei_supervision
        tile_sup_mask = nuclei_sup.get_sup_mask(len(centroids))[tile_indices]
        tile_nuclei_labels = nuclei_sup.get_targets(len(centroids))[tile_indices]
        assert tile.get("carcinoma") is not None, "Tile carcinoma label is required."
        tile_labels: Targets = {
            "nuclei": torch.as_tensor(tile_nuclei_labels),
            "graph": torch.tensor([float(tile["carcinoma"])]),
        }

        if len(tile_indices) == 0:
            tile_features = np.zeros((1, self.efd_order * 4 + 3), dtype=np.float32)
            tile_pos_centered = np.zeros((1, 2), dtype=np.float32)

            tile_sup_mask = np.array([False], dtype=bool)
            tile_labels["nuclei"] = torch.tensor([0.0], dtype=torch.float32)

            seq_len = 1
        else:
            tile_features = self.get_features(
                polygons[tile_indices], props["mpp_x"], props["mpp_y"]
            )
            scaled_centroids = centroids * np.array(
                [props["mpp_x"], props["mpp_y"]], dtype=np.float32
            )
            tile_pos = scaled_centroids[tile_indices]
            tile_pos_centered = tile_pos - tile_pos.mean(axis=0)

            if self.random_rotate:
                pos_rot, cos_rot, sin_rot = self.random_rotate_graph(
                    tile_pos_centered, tile_features[..., -2], tile_features[..., -1]
                )
                tile_pos_centered = pos_rot
                tile_features[..., -2], tile_features[..., -1] = cos_rot, sin_rot

            seq_len = len(tile_indices)

        return Sample(
            {
                "features": torch.as_tensor(tile_features, dtype=torch.float32),
                "labels": tile_labels,
                "pos": torch.as_tensor(tile_pos_centered, dtype=torch.float32),
                "sup_mask": torch.as_tensor(tile_sup_mask),
                "seq_len": torch.tensor(seq_len, dtype=torch.int32),
                "metadata": {
                    "slide_id": tile["stem"],
                    "x": int(tile["x"]),
                    "y": int(tile["y"]),
                },
            }
        )
