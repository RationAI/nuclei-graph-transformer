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
from nuclei_graph.nuclei_graph_typing import Targets, TileGraph


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

    def __getitem__(self, idx: int) -> TileGraph:
        tile = self.tiles.iloc[idx]
        props = self.slide_props[tile["stem"]]
        scaled_props = self.get_scaled_props(tile)

        # Tile-Crop Generation
        nuclei_path = self.slide_props[tile["stem"]]["slide_nuclei_path"]
        polygons, centroids, centroid_tree = get_slide_data(nuclei_path)
        crop_indices = self.get_crop_indices(scaled_props, centroids, centroid_tree)

        # Supervision
        nuclei_sup = self.supervision.supervision_map[tile["stem"]].nuclei_supervision
        crop_sup_mask = nuclei_sup.get_sup_mask(len(centroids))[crop_indices]
        crop_nuclei_labels = nuclei_sup.get_targets(len(centroids))[crop_indices]
        assert tile.get("carcinoma") is not None, "Tile carcinoma label is required."
        crop_labels: Targets = {
            "nuclei": torch.as_tensor(crop_nuclei_labels),
            "graph": torch.tensor([float(tile["carcinoma"])]),
        }

        # EFD Computation
        if len(crop_indices) == 0:
            crop_features = np.zeros((0, self.efd_order * 4 + 3), dtype=np.float32)
            crop_pos_centered = np.zeros((0, 2), dtype=np.float32)
        else:
            crop_features = self.get_features(
                polygons[crop_indices], props["mpp_x"], props["mpp_y"]
            )
            scaled_centroids = centroids * np.array(
                [props["mpp_x"], props["mpp_y"]], dtype=np.float32
            )
            crop_pos = scaled_centroids[crop_indices]
            crop_pos_centered = crop_pos - crop_pos.mean(axis=0)

            if self.random_rotate:
                pos_rot, cos_rot, sin_rot = self.random_rotate_graph(
                    crop_pos_centered, crop_features[..., -2], crop_features[..., -1]
                )
                crop_pos_centered = pos_rot
                crop_features[..., -2], crop_features[..., -1] = cos_rot, sin_rot

        return {
            "features": torch.as_tensor(crop_features, dtype=torch.float32),
            "labels": crop_labels,
            "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
            "sup_mask": torch.as_tensor(crop_sup_mask),
            "roi_mask": torch.from_numpy(
                self.get_roi_mask(scaled_props, centroids[crop_indices])
            ),
            "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
            "metadata": {
                "slide": tile["stem"],
                "x": int(tile["x"]),
                "y": int(tile["y"]),
            },
        }
