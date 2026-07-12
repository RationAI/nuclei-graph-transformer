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
        embedding_mode: str,
        margin: float | None = None,
        efd_order: int = 16,
        patch_size: int | None = None,
    ) -> None:
        super().__init__(
            metadata,
            uris,
            thresholds,
            efd_order,
            embedding_mode,
            carcinoma_filter=False,
            tile_size=tile_size,
            margin=margin,
            patch_size=patch_size,
        )

    def __getitem__(self, idx: int) -> Sample:
        tile = self.tiles.iloc[idx]
        props = self.slide_props[tile["stem"]]
        scaled_props = self.get_scaled_props(tile)

        # Nuclei Data
        nuclei_path = self.slide_props[tile["stem"]]["slide_nuclei_path"]
        polygons, centroids, centroid_tree, nuclei_ids = get_slide_data(nuclei_path)
        tile_indices = self.get_tile_indices(scaled_props, centroids, centroid_tree)

        # Supervision
        graph_label = float(tile.get("is_carcinoma", tile.get("carcinoma", 0.0)))
        tile_labels: Targets = {
            "nuclei": None,
            "graph": torch.tensor([graph_label], dtype=torch.float32),
        }

        # Embeddings
        tile_features, tile_bboxes = None, None

        if len(tile_indices) == 0:  # empty tile, no nuclei
            if self.embedding_mode == "efd":
                tile_features = np.zeros((1, self.efd_order * 4 + 3), dtype=np.float32)
            elif self.embedding_mode == "spatial":
                tile_features = np.zeros((1, 8), dtype=np.float32)
            elif self.embedding_mode == "bbox":
                assert self.patch_size is not None
                tile_bboxes = torch.zeros(
                    (1, 3, self.patch_size, self.patch_size), dtype=torch.uint8
                )
            tile_pos_centered = np.zeros((1, 2), dtype=np.float32)
            tile_sup_mask = torch.tensor([False], dtype=torch.bool)
            tile_nuclei_ids = np.full(1, -1, dtype=nuclei_ids.dtype)
            seq_len = 1

        else:
            scaled_centroids = centroids[tile_indices] * np.array(
                [props["mpp_x"], props["mpp_y"]], dtype=np.float32
            )
            if self.embedding_mode == "efd":
                tile_features = self.get_efd_features(
                    polygons[tile_indices], props["mpp_x"], props["mpp_y"]
                )
            elif self.embedding_mode == "spatial":
                tile_features = self.get_spatial_features(scaled_centroids)
            elif self.embedding_mode == "bbox":
                tile_bboxes = self.get_nuclei_bboxes(
                    centroids[tile_indices], props["slide_path"]
                )
            tile_pos_centered = scaled_centroids - scaled_centroids.mean(axis=0)
            tile_sup_mask = torch.ones(len(tile_indices), dtype=torch.bool)
            tile_nuclei_ids = nuclei_ids[tile_indices]
            seq_len = len(tile_indices)

        assert (
            tile_features is not None
            or tile_bboxes is not None
            or self.embedding_mode == "pos_only"
        )
        return Sample(
            {
                "features": torch.as_tensor(tile_features, dtype=torch.float32)
                if tile_features is not None
                else None,
                "bboxes": tile_bboxes,
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
