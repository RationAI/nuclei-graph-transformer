import math
from abc import abstractmethod
from collections.abc import Iterable
from functools import lru_cache
from random import uniform
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from numpy.typing import NDArray
from pandas import DataFrame
from rationai.mlkit.data.datasets import MetaTiledSlides
from ratiopath.openslide import OpenSlide
from scipy.spatial import KDTree
from torch.utils.data import Dataset

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)
from nuclei_graph.data.supervision import DatasetSupervision
from nuclei_graph.nuclei_graph_typing import Targets, TileCrop


PERCENTAGES = [
    "tissue_roi_percentage",
    "exclude_percentage",
    "another_pathology_percentage",
    "residual_percentage",
    "blur_percentage",
    "folding_percentage",
]
CARCINOMA_P = "carcinoma_roi_percentage"
T = TypeVar("T", covariant=True)


@lru_cache(maxsize=4)
def get_slide_data(nuclei_path: str) -> tuple[np.ndarray, np.ndarray, KDTree]:
    nuclei = pd.read_parquet(nuclei_path).sort_values("id").reset_index(drop=True)
    polygons = np.array(nuclei["polygon"].tolist(), dtype=np.float32)
    centroids = np.stack(nuclei["centroid"].tolist())
    kdtree = KDTree(centroids)
    return polygons, centroids, kdtree


class BaseTileDataset(MetaTiledSlides[T]):
    def __init__(
        self,
        metadata: DataFrame,
        uris: Iterable[str],
        thresholds: dict[str, float],
        efd_order: int,
        carcinoma_filter: bool,
        tile_size: int = 512,
        margin: float | None = None,
    ) -> None:
        super().__init__(uris=uris)
        self.metadata = metadata.set_index("slide_id")
        self.thresholds = thresholds
        self.efd_order = efd_order
        self.carcinoma_filter = (
            carcinoma_filter  # should only be used in the training mode
        )
        self.tile_size = tile_size
        self.margin = margin if margin is not None else tile_size / 4
        self.carcinoma_t = thresholds.get("carcinoma_roi_t")
        self.slide_props = self.get_slide_properties()

        # self.tiles metadataset is created in the parent class
        self.prepare_tiles()

    def __len__(self) -> int:
        return len(self.tiles)

    def generate_datasets(self) -> Iterable[Dataset[T]]:
        return [self]  # placeholder

    def get_slide_properties(self):
        slide_props = {}
        unique_stems = self.tiles["stem"].unique()

        for stem in unique_stems:
            slide_row = self.slides[self.slides["stem"] == stem].iloc[0]
            meta_row = self.metadata.loc[stem]

            # compute scaling factors for transforming from tile-level space to polygon space
            with OpenSlide(meta_row["slide_path"]) as slide:
                size_base = slide.level_dimensions[0]
                tile_level = slide_row["level"]
                size_level = slide.level_dimensions[tile_level]

                scale_x = size_level[0] / size_base[0]
                scale_y = size_level[1] / size_base[1]

            slide_props[stem] = {
                "scale_x": scale_x,
                "scale_y": scale_y,
                "scaled_extent_x": slide_row["tile_extent_x"] / scale_x,
                "scaled_extent_y": slide_row["tile_extent_y"] / scale_y,
                "mpp_x": float(meta_row["mpp_x"]),
                "mpp_y": float(meta_row["mpp_y"]),
                "slide_nuclei_path": str(meta_row["slide_nuclei_path"]),
            }
        return slide_props

    def filter_tiles_by_thresholds(self, tiles: pd.DataFrame) -> pd.DataFrame:
        for percentage in PERCENTAGES:
            if percentage in tiles.columns:
                key = percentage.replace("percentage", "t")
                assert key in self.thresholds

                mask = (
                    tiles[percentage] > self.thresholds[key]
                    if "tissue" in percentage
                    else tiles[percentage] <= self.thresholds[key]
                )
                tiles = tiles[mask]
        return tiles

    def prepare_tiles(self):
        # filter out slides that might have ended in a different data split (train/val set)
        self.tiles = self.tiles[
            self.tiles["stem"].isin(set(self.metadata.index))
        ].reset_index(drop=True)
        self.tiles = self.filter_tiles_by_thresholds(self.tiles)

        if self.carcinoma_t is not None:
            self.tiles["is_carcinoma"] = self.tiles[CARCINOMA_P] > self.carcinoma_t

            if self.carcinoma_filter:
                # filter out tiles from positive slides that don't meet the threshold
                slide_label = self.tiles["slide_id"].map(
                    dict(zip(self.slides["id"], self.slides["carcinoma"], strict=True))
                )
                mask = ~(
                    (slide_label.astype(int) == 1) & (self.tiles["is_carcinoma"] == 0)
                )
                self.tiles = self.tiles[mask]

    def get_features(self, polygons, mpp_x, mpp_y):
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        contours = rearrange(polygons, "b (v d) -> b v d", d=2) * mpps

        efds = elliptic_fourier_descriptors(contours.astype(np.float64), self.efd_order)
        efds, angles = normalize_efd_for_rotation(efds)
        cos_angles, sin_angles = np.cos(2.0 * angles), np.sin(2.0 * angles)
        efds, scales = normalize_efd_for_scale(efds)
        log_scales = np.log(scales + 1e-6)
        efds = rearrange(efds, "n order c -> n (order c)")

        return np.concatenate(
            [efds, log_scales, cos_angles, sin_angles], axis=-1
        ).astype(np.float32)

    def get_scaled_props(self, tile: pd.Series) -> dict[str, float]:
        """Computes tile properties in the coordinate space of the nuclei polygons."""
        props = self.slide_props[tile["stem"]]

        x_min = tile["x"] / props["scale_x"]
        y_min = tile["y"] / props["scale_y"]
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + props["scaled_extent_x"],
            "y_max": y_min + props["scaled_extent_y"],
            "x_extent": props["scaled_extent_x"],
            "y_extent": props["scaled_extent_y"],
        }

    def get_crop_indices(
        self, props: dict[str, float], centroids: NDArray[np.float32], tree: KDTree
    ) -> NDArray[np.int64]:
        center_x = props["x_min"] + props["x_extent"] / 2
        center_y = props["y_min"] + props["y_extent"] / 2
        center = np.array([center_x, center_y], dtype=np.float32)
        radius = np.sqrt((props["x_extent"] / 2) ** 2 + (props["y_extent"] / 2) ** 2)

        candidates = np.array(tree.query_ball_point(center, radius), dtype=np.int64)

        crop_indices = np.array([], dtype=np.int64)
        if len(candidates) > 0:
            cx, cy = centroids[candidates, 0], centroids[candidates, 1]
            crop_mask = (
                (cx >= props["x_min"])
                & (cx < props["x_max"])
                & (cy >= props["y_min"])
                & (cy < props["y_max"])
            )
            crop_indices = candidates[crop_mask]
        return crop_indices

    def get_roi_mask(
        self, props: dict[str, float], centroids: NDArray[np.float32]
    ) -> NDArray[np.bool_]:
        margin_x, margin_y = (
            props["x_extent"] * (self.margin / self.tile_size),
            props["y_extent"] * (self.margin / self.tile_size),
        )
        return (
            (centroids[:, 0] >= props["x_min"] + margin_x)
            & (centroids[:, 0] < props["x_max"] - margin_x)
            & (centroids[:, 1] >= props["y_min"] + margin_y)
            & (centroids[:, 1] < props["y_max"] - margin_y)
        )

    @abstractmethod
    def __getitem__(self, idx: int) -> TileCrop:
        raise NotImplementedError


class TileClassificationDataset(BaseTileDataset):
    def __init__(
        self,
        metadata: DataFrame,
        uris: Iterable[str],
        thresholds: dict[str, float],
        supervision: DatasetSupervision,
        random_rotate: bool,
        tile_size: int,
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

    def __getitem__(self, idx: int) -> TileCrop:
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
        assert tile.get("is_carcinoma") is not None, "Tile carcinoma label is required."
        crop_labels: Targets = {
            "nuclei": torch.as_tensor(crop_nuclei_labels),
            "graph": torch.tensor(float(tile["is_carcinoma"])),
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
                "x": int(scaled_props["x_min"]),
                "y": int(scaled_props["y_min"]),
            },
        }


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

    def __getitem__(self, idx: int) -> TileCrop:
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

        return {
            "features": torch.as_tensor(crop_features, dtype=torch.float32),
            "labels": {"nuclei": None, "graph": None},
            "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
            "sup_mask": torch.ones(len(crop_indices), dtype=torch.bool),
            "roi_mask": torch.from_numpy(
                self.get_roi_mask(scaled_props, centroids[crop_indices])
            ),
            "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
            "metadata": {
                "slide": tile["stem"],
                "x": int(scaled_props["x_min"]),
                "y": int(scaled_props["y_min"]),
            },
        }
