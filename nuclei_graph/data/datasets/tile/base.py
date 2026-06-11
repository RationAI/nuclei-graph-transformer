from abc import abstractmethod
from collections.abc import Iterable
from functools import lru_cache
from typing import TypeVar

import numpy as np
import pandas as pd
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
from nuclei_graph.nuclei_graph_typing import TileGraph


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
        self.carcinoma_filter = carcinoma_filter  # only in training mode
        self.tile_size = tile_size
        self.margin = margin if margin is not None else tile_size / 4
        self.carcinoma_t = thresholds.get("carcinoma_roi_t")

        # self.tiles metadataset is created in the parent class
        self.prepare_tiles()

        self.slide_props = self.get_slide_properties()

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
        id_to_stem = dict(zip(self.slides["id"], self.slides["stem"], strict=True))
        self.tiles["stem"] = self.tiles["slide_id"].map(id_to_stem)

        # filter out slides that might have ended in a different data split (train/val set)
        self.tiles = self.tiles[
            self.tiles["stem"].isin(set(self.metadata.index))
        ].reset_index(drop=True)
        self.tiles = self.filter_tiles_by_thresholds(self.tiles)

        if self.carcinoma_t is not None:
            self.tiles["carcinoma"] = self.tiles[CARCINOMA_P] > self.carcinoma_t

            if self.carcinoma_filter:
                # filter out tiles from positive slides that don't meet the threshold
                id_to_carcinoma = dict(
                    zip(self.slides["id"], self.slides["carcinoma"], strict=True)
                )
                slide_label = self.tiles["slide_id"].map(id_to_carcinoma)
                mask = ~(
                    (slide_label.astype(int) == 1) & (self.tiles["carcinoma"] == 0)
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
    def __getitem__(self, idx: int) -> TileGraph:
        raise NotImplementedError
