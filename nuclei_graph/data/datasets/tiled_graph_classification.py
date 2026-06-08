import math
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from random import uniform
from typing import TypeVar, cast

import numpy as np
import pandas as pd
import psutil
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
from nuclei_graph.data.supervision import DatasetSupervision, NucleiSupervision
from nuclei_graph.nuclei_graph_typing import Targets, TileCrop, TileMetadata


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")


@lru_cache(maxsize=3)
def get_slide(nuclei_path: str, mpp_x: float, mpp_y: float):
    nuclei = pd.read_parquet(nuclei_path).sort_values("id").reset_index(drop=True)
    raw_centroids = np.stack(nuclei["centroid"].tolist())
    scaled_centroids = raw_centroids * np.array([mpp_x, mpp_y], dtype=np.float32)
    tree = KDTree(raw_centroids)
    return nuclei, raw_centroids, scaled_centroids, tree


T = TypeVar("T", covariant=True)


def get_slide_name(slide: pd.Series) -> str:
    return Path(slide.path).stem


class FilterableDataset(MetaTiledSlides[T]):
    def __init__(
        self,
        uris: Iterable[str],
        thresholds: dict[str, float],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
    ) -> None:
        self.labeled = carcinoma_roi_t is not None and stratified_filter is not None
        self.stratified_filter = stratified_filter
        self.carcinoma_roi_t = carcinoma_roi_t
        self.thresholds = thresholds
        super().__init__(uris=uris)

    def filter_tiles_by_thresholds(self, tiles: pd.DataFrame) -> pd.DataFrame:
        for percentage in [
            "tissue_roi_percentage",
            "exclude_percentage",
            "another_pathology_percentage",
            "residual_percentage",
            "blur_percentage",
            "folding_percentage",
        ]:
            if percentage in tiles.columns:
                t = percentage.replace("percentage", "t")
                assert t in self.thresholds, f"{t} for {percentage}"
                mask = (
                    tiles[percentage] > self.thresholds[t]
                    if "tissue" in percentage
                    else tiles[percentage] <= self.thresholds[t]
                )
                tiles = tiles[mask]
        return tiles

    def prepare_tiles(self, tiles: pd.DataFrame) -> pd.DataFrame:
        assert self.labeled, "Only allowed for labeled dataset"
        tiles = self.filter_tiles_by_thresholds(tiles)
        tiles["carcinoma"] = tiles["carcinoma_roi_percentage"] > self.carcinoma_roi_t
        if self.stratified_filter:
            tiles = self.filter_non_carcinoma(tiles)
        return tiles

    def filter_non_carcinoma(self, tiles: pd.DataFrame) -> pd.DataFrame:
        assert self.labeled, "Only allowed for labeled dataset"
        tiles_slide_cancer = (
            tiles["slide_id"]
            .map(dict(zip(self.slides["id"], self.slides["carcinoma"], strict=True)))
            .astype(int)
        )
        return tiles[~((tiles_slide_cancer == 1) & (tiles["carcinoma"] == 0))]


T = TypeVar("T", covariant=True)


class NucleiTileDataset(FilterableDataset[T]):
    def __init__(
        self,
        metadata: DataFrame,
        supervision: DatasetSupervision | None,
        uris: Iterable[str],
        thresholds: dict[str, float],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
        window_size: int = 512,
        margin: int = 128,
        efd_order: int = 15,
        random_rotate: bool = False,
    ) -> None:
        self.metadata = metadata
        self.supervision = supervision
        self.window_size = window_size
        self.margin = margin
        self.efd_order = efd_order
        self.random_rotate = random_rotate

        super().__init__(
            uris=uris,
            thresholds=thresholds,
            carcinoma_roi_t=carcinoma_roi_t,
            stratified_filter=stratified_filter,
        )

    def generate_datasets(self) -> Iterable[Dataset[T]]:
        self.tiles = (
            self.prepare_tiles(self.tiles)
            if self.labeled
            else self.filter_tiles_by_thresholds(self.tiles)
        )
        stems = set(self.metadata["slide_id"].values)

        valid_slide_ids = set()
        for _, slide in self.slides.iterrows():
            if slide["stem"] in stems:
                valid_slide_ids.add(slide["id"])

        self.tiles = self.tiles[
            self.tiles["slide_id"].isin(valid_slide_ids)
        ].reset_index(drop=True)

        datasets = []
        for _, slide in self.slides.iterrows():
            if slide["id"] not in valid_slide_ids:
                continue

            slide_meta = self.metadata[self.metadata["slide_id"] == slide["stem"]]
            meta_row = slide_meta.iloc[0]

            slide_tiles = self.tiles[self.tiles["slide_id"] == slide["id"]]

            datasets.append(
                cast(
                    "Dataset[T]",
                    _TileNucleiSlide(
                        slide_metadata=slide,
                        slide_path=meta_row["slide_path"],
                        tiles=slide_tiles,
                        supervision=self.supervision,
                        nuclei_path=meta_row["slide_nuclei_path"],
                        mpp_x=float(meta_row["mpp_x"]),
                        mpp_y=float(meta_row["mpp_y"]),
                        include_label=self.labeled,
                        window_size=self.window_size,
                        margin=self.margin,
                        efd_order=self.efd_order,
                        random_rotate=self.random_rotate,
                    ),
                )
            )
        return datasets


class LabeledNucleiTileDataset(NucleiTileDataset[TileCrop]): ...


class UnlabeledNucleiTileDataset(NucleiTileDataset[TileCrop]): ...


class _TileNucleiSlide(Dataset[TileCrop]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        slide_path: str,
        tiles: pd.DataFrame,
        supervision: DatasetSupervision | None,
        nuclei_path: Path | str,
        mpp_x: float,
        mpp_y: float,
        include_label: bool,
        window_size: int,
        margin: int,
        efd_order: int,
        random_rotate: bool | None = False,
    ) -> None:
        super().__init__()
        self.slide_metadata = slide_metadata
        self.slide_path = slide_path
        self.tiles = tiles
        self.supervision = supervision
        self.include_label = include_label
        self.window_size = window_size
        self.margin = margin
        self.efd_order = efd_order
        self.mpp_x = mpp_x
        self.mpp_y = mpp_y
        self.random_rotate = random_rotate
        self.nuclei_path = nuclei_path
        self.scale_x, self.scale_y = self.tile2nuclei_ratio()

    def tile2nuclei_ratio(self) -> tuple[float, float]:
        with OpenSlide(self.slide_path) as slide:
            size_base = slide.level_dimensions[0]
            size = slide.level_dimensions[self.slide_metadata["level"]]
            scale_x = size[0] / size_base[0]
            scale_y = size[1] / size_base[1]
        return scale_x, scale_y

    def __len__(self) -> int:
        return len(self.tiles)

    def get_centroids(
        self,
        nuclei: pd.DataFrame,
        mpp_x: float | None = None,
        mpp_y: float | None = None,
    ) -> np.ndarray:
        if mpp_x is not None and mpp_y is not None:
            mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
            return np.stack(nuclei["centroid"].tolist()) * mpps
        return np.stack(nuclei["centroid"].tolist())

    def get_nuclei_sup(self, slide_id: str) -> NucleiSupervision:
        assert self.supervision is not None
        return self.supervision.supervision_map[slide_id].nuclei_supervision

    def random_rotate_graph(
        self,
        pos: NDArray[np.float32],
        cos_angles: NDArray[np.float32],
        sin_angles: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        theta = uniform(0, 2 * math.pi)

        rotation_matrix = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float32,
        )
        rotated_pos = pos @ rotation_matrix.T

        # the original angles are doubled
        c2 = math.cos(2 * theta)
        s2 = math.sin(2 * theta)

        rotated_cos = (cos_angles * c2 - sin_angles * s2).astype(np.float32)
        rotated_sin = (sin_angles * c2 + cos_angles * s2).astype(np.float32)

        return rotated_pos, rotated_cos, rotated_sin

    def get_features(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> NDArray[np.float32]:
        mpps = np.array([mpp_x, mpp_y], dtype=np.float32)
        contours = rearrange(polygons, "b (v d) -> b v d", d=2) * mpps
        efds = elliptic_fourier_descriptors(contours.astype(np.float64), self.efd_order)

        efds, angles = normalize_efd_for_rotation(efds)
        cos_angles = np.cos(2.0 * angles)
        sin_angles = np.sin(2.0 * angles)

        efds, scales = normalize_efd_for_scale(efds)
        log_scales = np.log(scales + 1e-6)

        efds = rearrange(efds, "n order c -> n (order c)")
        features = np.concatenate([efds, log_scales, cos_angles, sin_angles], axis=-1)
        return features.astype(np.float32)

    def __getitem__(self, idx: int) -> TileCrop:
        crop_indices = np.array([], dtype=np.int64)

        tile = self.tiles.iloc[idx]

        x_min = tile["x"] / self.scale_x
        y_min = tile["y"] / self.scale_y

        x_extent = self.slide_metadata.tile_extent_x / self.scale_x
        y_extent = self.slide_metadata.tile_extent_y / self.scale_y

        x_max = x_min + x_extent
        y_max = y_min + y_extent

        nuclei_df, raw_centroids, scaled_centroids, kdtree = get_slide(
            str(self.nuclei_path), self.mpp_x, self.mpp_y
        )

        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        radius = np.sqrt((x_extent / 2) ** 2 + (y_extent / 2) ** 2)
        candidates = np.array(kdtree.query_ball_point(center, radius), dtype=np.int64)

        crop_indices = np.array([], dtype=np.int64)
        if len(candidates) > 0:
            cx = raw_centroids[candidates, 0]
            cy = raw_centroids[candidates, 1]
            crop_mask = (cx >= x_min) & (cx < x_max) & (cy >= y_min) & (cy < y_max)
            crop_indices = candidates[crop_mask]

        nuclei = nuclei_df.iloc[crop_indices].reset_index(drop=True)

        crop_centroids_microns = scaled_centroids[crop_indices]
        crop_centroids_pixels = raw_centroids[crop_indices]

        # Supervision
        if self.supervision is not None:
            nuclei_sup = self.supervision.supervision_map[
                self.slide_metadata["stem"]
            ].nuclei_supervision
            global_sup_mask = nuclei_sup.get_sup_mask(len(nuclei_df))
            nuclei_targets = nuclei_sup.get_targets(len(nuclei_df))
            crop_sup_mask = torch.as_tensor(global_sup_mask[crop_indices])
            crop_nuclei_labels = torch.as_tensor(nuclei_targets[crop_indices])
        else:
            crop_sup_mask = torch.ones(len(crop_indices), dtype=torch.bool)
            crop_nuclei_labels = None

        # ROI Mask
        margin_x = x_extent * (self.margin / self.window_size)
        margin_y = y_extent * (self.margin / self.window_size)
        roi_mask = (
            (crop_centroids_pixels[:, 0] >= x_min + margin_x)
            & (crop_centroids_pixels[:, 0] < x_max - margin_x)
            & (crop_centroids_pixels[:, 1] >= y_min + margin_y)
            & (crop_centroids_pixels[:, 1] < y_max - margin_y)
        )
        roi_mask_t = torch.from_numpy(roi_mask).bool()

        crop_labels: Targets = {"nuclei": crop_nuclei_labels, "graph": None}
        if self.include_label:
            crop_labels["graph"] = torch.tensor(
                [float(tile["carcinoma"])], dtype=torch.float32
            )

        if len(crop_indices) == 0:
            crop_features = np.zeros((0, self.efd_order * 4 + 3), dtype=np.float32)
            crop_pos_centered = np.zeros((0, 2), dtype=np.float32)
        else:
            crop_polygons = np.array(nuclei["polygon"].tolist())
            crop_features = self.get_features(crop_polygons, self.mpp_x, self.mpp_y)
            crop_pos_centered = (
                crop_centroids_microns - crop_centroids_microns.mean(axis=0)
            ).astype(np.float32)

            if self.random_rotate:
                pos_rot, cos_rot, sin_rot = self.random_rotate_graph(
                    crop_pos_centered, crop_features[..., -2], crop_features[..., -1]
                )
                crop_pos_centered = pos_rot
                crop_features[..., -2] = cos_rot
                crop_features[..., -1] = sin_rot

        metadata: TileMetadata = {
            "slide": self.slide_metadata["stem"],
            "x": int(x_min),
            "y": int(y_min),
        }

        return TileCrop(
            {
                "features": crop_features,
                "labels": crop_labels,
                "pos": crop_pos_centered,
                "sup_mask": crop_sup_mask,
                "roi_mask": roi_mask_t,
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                "metadata": metadata,
            }
        )
