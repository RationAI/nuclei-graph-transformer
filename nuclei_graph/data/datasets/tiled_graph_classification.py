import math
import os
from collections.abc import Iterable
from functools import lru_cache
from random import uniform
from typing import TypeVar

import numpy as np
import pandas as pd
import psutil
import torch
from einops import rearrange
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


@lru_cache(maxsize=4)
def get_slide_data(nuclei_path: str, mpp_x: float, mpp_y: float):
    nuclei = pd.read_parquet(nuclei_path).sort_values("id").reset_index(drop=True)
    raw_centroids = np.stack(nuclei["centroid"].tolist())
    scaled_centroids = raw_centroids * np.array([mpp_x, mpp_y], dtype=np.float32)
    tree = KDTree(raw_centroids)
    raw_polygons = np.array(nuclei["polygon"].tolist(), dtype=np.float32)
    return raw_centroids, scaled_centroids, tree, raw_polygons


T = TypeVar("T", covariant=True)


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

    def generate_datasets(self) -> Iterable[Dataset[T]]:
        return [self]


class NucleiTileDataset(FilterableDataset[dict]):
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
        self.metadata = metadata.set_index("slide_id")
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

        self.tiles = (
            self.prepare_tiles(self.tiles)
            if self.labeled
            else self.filter_tiles_by_thresholds(self.tiles)
        )

        id_to_stem = dict(zip(self.slides["id"], self.slides["stem"], strict=True))
        self.tiles["stem"] = self.tiles["slide_id"].map(id_to_stem)

        valid_stems = set(self.metadata.index)
        self.tiles = self.tiles[self.tiles["stem"].isin(valid_stems)].reset_index(
            drop=True
        )

        if len(self.tiles) == 0:
            raise ValueError("Dataset initialization resulted in 0 tiles.")

        self.slide_props = {}
        unique_stems = self.tiles["stem"].unique()

        for stem in unique_stems:
            slide_row = self.slides[self.slides["stem"] == stem].iloc[0]
            meta_row = self.metadata.loc[stem]

            with OpenSlide(meta_row["slide_path"]) as slide:
                size_base = slide.level_dimensions[0]
                tile_level = slide_row["level"]
                size_level = slide.level_dimensions[tile_level]

                scale_x = size_level[0] / size_base[0]
                scale_y = size_level[1] / size_base[1]

            self.slide_props[stem] = {
                "scale_x": scale_x,
                "scale_y": scale_y,
                "tile_extent_x": slide_row["tile_extent_x"],
                "tile_extent_y": slide_row["tile_extent_y"],
                "mpp_x": float(meta_row["mpp_x"]),
                "mpp_y": float(meta_row["mpp_y"]),
                "slide_nuclei_path": str(meta_row["slide_nuclei_path"]),
            }

    def __len__(self) -> int:
        return len(self.tiles)

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

    def __getitem__(self, idx: int) -> dict:
        tile = self.tiles.iloc[idx]
        stem = tile["stem"]

        props = self.slide_props[stem]
        scale_x, scale_y = props["scale_x"], props["scale_y"]
        mpp_x, mpp_y = props["mpp_x"], props["mpp_y"]

        x_min, y_min = tile["x"] / scale_x, tile["y"] / scale_y
        x_extent, y_extent = (
            props["tile_extent_x"] / scale_x,
            props["tile_extent_y"] / scale_y,
        )
        x_max, y_max = x_min + x_extent, y_min + y_extent

        raw_centroids, scaled_centroids, kdtree, raw_polygons = get_slide_data(
            props["slide_nuclei_path"], mpp_x, mpp_y
        )

        # Crop Generation
        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        radius = np.sqrt((x_extent / 2) ** 2 + (y_extent / 2) ** 2)
        candidates = np.array(kdtree.query_ball_point(center, radius), dtype=np.int64)

        crop_indices = np.array([], dtype=np.int64)
        if len(candidates) > 0:
            cx, cy = raw_centroids[candidates, 0], raw_centroids[candidates, 1]
            crop_mask = (cx >= x_min) & (cx < x_max) & (cy >= y_min) & (cy < y_max)
            crop_indices = candidates[crop_mask]

        # Labels
        if self.supervision is not None:
            nuclei_sup = self.supervision.supervision_map[stem].nuclei_supervision
            global_sup_mask = nuclei_sup.get_sup_mask(len(raw_centroids))
            nuclei_targets = nuclei_sup.get_targets(len(raw_centroids))
            crop_sup_mask = torch.as_tensor(global_sup_mask[crop_indices])
            crop_nuclei_labels = torch.as_tensor(nuclei_targets[crop_indices])
        else:
            crop_sup_mask = torch.ones(len(crop_indices), dtype=torch.bool)
            crop_nuclei_labels = None

        crop_labels = {"nuclei": crop_nuclei_labels, "graph": None}
        if self.labeled:
            crop_labels["graph"] = torch.tensor(
                [float(tile["carcinoma"])], dtype=torch.float32
            )

        # ROI Mask
        margin_x, margin_y = (
            x_extent * (self.margin / self.window_size),
            y_extent * (self.margin / self.window_size),
        )
        crop_centroids_pixels = raw_centroids[crop_indices]
        roi_mask = (
            (crop_centroids_pixels[:, 0] >= x_min + margin_x)
            & (crop_centroids_pixels[:, 0] < x_max - margin_x)
            & (crop_centroids_pixels[:, 1] >= y_min + margin_y)
            & (crop_centroids_pixels[:, 1] < y_max - margin_y)
        )

        # EFD Computation
        if len(crop_indices) == 0:
            crop_features = np.zeros((0, self.efd_order * 4 + 3), dtype=np.float32)
            crop_pos_centered = np.zeros((0, 2), dtype=np.float32)
        else:
            crop_polygons = raw_polygons[crop_indices]
            crop_features = self.get_features(crop_polygons, mpp_x, mpp_y)
            crop_centroids_microns = scaled_centroids[crop_indices]
            crop_pos_centered = (
                crop_centroids_microns - crop_centroids_microns.mean(axis=0)
            ).astype(np.float32)

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
            "sup_mask": crop_sup_mask,
            "roi_mask": torch.from_numpy(roi_mask).bool(),
            "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
            "metadata": {"slide": stem, "x": int(x_min), "y": int(y_min)},
        }