"""Nuclei Segmentation Pipeline adapted from the Nuclei Foundational Model repository.

This pipeline processes whole-slide images to segment nuclei. The nuclei are saved in a
Parquet dataset under the configured `output_path`, partitioned by `slide_id`.

Output structure:
<OUTPUT_PATH>/
    <DATASET_NAME>/
        slide_id=<SLIDE_NAME>/
            *.parquet (segmented nuclei)

Each row in the saved Parquet files corresponds to a single nucleus and contains keys:
    - `polygon` (np.ndarray[float]): Nucleus segmentation polygon coordinates (64 points, flattened).
    - `centroid` (np.ndarray[float]): Nucleus centroid coordinates.
    - `id` (str): Unique hash identifier for the nucleus.

The `id` is intended to be used for determining a fixed ordering of nuclei within a slide
(reading partitioned Parquet files does not always guarantee a fixed order).
"""

import hashlib
from collections.abc import Iterator
from math import ceil, floor
from pathlib import Path
from typing import Any, TypedDict

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ray.data.expressions import col
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class SlideRecord(TypedDict):
    path: str
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    stride_x: int
    stride_y: int
    mpp_x: float
    mpp_y: float
    level: int
    downsample: float
    slide_id: str
    scale_factor: float


class TileRecord(TypedDict):
    tile_x: int
    tile_y: int
    path: str
    slide_id: str
    mpp_x: float
    mpp_y: float
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    level: int


class NucleusRecord(TypedDict):
    id: str
    slide_id: str
    polygon: NDArray[np.float32]
    centroid: NDArray[np.float32]


class TilePolygonRecord(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int
    polygons: list[NDArray[np.float32]]


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self.model = AutoModelForObjectDetection.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
        ).to(self.device)
        self.model = self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
        )

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def __call__(self, batch: dict[str, Any]) -> TilePolygonRecord:
        """Segments nuclei in a tile and extracts polygons.

        Args:
            batch (dict): Tile metadata and image.
                - "path" (str): Slide path.
                - "tile_x" (int): X-coordinate of the tile.
                - "tile_y" (int): Y-coordinate of the tile.
                - "tile" (PIL.Image or np.ndarray): Tile image.
        """
        inputs = self.processor(
            batch["tile"].copy(), device=self.device, return_tensors="pt"
        )
        outputs = self.model(**inputs)
        results = self.processor.post_process(outputs)

        return {
            "slide_id": batch["slide_id"],
            "tile_x": batch["tile_x"],
            "tile_y": batch["tile_y"],
            "polygons": [result["polygons"].cpu().numpy() for result in results],
        }


def tiling(slide_record: SlideRecord) -> Iterator[TileRecord]:
    """Yields metadata for unprocessed tiles of a slide.

    Note: The tiling step is not separated as per usual, as tiles are only used within
          this pipeline and persisting them would not have an additional benefit.
    """
    for x, y in grid_tiles(
        slide_extent=(slide_record["extent_x"], slide_record["extent_y"]),
        tile_extent=(slide_record["tile_extent_x"], slide_record["tile_extent_y"]),
        stride=(slide_record["stride_x"], slide_record["stride_y"]),
        last="keep",
    ):
        yield {
            "tile_x": x,
            "tile_y": y,
            "path": slide_record["path"],
            "slide_id": Path(slide_record["path"]).stem,
            "mpp_x": slide_record["mpp_x"],
            "mpp_y": slide_record["mpp_y"],
            "extent_x": slide_record["extent_x"],
            "extent_y": slide_record["extent_y"],
            "tile_extent_x": slide_record["tile_extent_x"],
            "tile_extent_y": slide_record["tile_extent_y"],
            "level": slide_record["level"],
        }


def drop_duplicates(
    tile_record: TilePolygonRecord, tile_extent: int, overlap: int
) -> Iterator[NucleusRecord]:
    """Filters out nuclei near tile borders to avoid duplicates.

    For each nucleus, its centroid is computed and checked to ensure it lies within
    the non-overlapping region. Remaining polygons and centroids are adjusted to
    absolute slide coordinates.
    """
    if len(tile_record["polygons"]) == 0:
        return
    polygons_arr = np.stack(tile_record["polygons"], axis=0)
    centroids = polygons_arr.mean(axis=1)
    keep = np.all(centroids >= overlap / 2, axis=-1) & np.all(
        centroids < tile_extent - overlap / 2, axis=-1
    )

    offset = np.array((tile_record["tile_x"], tile_record["tile_y"]), dtype=np.float32)
    polygons = polygons_arr[keep] + offset
    centroids = centroids[keep] + offset

    for i, (polygon, centroid) in enumerate(zip(polygons, centroids, strict=True)):
        nucleus_key = f"{tile_record['slide_id']}{tile_record['tile_x']}{tile_record['tile_y']}{i}"

        yield {
            "id": hashlib.sha256(nucleus_key.encode()).hexdigest(),
            "slide_id": tile_record["slide_id"],
            "polygon": polygon,
            "centroid": centroid,
        }


def filter_tissue_tiles(tile_record: TileRecord, tissue_masks_dir: Path) -> bool:
    """Returns True if a tile should be kept based on a binary tissue mask.

    If coverage > 0.0, keep the tile; otherwise drop it.
    """
    tissue_mask_path = tissue_masks_dir / f"{Path(tile_record['slide_id']).stem}.tiff"

    with OpenSlide(tissue_mask_path) as mask_slide:
        level = mask_slide.closest_level(
            (tile_record["mpp_x"] + tile_record["mpp_y"]) / 2
        )
        extent_x, extent_y = mask_slide.level_dimensions[level]
        scale_x = extent_x / tile_record["extent_x"]
        scale_y = extent_y / tile_record["extent_y"]

        x = floor(tile_record["tile_x"] * scale_x)
        y = floor(tile_record["tile_y"] * scale_y)
        width = ceil(tile_record["tile_extent_x"] * scale_x)
        height = ceil(tile_record["tile_extent_y"] * scale_y)

        mask_tile = mask_slide.read_region_relative((x, y), level, (width, height))
        # read_region returns RGBA; for binary masks all channels are identical — take the first one
        mask_tile_array = np.array(mask_tile)[..., 0]
        tissue_ratio = np.count_nonzero(mask_tile_array) / mask_tile_array.size

        if tissue_ratio > 0.0:
            return True

    return False


def run_segmentation(
    slide_paths: list[str], output_dir: Path, tissue_masks_dir: Path, config: DictConfig
) -> None:
    """Nuclei segmentation pipeline for whole-slide images specified by `slide_paths`.

    The pipeline proceeds in these steps:
        1) Slide Metadata Preparation: get slide metadata needed for processing.
        2) Tiling: split slides into tiles and filter out non-tissue tiles using binary masks.
        3) Nucleus Detection: segment nuclei in each tile and extract polygons.
        4) Duplicate Removal: discard nuclei near tile edges to prevent duplicates.
        5) Saving: store results as partitioned Parquet files organized by `slide_id`.
    """
    slides = read_slides(
        slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.tile_extent - config.overlap,
    )

    tiles = slides.flat_map(tiling, num_cpus=0.1, memory=128 * 1024**2).repartition(
        target_num_rows_per_block=128
    )
    tissue_tiles = (
        tiles.filter(
            filter_tissue_tiles,
            fn_kwargs={"tissue_masks_dir": tissue_masks_dir},
            memory=3 * 1024**3,
        )
        .repartition(target_num_rows_per_block=config.batch_size * 16)
        .with_column(
            "tile",
            read_slide_tiles(
                col("path"),
                col("tile_x"),
                col("tile_y"),
                col("tile_extent_x"),
                col("tile_extent_y"),
                col("level"),
            ),
            num_cpus=1,
            memory=5 * 1024**3,
        )
    )

    nuclei = tissue_tiles.map_batches(
        Model,
        num_gpus=1,
        num_cpus=0,
        batch_size=config.batch_size,
        memory=3 * 1024**3,
        concurrency=1,
        zero_copy_batch=True,
    )
    nuclei = nuclei.flat_map(
        drop_duplicates,
        fn_kwargs={"tile_extent": config.tile_extent, "overlap": config.overlap},
        num_cpus=0.1,
        memory=1.5 * 1024**3,
    )
    nuclei.write_parquet(str(output_dir), partition_cols=["slide_id"])


@with_cli_args(["+preprocessing=nuclei_segmentation"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    tissue_masks_dir = Path(download_artifacts(config.tissue_masks_uri))

    train_slides = pd.read_csv(download_artifacts(config.train_metadata_uri))
    run_segmentation(
        slide_paths=train_slides["slide_path"].tolist(),
        output_dir=Path(config.output_path, "tile_level_annotations"),
        tissue_masks_dir=tissue_masks_dir,
        config=config,
    )

    test_slides = pd.read_csv(download_artifacts(config.test_metadata_uri))
    run_segmentation(
        slide_paths=test_slides["slide_path"].tolist(),
        output_dir=Path(config.output_path, "tile_level_annotations_test"),
        tissue_masks_dir=tissue_masks_dir,
        config=config,
    )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
