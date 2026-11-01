import hashlib
import shutil
from collections.abc import Iterator
from math import ceil, floor
from pathlib import Path
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlparse

import cv2
import hydra
import numpy as np
import pandas as pd
import ray
import torch
from cellpose.models import CellposeModel
from cellpose.utils import outlines_list
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
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
    wb_scale: NotRequired[list[float]]


class NucleusRecord(TypedDict):
    id: str
    slide_id: str
    polygon: NDArray[np.float32]
    centroid: NDArray[np.float32]


class TilePolygonRecord(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int
    mpp_x: float
    mpp_y: float
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    polygons: list[NDArray[np.float32]]


# Ids of the slides whose nuclei are fully written, one per line, in the output folder.
COMPLETED_FILE = "_completed_slides.txt"

TILE_METADATA = (
    "slide_id",
    "tile_x",
    "tile_y",
    "mpp_x",
    "mpp_y",
    "extent_x",
    "extent_y",
    "tile_extent_x",
    "tile_extent_y",
)


def white_balanced_tiles(batch: dict[str, Any]) -> NDArray[np.uint8]:
    """The batch's tiles, scaled by their `wb_scale` factors if the batch has them."""
    tiles = batch["tile"]
    if "wb_scale" in batch:
        scale = np.stack(batch["wb_scale"]).astype(np.float32)[:, None, None, :]
        tiles = np.clip(tiles * scale, 0, 255).astype(np.uint8)
    return tiles


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
                - "wb_scale" (np.ndarray, optional): Per-channel white balance factors.
        """
        tiles = white_balanced_tiles(batch)
        inputs = self.processor(tiles.copy(), device=self.device, return_tensors="pt")
        outputs = self.model(**inputs)
        results = self.processor.post_process(outputs)

        return {
            **{key: batch[key] for key in TILE_METADATA},
            "polygons": [result["polygons"].cpu().numpy() for result in results],
        }


def resample_contour(
    contour: NDArray[np.float32], num_vertices: int
) -> NDArray[np.float32]:
    """`num_vertices` points spaced evenly along a closed contour, in LSP-DETR's winding.

    LSP-DETR polygons have a negative shoelace area in (x, y) pixel coordinates; the
    contour is reversed if its area is positive.
    """
    closed = np.concatenate([contour, contour[:1]]).astype(np.float64)
    arc = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    )
    t = np.linspace(0, arc[-1], num_vertices, endpoint=False)
    polygon = np.stack([np.interp(t, arc, closed[:, i]) for i in range(2)], axis=1)
    x, y = polygon[:, 0], polygon[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) > 0:
        polygon = polygon[::-1]
    return polygon.astype(np.float32)


class Cellpose:
    """Cellpose 4 (generalist `cpsam` model, default settings) instance segmentation.

    Each instance's outline is resampled to `num_vertices` points, so the polygons
    have the fixed size the rest of the pipeline expects from LSP-DETR.
    """

    def __init__(self, num_vertices: int) -> None:
        self.model = CellposeModel(gpu=True)
        self.num_vertices = num_vertices

    def __call__(self, batch: dict[str, Any]) -> TilePolygonRecord:
        """Segments nuclei in a batch of tiles (see `Model.__call__`)."""
        masks = self.model.eval(list(white_balanced_tiles(batch)))[0]

        polygons = [
            np.array(
                [
                    resample_contour(contour, self.num_vertices)
                    for contour in outlines_list(tile_masks, multiprocessing=False)
                    if len(contour) >= 3
                ],
                dtype=np.float32,
            ).reshape(-1, self.num_vertices, 2)
            for tile_masks in masks
        ]
        return {**{key: batch[key] for key in TILE_METADATA}, "polygons": polygons}


def glass_rgb(
    slide_path: str, glass_min_value: int, max_side: int
) -> NDArray[np.float64]:
    """Glass colour of a slide, read from its largest level no wider than `max_side`.

    The median of the bright pixels (max channel above `glass_min_value`) at their most
    common saturation; pure-white padding outside the scanned area is ignored. The same
    estimate as `correct_glass_tint` in `preprocessing/tissue_masks/icaird_cervix.py`,
    so tinted glass is found too.
    """
    with OpenSlide(slide_path) as slide:
        level = next(
            (
                level
                for level, (w, h) in enumerate(slide.level_dimensions)
                if max(w, h) <= max_side
            ),
            slide.level_count - 1,
        )
        region = slide.read_region((0, 0), level, slide.level_dimensions[level])
    pixels = np.asarray(region.convert("RGB")).reshape(-1, 3)
    pixels = pixels[(pixels.max(axis=1) > glass_min_value) & (pixels.min(axis=1) < 250)]
    if len(pixels) == 0:
        raise ValueError(f"{slide_path}: no glass pixels to white balance against")

    saturation = cv2.cvtColor(pixels[None], cv2.COLOR_RGB2HSV)[0, :, 1]
    glass_saturation = np.bincount(saturation).argmax()
    return np.median(pixels[saturation == glass_saturation], axis=0)


def tiling(
    slide_record: SlideRecord, white_balance: dict[str, Any] | None = None
) -> Iterator[TileRecord]:
    """Yields metadata for unprocessed tiles of a slide.

    With `white_balance`, every tile carries the per-channel factors that map the
    slide's glass colour to the grey `glass_target`, applied to the tile before
    segmentation.

    Note: The tiling step is not separated as per usual, as tiles are only used within
          this pipeline and persisting them would not have an additional benefit.
    """
    extra: dict[str, Any] = {}
    if white_balance is not None:
        glass = glass_rgb(
            slide_record["path"],
            white_balance["glass_min_value"],
            white_balance["max_side"],
        )
        extra["wb_scale"] = (white_balance["glass_target"] / glass).tolist()

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
            **extra,
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
        nucleus_key = f"{tile_record['slide_id']}_{tile_record['tile_x']}_{tile_record['tile_y']}_{i}"

        yield {
            "id": hashlib.sha256(nucleus_key.encode()).hexdigest(),
            "slide_id": tile_record["slide_id"],
            "polygon": polygon,
            "centroid": centroid,
        }


class TissueTileFilter:
    """Returns True if a tile should be kept based on a binary tissue mask.

    A slide's tiles arrive from `tiling()` (and stay grouped through repartition)
    in one contiguous run, so caching only the most recently opened mask avoids
    reopening the same WSI file for every one of a slide's tiles while still
    releasing the previous handle when a new slide starts.
    """

    def __init__(self, tissue_masks_dir: Path) -> None:
        self.tissue_masks_dir = tissue_masks_dir
        self._slide_id: str | None = None
        self._mask_slide: OpenSlide | None = None

    def _mask_slide_for(self, slide_id: str) -> OpenSlide:
        if slide_id != self._slide_id:
            if self._mask_slide is not None:
                self._mask_slide.close()
            self._mask_slide = OpenSlide(self.tissue_masks_dir / f"{slide_id}.tiff")
            self._slide_id = slide_id
        return self._mask_slide

    def _mask_tile(
        self, tile_record: TileRecord | TilePolygonRecord
    ) -> tuple[NDArray[np.uint8], int, int, float, float]:
        """The mask under a tile, its top-left corner and the tile-to-mask scale."""
        mask_slide = self._mask_slide_for(Path(tile_record["slide_id"]).stem)

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
        return np.array(mask_tile)[..., 0], x, y, scale_x, scale_y

    def __call__(self, tile_record: TileRecord) -> bool:
        """If coverage > 0.0, keep the tile; otherwise drop it."""
        return bool(np.count_nonzero(self._mask_tile(tile_record)[0]))


class TissueNucleiFilter(TissueTileFilter):
    """Drops the nuclei of a tile whose centroid lies outside the binary tissue mask.

    Cellpose outlines sensor noise on empty glass in tiles that touch tissue; LSP-DETR
    rarely does. Tiles arrive from the model mostly grouped by slide, so the mask cache
    of `TissueTileFilter` still saves most reopenings.
    """

    def __call__(self, tile_record: TilePolygonRecord) -> TilePolygonRecord:  # type: ignore[override]
        polygons = np.asarray(tile_record["polygons"])
        if len(polygons) == 0:
            return tile_record

        mask, x, y, scale_x, scale_y = self._mask_tile(tile_record)
        centroids = polygons.mean(axis=1)
        mask_x = np.floor((tile_record["tile_x"] + centroids[:, 0]) * scale_x) - x
        mask_y = np.floor((tile_record["tile_y"] + centroids[:, 1]) * scale_y) - y
        mask_x = np.clip(mask_x.astype(int), 0, mask.shape[1] - 1)
        mask_y = np.clip(mask_y.astype(int), 0, mask.shape[0] - 1)
        return {**tile_record, "polygons": polygons[mask[mask_y, mask_x] > 0]}


def run_segmentation(
    slide_paths: list[str], output_dir: Path, tissue_masks_dir: Path, config: DictConfig
) -> None:
    """Nuclei segmentation pipeline for whole-slide images specified by `slide_paths`.

    The pipeline proceeds in these steps:
        1) Slide Metadata Preparation: get slide metadata needed for processing.
        2) Tiling: split slides into tiles and filter out non-tissue tiles using binary masks.
        3) Nucleus Detection: segment nuclei in each tile with `config.model` (white
           balanced first, if `config.white_balance` is set) and extract polygons.
        4) Tissue Filter: with `config.tissue_filter_nuclei`, discard nuclei whose
           centroid lies outside the tissue mask.
        5) Duplicate Removal: discard nuclei near tile edges to prevent duplicates.
        6) Saving: store results as partitioned Parquet files organized by `slide_id`.
    """
    slides = read_slides(
        slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.tile_extent - config.overlap,
    )

    white_balance = (
        OmegaConf.to_container(config.white_balance)
        if config.white_balance is not None
        else None
    )
    tiles = slides.flat_map(
        tiling,
        fn_kwargs={"white_balance": white_balance},
        num_cpus=0.1,
        memory=1 * 1024**3,  # the white balance reads a low-resolution level
    ).repartition(target_num_rows_per_block=128)
    tissue_tiles = (
        tiles.filter(
            TissueTileFilter,
            fn_constructor_kwargs={"tissue_masks_dir": tissue_masks_dir},
            num_cpus=0.5,
            memory=1 * 1024**3,
            concurrency=(2, 16),
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

    models = {
        "lsp_detr": (Model, {}),
        "cellpose": (Cellpose, {"num_vertices": config.num_vertices}),
    }
    model, model_kwargs = models[config.model]
    nuclei = tissue_tiles.map_batches(
        model,
        fn_constructor_kwargs=model_kwargs,
        num_gpus=1 / config.workers_per_gpu,
        num_cpus=0,
        batch_size=config.batch_size,
        memory=3 * 1024**3 if config.model == "lsp_detr" else 6 * 1024**3,
        concurrency=config.gpus * config.workers_per_gpu,
        zero_copy_batch=True,
    )
    if config.tissue_filter_nuclei:
        nuclei = nuclei.map(
            TissueNucleiFilter,
            fn_constructor_kwargs={"tissue_masks_dir": tissue_masks_dir},
            num_cpus=0.5,
            memory=1 * 1024**3,
            concurrency=(2, 16),
        )
    nuclei = nuclei.flat_map(
        drop_duplicates,
        fn_kwargs={"tile_extent": config.tile_extent, "overlap": config.overlap},
        num_cpus=0.1,
        memory=1.5 * 1024**3,
    )
    nuclei.write_parquet(str(output_dir), partition_cols=["slide_id"])


def segment_resumably(
    slide_paths: list[str], output_dir: Path, tissue_masks_dir: Path, config: DictConfig
) -> None:
    """Runs `run_segmentation` in chunks of `config.slides_per_chunk` slides.

    A slide's nuclei are written in several files, so a slide whose chunk was
    interrupted may be only partly written. The ids of the slides of each finished
    chunk are appended to `COMPLETED_FILE`; a rerun skips them and deletes what the
    other slides left behind before segmenting them again.
    """
    completed_path = output_dir / COMPLETED_FILE
    if completed_path.exists():
        completed = set(completed_path.read_text().split())
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"{output_dir} holds nuclei but no {COMPLETED_FILE}, so it was not written "
            "by a resumable run and its finished slides are unknown. Set another "
            "`batch_name`."
        )
    else:
        completed = set()

    todo = [path for path in slide_paths if Path(path).stem not in completed]
    for start in range(0, len(todo), config.slides_per_chunk):
        chunk = todo[start : start + config.slides_per_chunk]
        for path in chunk:
            shutil.rmtree(
                output_dir / f"slide_id={Path(path).stem}", ignore_errors=True
            )

        run_segmentation(chunk, output_dir, tissue_masks_dir, config)
        output_dir.mkdir(parents=True, exist_ok=True)  # no nuclei, no folder
        with completed_path.open("a") as f:
            f.writelines(f"{Path(path).stem}\n" for path in chunk)


def output_batch_name(metadata_uri: str, batch_name: str | None) -> str:
    """Name of the output subfolder of a metadata CSV: `batch_name`, or its parent folder."""
    if batch_name:
        return batch_name

    parent = Path(urlparse(metadata_uri).path).parent.name
    if parent == "artifacts":
        raise ValueError(
            f"{metadata_uri} is at the root of its run's artifacts, so its folder name "
            "does not identify the batch. Set `batch_name`."
        )
    return parent


@with_cli_args(["+preprocessing=nuclei_segmentation"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    batch_names = [
        output_batch_name(metadata_uri, config.batch_name)
        for metadata_uri in config.metadata_uris
    ]
    tissue_masks_dir = Path(download_artifacts(config.tissue_masks_uri))

    for metadata_uri, batch_name in zip(config.metadata_uris, batch_names, strict=True):
        slides = pd.read_csv(download_artifacts(metadata_uri))
        segment_resumably(
            slide_paths=slides["slide_path"].tolist(),
            output_dir=Path(config.output_path, batch_name),
            tissue_masks_dir=tissue_masks_dir,
            config=config,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
