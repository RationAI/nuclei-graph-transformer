"""Script for rasterizing a single model's per-nucleus predictions into a mask.

Reads per-nucleus prediction scores already saved by `NucleiPredictionCallback`
during `mode=predict` (see `nuclei_graph/callbacks/predictions.py`, artifact
path "predictions") and the corresponding nuclei polygons, then rasterizes a
continuous-valued (0-255) heatmap mask at full resolution (level 0).

This lets masks be regenerated (e.g. to fix `NucleiPredictionMasksCallback`'s
level=2 masks, see `configs/callbacks/masks/nuclei.yaml`) without re-running
model inference, since the underlying predictions are already saved.

The result is a single-channel grayscale mask, matching the convention used by
`polygons2raster.py` and `NucleiPredictionMasksCallback` (0 = most negative
prediction, 255 = most positive).
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


def merge_predictions(
    nuclei: pd.DataFrame, predictions_dir: Path, slide_path: Path
) -> pd.DataFrame | None:
    predictions_path = predictions_dir / f"{slide_path.stem}.parquet"
    if not predictions_path.exists():
        return None

    predictions = pd.read_parquet(predictions_path, columns=["id", "nuclei_prediction"])
    return nuclei.merge(predictions, on="id", how="inner")


@ray.remote(memory=90 * 1024**3)
def process_slide(
    item: dict[str, Any],
    predictions_dir: Path,
    mask_tile_width: int,
    mask_tile_height: int,
    output_dir: Path,
) -> None:
    slide_path = Path(item["slide_path"])
    nuclei = pd.read_parquet(item["slide_nuclei_path"], columns=["id", "polygon"])
    nuclei = merge_predictions(nuclei, predictions_dir, slide_path)
    if nuclei is None:
        return

    with OpenSlide(item["slide_path"]) as slide:
        level = 0  # rasterize at the highest resolution level
        mask_mpp_x, mask_mpp_y = slide.slide_resolution(level)
        mask_size = slide.level_dimensions[level]

    mask = Image.new("L", size=mask_size)
    canvas = ImageDraw.Draw(mask)

    for row in nuclei.itertuples(index=False):
        pixel_val = round(row.nuclei_prediction * 255)
        canvas.polygon(xy=row.polygon, outline=pixel_val, fill=pixel_val)

    write_big_tiff(
        image=pyvips.Image.new_from_array(np.array(mask)),
        path=output_dir / slide_path.with_suffix(".tiff").name,
        mpp_x=mask_mpp_x,
        mpp_y=mask_mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


def uris2df(uris: list[str]) -> pd.DataFrame:
    """Loads and merges multiple metadata Parquet files into a single DataFrame."""
    batches = [pd.read_parquet(download_artifacts(uri)) for uri in uris]
    return pd.concat(batches, ignore_index=True).drop_duplicates(subset=["slide_path"])


@hydra.main(config_path="../configs", config_name="prediction_mask", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata = uris2df(config.metadata_uris)
    predictions_dir = Path(download_artifacts(config.predictions_uri))

    with TemporaryDirectory() as output_dir:
        process_items(
            items=metadata[["slide_path", "slide_nuclei_path"]].to_dict("records"),
            process_item=process_slide,
            fn_kwargs={
                "predictions_dir": predictions_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                "output_dir": Path(output_dir),
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
