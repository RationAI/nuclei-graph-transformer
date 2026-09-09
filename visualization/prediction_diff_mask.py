"""Script for visualizing the disagreement between two models' nuclei predictions.

Both models must score the same underlying nuclei polygon set (`slide_nuclei_path`).
Each model's `nuclei_prediction` is thresholded independently (`pred_thr_a`, `pred_thr_b`).
A nucleus is filled iff the two resulting binary predictions disagree
(positive for exactly one of the two models); nuclei both models agree on
(both positive or both negative) are left as background.

The result is a single-channel binary mask, matching the convention used by
`polygons2raster.py` (0 = background, 255 = filled).
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


def mark_disagreement(
    nuclei: pd.DataFrame,
    predictions_dir_a: Path,
    predictions_dir_b: Path,
    slide_path: Path,
    pred_thr_a: float,
    pred_thr_b: float,
) -> pd.DataFrame | None:
    predictions_path_a = predictions_dir_a / f"{slide_path.stem}.parquet"
    predictions_path_b = predictions_dir_b / f"{slide_path.stem}.parquet"
    if not predictions_path_a.exists() or not predictions_path_b.exists():
        return None 

    preds_a = pd.read_parquet(predictions_path_a)[["id", "nuclei_prediction"]].rename(
        columns={"nuclei_prediction": "pred_a"}
    )
    preds_b = pd.read_parquet(predictions_path_b)[["id", "nuclei_prediction"]].rename(
        columns={"nuclei_prediction": "pred_b"}
    )
    nuclei = nuclei.merge(preds_a, on="id", how="inner").merge(
        preds_b, on="id", how="inner"
    )

    pos_a = nuclei["pred_a"] >= pred_thr_a
    pos_b = nuclei["pred_b"] >= pred_thr_b
    nuclei["disagree"] = pos_a != pos_b
    return nuclei


@ray.remote(memory=90 * 1024**3)
def process_slide(
    item: dict[str, Any],
    predictions_dir_a: Path,
    predictions_dir_b: Path,
    pred_thr_a: float,
    pred_thr_b: float,
    mask_tile_width: int,
    mask_tile_height: int,
    output_dir: Path,
) -> None:
    slide_path = Path(item["slide_path"])
    nuclei = pd.read_parquet(item["slide_nuclei_path"], columns=["id", "polygon"])
    nuclei = mark_disagreement(
        nuclei, predictions_dir_a, predictions_dir_b, slide_path, pred_thr_a, pred_thr_b
    )
    if nuclei is None:
        return

    with OpenSlide(item["slide_path"]) as slide:
        level = 0  # rasterize at the highest resolution level
        mask_mpp_x, mask_mpp_y = slide.slide_resolution(level)
        mask_size = slide.level_dimensions[level]

    mask = Image.new("L", size=mask_size)
    canvas = ImageDraw.Draw(mask)

    for row in nuclei.itertuples(index=False):
        if row.disagree:
            canvas.polygon(xy=row.polygon, outline=255, fill=255)

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


@hydra.main(config_path="../configs", config_name="prediction_diff", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata = uris2df(config.metadata_uris)
    predictions_dir_a = Path(download_artifacts(config.predictions_uri_a))
    predictions_dir_b = Path(download_artifacts(config.predictions_uri_b))

    with TemporaryDirectory() as output_dir:
        process_items(
            items=metadata[["slide_path", "slide_nuclei_path"]].to_dict("records"),
            process_item=process_slide,
            fn_kwargs={
                "predictions_dir_a": predictions_dir_a,
                "predictions_dir_b": predictions_dir_b,
                "pred_thr_a": config.pred_thr_a,
                "pred_thr_b": config.pred_thr_b,
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
