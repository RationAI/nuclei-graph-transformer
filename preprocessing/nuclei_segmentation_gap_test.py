"""Re-segmentation of selected tiles, to tell lost output from model misses (GPU).

`nuclei_segmentation_qc.py` flags tissue tiles without nuclei whose H&E shows nucleus-like
hematoxylin blobs ("gap" tiles). This script runs the segmentation model of
`nuclei_segmentation.py` (same `Model`, tile extent, overlap and border rule) again on
such tiles and on "control" tiles that do have nuclei in the stored output.

- Controls re-segmented to about their stored count: the test reproduces the pipeline.
- Gap tiles then with nuclei: the original output of those tiles was lost; re-segmenting
  the slides fixes them.
- Gap tiles still (almost) empty: the model misses these nuclei; re-segmenting with the
  same model does not help.

Input: a CSV (`tiles_uri`) with `slide_id`, `slide_path`, `kind` (gap / control),
`tile_x`, `tile_y` (tile origin, segmentation-level pixels), `blobs`, `stored_nuclei`.
Logs `gap_test/results.csv` (adds `rerun_nuclei`) and `gap_test/overlays/*.png` (each gap
tile with the re-segmented polygons) to MLflow.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide

from preprocessing.nuclei_segmentation import Model, drop_duplicates


def read_tile(
    slide_path: str, tile_x: int, tile_y: int, mpp: float, extent: int
) -> np.ndarray:
    """RGB tile at the slide level closest to `mpp`; origin in that level's pixels."""
    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        downsample = slide.level_downsamples[level]
        location = (round(tile_x * downsample), round(tile_y * downsample))
        return np.asarray(
            slide.read_region(location, level, (extent, extent)).convert("RGB")
        )


def overlay(
    tile: np.ndarray, polygons: list[np.ndarray], tile_x: int, tile_y: int, size: int
) -> Image.Image:
    image = Image.fromarray(tile)
    canvas = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [tuple(p) for p in (polygon - [tile_x, tile_y])]
        canvas.line([*points, points[0]], fill=(0, 230, 118), width=3)
    return image.resize((size, size), Image.LANCZOS)


@with_cli_args(["+preprocessing=nuclei_segmentation_gap_test"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiles = pd.read_csv(download_artifacts(config.tiles_uri))
    model = Model()
    seg = config.segmentation

    counts, overlays = [], {}
    for start in range(0, len(tiles), config.batch_size):
        batch = tiles.iloc[start : start + config.batch_size]
        images = np.stack(
            [
                read_tile(t.slide_path, t.tile_x, t.tile_y, seg.mpp, seg.tile_extent)
                for t in batch.itertuples(index=False)
            ]
        )
        output = model(
            {
                "slide_id": batch["slide_id"].to_numpy(),
                "tile_x": batch["tile_x"].to_numpy(),
                "tile_y": batch["tile_y"].to_numpy(),
                "tile": images,
            }
        )
        for k, t in enumerate(batch.itertuples(index=False)):
            record = {
                "slide_id": t.slide_id,
                "tile_x": t.tile_x,
                "tile_y": t.tile_y,
                "polygons": output["polygons"][k],
            }
            nuclei = list(drop_duplicates(record, seg.tile_extent, seg.overlap))
            counts.append(len(nuclei))
            if t.kind == "gap":
                name = f"{t.slide_id}_{t.tile_x}_{t.tile_y}.png"
                overlays[name] = overlay(
                    images[k],
                    [n["polygon"] for n in nuclei],
                    t.tile_x,
                    t.tile_y,
                    config.overlay_size,
                )
        print(f"{min(start + config.batch_size, len(tiles))}/{len(tiles)} tiles")

    tiles["rerun_nuclei"] = counts
    controls = tiles[tiles["kind"] == "control"]
    gaps = tiles[tiles["kind"] == "gap"]
    ratio = controls["rerun_nuclei"] / controls["stored_nuclei"]
    summary = {
        "control_ratio_median": float(ratio.median()),
        "control_ratio_min": float(ratio.min()),
        "gap_tiles": len(gaps),
        "gap_tiles_now_with_nuclei": int(
            (gaps["rerun_nuclei"] >= config.min_nuclei).sum()
        ),
        "gap_rerun_nuclei_median": float(gaps["rerun_nuclei"].median()),
    }
    print(summary)
    print(tiles.to_string(index=False))
    logger.log_hyperparams(summary)

    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        tiles.to_csv(out / "results.csv", index=False)
        (out / "overlays").mkdir()
        for name, image in overlays.items():
            image.save(out / "overlays" / name)
        logger.log_artifacts(str(out), "gap_test")


if __name__ == "__main__":
    main()
