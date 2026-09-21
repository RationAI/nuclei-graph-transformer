"""Script to generate tissue masks for whole slide images (WSIs).

Source: Carcinoma Binary Classification Methods repository.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hydra
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, tissue_mask, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=(4 * 1024**3))
def process_slide(
    slide_path: Path,
    level: int,
    output_path: Path,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    pyvips.concurrency_set(1)

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)

    level_arg = (
        {"page": level}
        if slide_path.suffix.lower() in [".tiff", ".tif"]
        else {"level": level}
    )
    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, **level_arg))
    mask = tissue_mask(slide, mpp=mpp_x)
    mask_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(
        mask,
        path=mask_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.read_csv(download_artifacts(config.metadata_uri))
    slides_path = [Path(path) for path in df["slide_path"]]

    with TemporaryDirectory() as output_dir:
        process_items(
            slides_path,
            process_item=process_slide,
            fn_kwargs={
                "level": config.level,
                "output_path": Path(output_dir),
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(local_dir=output_dir, artifact_path="tissue_masks")


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
