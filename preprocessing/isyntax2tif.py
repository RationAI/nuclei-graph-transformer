"""Batch-convert iSyntax slides from the iCAIRD cervical dataset to pyramidal OpenSlide-compatible TIFF."""

from pathlib import Path

import hydra
from ratiopath.openslide import OpenSlide
import pandas as pd
import pyvips
import ray
from isyntax import ISyntax
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

MPP_TOLERANCE = 1e-3


@ray.remote(num_cpus=1, memory=(32 * 1024**3))
def convert_slide(
    slide_path: Path,
    output_dir: Path,
    tile_width: int,
    tile_height: int,
) -> None:
    output_path = output_dir / f"{slide_path.stem}.tiff"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ISyntax.open(str(slide_path)) as isx:
        width, height = isx.level_dimensions[0]
        mpp_x, mpp_y = isx.mpp_x, isx.mpp_y

        rgba = isx.read_region(0, 0, width, height, level=0)
        rgb = rgba[:, :, :3]

        write_big_tiff(
            image=pyvips.Image.new_from_array(rgb),
            path=str(output_path),
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )

    with OpenSlide(output_path) as slide:
        out_mpp_x, out_mpp_y =  slide.slide_resolution(level=0)

    assert abs(out_mpp_x - mpp_x) < MPP_TOLERANCE, (
        f"{slide_path.stem}: mpp-x mismatch, source={mpp_x} output={out_mpp_x}"
    )
    assert abs(out_mpp_y - mpp_y) < MPP_TOLERANCE, (
        f"{slide_path.stem}: mpp-y mismatch, source={mpp_y} output={out_mpp_y}"
    )


@with_cli_args(["+preprocessing=isyntax2tif"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    slides = list(Path(config.slides_dir).glob("*.isyntax"))

    process_items(
        items=slides,
        process_item=convert_slide,
        fn_kwargs={
            "output_dir": Path(config.output_dir),
            "tile_width": config.mask_tile_width,
            "tile_height": config.mask_tile_height,
        },
        max_concurrent=config.max_concurrent,
    )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()