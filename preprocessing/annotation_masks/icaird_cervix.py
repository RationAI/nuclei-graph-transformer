"""Script for generating iCAIRD cervical dataset annotation masks.

Annotation files are per-slide GeoJSON `FeatureCollection`s exported from QuPath, with
each feature's severity given by `properties.classification.name` (spelling/casing
varies, e.g. "Malignant"/"malignant", "High Grade"/"high grade"). Slides in the
"normal_inflammation" category (see `index.csv`) have no lesion annotations and are
skipped entirely.

The resulting single-channel mask encodes, per pixel, the most severe annotation class
covering it (malignant > high grade > low grade), by drawing classes in ascending
severity so that a more severe polygon overwrites an overlapping less severe one:
    0 = background/unannotated
    1 = low grade  (CIN1, HPV)
    2 = high grade (CIN2, CIN3)
    3 = malignant  (squamous carcinoma, adenocarcinoma, ...)
Rare isolated "normal"/"normal/inflammation" annotations found within otherwise
abnormal slides are not drawn (left as background).
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from ratiopath.parsers import GeoJSONParser


CLASS_PATTERNS = [
    (1, r"(?i)^low\s*grade$"),
    (2, r"(?i)^high\s*grade$"),
    (3, r"(?i)^malignant$"),
]


@ray.remote(num_cpus=1, memory=(3 * 1024**3))
def process_slide(
    slide_path: Path,
    annots_dir: Path,
    level: int,
    output_dir: str,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)
        mask_size_base = slide.level_dimensions[0]
        mask_size = slide.level_dimensions[level]

    scale_x = mask_size[0] / mask_size_base[0]
    scale_y = mask_size[1] / mask_size_base[1]

    mask = Image.new("L", size=mask_size)
    canvas = ImageDraw.Draw(mask)

    parser = GeoJSONParser(annots_dir / f"{slide_path.stem}.txt")
    for value, pattern in CLASS_PATTERNS:
        for polygon in parser.get_polygons(classification_name=pattern):
            exterior_coords = [
                (x * scale_x, y * scale_y) for x, y in polygon.exterior.coords
            ]
            canvas.polygon(xy=exterior_coords, fill=value)

            for interior in polygon.interiors:  # draw holes
                interior_coords = [
                    (x * scale_x, y * scale_y) for x, y in interior.coords
                ]
                canvas.polygon(xy=interior_coords, fill=0)

    output_path = Path(output_dir, slide_path.with_suffix(".tiff").name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_big_tiff(
        image=pyvips.Image.new_from_array(mask),
        path=output_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


@with_cli_args(["+preprocessing/annotation_masks=icaird_cervix"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    index_df = pd.read_csv(config.index_csv)
    annotated = index_df[index_df["category"] != "normal_inflammation"]

    slides_dir = Path(config.slides_dir)
    annots_dir = Path(config.annots_dir)
    slide_paths = [
        slides_dir / Path(name).with_suffix(".tiff").name
        for name in annotated["slide"]
    ]

    with TemporaryDirectory() as output_dir:
        process_items(
            slide_paths,
            process_item=process_slide,
            fn_kwargs={
                "annots_dir": annots_dir,
                "level": config.level,
                "output_dir": output_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(
            local_dir=output_dir,
            artifact_path=config.mlflow_artifact_path,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
