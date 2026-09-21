"""Script for generating per-class iCAIRD cervical dataset annotation masks.

Annotation files are per-slide GeoJSON `FeatureCollection`s exported from QuPath, with
each feature's severity given by `properties.classification.name` (spelling/casing
varies, e.g. "Malignant"/"malignant", "High Grade"/"high grade"). Slides with no
annotation file (per the exploration CSV's `has_annotation` column) are skipped entirely.

This
script writes a separate binary mask (255 = annotated region, 0 = background) per class,
logged as a top-level MLflow artifact folder per class:
    low_grade/           (CIN1, HPV)
    high_grade/          (CIN2, CIN3)
    malignant/           (squamous carcinoma, adenocarcinoma, CGIN, other)
    normal_inflammation/ (rare isolated "normal"/"normal/inflammation" regions annotated
                          within otherwise abnormal slides)
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from ratiopath.parsers import GeoJSONParser

CLASS_PATTERNS = [
    ("low_grade", r"(?i)^low\s*grade$"),
    ("high_grade", r"(?i)^high\s*grade$"),
    ("malignant", r"(?i)^malignant$"),
    ("normal_inflammation", r"(?i)^normal\s*(/\s*inflammation)?$"),
]


@ray.remote(num_cpus=1, memory=(4 * 1024**3))
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

    parser = GeoJSONParser(annots_dir / f"{slide_path.stem}.txt")

    for class_name, pattern in CLASS_PATTERNS:
        mask = Image.new("L", size=mask_size)
        canvas = ImageDraw.Draw(mask)

        for polygon in parser.get_polygons(classification_name=pattern):
            exterior_coords = [
                (x * scale_x, y * scale_y) for x, y in polygon.exterior.coords
            ]
            canvas.polygon(xy=exterior_coords, fill=255)

            for interior in polygon.interiors:  # draw holes
                interior_coords = [
                    (x * scale_x, y * scale_y) for x, y in interior.coords
                ]
                canvas.polygon(xy=interior_coords, fill=0)

        output_path = Path(output_dir, class_name, slide_path.with_suffix(".tiff").name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_big_tiff(
            image=pyvips.Image.new_from_array(mask),
            path=output_path,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=mask_tile_width,
            tile_height=mask_tile_height,
        )


@with_cli_args(["+preprocessing/annotation_masks=icaird_cervix_per_class"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata_df = pd.read_csv(download_artifacts(config.metadata_uri))
    annotated = metadata_df[metadata_df["has_annotation"]]

    annots_dir = Path(config.annots_dir)
    slide_paths = annotated["slide_path"].map(Path)

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
        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
