"""Script for generating per-class iCAIRD cervical dataset annotation masks.

Annotation files are per-slide GeoJSON `FeatureCollection`s exported from QuPath, with
each feature's severity given by `properties.classification.name` (spelling/casing
varies, e.g. "Malignant"/"malignant", "High Grade"/"high grade"). Slides in the
"normal_inflammation" category (see `index.csv`) have no lesion annotations and are
skipped entirely.

This
script writes a separate binary mask (255 = annotated region, 0 = background) per class,
logged as a top-level MLflow artifact folder per class:
    low_grade/           (CIN1, HPV)
    high_grade/          (CIN2, CIN3)
    malignant/           (squamous carcinoma, adenocarcinoma, ...)
    normal_inflammation/ (rare isolated "normal"/"normal/inflammation" regions annotated
                          within otherwise abnormal slides)
"""

import logging
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

py_logger = logging.getLogger(__name__)

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
    failed_dir: str,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    try:
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

            output_path = Path(
                output_dir, class_name, slide_path.with_suffix(".tiff").name
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            write_big_tiff(
                image=pyvips.Image.new_from_array(mask),
                path=output_path,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                tile_width=mask_tile_width,
                tile_height=mask_tile_height,
            )
    except Exception as e:
        failed_path = Path(failed_dir, f"{slide_path.stem}.txt")
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(str(e))
        py_logger.warning("Failed to process slide %s", slide_path.stem, exc_info=True)


@with_cli_args(["+preprocessing/annotation_masks=icaird_cervix_per_class"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata_df = pd.read_csv(config.metadata_csv)
    annotated = metadata_df[metadata_df["category"] != "normal_inflammation"]

    slides_dir = Path(config.slides_dir)
    annots_dir = Path(config.annots_dir)
    slide_paths = [
        slides_dir / Path(name).with_suffix(".tiff").name for name in annotated["slide"]
    ]

    with TemporaryDirectory() as output_dir, TemporaryDirectory() as failed_dir:
        process_items(
            slide_paths,
            process_item=process_slide,
            fn_kwargs={
                "annots_dir": annots_dir,
                "level": config.level,
                "output_dir": output_dir,
                "failed_dir": failed_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
            },
            max_concurrent=config.max_concurrent,
        )
        failed_slides = {p.stem: p.read_text() for p in Path(failed_dir).glob("*.txt")}

        if failed_slides:
            py_logger.warning(
                "%d slide(s) failed to process: %s",
                len(failed_slides),
                sorted(failed_slides),
            )
            pd.DataFrame(
                {
                    "slide_stem": list(failed_slides),
                    "error": list(failed_slides.values()),
                }
            ).to_csv(Path(output_dir, "failed_slides.csv"), index=False)

        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
