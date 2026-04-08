"""Script for generating prostate cancer annotation masks for annotated regions.

Annotation groups:
    - "Carcinoma": regions containing carcinoma (cancerous tissue)
    - "Exclude": regions that should be removed from the carcinoma mask (holes, artifacts)
    - "Another pathology": various non-cancerous regions (inflammation, pre-cancer, etc.); should be classified as negative
The resulting mask only marks regions that are "Carcinoma" and are neither "Exclude" nor "Another pathology".

Assumes the following structure of input data:
1. Exploratory Metadataset (`exploration/save_metadataset.py`):
<DATASET_NAME>/
    slides_metadata.csv (columns "slide_path" (str), "is_carcinoma" (bool), and "has_annotation" (bool))

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.tiff (binary single-channel mask of annotated regions for positive slides with annotations)
`missing_annotations.csv` (column "slide_path" (str))
"""

import os
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
from ratiopath.parsers import ASAPParser
from shapely import MultiPolygon, Polygon, make_valid
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def filter_carcinoma(slide_path: Path) -> list[BaseGeometry]:
    parser = ASAPParser(slide_path.with_suffix(".xml"))
    carcinoma = MultiPolygon(list(parser.get_polygons(part_of_group="Carcinoma")))
    exclude = MultiPolygon(list(parser.get_polygons(part_of_group="Exclude")))
    another = MultiPolygon(list(parser.get_polygons(part_of_group="Another pathology")))

    # fix self-intersections, etc.
    carcinoma, exclude, another = [
        make_valid(geom) if not geom.is_valid else geom
        for geom in [carcinoma, exclude, another]
    ]
    exclusions = unary_union([exclude, another])
    exclusions = make_valid(exclusions) if not exclusions.is_valid else exclusions
    result = carcinoma.difference(exclusions)
    result = make_valid(result) if not result.is_valid else result
    return [result] if isinstance(result, Polygon) else result.geoms


@ray.remote(num_cpus=1, memory=(3 * 1024**3))
def process_slide(
    slide_path: Path,
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
    filtered_carcinoma = filter_carcinoma(slide_path)

    for polygon in filtered_carcinoma:
        exterior_coords = [
            (x * scale_x, y * scale_y) for x, y in polygon.exterior.coords
        ]
        canvas.polygon(xy=exterior_coords, fill=255)

        for interior in polygon.interiors:  # draw holes
            interior_coords = [(x * scale_x, y * scale_y) for x, y in interior.coords]
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


@with_cli_args(["+preprocessing=annotation_masks"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    train_df = pd.read_csv(download_artifacts(config.train_metadata_uri))
    test_df = pd.read_csv(download_artifacts(config.test_metadata_uri))
    slides = pd.concat([train_df, test_df])
    slides_annots = slides[slides["has_annotation"] & (slides["is_carcinoma"])]
    missing_annots = slides[~slides["has_annotation"] & (slides["is_carcinoma"])]

    with TemporaryDirectory(dir=os.getcwd()) as output_dir:
        process_items(
            slides_annots["slide_path"].map(Path),
            process_item=process_slide,
            fn_kwargs={
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
        csv_path = Path(output_dir, "missing_annotations.csv")
        missing_annots.to_csv(csv_path, columns=["slide_path"], index=False)
        logger.log_artifact(local_path=str(csv_path))


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()