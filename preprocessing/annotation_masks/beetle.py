"""Script for generating per-class annotation masks for the BEETLE dataset.

Annotation groups (BEETLE):
    - "invasive epithelium"
    - "non-invasive epithelium"
    - "necrosis"
    - "other"
A separate binary mask (255 = annotated region, 0 = background) is written for each
class, for each slide.
"""

import io
import re
from pathlib import Path
from typing import TypedDict

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
from shapely import MultiPolygon, make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


# A handful of BEETLE XMLs contain coordinates exported with a comma as the decimal separator
DECIMAL_COMMA_COORD = re.compile(r'([XY])="(\d+),(\d+)"')


def slugify_class(group: str) -> str:
    return group.replace(" ", "_")


class SlideRecord(TypedDict):
    slide_id: str
    slide_path: str
    mask_path: str


def load_asap_parser(xml_path: Path) -> ASAPParser:
    xml_text = xml_path.read_text(encoding="utf-8")
    xml_text = DECIMAL_COMMA_COORD.sub(r'\1="\2.\3"', xml_text)
    return ASAPParser(io.StringIO(xml_text))


def get_class_geometry(parser: ASAPParser, group: str) -> BaseGeometry:
    geometry = MultiPolygon(list(parser.get_polygons(part_of_group=group)))
    return make_valid(geometry) if not geometry.is_valid else geometry


def iter_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [p for geom in geometry.geoms for p in iter_polygons(geom)]
    return []


def get_annotation_size(reference_mask_path: Path) -> tuple[int, int]:
    """Returns the pixel size of the frame the XML coordinates are expressed in.

    The annotations were drawn on ~0.5 mpp images, so for TCGA slides whose WSI is
    ~0.25 mpp (or finer) the coordinates are in a 2x (or 4x) downsampled frame
    relative to the WSI's level 0. The reference masks share the annotation frame.
    """
    with OpenSlide(reference_mask_path) as reference:
        return reference.level_dimensions[0]


def output_path_for(output_dir: str, group: str, slide_path: Path) -> Path:
    return Path(output_dir, slugify_class(group), slide_path.with_suffix(".tiff").name)


@ray.remote(num_cpus=1, memory=(6 * 1024**3))
def process_slide(
    slide_record: SlideRecord,
    xmls_dir: Path,
    classes: list[str],
    level: int,
    output_dir: str,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    slide_path = Path(slide_record["slide_path"])
    xml_path = xmls_dir / f"{slide_record['slide_id']}.xml"

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)
        mask_size = slide.level_dimensions[level]

    annotation_size = get_annotation_size(Path(slide_record["mask_path"]))
    scale_x = mask_size[0] / annotation_size[0]
    scale_y = mask_size[1] / annotation_size[1]

    parser = load_asap_parser(xml_path)
    for group in classes:
        geometry = get_class_geometry(parser, group)
        polygons = iter_polygons(geometry)

        mask = Image.new("L", size=mask_size)
        canvas = ImageDraw.Draw(mask)

        for polygon in polygons:
            if polygon.is_empty:
                continue

            exterior_coords = [
                (x * scale_x, y * scale_y) for x, y in polygon.exterior.coords
            ]
            canvas.polygon(xy=exterior_coords, fill=255)

            for interior in polygon.interiors:  # draw holes
                interior_coords = [
                    (x * scale_x, y * scale_y) for x, y in interior.coords
                ]
                canvas.polygon(xy=interior_coords, fill=0)

        output_path = output_path_for(output_dir, group, slide_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_big_tiff(
            image=pyvips.Image.new_from_array(mask),
            path=output_path,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=mask_tile_width,
            tile_height=mask_tile_height,
        )


@with_cli_args(["+preprocessing/annotation_masks=beetle"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    xmls_dir = Path(config.xmls_dir)
    classes = list(config.classes)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        download_artifacts(config.metadata_uri), keep_default_na=False
    )
    annotated = metadata[metadata["has_annotation_xml"]]
    slides: list[SlideRecord] = annotated[
        ["slide_id", "slide_path", "mask_path"]
    ].to_dict("records")  # type: ignore[assignment]

    process_items(
        slides,
        process_item=process_slide,
        fn_kwargs={
            "xmls_dir": xmls_dir,
            "classes": classes,
            "level": config.level,
            "output_dir": str(output_dir),
            "mask_tile_width": config.mask_tile_width,
            "mask_tile_height": config.mask_tile_height,
        },
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(
        local_dir=str(output_dir),
        artifact_path=config.mlflow_artifact_path,
    )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
