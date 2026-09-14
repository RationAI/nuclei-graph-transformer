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
import os
import re
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
from ratiopath.parsers import ASAPParser
from shapely import MultiPolygon, make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

WSI_EXTENSIONS = {".tif", ".tiff", ".svs"}

# A handful of BEETLE XMLs contain coordinates exported with a comma as the decimal
# separator (e.g. X="28881,7012" instead of X="28881.7012"), which ratiopath's
# ASAPParser cannot parse as a float.
DECIMAL_COMMA_COORD = re.compile(r'([XY])="(\d+),(\d+)"')


def slugify_class(group: str) -> str:
    return group.replace(" ", "_")


def find_wsi_path(wsis_dir: Path, stem: str) -> Path | None:
    for ext in WSI_EXTENSIONS:
        candidate = wsis_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_asap_parser(xml_path: Path) -> ASAPParser:
    xml_text = xml_path.read_text(encoding="utf-8")
    xml_text = DECIMAL_COMMA_COORD.sub(r'\1="\2.\3"', xml_text)
    return ASAPParser(io.StringIO(xml_text))


def get_class_geometry(parser: ASAPParser, group: str) -> BaseGeometry:
    geometry = MultiPolygon(list(parser.get_polygons(part_of_group=group)))
    return make_valid(geometry) if not geometry.is_valid else geometry


def iter_polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Flattens a (possibly nested) geometry into its constituent Polygons.

    `make_valid` on a self-intersecting MultiPolygon can return a GeometryCollection
    mixing Polygons with degenerate LineStrings/Points, instead of a clean Polygon or
    MultiPolygon -- those non-polygonal parts have zero area and are dropped here.
    """
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [p for geom in geometry.geoms for p in iter_polygons(geom)]
    return []


@ray.remote(num_cpus=1, memory=(3 * 1024**3))
def process_slide(
    slide_xml_paths: tuple[Path, Path],
    classes: list[str],
    level: int,
    output_dir: str,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    slide_path, xml_path = slide_xml_paths

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)
        mask_size_base = slide.level_dimensions[0]
        mask_size = slide.level_dimensions[level]

    scale_x = mask_size[0] / mask_size_base[0]
    scale_y = mask_size[1] / mask_size_base[1]

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

        output_path = Path(
            output_dir, slugify_class(group), slide_path.with_suffix(".tiff").name
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


@with_cli_args(["+preprocessing=annotation_masks_beetle"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    xmls_dir = Path(config.xmls_dir)
    wsis_dir = Path(config.wsis_dir)
    classes = list(config.classes)

    slides, missing_wsis = [], []
    for xml_path in sorted(xmls_dir.glob("*.xml")):
        wsi_path = find_wsi_path(wsis_dir, xml_path.stem)
        if wsi_path is None:
            missing_wsis.append(xml_path.stem)
        else:
            slides.append((wsi_path, xml_path))

    with TemporaryDirectory(dir=os.getcwd()) as output_dir:
        process_items(
            slides,
            process_item=process_slide,
            fn_kwargs={
                "classes": classes,
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
        csv_path = Path(output_dir, "missing_slides.csv")
        pd.DataFrame({"slide_stem": missing_wsis}).to_csv(csv_path, index=False)
        logger.log_artifact(local_path=str(csv_path))


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
