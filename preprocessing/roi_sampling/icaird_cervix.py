"""Script for sampling ROIs inside the positive annotations of the iCAIRD cervical dataset.

The ROIs are meant to be re-annotated precisely by a pathologist. For each annotated slide
of the configured split, rectangular ROIs are sampled at random such that:
    - the ROIs cover ~`roi.target_fraction` of the slide's positive (high grade /
      malignant) annotation area,
    - each ROI is `roi.min_area_mm2`-`roi.max_area_mm2` large (uniformly distributed),
    - at least `roi.min_coverage` of each ROI lies inside the positive annotation,
    - ROIs do not overlap and are at least `roi.min_gap_um` apart.

The sampled area is counted as the part of the ROIs inside the positive annotation. When
10 % of a slide's positive area is smaller than a minimum-sized ROI, the slide still gets a
single minimum-sized ROI (otherwise slides with small lesions would never be sampled).

Outputs (logged as MLflow artifacts):
    roi_masks/<SLIDE_NAME>.tiff  binary mask (255 = ROI), one per slide with >= 1 ROI
    rois.csv                     one row per ROI
    slides_summary.csv           one row per slide, incl. why a slide got no ROIs
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
import shapely
import shapely.affinity
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from ratiopath.parsers import GeoJSONParser
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def load_class_region(
    parser: GeoJSONParser, pattern: str, scale_x: float, scale_y: float
) -> BaseGeometry:
    """Union of the class's polygons (holes included) in mask pixel coordinates."""
    polygons = [
        shapely.affinity.scale(
            polygon, xfact=scale_x, yfact=scale_y, origin=(0, 0)
        ).buffer(0)
        for polygon in parser.get_polygons(classification_name=pattern)
    ]
    return shapely.union_all(polygons) if polygons else Polygon()


def sample_points_in(
    region: BaseGeometry, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform random points inside `region`, via rejection sampling in its bounds."""
    min_x, min_y, max_x, max_y = region.bounds
    points = np.empty((0, 2))
    while len(points) < n:
        candidates = np.column_stack(
            [rng.uniform(min_x, max_x, 4 * n), rng.uniform(min_y, max_y, 4 * n)]
        )
        inside = shapely.contains_xy(region, candidates[:, 0], candidates[:, 1])
        points = np.vstack([points, candidates[inside]])
    return points[:n]


def make_box(
    cx: float, cy: float, area: float, ratio: float, flip: bool
) -> tuple[int, int, int, int]:
    w, h = np.sqrt(area * ratio), np.sqrt(area / ratio)
    w, h = (h, w) if flip else (w, h)
    x0, y0 = round(cx - w / 2), round(cy - h / 2)
    return x0, y0, x0 + round(w), y0 + round(h)


def best_effort_roi(
    positive: BaseGeometry,
    mask_size: tuple[int, int],
    min_area: float,
    max_aspect_ratio: float,
    rng: np.random.Generator,
    n_candidates: int = 2000,
) -> list[tuple[int, int, int, int]]:
    """Minimum-sized ROI with the highest coverage of `positive`.

    For slides where no ROI reaches the coverage threshold (small or fragmented positive
    annotations).
    """
    centers = sample_points_in(positive, n_candidates, rng)
    ratios = np.exp(rng.uniform(0, np.log(max_aspect_ratio), n_candidates))
    flips = rng.random(n_candidates) < 0.5

    best, best_inside = None, 0.0
    for (cx, cy), ratio, flip in zip(centers, ratios, flips, strict=True):
        x0, y0, x1, y1 = box = make_box(cx, cy, min_area, ratio, flip)
        if x0 < 0 or y0 < 0 or x1 > mask_size[0] or y1 > mask_size[1]:
            continue
        inside = shapely.clip_by_rect(positive, x0, y0, x1, y1).area
        if inside > best_inside:
            best, best_inside = box, inside
    return [best] if best else []


def sample_rois(
    positive: BaseGeometry,
    mask_size: tuple[int, int],
    pixel_area_mm2: float,
    mpp: float,
    rng: np.random.Generator,
    target_fraction: float,
    min_area_mm2: float,
    max_area_mm2: float,
    max_aspect_ratio: float,
    min_coverage: float,
    min_gap_um: float,
    max_attempts: int,
) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Samples non-overlapping ROIs (x0, y0, x1, y1) in mask pixel coordinates.

    If no ROI reaches `min_coverage`, falls back to the single best-covering minimum-sized
    ROI (second return value True), so that no slide with positive annotation is left out.
    """
    min_area = min_area_mm2 / pixel_area_mm2
    max_area = max_area_mm2 / pixel_area_mm2
    gap = min_gap_um / mpp
    budget = target_fraction * positive.area  # of positive area to be covered by ROIs
    min_gain = min_coverage * min_area

    rois: list[tuple[int, int, int, int]] = []
    placed = Polygon()
    covered = 0.0
    attempts = 0

    while attempts < max_attempts and (not rois or 2 * (budget - covered) >= min_gain):
        # a budget below a minimum-sized ROI still gets a (minimum-sized) ROI
        area_range = (min_area, min_area) if budget < min_gain else (min_area, max_area)
        batch = min(256, max_attempts - attempts)
        centers = sample_points_in(positive, batch, rng)
        areas = rng.uniform(*area_range, batch)
        ratios = np.exp(rng.uniform(0, np.log(max_aspect_ratio), batch))
        flips = rng.random(batch) < 0.5

        for (cx, cy), area, ratio, flip in zip(
            centers, areas, ratios, flips, strict=True
        ):
            attempts += 1
            x0, y0, x1, y1 = make_box(cx, cy, area, ratio, flip)

            if x0 < 0 or y0 < 0 or x1 > mask_size[0] or y1 > mask_size[1]:
                continue
            roi_area = (x1 - x0) * (y1 - y0)
            inside = shapely.clip_by_rect(positive, x0, y0, x1, y1).area
            if inside < min_coverage * roi_area:
                continue
            if shapely.box(x0 - gap, y0 - gap, x1 + gap, y1 + gap).intersects(placed):
                continue
            # accept unless overshooting the budget is worse than the current shortfall
            if rois and inside > 2 * (budget - covered):
                continue

            rois.append((x0, y0, x1, y1))
            placed = placed.union(shapely.box(x0, y0, x1, y1))
            covered += inside
            if covered >= budget:
                return rois, False
            break  # draw new candidates, the remaining budget changed

    if rois:
        return rois, False
    return best_effort_roi(positive, mask_size, min_area, max_aspect_ratio, rng), True


@ray.remote(num_cpus=1, memory=(4 * 1024**3))
def process_slide(
    slide_path: Path,
    annots_dir: Path,
    level: int,
    seed: int,
    positive_patterns: dict[str, str],
    roi_params: dict[str, Any],
    output_dir: str,
    summary_dir: str,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    slide_id = slide_path.stem
    summary: dict[str, Any] = {"slide_id": slide_id, "status": "ok"}
    rois_df = pd.DataFrame()

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)
        mask_size_base = slide.level_dimensions[0]
        mask_size = slide.level_dimensions[level]

    scale_x = mask_size[0] / mask_size_base[0]
    scale_y = mask_size[1] / mask_size_base[1]
    pixel_area_mm2 = mpp_x * mpp_y / 1e6

    parser = GeoJSONParser(annots_dir / f"{slide_id}.txt")
    regions = {
        name: load_class_region(parser, pattern, scale_x, scale_y)
        for name, pattern in positive_patterns.items()
    }
    positive = shapely.union_all(list(regions.values()))
    summary["positive_area_mm2"] = positive.area * pixel_area_mm2

    rois: list[tuple[int, int, int, int]] = []
    if positive.is_empty:
        summary["status"] = "no_positive_annotation"
    else:
        # reproducible per slide, independent of the processing order
        seed_seq = np.random.SeedSequence([seed, *slide_id.encode()])
        rois, fallback = sample_rois(
            positive,
            mask_size,
            pixel_area_mm2,
            mpp=mpp_x,
            rng=np.random.default_rng(seed_seq),
            **roi_params,
        )
        if not rois:
            summary["status"] = "no_feasible_roi"
        elif fallback:
            summary["status"] = "low_coverage_fallback"

    if rois:
        mask = Image.new("L", size=mask_size)
        canvas = ImageDraw.Draw(mask)
        rows = []
        for i, (x0, y0, x1, y1) in enumerate(rois):
            canvas.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
            box = shapely.box(x0, y0, x1, y1)
            rows.append(
                {
                    "slide_id": slide_id,
                    "roi_id": i,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "area_mm2": box.area * pixel_area_mm2,
                    "coverage": box.intersection(positive).area / box.area,
                    **{
                        f"coverage_{name}": box.intersection(region).area / box.area
                        for name, region in regions.items()
                    },
                }
            )
        rois_df = pd.DataFrame(rows)

        output_path = Path(
            output_dir, "roi_masks", slide_path.with_suffix(".tiff").name
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

    summary["n_rois"] = len(rois)
    summary["roi_area_mm2"] = float(rois_df["area_mm2"].sum()) if len(rois_df) else 0.0
    summary["sampled_positive_area_mm2"] = (
        float((rois_df["area_mm2"] * rois_df["coverage"]).sum())
        if len(rois_df)
        else 0.0
    )

    rois_df.to_csv(Path(summary_dir, f"{slide_id}_rois.csv"), index=False)
    Path(summary_dir, f"{slide_id}_summary.json").write_text(json.dumps(summary))


@with_cli_args(["+preprocessing/roi_sampling=icaird_cervix"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata_df = pd.read_csv(download_artifacts(config.metadata_uri))
    selected = metadata_df[
        (metadata_df["split"] == config.split) & metadata_df["has_annotation"]
    ]
    slide_paths = selected["slide_path"].map(Path)

    with TemporaryDirectory() as output_dir, TemporaryDirectory() as summary_dir:
        process_items(
            slide_paths,
            process_item=process_slide,
            fn_kwargs={
                "annots_dir": Path(config.annots_dir),
                "level": config.level,
                "seed": config.seed,
                "positive_patterns": dict(config.positive_patterns),
                "roi_params": {
                    "target_fraction": config.roi.target_fraction,
                    "min_area_mm2": config.roi.min_area_mm2,
                    "max_area_mm2": config.roi.max_area_mm2,
                    "max_aspect_ratio": config.roi.max_aspect_ratio,
                    "min_coverage": config.roi.min_coverage,
                    "min_gap_um": config.roi.min_gap_um,
                    "max_attempts": config.roi.max_attempts,
                },
                "output_dir": output_dir,
                "summary_dir": summary_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
            },
            max_concurrent=config.max_concurrent,
        )

        summaries = pd.DataFrame(
            [
                json.loads(p.read_text())
                for p in sorted(Path(summary_dir).glob("*.json"))
            ]
        ).merge(selected[["slide_id", "category", "subcategory"]], on="slide_id")
        rois = pd.concat(
            [
                pd.read_csv(p)
                for p in sorted(Path(summary_dir).glob("*_rois.csv"))
                if p.stat().st_size > 1  # slides without ROIs have an empty file
            ]
        )
        summaries["sampled_fraction"] = (
            summaries["sampled_positive_area_mm2"] / summaries["positive_area_mm2"]
        )

        summaries.to_csv(Path(output_dir, "slides_summary.csv"), index=False)
        rois.to_csv(Path(output_dir, "rois.csv"), index=False)
        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
