"""Script for sampling ROIs inside the positive annotations of the iCAIRD cervical dataset.

The ROIs are meant to be re-annotated precisely by a pathologist. For each annotated slide
of the configured split, rectangular ROIs are sampled at random such that:
    - the ROIs cover ~`roi.target_fraction` of the slide's positive (high grade /
      malignant) annotation area on tissue,
    - each ROI is `roi.min_area_mm2`-`roi.max_area_mm2` large (uniformly distributed),
    - at least `coverage_steps[0]` of each ROI lies inside the positive annotation on
      tissue (positive annotation ∩ tissue mask),
    - ROIs do not overlap and are at least `roi.min_gap_um` apart.

Slides where no ROI reaches the first coverage step (small or fragmented lesions) are not
left out: the coverage requirement is relaxed step by step (`coverage_steps`), and if even
the last step fails, the slide gets a single minimum-sized ROI placed where it covers the
most positive tissue. The coverage requirement used is recorded per slide. When 10 % of a
slide's positive area is smaller than a minimum-sized ROI, the slide still gets one
minimum-sized ROI.

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
from mlflow.exceptions import MlflowException
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


Box = tuple[int, int, int, int]
BATCH_SIZE = 1024


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


def rasterize(
    region: BaseGeometry, shape: tuple[int, int], scale_x: float, scale_y: float
) -> np.ndarray:
    """Boolean raster (height, width) of `region` given in ROI mask pixel coordinates."""
    canvas = Image.new("L", size=(shape[1], shape[0]))
    draw = ImageDraw.Draw(canvas)
    for polygon in getattr(region, "geoms", [region]):
        draw.polygon(
            [(x * scale_x, y * scale_y) for x, y in polygon.exterior.coords], fill=255
        )
        for interior in polygon.interiors:  # draw holes
            draw.polygon(
                [(x * scale_x, y * scale_y) for x, y in interior.coords], fill=0
            )
    return np.asarray(canvas) > 0


def load_tissue(tissue_uri: str, slide_id: str, shape: tuple[int, int]) -> np.ndarray:
    """Tissue mask artifact of the slide as a boolean raster cropped/padded to `shape`."""
    with TemporaryDirectory() as tmp_dir:
        path = download_artifacts(f"{tissue_uri}/{slide_id}.tiff", dst_path=tmp_dir)
        tissue = pyvips.Image.new_from_file(path).numpy() > 0

    out = np.zeros(shape, dtype=bool)
    h, w = min(shape[0], tissue.shape[0]), min(shape[1], tissue.shape[1])
    out[:h, :w] = tissue[:h, :w]
    return out


class CoverageMap:
    """Area of a boolean raster inside boxes, computed via an integral image.

    The raster is coarser than the ROI mask: `scale_x`/`scale_y` convert ROI mask pixels
    to raster pixels.
    """

    def __init__(self, raster: np.ndarray, scale_x: float, scale_y: float) -> None:
        self.height, self.width = raster.shape
        self.scale_x, self.scale_y = scale_x, scale_y
        self.integral = np.zeros((self.height + 1, self.width + 1), dtype=np.int32)
        np.cumsum(raster, axis=0, dtype=np.int32, out=self.integral[1:, 1:])
        np.cumsum(self.integral[1:, 1:], axis=1, out=self.integral[1:, 1:])

    @property
    def total_area(self) -> float:
        """Area of the whole raster in ROI mask pixels."""
        return float(self.integral[-1, -1]) / (self.scale_x * self.scale_y)

    def area(
        self, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray
    ) -> np.ndarray:
        """Area (in ROI mask pixels) of the raster inside the boxes."""

        def cell(values: np.ndarray, scale: float, size: int) -> np.ndarray:
            return np.clip(np.rint(values * scale), 0, size).astype(np.intp)

        cx0, cx1 = (cell(v, self.scale_x, self.width) for v in (x0, x1))
        cy0, cy1 = (cell(v, self.scale_y, self.height) for v in (y0, y1))
        ii = self.integral
        count = ii[cy1, cx1] - ii[cy0, cx1] - ii[cy1, cx0] + ii[cy0, cx0]
        return count / (self.scale_x * self.scale_y)


class CandidateSampler:
    """Draws random candidate boxes centered on random points of a boolean raster."""

    def __init__(
        self,
        raster: np.ndarray,
        scale_x: float,
        scale_y: float,
        mask_size: tuple[int, int],
        max_aspect_ratio: float,
        rng: np.random.Generator,
    ) -> None:
        self.raster = raster
        self.scale_x, self.scale_y = scale_x, scale_y
        self.mask_size = mask_size
        self.max_aspect_ratio = max_aspect_ratio
        self.rng = rng

        rows, cols = (
            np.flatnonzero(raster.any(axis=1)),
            np.flatnonzero(raster.any(axis=0)),
        )
        self.y_range = (rows[0], rows[-1] + 1)
        self.x_range = (cols[0], cols[-1] + 1)

    def centers(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Uniform random points of the raster (in ROI mask pixels), by rejection."""
        xs, ys = [], []
        while sum(len(x) for x in xs) < n:
            x = self.rng.integers(*self.x_range, size=4 * n)
            y = self.rng.integers(*self.y_range, size=4 * n)
            keep = self.raster[y, x]
            xs.append(x[keep])
            ys.append(y[keep])
        x, y = np.concatenate(xs)[:n], np.concatenate(ys)[:n]
        return (
            (x + self.rng.random(n)) / self.scale_x,
            (y + self.rng.random(n)) / self.scale_y,
        )

    def boxes(
        self, n: int, area_range: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Candidate boxes (x0, y0, x1, y1); those outside the mask are marked by x1 < 0."""
        cx, cy = self.centers(n)
        area = self.rng.uniform(*area_range, n)
        ratio = np.exp(self.rng.uniform(0, np.log(self.max_aspect_ratio), n))
        w, h = np.sqrt(area * ratio), np.sqrt(area / ratio)
        w, h = np.where(self.rng.random(n) < 0.5, (h, w), (w, h))

        x0, y0 = np.rint(cx - w / 2).astype(int), np.rint(cy - h / 2).astype(int)
        x1, y1 = x0 + np.rint(w).astype(int), y0 + np.rint(h).astype(int)
        inside = (x0 >= 0) & (y0 >= 0) & (x1 <= self.mask_size[0])
        inside &= y1 <= self.mask_size[1]
        return x0, y0, np.where(inside, x1, -1), y1


def best_effort_roi(
    coverage_map: CoverageMap, sampler: CandidateSampler, min_area: float, n: int
) -> list[Box]:
    """Minimum-sized ROI with the highest coverage, for slides no coverage step works for."""
    x0, y0, x1, y1 = sampler.boxes(n, (min_area, min_area))
    inside = np.where(x1 >= 0, coverage_map.area(x0, y0, x1, y1), 0.0)
    best = int(np.argmax(inside))
    if inside[best] <= 0:
        return []
    return [(int(x0[best]), int(y0[best]), int(x1[best]), int(y1[best]))]


def sample_rois_at_coverage(
    coverage_map: CoverageMap,
    sampler: CandidateSampler,
    min_coverage: float,
    budget: float,
    min_area: float,
    max_area: float,
    gap: float,
    max_attempts: int,
) -> list[Box]:
    """Samples non-overlapping ROIs with at least `min_coverage` of their area covered."""
    min_gain = min_coverage * min_area
    # a budget below a minimum-sized ROI still gets a (minimum-sized) ROI
    area_range = (min_area, min_area) if budget < min_gain else (min_area, max_area)

    rois: list[Box] = []
    placed = Polygon()
    covered = 0.0

    while not rois or 2 * (budget - covered) >= min_gain:
        accepted = False
        for _ in range(max(1, max_attempts // BATCH_SIZE)):
            x0, y0, x1, y1 = sampler.boxes(BATCH_SIZE, area_range)
            inside = coverage_map.area(x0, y0, x1, y1)
            valid = (x1 >= 0) & (inside >= min_coverage * (x1 - x0) * (y1 - y0))

            for i in np.flatnonzero(valid):
                box = (int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i]))
                if shapely.box(
                    box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap
                ).intersects(placed):
                    continue
                # accept unless overshooting the budget is worse than the shortfall
                if rois and inside[i] > 2 * (budget - covered):
                    continue
                rois.append(box)
                placed = placed.union(shapely.box(*box))
                covered += float(inside[i])
                accepted = True
                break
            if accepted:
                break

        if not accepted or covered >= budget:
            break

    return rois


def sample_rois(
    coverage_map: CoverageMap,
    sampler: CandidateSampler,
    pixel_area_mm2: float,
    mpp: float,
    target_fraction: float,
    min_area_mm2: float,
    max_area_mm2: float,
    coverage_steps: list[float],
    min_gap_um: float,
    max_attempts: int,
) -> tuple[list[Box], float | None]:
    """Samples ROIs (x0, y0, x1, y1) in mask pixel coordinates.

    Tries the coverage steps in order and returns the ROIs of the first step that yields
    any, together with that step. If none does, returns the best-effort ROI and None.
    """
    min_area = min_area_mm2 / pixel_area_mm2
    max_area = max_area_mm2 / pixel_area_mm2
    budget = target_fraction * coverage_map.total_area

    for min_coverage in coverage_steps:
        rois = sample_rois_at_coverage(
            coverage_map,
            sampler,
            min_coverage,
            budget,
            min_area,
            max_area,
            gap=min_gap_um / mpp,
            max_attempts=max_attempts,
        )
        if rois:
            return rois, min_coverage

    return best_effort_roi(coverage_map, sampler, min_area, max_attempts), None


@ray.remote(num_cpus=1, memory=(8 * 1024**3))
def process_slide(
    slide_path: Path,
    annots_dir: Path,
    level: int,
    tissue_uri: str,
    tissue_level: int,
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
        raster_size = slide.level_dimensions[tissue_level]

    scale_x = mask_size[0] / mask_size_base[0]
    scale_y = mask_size[1] / mask_size_base[1]
    raster_scale_x = raster_size[0] / mask_size[0]
    raster_scale_y = raster_size[1] / mask_size[1]
    raster_shape = (raster_size[1], raster_size[0])
    pixel_area_mm2 = mpp_x * mpp_y / 1e6

    parser = GeoJSONParser(annots_dir / f"{slide_id}.txt")
    regions = {
        name: load_class_region(parser, pattern, scale_x, scale_y)
        for name, pattern in positive_patterns.items()
    }
    positive = shapely.union_all(list(regions.values()))
    summary["positive_area_mm2"] = positive.area * pixel_area_mm2

    rois: list[Box] = []
    positive_raster = np.zeros(raster_shape, dtype=bool)
    positive_tissue = positive_raster
    tissue = np.ones(raster_shape, dtype=bool)
    if positive.is_empty:
        summary["status"] = "no_positive_annotation"
    else:
        positive_raster = rasterize(
            positive, raster_shape, raster_scale_x, raster_scale_y
        )
        try:
            tissue = load_tissue(tissue_uri, slide_id, raster_shape)
        except (MlflowException, OSError):
            tissue = np.ones(raster_shape, dtype=bool)
            summary["tissue_mask"] = "missing"
        positive_tissue = positive_raster & tissue
        if not positive_tissue.any():  # annotation outside of the tissue mask
            positive_tissue, tissue = positive_raster, np.ones(raster_shape, dtype=bool)
            summary["tissue_mask"] = "ignored"
        summary.setdefault("tissue_mask", "used")

    if not positive.is_empty and not positive_tissue.any():
        summary["status"] = "no_feasible_roi"
    elif not positive.is_empty:
        coverage_map = CoverageMap(positive_tissue, raster_scale_x, raster_scale_y)
        summary["positive_tissue_area_mm2"] = coverage_map.total_area * pixel_area_mm2
        # reproducible per slide, independent of the processing order
        seed_seq = np.random.SeedSequence([seed, *slide_id.encode()])
        sampler = CandidateSampler(
            positive_tissue,
            raster_scale_x,
            raster_scale_y,
            mask_size,
            max_aspect_ratio=roi_params["max_aspect_ratio"],
            rng=np.random.default_rng(seed_seq),
        )
        rois, step = sample_rois(
            coverage_map,
            sampler,
            pixel_area_mm2,
            mpp=mpp_x,
            target_fraction=roi_params["target_fraction"],
            min_area_mm2=roi_params["min_area_mm2"],
            max_area_mm2=roi_params["max_area_mm2"],
            coverage_steps=list(roi_params["coverage_steps"]),
            min_gap_um=roi_params["min_gap_um"],
            max_attempts=roi_params["max_attempts"],
        )
        summary["min_coverage_used"] = step
        if not rois:
            summary["status"] = "no_feasible_roi"
        elif step is None:
            summary["status"] = "best_effort"
        elif step < roi_params["coverage_steps"][0]:
            summary["status"] = "relaxed_coverage"

    if rois:
        mask = Image.new("L", size=mask_size)
        canvas = ImageDraw.Draw(mask)
        rows = []
        for i, (x0, y0, x1, y1) in enumerate(rois):
            canvas.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
            box = shapely.box(x0, y0, x1, y1)
            window = (
                slice(round(y0 * raster_scale_y), max(round(y1 * raster_scale_y), 1)),
                slice(round(x0 * raster_scale_x), max(round(x1 * raster_scale_x), 1)),
            )
            rows.append(
                {
                    "slide_id": slide_id,
                    "roi_id": i,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "area_mm2": box.area * pixel_area_mm2,
                    # fraction of the ROI inside the positive annotation on tissue
                    "coverage": float(positive_tissue[window].mean()),
                    "tissue_fraction": float(tissue[window].mean()),
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
                "tissue_uri": config.tissue_uri,
                "tissue_level": config.tissue_level,
                "seed": config.seed,
                "positive_patterns": dict(config.positive_patterns),
                "roi_params": {
                    "target_fraction": config.roi.target_fraction,
                    "min_area_mm2": config.roi.min_area_mm2,
                    "max_area_mm2": config.roi.max_area_mm2,
                    "max_aspect_ratio": config.roi.max_aspect_ratio,
                    "coverage_steps": list(config.roi.coverage_steps),
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
            summaries["sampled_positive_area_mm2"]
            / summaries["positive_tissue_area_mm2"]
        )

        summaries.to_csv(Path(output_dir, "slides_summary.csv"), index=False)
        rois.to_csv(Path(output_dir, "rois.csv"), index=False)
        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
