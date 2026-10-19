"""Quality check of the nuclei segmentation (`nuclei_segmentation.py`) of a dataset.

For every slide of the dataset index, the output of the segmentation is read and checked:

1. Presence: a `slide_id=<slide>` folder with readable parquet parts and at least one nucleus.
2. Integrity: unique ids, finite coordinates inside the slide, centroids equal to the polygon
   means (as the pipeline computes them), plausible polygon areas (µm²).
3. Alignment: the share of nucleus centroids on the tissue mask.
4. Coverage: the segmentation keeps a nucleus only in the tile whose inner region (tile
   minus `overlap / 2` at each side) holds its centroid. Each inner region with at least
   `coverage.min_tissue_fraction` tissue (tissue mask) and no nucleus is a gap tile.
5. Missed nuclei: in every gap tile, nucleus-like hematoxylin blobs are counted on the
   H&E (at about `stain.mpp`). The threshold is the `stain.threshold_quantile` quantile of
   the hematoxylin at the slide's own segmented nuclei, sampled in up to
   `stain.calibration_tiles` segmented tiles. A gap tile with at least `stain.min_blobs`
   blobs most likely lost its nuclei; one without blobs is probably empty tissue.
6. Density: nuclei per mm² of tissue, compared with the median of the dataset.

Logs to MLflow `segmentation_qc.csv` (one row per slide: the metrics, `problems` = the
segmentation is incomplete or wrong, `review` = unusual but not necessarily wrong),
`flagged.csv` (slides with either) and `gap_tiles.csv` (every gap tile with its blobs).
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyvips
from joblib import Parallel, delayed
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from scipy.ndimage import label
from skimage.color import rgb2hed


def read_nuclei(nuclei_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Ids, polygons (n x vertices x 2), centroids (n x 2) and the number of parquet parts."""
    parts = sorted(nuclei_dir.glob("*.parquet"))
    table = ds.dataset(parts, format="parquet").to_table(
        columns=["id", "polygon", "centroid"]
    )
    ids = table.column("id").to_numpy()
    polygons = flat_values(table.column("polygon")).reshape(len(ids), -1, 2)
    centroids = flat_values(table.column("centroid")).reshape(len(ids), 2)
    return ids, polygons, centroids, len(parts)


def flat_values(column: pa.ChunkedArray) -> np.ndarray:
    """The values of a list column as float32.

    With Ray imported, Ray-written columns are read as its tensor extension type, whose
    storage is the plain list array.
    """
    array = column.combine_chunks()
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    return np.asarray(array.flatten(), dtype=np.float32)


def read_tissue(
    path: Path, slide_width: int, max_um_per_px: float, mpp: float
) -> tuple[np.ndarray, float]:
    """Coarsest tissue-mask page at ≤ `max_um_per_px`, and level-0 px per mask px."""
    best = None
    for page in range(pyvips.Image.new_from_file(str(path)).get_n_pages()):
        image = pyvips.Image.new_from_file(str(path), page=page)
        if slide_width / image.width * mpp <= max_um_per_px:
            best = image
    assert best is not None, f"No tissue mask page at ≤ {max_um_per_px} µm/px: {path}"
    array = best.numpy()
    array = array[..., 0] if array.ndim == 3 else array
    return array > 0, slide_width / best.width


def tile_grid(
    size: tuple[int, int], tile_extent: int, overlap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tile origins along x and y, as `grid_tiles(..., last="keep")` of the segmentation."""
    stride = tile_extent - overlap
    return tuple(
        np.arange(int(np.ceil(max(extent - tile_extent, 0) / stride)) + 1) * stride
        for extent in size
    )


def region_fraction(
    mask: np.ndarray,
    scale: float,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """Mean of `mask` over the level-0 boxes [x0, x1) x [y0, y1) (grid of x by y boxes)."""
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = mask.shape
    cx0, cx1 = (np.clip(np.round(v / scale).astype(int), 0, w) for v in (x0, x1))
    cy0, cy1 = (np.clip(np.round(v / scale).astype(int), 0, h) for v in (y0, y1))
    total = (
        integral[cy1[None, :], cx1[:, None]]
        - integral[cy0[None, :], cx1[:, None]]
        - integral[cy1[None, :], cx0[:, None]]
        + integral[cy0[None, :], cx0[:, None]]
    )
    area = np.maximum((cx1 - cx0)[:, None] * (cy1 - cy0)[None, :], 1)
    return total / area


def hematoxylin(
    slide: OpenSlide, level: int, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Hematoxylin optical density of a level-0 box, read at `level`."""
    x0, y0, x1, y1 = box
    downsample = slide.level_downsamples[level]
    size = (
        max(1, round((x1 - x0) / downsample)),
        max(1, round((y1 - y0) / downsample)),
    )
    rgb = np.asarray(slide.read_region((x0, y0), level, size).convert("RGB"))
    return rgb2hed(rgb)[..., 0]


def count_blobs(
    h: np.ndarray, threshold: float, um_per_px: float, stain: DictConfig
) -> int:
    """Connected hematoxylin-positive components with a nucleus-like area."""
    labels, n = label(h >= threshold)
    if n == 0:
        return 0
    areas = np.bincount(labels.ravel())[1:] * um_per_px**2
    return int(
        ((areas >= stain.min_blob_area_um2) & (areas <= stain.max_blob_area_um2)).sum()
    )


def check_slide(
    item: dict[str, Any], config: DictConfig
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seg, cov, stain = config.segmentation, config.coverage, config.stain
    row: dict[str, Any] = {"slide_id": item["slide_id"]}
    nuclei_dir = Path(item["nuclei_dir"])
    if not nuclei_dir.is_dir():
        return {**row, "status": "missing_output"}, []
    try:
        ids, polygons, centroids, row["n_parts"] = read_nuclei(nuclei_dir)
    except Exception as e:  # noqa: BLE001 - reported as an unreadable output
        return {**row, "status": "unreadable", "error": repr(e)}, []
    row["n_nuclei"] = len(ids)
    if len(ids) == 0:
        return {**row, "status": "empty"}, []

    with OpenSlide(item["slide_path"]) as slide:
        level = slide.closest_level(seg.mpp)
        width, height = slide.level_dimensions[level]
        mpp = float(np.mean(slide.slide_resolution(level)))
        stain_level = slide.closest_level(stain.mpp)
        stain_mpp = float(np.mean(slide.slide_resolution(stain_level)))
        level_downsample = slide.level_downsamples[level]

    # integrity
    row["duplicate_ids"] = int(len(ids) - len(np.unique(ids)))
    finite = np.isfinite(polygons).all((1, 2)) & np.isfinite(centroids).all(1)
    row["nonfinite"] = int((~finite).sum())
    row["out_of_bounds"] = int(
        (
            (centroids[:, 0] < 0)
            | (centroids[:, 0] >= width)
            | (centroids[:, 1] < 0)
            | (centroids[:, 1] >= height)
        ).sum()
    )
    row["max_centroid_offset_px"] = float(
        np.nanmax(np.abs(polygons.mean(1, dtype=np.float64) - centroids))
    )
    x, y = polygons[..., 0].astype(np.float64), polygons[..., 1].astype(np.float64)
    area = 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1)) * mpp**2
    row["median_area_um2"] = float(np.median(area))
    row["area_outlier_fraction"] = float(
        ((area < cov.min_nucleus_area_um2) | (area > cov.max_nucleus_area_um2)).mean()
    )

    if item["tissue_path"] is None:
        return {**row, "status": "no_tissue_mask"}, []
    mask, scale = read_tissue(
        Path(item["tissue_path"]),
        int(width * level_downsample),
        cov.mask_max_um_per_px,
        mpp / level_downsample,
    )
    scale /= level_downsample  # mask px -> segmentation-level px
    mx = np.clip((centroids[:, 0] / scale).astype(int), 0, mask.shape[1] - 1)
    my = np.clip((centroids[:, 1] / scale).astype(int), 0, mask.shape[0] - 1)
    row["on_tissue_fraction"] = float(mask[my, mx].mean())
    tissue_mm2 = mask.sum() * (scale * mpp) ** 2 / 1e6
    row["tissue_mm2"] = float(tissue_mm2)
    row["density_per_mm2"] = float(len(ids) / tissue_mm2) if tissue_mm2 > 0 else np.nan

    # coverage on the inner regions of the segmentation tiles
    half = seg.overlap / 2
    stride = seg.tile_extent - seg.overlap
    xs, ys = tile_grid((width, height), seg.tile_extent, seg.overlap)
    tissue = region_fraction(
        mask,
        scale,
        xs + half,
        np.minimum(xs + seg.tile_extent - half, width),
        ys + half,
        np.minimum(ys + seg.tile_extent - half, height),
    )
    owner = np.floor((centroids - half) / stride).astype(int)
    inside = (owner >= 0).all(1) & (owner[:, 0] < len(xs)) & (owner[:, 1] < len(ys))
    counts = np.zeros((len(xs), len(ys)), dtype=np.int64)
    np.add.at(counts, (owner[inside, 0], owner[inside, 1]), 1)
    tissue_tiles = tissue >= cov.min_tissue_fraction
    gaps = tissue_tiles & (counts == 0)
    row["tissue_tiles"] = int(tissue_tiles.sum())
    row["gap_tiles"] = int(gaps.sum())
    row["largest_gap_cluster"] = int(
        np.bincount(label(gaps)[0].ravel())[1:].max(initial=0)
    )

    # missed nuclei: hematoxylin blobs in the gap tiles, threshold from segmented nuclei
    rng = np.random.default_rng(stain.seed)
    segmented = np.argwhere(tissue_tiles & (counts >= stain.min_calibration_nuclei))
    gap_rows: list[dict[str, Any]] = []
    if len(segmented) == 0:
        row["stain_threshold"] = np.nan
        row["status"] = "ok"
        return row, gap_rows
    sample = segmented[rng.permutation(len(segmented))[: stain.calibration_tiles]]
    to_level0 = level_downsample
    with OpenSlide(item["slide_path"]) as slide:
        values, blobs, calib_nuclei = [], 0, 0
        h_tiles = []
        for i, j in sample:
            box = (
                int(xs[i] + half),
                int(ys[j] + half),
                int(min(xs[i] + seg.tile_extent - half, width)),
                int(min(ys[j] + seg.tile_extent - half, height)),
            )
            h = hematoxylin(slide, stain_level, tuple(int(v * to_level0) for v in box))
            sel = (owner[:, 0] == i) & (owner[:, 1] == j)
            px = ((centroids[sel] - [box[0], box[1]]) * mpp / stain_mpp).astype(int)
            px = np.clip(px, 0, [h.shape[1] - 1, h.shape[0] - 1])
            values.append(h[px[:, 1], px[:, 0]])
            h_tiles.append(h)
            calib_nuclei += int(sel.sum())
        threshold = float(np.quantile(np.concatenate(values), stain.threshold_quantile))
        blobs = sum(count_blobs(h, threshold, stain_mpp, stain) for h in h_tiles)
        row["stain_threshold"] = threshold
        row["calibration_blobs_per_nucleus"] = blobs / max(calib_nuclei, 1)

        for i, j in np.argwhere(gaps)[: stain.max_gap_tiles]:
            box = (
                int(xs[i] + half),
                int(ys[j] + half),
                int(min(xs[i] + seg.tile_extent - half, width)),
                int(min(ys[j] + seg.tile_extent - half, height)),
            )
            h = hematoxylin(slide, stain_level, tuple(int(v * to_level0) for v in box))
            n_blobs = count_blobs(h, threshold, stain_mpp, stain)
            gap_rows.append(
                {
                    "slide_id": item["slide_id"],
                    "x0": int(box[0] * to_level0),
                    "y0": int(box[1] * to_level0),
                    "x1": int(box[2] * to_level0),
                    "y1": int(box[3] * to_level0),
                    "tissue_fraction": float(tissue[i, j]),
                    "blobs": n_blobs,
                    "missed_nuclei": n_blobs >= stain.min_blobs,
                }
            )
    row["gap_tiles_checked"] = len(gap_rows)
    row["gap_tiles_missed_nuclei"] = int(sum(g["missed_nuclei"] for g in gap_rows))
    row["missed_blobs"] = int(sum(g["blobs"] for g in gap_rows if g["missed_nuclei"]))
    row["status"] = "ok"
    return row, gap_rows


PROBLEMS = (
    "missing_output",
    "unreadable",
    "empty",
    "no_tissue_mask",
    "duplicate_ids",
    "invalid_geometry",
    "off_tissue",
    "missed_nuclei",
)


def flag(slides: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Per slide, `problems` and `review`: ';'-joined reasons ('' = none).

    `problems`: the segmentation is incomplete or wrong; `review`: unusual, to look at.
    """
    f = config.flags
    density_median = slides["density_per_mm2"].median()
    rules = {
        "missing_output": slides["status"] == "missing_output",
        "unreadable": slides["status"] == "unreadable",
        "empty": slides["status"] == "empty",
        "no_tissue_mask": slides["status"] == "no_tissue_mask",
        "duplicate_ids": slides["duplicate_ids"].fillna(0) > 0,
        "invalid_geometry": (slides["nonfinite"].fillna(0) > 0)
        | (slides["out_of_bounds"].fillna(0) > 0)
        | (slides["max_centroid_offset_px"].fillna(0) > f.max_centroid_offset_px),
        "off_tissue": slides["on_tissue_fraction"].fillna(1) < f.min_on_tissue_fraction,
        "missed_nuclei": slides["gap_tiles_missed_nuclei"].fillna(0)
        >= f.min_missed_gap_tiles,
        "implausible_areas": slides["area_outlier_fraction"].fillna(0)
        > f.max_area_outlier_fraction,
        "low_density": slides["density_per_mm2"] < f.min_density_ratio * density_median,
        "no_stain_calibration": (slides["status"] == "ok")
        & slides["stain_threshold"].isna(),
    }
    reasons = pd.DataFrame(rules)
    joined = {
        name: reasons[columns].apply(lambda r: ";".join(r.index[r.to_numpy()]), axis=1)
        for name, columns in (
            ("problems", [c for c in rules if c in PROBLEMS]),
            ("review", [c for c in rules if c not in PROBLEMS]),
        )
    }
    return pd.DataFrame(joined)


@with_cli_args(["+preprocessing=nuclei_segmentation_qc"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    index = pd.read_csv(config.slides_index)
    slide_ids = sorted({Path(name).stem for name in index[config.slide_column]})
    if config.slide_ids is not None:
        slide_ids = [s for s in slide_ids if s in set(config.slide_ids)]
    tissue_dir = Path(download_artifacts(config.tissue_masks_uri))
    items = []
    for slide_id in slide_ids:
        tissue_path = tissue_dir / f"{slide_id}.tiff"
        items.append(
            {
                "slide_id": slide_id,
                "slide_path": str(Path(config.slides_dir) / f"{slide_id}.tiff"),
                "nuclei_dir": str(Path(config.nuclei_dir) / f"slide_id={slide_id}"),
                "tissue_path": str(tissue_path) if tissue_path.exists() else None,
            }
        )

    results = Parallel(n_jobs=config.n_jobs, verbose=5)(
        delayed(check_slide)(item, config) for item in items
    )
    slides = pd.DataFrame([row for row, _ in results])
    for column in (
        "duplicate_ids",
        "nonfinite",
        "out_of_bounds",
        "max_centroid_offset_px",
        "area_outlier_fraction",
        "on_tissue_fraction",
        "density_per_mm2",
        "gap_tiles_missed_nuclei",
        "stain_threshold",
    ):
        if column not in slides:
            slides[column] = np.nan
    slides = pd.concat([slides, flag(slides, config)], axis=1)
    gaps = pd.DataFrame([g for _, rows in results for g in rows])

    flagged = (slides["problems"] != "") | (slides["review"] != "")
    summary = {
        "slides": len(slides),
        "slides_with_problems": int((slides["problems"] != "").sum()),
        "slides_to_review": int(
            ((slides["problems"] == "") & (slides["review"] != "")).sum()
        ),
    }
    print(summary)
    for column in ("problems", "review"):
        reasons = slides[column].str.split(";").explode()
        print(reasons[reasons != ""].value_counts().to_string())
    logger.log_hyperparams(summary)
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        slides.to_csv(out / "segmentation_qc.csv", index=False)
        slides[flagged].to_csv(out / "flagged.csv", index=False)
        gaps.to_csv(out / "gap_tiles.csv", index=False)
        logger.log_artifacts(str(out), "segmentation_qc")


if __name__ == "__main__":
    main()
