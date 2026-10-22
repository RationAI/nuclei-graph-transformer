"""Classic nuclear morphometry of every nucleus of a split (step 1 of the concept morphometry).

Writes `morphometry_<split>.parquet` to `output_dir`, one row per nucleus in the order of
`activations/<split>/index.parquet` (so row i matches row i of `codes_<split>.npy`):

- polygon features of all nuclei, from the segmentation polygons (64 vertices, level-0
  px, converted to µm with the slide's mpp): area (shoelace), perimeter, long and short
  axis (2 x the semi-axes of the polygon's moment ellipse, i.e. 4 sqrt of the eigenvalues
  of its area covariance), axis ratio (long / short) and circularity (4 pi area /
  perimeter^2);
- (if `od.enabled`, test only) hematoxylin optical density on a subsample (`od.max_per_slide` random nuclei per slide
  plus each concept's `od.top_per_concept` most strongly activated test nuclei): the
  slide is read at level 0, colour-deconvolved (`skimage.color.rgb2hed`, Ruifrok-Johnston)
  and the hematoxylin channel averaged over the pixels whose centres lie in the polygon
  (`h_od_mean`, in rgb2hed units: ln-OD / -ln(1e-6)); `h_od_integrated` = mean x area.
  Per slide, both are divided by the median `h_od_mean` of the slide's random-sample
  nuclei (`*_norm`; the top-activated nuclei are left out of the reference so that they
  cannot shift it). NaN where not measured.

The polygons of a slide are its nuclei parquet sorted by `id` (as in the datasets); their
id checksum must equal `ids_crc32` of `slides.parquet`, and the written nucleus ids must
equal the index's. Nothing here names or orders concepts (`od_top_lists` only counts the
concepts' top lists a nucleus is in).
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from ratiopath.openslide import OpenSlide
from skimage.color import rgb2hed
from skimage.draw import polygon2mask

from xai.prepare_nuclei import ids_checksum


POLYGON_COLUMNS = [
    "area_um2",
    "perimeter_um",
    "long_axis_um",
    "short_axis_um",
    "axis_ratio",
    "circularity",
]


def polygon_features(polygons_um: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Features of closed polygons (n x v x 2, µm); vertex order does not matter."""
    p = polygons_um - polygons_um.mean(axis=1, keepdims=True)  # for numerical stability
    x, y = p[..., 0], p[..., 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    signed_area = cross.sum(axis=1) / 2
    # area moments by Green's theorem (signed; the signs cancel in the ratios below)
    cx = ((x + xn) * cross).sum(axis=1) / (6 * signed_area)
    cy = ((y + yn) * cross).sum(axis=1) / (6 * signed_area)
    xx = ((x**2 + x * xn + xn**2) * cross).sum(axis=1) / (12 * signed_area) - cx**2
    yy = ((y**2 + y * yn + yn**2) * cross).sum(axis=1) / (12 * signed_area) - cy**2
    xy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cross).sum(axis=1) / (24 * signed_area) - cx * cy
    # eigenvalues of [[xx, xy], [xy, yy]]; an ellipse with semi-axes a, b has a^2 / 4, b^2 / 4
    mean, half_gap = (xx + yy) / 2, np.sqrt(((xx - yy) / 2) ** 2 + xy**2)
    long_axis = 4 * np.sqrt(mean + half_gap)
    short_axis = 4 * np.sqrt(np.maximum(mean - half_gap, 0.0))
    area = np.abs(signed_area)
    perimeter = np.hypot(xn - x, yn - y).sum(axis=1)
    return {
        "area_um2": area,
        "perimeter_um": perimeter,
        "long_axis_um": long_axis,
        "short_axis_um": short_axis,
        "axis_ratio": long_axis / short_axis,
        "circularity": 4 * np.pi * area / perimeter**2,
        # long axis angle in image coordinates (x right, y down), modulo pi
        "orientation_rad": 0.5 * np.arctan2(2 * xy, xx - yy),
    }


def hematoxylin_means(slide_path: str, polygons_px: NDArray[np.float32]) -> tuple[NDArray, NDArray]:
    """Mean hematoxylin (rgb2hed) over each polygon's pixels at level 0, and pixel counts."""
    means = np.full(len(polygons_px), np.nan)
    counts = np.zeros(len(polygons_px), dtype=np.int64)
    with OpenSlide(slide_path) as wsi:
        for i, polygon in enumerate(polygons_px):
            lx, ly = np.floor(polygon.min(axis=0)).astype(int) - 1
            rx, ry = np.ceil(polygon.max(axis=0)).astype(int) + 2
            rgb = np.asarray(wsi.read_region((int(lx), int(ly)), 0, (int(rx - lx), int(ry - ly))).convert("RGB"))
            # pixel (i, j) is centred at (lx + j, ly + i), as in nucleus_patch; only the
            # pixels whose centre lies inside the polygon (cv2.fillPoly also fills the
            # pixels the outline touches, ~20 % more at this resolution)
            local = polygon.astype(np.float64) - [lx, ly]
            inside = polygon2mask(rgb.shape[:2], local[:, ::-1])
            counts[i] = inside.sum()
            if counts[i]:
                means[i] = rgb2hed(rgb)[..., 0][inside].mean()
    return means, counts


def process_slide(slide: dict, od_rows: NDArray[np.int64]) -> dict:
    """Polygon features of all of the slide's nuclei, OD of `od_rows` (slide-local rows)."""
    nuclei = pd.read_parquet(slide["slide_nuclei_path"], columns=["id", "polygon"])
    nuclei = nuclei.sort_values("id").reset_index(drop=True)
    ids = nuclei["id"].to_numpy()
    assert len(ids) == slide["n_nuclei"], slide["slide_id"]
    assert ids_checksum(ids) == slide["ids_crc32"], f"nuclei differ from the index: {slide['slide_id']}"
    polygons_px = np.array(nuclei["polygon"].tolist(), dtype=np.float32).reshape(len(nuclei), -1, 2)
    mpp = np.array([slide["mpp_x"], slide["mpp_y"]])
    features = polygon_features(polygons_px.astype(np.float64) * mpp)
    od_mean, od_pixels = hematoxylin_means(slide["slide_path"], polygons_px[od_rows])
    return {"nucleus_id": ids, "n_vertices": polygons_px.shape[1], **features,
            "od_mean": od_mean, "od_pixels": od_pixels}


def od_subsample(slides: pd.DataFrame, codes: NDArray, max_per_slide: int, top_per_concept: int,
                 seed: int) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """Random rows per slide (flag) and, per row, the number of concepts' top lists it is in."""
    rng = np.random.default_rng(seed)
    random = np.zeros(len(codes), dtype=bool)
    for offset, n in zip(slides["offset"], slides["n_nuclei"], strict=True):
        random[offset + rng.choice(n, size=min(n, max_per_slide), replace=False)] = True
    top_lists = np.zeros(len(codes), dtype=np.int64)
    for j in range(codes.shape[1]):
        column = np.asarray(codes[:, j])
        top = np.argpartition(-column, top_per_concept)[:top_per_concept]
        top_lists[top[column[top] > 0]] += 1
    return random, top_lists


@with_cli_args(["+xai=morphometry_features"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
def main(config: DictConfig) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = config.split

    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == split].sort_values("offset").reset_index(drop=True)
    index = pq.read_table(Path(config.activations_dir) / split / "index.parquet",
                          columns=["slide_id", "nucleus_id"])
    n = index.num_rows
    assert n == slides["n_nuclei"].sum()

    # OD (test only): the random sample per slide and the concepts' top lists
    random, top_lists = np.zeros(n, dtype=bool), np.zeros(n, dtype=np.int64)
    if config.od.enabled:
        codes = np.load(Path(config.concept_dir) / f"codes_{split}.npy", mmap_mode="r")
        assert len(codes) == n
        random, top_lists = od_subsample(slides, codes, config.od.max_per_slide,
                                         config.od.top_per_concept, config.seed)
    measured = random | (top_lists > 0)

    records = slides.to_dict("records")
    results = Parallel(n_jobs=config.n_jobs, verbose=5)(
        delayed(process_slide)(s, np.flatnonzero(measured[s["offset"]: s["offset"] + s["n_nuclei"]]))
        for s in records
    )

    table = {"slide_id": index.column("slide_id").to_numpy(),
             "nucleus_id": np.concatenate([r["nucleus_id"] for r in results])}
    assert (table["nucleus_id"] == index.column("nucleus_id").to_numpy()).all(), "nucleus ids differ"
    for column in POLYGON_COLUMNS:
        table[column] = np.concatenate([r[column] for r in results]).astype(np.float32)
    summary = {"split": split, "seed": config.seed, "n_nuclei": n, "n_slides": len(slides)}

    if config.od.enabled:
        od_mean = np.full(n, np.nan)
        od_pixels = np.zeros(n, dtype=np.int32)
        od_mean[measured] = np.concatenate([r["od_mean"] for r in results])
        od_pixels[measured] = np.concatenate([r["od_pixels"] for r in results])
        area = table["area_um2"].astype(np.float64)

        # per-slide reference: the median mean OD of the slide's random-sample nuclei
        slide_of_row = np.repeat(np.arange(len(slides)), slides["n_nuclei"])
        reference = (pd.Series(od_mean[random]).groupby(slide_of_row[random]).median()
                     .reindex(range(len(slides))).to_numpy())
        table |= {
            "h_od_mean": od_mean.astype(np.float32),
            "h_od_integrated": (od_mean * area).astype(np.float32),
            "h_od_mean_norm": (od_mean / reference[slide_of_row]).astype(np.float32),
            "h_od_integrated_norm": (od_mean / reference[slide_of_row] * area).astype(np.float32),
            "od_pixels": od_pixels,
            "od_random_sample": random,
            "od_top_lists": top_lists.astype(np.int8),
        }
        (out_dir / f"od_reference_{split}.csv").write_text(
            pd.DataFrame({"slide_id": slides["slide_id"], "median_h_od_mean_random": reference})
            .to_csv(index=False)
        )
        summary |= {"od": dict(config.od), "n_od_measured": int(measured.sum()),
                    "n_od_random": int(random.sum()), "n_od_top_only": int((measured & ~random).sum())}

    pd.DataFrame(table).to_parquet(out_dir / f"morphometry_{split}.parquet", index=False)
    (out_dir / f"morphometry_features_{split}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
