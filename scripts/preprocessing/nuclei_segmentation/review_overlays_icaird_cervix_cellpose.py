"""Overlay masks and review regions for the flagged iCAIRD cervix Cellpose slides.

For every slide in icaird_cervix_cellpose_review.csv that has nuclei and still has a
problem (not the re-segmented white-balance slides):
    - `masks/<slide_id>.tiff`: the nuclei rasterised at level 1 (~0.5 µm/px), 128 inside
      a nucleus, 255 inside an object larger than 150 µm² (whole cells, muscle fibres),
      0 on a 1 px border so touching nuclei stay apart;
    - `regions.json`: per slide, the level-0 box of a 2 x 2-tile region showing the
      problem, for the report to open the viewer on:
        red blood cells   densest cluster of sampled objects without hematoxylin;
        whole cells,
        muscle fibres     densest cluster of objects > 150 µm²;
        out of focus      the largest group of tissue tiles without nuclei;
        faint nuclei      a tissue tile of median nucleus density.

Usage:
    uv run python scripts/preprocessing/nuclei_segmentation/review_overlays_icaird_cervix_cellpose.py \
        GRIDS_DIR [--workers 2]
GRIDS_DIR holds the `grids/<slide_id>.npz` of deep_check_icaird_cervix_cellpose.py.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvips
from openslide import OpenSlide
from rationai.masks import write_big_tiff
from skimage.color import rgb2hed


HERE = Path(__file__).parent
SEG = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/iCAIRD/cervix_cellpose"
)
SLIDES = Path("/mnt/data/Public/Public/cervix/iCAIRD/slides")
OUT = Path("/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/icaird_cervix_review")
LEVEL, LARGE_UM2 = 1, 150
STRIDE, MARGIN = 1984, 32
H_MIN, H_E_RATIO = 0.005, 0.15  # as in stain_check_icaird_cervix_cellpose.py


def polygons(slide_id: str):
    for f in sorted((SEG / f"slide_id={slide_id}").glob("*.parquet")):
        col = pq.read_table(f, columns=["polygon"]).column("polygon").combine_chunks()
        yield np.asarray(col.flatten()).reshape(-1, 64, 2)


def areas_um2(polys: np.ndarray, mpp: float) -> np.ndarray:
    local = polys.astype(np.float64) - polys.mean(axis=1, keepdims=True)
    x, y = local[..., 0], local[..., 1]
    return 0.5 * np.abs((x * np.roll(y, -1, 1)).sum(1) - (np.roll(x, -1, 1) * y).sum(1)) * mpp**2


def write_mask(slide_id: str) -> dict:
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        w, h = slide.level_dimensions[LEVEL]
        scale = w / slide.level_dimensions[0][0]
        mpp = float(slide.properties["openslide.mpp-x"])
    mask = np.zeros((h, w), np.uint8)
    large_xy = []
    for polys in polygons(slide_id):
        big = areas_um2(polys, mpp) > LARGE_UM2
        large_xy.append(polys[big].mean(axis=1))
        scaled = [np.round(p * scale).astype(np.int32) for p in polys]
        for value, chosen in ((128, ~big), (255, big)):
            pts = [scaled[i] for i in np.flatnonzero(chosen)]
            if pts:
                cv2.fillPoly(mask, pts, value)
                cv2.polylines(mask, pts, True, 0, 1)
    out = OUT / "masks" / f"{slide_id}.tiff"
    write_big_tiff(
        pyvips.Image.new_from_array(mask),
        path=out,
        mpp_x=mpp / scale,
        mpp_y=mpp / scale,
    )
    del mask
    return {"slide_id": slide_id, "large_xy": np.concatenate(large_xy).tolist()}


def densest_box(points: np.ndarray) -> list[int] | None:
    """Level-0 box of the 2 x 2 tiles holding the most points."""
    if len(points) == 0:
        return None
    cells = np.floor((points - MARGIN) / STRIDE).astype(int)
    ids, counts = np.unique(cells, axis=0, return_counts=True)
    score = {tuple(c): n for c, n in zip(ids, counts, strict=True)}
    best = max(
        score,
        key=lambda c: sum(score.get((c[0] + dx, c[1] + dy), 0) for dx in (0, 1) for dy in (0, 1)),
    )
    x0, y0 = best[0] * STRIDE + MARGIN, best[1] * STRIDE + MARGIN
    return [x0, y0, x0 + 2 * STRIDE, y0 + 2 * STRIDE]


def no_hematoxylin_points(slide_id: str, n: int = 1500) -> np.ndarray:
    """Centroids of sampled objects without hematoxylin."""
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    tables = [pq.read_table(f, columns=["polygon"]) for f in files]
    polys = np.concatenate(
        [np.asarray(t.column("polygon").combine_chunks().flatten()).reshape(-1, 64, 2) for t in tables]
    )
    rng = np.random.default_rng(0)
    polys = polys[rng.choice(len(polys), min(n, len(polys)), replace=False)]
    out = []
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        for p in polys:
            x0, y0 = np.floor(p.min(axis=0)).astype(int) - 1
            x1, y1 = np.ceil(p.max(axis=0)).astype(int) + 1
            img = np.asarray(
                slide.read_region((int(x0), int(y0)), 0, (int(x1 - x0), int(y1 - y0))).convert("RGB")
            )
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [(p - [x0, y0]).astype(np.int32)], 1)
            if m.any():
                hh, e, _ = rgb2hed(img)[m > 0].mean(axis=0)
                if hh < H_MIN and hh < H_E_RATIO * e:
                    out.append(p.mean(axis=0))
    return np.array(out).reshape(-1, 2)


def grid_box(grid: np.lib.npyio.NpzFile, category: str) -> list[int]:
    tissue, nuclei = grid["tissue"], grid["nuclei"]
    if category == "out_of_focus":
        empty = ((tissue >= 0.5) & (nuclei == 0)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(empty, connectivity=4)
        i0, j0, cw, ch = stats[1 + np.argmax(stats[1:, 4]), :4]
        ci, cj = i0 + cw // 2 - 1, j0 + ch // 2
    else:  # a tile of median density
        full = tissue >= 0.9
        js, is_ = np.nonzero(full)
        k = np.argsort(nuclei[full] / tissue[full])[len(js) // 2]
        cj, ci = js[k], is_[k]
    x0, y0 = int(ci * STRIDE + MARGIN), int(cj * STRIDE + MARGIN)
    return [x0, y0, x0 + 2 * STRIDE, y0 + 2 * STRIDE]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("grids", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    review = pd.read_csv(HERE / "icaird_cervix_cellpose_review.csv")
    review = review[~review.category.isin(["white_balance_broken", "not_segmented"])]
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    todo = [s for s in review.slide_id if not (OUT / "masks" / f"{s}.tiff").exists()]
    large = {}
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(write_mask, todo):
            large[r["slide_id"]] = np.array(r["large_xy"]).reshape(-1, 2)
            print("mask", r["slide_id"], flush=True)

    regions_path = OUT / "regions.json"
    regions = json.loads(regions_path.read_text()) if regions_path.exists() else {}
    for _, row in review.iterrows():
        s, category = row.slide_id, row.category
        if s in regions:
            continue
        if category == "red_blood_cells_as_nuclei":
            box = densest_box(no_hematoxylin_points(s))
        elif category in ("whole_cells_as_nuclei", "muscle_fibres_as_nuclei"):
            pts = large.get(s)
            if pts is None:
                with OpenSlide(str(SLIDES / f"{s}.tiff")) as slide:
                    mpp = float(slide.properties["openslide.mpp-x"])
                pts = np.concatenate([p[areas_um2(p, mpp) > LARGE_UM2].mean(axis=1) for p in polygons(s)])
            box = densest_box(pts)
        else:
            box = grid_box(np.load(args.grids / f"{s}.npz"), category)
        regions[s] = [int(v) for v in box] if box else None
        print("region", s, box, flush=True)
    regions_path.write_text(json.dumps(regions, indent=1))


if __name__ == "__main__":
    main()
