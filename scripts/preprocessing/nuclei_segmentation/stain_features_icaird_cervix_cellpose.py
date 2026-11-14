"""Per-object colour features for the iCAIRD cervix Cellpose batch.

Objects: every object's mean colour under its polygon, measured at level 1 (~0.5 µm/px).
Objects are grouped by the segmentation tile holding their centroid; each tile is read
once, the objects are rasterised into a label image, and the means come from bincounts.
Per slide, `OUT/objects/<slide_id>.npz` holds, in the order of the slide's sorted parquet
files and their rows:
    file_index, row index of the parquet file (sorted names) and the row in it;
    area_um2        polygon area;
    h, e            mean hematoxylin and eosin optical density (skimage `rgb2hed`);
    r, g, b         mean RGB;
    npx             pixels measured (0 for objects too small to cover a level-1 pixel);
    centroid        polygon centroid, level-0 px.
`OUT/slides.csv` holds per-slide summaries.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose \
        [--workers 14] [--slides ID ...]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openslide import OpenSlide
from skimage.color import rgb2hed

from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    MARGIN,
    SEG,
    SLIDES,
    STRIDE,
    areas_um2,
)


OUT = Path("/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/icaird_cervix_review/stain")
LEVEL = 1


def object_features(slide: OpenSlide, slide_id: str, mpp: float) -> dict[str, np.ndarray]:
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    polys, file_idx, rows = [], [], []
    for i, f in enumerate(files):
        col = pq.ParquetFile(f).read(columns=["polygon"]).column("polygon").combine_chunks()
        if isinstance(col, pa.ExtensionArray):  # Ray's tensor type, once ray is imported
            col = col.storage
        p = np.asarray(col.flatten()).reshape(-1, 64, 2)
        polys.append(p)
        file_idx.append(np.full(len(p), i, np.uint16))
        rows.append(np.arange(len(p), dtype=np.uint32))
    polys = np.concatenate(polys) if polys else np.zeros((0, 64, 2), np.float32)
    return {
        "file_index": np.concatenate(file_idx) if file_idx else np.zeros(0, np.uint16),
        "row": np.concatenate(rows) if rows else np.zeros(0, np.uint32),
        **polygon_features(slide, polys, mpp),
    }


def polygon_features(slide: OpenSlide, polys: np.ndarray, mpp: float) -> dict[str, np.ndarray]:
    """Colour features of `polys` (n, 64, 2), level-0 px; the keys of `object_features` but the file and row."""
    n = len(polys)
    sums = np.zeros((n, 5), np.float64)  # h, e, r, g, b
    npx = np.zeros(n, np.int64)

    scale = 1 / slide.level_downsamples[LEVEL]
    cells = np.floor((polys.mean(axis=1) - MARGIN) / STRIDE).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    bounds = np.flatnonzero(np.any(np.diff(cells[order], axis=0) != 0, axis=1)) + 1
    for group in np.split(order, bounds):
        if len(group) == 0:
            continue
        p = polys[group]
        x0, y0 = np.floor(p.min(axis=(0, 1))).astype(int) - 2
        x1, y1 = np.ceil(p.max(axis=(0, 1))).astype(int) + 2
        w, h = int((x1 - x0) * scale) + 1, int((y1 - y0) * scale) + 1
        img = np.asarray(slide.read_region((int(x0), int(y0)), LEVEL, (w, h)).convert("RGB"))
        labels = np.zeros((h, w), np.int32)
        local = [np.round((q - [x0, y0]) * scale).astype(np.int32) for q in p]
        for k, q in enumerate(local, 1):
            cv2.fillPoly(labels, [q], k)
        inside = labels > 0
        lab = labels[inside] - 1
        rgb = img[inside]
        hed = rgb2hed(rgb[:, None, :])[:, 0, :2]
        counts = np.bincount(lab, minlength=len(group))
        npx[group] = counts
        for c, values in enumerate((hed[:, 0], hed[:, 1], *rgb.T.astype(np.float64))):
            sums[group, c] = np.bincount(lab, weights=values, minlength=len(group))
    means = sums / np.maximum(npx, 1)[:, None]
    means[npx == 0] = np.nan
    return {
        "area_um2": areas_um2(polys, mpp).astype(np.float32),
        "h": means[:, 0].astype(np.float32),
        "e": means[:, 1].astype(np.float32),
        "r": means[:, 2].astype(np.float32),
        "g": means[:, 3].astype(np.float32),
        "b": means[:, 4].astype(np.float32),
        "npx": npx.astype(np.int32),
        "centroid": polys.mean(axis=1).astype(np.float32),
    }


def process(slide_id: str) -> dict:
    path = SLIDES / f"{slide_id}.tiff"
    with OpenSlide(str(path)) as slide:
        mpp = (float(slide.properties["openslide.mpp-x"]) + float(slide.properties["openslide.mpp-y"])) / 2
        feats = object_features(slide, slide_id, mpp)
    np.savez(OUT / "objects" / f"{slide_id}.npz", **feats)
    return {
        "slide_id": slide_id,
        "objects": len(feats["h"]),
        "h_med": float(np.nanmedian(feats["h"])) if len(feats["h"]) else np.nan,
        "unmeasured": int((feats["npx"] == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--slides", nargs="*")
    args = parser.parse_args()
    (OUT / "objects").mkdir(parents=True, exist_ok=True)

    slides = args.slides or sorted(p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*"))
    rows = []
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for i, row in enumerate(pool.imap_unordered(process, slides), 1):
            rows.append(row)
            if i % 100 == 0 or args.slides:
                print(f"{i}/{len(slides)} {row}", flush=True)
    table = pd.DataFrame(rows).sort_values("slide_id")
    csv = OUT / ("slides.csv" if not args.slides else "slides_subset.csv")
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
