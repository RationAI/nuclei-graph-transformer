"""Removes empty cells from the iCAIRD cervix Cellpose batch: outlines with no hematoxylin.

Cellpose-SAM sometimes outlines whole squamous cells, or other pale structures, instead of
nuclei. Where the outline holds no nucleus it can be removed by image processing alone,
with no second segmenter. An object of the batch (after the blood clean-up,
clean_blood_icaird_cervix_cellpose.py) is an empty cell when all three hold:
    1. area >= MIN_AREA_UM2;
    2. mean H OD < H_RATIO x the slide's typical nucleus, which is the median H OD of
       the slide's objects under REF_AREA_UM2 that are not red blood cells (all measured
       by stain_features_icaird_cervix_cellpose.py, before the blood clean-up);
    3. local hematoxylin contrast <= MAX_CONTRAST. The contrast is the mean H OD inside
       the outline minus that in a ring 1-3 µm around it (9 and 25 px dilations at about
       0.25 µm/px). The ring excludes glass (mean RGB > GLASS) and every other Cellpose
       object, including the red blood cells removed by the blood clean-up. H is the
       hematoxylin channel of skimage's rgb2hed on the level-0 pixels as stored (not
       white-balanced). Objects whose ring has fewer than MIN_RING_PX pixels are kept.

Calibration (800 random tiles, 2 from each of 400 slides, 703k objects; LSP-DETR used only
as a referee, never as a source of nuclei): the rule removed 388 objects (0.055%), of which
314 held no LSP-DETR nucleus, 64 were whole cells with a smaller LSP-DETR nucleus inside,
and 10 matched an LSP-DETR nucleus of about the same size. By eye, about 3 of those 10 are
real nuclei. The contrast threshold is sensitive: at 0.001 instead of 0 the rule removes
915 objects, and 231 of them match an LSP-DETR nucleus. Whole cells that do hold a nucleus
have positive contrast and are not touched by this rule.

As in clean_blood_icaird_cervix_cellpose.py, the objects are matched to the colour features
by file and row, and each file's row count and centroids are checked first. For each slide
the measurements of every candidate (conditions 1 and 2) are saved first to
REMOVED/_measures/<slide>.npz, and a rerun uses them instead of measuring again. The removed
rows (with `reason`, `rel_h`, `contrast` columns) are saved to REMOVED/slide_id=<slide>/<file>.
Each filtered file is written as a hidden temporary file and read back; its row count,
schema and ids are checked before it is moved over the original. A slide is listed in
REMOVED/_done.txt once finished, with its counts in REMOVED/_slides.csv, and a summary is
appended to the batch's `_empty_cell_cleanup.json`.

After this clean-up the stain features no longer match the batch row by row. The current
rows of a slide are the feature rows that are not red blood cells, not in a blood region,
and not `removed` in REMOVED/_measures/<slide>.npz (its `obj` indexes the feature rows).

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.clean_empty_cells_icaird_cervix_cellpose \
        [--workers 18] [--slides ID ...] [--dry-run DIR]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openslide import OpenSlide
from scipy.spatial import cKDTree
from skimage.color import rgb2hed

from scripts.preprocessing.nuclei_segmentation.blood_regions_icaird_cervix_cellpose import (
    in_regions,
    red_blood_cells,
)
from scripts.preprocessing.nuclei_segmentation.clean_blood_icaird_cervix_cellpose import (
    REMOVED as BLOOD_REMOVED,
)
from scripts.preprocessing.nuclei_segmentation.clean_blood_icaird_cervix_cellpose import (
    column,
    regions_of,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


REMOVED = SEG.with_name("cervix_cellpose_removed_empty_cells")
MIN_AREA_UM2, H_RATIO, MAX_CONTRAST = 15.0, 0.5, 0.0
REF_AREA_UM2 = 40.0
PAD, RING_IN, RING_OUT, GLASS, MIN_RING_PX = 20, 9, 25, 215, 30
BLOCK = 1024  # level-0 px; candidates are measured block by block, one slide read per block
CALIBRATION = {
    "sample": "800 random tiles (2048 px), 2 from each of 400 slides, 702,964 Cellpose objects",
    "removed": 388,
    "removed_without_lsp_detr_nucleus": 314,
    "removed_whole_cells_with_smaller_lsp_detr_nucleus": 64,
    "removed_matching_lsp_detr_nucleus": 10,
    "of_which_real_nuclei_by_eye": "about 3",
    "contrast_sensitivity": (
        "the contrast threshold is sharp: at 0 the rule removes 388 objects, 10 matching an LSP-DETR "
        "nucleus of about the same size; at 0.001 it removes 915, 231 matching; at 0.002 1,808, 707 matching"
    ),
}


def ring_contrast(
    slide: OpenSlide, polys: np.ndarray, objs: np.ndarray, occupied: np.ndarray
) -> np.ndarray:
    """[h_in, h_ring, ring_px] for each object in `objs` (indices into `polys`).

    `occupied` holds the polygons kept out of every ring: all Cellpose objects of the slide,
    with `polys` first, so that index k of `polys` is index k of `occupied`.
    """
    out = np.full((len(objs), 3), np.nan)
    if not len(objs):
        return out
    centres = occupied.mean(1)
    tree = cKDTree(centres)
    lo = np.floor(polys[objs].min(1)).astype(int) - PAD
    hi = np.ceil(polys[objs].max(1)).astype(int) + PAD
    blocks = np.floor(polys[objs].mean(1) / BLOCK).astype(int)
    _, group = np.unique(blocks, axis=0, return_inverse=True)
    for g in np.unique(group):
        members = np.flatnonzero(group == g)
        b_lo, b_hi = lo[members].min(0), hi[members].max(0)
        size = b_hi - b_lo
        im = np.asarray(slide.read_region(tuple(int(v) for v in b_lo), 0, tuple(int(v) for v in size)).convert("RGB"))
        hema = rgb2hed(im)[..., 0]
        glass = im.mean(2) > GLASS
        for n in members:
            k = objs[n]
            p = polys[k]
            (x0, y0), (x1, y1) = lo[n] - b_lo, hi[n] - b_lo
            w, h = hi[n] - lo[n]
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [(p - lo[n]).astype(np.int32)], 1)
            occ = np.zeros((h, w), np.uint8)
            near = np.asarray(tree.query_ball_point(p.mean(0), max(w, h), p=np.inf), int)
            near = near[np.all(np.abs(centres[near] - p.mean(0)) < max(w, h), 1) & (near != k)]
            if len(near):
                cv2.fillPoly(occ, [(occupied[j] - lo[n]).astype(np.int32) for j in near], 1)
            ring = (
                (cv2.dilate(m, np.ones((RING_OUT, RING_OUT), np.uint8)) > 0)
                & ~(cv2.dilate(m, np.ones((RING_IN, RING_IN), np.uint8)) > 0)
                & (occ == 0)
                & ~glass[y0:y1, x0:x1]
            )
            if ring.sum() < MIN_RING_PX:
                continue
            H = hema[y0:y1, x0:x1]
            out[n] = [H[m > 0].mean(), H[ring].mean(), ring.sum()]
    return out


def write_checked(table: pa.Table, path: Path) -> None:
    """Write via a hidden temporary file; check rows, schema and ids before replacing `path`."""
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="snappy")
    back = pq.ParquetFile(tmp)
    ok = back.metadata.num_rows == len(table) and back.schema_arrow.equals(table.schema, check_metadata=True)
    ok = ok and back.read(columns=["id"]).column("id").equals(table.column("id"))
    if not ok:
        tmp.unlink()
        raise RuntimeError(f"{tmp}: read-back check failed")
    os.replace(tmp, path)


def measure(slide_id: str, feats: np.lib.npyio.NpzFile, current: np.ndarray, files: list[Path]) -> dict:
    """Candidates and their contrast; checks the batch files against the colour features."""
    rbc = red_blood_cells(feats)
    ref = (feats["npx"] > 0) & (feats["area_um2"] < REF_AREA_UM2) & ~rbc
    h_ref = float(np.median(feats["h"][ref]))
    obj = np.flatnonzero(current)
    polys = []
    for i, path in enumerate(files):
        table = pq.ParquetFile(path).read(columns=["polygon", "centroid"])
        sel = feats["file_index"][obj] == i
        if len(table) != sel.sum():
            raise RuntimeError(f"{path}: {len(table)} rows, colour features hold {sel.sum()} after the blood clean-up")
        if not np.allclose(column(table, "centroid", (2,)), feats["centroid"][obj[sel]], atol=0.5):
            raise RuntimeError(f"{path}: centroids differ from the colour features")
        polys.append(column(table, "polygon", (64, 2)))
    if len(np.unique(feats["file_index"])) > len(files):
        raise RuntimeError(f"{slide_id}: colour features refer to more files than the batch holds")
    polys = np.concatenate(polys).astype(np.float32)
    order = np.concatenate([obj[feats["file_index"][obj] == i] for i in range(len(files))])
    polys_by_obj = np.empty_like(polys)
    polys_by_obj[np.searchsorted(obj, order)] = polys

    blood = sorted((BLOOD_REMOVED / f"slide_id={slide_id}").glob("*.parquet"))
    gone = [column(pq.ParquetFile(p).read(columns=["polygon"]), "polygon", (64, 2)) for p in blood]
    occupied = np.concatenate([polys_by_obj, *gone]).astype(np.float32)

    rel_h = feats["h"][obj] / h_ref
    cand = (feats["npx"][obj] > 0) & (feats["area_um2"][obj] >= MIN_AREA_UM2) & (rel_h < H_RATIO)
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        m = ring_contrast(slide, polys_by_obj, np.flatnonzero(cand), occupied)
    contrast = m[:, 0] - m[:, 1]
    return {
        "obj": obj[cand],
        "area_um2": feats["area_um2"][obj[cand]],
        "rel_h": rel_h[cand],
        "h_in": m[:, 0],
        "h_ring": m[:, 1],
        "ring_px": m[:, 2],
        "contrast": contrast,
        "removed": contrast <= MAX_CONTRAST,  # NaN (ring too small) compares False: kept
        "h_ref": np.float64(h_ref),
    }


def clean_slide(slide_id: str, dry_run: Path | None = None) -> dict:
    stats = {"slide_id": slide_id, "objects": 0, "candidates": 0, "no_ring": 0, "removed": 0, "files_rewritten": 0}
    path = STAIN / "objects" / f"{slide_id}.npz"
    if not path.exists():
        return stats
    feats = np.load(path)
    rbc = red_blood_cells(feats)
    regions = regions_of(slide_id, feats, rbc)
    current = ~(rbc | (in_regions(feats["centroid"], regions) if regions is not None else False))
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    stats["objects"] = int(current.sum())

    store = (dry_run or REMOVED / "_measures") / f"{slide_id}.npz"
    if store.exists():
        meas = dict(np.load(store))
    else:
        meas = measure(slide_id, feats, current, files)
        tmp = store.with_name(f".{store.stem}.tmp.npz")
        store.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, **meas)
        os.replace(tmp, store)
    stats["candidates"] = len(meas["obj"])
    stats["no_ring"] = int(np.isnan(meas["contrast"]).sum())
    stats["removed"] = int(meas["removed"].sum())
    if dry_run is not None:
        return stats

    drop = np.zeros(len(current), bool)
    drop[meas["obj"][meas["removed"]]] = True
    by_obj = {int(k): j for j, k in enumerate(meas["obj"])}
    for i, path in enumerate(files):
        obj = np.flatnonzero(current & (feats["file_index"] == i))
        d = drop[obj]
        if not d.any():
            continue
        table = pq.ParquetFile(path).read()
        out = REMOVED / f"slide_id={slide_id}" / path.name
        if out.exists() and len(table) == len(obj) - d.sum():
            continue  # filtered by an interrupted earlier run
        if len(table) != len(obj):
            raise RuntimeError(f"{path}: {len(table)} rows, expected {len(obj)}")
        if not np.allclose(column(table, "centroid", (2,)), feats["centroid"][obj], atol=0.5):
            raise RuntimeError(f"{path}: centroids differ from the colour features")
        j = np.array([by_obj[int(k)] for k in obj[d]])
        removed = (
            table.filter(pa.array(d))
            .append_column("reason", pa.array(["empty_cell"] * int(d.sum())))
            .append_column("rel_h", pa.array(meas["rel_h"][j]))
            .append_column("contrast", pa.array(meas["contrast"][j]))
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        write_checked(removed, out)
        write_checked(table.filter(pa.array(~d)), path)
        stats["files_rewritten"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--slides", nargs="*")
    parser.add_argument("--dry-run", type=Path, help="only measure, saving to this directory; the batch is not changed")
    args = parser.parse_args()
    slides = args.slides or sorted(p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*"))
    if args.dry_run:
        args.dry_run.mkdir(parents=True, exist_ok=True)
        with Pool(args.workers, maxtasksperchild=20) as pool:
            rows = pool.starmap(clean_slide, [(s, args.dry_run) for s in slides], chunksize=1)
        print(pd.DataFrame(rows).sum(numeric_only=True).to_string())
        return

    REMOVED.mkdir(parents=True, exist_ok=True)
    done_path = REMOVED / "_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [s for s in slides if s not in done]
    slides_csv = REMOVED / "_slides.csv"
    new_csv = not slides_csv.exists()
    rows = []
    start = time.time()
    with (
        Pool(args.workers, maxtasksperchild=20) as pool,
        done_path.open("a") as done_file,
        slides_csv.open("a") as csv_file,
    ):
        for i, row in enumerate(pool.imap_unordered(clean_slide, todo, chunksize=1), 1):
            rows.append(row)
            if new_csv:
                csv_file.write(",".join(row) + "\n")
                new_csv = False
            csv_file.write(",".join(str(v) for v in row.values()) + "\n")
            csv_file.flush()
            done_file.write(f"{row['slide_id']}\n")
            done_file.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)} slides, {(time.time() - start) / 3600:.1f} h", flush=True)

    summary_path = SEG / "_empty_cell_cleanup.json"
    previous = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else []
    run = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slides": len(rows),
        "objects_before": sum(r["objects"] for r in rows),
        "candidates": sum(r["candidates"] for r in rows),
        "candidates_without_ring": sum(r["no_ring"] for r in rows),
        "removed_empty_cells": sum(r["removed"] for r in rows),
        "files_rewritten": sum(r["files_rewritten"] for r in rows),
        "rule": {
            "min_area_um2": MIN_AREA_UM2,
            "max_h_relative_to_typical_nucleus": H_RATIO,
            "typical_nucleus": f"median H OD of non-RBC objects under {REF_AREA_UM2} µm²",
            "max_contrast": MAX_CONTRAST,
            "ring": f"{RING_IN}-{RING_OUT} px dilations (1-3 µm), glass (mean RGB > {GLASS}) and other objects excluded",
            "min_ring_px": MIN_RING_PX,
        },
        "calibration": CALIBRATION,
        "removed_rows_saved_in": str(REMOVED),
        "measures_saved_in": str(REMOVED / "_measures"),
        "script": "scripts/preprocessing/nuclei_segmentation/clean_empty_cells_icaird_cervix_cellpose.py",
    }
    summary_path.write_text(json.dumps({"runs": [*previous, run]}, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
