"""Removes red blood cells and blood regions from the iCAIRD cervix Cellpose batch.

Per slide:
    1. every Cellpose object that is a red blood cell by colour is removed
       (`red_blood_cells` in blood_regions_icaird_cervix_cellpose.py);
    2. every other Cellpose object inside a blood region is removed too: there Cellpose
       mostly garbles what lies between the red blood cells (on IC-CX-01429-01, 61% of
       those objects match no LSP-DETR nucleus).
Blood regions are left without objects. The batch holds Cellpose objects only: the
first run of this clean-up (2026-11-13) also added LSP-DETR nuclei in the blood regions;
they were taken out the same day and archived in
_archive/lsp_detr_blood_regions_added (next to the batch).

The objects are matched to the colour features by file and row; before a file is
changed its row count and centroids are checked against the features, which were
measured after the area filter (since dropped: restore_large_objects_icaird_cervix_cellpose.py
put its objects back and ran this clean-up's rules on them). As in the area filter, the
removed rows (with a `reason` column) are saved first to REMOVED/slide_id=<slide>/<file>,
and each filtered file is written as a hidden temporary file, read back, checked and
moved over the original. A slide is listed in REMOVED/_done.txt once finished, with its
counts in REMOVED/_slides.csv; a rerun skips listed slides and redoes the others safely
(the counts of a slide interrupted midway then miss its already filtered files). A
summary is appended to the batch's `_blood_cleanup.json`.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.clean_blood_icaird_cervix_cellpose \
        [--workers 14] [--slides ID ...]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openslide import OpenSlide

from scripts.preprocessing.nuclei_segmentation.blood_regions_icaird_cervix_cellpose import (
    CELL_PX,
    blood_regions,
    in_regions,
    red_blood_cells,
)
from scripts.preprocessing.nuclei_segmentation.blood_regions_icaird_cervix_cellpose import (
    OUT as BLOOD,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


REMOVED = SEG.with_name("cervix_cellpose_removed_blood")
BLOOD_SLIDES = pd.read_csv(BLOOD / "slides.csv").set_index("slide_id")


def column(table: pa.Table, name: str, shape: tuple[int, ...]) -> np.ndarray:
    col = table.column(name).combine_chunks()
    if isinstance(col, pa.ExtensionArray):
        col = col.storage
    return np.asarray(col.flatten()).reshape(-1, *shape)


def regions_of(slide_id: str, feats: np.lib.npyio.NpzFile | dict[str, np.ndarray], rbc: np.ndarray) -> np.ndarray | None:
    """The slide's blood regions, recomputed from the colour features.

    They are computed exactly as in blood_regions_icaird_cervix_cellpose.py, not read
    from the mask TIFF: OpenSlide garbled small masks once.
    """
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        width, height = slide.level_dimensions[0]
    regions = blood_regions(feats["centroid"], rbc, (-(-height // CELL_PX), -(-width // CELL_PX)))
    return regions if regions.any() else None


def write_checked(table: pa.Table, path: Path, schema: pa.Schema) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="snappy")
    check = pq.ParquetFile(tmp)
    if check.metadata.num_rows != len(table) or not check.schema_arrow.equals(schema, check_metadata=True):
        tmp.unlink()
        raise RuntimeError(f"{tmp}: read-back check failed")
    os.replace(tmp, path)


def clean_slide(slide_id: str) -> dict:
    feats = np.load(STAIN / "objects" / f"{slide_id}.npz")
    rbc = red_blood_cells(feats)
    regions = regions_of(slide_id, feats, rbc)
    inside = in_regions(feats["centroid"], regions) if regions is not None else np.zeros(len(rbc), bool)
    drop = rbc | inside
    expected = BLOOD_SLIDES.loc[slide_id]
    if (inside & ~rbc).sum() != expected.other_in_regions or (inside & rbc).sum() != expected.rbc_in_regions:
        raise RuntimeError(f"{slide_id}: blood regions differ from {BLOOD / 'slides.csv'}")

    stats = {"slide_id": slide_id, "files_rewritten": 0, "removed_rbc": 0, "removed_in_regions": 0,
             "kept": 0}
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    for i, path in enumerate(files):
        sel = feats["file_index"] == i
        rows = feats["row"][sel]
        table = pq.ParquetFile(path).read()
        out = REMOVED / f"slide_id={slide_id}" / path.name
        if out.exists() and len(table) == len(rows) - int(drop[sel].sum()):
            # filtered by an interrupted earlier run
            stats["kept"] += len(table)
            continue
        if len(table) != len(rows) or not np.array_equal(rows, np.arange(len(rows))):
            raise RuntimeError(f"{path}: {len(table)} rows, colour features hold {len(rows)}")
        centroids = column(table, "centroid", (2,))
        if not np.allclose(centroids, feats["centroid"][sel], atol=0.5):
            raise RuntimeError(f"{path}: centroids differ from the colour features")
        d, r = drop[sel], rbc[sel]
        stats["kept"] += int((~d).sum())
        if not d.any():
            continue
        reason = np.where(r[d], "red_blood_cell", "in_blood_region")
        removed = table.filter(pa.array(d)).append_column("reason", pa.array(reason))
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(removed, out, compression="snappy")
        write_checked(table.filter(pa.array(~d)), path, table.schema)
        stats["files_rewritten"] += 1
        stats["removed_rbc"] += int(r.sum())
        stats["removed_in_regions"] += int((d & ~r).sum())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--slides", nargs="*")
    args = parser.parse_args()
    REMOVED.mkdir(parents=True, exist_ok=True)
    done_path = REMOVED / "_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    slides = args.slides or sorted(p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*"))
    todo = [s for s in slides if s not in done]
    rows = []
    slides_csv = REMOVED / "_slides.csv"
    new_csv = not slides_csv.exists()
    with (
        Pool(args.workers, maxtasksperchild=50) as pool,
        done_path.open("a") as done_file,
        slides_csv.open("a") as csv_file,
    ):
        for i, row in enumerate(pool.imap_unordered(clean_slide, todo, chunksize=2), 1):
            rows.append(row)
            if new_csv:
                csv_file.write(",".join(row) + "\n")
                new_csv = False
            csv_file.write(",".join(str(v) for v in row.values()) + "\n")
            csv_file.flush()
            done_file.write(f"{row['slide_id']}\n")
            done_file.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)

    summary_path = SEG / "_blood_cleanup.json"
    previous = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else []
    run = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slides": len(rows),
        "files_rewritten": sum(r["files_rewritten"] for r in rows),
        "removed_red_blood_cells": sum(r["removed_rbc"] for r in rows),
        "removed_in_blood_regions": sum(r["removed_in_regions"] for r in rows),
        "kept": sum(r["kept"] for r in rows),
        "removed_rows_saved_in": str(REMOVED),
        "blood_regions": str(BLOOD),
        "grid_um": CELL_PX * 0.25,
        "script": "scripts/preprocessing/nuclei_segmentation/clean_blood_icaird_cervix_cellpose.py",
    }
    summary_path.write_text(json.dumps({"runs": [*previous, run]}, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
