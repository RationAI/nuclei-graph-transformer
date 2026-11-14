"""Stain features of the cleaned iCAIRD cervix Cellpose batch, row for row and with nucleus ids.

stain_features_icaird_cervix_cellpose.py measured every object of the batch as it was after
the (since dropped) area filter. The blood clean-up (clean_blood_icaird_cervix_cellpose.py)
and the empty-cell clean-up (clean_empty_cells_icaird_cervix_cellpose.py) then removed rows,
and restore_large_objects_icaird_cervix_cellpose.py appended the restored objects over
250 µm² to the end of their files, so those features no longer match the batch row by row. This script writes, per slide, the features of
exactly the objects left in the batch, in batch order: the slide's parquet files sorted
by name, rows in file order. That is the order of `pd.read_parquet` on the slide directory
before any sorting. Each file also holds the objects' `id`, so consumers can join on it.

The features are `all_features` of the restore script: the original rows, then those of the
restored objects. Kept rows: not a red blood cell, not in a blood region (the regions of the
blood clean-up), not `removed` in the empty-cell clean-up's _measures/<slide>.npz, and, for
restored objects, not removed again by the restore (`removed_after_restore`). For every batch file the row count and
centroids (to 0.5 px) are checked against the kept features, and the ids are taken from the
file itself.

Writes OUT/<slide_id>.npz with the original keys (area_um2, h, e, r, g, b, npx, centroid),
plus `id` (nucleus id), `file_index` and `row` (position in the current batch files), `obj`
(row in `all_features`) and `restored` (an object over 250 µm² put back by the restore). Each file is written as a hidden temporary
file, read back and checked, then moved into place. Slides with no stain features (no
nuclei) are skipped. A summary goes to OUT/_summary.json.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.align_stain_features_icaird_cervix_cellpose \
        [--workers 18] [--slides ID ...]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from scripts.preprocessing.nuclei_segmentation.blood_regions_icaird_cervix_cellpose import (
    in_regions,
    red_blood_cells,
)
from scripts.preprocessing.nuclei_segmentation.clean_blood_icaird_cervix_cellpose import (
    column,
    regions_of,
)
from scripts.preprocessing.nuclei_segmentation.clean_empty_cells_icaird_cervix_cellpose import (
    REMOVED as EMPTY_REMOVED,
)
from scripts.preprocessing.nuclei_segmentation.restore_large_objects_icaird_cervix_cellpose import (
    all_features,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


OUT = STAIN / "objects_cleaned"
FEATURES = ["area_um2", "h", "e", "r", "g", "b", "npx", "centroid"]


def current_objects(slide_id: str, feats: dict[str, np.ndarray]) -> np.ndarray:
    """Mask over the rows of `all_features` of the objects left in the batch."""
    rbc = red_blood_cells(feats)
    original = ~feats["restored"]
    regions = regions_of(slide_id, {"centroid": feats["centroid"][original]}, rbc[original])
    keep = ~(rbc | (in_regions(feats["centroid"], regions) if regions is not None else False))
    measures = np.load(EMPTY_REMOVED / "_measures" / f"{slide_id}.npz")
    keep[measures["obj"][measures["removed"]]] = False
    keep[feats["removed_after_restore"]] = False
    return keep


def align(slide_id: str) -> dict:
    path = STAIN / "objects" / f"{slide_id}.npz"
    if not path.exists():
        return {"slide_id": slide_id, "objects": 0, "skipped": True}
    feats = all_features(slide_id)
    keep = current_objects(slide_id, feats)
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    if len(np.unique(feats["file_index"])) > len(files):
        raise RuntimeError(f"{slide_id}: colour features refer to more files than the batch holds")

    objs, ids, file_index, rows = [], [], [], []
    for i, file in enumerate(files):
        obj = np.flatnonzero(keep & (feats["file_index"] == i))
        table = pq.ParquetFile(file).read(columns=["id", "centroid"])
        if len(table) != len(obj):
            raise RuntimeError(f"{file}: {len(table)} rows, the kept features hold {len(obj)}")
        if not np.allclose(column(table, "centroid", (2,)), feats["centroid"][obj], atol=0.5):
            raise RuntimeError(f"{file}: centroids differ from the kept features")
        objs.append(obj)
        ids.append(np.asarray(table.column("id").to_pylist(), dtype=str))
        file_index.append(np.full(len(obj), i, np.int32))
        rows.append(np.arange(len(obj), dtype=np.int64))
    obj = np.concatenate(objs) if objs else np.zeros(0, np.int64)
    out = {k: feats[k][obj] for k in FEATURES}
    out |= {
        "id": np.concatenate(ids) if ids else np.zeros(0, str),
        "file_index": np.concatenate(file_index) if file_index else np.zeros(0, np.int32),
        "row": np.concatenate(rows) if rows else np.zeros(0, np.int64),
        "obj": obj,
        "restored": feats["restored"][obj],
    }
    if len(np.unique(out["id"])) != len(obj):
        raise RuntimeError(f"{slide_id}: nucleus ids are not unique")

    tmp = OUT / f".{slide_id}.tmp.npz"
    np.savez(tmp, **out)
    back = np.load(tmp)
    if set(back.files) != set(out) or not all(np.array_equal(back[k], out[k], equal_nan=k != "id") for k in out):
        tmp.unlink()
        raise RuntimeError(f"{tmp}: read-back check failed")
    os.replace(tmp, OUT / f"{slide_id}.npz")
    return {"slide_id": slide_id, "objects": len(obj), "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--slides", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = args.slides or sorted(p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*"))
    with Pool(args.workers, maxtasksperchild=50) as pool:
        rows = pool.map(align, slides, chunksize=4)
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slides": len(rows),
        "slides_skipped": [r["slide_id"] for r in rows if r["skipped"]],
        "objects": sum(r["objects"] for r in rows),
        "order": "batch order: parquet files sorted by name, rows in file order",
        "script": "scripts/preprocessing/nuclei_segmentation/align_stain_features_icaird_cervix_cellpose.py",
    }
    if not args.slides:
        (OUT / "_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
