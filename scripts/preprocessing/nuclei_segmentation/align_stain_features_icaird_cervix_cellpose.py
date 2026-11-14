"""Stain features of the cleaned iCAIRD cervix Cellpose batch, row for row and with nucleus ids.

The stain features of every original Cellpose object are in STAIN/objects_original (written by
reconstruct_original_icaird_cervix_cellpose.py, in the order of the original files, with ids).
The clean-up (clean_icaird_cervix_cellpose.py) removes rows but keeps the order, so the
features of the batch are those rows, picked by id. This script writes them per slide in
batch order: the slide's parquet files sorted by name, rows in file order. That is the order
of `pd.read_parquet` on the slide directory before any sorting.

Checks per batch file: every id is an original object of the same file, the ids keep the
original order, are unique, and the centroids match the features (to 0.5 px).

Writes OUT/<slide_id>.npz with area_um2, h, e, r, g, b, npx, centroid, plus `id`,
`file_index` and `row` (position in the batch files), `obj` (row in objects_original) and
`appended` (an object over 250 µm², whose position in its file is not the original one).
Each file is written as a hidden temporary file, read back and checked, then moved into
place. A summary goes to OUT/_summary.json.

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
import pandas as pd
import pyarrow.parquet as pq

from scripts.preprocessing.nuclei_segmentation.clean_icaird_cervix_cellpose import (
    FEATURES,
    SEG,
    column,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


OUT = STAIN / "objects_cleaned"
KEYS = ["area_um2", "h", "e", "r", "g", "b", "npx", "centroid"]


def align(slide_id: str) -> dict:
    f = np.load(FEATURES / f"{slide_id}.npz")
    index = pd.Index(f["id"])
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    objs, ids, file_index, rows = [], [], [], []
    for i, file in enumerate(files):
        table = pq.ParquetFile(file).read(columns=["id", "centroid"])
        file_ids = np.asarray(table.column("id").to_pylist(), dtype=str)
        obj = index.get_indexer(file_ids)
        if (obj < 0).any() or (len(obj) and not (f["file_index"][obj] == i).all()):
            raise RuntimeError(
                f"{file}: ids that are not original objects of this file"
            )
        if np.any(np.diff(obj) <= 0):
            raise RuntimeError(f"{file}: rows are not in the original order")
        if not np.allclose(
            column(table, "centroid", (2,)), f["centroid"][obj], atol=0.5
        ):
            raise RuntimeError(f"{file}: centroids differ from the features")
        objs.append(obj)
        ids.append(file_ids)
        file_index.append(np.full(len(obj), i, np.int32))
        rows.append(np.arange(len(obj), dtype=np.int64))
    obj = np.concatenate(objs) if objs else np.zeros(0, np.int64)
    out = {k: f[k][obj] for k in KEYS}
    out |= {
        "id": np.concatenate(ids) if ids else np.zeros(0, str),
        "file_index": np.concatenate(file_index)
        if file_index
        else np.zeros(0, np.int32),
        "row": np.concatenate(rows) if rows else np.zeros(0, np.int64),
        "obj": obj,
        "appended": f["appended"][obj],
    }
    if len(np.unique(out["id"])) != len(obj):
        raise RuntimeError(f"{slide_id}: nucleus ids are not unique")

    tmp = OUT / f".{slide_id}.tmp.npz"
    np.savez(tmp, **out)
    back = np.load(tmp)
    if set(back.files) != set(out) or not all(
        np.array_equal(back[k], out[k], equal_nan=k != "id") for k in out
    ):
        tmp.unlink()
        raise RuntimeError(f"{tmp}: read-back check failed")
    os.replace(tmp, OUT / f"{slide_id}.npz")
    return {"slide_id": slide_id, "objects": len(obj)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--slides", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = args.slides or sorted(
        p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*")
    )
    with Pool(args.workers, maxtasksperchild=50) as pool:
        rows = pool.map(align, slides, chunksize=4)
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slides": len(rows),
        "objects": sum(r["objects"] for r in rows),
        "order": "batch order: parquet files sorted by name, rows in file order",
        "features": str(FEATURES),
        "script": "scripts/preprocessing/nuclei_segmentation/align_stain_features_icaird_cervix_cellpose.py",
    }
    if not args.slides:
        (OUT / "_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
