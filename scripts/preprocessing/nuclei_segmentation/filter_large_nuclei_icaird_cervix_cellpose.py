"""Removes objects larger than 250 µm² from the iCAIRD cervix Cellpose batch.

Above 250 µm² (equivalent diameter ~18 µm) about 9 in 10 Cellpose objects are not nuclei
but whole cells outlined along their membranes, spaces between membranes, clusters of
cells or debris (see area_analysis_icaird_cervix_cellpose.py; the cutoff was chosen on
12 Nov 2026 from crops of random objects per area range).

Each parquet file of the batch holding such objects is rewritten without them:
    1. the removed rows (id, polygon, centroid, with their area) are saved first to
       REMOVED/slide_id=<slide>/<file name>, so the change can be undone;
    2. the filtered table is written next to the original as a hidden `.<name>.tmp`
       (ignored by parquet readers), read back and checked, then moved over it.
Files without such objects are left untouched. A rerun skips files already filtered.
A summary is written to the batch as `_area_filter.json`.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.filter_large_nuclei_icaird_cervix_cellpose \
        [--workers 14]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openslide import OpenSlide

from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
    areas_um2,
)


MAX_AREA_UM2 = 250.0
REMOVED = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/nuclei_segmentation/iCAIRD/"
    "cervix_cellpose_removed_area_gt250um2"
)


def filter_slide(slide_id: str) -> dict:
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        mpp = (float(slide.properties["openslide.mpp-x"]) + float(slide.properties["openslide.mpp-y"])) / 2
    stats = {"slide_id": slide_id, "files": 0, "files_rewritten": 0, "kept": 0, "removed": 0}
    for path in sorted((SEG / f"slide_id={slide_id}").glob("*.parquet")):
        stats["files"] += 1
        table = pq.ParquetFile(path).read()
        polys = np.asarray(table.column("polygon").combine_chunks().flatten()).reshape(-1, 64, 2)
        area = areas_um2(polys, mpp)
        drop = area > MAX_AREA_UM2
        n_drop = int(drop.sum())
        stats["kept"] += len(table) - n_drop
        if n_drop == 0:
            continue

        removed = table.filter(pa.array(drop)).append_column(
            "area_um2", pa.array(area[drop].astype(np.float32))
        )
        out = REMOVED / f"slide_id={slide_id}" / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(removed, out, compression="snappy")

        kept = table.filter(pa.array(~drop))
        tmp = path.with_name(f".{path.name}.tmp")
        pq.write_table(kept, tmp, compression="snappy")
        check = pq.ParquetFile(tmp)
        if check.metadata.num_rows != len(kept) or not check.schema_arrow.equals(
            table.schema, check_metadata=True
        ):
            tmp.unlink()
            raise RuntimeError(f"{tmp}: read-back check failed")
        os.replace(tmp, path)
        stats["files_rewritten"] += 1
        stats["removed"] += n_drop
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    slides = sorted(p.name.removeprefix("slide_id=") for p in SEG.glob("slide_id=*"))
    rows = []
    with Pool(args.workers, maxtasksperchild=50) as pool:
        for i, row in enumerate(pool.imap_unordered(filter_slide, slides, chunksize=2), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"{i}/{len(slides)}: removed {sum(r['removed'] for r in rows):,}", flush=True)

    summary_path = SEG / "_area_filter.json"
    previous = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else []
    run = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_area_um2": MAX_AREA_UM2,
        "removed_rows_saved_in": str(REMOVED),
        "slides": len(rows),
        "files": sum(r["files"] for r in rows),
        "files_rewritten": sum(r["files_rewritten"] for r in rows),
        "removed": sum(r["removed"] for r in rows),
        "kept": sum(r["kept"] for r in rows),
        "script": "scripts/preprocessing/nuclei_segmentation/filter_large_nuclei_icaird_cervix_cellpose.py",
    }
    summary_path.write_text(json.dumps({"runs": [*previous, run]}, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
