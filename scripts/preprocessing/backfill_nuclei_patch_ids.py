"""Backfills `ids.npy` into nuclei patch folders extracted before it existed, without re-extracting.

`preprocessing/nuclei_patches.py` now saves, next to `patches.npy` and `rows.npy`, the
nuclei ids in `get_nuclei` order (sorted by `id`), and the datasets check them against
the slide's current nuclei. Folders written earlier lack them. For each such folder this
script writes the sorted ids of the slide's nuclei (`<nuclei>/slide_id=<id>/`), but only
when both hold:
    1. every nucleus file of the slide is older than the patch folder's oldest file, so
       the patches were extracted from these nuclei and not from an earlier version;
    2. the slide has exactly as many nuclei as `rows.npy` has rows.
Folders that fail either check, or have no nuclei, are listed in `<patches>/_ids_backfill_failed.csv`
with the reason and left unchanged; they must be re-extracted. Folders that already have
`ids.npy` are skipped. Each file is written as a hidden temporary file, read back and
compared, then moved into place. A summary goes to `<patches>/_ids_backfill.json`.

Run it where the patch folders are writable: /mnt/public_projects (the PANDA patches) is
mounted read-only on the sensitive cluster, so there it only works with --dry-run.

Usage:
    uv run python -m scripts.preprocessing.backfill_nuclei_patch_ids \
        --patches /mnt/public_projects/nuclei_based_wsi_analysis/nuclei_patches/PANDA_45deg_18um_40px \
        --nuclei /mnt/public_projects/nuclei_based_wsi_analysis/nuclei_segmentation/PANDA \
        [--workers 16] [--dry-run]
"""

import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nuclei_graph.data.datasets.nuclei_features import (
    IDS_FILE,
    PATCHES_FILE,
    ROWS_FILE,
)


def backfill(slide_dir: Path, nuclei_root: Path, dry_run: bool) -> dict[str, str]:
    slide = slide_dir.name
    row = {"slide": slide, "status": "", "reason": ""}
    if (slide_dir / IDS_FILE).exists():
        return row | {"status": "skipped", "reason": "ids.npy exists"}
    if not (slide_dir / ROWS_FILE).exists():
        return row | {
            "status": "failed",
            "reason": "no rows.npy (extraction unfinished)",
        }
    nuclei_files = sorted((nuclei_root / slide).glob("*.parquet"))
    if not nuclei_files:
        return row | {
            "status": "failed",
            "reason": f"no nuclei in {nuclei_root / slide}",
        }

    patch_files = [slide_dir / PATCHES_FILE, slide_dir / ROWS_FILE]
    oldest_patch = min(p.stat().st_mtime for p in patch_files if p.exists())
    newest_nuclei = max(p.stat().st_mtime for p in nuclei_files)
    if newest_nuclei >= oldest_patch:
        return row | {"status": "failed", "reason": "nuclei newer than the patches"}

    ids = np.sort(
        np.concatenate(
            [
                pq.read_table(p, columns=["id"]).column("id").to_numpy()
                for p in nuclei_files
            ]
        ).astype(str)
    )
    n_rows = len(np.load(slide_dir / ROWS_FILE, mmap_mode="r"))
    if len(ids) != n_rows:
        return row | {
            "status": "failed",
            "reason": f"{len(ids)} nuclei, rows.npy has {n_rows}",
        }
    if len(np.unique(ids)) != len(ids):
        return row | {"status": "failed", "reason": "duplicate nuclei ids"}
    if dry_run:
        return row | {"status": "would_write"}

    tmp = slide_dir / f".{IDS_FILE}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, ids)
    if not np.array_equal(np.load(tmp), ids):
        tmp.unlink()
        return row | {"status": "failed", "reason": "read-back check failed"}
    os.replace(tmp, slide_dir / IDS_FILE)
    return row | {"status": "written"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--nuclei", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slide_dirs = sorted(p for p in args.patches.glob("slide_id=*") if p.is_dir())
    with Pool(args.workers) as pool:
        rows = pool.map(
            partial(backfill, nuclei_root=args.nuclei, dry_run=args.dry_run),
            slide_dirs,
            chunksize=8,
        )
    table = pd.DataFrame(rows)
    counts = table.status.value_counts().to_dict()
    print(json.dumps(counts, indent=1))
    failed = table[table.status == "failed"]
    if len(failed):
        print(failed.to_string(index=False))
    if args.dry_run:
        return
    failed.to_csv(args.patches / "_ids_backfill_failed.csv", index=False)
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patches": str(args.patches),
        "nuclei": str(args.nuclei),
        "folders": len(table),
        **{k: int(v) for k, v in counts.items()},
        "rule": "nuclei files older than patches.npy and rows.npy; nuclei count == rows.npy rows",
        "failed_listed_in": str(args.patches / "_ids_backfill_failed.csv"),
        "script": "scripts/preprocessing/backfill_nuclei_patch_ids.py",
    }
    path = args.patches / "_ids_backfill.json"
    previous = json.loads(path.read_text())["runs"] if path.exists() else []
    path.write_text(json.dumps({"runs": [*previous, summary]}, indent=1))


if __name__ == "__main__":
    main()
