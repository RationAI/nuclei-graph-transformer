"""Rebuilds the original iCAIRD cervix Cellpose segmentation from the batch and its removed rows.

The Cellpose output (SEG) was cleaned in place, step by step (2026-11-12 to 11-14): the area
filter (since dropped), the blood clean-up, the empty-cell clean-up and the restore of the
objects over 250 µm². Every removed row was saved, so the original is the batch plus:
    _archive/cervix_cellpose_removed_area_gt250um2   (the restored objects, as removed),
    cervix_cellpose_removed_blood,
    cervix_cellpose_removed_empty_cells,
    cervix_cellpose_removed_restored_large            (restored, then removed again).
No untouched copy was kept, so this script rebuilt it once (2026-11-14) and wrote it to
ORIGINAL, from which clean_icaird_cervix_cellpose.py runs the whole clean-up again. Run 1 (the
batch and its removed rows) was then moved to _archive/cleanup_run1; the paths below point
there, so the rebuild can be repeated.

Order. The stain features (stain_features_icaird_cervix_cellpose.py, STAIN/objects) were
measured on the batch after the area filter, so they keep the original Cellpose order of
every other object. The objects over 250 µm² have no recorded position (the ids are
content hashes); they are appended to the end of the file they came from, in the order of
the area filter's removed rows. Each original file is thus: the feature rows of the file,
then its objects over 250 µm².

Ids. Each feature row's id is found where its step left it, by position: kept rows from
objects_cleaned (`obj`, `id`); blood-removed rows are the rows the blood rule drops, in
order; empty-cell rows those `removed` in the empty-cell _measures, in order; the objects over
250 µm² carry their ids in objects_restored. Checks: every feature row gets exactly one id,
the ids are those of the sources and unique, every row's centroid matches its features (to
0.5 px), the objects over 250 µm² match the area filter's rows exactly, and the totals add up.

Writes
    ORIGINAL/slide_id=<slide>/<file>.parquet   id, polygon, centroid (the batch schema); each
                                               file written as a hidden temporary file,
                                               read back and checked;
    STAIN/objects_original/<slide>.npz         the stain features in that order (area_um2,
                                               h, e, r, g, b, npx, centroid), with id,
                                               file_index, row and `appended` (an object
                                               over 250 µm², position not original);
    ORIGINAL/_reconstruction.json              the counts and checks.
A slide is listed in ORIGINAL/_done.txt when finished; a rerun skips it.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.reconstruct_original_icaird_cervix_cellpose \
        [--workers 16] [--slides ID ...]
"""

import argparse
import json
import os
import shutil
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
from scripts.preprocessing.nuclei_segmentation.clean_icaird_cervix_cellpose import (
    FEATURES as FEATURES_OUT,
)
from scripts.preprocessing.nuclei_segmentation.clean_icaird_cervix_cellpose import (
    ORIGINAL,
    column,
    write_checked,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


# Run 1 was moved to the archive after this script ran (2026-11-14); the paths point there.
RUN1 = SEG.parent / "_archive" / "cleanup_run1"
BATCH = RUN1 / "cervix_cellpose"
BLOOD_REMOVED = RUN1 / "cervix_cellpose_removed_blood"
EMPTY_REMOVED = RUN1 / "cervix_cellpose_removed_empty_cells"
RESTORE_REMOVED = RUN1 / "cervix_cellpose_removed_restored_large"
ALIGNED = STAIN / "_archive_run1" / "objects_cleaned"
SOURCE = SEG.parent / "_archive" / "cervix_cellpose_removed_area_gt250um2"
RESTORED = STAIN / "objects_restored"
FEATURES = ["area_um2", "h", "e", "r", "g", "b", "npx", "centroid"]
BATCH_COLUMNS = ["id", "polygon", "centroid"]
SOURCES = {
    "batch": BATCH,
    "blood": BLOOD_REMOVED,
    "empty": EMPTY_REMOVED,
    "restored_large": RESTORE_REMOVED,
}


def all_features(slide_id: str) -> dict[str, np.ndarray]:
    """The stain features measured after the area filter, then those of the objects over 250 µm²."""
    orig = np.load(STAIN / "objects" / f"{slide_id}.npz")
    out = {k: orig[k] for k in [*FEATURES, "file_index"]}
    out["restored"] = np.zeros(len(out["h"]), bool)
    path = RESTORED / f"{slide_id}.npz"
    if path.exists():
        r = np.load(path)
        for k in [*FEATURES, "file_index"]:
            out[k] = np.concatenate([out[k], r[k].astype(out[k].dtype)])
        out["restored"] = np.concatenate([out["restored"], np.ones(len(r["h"]), bool)])
    return out


def regions_of(
    slide_id: str, feats: dict[str, np.ndarray], rbc: np.ndarray
) -> np.ndarray | None:
    """The blood regions of run 1 (computed from the features measured after the area filter)."""
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        width, height = slide.level_dimensions[0]
    regions = blood_regions(
        feats["centroid"], rbc, (-(-height // CELL_PX), -(-width // CELL_PX))
    )
    return regions if regions.any() else None


def read(directory: Path, slide_id: str, schema: pa.Schema) -> dict[str, pa.Table]:
    """The rows of `directory` per file name, cut to the batch columns and cast to the batch schema."""
    out = {}
    for path in sorted((directory / f"slide_id={slide_id}").glob("*.parquet")):
        table = pq.ParquetFile(path).read(columns=BATCH_COLUMNS)
        if not table.schema.equals(schema, check_metadata=False):
            raise RuntimeError(f"{path}: schema differs from the batch")
        out[path.name] = table.cast(schema)
    return out


def ids_by_position(
    tables: dict[str, pa.Table],
    names: list[str],
    file_index: np.ndarray,
    mask: np.ndarray,
    centroids: np.ndarray,
) -> dict[int, str]:
    """Feature row -> id, pairing the rows of `mask` in each file with that file's rows in order."""
    out = {}
    for name, table in tables.items():
        rows = np.flatnonzero(mask & (file_index == names.index(name)))
        if len(table) != len(rows):
            raise RuntimeError(
                f"{name}: {len(table)} rows, the rule selects {len(rows)}"
            )
        if not np.allclose(column(table, "centroid", (2,)), centroids[rows], atol=0.5):
            raise RuntimeError(f"{name}: centroids differ from the features")
        out |= dict(zip(rows.tolist(), table.column("id").to_pylist(), strict=True))
    if len(out) != mask.sum():
        raise RuntimeError(
            f"{mask.sum()} rows selected, {len(out)} found in the removed files"
        )
    return out


def reconstruct(slide_id: str) -> dict:
    files = sorted((BATCH / f"slide_id={slide_id}").glob("*.parquet"))
    names = [f.name for f in files]
    stats = {"slide_id": slide_id, "objects": 0, "appended": 0, "files": len(files)}
    stats |= dict.fromkeys(SOURCES, 0)
    if not files:
        return stats
    schema = pq.ParquetFile(files[0]).schema_arrow
    tables = {k: read(d, slide_id, schema) for k, d in SOURCES.items()}
    for k, t in tables.items():
        stats[k] = sum(len(v) for v in t.values())
        if any(n not in names for n in t):
            raise RuntimeError(
                f"{slide_id}: {SOURCES[k]} holds a file the batch does not"
            )

    feats = all_features(slide_id)
    n_orig = int((~feats["restored"]).sum())
    orig = {
        k: feats[k][:n_orig]
        for k in ("file_index", "centroid", "npx", "r", "g", "b", "h")
    }
    ids = np.full(len(feats["h"]), None, object)

    # kept rows: objects_cleaned
    aligned = np.load(ALIGNED / f"{slide_id}.npz")
    ids[aligned["obj"]] = aligned["id"]
    kept = np.zeros(len(ids), bool)
    kept[aligned["obj"]] = True
    # blood: the rows the blood rule drops, in order
    rbc = red_blood_cells(orig)
    regions = regions_of(slide_id, orig, rbc)
    blood = rbc | (
        in_regions(orig["centroid"], regions) if regions is not None else False
    )
    by_pos = ids_by_position(
        tables["blood"], names, orig["file_index"], blood, orig["centroid"]
    )
    # empty cells: the rows `removed` in the measures, in order
    meas = np.load(EMPTY_REMOVED / "_measures" / f"{slide_id}.npz")
    empty = np.zeros(n_orig, bool)
    empty[meas["obj"][meas["removed"]]] = True
    by_pos_e = ids_by_position(
        tables["empty"], names, orig["file_index"], empty, orig["centroid"]
    )
    if (blood & empty).any() or (kept[:n_orig] & (blood | empty)).any():
        raise RuntimeError(f"{slide_id}: a feature row is assigned twice")
    for row, i in (by_pos | by_pos_e).items():
        ids[row] = i
    # objects over 250 µm²: ids saved by the restore
    if n_orig < len(ids):
        ids[n_orig:] = np.load(RESTORED / f"{slide_id}.npz")["id"]
    if any(i is None for i in ids):
        raise RuntimeError(
            f"{slide_id}: {sum(i is None for i in ids)} feature rows have no id"
        )

    pool = pa.concat_tables([t for k in SOURCES for t in tables[k].values()])
    index = pd.Index(pool.column("id").to_pylist())
    if not index.is_unique or len(index) != len(ids):
        raise RuntimeError(
            f"{slide_id}: {len(index)} source rows, {len(ids)} feature rows, unique {index.is_unique}"
        )
    take = index.get_indexer(ids.astype(str))
    if (take < 0).any() or len(np.unique(take)) != len(take):
        raise RuntimeError(
            f"{slide_id}: feature ids do not match the source rows one to one"
        )
    table = pool.take(pa.array(take))
    if not np.allclose(column(table, "centroid", (2,)), feats["centroid"], atol=0.5):
        raise RuntimeError(f"{slide_id}: centroids differ from the features")

    # objects over 250 µm² equal the area filter's rows (polygons exactly)
    source = read(SOURCE, slide_id, schema)
    if source:
        src = pa.concat_tables(source.values())
        appended = table.slice(n_orig)
        if not (
            src.column("id").equals(appended.column("id"))
            and np.array_equal(
                column(src, "polygon", (64, 2)), column(appended, "polygon", (64, 2))
            )
        ):
            raise RuntimeError(f"{slide_id}: objects over 250 µm² differ from {SOURCE}")

    # per file: its feature rows, then its objects over 250 µm²
    # all_features holds the feature rows, then the objects over 250 µm²: group them by file
    order = np.lexsort((np.arange(len(ids)), feats["restored"], feats["file_index"]))
    table = table.take(pa.array(order))
    file_index = feats["file_index"][order].astype(np.int32)
    out_dir = ORIGINAL / f"slide_id={slide_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = np.zeros(len(order), np.int64)
    for i, name in enumerate(names):
        sel = np.flatnonzero(file_index == i)
        rows[sel] = np.arange(len(sel))
        write_checked(table.take(pa.array(sel)), out_dir / name)

    out = {k: feats[k][order] for k in FEATURES}
    out |= {
        "id": ids[order].astype(str),
        "file_index": file_index,
        "row": rows,
        "appended": feats["restored"][order],
    }
    tmp = FEATURES_OUT / f".{slide_id}.tmp.npz"
    np.savez(tmp, **out)
    back = np.load(tmp)
    if not all(np.array_equal(back[k], out[k], equal_nan=k != "id") for k in out):
        tmp.unlink()
        raise RuntimeError(f"{tmp}: read-back check failed")
    os.replace(tmp, FEATURES_OUT / f"{slide_id}.npz")
    stats["objects"] = len(ids)
    stats["appended"] = int(feats["restored"].sum())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--slides", nargs="*")
    args = parser.parse_args()
    ORIGINAL.mkdir(exist_ok=True)
    FEATURES_OUT.mkdir(exist_ok=True)
    for f in BATCH.glob("_completed_slides_*.txt"):
        shutil.copy2(f, ORIGINAL / f.name)
    slides = args.slides or sorted(
        p.name.removeprefix("slide_id=") for p in BATCH.glob("slide_id=*")
    )
    done_path = ORIGINAL / "_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [s for s in slides if s not in done]
    slides_csv = ORIGINAL / "_slides.csv"
    rows = []
    with (
        Pool(args.workers, maxtasksperchild=20) as pool,
        done_path.open("a") as done_file,
    ):
        for i, row in enumerate(pool.imap_unordered(reconstruct, todo, chunksize=1), 1):
            rows.append(row)
            done_file.write(f"{row['slide_id']}\n")
            done_file.flush()
            if i % 200 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    table = pd.DataFrame(rows)
    if slides_csv.exists():
        table = pd.concat([pd.read_csv(slides_csv), table]).drop_duplicates(
            "slide_id", keep="last"
        )
    table.sort_values("slide_id").to_csv(slides_csv, index=False)
    if args.slides:
        print(table.drop(columns="slide_id").sum().to_string())
        return

    total = int(table.objects.sum())
    sources = {k: int(table[k].sum()) for k in SOURCES}
    written = sum(
        pq.ParquetFile(p).metadata.num_rows
        for p in ORIGINAL.glob("slide_id=*/*.parquet")
    )
    expected = json.loads((SOURCE / "_area_filter.json").read_text())["runs"][-1]
    expected = expected["kept"] + expected["removed"]
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slides": len(table),
        "objects": total,
        "objects_written": written,
        "objects_from": sources,
        "appended_objects_over_250um2": int(table.appended.sum()),
        "cellpose_objects_before_any_filter": expected,
        "totals_add_up": total == written == sum(sources.values()) == expected,
        "order": (
            "per file: the rows of the stain features (original Cellpose order, objects over 250 µm² "
            "left out), then that file's objects over 250 µm² in the area filter's order"
        ),
        "features": str(FEATURES_OUT),
        "script": "scripts/preprocessing/nuclei_segmentation/reconstruct_original_icaird_cervix_cellpose.py",
    }
    (ORIGINAL / "_reconstruction.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not summary["totals_add_up"]:
        raise SystemExit("the totals do not add up")


if __name__ == "__main__":
    main()
