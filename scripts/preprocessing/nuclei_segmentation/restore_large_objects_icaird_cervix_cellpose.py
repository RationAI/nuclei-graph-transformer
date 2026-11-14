"""Restores the objects over 250 µm² to the iCAIRD cervix Cellpose batch: the area filter is dropped.

The area filter (filter_large_nuclei_icaird_cervix_cellpose.py, run 2026-11-12) removed
75,460 objects over 250 µm². A gallery of 24 random ones from malignant lesions showed about
10 real, enlarged tumour nuclei, so the filter removed the most atypical nuclei and is no
longer part of the pipeline. Its removed rows (id, polygon, centroid, area_um2), with its
summary `_area_filter.json`, are in SOURCE (moved there from the batch's side).

The blood and empty-cell clean-ups ran after the filter and never saw these objects. So
that the batch is what the pipeline gives without the filter, each restored object first
goes through the same rules, with the same parameters, slide by slide:
    1. its colour features are measured as in stain_features_icaird_cervix_cellpose.py
       (`polygon_features`) and saved to RESTORED/<slide>.npz;
    2. red blood cells by colour, and objects in the slide's blood regions (the regions
       computed by the blood clean-up from the other objects), are removed;
    3. empty cells are removed by the rule of clean_empty_cells_icaird_cervix_cellpose.py,
       with the slide's typical nucleus H OD (`h_ref`) of that clean-up; the ring excludes
       every Cellpose object of the slide (batch, removed blood and empty cells, restored).
Rows removed by steps 2-3 are saved, with `reason`, `rel_h` and `contrast`, to
REMOVED/slide_id=<slide>/<file>; the others are appended to the end of the batch file
they came from (their positions in it were not saved; the loaders sort by id). Before a
file is changed, its rows and centroids are checked against the features (as in the other
clean-ups); each rewritten file is read back and its rows, schema and ids checked.

Two effects are not redone, and the dry run reports their size: the restored objects
could have changed the blood regions (they count as "other" objects there) and the rings
of earlier empty-cell candidates nearby. `regions_changed` counts the batch objects whose
blood-region status would change; `empty_candidates_near` counts earlier empty-cell
candidates within reach of a restored object (an upper bound on changed decisions).

Afterwards the features of the batch are those of `all_features`: the original stain
features, then the restored ones (`restored` True, `removed_after_restore` for steps 2-3).
A slide is listed in REMOVED/_done.txt once finished; a rerun skips listed slides and
redoes the others safely (files already restored are detected and left as they are). A
summary is appended to the batch's `_restore_large_objects.json`.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.restore_large_objects_icaird_cervix_cellpose \
        [--workers 14] [--slides ID ...] [--dry-run]
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
from scipy.spatial import cKDTree

from scripts.preprocessing.nuclei_segmentation.blood_regions_icaird_cervix_cellpose import (
    CELL_PX,
    blood_regions,
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
from scripts.preprocessing.nuclei_segmentation.clean_empty_cells_icaird_cervix_cellpose import (
    H_RATIO,
    MAX_CONTRAST,
    MIN_AREA_UM2,
    RING_OUT,
    ring_contrast,
    write_checked,
)
from scripts.preprocessing.nuclei_segmentation.clean_empty_cells_icaird_cervix_cellpose import (
    REMOVED as EMPTY_REMOVED,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    polygon_features,
)


SOURCE = SEG.parent / "_archive" / "cervix_cellpose_removed_area_gt250um2"
RESTORED = STAIN / "objects_restored"
REMOVED = SEG.with_name("cervix_cellpose_removed_restored_large")
FEATURES = ["file_index", "area_um2", "h", "e", "r", "g", "b", "npx", "centroid"]
BATCH_COLUMNS = ["id", "polygon", "centroid"]


def all_features(slide_id: str) -> dict[str, np.ndarray]:
    """The original stain features, then those of the restored objects (in source-file order)."""
    orig = np.load(STAIN / "objects" / f"{slide_id}.npz")
    out = {k: orig[k] for k in FEATURES}
    n = len(out["h"])
    out["restored"] = np.zeros(n, bool)
    out["removed_after_restore"] = np.zeros(n, bool)
    path = RESTORED / f"{slide_id}.npz"
    if path.exists():
        r = np.load(path)
        for k in FEATURES:
            out[k] = np.concatenate([out[k], r[k].astype(out[k].dtype)])
        out["restored"] = np.concatenate([out["restored"], np.ones(len(r["h"]), bool)])
        out["removed_after_restore"] = np.concatenate([out["removed_after_restore"], r["reason"] != ""])
    return out


def current_before(slide_id: str, feats: np.lib.npyio.NpzFile) -> np.ndarray:
    """Mask over the original features of the objects in the batch before the restore."""
    rbc = red_blood_cells(feats)
    regions = regions_of(slide_id, feats, rbc)
    keep = ~(rbc | (in_regions(feats["centroid"], regions) if regions is not None else False))
    measures = np.load(EMPTY_REMOVED / "_measures" / f"{slide_id}.npz")
    keep[measures["obj"][measures["removed"]]] = False
    return keep


def read_source(slide_id: str, files: list[Path]) -> tuple[list[tuple[int, pa.Table]], pa.Schema]:
    """The removed rows per batch file (index into `files`), with the batch schema."""
    out = []
    names = {f.name: i for i, f in enumerate(files)}
    schema = pq.ParquetFile(files[0]).schema_arrow
    for src in sorted((SOURCE / f"slide_id={slide_id}").glob("*.parquet")):
        if src.name not in names:
            raise RuntimeError(f"{src}: no batch file of that name")
        table = pq.ParquetFile(src).read(columns=BATCH_COLUMNS)
        if not table.schema.equals(schema, check_metadata=False):
            raise RuntimeError(f"{src}: schema differs from the batch")
        out.append((names[src.name], table.cast(schema)))
    return out, schema


def measure(slide_id: str) -> dict:
    """Features, rule decisions and diagnostics of the restored objects of one slide."""
    feats = np.load(STAIN / "objects" / f"{slide_id}.npz")
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    source, _ = read_source(slide_id, files)
    file_index = np.concatenate([np.full(len(t), i, np.int64) for i, t in source])
    ids = np.concatenate([np.asarray(t.column("id").to_pylist(), str) for _, t in source])
    polys = np.concatenate([column(t, "polygon", (64, 2)) for _, t in source]).astype(np.float32)

    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        mpp = (float(slide.properties["openslide.mpp-x"]) + float(slide.properties["openslide.mpp-y"])) / 2
        width, height = slide.level_dimensions[0]
        r = polygon_features(slide, polys, mpp)
    r |= {"file_index": file_index, "id": ids}

    # 2. blood: the regions of the blood clean-up
    rbc_orig = red_blood_cells(feats)
    regions = regions_of(slide_id, feats, rbc_orig)
    rbc = red_blood_cells(r)
    inside = in_regions(r["centroid"], regions) if regions is not None else np.zeros(len(ids), bool)
    grid = (-(-height // CELL_PX), -(-width // CELL_PX))
    with_restored = blood_regions(
        np.concatenate([feats["centroid"], r["centroid"]]), np.concatenate([rbc_orig, rbc]), grid
    )
    before = regions if regions is not None else np.zeros(grid, bool)
    regions_changed = int((in_regions(feats["centroid"], before) != in_regions(feats["centroid"], with_restored)).sum())

    # 3. empty cells, on the restored objects left after step 2
    measures = np.load(EMPTY_REMOVED / "_measures" / f"{slide_id}.npz")
    h_ref = float(measures["h_ref"])
    rel_h = r["h"] / h_ref
    cand = ~(rbc | inside) & (r["npx"] > 0) & (r["area_um2"] >= MIN_AREA_UM2) & (rel_h < H_RATIO)
    m = np.full((len(ids), 3), np.nan)
    if cand.any():
        others = [
            column(pq.ParquetFile(p).read(columns=["polygon"]), "polygon", (64, 2))
            for p in [
                *files,
                *sorted((BLOOD_REMOVED / f"slide_id={slide_id}").glob("*.parquet")),
                *sorted((EMPTY_REMOVED / f"slide_id={slide_id}").glob("*.parquet")),
            ]
        ]
        occupied = np.concatenate([polys, *others]).astype(np.float32)
        with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
            m[cand] = ring_contrast(slide, polys, np.flatnonzero(cand), occupied)
    contrast = m[:, 0] - m[:, 1]
    empty = cand & (contrast <= MAX_CONTRAST)

    # earlier empty-cell candidates whose ring a restored object could reach
    reach = np.linalg.norm(polys - polys.mean(1, keepdims=True), axis=2).max(1)
    cand_c = feats["centroid"][measures["obj"]]
    cand_r = 2 * np.sqrt(feats["area_um2"][measures["obj"]] / np.pi) / mpp + RING_OUT
    tree = cKDTree(r["centroid"])
    near = [
        bool(tree.query_ball_point(c, rad + reach.max()))
        and bool(np.any(np.linalg.norm(r["centroid"] - c, axis=1) < rad + reach))
        for c, rad in zip(cand_c, cand_r, strict=True)
    ]

    reason = np.where(rbc, "red_blood_cell", np.where(inside, "in_blood_region", np.where(empty, "empty_cell", "")))
    return r | {
        "rbc": rbc,
        "in_region": inside,
        "rel_h": rel_h.astype(np.float32),
        "h_in": m[:, 0],
        "h_ring": m[:, 1],
        "ring_px": m[:, 2],
        "contrast": contrast,
        "empty_candidate": cand,
        "reason": reason,
        "h_ref": np.float64(h_ref),
        "regions_changed": np.int64(regions_changed),
        "empty_candidates_near": np.int64(sum(near)),
    }


def restore_slide(slide_id: str, dry_run: bool = False) -> dict:
    store = RESTORED / f"{slide_id}.npz"
    if store.exists():
        r = dict(np.load(store))
    else:
        r = measure(slide_id)
        tmp = store.with_name(f".{store.stem}.tmp.npz")
        np.savez(tmp, **r)
        os.replace(tmp, store)
    reason = r["reason"]
    stats = {
        "slide_id": slide_id,
        "restored": len(reason),
        "kept": int((reason == "").sum()),
        "red_blood_cell": int((reason == "red_blood_cell").sum()),
        "in_blood_region": int((reason == "in_blood_region").sum()),
        "empty_cell": int((reason == "empty_cell").sum()),
        "empty_candidates": int(r["empty_candidate"].sum()),
        "empty_candidates_without_ring": int((r["empty_candidate"] & np.isnan(r["contrast"])).sum()),
        "regions_changed": int(r["regions_changed"]),
        "empty_candidates_near": int(r["empty_candidates_near"]),
        "files_rewritten": 0,
    }
    if dry_run:
        return stats

    feats = np.load(STAIN / "objects" / f"{slide_id}.npz")
    keep = current_before(slide_id, feats)
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    source, schema = read_source(slide_id, files)
    start = 0
    for i, src in source:
        sel = slice(start, start + len(src))
        start += len(src)
        if not np.array_equal(np.asarray(src.column("id").to_pylist(), str), r["id"][sel]):
            raise RuntimeError(f"{files[i]}: restored rows differ from {store}")
        back = reason[sel] == ""
        path = files[i]
        table = pq.ParquetFile(path).read()
        obj = np.flatnonzero(keep & (feats["file_index"] == i))
        add = src.filter(pa.array(back))
        done = (
            len(add) > 0
            and len(table) == len(obj) + len(add)
            and table.slice(len(obj)).column("id").equals(add.column("id"))
        )  # restored by an earlier run
        if not done and len(table) != len(obj):
            raise RuntimeError(f"{path}: {len(table)} rows, expected {len(obj)}")
        if not np.allclose(column(table, "centroid", (2,))[: len(obj)], feats["centroid"][obj], atol=0.5):
            raise RuntimeError(f"{path}: centroids differ from the colour features")
        if (~back).any():
            removed = (
                src.filter(pa.array(~back))
                .append_column("reason", pa.array(reason[sel][~back]))
                .append_column("rel_h", pa.array(r["rel_h"][sel][~back]))
                .append_column("contrast", pa.array(r["contrast"][sel][~back]))
            )
            out = REMOVED / f"slide_id={slide_id}" / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            write_checked(removed, out)
        if len(add) and not done:
            if not table.schema.equals(schema, check_metadata=True):
                raise RuntimeError(f"{path}: schema differs")
            write_checked(pa.concat_tables([table, add]), path)
            stats["files_rewritten"] += 1
    if start != len(reason):
        raise RuntimeError(f"{slide_id}: {start} source rows, {store} holds {len(reason)}")
    return stats


def count_rows(directory: Path) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in directory.glob("slide_id=*/*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--slides", nargs="*")
    parser.add_argument("--dry-run", action="store_true", help="only measure and decide; the batch is not changed")
    args = parser.parse_args()
    RESTORED.mkdir(parents=True, exist_ok=True)
    slides = args.slides or sorted(p.name.removeprefix("slide_id=") for p in SOURCE.glob("slide_id=*"))
    done_path = REMOVED / "_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = slides if args.dry_run else [s for s in slides if s not in done]

    rows = []
    with Pool(args.workers, maxtasksperchild=20) as pool:
        if args.dry_run:
            rows = pool.starmap(restore_slide, [(s, True) for s in todo], chunksize=1)
        else:
            REMOVED.mkdir(parents=True, exist_ok=True)
            with done_path.open("a") as done_file:
                for i, row in enumerate(pool.imap_unordered(restore_slide, todo, chunksize=1), 1):
                    rows.append(row)
                    done_file.write(f"{row['slide_id']}\n")
                    done_file.flush()
                    if i % 200 == 0:
                        print(f"{i}/{len(todo)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESTORED / ("_dry_run.csv" if args.dry_run else "_slides.csv"), index=False)
    totals = {k: int(v) for k, v in table.drop(columns="slide_id").sum().items()}
    print(json.dumps(totals, indent=1))
    if args.dry_run or args.slides:
        return

    batch = count_rows(SEG)
    removed = {d.name: count_rows(d) for d in (BLOOD_REMOVED, EMPTY_REMOVED, REMOVED)}
    before_filter = json.loads((SOURCE / "_area_filter.json").read_text())["runs"][-1]
    before_filter = before_filter["kept"] + before_filter["removed"]
    run = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "area_filter": "dropped: not part of the pipeline; its removed rows and summary are in " + str(SOURCE),
        "slides": len(rows),
        **totals,
        "restored_rows_removed_again_saved_in": str(REMOVED),
        "restored_features": str(RESTORED),
        "batch_objects": batch,
        "removed_objects": removed,
        "objects_before_area_filter": before_filter,
        "totals_add_up": batch + sum(removed.values()) == before_filter,
        "not_redone": (
            "blood regions and earlier empty-cell rings were not recomputed with the restored objects; "
            "regions_changed and empty_candidates_near give the size of that"
        ),
        "script": "scripts/preprocessing/nuclei_segmentation/restore_large_objects_icaird_cervix_cellpose.py",
    }
    summary_path = SEG / "_restore_large_objects.json"
    previous = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else []
    summary_path.write_text(json.dumps({"runs": [*previous, run]}, indent=1))
    print(json.dumps(run, indent=1))
    if not run["totals_add_up"]:
        raise SystemExit("the totals do not add up")


if __name__ == "__main__":
    main()
