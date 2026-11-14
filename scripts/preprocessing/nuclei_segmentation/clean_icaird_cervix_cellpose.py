"""Cleans the iCAIRD cervix Cellpose segmentation: red blood cells, blood regions, empty cells.

One pass from the original Cellpose objects (ORIGINAL, rebuilt by
reconstruct_original_icaird_cervix_cellpose.py) and their stain features (FEATURES,
measured by stain_features_icaird_cervix_cellpose.py), slide by slide:
    1. red blood cells: objects whose mean colour is red with almost no hematoxylin
       (`red_blood_cells` in blood_regions_icaird_cervix_cellpose.py);
    2. blood regions: where the red blood cells dominate on a 25 µm grid
       (`blood_regions`, computed from every object of the slide); every other object
       inside is removed, since there Cellpose mostly garbles what lies between the red
       blood cells (on IC-CX-01429-01, 61% of those objects match no LSP-DETR nucleus);
    3. empty cells: outlines that hold no hematoxylin. Cellpose-SAM sometimes outlines whole
       squamous cells or other pale structures instead of nuclei. An object left after
       steps 1-2 is an empty cell when all three hold:
         a. area >= MIN_AREA_UM2;
         b. mean H OD < H_RATIO x the slide's typical nucleus, the median H OD of the
            slide's objects under REF_AREA_UM2 that are not red blood cells;
         c. local hematoxylin contrast <= MAX_CONTRAST: the mean H OD inside the outline
            minus that in a ring 1-3 µm around it (9 and 25 px dilations at about
            0.25 µm/px). The ring excludes glass (mean RGB > GLASS) and every other Cellpose
            object of the slide. H is the hematoxylin channel of skimage's rgb2hed on the
            level-0 pixels as stored (not white-balanced). Objects whose ring has fewer
            than MIN_RING_PX pixels are kept.
No area filter and no fragment filter: both were checked and rejected (the 250 µm² filter
removed enlarged tumour nuclei; the objects under 5 µm² are about half real small nuclei
and are 2-5x rarer inside lesions than outside).

Calibration of step 3 (800 random tiles, 2 from each of 400 slides, 703k objects; LSP-DETR
used only as a referee, never as a source of nuclei): the rule removed 388 objects
(0.055%), of which 314 held no LSP-DETR nucleus, 64 were whole cells with a smaller LSP-DETR
nucleus inside, and 10 matched an LSP-DETR nucleus of about the same size; by eye about 3
of those 10 are real nuclei. The contrast threshold is sharp: at 0.001 instead of 0 the
rule removes 915 objects, 231 of them matching an LSP-DETR nucleus. Whole cells that hold
a nucleus have positive contrast and are kept.

Writes
    SEG/slide_id=<slide>/<file>.parquet      the kept rows, in original order;
    REMOVED/slide_id=<slide>/<file>.parquet  the removed rows with `reason`
                                             (red_blood_cell, in_blood_region, empty_cell),
                                             `rel_h` and `contrast` (NaN for steps 1-2);
    REMOVED/_measures/<slide>.npz            per slide the step-3 candidates (`obj` indexes
                                             the FEATURES rows) and their measurements,
                                             `h_ref` and the blood-region grid; a rerun
                                             uses them instead of measuring again;
    REMOVED/_slides.csv, SEG/_cleanup.json   counts per slide and the summary.
Every file is written as a hidden temporary file, read back (rows, schema, ids) and moved
into place. Before a slide is cleaned, its original files are checked against the
features (rows, ids, centroids to 0.5 px). A slide is listed in REMOVED/_done.txt when
finished; a rerun skips listed slides and rewrites the others whole, from ORIGINAL.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.clean_icaird_cervix_cellpose \
        [--workers 18] [--slides ID ...]
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
    CELL_PX,
    GROW,
    H_MAX,
    HUE_MIN,
    MIN_CELLS,
    MIN_RBC,
    RATIO,
    SIGMA,
    blood_regions,
    in_regions,
    red_blood_cells,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.stain_features_icaird_cervix_cellpose import (
    OUT as STAIN,
)


ORIGINAL = SEG.with_name("cervix_cellpose_original")
FEATURES = STAIN / "objects_original"
REMOVED = SEG.with_name("cervix_cellpose_removed")
MIN_AREA_UM2, H_RATIO, MAX_CONTRAST = 15.0, 0.5, 0.0
REF_AREA_UM2 = 40.0
PAD, RING_IN, RING_OUT, GLASS, MIN_RING_PX = 20, 9, 25, 215, 30
BLOCK = (
    1024  # level-0 px; candidates are measured block by block, one slide read per block
)
REASONS = ["red_blood_cell", "in_blood_region", "empty_cell"]
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


def column(table: pa.Table, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """A tensor column as an array (Ray's extension type is read through its storage)."""
    col = table.column(name).combine_chunks()
    if isinstance(col, pa.ExtensionArray):
        col = col.storage
    return np.asarray(col.flatten()).reshape(-1, *shape)


def write_checked(table: pa.Table, path: Path) -> None:
    """Write via a hidden temporary file; check rows, schema and ids before replacing `path`."""
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="snappy")
    back = pq.ParquetFile(tmp)
    ok = back.metadata.num_rows == len(table) and back.schema_arrow.equals(
        table.schema, check_metadata=True
    )
    ok = ok and back.read(columns=["id"]).column("id").equals(table.column("id"))
    if not ok:
        tmp.unlink()
        raise RuntimeError(f"{tmp}: read-back check failed")
    os.replace(tmp, path)


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
        im = np.asarray(
            slide.read_region(
                tuple(int(v) for v in b_lo), 0, tuple(int(v) for v in size)
            ).convert("RGB")
        )
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
            near = np.asarray(
                tree.query_ball_point(p.mean(0), max(w, h), p=np.inf), int
            )
            near = near[
                np.all(np.abs(centres[near] - p.mean(0)) < max(w, h), 1) & (near != k)
            ]
            if len(near):
                cv2.fillPoly(
                    occ, [(occupied[j] - lo[n]).astype(np.int32) for j in near], 1
                )
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


def read_original(
    slide_id: str, f: np.lib.npyio.NpzFile
) -> tuple[list[Path], list[pa.Table]]:
    """The slide's original files, checked against the features row by row."""
    files = sorted((ORIGINAL / f"slide_id={slide_id}").glob("*.parquet"))
    tables = [pq.ParquetFile(p).read() for p in files]
    fi = f["file_index"]
    if len(fi) and (np.any(np.diff(fi) < 0) or fi.max() >= len(files)):
        raise RuntimeError(f"{slide_id}: features are not grouped by file")
    for i, (path, table) in enumerate(zip(files, tables, strict=True)):
        sel = fi == i
        if len(table) != sel.sum() or not np.array_equal(
            f["row"][sel], np.arange(len(table))
        ):
            raise RuntimeError(f"{path}: {len(table)} rows, features hold {sel.sum()}")
        if not np.array_equal(
            np.asarray(table.column("id").to_pylist(), str), f["id"][sel]
        ):
            raise RuntimeError(f"{path}: ids differ from the features")
        if not np.allclose(
            column(table, "centroid", (2,)), f["centroid"][sel], atol=0.5
        ):
            raise RuntimeError(f"{path}: centroids differ from the features")
    return files, tables


def measure(
    slide_id: str, f: np.lib.npyio.NpzFile, polys: np.ndarray
) -> dict[str, np.ndarray]:
    """Steps 1-3 for one slide: blood regions, and the empty-cell candidates with their contrast."""
    rbc = red_blood_cells(f)
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        width, height = slide.level_dimensions[0]
    regions = blood_regions(
        f["centroid"], rbc, (-(-height // CELL_PX), -(-width // CELL_PX))
    )
    inside = in_regions(f["centroid"], regions) & ~rbc
    ref = (f["npx"] > 0) & (f["area_um2"] < REF_AREA_UM2) & ~rbc
    h_ref = float(np.median(f["h"][ref])) if ref.any() else np.nan
    rel_h = f["h"] / h_ref
    cand = (
        ~(rbc | inside)
        & (f["npx"] > 0)
        & (f["area_um2"] >= MIN_AREA_UM2)
        & (rel_h < H_RATIO)
    )
    obj = np.flatnonzero(cand)
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as slide:
        m = ring_contrast(slide, polys, obj, polys)
    contrast = m[:, 0] - m[:, 1]
    return {
        "obj": obj,
        "area_um2": f["area_um2"][obj],
        "rel_h": rel_h[obj].astype(np.float32),
        "h_in": m[:, 0],
        "h_ring": m[:, 1],
        "ring_px": m[:, 2],
        "contrast": contrast,
        "removed": contrast
        <= MAX_CONTRAST,  # NaN (ring too small) compares False: kept
        "h_ref": np.float64(h_ref),
        "rbc": np.flatnonzero(rbc),
        "in_blood_region": np.flatnonzero(inside),
        "regions": regions,
    }


def clean_slide(slide_id: str) -> dict:
    f = np.load(FEATURES / f"{slide_id}.npz")
    files, tables = read_original(slide_id, f)
    n = len(f["id"])
    stats = {
        "slide_id": slide_id,
        "objects": n,
        **dict.fromkeys(REASONS, 0),
        "kept": 0,
        "empty_candidates": 0,
        "empty_candidates_without_ring": 0,
        "blood_region_mm2": 0.0,
    }
    if not files:
        return stats

    store = REMOVED / "_measures" / f"{slide_id}.npz"
    if store.exists():
        meas = dict(np.load(store))
    else:
        polys = (
            np.concatenate([column(t, "polygon", (64, 2)) for t in tables]).astype(
                np.float32
            )
            if n
            else np.zeros((0, 64, 2), np.float32)
        )
        meas = measure(slide_id, f, polys)
        tmp = store.with_name(f".{store.stem}.tmp.npz")
        np.savez_compressed(tmp, **meas)
        os.replace(tmp, store)

    reason = np.full(n, "", object)
    reason[meas["obj"][meas["removed"]]] = "empty_cell"
    reason[meas["in_blood_region"]] = "in_blood_region"
    reason[meas["rbc"]] = "red_blood_cell"
    rel_h = np.full(n, np.nan, np.float32)
    contrast = np.full(n, np.nan)
    rel_h[meas["obj"]] = meas["rel_h"]
    contrast[meas["obj"]] = meas["contrast"]

    out_dir = SEG / f"slide_id={slide_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (path, table) in enumerate(zip(files, tables, strict=True)):
        sel = np.flatnonzero(f["file_index"] == i)
        drop = reason[sel] != ""
        write_checked(table.filter(pa.array(~drop)), out_dir / path.name)
        out = REMOVED / f"slide_id={slide_id}" / path.name
        if drop.any():
            removed = (
                table.filter(pa.array(drop))
                .append_column("reason", pa.array(reason[sel][drop].astype(str)))
                .append_column("rel_h", pa.array(rel_h[sel][drop]))
                .append_column("contrast", pa.array(contrast[sel][drop]))
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            write_checked(removed, out)
        elif out.exists():
            out.unlink()
    for r in REASONS:
        stats[r] = int((reason == r).sum())
    stats["kept"] = int((reason == "").sum())
    stats["empty_candidates"] = len(meas["obj"])
    stats["empty_candidates_without_ring"] = int(np.isnan(meas["contrast"]).sum())
    stats["blood_region_mm2"] = float(
        meas["regions"].sum() * (CELL_PX * 0.25 / 1000) ** 2
    )
    return stats


def count_rows(directory: Path, by_reason: bool = False) -> int | dict[str, int]:
    files = sorted(directory.glob("slide_id=*/*.parquet"))
    if not by_reason:
        return sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    counts: dict[str, int] = {}
    for p in files:
        for k, v in (
            pd.Series(
                pq.ParquetFile(p).read(columns=["reason"]).column("reason").to_pylist()
            )
            .value_counts()
            .items()
        ):
            counts[k] = counts.get(k, 0) + int(v)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--slides", nargs="*")
    args = parser.parse_args()
    (REMOVED / "_measures").mkdir(parents=True, exist_ok=True)
    SEG.mkdir(exist_ok=True)
    slides = args.slides or sorted(
        p.name.removeprefix("slide_id=") for p in ORIGINAL.glob("slide_id=*")
    )
    done_path = REMOVED / "_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [s for s in slides if s not in done]
    start = time.time()
    rows = []
    with (
        Pool(args.workers, maxtasksperchild=20) as pool,
        done_path.open("a") as done_file,
    ):
        for i, row in enumerate(pool.imap_unordered(clean_slide, todo, chunksize=1), 1):
            rows.append(row)
            done_file.write(f"{row['slide_id']}\n")
            done_file.flush()
            if i % 100 == 0:
                print(
                    f"{i}/{len(todo)} slides, {(time.time() - start) / 3600:.2f} h",
                    flush=True,
                )
    slides_csv = REMOVED / "_slides.csv"
    table = pd.DataFrame(rows)
    if slides_csv.exists():
        table = pd.concat([pd.read_csv(slides_csv), table]).drop_duplicates(
            "slide_id", keep="last"
        )
    table = table.sort_values("slide_id")
    table.to_csv(slides_csv, index=False)
    if args.slides:
        print(table.drop(columns="slide_id").sum().to_string())
        return

    original = json.loads((ORIGINAL / "_reconstruction.json").read_text())
    batch = count_rows(SEG)
    removed = count_rows(REMOVED, by_reason=True)
    totals = {k: int(table[k].sum()) for k in ["objects", *REASONS, "kept"]}
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": [
            {
                "step": "segmentation",
                "what": "Cellpose-SAM on the white-balanced slides (preprocessing/nuclei_segmentation.py, "
                "configs/experiment/preprocessing/nuclei_segmentation/icaird_cervix_cellpose.yaml), tissue only",
                "objects": original["objects"],
                "slides": len(table),
                "original_rows": str(ORIGINAL),
            },
            {
                "step": "stain features",
                "what": "mean H, E OD (skimage rgb2hed) and RGB under each polygon at level 1, area from the polygon",
                "features": str(FEATURES),
                "script": "scripts/preprocessing/nuclei_segmentation/stain_features_icaird_cervix_cellpose.py",
            },
            {
                "step": "1. red blood cells",
                "rule": f"hue of the mean RGB > {HUE_MIN}° (0° = red) and mean H OD < {H_MAX}",
                "removed": totals["red_blood_cell"],
            },
            {
                "step": "2. blood regions",
                "rule": (
                    f"red blood cells and other objects counted on a {CELL_PX * 0.25:g} µm grid, smoothed "
                    f"(Gaussian sigma {SIGMA} cells); blood where smoothed RBC >= {MIN_RBC} and >= {RATIO} x the "
                    f"others; regions under {MIN_CELLS} cells dropped, the rest grown by {GROW} cell; every "
                    "other object inside removed"
                ),
                "removed": totals["in_blood_region"],
                "slides_with_regions": int((table.blood_region_mm2 > 0).sum()),
                "region_mm2": round(float(table.blood_region_mm2.sum()), 2),
            },
            {
                "step": "3. empty cells",
                "rule": {
                    "min_area_um2": MIN_AREA_UM2,
                    "max_h_relative_to_typical_nucleus": H_RATIO,
                    "typical_nucleus": f"median H OD of the slide's non-RBC objects under {REF_AREA_UM2} µm²",
                    "max_contrast": MAX_CONTRAST,
                    "ring": f"{RING_IN}-{RING_OUT} px dilations (1-3 µm); glass (mean RGB > {GLASS}) and every "
                    "other Cellpose object of the slide excluded",
                    "min_ring_px": MIN_RING_PX,
                },
                "candidates": int(table.empty_candidates.sum()),
                "candidates_without_ring_kept": int(
                    table.empty_candidates_without_ring.sum()
                ),
                "removed": totals["empty_cell"],
                "calibration": CALIBRATION,
            },
        ],
        "not_applied": {
            "area_filter_over_250um2": "removed enlarged tumour nuclei (about 10 of 24 in a gallery from malignant lesions)",
            "fragment_filter_under_5um2": (
                "about half are real small nuclei by eye (14/24 random, 15/24 in malignant lesions), "
                "and their share is 2-5x lower inside lesions than outside"
            ),
            "blur_at_object_level": "the QC blur mask is patchy and only about a third of flagged crops are visibly blurred",
        },
        "objects": totals["objects"],
        "kept": totals["kept"],
        "removed": {r: totals[r] for r in REASONS},
        "batch_rows_on_disk": batch,
        "removed_rows_on_disk": removed,
        "totals_add_up": (
            totals["objects"] == original["objects"]
            and batch == totals["kept"]
            and removed == {r: totals[r] for r in REASONS if totals[r]}
            and batch + sum(removed.values()) == original["objects"]
        ),
        "batch": str(SEG),
        "removed_rows": str(REMOVED),
        "order": "original Cellpose order per file (objects over 250 µm² appended per file; see the reconstruction)",
        "script": "scripts/preprocessing/nuclei_segmentation/clean_icaird_cervix_cellpose.py",
    }
    (SEG / "_cleanup.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if not summary["totals_add_up"]:
        raise SystemExit("the totals do not add up")


if __name__ == "__main__":
    main()
