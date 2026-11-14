"""Fragments in the iCAIRD cervix Cellpose batch: objects under 5 µm², measured before any filter (read-only).

Analysis only, not a pipeline step. It checked a fragment filter (remove objects under
5 µm²) on 2026-11-14, on the batch of the first clean-up run. The filter was not applied:
the fragments are 8.2% of the objects, but 2-5x rarer inside lesions than outside (2.8-6.3%
vs 8.9-15.3% by slide diagnosis), and about half are real small nuclei by eye (14/24 random,
15/24 from malignant lesions; LSP-DETR, as a reference only, matches 56% of them).

Area: the polygon area in µm² at the slide's mpp (mean of x and y), as in the aligned stain
features (align_stain_features_icaird_cervix_cellpose.py), which hold every object of the
current batch with its id and centroid.

Writes to OUT:
    counts_by_slide_lesion.csv   per slide and lesion class: objects, fragments;
    by_category_lesion.csv       share by slide diagnosis, inside vs outside lesions;
    by_lesion_class.csv          share by the class of the lesion annotation;
    fragments.parquet            every fragment: slide, id, centroid, area, lesion;
    sample_lsp_detr.parquet      fragments in the 800-tile sample (wholecell_sample/tiles.csv,
                                 tile cores as in whole_cells_sample_icaird_cervix_cellpose.py),
                                 with the LSP-DETR reference (never a source of nuclei):
        lsp_match      an LSP-DETR nucleus holds the fragment's centroid, or its centroid
                       lies in the fragment;
        lsp_covered    that nucleus's centroid also lies in another Cellpose object (not a
                       fragment), so the fragment duplicates a nucleus the batch has;
    gallery_random.jpg, gallery_malignant.jpg   24 random fragments, and 24 from inside
                       malignant lesions: raw crop | outlines (red = fragment, green = other
                       Cellpose objects), 20 µm wide.

Lesion classes follow the annotations (low grade, high grade, malignant; overlaps take the
higher grade); the slide diagnosis is the iCAIRD index category.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.fragments_icaird_cervix_cellpose [--workers 18]
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.path import Path as Outline
from openslide import OpenSlide
from scipy.spatial import cKDTree

from scripts.preprocessing.nuclei_segmentation.align_stain_features_icaird_cervix_cellpose import (
    OUT as ALIGNED,
)
from scripts.preprocessing.nuclei_segmentation.clean_icaird_cervix_cellpose import (
    column,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    OUT as REVIEW,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)
from scripts.preprocessing.nuclei_segmentation.whole_cells_sample_icaird_cervix_cellpose import (
    CORE_FROM,
    CORE_TO,
    LSP_DETR,
    SAMPLE,
)


OUT = REVIEW / "fragment_analysis"
MAX_AREA_UM2 = 5.0
CLASSES = ["outside", "low grade", "high grade", "malignant", "other"]  # code = index
ROOT = SLIDES.parent
GALLERY_UM, TILE = 20.0, 240


def lesion_polys(slide_id: str) -> list[tuple[int, np.ndarray]]:
    out = []
    for ft in json.loads((ROOT / "annotations" / f"{slide_id}.txt").read_text())[
        "features"
    ]:
        g = ft["geometry"]
        rings = {
            "Polygon": lambda c: [c[0]],
            "MultiPolygon": lambda c: [p[0] for p in c],
        }
        name = (
            (ft.get("properties", {}).get("classification", {}) or {})
            .get("name", "")
            .lower()
            .strip()
        )
        code = CLASSES.index(name) if name in CLASSES else 4
        out += [
            (code, np.asarray(r, float))
            for r in rings.get(g["type"], lambda c: [])(g["coordinates"])
            if len(r) > 2
        ]
    return out


def classify(c: np.ndarray, polys: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Lesion class of each centroid; overlapping annotations take the higher grade."""
    code = np.zeros(len(c), np.int8)
    rank = np.array([0, 2, 3, 4, 1])  # outside < other < low < high < malignant
    for k, r in polys:
        box = np.flatnonzero(np.all((c >= r.min(0)) & (c <= r.max(0)), 1))
        if not len(box):
            continue
        hit = box[Outline(r).contains_points(c[box])]
        code[hit[rank[k] > rank[code[hit]]]] = k
    return code


def count(slide_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = np.load(ALIGNED / f"{slide_id}.npz")
    code = classify(a["centroid"].astype(float), lesion_polys(slide_id))
    frag = a["area_um2"] < MAX_AREA_UM2
    counts = pd.DataFrame(
        {
            "slide_id": slide_id,
            "lesion": CLASSES,
            "objects": np.bincount(code, minlength=5),
            "fragments": np.bincount(code[frag], minlength=5),
        }
    )
    f = pd.DataFrame(
        {
            "slide_id": slide_id,
            "id": a["id"][frag],
            "cx": a["centroid"][frag, 0],
            "cy": a["centroid"][frag, 1],
            "area_um2": a["area_um2"][frag],
            "lesion": np.asarray(CLASSES, object)[code[frag]],
        }
    )
    return counts, f


def polygons(slide_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ids, polygons and areas (from the aligned features) of the slide's batch objects, in batch order."""
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    p = np.concatenate(
        [
            column(pq.ParquetFile(f).read(columns=["polygon"]), "polygon", (64, 2))
            for f in files
        ]
    )
    a = np.load(ALIGNED / f"{slide_id}.npz")
    if len(p) != len(a["id"]) or not np.allclose(p.mean(1), a["centroid"], atol=0.5):
        raise RuntimeError(f"{slide_id}: batch and aligned features differ")
    return a["id"], p, a["area_um2"]


def sample_slide(args: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    slide_id, tiles = args
    ids, polys, area = polygons(slide_id)
    c = polys.mean(1)
    boxes = [
        (x + CORE_FROM, y + CORE_FROM, x + CORE_TO, y + CORE_TO)
        for x, y in tiles[["tile_x", "tile_y"]].to_numpy()
    ]
    core = np.zeros(len(c), bool)
    for x0, y0, x1, y1 in boxes:
        core |= (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
    frag = np.flatnonzero(core & (area < MAX_AREA_UM2))
    rows = []
    lc, lp = [], []
    for f in sorted((LSP_DETR / f"slide_id={slide_id}").glob("*.parquet")):
        t = pq.ParquetFile(f).read(columns=["centroid", "polygon"])
        cc = column(t, "centroid", (2,))
        sel = np.zeros(len(cc), bool)
        for x0, y0, x1, y1 in boxes:
            sel |= (
                (cc[:, 0] >= x0 - 100)
                & (cc[:, 0] < x1 + 100)
                & (cc[:, 1] >= y0 - 100)
                & (cc[:, 1] < y1 + 100)
            )
        lc.append(cc[sel])
        lp.append(column(t, "polygon", (64, 2))[sel])
    lc, lp = np.concatenate(lc), np.concatenate(lp)
    ltree, ctree = cKDTree(lc), cKDTree(c)
    lr = np.linalg.norm(lp - lc[:, None], axis=2).max(1)
    cr = np.linalg.norm(polys - c[:, None], axis=2).max(1)
    for k in frag:
        match = -1
        for j in ltree.query_ball_point(c[k], lr.max()):
            if Outline(lp[j]).contains_point(c[k]) or Outline(polys[k]).contains_point(
                lc[j]
            ):
                match = j
                break
        covered = False
        if match >= 0:
            for o in ctree.query_ball_point(lc[match], cr.max()):
                if (
                    o != k
                    and area[o] >= MAX_AREA_UM2
                    and Outline(polys[o]).contains_point(lc[match])
                ):
                    covered = True
                    break
        rows.append(
            {
                "slide_id": slide_id,
                "id": ids[k],
                "area_um2": area[k],
                "lsp_match": match >= 0,
                "lsp_covered": covered,
            }
        )
    n_core = int(core.sum())
    return (
        pd.DataFrame(rows).assign(core_objects=n_core)
        if rows
        else pd.DataFrame(
            {"slide_id": [slide_id], "id": [None], "core_objects": [n_core]}
        )
    )


def gallery(sel: pd.DataFrame, path: Path) -> None:
    tiles = []
    for _, r in sel.sort_values("slide_id").iterrows():
        ids, polys, _ = polygons(r.slide_id)
        k = int(np.flatnonzero(ids == r.id)[0])
        with OpenSlide(str(SLIDES / f"{r.slide_id}.tiff")) as sl:
            mpp = float(sl.properties["openslide.mpp-x"])
            size = int(GALLERY_UM / mpp)
            x0, y0 = int(r.cx - size / 2), int(r.cy - size / 2)
            im = np.asarray(
                sl.read_region((x0, y0), 0, (size, size)).convert("RGB")
            ).copy()
        raw = im.copy()
        near = np.all(np.abs(polys.mean(1) - [r.cx, r.cy]) < size, 1)
        near[k] = False
        s = TILE / size
        big = [
            cv2.resize(a, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
            for a in (raw, im)
        ]
        cv2.polylines(
            big[1],
            [np.round((p - [x0, y0]) * s).astype(np.int32) for p in polys[near]],
            True,
            (0, 170, 70),
            1,
        )
        cv2.polylines(
            big[1],
            [np.round((polys[k] - [x0, y0]) * s).astype(np.int32)],
            True,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            big[1],
            f"{r.slide_id[6:11]} {r.area_um2:.1f}um2",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
        )
        tiles.append(np.concatenate(big, 1))
    while len(tiles) % 4:
        tiles.append(np.full((TILE, 2 * TILE, 3), 255, np.uint8))
    g = np.concatenate(
        [np.concatenate(tiles[i : i + 4], 1) for i in range(0, len(tiles), 4)], 0
    )
    cv2.imwrite(
        str(path), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )


def share(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "objects": g.objects.sum(),
            "fragments": g.fragments.sum(),
            "pct": 100 * g.fragments.sum() / max(g.objects.sum(), 1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = sorted(p.stem for p in ALIGNED.glob("IC-CX-*.npz"))
    with Pool(args.workers, maxtasksperchild=50) as pool:
        res = pool.map(count, slides, chunksize=4)
    counts = pd.concat([r[0] for r in res], ignore_index=True)
    frags = pd.concat([r[1] for r in res], ignore_index=True)
    idx = pd.read_csv(ROOT / "index.csv")
    cat = dict(zip(idx.slide.str.removesuffix(".isyntax"), idx.category, strict=True))
    counts["category"], frags["category"] = (
        counts.slide_id.map(cat),
        frags.slide_id.map(cat),
    )
    counts["inside_lesion"] = counts.lesion != "outside"
    counts.to_csv(OUT / "counts_by_slide_lesion.csv", index=False)
    frags.to_parquet(OUT / "fragments.parquet")
    overall = share(counts)
    by_cat = counts.groupby(["category", "inside_lesion"])[
        ["objects", "fragments"]
    ].apply(share)
    by_cls = counts.groupby("lesion")[["objects", "fragments"]].apply(share)
    by_cat.to_csv(OUT / "by_category_lesion.csv")
    by_cls.to_csv(OUT / "by_lesion_class.csv")
    print(overall.to_string(), by_cat.to_string(), by_cls.to_string(), sep="\n\n")
    per_slide = counts.groupby("slide_id")[["objects", "fragments"]].sum()
    print(
        "\nper slide pct: median",
        (100 * per_slide.fragments / per_slide.objects).median(),
        "p90",
        (100 * per_slide.fragments / per_slide.objects).quantile(0.9),
        "max",
        (100 * per_slide.fragments / per_slide.objects).max(),
    )

    tiles = pd.read_csv(SAMPLE / "tiles.csv")
    with Pool(args.workers) as pool:
        sample = pd.concat(
            pool.map(sample_slide, list(tiles.groupby("slide_id")), chunksize=2),
            ignore_index=True,
        )
    sample.to_parquet(OUT / "sample_lsp_detr.parquet")
    f = sample.dropna(subset=["id"])
    core = sample.groupby("slide_id").core_objects.first().sum()
    print(
        f"\nsample: {len(f)} fragments of {core} objects in the tile cores ({100 * len(f) / core:.3f}%); "
        f"LSP-DETR match {f.lsp_match.sum()}, of which covered by another Cellpose object {f.lsp_covered.sum()}"
    )

    gallery(frags.sample(24, random_state=0), OUT / "gallery_random.jpg")
    gallery(
        frags[frags.lesion == "malignant"].sample(24, random_state=0),
        OUT / "gallery_malignant.jpg",
    )


if __name__ == "__main__":
    main()
