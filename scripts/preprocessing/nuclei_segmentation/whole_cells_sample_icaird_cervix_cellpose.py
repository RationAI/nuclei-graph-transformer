"""How often Cellpose outlines a whole cell around a nucleus, on 800 random iCAIRD cervix tiles.

The cleaned batch keeps whole cells that hold a nucleus: the empty-cell clean-up
(step 3 of clean_icaird_cervix_cellpose.py) removes only outlines without hematoxylin.
This script measures how many remain, using LSP-DETR as a referee only (its nuclei are
never added to the batch). The sample is 800 random 2,048 px tiles, 2 from each of 400
slides with at least 30 objects (icaird_cervix_review/wholecell_sample/tiles.csv). Each
object whose centroid lies in a tile's core ([x+32, x+2016) in both axes) is classed by
the LSP-DETR centroids inside its outline:
    nucleus                  the largest has at least MATCH_RATIO x the object's area;
    whole_cell_with_nucleus  there are some, but all are smaller;
    no_lsp                   there are none.
The objects are those of the batch after the blood steps: the current batch (stain
features aligned by align_stain_features_icaird_cervix_cellpose.py) plus the rows the
empty-cell step removed (`empty_removed`). The slide diagnosis comes from the
iCAIRD index. `lesion` is the class of the annotation polygon holding the centroid
("" outside every annotation); the annotations mark lesions only, not tissue types.

Writes to OUT: objects.parquet (one row per object) and the tables
whole_cells_overall.csv, whole_cells_by_category_lesion.csv and whole_cells_by_slide.csv.
A whole cell counted there is a whole_cell_with_nucleus that the empty-cell clean-up kept.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.whole_cells_sample_icaird_cervix_cellpose \
        [--workers 18]
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.path import Path as Outline
from scipy.spatial import cKDTree

from scripts.preprocessing.nuclei_segmentation.align_stain_features_icaird_cervix_cellpose import (
    OUT as ALIGNED,
)
from scripts.preprocessing.nuclei_segmentation.clean_icaird_cervix_cellpose import (
    FEATURES,
    REMOVED,
    column,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    OUT as REVIEW,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
    SLIDES,
)


SAMPLE = REVIEW / "wholecell_sample"
OUT = SAMPLE / "analysis"
LSP_DETR = Path("/mnt/public_projects/nuclei_based_wsi_analysis/nuclei_segmentation/iCAIRD/cervix_lsp-detr")
ANNOTATIONS, INDEX = SLIDES.parent / "annotations", SLIDES.parent / "index.csv"
MATCH_RATIO = 0.4
CORE_FROM, CORE_TO, MARGIN = 32, 2016, 200  # tile core in px; LSP-DETR read beyond it


def polygon_areas(p: np.ndarray) -> np.ndarray:
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y, -1))


def lesions(slide_id: str) -> list[tuple[str, Outline]]:
    features = json.loads((ANNOTATIONS / f"{slide_id}.txt").read_text())["features"]
    out = []
    for feature in features:
        geometry = feature["geometry"]
        rings = {"Polygon": lambda c: [c[0]], "MultiPolygon": lambda c: [p[0] for p in c]}
        name = feature.get("properties", {}).get("classification", {}).get("name", "").lower()
        for ring in rings.get(geometry["type"], lambda c: [])(geometry["coordinates"]):
            if len(ring) > 2:
                out.append((name, Outline(np.asarray(ring, float))))
    return out


def objects(slide_id: str) -> pd.DataFrame:
    """Objects after the blood steps of the clean-up: the batch rows plus the removed empty cells."""
    aligned = np.load(ALIGNED / f"{slide_id}.npz")
    files = sorted((SEG / f"slide_id={slide_id}").glob("*.parquet"))
    polys = [column(pq.ParquetFile(f).read(columns=["polygon"]), "polygon", (64, 2)) for f in files]
    current = pd.DataFrame({"obj": aligned["obj"], "area": aligned["area_um2"], "empty_removed": False})
    current["polygon"] = list(np.concatenate(polys))

    feats = np.load(FEATURES / f"{slide_id}.npz")  # `obj` of the aligned features indexes these
    removed = [pq.ParquetFile(f).read(columns=["id", "polygon", "reason"])
               for f in sorted((REMOVED / f"slide_id={slide_id}").glob("*.parquet"))]
    removed = [t.filter(pc.equal(t.column("reason"), "empty_cell")) for t in removed]
    ids = np.concatenate([np.asarray(t.column("id").to_pylist(), str) for t in removed]) if removed else np.zeros(0, str)
    rp = np.concatenate([column(t, "polygon", (64, 2)) for t in removed]) if removed else np.zeros((0, 64, 2), np.float32)
    gone = pd.Index(feats["id"]).get_indexer(ids)
    if (gone < 0).any():
        raise RuntimeError(f"{slide_id}: removed empty cells that are not original objects")
    gone_df = pd.DataFrame({"obj": gone, "area": feats["area_um2"][gone], "empty_removed": True})
    gone_df["polygon"] = list(rp)
    out = pd.concat([current, gone_df], ignore_index=True)
    out["centroid"] = list(feats["centroid"][out.obj])
    return out


def run(args: tuple[str, pd.DataFrame, str]) -> pd.DataFrame:
    slide_id, tiles, category = args
    d = objects(slide_id)
    c = np.stack(d.centroid)
    boxes = [(x + CORE_FROM, y + CORE_FROM, x + CORE_TO, y + CORE_TO) for x, y in tiles[["tile_x", "tile_y"]].to_numpy()]
    inside = np.zeros(len(c), bool)
    for x0, y0, x1, y1 in boxes:
        inside |= (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
    d = d[inside].reset_index(drop=True)
    c = c[inside]

    lc, lp = [], []
    for f in sorted((LSP_DETR / f"slide_id={slide_id}").glob("*.parquet")):
        t = pq.ParquetFile(f).read(columns=["centroid", "polygon"])
        cc = column(t, "centroid", (2,))
        sel = np.zeros(len(cc), bool)
        for x0, y0, x1, y1 in boxes:
            sel |= (cc[:, 0] >= x0 - MARGIN) & (cc[:, 0] < x1 + MARGIN) & (cc[:, 1] >= y0 - MARGIN) & (cc[:, 1] < y1 + MARGIN)
        lc.append(cc[sel])
        lp.append(column(t, "polygon", (64, 2))[sel])
    lc, la = np.concatenate(lc), polygon_areas(np.concatenate(lp).astype(float))
    tree = cKDTree(lc)
    cat, n_lsp = [], []
    for poly in d.polygon:
        r = np.linalg.norm(poly - poly.mean(0), axis=1).max()
        near = np.asarray(tree.query_ball_point(poly.mean(0), r), int)
        hit = near[Outline(poly).contains_points(lc[near])] if len(near) else near
        n_lsp.append(len(hit))
        if not len(hit):
            cat.append("no_lsp")
        elif la[hit].max() >= MATCH_RATIO * polygon_areas(poly.astype(float)):
            cat.append("nucleus")
        else:
            cat.append("whole_cell_with_nucleus")

    lesion = np.full(len(d), "", object)
    for name, outline in lesions(slide_id):
        hit = outline.contains_points(c)
        lesion[hit & (lesion == "")] = name
    return pd.DataFrame({"slide_id": slide_id, "category": category, "obj": d.obj, "area": d.area, "cat": cat,
                         "n_lsp": n_lsp, "empty_removed": d.empty_removed, "lesion": lesion})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    tiles = pd.read_csv(SAMPLE / "tiles.csv")
    index = pd.read_csv(INDEX)
    category = dict(zip(index.slide.str.removesuffix(".isyntax"), index.category, strict=True))
    jobs = [(s, g, category[s]) for s, g in tiles.groupby("slide_id")]
    with Pool(args.workers) as pool:
        d = pd.concat(pool.map(run, jobs, chunksize=2), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "objects.parquet")

    d["whole_cell"] = (d["cat"] == "whole_cell_with_nucleus") & ~d.empty_removed
    d["inside_lesion"] = d.lesion != ""

    def share(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"objects": len(g), "slides": g.slide_id.nunique(),
                          "whole_cells_with_nucleus": int(g.whole_cell.sum()), "pct": 100 * g.whole_cell.mean()})

    overall = share(d).to_frame("all").T
    overall.to_csv(OUT / "whole_cells_overall.csv")
    d.groupby(["category", "inside_lesion"])[d.columns].apply(share).to_csv(OUT / "whole_cells_by_category_lesion.csv")
    by_slide = d.groupby("slide_id").agg(category=("category", "first"), objects=("whole_cell", "size"),
                                         whole_cells_with_nucleus=("whole_cell", "sum"))
    by_slide["pct"] = 100 * by_slide.whole_cells_with_nucleus / by_slide.objects
    by_slide.to_csv(OUT / "whole_cells_by_slide.csv")
    print(overall.to_string())
    print(d[d.empty_removed]["cat"].value_counts().to_string())


if __name__ == "__main__":
    main()
