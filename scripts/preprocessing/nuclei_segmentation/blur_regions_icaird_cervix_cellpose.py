"""How many iCAIRD cervix Cellpose objects lie in regions the QC marked blurred (read-only).

Analysis only, not a pipeline step. It checked on 2026-11-14 (batch of the first clean-up
run) whether the QC blur mask could remove objects in blurred regions. It is not used at the
object level: 2.3% of the objects lie on blurred pixels, twice as often inside low and high
grade lesions (3.6-3.7%) as outside (1.9-2.0%), and only about a third of the flagged crops
look blurred (the mask is patchy at 16 µm). Whole slides are excluded instead, by the PANDA -
Radboud rule (scripts/preprocessing/quality_control/blur_exclusion_icaird_cervix.py).

The QC run "QC Masks: iCAIRD Cervix" (public MLflow, run 149a9da90b6046db8ac880b9c4cfc318;
mask level 3 = 2 µm/px, sample level 1, white-balance correction) wrote per slide:
    blur_per_pixel/<slide>.tiff   255 where PIQE marks the pixel blurred, else 0;
    blur_per_tile/<slide>.tiff    the blurred fraction of each 512 px tile, 0-255.
They were downloaded to QC (2,524 of the 2,539 slides; the QC failed on the others). An
object is in a blurred region when its centroid falls on a blurred pixel.

Writes to OUT:
    counts_by_slide_lesion.csv  per slide and lesion class: objects, in blurred pixels;
    by_category_lesion.csv, by_lesion_class.csv   shares as for the other filters;
    gallery_random.jpg, gallery_malignant.jpg   24 objects in blurred pixels, and 24 from
        inside malignant lesions, each from a random slide (up to 5 candidates per slide): each a 100 µm crop with the blurred region
        outlined (yellow), next to a crop around a random object of the same slide outside
        blurred pixels.

Usage:
    uv run python -m scripts.preprocessing.nuclei_segmentation.blur_regions_icaird_cervix_cellpose [--workers 18]
"""

import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import tifffile
from openslide import OpenSlide

from scripts.preprocessing.nuclei_segmentation.align_stain_features_icaird_cervix_cellpose import (
    OUT as ALIGNED,
)
from scripts.preprocessing.nuclei_segmentation.fragments_icaird_cervix_cellpose import (
    CLASSES,
    ROOT,
    classify,
    lesion_polys,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    OUT as REVIEW,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SLIDES,
)


QC = REVIEW / "qc_blur"
OUT = QC / "analysis"
MASK_DS = 8  # mask level 3
CROP_UM, TILE = 100.0, 300


def blur_mask(slide_id: str) -> np.ndarray | None:
    path = QC / "blur_per_pixel" / f"{slide_id}.tiff"
    return tifffile.imread(path) > 0 if path.exists() else None


def blurred(centroids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ij = np.clip(
        np.floor(centroids / MASK_DS).astype(int), 0, np.array(mask.shape[::-1]) - 1
    )
    return mask[ij[:, 1], ij[:, 0]]


def count(slide_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = blur_mask(slide_id)
    if mask is None:
        return pd.DataFrame(), pd.DataFrame()
    a = np.load(ALIGNED / f"{slide_id}.npz")
    c = a["centroid"].astype(float)
    code = classify(c, lesion_polys(slide_id))
    b = blurred(c, mask)
    counts = pd.DataFrame(
        {
            "slide_id": slide_id,
            "lesion": CLASSES,
            "objects": np.bincount(code, minlength=5),
            "blurred": np.bincount(code[b], minlength=5),
        }
    )
    rng = np.random.default_rng(0)
    pick = lambda m: (
        rng.choice(np.flatnonzero(m), min(5, int(m.sum())), replace=False)
        if m.any()
        else []
    )
    rows = [(k, "blurred") for k in pick(b)] + [
        (k, "malignant") for k in pick(b & (code == 3))
    ]
    sharp = np.flatnonzero(~b)
    picks = pd.DataFrame(
        [
            {
                "slide_id": slide_id,
                "kind": kind,
                "cx": c[k, 0],
                "cy": c[k, 1],
                "sx": c[s, 0],
                "sy": c[s, 1],
            }
            for k, kind in rows
            for s in [rng.choice(sharp)]
            if len(sharp)
        ]
    )
    return counts, picks


def crop(slide_id: str, x: float, y: float, mask: np.ndarray) -> np.ndarray:
    with OpenSlide(str(SLIDES / f"{slide_id}.tiff")) as sl:
        size = int(CROP_UM / float(sl.properties["openslide.mpp-x"]))
        x0, y0 = int(x - size / 2), int(y - size / 2)
        im = np.asarray(sl.read_region((x0, y0), 0, (size, size)).convert("RGB"))
    im = cv2.resize(im, (TILE, TILE), interpolation=cv2.INTER_AREA)
    m0, m1 = max(y0 // MASK_DS, 0), max(x0 // MASK_DS, 0)
    sub = mask[m0 : m0 + size // MASK_DS + 1, m1 : m1 + size // MASK_DS + 1].astype(
        np.uint8
    )
    full = np.zeros((size // MASK_DS + 1, size // MASK_DS + 1), np.uint8)
    full[: sub.shape[0], : sub.shape[1]] = sub
    full = cv2.resize(full, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours, -1, (255, 220, 0), 2)
    return im


def gallery(sel: pd.DataFrame, path: str) -> None:
    tiles = []
    for _, r in sel.sort_values("slide_id").iterrows():
        mask = blur_mask(r.slide_id)
        pair = [crop(r.slide_id, r.cx, r.cy, mask), crop(r.slide_id, r.sx, r.sy, mask)]
        cv2.putText(
            pair[0],
            f"{r.slide_id[6:11]} blurred",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            pair[1],
            "same slide, sharp",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        tiles.append(np.concatenate(pair, 1))
    while len(tiles) % 4:
        tiles.append(np.full((TILE, 2 * TILE, 3), 255, np.uint8))
    g = np.concatenate(
        [np.concatenate(tiles[i : i + 4], 1) for i in range(0, len(tiles), 4)], 0
    )
    cv2.imwrite(
        path, cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )


def share(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "objects": g.objects.sum(),
            "blurred": g.blurred.sum(),
            "pct": 100 * g.blurred.sum() / max(g.objects.sum(), 1),
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
    picks = pd.concat([r[1] for r in res], ignore_index=True)
    idx = pd.read_csv(ROOT / "index.csv")
    cat = dict(zip(idx.slide.str.removesuffix(".isyntax"), idx.category, strict=True))
    counts["category"] = counts.slide_id.map(cat)
    counts["inside_lesion"] = counts.lesion != "outside"
    counts.to_csv(OUT / "counts_by_slide_lesion.csv", index=False)
    by_cat = counts.groupby(["category", "inside_lesion"])[
        ["objects", "blurred"]
    ].apply(share)
    by_cls = counts.groupby("lesion")[["objects", "blurred"]].apply(share)
    by_cat.to_csv(OUT / "by_category_lesion.csv")
    by_cls.to_csv(OUT / "by_lesion_class.csv")
    per_slide = counts.groupby("slide_id")[["objects", "blurred"]].sum()
    pct = 100 * per_slide.blurred / per_slide.objects
    print(share(counts).to_string(), by_cat.to_string(), by_cls.to_string(), sep="\n\n")
    print(
        f"\nslides {len(per_slide)}; per-slide pct median {pct.median():.2f}, p90 {pct.quantile(0.9):.2f}, "
        f"max {pct.max():.1f}; slides over 20%: {(pct > 20).sum()}, over 50%: {(pct > 50).sum()}"
    )
    gallery(
        picks[picks.kind == "blurred"].sample(24, random_state=0),
        str(OUT / "gallery_random.jpg"),
    )
    gallery(
        picks[picks.kind == "malignant"].sample(24, random_state=0),
        str(OUT / "gallery_malignant.jpg"),
    )


if __name__ == "__main__":
    main()
