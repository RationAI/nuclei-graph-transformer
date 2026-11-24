"""Masks of the iCAIRD test labels (label-1 nuclei inside the pathologist's regions) for the report.

Drawn exactly as the evaluation's prediction masks (``NucleiPredictionMasksCallback`` at level 1:
nucleus polygons scaled to the level, tiled BigTIFF with the level's mpp), with 255 for every
nucleus labelled 1 in the re-annotation labels and 0 elsewhere. One TIFF per positive test slide.

Usage: python render_label_masks.py <out_dir>
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
from einops import rearrange
from openslide import OpenSlide
from PIL import Image as PILImage
from PIL import ImageDraw
from rationai.masks import slide_resolution, write_big_tiff


LEVEL = 1  # as the evaluation runs' nuclei_masks callback
TILE = 512
TEST_SLIDES = Path("/mnt/projects/nuclei_based_wsi_analysis/evaluation/icaird_test/inference/test_slides.parquet")
LABELS = Path("/mnt/projects/nuclei_based_wsi_analysis/nuclei_supervision/iCAIRD/reannotation")


def render(args: tuple[str, str, str, Path]) -> tuple[str, int]:
    slide_id, slide_path, nuclei_path, out_dir = args
    with OpenSlide(slide_path) as slide:
        size = slide.level_dimensions[LEVEL]
        base_x, base_y = slide_resolution(slide, 0)
        mpp_x, mpp_y = slide_resolution(slide, LEVEL)
    sx, sy = base_x / mpp_x, base_y / mpp_y
    labels = pd.read_parquet(LABELS / f"{slide_id}.parquet", columns=["id", "annot_label"])
    positive = set(labels.loc[labels["annot_label"] == 1, "id"])
    nuclei = pd.read_parquet(nuclei_path, columns=["id", "polygon"])
    nuclei = nuclei[nuclei["id"].isin(positive)]
    assert len(nuclei) == len(positive), f"{slide_id}: labelled nuclei missing from the nuclei file"
    mask = PILImage.new("L", size, color=0)
    canvas = ImageDraw.Draw(mask)
    for poly in nuclei["polygon"]:
        pts = [(x * sx, y * sy) for x, y in rearrange(poly, "(n c) -> n c", c=2)]
        canvas.polygon(pts, fill=255, outline=255)
    write_big_tiff(
        image=pyvips.Image.new_from_array(np.array(mask)),
        path=out_dir / f"{slide_id}.tiff",
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=TILE,
        tile_height=TILE,
    )
    return slide_id, len(nuclei)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    slides = pd.read_parquet(TEST_SLIDES)
    slides = slides[slides["is_carcinoma"]]
    assert len(slides) == 60
    jobs = [(r.slide_id, r.slide_path, r.slide_nuclei_path, out) for r in slides.itertuples()]
    with ProcessPoolExecutor(16) as pool:
        done = list(pool.map(render, jobs))
    print(f"{len(done)} masks, {sum(n for _, n in done):,} label-1 nuclei -> {out}")
