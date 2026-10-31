"""Thesis figure: the tissue around the nucleus of the embeddings figure.

For each of VIEWS, a square field of level-0 H&E around the nucleus of `thesis_embeddings_figure`, with
every segmented (LSP-DETR) nucleus outlined that has a vertex in the field. The nucleus
of the embeddings figure is outlined in yellow with a thin dark outline, as in its
panel (a). Black scale bar with its label in the thesis font (Computer Modern, as the
other figures); the figure is drawn at print size (FIG_W_CM) and exported unscaled.

Usage:
    uv run -m visualization.thesis_embeddings_context_figure [--log]

`--log` logs the figure, checks and this script to the embeddings figure's MLflow run
(under `context/`).
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import PolyCollection
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

from nuclei_graph.data.datasets.nuclei_features import NucleiFeatureExtractor
from nuclei_graph.nuclei_graph_typing import Box
from visualization.thesis_embeddings_figure import (
    METADATA_URI,
    NUCLEI,
    NUCLEUS_IDS,
    POLYGON_COLOR,
    POLYGON_EDGE,
    POLYGON_EDGE_LW,
    SCALE_BAR_LW,
    SLIDE_ID,
    THESIS_RC,
)


EMBEDDINGS_RUN_ID = "2be3387b57b34f97a4941d54a93fbecd"
# name -> (side of the field in µm; scale bar in µm; placement): "centred" on the nucleus,
# or "segmented": the field holding the nucleus where the segmentation misses least
VIEWS = {
    "embeddings_context": (160.0, 50.0, "centred"),  # 10x the texture window
    "embeddings_context_closeup": (40.0, 10.0, "segmented"),  # the nucleus and a few neighbours
}
# "segmented": hematoxylin (colour deconvolution, Gaussian sigma 1 px, Otsu over the tissue)
# not covered by a polygon dilated by ~1 µm, in blobs of >= MIN_MISSED_UM2, as a share of
# all hematoxylin in the field; searched within SEARCH_UM, the nucleus's centroid kept
# >= NUCLEUS_MARGIN_UM from the field's edges
SEARCH_UM = 120.0
NUCLEUS_MARGIN_UM = 8.0
MIN_MISSED_UM2 = 3.0
FIG_W_CM = 10.0
HE_MIN_DPI = 300  # the H&E is shown bilinearly upsampled to at least this (display only)
NUCLEI_COLOR, NUCLEI_LW = "#7FDBFF", 0.7  # pt, the other segmented nuclei
HIGHLIGHT_LW = 1.0  # pt, the nucleus of the embeddings figure (yellow, thin dark outline)
BAR_INSET_PT = 12  # scale bar to the field's edges
OUT_DIR = Path("figures")


def load(slide: pd.Series) -> dict:
    (name, idx), = NUCLEI.items()
    nuclei = pd.read_parquet(slide.slide_nuclei_path).sort_values("id").reset_index(drop=True)
    assert nuclei["id"].iloc[idx] == NUCLEUS_IDS[name]
    polygons = np.array(nuclei["polygon"].tolist(), dtype=np.float32).reshape(len(nuclei), -1, 2)
    return {"name": name, "idx": idx, "polygons": polygons,
            "centroid": np.asarray(nuclei["centroid"].iloc[idx], dtype=np.float32)}


def read_around(slide: pd.Series, centroid: NDArray, side_um: float) -> tuple[NDArray, Box]:
    half_px = int(np.ceil(side_um / 2 / min(float(slide.mpp_x), float(slide.mpp_y)))) + 2
    box = Box(int(centroid[0]) - half_px, int(centroid[1]) - half_px,
              int(centroid[0]) + half_px, int(centroid[1]) + half_px)
    with OpenSlide(slide.slide_path) as wsi:
        return NucleiFeatureExtractor().read_region(wsi, box).array, box


def best_segmented_field(slide: pd.Series, data: dict, field_um: float) -> dict:
    """The field holding the nucleus where the segmentation misses least.

    Of the fields of side `field_um` that keep the nucleus's centroid NUCLEUS_MARGIN_UM
    from their edges, the one with the smallest share of hematoxylin outside the
    segmentation (ties: the least shifted); its centre relative to the centroid, in µm.
    """
    mpp = float(slide.mpp_x)
    rgb, box = read_around(slide, data["centroid"], SEARCH_UM)
    stain = cv2.GaussianBlur(rgb2hed(rgb)[..., 0], (0, 0), 1.0)
    tissue = rgb.mean(axis=-1) < 220
    nuclear = (stain > threshold_otsu(stain[tissue])) & tissue
    nuclear = cv2.morphologyEx(nuclear.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    segmented = np.zeros(nuclear.shape, np.uint8)
    for polygon in data["polygons"]:
        local = polygon - [box.lx, box.ly]
        if (local.max(axis=0) > 0).all() and (local.min(axis=0) < nuclear.shape[::-1]).all():
            cv2.fillPoly(segmented, [np.round(local * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    radius = max(1, round(1.0 / mpp))
    segmented = cv2.dilate(segmented, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats((nuclear & ~segmented).astype(np.uint8))
    missed = np.isin(labels, np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] * mpp**2 >= MIN_MISSED_UM2) + 1)

    def integral(mask: NDArray) -> NDArray:
        return np.pad(mask, ((1, 0), (1, 0))).astype(np.int64).cumsum(0).cumsum(1)

    sums = {"missed": integral(missed), "nuclear": integral(nuclear)}
    f = round(field_um / mpp)
    centre = np.array(nuclear.shape[::-1]) // 2  # the nucleus's centroid
    reach = f // 2 - round(NUCLEUS_MARGIN_UM / mpp)
    rows = []
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            x0, y0 = centre + np.array([dx, dy]) - f // 2
            total = {k: s[y0 + f, x0 + f] - s[y0, x0 + f] - s[y0 + f, x0] + s[y0, x0] for k, s in sums.items()}
            rows.append((dx, dy, total["missed"] / max(total["nuclear"], 1)))
    table = pd.DataFrame(rows, columns=["dx", "dy", "missed"])
    table["shift"] = np.hypot(table["dx"], table["dy"])
    best = table.sort_values(["missed", "shift"]).iloc[0]
    centred = table[(table["dx"] == 0) & (table["dy"] == 0)].iloc[0]
    return {"offset_um": [float(best["dx"] * mpp), float(best["dy"] * mpp)],
            "missed_hematoxylin_share": round(float(best["missed"]), 3),
            "missed_hematoxylin_share_centred": round(float(centred["missed"]), 3)}


def draw(slide: pd.Series, data: dict, field_um: float, bar_um: float,
         offset_um: NDArray) -> tuple[plt.Figure, dict]:
    """The field of side `field_um` centred at `offset_um` (µm) from the nucleus's centroid."""
    idx, polygons, centroid = data["idx"], data["polygons"], data["centroid"]
    mpp_x, mpp_y = float(slide.mpp_x), float(slide.mpp_y)
    mpp = np.array([mpp_x, mpp_y])
    offset_um = np.asarray(offset_um, dtype=np.float64)

    half_px = int(np.ceil((field_um / 2 + np.abs(offset_um).max()) / min(mpp_x, mpp_y))) + 2
    box = Box(int(centroid[0]) - half_px, int(centroid[1]) - half_px,
              int(centroid[0]) + half_px, int(centroid[1]) + half_px)
    with OpenSlide(slide.slide_path) as wsi:
        crop = NucleiFeatureExtractor().read_region(wsi, box).array

    def to_um(p: NDArray) -> NDArray:
        return (np.asarray(p, dtype=np.float64) - centroid) * mpp

    h = field_um / 2
    (cx, cy) = offset_um
    in_field = (np.abs(to_um(polygons) - offset_um) < h).all(axis=2).any(axis=1)
    others = in_field.copy()
    others[idx] = False

    side_in = FIG_W_CM / 2.54
    fig = plt.figure(figsize=(side_in, side_in))
    ax = fig.add_axes([0, 0, 1, 1])
    # pixel i is centred at coordinate lx + i
    left, top = (np.array([box.lx, box.ly]) - 0.5 - centroid) * mpp
    right, bottom = (np.array([box.rx, box.ry]) - 0.5 - centroid) * mpp
    upsample = int(np.ceil(HE_MIN_DPI * side_in / (field_um / mpp_x)))
    shown = cv2.resize(crop, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_LINEAR)
    ax.imshow(shown, extent=(left, right, bottom, top), interpolation="none")
    ax.add_collection(PolyCollection([to_um(p) for p in polygons[others]], closed=True,
                                     facecolors="none", edgecolors=NUCLEI_COLOR,
                                     linewidths=NUCLEI_LW, zorder=3))
    vertices = to_um(polygons[idx])
    ax.plot(*np.vstack([vertices, vertices[:1]]).T, color=POLYGON_COLOR, lw=HIGHLIGHT_LW, zorder=4,
            path_effects=[pe.withStroke(linewidth=HIGHLIGHT_LW + 2 * POLYGON_EDGE_LW,
                                        foreground=POLYGON_EDGE)])
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cy + h, cy - h)
    ax.set_aspect("equal")
    ax.axis("off")

    # black scale bar, its label in the thesis font as in the other figures, in the first
    # corner where neither touches a nucleus outline
    um_per_pt = field_um / (side_in * 72)
    inset = BAR_INSET_PT * um_per_pt
    outlines = shapely.MultiLineString([to_um(np.vstack([q, q[:1]])) for q in polygons[in_field]])
    corners = {"top right": (cx + h - inset, cy - h + inset + 10 * um_per_pt),
               "top left": (cx - h + inset + bar_um, cy - h + inset + 10 * um_per_pt),
               "bottom right": (cx + h - inset, cy + h - inset),
               "bottom left": (cx - h + inset + bar_um, cy + h - inset)}
    corner = None
    for name, (x1, y0) in corners.items():
        bar = ax.plot([x1 - bar_um, x1], [y0, y0], color="black", lw=SCALE_BAR_LW, solid_capstyle="butt")[0]
        label = ax.text(x1 - bar_um / 2, y0 - 2 * um_per_pt, rf"${bar_um:g}\,\mu\mathrm{{m}}$",
                        ha="center", va="bottom", fontsize=8, color="black")
        bb = label.get_window_extent(fig.canvas.get_renderer())
        (lx0, ly0), (lx1, ly1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        scale_bar = shapely.union(
            shapely.LineString([[x1 - bar_um, y0], [x1, y0]]).buffer(SCALE_BAR_LW / 2 * um_per_pt),
            shapely.box(min(lx0, lx1), min(ly0, ly1), max(lx0, lx1), max(ly0, ly1)))
        clearance_um = float(scale_bar.distance(outlines))
        if clearance_um > 2 * um_per_pt:
            corner = name
            break
        bar.remove()
        label.remove()
    else:
        raise AssertionError("no corner keeps the scale bar clear of the outlines")

    checks = {
        "slide_id": slide.slide_id,
        "nucleus_index": idx,
        "nucleus_id": NUCLEUS_IDS[data["name"]],
        "field_um": field_um,
        "field_centre_offset_from_nucleus_um": [round(float(v), 2) for v in offset_um],
        "nuclei_outlined": int(in_field.sum()),
        "he_upsample": upsample,
        "scale_bar_um": bar_um,
        "scale_bar_corner": corner,
        "scale_bar_clearance_to_outlines_um": round(clearance_um, 2),
        "scale_bar_clearance_to_outlines_pt": round(clearance_um / um_per_pt, 1),
        "mpp": mpp_x,
    }
    return fig, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="log the outputs to MLflow")
    args = parser.parse_args()

    meta = pd.read_parquet(download_artifacts(METADATA_URI))
    slide = meta.set_index("slide_id", drop=False).loc[SLIDE_ID]
    data = load(slide)

    OUT_DIR.mkdir(exist_ok=True)
    outputs = []
    for stem, (field_um, bar_um, placement) in VIEWS.items():
        field = {"offset_um": [0.0, 0.0]}
        if placement == "segmented":
            field = best_segmented_field(slide, data, field_um)
        with mpl.rc_context(THESIS_RC):  # fonts resolve at draw time: save inside
            fig, checks = draw(slide, data, field_um, bar_um, field["offset_um"])
            fig.savefig(OUT_DIR / f"{stem}.pdf")  # at print size, no bbox adjustment
            fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
            plt.close(fig)
        checks |= {"placement": placement} | {k: v for k, v in field.items() if k != "offset_um"}

        pdf = (OUT_DIR / f"{stem}.pdf").read_bytes()
        media = [float(v) for v in re.search(rb"/MediaBox \[\s*([\d.\s]+)\]", pdf).group(1).split()]
        checks["pdf_width_cm"] = round((media[2] - media[0]) / 72 * 2.54, 3)
        checks["pdf_fonts"] = sorted({f.decode() for f in re.findall(rb"/BaseFont /([\w+-]+)", pdf)})
        assert abs(checks["pdf_width_cm"] - FIG_W_CM) < 0.01, checks["pdf_width_cm"]
        assert checks["pdf_fonts"], "no font embedded"
        assert all(f.split("+")[-1].lower().startswith("cm") for f in checks["pdf_fonts"]), checks["pdf_fonts"]
        (OUT_DIR / f"{stem}_checks.json").write_text(json.dumps(checks, indent=2))
        print(stem, json.dumps(checks, indent=1))
        outputs += [f"{stem}.pdf", f"{stem}.png", f"{stem}_checks.json"]

    if args.log:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with mlflow.start_run(run_id=EMBEDDINGS_RUN_ID):
            for name in outputs:
                mlflow.log_artifact(str(OUT_DIR / name), "context")
            mlflow.log_artifact(__file__, "context")
            mlflow.set_tag("git_commit", commit)


if __name__ == "__main__":
    main()
