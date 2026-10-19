"""Thesis figure: architecture panel (a), one tissue region through the pipeline.

Four images of the same 250 × 250 µm region of a PANDA Radboud test slide, each
exactly the region (no margin), drawn with the same µm coordinates so that they
overlay exactly:
(a1) the level-0 H&E with the nucleus polygons and a 50 µm scale bar,
(a2) the polygons and centroids alone,
(a4) the centroids and the k = 24 neighbourhood of one nucleus, built like
     `GraphCollator` (sklearn `NearestNeighbors` on the whole slide's centred µm
     positions, the query nucleus itself included, so 23 lines),
(a5) the polygons filled by sigmoid(z_i), the explained model's stored test logit
     (`xai/<run>/activations/test/logits.npy`, no inference is run).

They are then composed into the panel (`panel_a.pdf` / `.png`): a1 → a2 → embedding
box → a4 → a5 with its colorbar, and below a5 the attention pooling to the slide
logit. Text, arrows and boxes are vector; the four images are embedded unresampled.

Usage:
    uv run -m visualization.thesis_architecture_panel_a [--log]

`--log` logs the images, the panel, checks and this script to the panel's MLflow run
(experiment 37).
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


MODEL_RUN_ID = "c8f268536be8416f9fe395eea95c458e"  # Radboud, k=24: Polygons + RoVE + k-NN Attn
PANEL_RUN_ID = "21b298d40aa84d268093bd24edbf5665"  # "Thesis figure: architecture panel (a)"
XAI_DIR = Path("/mnt/projects/nuclei_based_wsi_analysis/xai") / MODEL_RUN_ID
SLIDE_ID = "89b62c165838d36fd294a57624db8e4f"  # PANDA Radboud test, Gleason 4+3
X_UM = (275.0, 525.0)
Y_UM = (2465.0, 2715.0)
K = 24  # as in training and in the activation run

# images: 720 px is the region box of the first (1000 px, margin 140) renders, so all
# line widths and marker sizes in px are unchanged
IMG_PX = 720
DPI = 200
SCALE_BAR_UM = 50.0
SCALE_BAR_H_UM = 5.0
SCALE_BAR_PAD_UM = 15.0

PX_PT = 72 / DPI  # one image pixel in points
OUTLINE_LW = 1 * PX_PT
HE_OUTLINE = (0.0, 0.0, 0.0, 0.6)
POLYGON_GREY = "#8c8c8c"
CENTROID_DARK = "#262626"
CENTROID_GREY = "#a6a6a6"
ACCENT_COLOR = "#E4572E"
CMAP = "viridis"

# panel layout, cm
CM = 1 / 2.54
PANEL_W = 16.0
IMG_CM = 2.35
EMBED_BOX = (3.7, 1.5)  # fits "(shape / texture / blank)" at 8 pt
POOL_BOX = (2.75, 1.25)
EDGE = 0.02  # figure edge to the outermost artists
CBAR_GAP, CBAR_W, CBAR_BLOCK = 0.1, 0.15, 1.15  # the block includes ticks and label
TOP, CAPTION_H, CAPTION_PAD = 0.03, 0.75, 0.1
ARROW_LEN, ARROW_PAD = 0.24, 0.08  # every arrow; pad to a box or an image below it
MIN_ARROW_CLEARANCE = 0.05  # horizontal arrow to the images beside it
FONT_SIZE, SMALL_FONT_SIZE = 8, 7
LINE_PT = 0.8
BORDER_GREY = "#999999"
EMBED_FILL = "#D5E8D4"
POOL_FILL = "#D1D6D1"
ARROW_STYLE = "-|>,head_length=3,head_width=1.5"

CAPTIONS = {
    "a1": "H&E with\nsegmented nuclei",
    "a2": "Point cloud: polygon\nand centroid $p_i$",
    "embed": "Per-nucleus input",
    "a4": "$L$ layers: attention to\n$k$ nearest neighbors",
    "a5": "Nucleus logits $z_i$",
}
EMBED_TEXT = "Nucleus embedding\n(shape / texture / blank)\n→ $x_i$"
POOL_TEXT = "Attention pooling\n$Z = \\sum_i \\alpha_i z_i$"
SLIDE_TEXT = "Slide logit $\\mathbfit{Z}$"

IMAGES = {"a1": "a1_he.png", "a2": "a2_points.png", "a4": "a4_knn.png", "a5": "a5_heatmap.png"}
OUT_DIR = Path("figures/arch_panel_a")


def load_region() -> dict:
    slides = pd.read_parquet(XAI_DIR / "slides.parquet").set_index("slide_id", drop=False)
    slide = slides.loc[SLIDE_ID]
    assert slide.split == "test"
    deck_slides = set(pd.read_csv(XAI_DIR / "session" / "regions_key.csv").slide_id)
    assert SLIDE_ID not in deck_slides, "slide is in the pathologist deck"

    # as BaseCropDataset: nuclei sorted by id, positions = level-0 centroid × mpp
    nuclei = pd.read_parquet(slide.slide_nuclei_path).sort_values("id").reset_index(drop=True)
    mpps = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
    centroids_um = np.stack(nuclei["centroid"].tolist()) * mpps
    polygons_um = np.array(nuclei["polygon"].tolist()).reshape(len(nuclei), -1, 2) * mpps

    # stored logits, in the index's row order (the slide's offset into the arrays)
    index = pd.read_parquet(
        XAI_DIR / "activations" / "test" / "index.parquet",
        filters=[("slide_id", "==", SLIDE_ID)],
    )
    assert len(index) == len(nuclei) == slide.n_nuclei
    assert (index["nucleus_id"].to_numpy() == nuclei["id"].to_numpy()).all()
    index_um = index[["centroid_x_um", "centroid_y_um"]].to_numpy()
    assert np.allclose(index_um, centroids_um, atol=1e-3)
    logits_all = np.load(XAI_DIR / "activations" / "test" / "logits.npy", mmap_mode="r")
    logits = np.asarray(logits_all[slide.offset : slide.offset + slide.n_nuclei], dtype=np.float64)

    # as GraphCollator: k-NN over the whole slide, on the dataset's centred positions
    pos = (centroids_um - centroids_um.mean(axis=0)).astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(pos)

    inside = (
        (centroids_um[:, 0] >= X_UM[0]) & (centroids_um[:, 0] < X_UM[1])
        & (centroids_um[:, 1] >= Y_UM[0]) & (centroids_um[:, 1] < Y_UM[1])
    )
    region = np.flatnonzero(inside)
    centre = np.array([np.mean(X_UM), np.mean(Y_UM)])
    query = int(region[np.argmin(np.linalg.norm(centroids_um[region] - centre, axis=1))])
    dist, knn = nbrs.kneighbors(pos[[query]])
    knn, dist = knn[0], dist[0]
    assert knn[0] == query and dist[0] == 0.0

    # the neighbourhood is unambiguous (no tie at the k-th distance) and the same with
    # a KD-tree on the uncentred µm positions
    d_next, _ = KDTree(centroids_um).query(centroids_um[query], k=K + 1)
    assert d_next[K] - d_next[K - 1] > 1e-3
    _, knn_tree = KDTree(centroids_um).query(centroids_um[query], k=K)
    assert set(knn_tree) == set(knn)
    assert inside[knn].all(), "neighbourhood leaves the region"

    with OpenSlide(slide.slide_path) as wsi:
        lx, ly = int(X_UM[0] / slide.mpp_x) - 2, int(Y_UM[0] / slide.mpp_y) - 2
        rx, ry = int(np.ceil(X_UM[1] / slide.mpp_x)) + 2, int(np.ceil(Y_UM[1] / slide.mpp_y)) + 2
        he = np.asarray(wsi.read_region((lx, ly), 0, (rx - lx, ry - ly)).convert("RGB"))
    # pixel i is centred at level-0 coordinate i (as for the polygons and centroids)
    he_extent = (
        (lx - 0.5) * slide.mpp_x, (rx - 0.5) * slide.mpp_x,
        (ry - 0.5) * slide.mpp_y, (ly - 0.5) * slide.mpp_y,
    )

    labels = index["label"].to_numpy()
    return {
        "slide": slide,
        "nuclei": nuclei,
        "region": region,
        "centroids": centroids_um[region],
        "polygons": polygons_um[region],
        "prob": 1.0 / (1.0 + np.exp(-logits[region])),
        "labels": labels[region],
        "query": query,
        "knn": knn,
        "knn_dist": dist,
        "kth_gap_um": float(d_next[K] - d_next[K - 1]),
        "all_centroids": centroids_um,
        "logits": logits,
        "he": he,
        "he_extent": he_extent,
    }


def region_axes() -> tuple[plt.Figure, plt.Axes]:
    """An image that is exactly the region, y down as in the slide."""
    fig = plt.figure(figsize=(IMG_PX / DPI, IMG_PX / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(*X_UM)
    ax.set_ylim(Y_UM[1], Y_UM[0])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def outlines(polygons: NDArray, color, lw: float = OUTLINE_LW) -> PolyCollection:
    return PolyCollection(polygons, closed=True, facecolors="none", edgecolors=[color], linewidths=lw)


def scale_bar_um() -> tuple[float, float, float, float]:
    """Outer edges (x0, x1, y_top, y_bottom) of the scale bar, µm."""
    x0 = X_UM[0] + SCALE_BAR_PAD_UM
    y1 = Y_UM[1] - SCALE_BAR_PAD_UM
    return x0, x0 + SCALE_BAR_UM, y1 - SCALE_BAR_H_UM, y1


def draw_he(r: dict) -> plt.Figure:
    fig, ax = region_axes()
    ax.imshow(r["he"], extent=r["he_extent"], interpolation="bilinear")
    ax.add_collection(outlines(r["polygons"], HE_OUTLINE))

    # white bar with a black outline drawn inside it, so its outer length is 50 µm
    stroke_px = 3
    inset = stroke_px / 2 * (X_UM[1] - X_UM[0]) / IMG_PX
    x0, x1, y0, y1 = scale_bar_um()
    ax.add_patch(
        plt.Rectangle((x0 + inset, y0 + inset), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset,
                      facecolor="white", edgecolor="black", lw=stroke_px * PX_PT)
    )
    return fig


def draw_points(r: dict) -> plt.Figure:
    fig, ax = region_axes()
    ax.add_collection(outlines(r["polygons"], POLYGON_GREY))
    ax.scatter(*r["centroids"].T, s=(4 * PX_PT) ** 2, c=CENTROID_DARK, lw=0)
    return fig


def draw_knn(r: dict) -> plt.Figure:
    fig, ax = region_axes()
    in_knn = np.isin(r["region"], r["knn"])
    ax.scatter(*r["centroids"][~in_knn].T, s=(5 * PX_PT) ** 2, c=CENTROID_GREY, lw=0)
    centre = r["all_centroids"][r["query"]]
    others = r["all_centroids"][r["knn"][1:]]
    for p in others:
        ax.plot(*np.stack([centre, p]).T, color=ACCENT_COLOR, lw=1.5 * PX_PT, zorder=2)
    ax.scatter(*others.T, s=(8 * PX_PT) ** 2, c=ACCENT_COLOR, lw=0, zorder=3)
    ax.scatter(*centre, s=(14 * PX_PT) ** 2, c=ACCENT_COLOR, edgecolors="white",
               linewidths=2 * PX_PT, zorder=4)
    return fig


def draw_heatmap(r: dict) -> plt.Figure:
    fig, ax = region_axes()
    cells = PolyCollection(r["polygons"], closed=True, cmap=CMAP,
                           norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                           edgecolors="none", antialiased=True)
    cells.set_array(r["prob"])
    ax.add_collection(cells)
    return fig


def image_checks(figs: dict[str, plt.Figure], r: dict) -> dict:
    """Identical data transform in every image, nothing outside it, same nuclei."""
    checks: dict = {}
    ref = None
    for name, fig in figs.items():
        fig.canvas.draw()
        ax = fig.axes[0]
        assert ax.get_position().bounds == (0.0, 0.0, 1.0, 1.0)
        px = ax.transData.transform(r["centroids"])
        if ref is None:
            ref = px
        assert np.array_equal(px, ref), f"{name}: centroids not aligned"
        bbox = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
        assert np.allclose(bbox.extents, [0, 0, IMG_PX, IMG_PX]), f"{name}: {bbox}"
        size = fig.canvas.get_width_height()
        assert size == (IMG_PX, IMG_PX), f"{name}: {size}"

    n = len(r["region"])
    counts = {
        "a2_points.png": [len(figs["a2"].axes[0].collections[0].get_paths()),
                          len(figs["a2"].axes[0].collections[1].get_offsets())],
        "a4_knn.png": [sum(len(c.get_offsets()) for c in figs["a4"].axes[0].collections)],
        "a5_heatmap.png": [len(figs["a5"].axes[0].collections[0].get_paths())],
    }
    assert all(c == n for v in counts.values() for c in v), counts
    a4_lines = len(figs["a4"].axes[0].lines)
    assert a4_lines == K - 1
    checks["nuclei_drawn"] = counts
    checks["a4_lines"] = a4_lines
    checks["image_px"] = IMG_PX
    checks["um_per_px"] = (X_UM[1] - X_UM[0]) / IMG_PX
    return checks


def to_array(fig: plt.Figure) -> NDArray[np.uint8]:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def arrow(lay: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> FancyArrowPatch:
    patch = FancyArrowPatch(start, end, arrowstyle=ARROW_STYLE, mutation_scale=1,
                            lw=LINE_PT, color="black", shrinkA=0, shrinkB=0)
    lay.add_patch(patch)
    return patch


def text_box(lay: plt.Axes, centre: tuple[float, float], size: tuple[float, float],
             fill: str, text: str) -> tuple[FancyBboxPatch, plt.Text]:
    (cx, cy), (w, h) = centre, size
    patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0,rounding_size=0.15",
                           facecolor=fill, edgecolor="black", lw=LINE_PT)
    lay.add_patch(patch)
    label = lay.text(cx, cy, text, ha="center", va="center", ma="center", fontsize=FONT_SIZE)
    return patch, label


def draw_panel(images: dict[str, NDArray[np.uint8]]) -> tuple[plt.Figure, dict]:
    widths = {"a1": IMG_CM, "a2": IMG_CM, "embed": EMBED_BOX[0], "a4": IMG_CM, "a5": IMG_CM}
    height = TOP + CAPTION_H + CAPTION_PAD + IMG_CM + ARROW_PAD + ARROW_LEN + POOL_BOX[1] + EDGE
    img_top = height - TOP - CAPTION_H - CAPTION_PAD
    img_bottom = img_top - IMG_CM
    img_mid = (img_top + img_bottom) / 2

    fig = plt.figure(figsize=(PANEL_W * CM, height * CM), facecolor="white")
    lay = fig.add_axes((0, 0, 1, 1))  # layout axes in cm, under the images
    lay.set_xlim(0, PANEL_W)
    lay.set_ylim(0, height)
    lay.axis("off")

    def cm_rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / PANEL_W, y / height, w / PANEL_W, h / height)

    items: dict = {"captions": {}, "arrows": {}, "images": {}}
    for name, text in CAPTIONS.items():
        items["captions"][name] = lay.text(0, img_top + CAPTION_PAD, text, ha="center",
                                           va="bottom", ma="center", fontsize=FONT_SIZE)

    # captions wider than their element overhang into the gaps: each gap takes the
    # overhangs on its two sides plus an equal share of the remaining width, so all
    # neighbouring captions are equally far apart; the row starts by the first overhang
    renderer = fig.canvas.get_renderer()
    overhang = {
        name: max(0.0, (t.get_window_extent(renderer).width / fig.dpi * 2.54 - widths[name]) / 2)
        for name, t in items["captions"].items()
    }
    names = list(widths)
    start = EDGE + overhang[names[0]]
    needed = [overhang[a] + overhang[b] for a, b in zip(names[:-1], names[1:], strict=True)]
    free = PANEL_W - start - EDGE - sum(widths.values()) - CBAR_BLOCK - sum(needed)
    gaps = [n + free / len(needed) for n in needed]
    assert min(gaps) >= ARROW_LEN + 2 * MIN_ARROW_CLEARANCE, gaps
    left = {}
    x = start
    for name, gap in zip(names, [*gaps, 0.0], strict=True):
        left[name] = x
        x += widths[name] + gap

    for name in IMAGES:
        ax = fig.add_axes(cm_rect(left[name], img_bottom, IMG_CM, IMG_CM))
        ax.imshow(images[name], extent=(X_UM[0], X_UM[1], Y_UM[1], Y_UM[0]),
                  interpolation="none", aspect="auto")
        ax.set_xlim(*X_UM)
        ax.set_ylim(Y_UM[1], Y_UM[0])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BORDER_GREY)
            spine.set_linewidth(0.5)
        items["images"][name] = ax

    # the "50 µm" label right of the bar, on the image (no room below it)
    x0, x1, y0, y1 = scale_bar_um()
    items["scale_text"] = items["images"]["a1"].text(
        x1 + 5.0, (y0 + y1) / 2, f"{SCALE_BAR_UM:g} µm", ha="left", va="center",
        fontsize=SMALL_FONT_SIZE, color="white",
        path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
    )
    items["scale_bar_um"] = (x0, x1, y0, y1)

    embed_cx = left["embed"] + EMBED_BOX[0] / 2
    items["embed_box"] = text_box(lay, (embed_cx, img_mid), EMBED_BOX, EMBED_FILL, EMBED_TEXT)

    for name, caption in items["captions"].items():
        caption.set_x(left[name] + widths[name] / 2)

    for a, b in zip(names[:-1], names[1:], strict=True):
        mid = (left[a] + widths[a] + left[b]) / 2
        items["arrows"][f"{a}->{b}"] = arrow(
            lay, (mid - ARROW_LEN / 2, img_mid), (mid + ARROW_LEN / 2, img_mid)
        )

    # colorbar of a5: thin, the image's height
    cax = fig.add_axes(cm_rect(left["a5"] + IMG_CM + CBAR_GAP, img_bottom, CBAR_W, IMG_CM))
    mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, 1.0), cmap=CMAP)
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, width=0.5, length=2, pad=1.5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(r"sigmoid($z_i$)", fontsize=SMALL_FONT_SIZE, labelpad=2)
    cbar.solids.set_rasterized(False)  # matplotlib rasterizes long colorbars by default
    cbar.solids.set_edgecolor("face")
    items["colorbar"] = cax

    # attention pooling below a5, then the slide logit to its left
    a5_cx = left["a5"] + IMG_CM / 2
    pool_top = img_bottom - ARROW_PAD - ARROW_LEN
    pool_cy = pool_top - POOL_BOX[1] / 2
    items["arrows"]["a5->pool"] = arrow(lay, (a5_cx, img_bottom - ARROW_PAD), (a5_cx, pool_top))
    items["pool_box"] = text_box(lay, (a5_cx, pool_cy), POOL_BOX, POOL_FILL, POOL_TEXT)
    tail = a5_cx - POOL_BOX[0] / 2 - ARROW_PAD
    items["arrows"]["pool->slide"] = arrow(lay, (tail, pool_cy), (tail - ARROW_LEN, pool_cy))
    items["slide_text"] = lay.text(tail - ARROW_LEN - ARROW_PAD, pool_cy, SLIDE_TEXT, ha="right",
                                   va="center", fontsize=FONT_SIZE, fontweight="bold")
    items["size_cm"] = (PANEL_W, height)
    items["gaps_cm"] = gaps
    return fig, items


def panel_checks(fig: plt.Figure, items: dict) -> dict:
    """No overlaps between text, arrows, boxes and images; everything inside the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_cm = fig.dpi_scale_trans.inverted() + mpl.transforms.Affine2D().scale(2.54)

    def cm_box(bbox: Bbox) -> Bbox:
        return bbox.transformed(to_cm)

    boxes: dict[str, Bbox] = {}
    for name, t in items["captions"].items():
        boxes[f"caption:{name}"] = cm_box(t.get_window_extent(renderer))
    for name, ax in items["images"].items():
        boxes[f"image:{name}"] = cm_box(ax.get_window_extent(renderer))
    for name, a in items["arrows"].items():
        boxes[f"arrow:{name}"] = cm_box(a.get_window_extent(renderer))
    for name in ("embed_box", "pool_box"):
        patch, label = items[name]
        box, text = cm_box(patch.get_window_extent(renderer)), cm_box(label.get_window_extent(renderer))
        assert box.x0 < text.x0 and text.x1 < box.x1 and box.y0 < text.y0 and text.y1 < box.y1, (
            f"{name}: text not inside its box"
        )
        boxes[name] = box
    boxes["colorbar"] = cm_box(items["colorbar"].get_tightbbox(renderer))
    boxes["slide_text"] = cm_box(items["slide_text"].get_window_extent(renderer))

    width, height = items["size_cm"]
    clearances = {}
    names = list(boxes)
    for i, a in enumerate(names):
        b = boxes[a]
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= width and b.y1 <= height, f"{a} outside: {b}"
        for c in names[i + 1 :]:
            assert not b.overlaps(boxes[c]), f"{a} overlaps {c}"
            dx = max(boxes[c].x0 - b.x1, b.x0 - boxes[c].x1, 0.0)
            dy = max(boxes[c].y0 - b.y1, b.y0 - boxes[c].y1, 0.0)
            clearances[f"{a} | {c}"] = round(max(dx, dy), 3)

    # the scale label sits on a1, right of the bar, without touching it
    a1 = items["images"]["a1"]
    label = items["scale_text"].get_window_extent(renderer)
    x0, x1, y0, y1 = items["scale_bar_um"]
    bar = Bbox(a1.transData.transform([[x0, y1], [x1, y0]]))
    assert a1.get_window_extent(renderer).contains(label.x0, label.y0)
    assert a1.get_window_extent(renderer).contains(label.x1, label.y1)
    assert label.x0 > bar.x1 and not label.overlaps(bar)

    # equal aspect: square axes boxes showing the square region
    for name, ax in items["images"].items():
        b = boxes[f"image:{name}"]
        assert abs(b.width - b.height) < 1e-9, f"{name} not square: {b}"
        assert ax.get_images()[0].get_array().shape[:2] == (IMG_PX, IMG_PX)

    return {
        "size_cm": [round(width, 3), round(height, 3)],
        "image_cm": IMG_CM,
        "arrow_gaps_cm": [round(g, 3) for g in items["gaps_cm"]],
        "smallest_clearances_cm": dict(sorted(clearances.items(), key=lambda kv: kv[1])[:6]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="log the outputs to MLflow")
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": FONT_SIZE,
            "pdf.fonttype": 42,
        }
    )
    r = load_region()
    figs = {"a1": draw_he(r), "a2": draw_points(r), "a4": draw_knn(r), "a5": draw_heatmap(r)}
    checks = image_checks(figs, r)

    q = r["query"]
    y, p = r["labels"], r["prob"]
    checks |= {
        "model_run_id": MODEL_RUN_ID,
        "slide_id": SLIDE_ID,
        "gleason_score": r["slide"].gleason_score,
        "mpp": [float(r["slide"].mpp_x), float(r["slide"].mpp_y)],
        "region_um": {"x": list(X_UM), "y": list(Y_UM)},
        "region_rule": "nuclei with centroid in [x0, x1) × [y0, y1)",
        "n_nuclei": len(r["region"]),
        "n_carcinoma": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
        "mean_sigmoid_carcinoma": float(p[y == 1].mean()),
        "mean_sigmoid_benign": float(p[y == 0].mean()),
        "accuracy_at_0.5": float(((p > 0.5) == (y == 1)).mean()),
        "knn": {
            "k": K,
            "includes_query": True,
            "query_index": q,
            "query_nucleus_id": str(r["nuclei"]["id"].iloc[q]),
            "query_centroid_um": [float(v) for v in r["all_centroids"][q]],
            "query_label": int(r["labels"][np.flatnonzero(r["region"] == q)[0]]),
            "query_logit": float(r["logits"][q]),
            "neighbour_indices": [int(i) for i in r["knn"][1:]],
            "max_neighbour_distance_um": float(r["knn_dist"].max()),
            "kth_distance_gap_um": r["kth_gap_um"],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, fig in figs.items():
        fig.savefig(OUT_DIR / IMAGES[name], dpi=DPI, facecolor="white")
        fig.savefig(OUT_DIR / IMAGES[name].replace(".png", ".pdf"), facecolor="white")
        images[name] = to_array(fig)

    panel, items = draw_panel(images)
    checks["panel"] = panel_checks(panel, items)
    print(json.dumps(checks, indent=1))
    # unresampled images in the PDF, smooth downsampling in the PNG
    panel.savefig(OUT_DIR / "panel_a.pdf")
    for ax in items["images"].values():
        ax.get_images()[0].set_interpolation("antialiased")
    panel.savefig(OUT_DIR / "panel_a.png", dpi=300)
    (OUT_DIR / "checks.json").write_text(json.dumps(checks, indent=2))

    if args.log:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with mlflow.start_run(run_id=PANEL_RUN_ID) as run:
            mlflow.set_tag("git_commit", commit)
            mlflow.log_params(
                {
                    "model_run_id": MODEL_RUN_ID,
                    "slide_id": SLIDE_ID,
                    "region_x_um": f"{X_UM[0]:g}-{X_UM[1]:g}",
                    "region_y_um": f"{Y_UM[0]:g}-{Y_UM[1]:g}",
                    "k": K,
                    "knn_query_index": q,
                    "image_px": IMG_PX,
                    "panel_size_cm": "x".join(f"{v:.2f}" for v in checks["panel"]["size_cm"]),
                }
            )
            mlflow.log_artifacts(str(OUT_DIR), artifact_path="arch_panel_a")
            mlflow.log_artifact(__file__)
            print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
