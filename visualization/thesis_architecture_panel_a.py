"""Thesis figure: architecture panel (a), one tissue region through the pipeline.

One row of three equal framed squares of the same 250 × 250 µm region of a PANDA
Radboud test slide, captions above, labelled arrows between them:
(a1) the level-0 H&E with the nucleus polygons and a 50 µm scale bar,
(a2) the nuclei as tokens: all polygons and centroids, the k-NN query nucleus filled
     and its k - 1 = 23 neighbours outlined and joined to it, built like
     `GraphCollator` (sklearn `NearestNeighbors` on the whole slide's centred µm
     positions, the query nucleus itself included),
(a3) the polygons filled by sigmoid(z_i), the explained model's stored test logit
     (`xai/<run>/activations/test/logits.npy`, no inference is run), with its colorbar.
The query nucleus is marked in all three.

The panel is drawn at its print size (16 cm wide, fonts in pt) and exported unscaled:
the squares are drawn in µm directly into the panel, so line widths and marker sizes
are in points at print size. Only the H&E is a raster. Each square is also saved on
its own at the same size.

Usage:
    uv run -m visualization.thesis_architecture_panel_a [--log]

`--log` logs the squares, the panel, checks and this script to the panel's MLflow run
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
from matplotlib.patches import FancyArrowPatch, Rectangle
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

SCALE_BAR_UM = 50.0
SCALE_BAR_H_UM = 5.0
SCALE_BAR_PAD_UM = 15.0

# colours; line widths and marker diameters in pt at print size
ORANGE = "#E6550D"
HE_OUTLINE, HE_OUTLINE_LW = (0.0, 0.0, 0.0, 0.5), 0.4
TOKEN_GREY, TOKEN_LW = "#BBBBBB", 0.4
CENTROID_GREY, CENTROID_D = "#888888", 1.0
NEIGHBOUR_LW = 0.8
EDGE_LW, EDGE_ALPHA = 0.5, 0.7
MARKER_LW, MARKER_RADIUS_UM = 0.8, 6.0  # open circle around the query nucleus
SCALE_BAR_LW = 0.5  # black outline of the white bar
TEXT_STROKE = 1.0  # black outline of the white scale-bar text (half of it shows)
CMAP = "viridis"

# panel layout, cm
CM = 1 / 2.54
PANEL_W = 16.0
EDGE = 0.02  # figure edge to the outermost artists
MIN_GAP = 1.6  # between the squares; widened if an arrow label needs more
CBAR_GAP, CBAR_W, CBAR_BLOCK = 0.12, 0.18, 1.16  # the block includes ticks and label
TOP, CAPTION_PAD, BOTTOM = 0.03, 0.12, 0.16  # BOTTOM: room for the lowest colorbar tick
ARROW_LEN = 0.9
ARROW_LABEL_PAD = 0.08  # arrow to the bottom of its label
MIN_CLEARANCE = 0.1  # arrow labels to the squares
FONT_SIZE, SMALL_FONT_SIZE = 8, 7
LINE_PT = 0.8
BORDER_GREY, BORDER_LW = "#999999", 0.5
ARROW_STYLE = "-|>,head_length=3,head_width=1.5"
DPI_PNG = 300

SQUARES = ("a1", "a2", "a3")
CAPTIONS = {
    "a1": "H&E with segmented nuclei",
    "a2": "Nuclei as tokens",
    "a3": "Nucleus logits $z_i$",
}
ARROW_LABELS = {
    "a1->a2": "shape (EFD)\n+ position $p_i$",
    "a2->a3": "$L$ layers",
}
FILES = {"a1": "a1_he", "a2": "a2_tokens", "a3": "a3_logits"}
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
        "all_polygons": polygons_um,
        "logits": logits,
        "he": he,
        "he_extent": he_extent,
    }


def frame(ax: plt.Axes) -> None:
    """Square axes over the region, y down as in the slide, with a thin grey frame."""
    ax.set_xlim(*X_UM)
    ax.set_ylim(Y_UM[1], Y_UM[0])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER_GREY)
        spine.set_linewidth(BORDER_LW)


def outlines(polygons: NDArray, color, lw: float, **kwargs) -> PolyCollection:
    return PolyCollection(polygons, closed=True, facecolors="none", edgecolors=[color],
                          linewidths=lw, **kwargs)


def query_circle(ax: plt.Axes, r: dict) -> plt.Circle:
    return ax.add_patch(plt.Circle(r["all_centroids"][r["query"]], MARKER_RADIUS_UM, fill=False,
                                   edgecolor=ORANGE, lw=MARKER_LW, zorder=5))


def scale_bar_um() -> tuple[float, float, float, float]:
    """Outer edges (x0, x1, y_top, y_bottom) of the scale bar, µm."""
    x0 = X_UM[0] + SCALE_BAR_PAD_UM
    y1 = Y_UM[1] - SCALE_BAR_PAD_UM
    return x0, x0 + SCALE_BAR_UM, y1 - SCALE_BAR_H_UM, y1


def draw_he(ax: plt.Axes, r: dict) -> dict:
    frame(ax)
    ax.imshow(r["he"], extent=r["he_extent"], interpolation="antialiased")
    ax.add_collection(outlines(r["polygons"], HE_OUTLINE, HE_OUTLINE_LW))

    # white bar with a black outline drawn inside it, so its outer length is 50 µm;
    # the label just above it
    um_per_pt = (X_UM[1] - X_UM[0]) / (ax.get_window_extent().width * 72 / ax.figure.dpi)
    inset = SCALE_BAR_LW / 2 * um_per_pt
    x0, x1, y0, y1 = scale_bar_um()
    ax.add_patch(Rectangle((x0 + inset, y0 + inset), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset,
                           facecolor="white", edgecolor="black", lw=SCALE_BAR_LW))
    text = ax.text((x0 + x1) / 2, y0 - 2.0, f"{SCALE_BAR_UM:g} µm", ha="center", va="bottom",
                   fontsize=SMALL_FONT_SIZE, color="white",
                   path_effects=[pe.withStroke(linewidth=TEXT_STROKE, foreground="black")])
    return {"scale_text": text, "scale_bar_um": (x0, x1, y0, y1), "marker": query_circle(ax, r)}


def draw_tokens(ax: plt.Axes, r: dict) -> dict:
    frame(ax)
    ax.add_collection(outlines(r["polygons"], TOKEN_GREY, TOKEN_LW))
    ax.scatter(*r["centroids"].T, s=CENTROID_D**2, c=CENTROID_GREY, lw=0, zorder=2)
    centre = r["all_centroids"][r["query"]]
    others = r["knn"][1:]
    for p in r["all_centroids"][others]:
        ax.plot(*np.stack([centre, p]).T, color=ORANGE, lw=EDGE_LW, alpha=EDGE_ALPHA, zorder=3)
    ax.add_collection(outlines(r["all_polygons"][others], ORANGE, NEIGHBOUR_LW, zorder=4))
    ax.add_collection(PolyCollection(r["all_polygons"][[r["query"]]], closed=True,
                                     facecolors=[ORANGE], edgecolors="none", zorder=5))
    return {}


def draw_heatmap(ax: plt.Axes, r: dict) -> dict:
    frame(ax)
    cells = PolyCollection(r["polygons"], closed=True, cmap=CMAP,
                           norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                           edgecolors="none", antialiased=True)
    cells.set_array(r["prob"])
    ax.add_collection(cells)
    return {"marker": query_circle(ax, r)}


DRAW = {"a1": draw_he, "a2": draw_tokens, "a3": draw_heatmap}


def arrow(lay: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> FancyArrowPatch:
    patch = FancyArrowPatch(start, end, arrowstyle=ARROW_STYLE, mutation_scale=1,
                            lw=LINE_PT, color="black", shrinkA=0, shrinkB=0)
    lay.add_patch(patch)
    return patch


def draw_panel(r: dict) -> tuple[plt.Figure, dict]:
    # caption height (one line each) and arrow-label widths at print size
    probe = plt.figure(figsize=(PANEL_W * CM, 2 * CM))
    renderer = probe.canvas.get_renderer()

    def size_cm(text: str, fontsize: float) -> tuple[float, float]:
        extent = probe.text(0, 0, text, fontsize=fontsize, ma="center").get_window_extent(renderer)
        return extent.width / probe.dpi * 2.54, extent.height / probe.dpi * 2.54

    caption_h = max(size_cm(t, FONT_SIZE)[1] for t in CAPTIONS.values())
    first_caption_w = size_cm(CAPTIONS[SQUARES[0]], FONT_SIZE)[0]
    label_w = max(size_cm(t, SMALL_FONT_SIZE)[0] for t in ARROW_LABELS.values())
    plt.close(probe)
    gap = max(MIN_GAP, label_w + 2 * MIN_CLEARANCE + 0.02)
    # the row starts far enough in for the first caption, if it is wider than its square
    lead = 0.0
    for _ in range(3):
        side = (PANEL_W - 2 * EDGE - lead - CBAR_BLOCK - (len(SQUARES) - 1) * gap) / len(SQUARES)
        lead = max(0.0, (first_caption_w - side) / 2 + 0.01)
    height = TOP + caption_h + CAPTION_PAD + side + BOTTOM
    bottom, mid = BOTTOM, BOTTOM + side / 2

    fig = plt.figure(figsize=(PANEL_W * CM, height * CM), facecolor="white")
    lay = fig.add_axes((0, 0, 1, 1))  # layout axes in cm, under the squares
    lay.set_xlim(0, PANEL_W)
    lay.set_ylim(0, height)
    lay.axis("off")

    def cm_rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / PANEL_W, y / height, w / PANEL_W, h / height)

    items: dict = {"axes": {}, "captions": {}, "arrows": {}, "arrow_labels": {}, "extras": {},
                   "side": side, "gap": gap, "size_cm": (PANEL_W, height)}
    left = {n: EDGE + lead + i * (side + gap) for i, n in enumerate(SQUARES)}
    # one shared baseline: the descent of the tallest caption above the squares
    baseline = bottom + side + CAPTION_PAD + 0.06
    for name in SQUARES:
        ax = fig.add_axes(cm_rect(left[name], bottom, side, side))
        items["axes"][name] = ax
        items["extras"][name] = DRAW[name](ax, r)
        items["captions"][name] = lay.text(left[name] + side / 2, baseline, CAPTIONS[name],
                                           ha="center", va="baseline", fontsize=FONT_SIZE)
    for a, b in zip(SQUARES[:-1], SQUARES[1:], strict=True):
        name = f"{a}->{b}"
        centre = left[a] + side + gap / 2
        items["arrows"][name] = arrow(lay, (centre - ARROW_LEN / 2, mid), (centre + ARROW_LEN / 2, mid))
        items["arrow_labels"][name] = lay.text(centre, mid + ARROW_LABEL_PAD, ARROW_LABELS[name],
                                               ha="center", va="bottom", ma="center",
                                               fontsize=SMALL_FONT_SIZE)

    cax = fig.add_axes(cm_rect(left["a3"] + side + CBAR_GAP, bottom, CBAR_W, side))
    mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, 1.0), cmap=CMAP)
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, width=0.5, length=2, pad=1.5)
    cbar.outline.set_linewidth(BORDER_LW)
    cbar.set_label(r"$\sigma(z_i)$", fontsize=SMALL_FONT_SIZE, labelpad=2)
    cbar.solids.set_rasterized(False)  # matplotlib rasterizes long colorbars by default
    cbar.solids.set_edgecolor("face")
    items["colorbar"] = cax
    return fig, items


def panel_checks(fig: plt.Figure, items: dict, r: dict) -> dict:
    """Equal aligned squares, equal gaps, centred arrows and labels; text sizes; no
    overlaps; the same µm mapping in all squares; the query nucleus marked in each.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_cm = fig.dpi_scale_trans.inverted() + mpl.transforms.Affine2D().scale(2.54)

    def cm_box(bbox: Bbox) -> Bbox:
        return bbox.transformed(to_cm)

    width, height = items["size_cm"]
    w_in, h_in = fig.get_size_inches()
    assert np.isclose(w_in * 2.54, PANEL_W) and np.isclose(h_in * 2.54, height)

    side = items["side"]
    squares = {n: cm_box(ax.get_window_extent(renderer)) for n, ax in items["axes"].items()}
    for name, b in squares.items():
        assert np.isclose(b.width, side) and np.isclose(b.height, side), f"{name}: {b}"
        assert np.isclose(b.y0, squares["a1"].y0), f"{name} not aligned"
    gaps = [squares[b].x0 - squares[a].x1 for a, b in zip(SQUARES[:-1], SQUARES[1:], strict=True)]
    assert np.allclose(gaps, items["gap"]), gaps

    for name, a in items["arrows"].items():
        p, q = name.split("->")
        (sx, sy), (ex, ey) = a._posA_posB  # the arrow's own endpoints, cm
        centre = (squares[p].x1 + squares[q].x0) / 2
        assert abs((sx + ex) / 2 - centre) < 1e-9, f"arrow {name} not centred"
        assert np.isclose(sy, squares[p].y0 + side / 2) and sy == ey, f"arrow {name} not at mid-height"
        label = cm_box(items["arrow_labels"][name].get_window_extent(renderer))
        assert abs((label.x0 + label.x1) / 2 - centre) < 0.01, f"label of {name} not centred"
        assert label.x0 - squares[p].x1 >= MIN_CLEARANCE and squares[q].x0 - label.x1 >= MIN_CLEARANCE, (
            f"label of {name} too close to the squares"
        )

    # text sizes at print size; captions one line on one baseline
    baselines = set()
    for name, t in items["captions"].items():
        assert t.get_fontsize() == FONT_SIZE and "\n" not in t.get_text(), name
        assert t.get_va() == "baseline"
        baselines.add(t.get_position()[1])
    assert len(baselines) == 1
    assert all(t.get_fontsize() == SMALL_FONT_SIZE for t in items["arrow_labels"].values())
    cax = items["colorbar"]
    assert all(t.get_fontsize() == SMALL_FONT_SIZE for t in cax.get_yticklabels())
    assert cax.yaxis.label.get_fontsize() == SMALL_FONT_SIZE
    assert items["extras"]["a1"]["scale_text"].get_fontsize() == SMALL_FONT_SIZE

    boxes: dict[str, Bbox] = {f"square:{n}": b for n, b in squares.items()}
    boxes |= {f"caption:{n}": cm_box(t.get_window_extent(renderer)) for n, t in items["captions"].items()}
    boxes |= {f"arrow:{n}": cm_box(a.get_window_extent(renderer)) for n, a in items["arrows"].items()}
    boxes |= {f"label:{n}": cm_box(t.get_window_extent(renderer)) for n, t in items["arrow_labels"].items()}
    boxes["colorbar"] = cm_box(cax.get_tightbbox(renderer))
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

    # the scale label sits on a1, just above the bar, without touching it
    a1 = items["axes"]["a1"]
    label = items["extras"]["a1"]["scale_text"].get_window_extent(renderer)
    x0, x1, y0, y1 = items["extras"]["a1"]["scale_bar_um"]
    bar = Bbox(a1.transData.transform([[x0, y1], [x1, y0]]))
    assert a1.get_window_extent(renderer).contains(label.x0, label.y0)
    assert a1.get_window_extent(renderer).contains(label.x1, label.y1)
    assert label.y0 > bar.y1 and not label.overlaps(bar)

    # the same µm mapping in all squares: the centroids land on the same pixels up to
    # each square's offset
    ref = None
    for ax in items["axes"].values():
        px = ax.transData.transform(r["centroids"]) - ax.get_window_extent(renderer).p0
        ref = px if ref is None else ref
        assert np.allclose(px, ref, atol=1e-6)

    # the query nucleus: circled in a1 and a3, filled in a2, inside every square
    q = r["all_centroids"][r["query"]]
    assert X_UM[0] < q[0] < X_UM[1] and Y_UM[0] < q[1] < Y_UM[1]
    for name in ("a1", "a3"):
        assert np.allclose(items["extras"][name]["marker"].center, q)
    return {
        "size_cm": [round(width, 3), round(height, 3)],
        "square_cm": round(side, 4),
        "gap_cm": round(items["gap"], 4),
        "smallest_clearances_cm": dict(sorted(clearances.items(), key=lambda kv: kv[1])[:6]),
    }


def data_checks(items: dict, r: dict) -> dict:
    n = len(r["region"])
    axes = items["axes"]
    tokens = axes["a2"].collections  # grey outlines, centroids, neighbour outlines, query
    counts = {
        "a1_outlines": len(axes["a1"].collections[0].get_paths()),
        "a2_outlines": len(tokens[0].get_paths()),
        "a2_centroids": len(tokens[1].get_offsets()),
        "a3_cells": len(axes["a3"].collections[0].get_paths()),
    }
    assert all(c == n for c in counts.values()), counts
    assert len(tokens[2].get_paths()) == K - 1 and len(tokens[3].get_paths()) == 1
    assert len(axes["a2"].lines) == K - 1
    return {"nuclei_drawn": counts, "a2_neighbours": K - 1, "a2_lines": len(axes["a2"].lines)}


def save_squares(r: dict, side: float) -> None:
    """Each square on its own, at its size in the panel (no caption, no arrows)."""
    for name in SQUARES:
        fig = plt.figure(figsize=(side * CM, side * CM), facecolor="white")
        DRAW[name](fig.add_axes((0, 0, 1, 1)), r)
        fig.savefig(OUT_DIR / f"{FILES[name]}.pdf", facecolor="white")
        fig.savefig(OUT_DIR / f"{FILES[name]}.png", dpi=600, facecolor="white")
        plt.close(fig)


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
    panel, items = draw_panel(r)
    checks = data_checks(items, r)
    checks["panel"] = panel_checks(panel, items, r)

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
    print(json.dumps(checks, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.savefig(OUT_DIR / "panel_a.pdf")  # at print size, no bbox adjustment
    panel.savefig(OUT_DIR / "panel_a.png", dpi=DPI_PNG)
    save_squares(r, items["side"])
    (OUT_DIR / "checks.json").write_text(json.dumps(checks, indent=2))

    if args.log:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with mlflow.start_run(run_id=PANEL_RUN_ID) as run:
            mlflow.set_tags(
                {
                    "git_commit": commit,
                    "panel_layout": "one row of three squares",
                    "panel_size_cm": "x".join(f"{v:.2f}" for v in checks["panel"]["size_cm"]),
                }
            )
            mlflow.log_artifacts(str(OUT_DIR), artifact_path="arch_panel_a")
            mlflow.log_artifact(__file__)
            print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
