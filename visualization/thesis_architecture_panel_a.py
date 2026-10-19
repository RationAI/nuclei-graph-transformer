"""Thesis figure: architecture panel (a), one tissue region through the pipeline.

One row of five equal framed squares, captions above and arrows between them:
(a1) the level-0 H&E of a 250 × 250 µm region of a PANDA Radboud test slide with the
     nucleus polygons, a 50 µm scale bar and the zoom square of a3,
(a2) the polygons and centroids p_i,
(a3) the shape input of the k-NN query nucleus: a 12 × 12 µm H&E zoom with its 64
     polygon vertices, the order-16 EFD reconstruction in µm and the order-1 ellipse
     with its major axis (the orientation ψ); the features themselves are checked
     against the dataset's vector for this nucleus,
(a4) the centroids and the k = 24 neighbourhood of the query nucleus, built like
     `GraphCollator` (sklearn `NearestNeighbors` on the whole slide's centred µm
     positions, the query nucleus itself included, so 23 lines),
(a5) the polygons filled by sigmoid(z_i), the explained model's stored test logit
     (`xai/<run>/activations/test/logits.npy`, no inference is run), with its colorbar.

The panel is drawn at its print size (16 cm wide, 8 pt captions) and exported
unscaled: every square is drawn in µm directly into the panel, so line widths and
marker sizes are in points at print size. Only the H&E is a raster. Each square is also
saved on its own at the same size.

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

from nuclei_graph.data.datasets.crop.prediction import PredictionDataset
from nuclei_graph.data.datasets.nuclei_features import NucleiFeatureExtractor
from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)


MODEL_RUN_ID = "c8f268536be8416f9fe395eea95c458e"  # Radboud, k=24: Polygons + RoVE + k-NN Attn
PANEL_RUN_ID = "21b298d40aa84d268093bd24edbf5665"  # "Thesis figure: architecture panel (a)"
XAI_DIR = Path("/mnt/projects/nuclei_based_wsi_analysis/xai") / MODEL_RUN_ID
SLIDE_ID = "89b62c165838d36fd294a57624db8e4f"  # PANDA Radboud test, Gleason 4+3
X_UM = (275.0, 525.0)
Y_UM = (2465.0, 2715.0)
K = 24  # as in training and in the activation run
EFD_ORDER = 16
ZOOM_UM = 12.0  # side of the a3 zoom, centred on the query nucleus's centroid

SCALE_BAR_UM = 50.0
SCALE_BAR_H_UM = 5.0
SCALE_BAR_PAD_UM = 15.0

# line widths and marker diameters, pt at print size
HE_OUTLINE, HE_OUTLINE_LW = (0.0, 0.0, 0.0, 0.6), 0.3
POLYGON_GREY, POLYGON_LW = "#8c8c8c", 0.6
CENTROID_DARK, CENTROID_D = "#262626", 1.5
CENTROID_GREY, GREY_DOT_D = "#a6a6a6", 2.0
ACCENT_COLOR = "#E4572E"
NEIGHBOUR_D, NEIGHBOUR_LW, QUERY_D = 1.5, 0.8, 4.0
MARKER_LW, MARKER_RADIUS_UM = 1.0, 6.0  # a5 circle; a1 shows the zoom square instead
SCALE_BAR_LW = 0.5
TEXT_STROKE = 1.0  # black outline of the white scale-bar text (half of it shows)
ORDER16_COLOR, ORDER16_LW = "white", 0.7
ORDER1_COLOR, ORDER1_LW = "#9ecae1", 0.6
VERTEX_D = 1.2  # the 64 vertices are ~1.4 pt apart at the zoom's scale
CMAP = "viridis"

# panel layout, cm
CM = 1 / 2.54
PANEL_W = 16.0
EDGE = 0.02  # figure edge to the outermost artists
CBAR_GAP, CBAR_W, CBAR_BLOCK = 0.1, 0.15, 1.1  # the block includes ticks and label
BOTTOM = 0.16  # below the squares: room for the colorbar's lowest tick label
TOP, CAPTION_PAD = 0.03, 0.1
ARROW_LEN = 0.24
MIN_GAP = ARROW_LEN + 2 * 0.06  # arrow plus its clearance to the squares
CAPTION_CLEARANCE = 0.15  # between neighbouring captions
FONT_SIZE, SMALL_FONT_SIZE = 8, 7
LINE_PT = 0.8
BORDER_GREY, BORDER_LW = "#999999", 0.5
ARROW_STYLE = "-|>,head_length=3,head_width=1.5"
DPI_PNG = 300

SQUARES = ("a1", "a2", "a3", "a4", "a5")
CAPTIONS = {
    "a1": "H&E with\nsegmented nuclei",
    "a2": "Polygons and\ncentroids $p_i$",
    "a3": "Shape: EFD of order 16\n+ size and orientation",
    "a4": "$k$-NN attention\n($L$ layers)",
    "a5": "Nucleus logits $z_i$",
}
FILES = {
    "a1": "a1_he",
    "a2": "a2_points",
    "a3": "a3_efd",
    "a4": "a4_knn",
    "a5": "a5_heatmap",
}
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
        "query_polygon": polygons_um[query],
        "all_centroids": centroids_um,
        "logits": logits,
        "he": he,
        "he_origin_px": (lx, ly),
        "he_extent": he_extent,
    }


def efd_curve(coeffs: NDArray[np.float64], n: int = 1024) -> NDArray[np.float64]:
    """Contour of EFD coefficients (orders x [a, b, c, d]), without the DC term."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    phase = 2 * np.pi * t * np.arange(1, len(coeffs) + 1)[None]
    cos, sin = np.cos(phase), np.sin(phase)
    return np.stack(
        [cos @ coeffs[:, 0] + sin @ coeffs[:, 1], cos @ coeffs[:, 2] + sin @ coeffs[:, 3]],
        axis=1,
    )


def shape_embedding(r: dict) -> dict:
    """The k-NN query nucleus's EFD shape, in µm, and the features the model receives.

    The features are `NucleiFeatureExtractor.get_efd_features` of the level-0 polygon
    (order 16; normalised for the starting point, rotated by -ψ and divided by the scale
    s): 64 coefficients, log s, cos 2ψ, sin 2ψ. Checked against the dataset's own
    feature vector for this nucleus (`PredictionDataset`, the full-slide sample). The
    contour is the same order-16 EFD before the rotation and scale normalisation, plus
    the DC term (the arc-length weighted mean of the closed polygon), so it is in µm.
    """
    slide, q = r["slide"], r["query"]
    polygon = np.asarray(r["nuclei"]["polygon"].iloc[q], dtype=np.float32).reshape(-1, 2)
    extractor = NucleiFeatureExtractor()
    extractor.efd_order = EFD_ORDER
    features = extractor.get_efd_features(polygon[None], slide.mpp_x, slide.mpp_y)[0]

    dataset = PredictionDataset(
        metadata=pd.DataFrame([slide]).reset_index(drop=True),
        embedding_mode="efd",
        efd_order=EFD_ORDER,
    )
    sample = dataset[0]
    assert sample["metadata"]["nuclei_ids"][q] == r["nuclei"]["id"].iloc[q]
    max_abs_diff = float(np.abs(features - sample["features"][q].numpy()).max())
    assert max_abs_diff < 1e-6, max_abs_diff

    # the steps of get_efd_features
    contour = (polygon * np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)).astype(np.float64)
    raw = elliptic_fourier_descriptors(contour[None], EFD_ORDER)  # start point normalised
    coeffs, psi = normalize_efd_for_rotation(raw.copy())
    coeffs, scale = normalize_efd_for_scale(coeffs)
    psi, scale = float(psi[0, 0]), float(scale[0, 0])
    assert np.array_equal(coeffs[0].ravel().astype(np.float32), features[: 4 * EFD_ORDER])
    assert np.isclose(features[4 * EFD_ORDER], np.log(scale + 1e-6))

    closed = np.vstack([contour, contour[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    centre = ((closed[:-1] + closed[1:]) / 2 * lengths[:, None]).sum(0) / lengths.sum()
    order16 = efd_curve(raw[0]) + centre
    order1 = efd_curve(raw[0, :1]) + centre
    axis = centre + scale * np.array([[-np.cos(psi), -np.sin(psi)], [np.cos(psi), np.sin(psi)]])

    distance, _ = KDTree(efd_curve(raw[0], n=16384) + centre).query(contour)
    return {
        "features": features,
        "vertices": contour,
        "order16": order16,
        "order1": order1,
        "major_axis": axis,
        "psi_deg": float(np.degrees(psi)),
        "scale_um": scale,
        "recon_rms_um": float(np.sqrt(np.mean(distance**2))),
        "recon_max_um": float(distance.max()),
        "max_abs_diff_dataset": max_abs_diff,
    }


def frame(ax: plt.Axes, x: tuple[float, float], y: tuple[float, float]) -> None:
    """Square axes over x × y µm, y down as in the slide, with a thin grey frame."""
    ax.set_xlim(*x)
    ax.set_ylim(y[1], y[0])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER_GREY)
        spine.set_linewidth(BORDER_LW)


def dots(ax: plt.Axes, xy: NDArray, diameter_pt: float, **kwargs):
    return ax.scatter(*np.asarray(xy).T, s=diameter_pt**2, lw=0, **kwargs)


def outlines(polygons: NDArray, color, lw: float) -> PolyCollection:
    return PolyCollection(polygons, closed=True, facecolors="none", edgecolors=[color], linewidths=lw)


def zoom_box(r: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    c = r["all_centroids"][r["query"]]
    return (c[0] - ZOOM_UM / 2, c[0] + ZOOM_UM / 2), (c[1] - ZOOM_UM / 2, c[1] + ZOOM_UM / 2)


def scale_bar_um() -> tuple[float, float, float, float]:
    """Outer edges (x0, x1, y_top, y_bottom) of the scale bar, µm."""
    x0 = X_UM[0] + SCALE_BAR_PAD_UM
    y1 = Y_UM[1] - SCALE_BAR_PAD_UM
    return x0, x0 + SCALE_BAR_UM, y1 - SCALE_BAR_H_UM, y1


def draw_he(ax: plt.Axes, r: dict) -> dict:
    frame(ax, X_UM, Y_UM)
    ax.imshow(r["he"], extent=r["he_extent"], interpolation="antialiased")
    ax.add_collection(outlines(r["polygons"], HE_OUTLINE, HE_OUTLINE_LW))

    # white bar with a black outline drawn inside it, so its outer length is 50 µm
    um_per_pt = (X_UM[1] - X_UM[0]) / (ax.get_window_extent().width * 72 / ax.figure.dpi)
    inset = SCALE_BAR_LW / 2 * um_per_pt
    x0, x1, y0, y1 = scale_bar_um()
    ax.add_patch(Rectangle((x0 + inset, y0 + inset), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset,
                           facecolor="white", edgecolor="black", lw=SCALE_BAR_LW))
    text = ax.text(x1 + 5.0, (y0 + y1) / 2, f"{SCALE_BAR_UM:g} µm", ha="left", va="center",
                   fontsize=SMALL_FONT_SIZE, color="white",
                   path_effects=[pe.withStroke(linewidth=TEXT_STROKE, foreground="black")])

    (zx0, zx1), (zy0, zy1) = zoom_box(r)
    square = ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                                    edgecolor=ACCENT_COLOR, lw=MARKER_LW, zorder=5))
    return {"scale_text": text, "scale_bar_um": (x0, x1, y0, y1), "zoom_square": square}


def draw_points(ax: plt.Axes, r: dict) -> dict:
    frame(ax, X_UM, Y_UM)
    ax.add_collection(outlines(r["polygons"], POLYGON_GREY, POLYGON_LW))
    dots(ax, r["centroids"], CENTROID_D, c=CENTROID_DARK)
    return {}


def draw_zoom(ax: plt.Axes, r: dict, shape: dict) -> dict:
    """The H&E around the query nucleus (the same pixels and µm mapping as a1)."""
    x, y = zoom_box(r)
    frame(ax, x, y)
    mpp = np.array([r["slide"].mpp_x, r["slide"].mpp_y])
    lx, ly = r["he_origin_px"]
    # the H&E pixels covering the zoom, with the extent convention of load_region
    c0 = np.floor(np.array([x[0], y[0]]) / mpp + 0.5).astype(int) - 1
    c1 = np.ceil(np.array([x[1], y[1]]) / mpp + 0.5).astype(int) + 1
    crop = r["he"][c0[1] - ly : c1[1] - ly, c0[0] - lx : c1[0] - lx]
    extent = ((c0[0] - 0.5) * mpp[0], (c1[0] - 0.5) * mpp[0],
              (c1[1] - 0.5) * mpp[1], (c0[1] - 0.5) * mpp[1])
    ax.imshow(crop, extent=extent, interpolation="nearest")

    ax.plot(*np.vstack([shape["order1"], shape["order1"][:1]]).T, color=ORDER1_COLOR,
            lw=ORDER1_LW, ls=(0, (3, 2)), zorder=2)
    ax.plot(*shape["major_axis"].T, color=ORDER1_COLOR, lw=ORDER1_LW, ls=(0, (3, 2)), zorder=2)
    dots(ax, shape["vertices"], VERTEX_D, c=ACCENT_COLOR, zorder=3)
    ax.plot(*np.vstack([shape["order16"], shape["order16"][:1]]).T, color=ORDER16_COLOR,
            lw=ORDER16_LW, zorder=4)
    return {"crop": crop, "crop_extent": extent, "crop_origin_px": tuple(c0)}


def draw_knn(ax: plt.Axes, r: dict) -> dict:
    frame(ax, X_UM, Y_UM)
    in_knn = np.isin(r["region"], r["knn"])
    dots(ax, r["centroids"][~in_knn], GREY_DOT_D, c=CENTROID_GREY)
    centre = r["all_centroids"][r["query"]]
    others = r["all_centroids"][r["knn"][1:]]
    for p in others:
        ax.plot(*np.stack([centre, p]).T, color=ACCENT_COLOR, lw=NEIGHBOUR_LW, zorder=2)
    dots(ax, others, NEIGHBOUR_D, c=ACCENT_COLOR, zorder=3)
    ax.scatter(*centre, s=QUERY_D**2, c=ACCENT_COLOR, edgecolors="white", linewidths=0.4, zorder=4)
    return {}


def draw_heatmap(ax: plt.Axes, r: dict) -> dict:
    frame(ax, X_UM, Y_UM)
    cells = PolyCollection(r["polygons"], closed=True, cmap=CMAP,
                           norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                           edgecolors="none", antialiased=True)
    cells.set_array(r["prob"])
    ax.add_collection(cells)
    ax.add_patch(plt.Circle(r["all_centroids"][r["query"]], MARKER_RADIUS_UM, fill=False,
                            edgecolor=ACCENT_COLOR, lw=MARKER_LW, zorder=5))
    return {}


def draw_square(name: str, ax: plt.Axes, r: dict, shape: dict) -> dict:
    if name == "a3":
        return draw_zoom(ax, r, shape)
    return {"a1": draw_he, "a2": draw_points, "a4": draw_knn, "a5": draw_heatmap}[name](ax, r)


def arrow(lay: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> FancyArrowPatch:
    patch = FancyArrowPatch(start, end, arrowstyle=ARROW_STYLE, mutation_scale=1,
                            lw=LINE_PT, color="black", shrinkA=0, shrinkB=0)
    lay.add_patch(patch)
    return patch


def caption_sizes() -> dict[str, tuple[float, float]]:
    """Width and height (cm) of every caption at its print size."""
    fig = plt.figure(figsize=(PANEL_W * CM, 3 * CM))
    renderer = fig.canvas.get_renderer()
    sizes = {}
    for name, text in CAPTIONS.items():
        assert text.count("\n") <= 1, f"{name}: more than 2 lines"
        extent = fig.text(0, 0, text, ha="center", ma="center", fontsize=FONT_SIZE).get_window_extent(renderer)
        sizes[name] = (extent.width / fig.dpi * 2.54, extent.height / fig.dpi * 2.54)
    plt.close(fig)
    return sizes


def layout(captions: dict[str, tuple[float, float]]) -> dict:
    """Square side and gap (cm): five equal squares, four equal gaps, the colorbar block.

    The gap is the smallest one (>= MIN_GAP) at which neighbouring captions keep
    CAPTION_CLEARANCE and the first caption stays inside the figure.
    """
    widths = [captions[n][0] for n in SQUARES]
    lead = 0.0
    for gap in np.arange(MIN_GAP, 2.0, 0.005):
        side = (PANEL_W - 2 * EDGE - lead - CBAR_BLOCK - 4 * gap) / 5
        lead = max(0.0, (widths[0] - side) / 2)
        side = (PANEL_W - 2 * EDGE - lead - CBAR_BLOCK - 4 * gap) / 5
        if all((a + b) / 2 + CAPTION_CLEARANCE <= side + gap for a, b in zip(widths[:-1], widths[1:], strict=True)):
            return {"side": side, "gap": float(gap), "left0": EDGE + lead}
    raise ValueError("captions do not fit")


def draw_panel(r: dict, shape: dict) -> tuple[plt.Figure, dict]:
    captions = caption_sizes()
    geo = layout(captions)
    side, gap = geo["side"], geo["gap"]
    caption_h = max(h for _, h in captions.values())
    height = TOP + caption_h + CAPTION_PAD + side + BOTTOM
    bottom = BOTTOM
    mid = bottom + side / 2

    fig = plt.figure(figsize=(PANEL_W * CM, height * CM), facecolor="white")
    lay = fig.add_axes((0, 0, 1, 1))  # layout axes in cm, under the squares
    lay.set_xlim(0, PANEL_W)
    lay.set_ylim(0, height)
    lay.axis("off")

    def cm_rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / PANEL_W, y / height, w / PANEL_W, h / height)

    items: dict = {"axes": {}, "captions": {}, "arrows": {}, "extras": {}, "side": side,
                   "gap": gap, "size_cm": (PANEL_W, height)}
    left = {n: geo["left0"] + i * (side + gap) for i, n in enumerate(SQUARES)}
    for name in SQUARES:
        ax = fig.add_axes(cm_rect(left[name], bottom, side, side))
        items["axes"][name] = ax
        items["extras"][name] = draw_square(name, ax, r, shape)
        items["captions"][name] = lay.text(left[name] + side / 2, bottom + side + CAPTION_PAD,
                                           CAPTIONS[name], ha="center", va="bottom", ma="center",
                                           fontsize=FONT_SIZE)
    for a, b in zip(SQUARES[:-1], SQUARES[1:], strict=True):
        centre = left[a] + side + gap / 2
        items["arrows"][f"{a}->{b}"] = arrow(lay, (centre - ARROW_LEN / 2, mid), (centre + ARROW_LEN / 2, mid))

    cax = fig.add_axes(cm_rect(left["a5"] + side + CBAR_GAP, bottom, CBAR_W, side))
    mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, 1.0), cmap=CMAP)
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, width=0.5, length=2, pad=1.5)
    cbar.outline.set_linewidth(BORDER_LW)
    cbar.set_label(r"$\sigma(z_i)$", fontsize=SMALL_FONT_SIZE, labelpad=2)
    cbar.solids.set_rasterized(False)  # matplotlib rasterizes long colorbars by default
    cbar.solids.set_edgecolor("face")
    items["colorbar"] = cax
    items["left"] = left
    return fig, items


def panel_checks(fig: plt.Figure, items: dict, r: dict, shape: dict) -> dict:
    """Equal aligned squares, equal gaps and centred arrows; no overlaps; text sizes;
    the zoom square in a1 and the zoom in a3 registered with the H&E."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_cm = fig.dpi_scale_trans.inverted() + mpl.transforms.Affine2D().scale(2.54)

    def cm_box(bbox: Bbox) -> Bbox:
        return bbox.transformed(to_cm)

    width, height = items["size_cm"]
    w_in, h_in = fig.get_size_inches()
    assert np.isclose(w_in * 2.54, PANEL_W) and np.isclose(h_in * 2.54, height)

    squares = {n: cm_box(ax.get_window_extent(renderer)) for n, ax in items["axes"].items()}
    side = items["side"]
    for name, b in squares.items():
        assert np.isclose(b.width, side) and np.isclose(b.height, side), f"{name}: {b}"
        assert np.isclose(b.y0, squares["a1"].y0), f"{name} not aligned"
    gaps = [squares[b].x0 - squares[a].x1 for a, b in zip(SQUARES[:-1], SQUARES[1:], strict=True)]
    assert np.allclose(gaps, gaps[0]), gaps
    for name, a in items["arrows"].items():
        p, q = name.split("->")
        (sx, sy), (ex, ey) = a._posA_posB  # the arrow's own endpoints, cm
        centre = (squares[p].x1 + squares[q].x0) / 2
        assert abs((sx + ex) / 2 - centre) < 1e-6, f"arrow {name} not centred"
        assert abs(sy - (squares[p].y0 + side / 2)) < 1e-6 and sy == ey, f"arrow {name} not at mid-height"

    # text sizes at print size
    for name, t in items["captions"].items():
        assert t.get_fontsize() == FONT_SIZE and t.get_text().count("\n") <= 1, name
    cax = items["colorbar"]
    assert all(t.get_fontsize() == SMALL_FONT_SIZE for t in cax.get_yticklabels())
    assert cax.yaxis.label.get_fontsize() == SMALL_FONT_SIZE
    assert items["extras"]["a1"]["scale_text"].get_fontsize() == SMALL_FONT_SIZE

    boxes: dict[str, Bbox] = {f"square:{n}": b for n, b in squares.items()}
    boxes |= {f"caption:{n}": cm_box(t.get_window_extent(renderer)) for n, t in items["captions"].items()}
    boxes |= {f"arrow:{n}": cm_box(a.get_window_extent(renderer)) for n, a in items["arrows"].items()}
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

    # the scale label sits on a1, right of the bar, without touching it
    a1 = items["axes"]["a1"]
    label = items["extras"]["a1"]["scale_text"].get_window_extent(renderer)
    x0, x1, y0, y1 = items["extras"]["a1"]["scale_bar_um"]
    bar = Bbox(a1.transData.transform([[x0, y1], [x1, y0]]))
    assert a1.get_window_extent(renderer).contains(label.x0, label.y0)
    assert a1.get_window_extent(renderer).contains(label.x1, label.y1)
    assert label.x0 > bar.x1 and not label.overlaps(bar)

    # same µm mapping in the four region squares (a1, a2, a4, a5): the centroids land on
    # the same pixels up to each square's offset
    ref = None
    for name in ("a1", "a2", "a4", "a5"):
        ax = items["axes"][name]
        px = ax.transData.transform(r["centroids"]) - ax.get_window_extent(renderer).p0
        ref = px if ref is None else ref
        assert np.allclose(px, ref, atol=1e-6), f"{name} not aligned with a1"

    # zoom square in a1 = the limits of a3; the a3 H&E is the a1 pixels with a1's extent rule
    square = items["extras"]["a1"]["zoom_square"]
    a3 = items["axes"]["a3"]
    sx, sy = square.get_x(), square.get_y()
    assert np.allclose([sx, sx + square.get_width()], a3.get_xlim())
    assert np.allclose([sy + square.get_height(), sy], a3.get_ylim())
    zoom = items["extras"]["a3"]
    lx, ly = r["he_origin_px"]
    c0 = zoom["crop_origin_px"]
    assert np.array_equal(zoom["crop"][0, 0], r["he"][c0[1] - ly, c0[0] - lx])
    mpp = (r["slide"].mpp_x, r["slide"].mpp_y)
    assert np.isclose(zoom["crop_extent"][0], (c0[0] - 0.5) * mpp[0])
    assert np.isclose(r["he_extent"][0], (lx - 0.5) * mpp[0])
    xl, yl = a3.get_xlim(), a3.get_ylim()
    for curve in (shape["order16"], shape["vertices"]):
        assert ((curve[:, 0] > xl[0]) & (curve[:, 0] < xl[1])).all()
        assert ((curve[:, 1] > yl[1]) & (curve[:, 1] < yl[0])).all()

    return {
        "size_cm": [round(width, 3), round(height, 3)],
        "square_cm": round(side, 4),
        "gap_cm": round(items["gap"], 4),
        "smallest_clearances_cm": dict(sorted(clearances.items(), key=lambda kv: kv[1])[:6]),
    }


def data_checks(items: dict, r: dict) -> dict:
    n = len(r["region"])
    axes = items["axes"]
    counts = {
        "a1": [len(axes["a1"].collections[0].get_paths())],
        "a2": [len(axes["a2"].collections[0].get_paths()), len(axes["a2"].collections[1].get_offsets())],
        "a4": [sum(len(c.get_offsets()) for c in axes["a4"].collections)],
        "a5": [len(axes["a5"].collections[0].get_paths())],
    }
    assert all(c == n for v in counts.values() for c in v), counts
    a4_lines = len(axes["a4"].lines)
    assert a4_lines == K - 1
    return {"nuclei_drawn": counts, "a4_lines": a4_lines}


def save_squares(r: dict, shape: dict, side: float) -> None:
    """Each square on its own, at its size in the panel (no caption, no arrows)."""
    for name in SQUARES:
        fig = plt.figure(figsize=(side * CM, side * CM), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        draw_square(name, ax, r, shape)
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
    shape = shape_embedding(r)
    panel, items = draw_panel(r, shape)
    checks = data_checks(items, r)
    checks["panel"] = panel_checks(panel, items, r, shape)

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
        "shape_embedding": {
            "nucleus_id": str(r["nuclei"]["id"].iloc[q]),
            "efd_order": EFD_ORDER,
            "zoom_um": ZOOM_UM,
            "psi_deg": shape["psi_deg"],
            "scale_um": shape["scale_um"],
            "features": [float(v) for v in shape["features"]],
            "max_abs_diff_to_dataset": shape["max_abs_diff_dataset"],
            "order16_rms_to_polygon_um": shape["recon_rms_um"],
            "order16_max_distance_um": shape["recon_max_um"],
        },
    }
    print(json.dumps(checks, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.savefig(OUT_DIR / "panel_a.pdf")  # at print size, no bbox adjustment
    panel.savefig(OUT_DIR / "panel_a.png", dpi=DPI_PNG)
    save_squares(r, shape, items["side"])
    (OUT_DIR / "checks.json").write_text(json.dumps(checks, indent=2))

    if args.log:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with mlflow.start_run(run_id=PANEL_RUN_ID) as run:
            mlflow.set_tags(
                {
                    "git_commit": commit,
                    "panel_layout": "one row of five squares",
                    "panel_size_cm": "x".join(f"{v:.2f}" for v in checks["panel"]["size_cm"]),
                }
            )
            mlflow.log_artifacts(str(OUT_DIR), artifact_path="arch_panel_a")
            mlflow.log_artifact(__file__)
            print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
