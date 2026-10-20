"""Thesis figure: architecture panel (a), one tissue window through the pipeline.

One row, left to right, top-aligned, of a PANDA Radboud test slide:
(A) the level-0 H&E of a square window with the nucleus polygons, the highlighted
    nucleus outlined in orange and a 25 µm scale bar; dashed callout lines lead from
    the nucleus to B,
(B) the highlighted nucleus's shape encoding in µm and its true orientation: the 64
    polygon vertices, the order-16 EFD reconstruction, the order-1 ellipse with its
    semi-major axis s (solid), semi-minor axis (dotted) and the angle ψ from the
    horizontal (dotted reference, arc); the EFD is `NucleiFeatureExtractor.
    get_efd_features` (order 16, checked against the dataset's feature vector), with
    the normalisation inverted for display (scaled by s, rotated by ψ, shifted to the
    EFD centre),
(C) the nuclei as tokens: all polygons and centroids, the highlighted nucleus filled
    and its k - 1 = 23 neighbours outlined and joined to it, built like
    `GraphCollator` (sklearn `NearestNeighbors` on the whole slide's centred µm
    positions, the nucleus itself included),
(D) the polygons filled by sigmoid(z_i), the explained model's stored test logit
    (`xai/<run>/activations/test/logits.npy`, no inference is run), with its colorbar.

The highlighted nucleus (`choose_nucleus`) is elongated, mid-sized, cleanly segmented
and at a gland-stroma boundary: order-1 axis ratio in RATIO_RANGE, s between the
slide's S_PERCENTILES, solidity >= MIN_SOLIDITY, no overlap with other polygons, at
least MIN_KNN_EACH of its 23 neighbours on each side of sigmoid(z) = 0.5, and a window
as below. For a readable ψ in B, |ψ| in PSI_RANGE_DEG and the ψ arc (ARC_RADIUS · s)
at least MIN_ARC_CLEARANCE · s inside the ellipse. Ranked by the kNN balance, the
axis ratio's distance to RATIO_TARGET, the window's balance and its margin.

A, C and D show the same window and only the nuclei whose centroid lies in it: the
WINDOW_UM square that keeps the nucleus at least MIN_MARGIN_UM from every edge, holds
its 23 neighbours and at least MIN_CLASS_SHARE of nuclei on each side of sigmoid(z) =
0.5; of those, the one with the largest margin, then the most balanced.

ψ is the angle of the semi-major axis in image coordinates (x right, y down), taken
modulo 180° (the model sees cos 2ψ, sin 2ψ) in (-90°, 90°]: positive is clockwise on
screen, i.e. the axis points down to the right.

The panel is drawn at its print size (16 cm wide, fonts in pt) and exported unscaled:
the elements are drawn in µm directly into the panel, so line widths and marker sizes
are in points at print size. Only the H&E is a raster. Each element is also saved on
its own at the same size.

Usage:
    uv run -m visualization.thesis_architecture_panel_a [--log]

`--log` logs the elements, the panel, checks and this script to the panel's MLflow run
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
import shapely
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
K = 24  # as in training and in the activation run
EFD_ORDER = 16

# the highlighted nucleus
RATIO_RANGE = (2.0, 3.0)  # semi-major / semi-minor of the order-1 ellipse
RATIO_TARGET = 2.5
S_PERCENTILES = (40, 90)  # of the slide's semi-major axes s
MIN_SOLIDITY = 0.9
OVERLAP_UM2 = 1e-6  # intersection area that counts as an overlap
MIN_KNN_EACH = 5  # neighbours with sigmoid(z) >= 0.5 and < 0.5, each
PSI_RANGE_DEG = (20.0, 50.0)  # |ψ|, so the angle reads in B
ARC_RADIUS = 0.4  # of s
MIN_ARC_CLEARANCE = 0.1  # of s, from the ψ arc to the ellipse
N_CANDIDATES = 5  # reported

WINDOW_UM = 130.0
MIN_MARGIN_UM = 40.0  # highlighted nucleus to every window edge
MIN_CLASS_SHARE = 0.25  # of the window's nuclei on each side of sigmoid(z) = 0.5
WINDOW_STEP_UM = 1.0

SCALE_BAR_UM = 25.0
SCALE_BAR_H_UM = 2.5
SCALE_BAR_PAD_UM = 8.0

# colours; line widths and marker diameters in pt at print size
ORANGE = "#E6550D"
HE_OUTLINE, HE_OUTLINE_LW = (0.0, 0.0, 0.0, 0.5), 0.5
QUERY_LW = 1.0
TOKEN_GREY, TOKEN_LW = "#9A9A9A", 0.5
CENTROID_GREY, CENTROID_D = "#888888", 1.0
NEIGHBOUR_LW = 0.8
EDGE_LW, EDGE_ALPHA = 0.5, 0.7
VERTEX_COLOR, VERTEX_D = "#333333", 1.5
ORDER16_COLOR, ORDER16_LW = "#1F5FA8", 1.1
ORDER1_COLOR, ORDER1_LW = "#7FB2E5", 0.9
AXIS_LW = 0.6
CALLOUT_COLOR, CALLOUT_LW = "#999999", 0.5
SCALE_BAR_LW = 0.5  # black outline of the white bar
TEXT_STROKE = 1.0  # black outline of white text (half of it shows)
CMAP = "viridis"
B_PAD = 0.2  # padding around the shape in B, of its extent

# panel layout, cm
CM = 1 / 2.54
PANEL_W = 16.0
EDGE = 0.02  # figure edge to the outermost artists
B_W = 2.6
AB_GAP = 0.7  # room for the callout lines
MIN_LABEL_CLEARANCE = 0.1  # arrow labels to the elements
CBAR_GAP, CBAR_W, CBAR_BLOCK = 0.12, 0.18, 1.16  # the block includes ticks and label
TOP, CAPTION_PAD, BOTTOM = 0.03, 0.12, 0.16  # BOTTOM: room for the lowest colorbar tick
ARROW_LEN = 0.6
ARROW_LABEL_PAD = 0.08  # arrow to the bottom of its label
FONT_SIZE, SMALL_FONT_SIZE = 8, 7
LINE_PT = 0.8
BORDER_GREY, BORDER_LW = "#999999", 0.5
ARROW_STYLE = "-|>,head_length=3,head_width=1.5"
DPI_PNG = 300

ELEMENTS = ("A", "B", "C", "D")
CAPTIONS = {
    "A": "H&E, segmented nuclei",
    "B": "Shape encoding (EFD)",
    "C": "Nuclei as tokens",
    "D": "Nucleus logits $z_i$",
}
ARROW_LABELS = {"B->C": "$\\mathbf{x}_i$, $p_i$", "C->D": "$L$ layers"}
B_TEXT = "EFD (64 coeff.)\n+ log $s$ + $\\psi$"
FILES = {"A": "a_he", "B": "b_efd", "C": "c_tokens", "D": "d_logits"}
OUT_DIR = Path("figures/arch_panel_a")


def choose_window(centroids: NDArray, prob: NDArray, q: int, knn: NDArray) -> dict | None:
    """The WINDOW_UM square around nucleus q, or None.

    q at least MIN_MARGIN_UM from every edge, its k-NN inside, at least MIN_CLASS_SHARE of
    the nuclei on each side of sigmoid(z) = 0.5; then the largest margin, then the most
    balanced.
    """
    cq = centroids[q]
    near = np.flatnonzero(np.abs(centroids - cq).max(axis=1) < WINDOW_UM)
    c, pr = centroids[near], prob[near]
    knn_c = centroids[knn]
    best = None
    offsets = np.arange(MIN_MARGIN_UM, WINDOW_UM - MIN_MARGIN_UM + 1e-9, WINDOW_STEP_UM)
    for dx in offsets:
        for dy in offsets:
            x0, y0 = cq[0] - dx, cq[1] - dy
            if not ((knn_c >= [x0, y0]) & (knn_c < [x0 + WINDOW_UM, y0 + WINDOW_UM])).all():
                continue
            inside = (c[:, 0] >= x0) & (c[:, 0] < x0 + WINDOW_UM) & (c[:, 1] >= y0) & (c[:, 1] < y0 + WINDOW_UM)
            low = float((pr[inside] < 0.5).mean())
            if min(low, 1 - low) < MIN_CLASS_SHARE:
                continue
            margin = float(min(dx, dy, WINDOW_UM - dx, WINDOW_UM - dy))
            key = (margin, min(low, 1 - low))
            if best is None or key > (best["margin"], best["balance"]):
                best = {"x0": float(x0), "y0": float(y0), "margin": margin, "balance": key[1],
                        "share_below_0.5": low, "n": int(inside.sum())}
    return best


def choose_nucleus(nuclei: pd.DataFrame, centroids: NDArray, polygons: NDArray, prob: NDArray,
                   knn: NDArray) -> tuple[int, pd.DataFrame]:
    """The highlighted nucleus (see the module docstring) and the top candidates."""
    coeffs = elliptic_fourier_descriptors(polygons.astype(np.float64), 1)
    coeffs, psi = normalize_efd_for_rotation(coeffs)
    coeffs, scale = normalize_efd_for_scale(coeffs)
    ratio = 1.0 / np.abs(coeffs[:, 0, 3])
    scale = scale[:, 0]
    psi_deg = np.degrees(0.5 * np.arctan2(np.sin(2 * psi[:, 0]), np.cos(2 * psi[:, 0])))
    s_lo, s_hi = np.percentile(scale, S_PERCENTILES)
    shapes = shapely.polygons(polygons)
    solidity = shapely.area(shapes) / shapely.area(shapely.convex_hull(shapes))
    high = (prob[knn[:, 1:]] >= 0.5).sum(axis=1)
    low = (K - 1) - high

    # ψ arc of radius ARC_RADIUS · s from the horizontal to the axis, against the ellipse
    theta = np.radians(np.linspace(0.0, 1.0, 64))[None] * np.abs(psi_deg)[:, None]
    ellipse_r = 1.0 / np.sqrt(np.cos(theta) ** 2 + (ratio[:, None] * np.sin(theta)) ** 2)
    arc_clearance = ellipse_r.min(axis=1) - ARC_RADIUS

    hard = (
        (ratio >= RATIO_RANGE[0]) & (ratio <= RATIO_RANGE[1])
        & (scale >= s_lo) & (scale <= s_hi)
        & (solidity >= MIN_SOLIDITY)
        & (high >= MIN_KNN_EACH) & (low >= MIN_KNN_EACH)
    )
    readable = (
        (np.abs(psi_deg) >= PSI_RANGE_DEG[0]) & (np.abs(psi_deg) <= PSI_RANGE_DEG[1])
        & (arc_clearance >= MIN_ARC_CLEARANCE)
    )
    tree = shapely.STRtree(shapes)
    rows = []
    for q in np.flatnonzero(hard & readable):
        touching = [j for j in tree.query(shapes[q]) if j != q]
        if any(shapely.intersection(shapes[q], shapes[j]).area > OVERLAP_UM2 for j in touching):
            continue
        window = choose_window(centroids, prob, q, knn[q])
        if window is None:
            continue
        rows.append({
            "index": int(q), "nucleus_id": str(nuclei["id"].iloc[q]),
            "centroid_um": [round(float(v), 2) for v in centroids[q]],
            "axis_ratio": float(ratio[q]), "s_um": float(scale[q]), "psi_deg": float(psi_deg[q]),
            "solidity": float(solidity[q]), "sigmoid_z": float(prob[q]),
            "knn_high": int(high[q]), "knn_low": int(low[q]), "arc_clearance_s": float(arc_clearance[q]),
            "window_margin_um": window["margin"], "window_share_below_0.5": window["share_below_0.5"],
            "window_n": window["n"],
        })
    table = pd.DataFrame(rows)
    assert len(table), "no nucleus meets the criteria"
    table["knn_balance"] = table[["knn_high", "knn_low"]].min(axis=1)
    table["ratio_distance"] = (table["axis_ratio"] - RATIO_TARGET).abs()
    table["window_imbalance"] = (table["window_share_below_0.5"] - 0.5).abs()
    table = table.sort_values(
        ["knn_balance", "ratio_distance", "window_imbalance", "window_margin_um"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    table.attrs = {"s_percentiles_um": [float(s_lo), float(s_hi)], "n_hard": int(hard.sum()),
                   "n_hard_and_readable": int((hard & readable).sum()), "n_eligible": len(table)}
    return int(table["index"].iloc[0]), table


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
    prob = 1.0 / (1.0 + np.exp(-logits))

    # as GraphCollator: k-NN over the whole slide, on the dataset's centred positions
    pos = (centroids_um - centroids_um.mean(axis=0)).astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(pos)
    knn_dist_all, knn_all = nbrs.kneighbors(pos)
    query, candidates = choose_nucleus(nuclei, centroids_um, polygons_um, prob, knn_all)
    knn, dist = knn_all[query], knn_dist_all[query]
    assert knn[0] == query and dist[0] == 0.0
    # the neighbourhood is unambiguous (no tie at the k-th distance) and the same with
    # a KD-tree on the uncentred µm positions
    d_next, _ = KDTree(centroids_um).query(centroids_um[query], k=K + 1)
    assert d_next[K] - d_next[K - 1] > 1e-3
    _, knn_tree = KDTree(centroids_um).query(centroids_um[query], k=K)
    assert set(knn_tree) == set(knn)

    window = choose_window(centroids_um, prob, query, knn)
    assert window is not None
    x_um = (window["x0"], window["x0"] + WINDOW_UM)
    y_um = (window["y0"], window["y0"] + WINDOW_UM)
    inside = (
        (centroids_um[:, 0] >= x_um[0]) & (centroids_um[:, 0] < x_um[1])
        & (centroids_um[:, 1] >= y_um[0]) & (centroids_um[:, 1] < y_um[1])
    )
    assert inside[knn].all(), "neighbourhood leaves the window"
    region = np.flatnonzero(inside)

    with OpenSlide(slide.slide_path) as wsi:
        lx, ly = int(x_um[0] / slide.mpp_x) - 2, int(y_um[0] / slide.mpp_y) - 2
        rx, ry = int(np.ceil(x_um[1] / slide.mpp_x)) + 2, int(np.ceil(y_um[1] / slide.mpp_y)) + 2
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
        "x_um": x_um,
        "y_um": y_um,
        "window": window,
        "candidates": candidates,
        "region": region,
        "centroids": centroids_um[region],
        "polygons": polygons_um[region],
        "prob": prob[region],
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


def efd_curve(coeffs: NDArray[np.float64], n: int = 2048) -> NDArray[np.float64]:
    """Contour of EFD coefficients (orders x [a, b, c, d]), without the DC term."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    phase = 2 * np.pi * t * np.arange(1, len(coeffs) + 1)[None]
    cos, sin = np.cos(phase), np.sin(phase)
    return np.stack(
        [cos @ coeffs[:, 0] + sin @ coeffs[:, 1], cos @ coeffs[:, 2] + sin @ coeffs[:, 3]],
        axis=1,
    )


def shape_encoding(r: dict) -> dict:
    """The highlighted nucleus's EFD encoding, and its curves back in µm.

    The features are `NucleiFeatureExtractor.get_efd_features` of the level-0 polygon:
    the order-16 coefficients normalised for the starting point, rotated by -ψ and
    divided by s, then log s, cos 2ψ, sin 2ψ; checked against the dataset's feature
    vector (`PredictionDataset`, full slide). For display the normalisation is
    inverted: scaled by s, rotated by ψ and shifted to the EFD centre (its DC term, the
    arc-length weighted mean of the closed polygon), in µm relative to the centroid.
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

    coeffs = features[: 4 * EFD_ORDER].reshape(EFD_ORDER, 4).astype(np.float64)
    scale = float(np.exp(features[4 * EFD_ORDER])) - 1e-6  # the features hold log(s + 1e-6)
    psi_feature = float(extractor.get_orientations(polygon[None], slide.mpp_x, slide.mpp_y)[0])
    # ψ as the model sees it (cos 2ψ, sin 2ψ): modulo 180°, in (-90°, 90°]
    psi = 0.5 * float(np.arctan2(features[4 * EFD_ORDER + 2], features[4 * EFD_ORDER + 1]))
    assert np.isclose(np.cos(2 * (psi - psi_feature)), 1.0, atol=1e-6)

    contour = (polygon * np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)).astype(np.float64)
    closed = np.vstack([contour, contour[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    dc = ((closed[:-1] + closed[1:]) / 2 * lengths[:, None]).sum(0) / lengths.sum()
    centroid = r["all_centroids"][q].astype(np.float64)
    # back from the normalised frame: by +ψ_feature (rotating by ψ or ψ + 180° is the
    # same curve up to the order-1 sign, so the extractor's own branch is used)
    rotation = np.array([[np.cos(psi_feature), -np.sin(psi_feature)],
                         [np.sin(psi_feature), np.cos(psi_feature)]])

    def to_um(normalised: NDArray[np.float64]) -> NDArray[np.float64]:
        return normalised * scale @ rotation.T + dc - centroid

    order16 = to_um(efd_curve(coeffs))
    order1 = to_um(efd_curve(coeffs[:1]))
    vertices = contour - centroid
    distance, _ = KDTree(to_um(efd_curve(coeffs, n=16384))).query(vertices)
    rms = float(np.sqrt(np.mean(distance**2)))
    assert rms < 0.05 * scale, rms
    centre = dc - centroid
    major = np.array([np.cos(psi), np.sin(psi)])
    minor_len = scale * abs(coeffs[0, 3])  # the normalised order-1 ellipse is (1, |d1|)
    # the order-1 ellipse reaches s along ±ψ and b perpendicular to it
    radii = np.linalg.norm(order1 - centre, axis=1)
    assert np.isclose(radii.max(), scale, rtol=1e-3) and np.isclose(radii.min(), minor_len, rtol=1e-3)
    # the ψ arc stays inside the order-16 contour
    t = np.linspace(0.0, psi, 64)
    arc = centre + ARC_RADIUS * scale * np.stack([np.cos(t), np.sin(t)], axis=1)
    arc_gap, _ = KDTree(order16).query(arc)
    assert arc_gap.min() > 0.05 * scale, arc_gap.min()
    return {
        "features": features,
        "vertices": vertices,
        "order16": order16,
        "order1": order1,
        "centre": centre,
        "major": major,
        "minor_len": minor_len,
        "psi": psi,
        "scale_um": scale,
        "axis_ratio": scale / minor_len,
        "arc_gap_to_contour_um": float(arc_gap.min()),
        "dc_offset_um": float(np.linalg.norm(centre)),
        "recon_rms_um": rms,
        "recon_max_um": float(distance.max()),
        "max_abs_diff_dataset": max_abs_diff,
    }


def frame(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER_GREY)
        spine.set_linewidth(BORDER_LW)


def window_axes(ax: plt.Axes, r: dict) -> None:
    """The window, y down as in the slide, equal aspect."""
    ax.set_xlim(*r["x_um"])
    ax.set_ylim(r["y_um"][1], r["y_um"][0])
    ax.set_aspect("equal")
    frame(ax)


def outlines(polygons: NDArray, color, lw: float, **kwargs) -> PolyCollection:
    return PolyCollection(polygons, closed=True, facecolors="none", edgecolors=[color],
                          linewidths=lw, **kwargs)


def query_outline(ax: plt.Axes, r: dict) -> PolyCollection:
    return ax.add_collection(outlines(r["all_polygons"][[r["query"]]], ORANGE, QUERY_LW, zorder=5))


def draw_he(ax: plt.Axes, r: dict, shape: dict) -> dict:
    window_axes(ax, r)
    ax.imshow(r["he"], extent=r["he_extent"], interpolation="antialiased")
    ax.add_collection(outlines(r["polygons"], HE_OUTLINE, HE_OUTLINE_LW))
    marker = query_outline(ax, r)

    # white bar with a black outline drawn inside it, so its outer length is 25 µm; the
    # label just above it
    x_um, y_um = r["x_um"], r["y_um"]
    um_per_pt = WINDOW_UM / (ax.get_window_extent().width * 72 / ax.figure.dpi)
    inset = SCALE_BAR_LW / 2 * um_per_pt
    x0 = x_um[0] + SCALE_BAR_PAD_UM
    y1 = y_um[1] - SCALE_BAR_PAD_UM
    x1, y0 = x0 + SCALE_BAR_UM, y1 - SCALE_BAR_H_UM
    ax.add_patch(Rectangle((x0 + inset, y0 + inset), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset,
                           facecolor="white", edgecolor="black", lw=SCALE_BAR_LW))
    text = ax.text((x0 + x1) / 2, y0 - 1.0, f"{SCALE_BAR_UM:g} µm", ha="center", va="bottom",
                   fontsize=SMALL_FONT_SIZE, color="white",
                   path_effects=[pe.withStroke(linewidth=TEXT_STROKE, foreground="black")])
    return {"scale_text": text, "scale_bar_um": (x0, x1, y0, y1), "marker": marker}


def draw_efd(ax: plt.Axes, r: dict, shape: dict, text_h_cm: float) -> dict:
    """The shape in µm around the centroid; the plot above the two lines of text."""
    frame(ax)
    curves = np.vstack([shape["vertices"], shape["order16"], shape["order1"], [shape["centre"]]])
    lo, hi = curves.min(axis=0), curves.max(axis=0)
    extent = hi - lo
    pad = B_PAD * extent.max()
    w_cm, h_cm = (ax.get_position().width * ax.figure.get_figwidth() / CM,
                  ax.get_position().height * ax.figure.get_figheight() / CM)
    plot_h_cm = h_cm - text_h_cm
    um_per_cm = max((extent[0] + 2 * pad) / w_cm, (extent[1] + 2 * pad) / plot_h_cm)
    cx = (lo[0] + hi[0]) / 2
    top = lo[1] - ((plot_h_cm * um_per_cm) - extent[1]) / 2
    ax.set_xlim(cx - w_cm * um_per_cm / 2, cx + w_cm * um_per_cm / 2)
    ax.set_ylim(top + h_cm * um_per_cm, top)  # y down, as in the slide
    ax.set_aspect("auto")  # equal by construction: the same µm per cm on both axes

    closed = np.vstack
    ax.plot(*closed([shape["order1"], shape["order1"][:1]]).T, color=ORDER1_COLOR, lw=ORDER1_LW,
            ls=(0, (3, 2)), zorder=2)
    ax.plot(*closed([shape["order16"], shape["order16"][:1]]).T, color=ORDER16_COLOR,
            lw=ORDER16_LW, zorder=3)
    ax.scatter(*shape["vertices"].T, s=VERTEX_D**2, c=VERTEX_COLOR, lw=0, zorder=4)

    # semi-major axis (solid), semi-minor axis (dotted), the horizontal reference
    # (dotted) and the ψ arc between it and the major axis
    c, u, psi, s = shape["centre"], shape["major"], shape["psi"], shape["scale_um"]
    v = np.array([-u[1], u[0]])
    wedge = np.array([1.0, 0.0]) + u  # bisector of the ψ wedge
    if v @ wedge > 0:  # the minor axis on the side away from the wedge
        v = -v
    t = np.linspace(0.0, psi, 64)
    guides = {
        "major": np.stack([c, c + s * u]),
        "minor": np.stack([c, c + shape["minor_len"] * v]),
        "reference": np.stack([c, c + (ARC_RADIUS + 0.35) * s * np.array([1.0, 0.0])]),
        "arc": c + ARC_RADIUS * s * np.stack([np.cos(t), np.sin(t)], axis=1),
    }
    ax.plot(*guides["major"].T, color=ORDER16_COLOR, lw=AXIS_LW, zorder=3)
    ax.plot(*guides["minor"].T, color=ORDER16_COLOR, lw=AXIS_LW / 2, ls=(0, (1, 1.5)), zorder=3)
    ax.plot(*guides["reference"].T, color=VERTEX_COLOR, lw=AXIS_LW / 2, ls=(0, (1, 1.5)), zorder=3)
    ax.plot(*guides["arc"].T, color=VERTEX_COLOR, lw=AXIS_LW, zorder=3)

    # everything a label must keep clear of, densely sampled (µm)
    def dense(line: NDArray) -> NDArray:
        steps = np.linspace(0.0, 1.0, 40)[:, None]
        return np.concatenate([a + steps * (b - a) for a, b in zip(line[:-1], line[1:], strict=True)])

    obstacles = np.vstack([
        dense(closed([shape["order16"], shape["order16"][:1]])),
        dense(closed([shape["order1"], shape["order1"][:1]])),
        shape["vertices"], *(dense(g) for g in guides.values()),
    ])
    renderer = ax.figure.canvas.get_renderer()
    frame_box = ax.get_window_extent(renderer)

    def place(text: str, color: str, positions: list[NDArray], placed: list) -> plt.Text:
        """The first position whose text box (padded by 1 pt) clears every obstacle and
        every label placed before."""
        pad = ax.figure.dpi / 72
        px = ax.transData.transform(obstacles)
        for xy in positions:
            label = ax.text(*xy, text, ha="center", va="center", fontsize=SMALL_FONT_SIZE,
                            color=color, zorder=6)
            box = label.get_window_extent(renderer).padded(pad)
            free = not ((px[:, 0] > box.x0) & (px[:, 0] < box.x1)
                        & (px[:, 1] > box.y0) & (px[:, 1] < box.y1)).any()
            inside = frame_box.contains(box.x0, box.y0) and frame_box.contains(box.x1, box.y1)
            if free and inside and not any(box.overlaps(p.get_window_extent(renderer)) for p in placed):
                return label
            label.remove()
        raise ValueError(f"no free position for {text}")

    # "s" beside the middle of the major axis, on the side away from the wedge; "ψ" on
    # the wedge's bisector beyond the arc; both as close as they fit, inside the
    # nucleus if there is room, else just outside it
    s_label = place("$s$", ORDER16_COLOR,
                    [c + 0.5 * s * u + d * s * v for d in np.arange(0.15, 1.5, 0.02)], [])
    # ψ: in the wedge beyond the arc; else next to the arc's start, on the other side of
    # the reference line; else further out along the bisector
    mid = np.array([np.cos(psi / 2), np.sin(psi / 2)])
    side = -np.sign(psi)  # away from the wedge, across the reference line
    wedge_spots = [c + d * s * mid for d in np.arange(ARC_RADIUS + 0.12, 0.95, 0.02)]
    start_spots = [
        c + d * s * np.array([np.cos(a), np.sin(a)])
        for d in np.arange(ARC_RADIUS + 0.05, 0.9, 0.03)
        for a in side * np.radians(np.arange(12.0, 50.0, 4.0))
    ]
    outer_spots = [c + d * s * mid for d in np.arange(0.95, 2.0, 0.02)]
    psi_label = place("$\\psi$", VERTEX_COLOR, wedge_spots + start_spots + outer_spots, [s_label])
    text = ax.text(0.5, 0.03, B_TEXT, transform=ax.transAxes, ha="center", va="bottom",
                   ma="center", fontsize=SMALL_FONT_SIZE, linespacing=1.15)
    return {"text": text, "s_label": s_label, "psi_label": psi_label, "um_per_cm": um_per_cm,
            "curves_um": curves, "obstacles_um": obstacles, "guides": guides}


def draw_tokens(ax: plt.Axes, r: dict, shape: dict) -> dict:
    window_axes(ax, r)
    ax.add_collection(outlines(r["polygons"], TOKEN_GREY, TOKEN_LW))
    ax.scatter(*r["centroids"].T, s=CENTROID_D**2, c=CENTROID_GREY, lw=0, zorder=2)
    centre = r["all_centroids"][r["query"]]
    others = r["knn"][1:]
    for p in r["all_centroids"][others]:
        ax.plot(*np.stack([centre, p]).T, color=ORANGE, lw=EDGE_LW, alpha=EDGE_ALPHA, zorder=3)
    ax.add_collection(outlines(r["all_polygons"][others], ORANGE, NEIGHBOUR_LW, zorder=4))
    marker = ax.add_collection(PolyCollection(r["all_polygons"][[r["query"]]], closed=True,
                                              facecolors=[ORANGE], edgecolors="none", zorder=5))
    return {"marker": marker}


def draw_heatmap(ax: plt.Axes, r: dict, shape: dict) -> dict:
    window_axes(ax, r)
    cells = PolyCollection(r["polygons"], closed=True, cmap=CMAP,
                           norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                           edgecolors="none", antialiased=True)
    cells.set_array(r["prob"])
    ax.add_collection(cells)
    return {"marker": query_outline(ax, r)}


def arrow(lay: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> FancyArrowPatch:
    patch = FancyArrowPatch(start, end, arrowstyle=ARROW_STYLE, mutation_scale=1,
                            lw=LINE_PT, color="black", shrinkA=0, shrinkB=0)
    lay.add_patch(patch)
    return patch


def text_size_cm(texts: dict[str, str], fontsize: float) -> dict[str, tuple[float, float]]:
    probe = plt.figure(figsize=(PANEL_W * CM, 2 * CM))
    renderer = probe.canvas.get_renderer()
    sizes = {}
    for key, text in texts.items():
        extent = probe.text(0, 0, text, fontsize=fontsize, ma="center",
                            linespacing=1.15).get_window_extent(renderer)
        sizes[key] = (extent.width / probe.dpi * 2.54, extent.height / probe.dpi * 2.54)
    plt.close(probe)
    return sizes


def draw_panel(r: dict, shape: dict) -> tuple[plt.Figure, dict]:
    captions = text_size_cm(CAPTIONS, FONT_SIZE)
    labels = text_size_cm(ARROW_LABELS, SMALL_FONT_SIZE)
    b_text_h = text_size_cm({"b": B_TEXT}, SMALL_FONT_SIZE)["b"][1] + 0.12

    gaps = {"A->B": AB_GAP}
    gaps |= {name: labels[name][0] + 2 * MIN_LABEL_CLEARANCE + 0.02 for name in ARROW_LABELS}
    lead = 0.0
    for _ in range(3):  # the row starts far enough in for the first caption
        side = (PANEL_W - 2 * EDGE - lead - CBAR_BLOCK - B_W - sum(gaps.values())) / 3
        lead = max(0.0, (captions["A"][0] - side) / 2 + 0.01)
    widths = {"A": side, "B": B_W, "C": side, "D": side}
    caption_h = max(h for _, h in captions.values())
    height = TOP + caption_h + CAPTION_PAD + side + BOTTOM
    bottom, mid = BOTTOM, BOTTOM + side / 2

    fig = plt.figure(figsize=(PANEL_W * CM, height * CM), facecolor="white")
    lay = fig.add_axes((0, 0, 1, 1))  # layout axes in cm, under the elements
    lay.set_xlim(0, PANEL_W)
    lay.set_ylim(0, height)
    lay.axis("off")

    def cm_rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / PANEL_W, y / height, w / PANEL_W, h / height)

    left, x = {}, EDGE + lead
    for a, b in zip(ELEMENTS, [*ELEMENTS[1:], None], strict=True):
        left[a] = x
        x += widths[a] + (gaps[f"{a}->{b}"] if b else 0.0)

    items: dict = {"axes": {}, "captions": {}, "arrows": {}, "arrow_labels": {}, "extras": {},
                   "widths": widths, "side": side, "gaps": gaps, "size_cm": (PANEL_W, height)}
    baseline = bottom + side + CAPTION_PAD + 0.06  # one baseline for all captions
    draw = {"A": draw_he, "C": draw_tokens, "D": draw_heatmap}
    for name in ELEMENTS:
        ax = fig.add_axes(cm_rect(left[name], bottom, widths[name], side))
        items["axes"][name] = ax
        if name == "B":
            items["extras"][name] = draw_efd(ax, r, shape, b_text_h)
        else:
            items["extras"][name] = draw[name](ax, r, shape)
        items["captions"][name] = lay.text(left[name] + widths[name] / 2, baseline, CAPTIONS[name],
                                           ha="center", va="baseline", fontsize=FONT_SIZE)
    for name in ARROW_LABELS:
        a, b = name.split("->")
        centre = left[a] + widths[a] + gaps[name] / 2
        items["arrows"][name] = arrow(lay, (centre - ARROW_LEN / 2, mid), (centre + ARROW_LEN / 2, mid))
        items["arrow_labels"][name] = lay.text(centre, mid + ARROW_LABEL_PAD, ARROW_LABELS[name],
                                               ha="center", va="bottom", fontsize=SMALL_FONT_SIZE)

    # zoom callout: from the nucleus in A to the left corners of B, over the images
    over = fig.add_axes((0, 0, 1, 1), zorder=10)
    over.set_xlim(0, PANEL_W)
    over.set_ylim(0, height)
    over.axis("off")
    over.patch.set_alpha(0.0)
    q = r["all_centroids"][r["query"]]
    x_um, y_um = r["x_um"], r["y_um"]
    q_cm = np.array([left["A"] + (q[0] - x_um[0]) / WINDOW_UM * side,
                     bottom + (y_um[1] - q[1]) / WINDOW_UM * side])
    radius_cm = (np.linalg.norm(r["all_polygons"][r["query"]] - q, axis=1).max() + 1.0) / WINDOW_UM * side
    corners = {"top": np.array([left["B"], bottom + side]), "bottom": np.array([left["B"], bottom])}
    items["callout"] = {}
    for key, corner in corners.items():
        u = (corner - q_cm) / np.linalg.norm(corner - q_cm)
        start = q_cm + radius_cm * u
        (line,) = over.plot([start[0], corner[0]], [start[1], corner[1]], color=CALLOUT_COLOR,
                            lw=CALLOUT_LW, ls=(0, (3, 2)), solid_capstyle="butt")
        items["callout"][key] = (start, corner, line)

    cax = fig.add_axes(cm_rect(left["D"] + side + CBAR_GAP, bottom, CBAR_W, side))
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
    items["q_cm"] = q_cm
    return fig, items


def panel_checks(fig: plt.Figure, items: dict, r: dict, shape: dict) -> dict:
    """Aligned elements; centred arrows and labels; text sizes; no overlaps; text inside
    the figure; the same µm mapping in A, C, D; the nucleus marked; the callout.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_cm = fig.dpi_scale_trans.inverted() + mpl.transforms.Affine2D().scale(2.54)

    def cm_box(bbox: Bbox) -> Bbox:
        return bbox.transformed(to_cm)

    width, height = items["size_cm"]
    w_in, h_in = fig.get_size_inches()
    assert np.isclose(w_in * 2.54, PANEL_W) and np.isclose(h_in * 2.54, height)

    side, widths = items["side"], items["widths"]
    boxes_el = {n: cm_box(ax.get_window_extent(renderer)) for n, ax in items["axes"].items()}
    for name, b in boxes_el.items():
        assert np.isclose(b.width, widths[name]) and np.isclose(b.height, side), f"{name}: {b}"
        assert np.isclose(b.y0, boxes_el["A"].y0) and np.isclose(b.y1, boxes_el["A"].y1), f"{name} not aligned"

    for name, a in items["arrows"].items():
        p, q = name.split("->")
        (sx, sy), (ex, ey) = a._posA_posB  # the arrow's own endpoints, cm
        centre = (boxes_el[p].x1 + boxes_el[q].x0) / 2
        assert abs((sx + ex) / 2 - centre) < 1e-9, f"arrow {name} not centred"
        assert np.isclose(sy, boxes_el[p].y0 + side / 2) and sy == ey, f"arrow {name} not at mid-height"
        label = cm_box(items["arrow_labels"][name].get_window_extent(renderer))
        assert abs((label.x0 + label.x1) / 2 - centre) < 0.01, f"label of {name} not centred"
        assert (label.x0 - boxes_el[p].x1 >= MIN_LABEL_CLEARANCE
                and boxes_el[q].x0 - label.x1 >= MIN_LABEL_CLEARANCE), f"label of {name} too close"

    # text sizes; captions one line on one baseline
    baselines = set()
    for name, t in items["captions"].items():
        assert t.get_fontsize() == FONT_SIZE and "\n" not in t.get_text(), name
        assert t.get_va() == "baseline"
        baselines.add(t.get_position()[1])
    assert len(baselines) == 1
    b_extras = items["extras"]["B"]
    small = [*items["arrow_labels"].values(), items["extras"]["A"]["scale_text"],
             b_extras["text"], b_extras["s_label"], b_extras["psi_label"]]
    assert all(t.get_fontsize() == SMALL_FONT_SIZE for t in small)
    cax = items["colorbar"]
    assert all(t.get_fontsize() == SMALL_FONT_SIZE for t in cax.get_yticklabels())
    assert cax.yaxis.label.get_fontsize() == SMALL_FONT_SIZE

    boxes: dict[str, Bbox] = {f"element:{n}": b for n, b in boxes_el.items()}
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

    # the scale label sits on A, just above the bar
    a_ax = items["axes"]["A"]
    label = items["extras"]["A"]["scale_text"].get_window_extent(renderer)
    x0, x1, y0, y1 = items["extras"]["A"]["scale_bar_um"]
    bar = Bbox(a_ax.transData.transform([[x0, y1], [x1, y0]]))
    assert a_ax.get_window_extent(renderer).contains(label.x0, label.y0)
    assert a_ax.get_window_extent(renderer).contains(label.x1, label.y1)
    assert label.y0 > bar.y1 and not label.overlaps(bar)

    # B: its texts inside it; the two lines below every curve; s and ψ off the curves
    b_ax = items["axes"]["B"]
    b_box = b_ax.get_window_extent(renderer)
    curves_px = Bbox.from_extents(*b_ax.transData.transform(b_extras["curves_um"]).min(0),
                                  *b_ax.transData.transform(b_extras["curves_um"]).max(0))
    for key in ("text", "s_label", "psi_label"):
        t = b_extras[key].get_window_extent(renderer)
        assert b_box.contains(t.x0, t.y0) and b_box.contains(t.x1, t.y1), f"B {key} outside B"
    assert b_extras["text"].get_window_extent(renderer).y1 < curves_px.y0, "B text overlaps the shape"
    x_scale = np.diff(b_ax.transData.transform([[0, 0], [1, 0]]), axis=0)[0, 0]
    y_scale = -np.diff(b_ax.transData.transform([[0, 0], [0, 1]]), axis=0)[0, 1]
    assert np.isclose(x_scale, y_scale), "B not equal aspect"
    obstacles_px = b_ax.transData.transform(b_extras["obstacles_um"])
    for key in ("s_label", "psi_label"):
        t = b_extras[key].get_window_extent(renderer)
        hit = ((obstacles_px[:, 0] > t.x0) & (obstacles_px[:, 0] < t.x1)
               & (obstacles_px[:, 1] > t.y0) & (obstacles_px[:, 1] < t.y1))
        assert not hit.any(), f"B {key} on a line"
    assert not b_extras["s_label"].get_window_extent(renderer).overlaps(
        b_extras["psi_label"].get_window_extent(renderer)), "B labels overlap"

    # the same µm mapping in A, C, D
    ref = None
    for name in ("A", "C", "D"):
        ax = items["axes"][name]
        px = ax.transData.transform(r["centroids"]) - ax.get_window_extent(renderer).p0
        ref = px if ref is None else ref
        assert np.allclose(px, ref, atol=1e-6)
    # the nucleus is marked in A, C, D and lies where the callout starts
    for name in ("A", "C", "D"):
        marker = items["extras"][name]["marker"]
        assert np.allclose(marker.get_paths()[0].vertices[:-1].mean(0),
                           r["all_polygons"][r["query"]].mean(0), atol=0.5)
    q_px = a_ax.transData.transform(r["all_centroids"][r["query"]])
    assert np.allclose(cm_box(Bbox([q_px, q_px])).p0, items["q_cm"], atol=1e-6)
    for key, (start, corner, _) in items["callout"].items():
        target = (boxes_el["B"].x0, boxes_el["B"].y1 if key == "top" else boxes_el["B"].y0)
        assert np.allclose(corner, target), f"callout {key} does not end at B"

    return {
        "size_cm": [round(width, 3), round(height, 3)],
        "square_cm": round(side, 4),
        "b_cm": [B_W, round(side, 4)],
        "gaps_cm": {k: round(v, 4) for k, v in items["gaps"].items()},
        "b_um_per_cm": round(b_extras["um_per_cm"], 3),
        "smallest_clearances_cm": dict(sorted(clearances.items(), key=lambda kv: kv[1])[:6]),
    }


def data_checks(items: dict, r: dict) -> dict:
    n = len(r["region"])
    axes = items["axes"]
    tokens = axes["C"].collections  # grey outlines, centroids, neighbour outlines, query
    counts = {
        "A_outlines": len(axes["A"].collections[0].get_paths()),
        "C_outlines": len(tokens[0].get_paths()),
        "C_centroids": len(tokens[1].get_offsets()),
        "D_cells": len(axes["D"].collections[0].get_paths()),
    }
    assert all(c == n for c in counts.values()), counts
    assert len(tokens[2].get_paths()) == K - 1 and len(tokens[3].get_paths()) == 1
    assert len(axes["C"].lines) == K - 1
    return {"nuclei_drawn": counts, "C_neighbours": K - 1, "C_lines": len(axes["C"].lines)}


def save_elements(r: dict, shape: dict, items: dict) -> None:
    """Each element on its own, at its size in the panel (no caption, no arrows)."""
    side = items["side"]
    for name in ELEMENTS:
        w = items["widths"][name]
        fig = plt.figure(figsize=(w * CM, side * CM), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        if name == "B":
            draw_efd(ax, r, shape, text_size_cm({"b": B_TEXT}, SMALL_FONT_SIZE)["b"][1] + 0.12)
        else:
            {"A": draw_he, "C": draw_tokens, "D": draw_heatmap}[name](ax, r, shape)
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
    shape = shape_encoding(r)
    panel, items = draw_panel(r, shape)
    checks = data_checks(items, r)
    checks["panel"] = panel_checks(panel, items, r, shape)

    q = r["query"]
    y, p = r["labels"], r["prob"]
    w = r["window"]
    checks |= {
        "model_run_id": MODEL_RUN_ID,
        "slide_id": SLIDE_ID,
        "gleason_score": r["slide"].gleason_score,
        "mpp": [float(r["slide"].mpp_x), float(r["slide"].mpp_y)],
        "window_um": {"x": list(r["x_um"]), "y": list(r["y_um"])},
        "window_rule": (
            f"{WINDOW_UM:g} µm square, nuclei with centroid in [x0, x1) × [y0, y1); nucleus "
            f">= {MIN_MARGIN_UM:g} µm from every edge, its k-NN inside; >= "
            f"{MIN_CLASS_SHARE:.0%} of nuclei on each side of sigmoid(z) = 0.5; then the "
            "largest margin, then the most balanced"
        ),
        "nucleus_margin_um": w["margin"],
        "n_nuclei": len(r["region"]),
        "share_sigmoid_below_0.5": float((p < 0.5).mean()),
        "share_benign_label": float((y == 0).mean()),
        "n_carcinoma": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
        "accuracy_at_0.5": float(((p > 0.5) == (y == 1)).mean()),
        "knn": {
            "k": K,
            "includes_query": True,
            "query_index": q,
            "query_nucleus_id": str(r["nuclei"]["id"].iloc[q]),
            "query_centroid_um": [float(v) for v in r["all_centroids"][q]],
            "query_logit": float(r["logits"][q]),
            "neighbour_indices": [int(i) for i in r["knn"][1:]],
            "max_neighbour_distance_um": float(r["knn_dist"].max()),
            "kth_distance_gap_um": r["kth_gap_um"],
        },
        "nucleus_selection": {
            **r["candidates"].attrs,
            "criteria": {
                "axis_ratio": list(RATIO_RANGE), "s_percentiles": list(S_PERCENTILES),
                "min_solidity": MIN_SOLIDITY, "no_overlap": True,
                "min_knn_each_side_of_0.5": MIN_KNN_EACH,
                "abs_psi_deg_for_display": list(PSI_RANGE_DEG),
                "min_arc_clearance_s": MIN_ARC_CLEARANCE,
                "window": "see window_rule",
            },
            "ranking": "kNN balance desc, |axis ratio - 2.5| asc, window imbalance asc, margin desc",
            "top": json.loads(r["candidates"].head(N_CANDIDATES).to_json(orient="records")),
        },
        "shape_encoding": {
            "efd_order": EFD_ORDER,
            "psi_deg": float(np.degrees(shape["psi"])),
            "psi_convention": "image coordinates (x right, y down), modulo 180° in (-90°, 90°]; "
            "positive = clockwise on screen",
            "scale_um": shape["scale_um"],
            "axis_ratio": shape["axis_ratio"],
            "sigmoid_z": float(1.0 / (1.0 + np.exp(-r["logits"][q]))),
            "arc_gap_to_contour_um": shape["arc_gap_to_contour_um"],
            "efd_centre_to_centroid_um": shape["dc_offset_um"],
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
    save_elements(r, shape, items)
    (OUT_DIR / "checks.json").write_text(json.dumps(checks, indent=2))

    if args.log:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with mlflow.start_run(run_id=PANEL_RUN_ID) as run:
            mlflow.set_tags(
                {
                    "git_commit": commit,
                    "panel_layout": "A H&E -> B EFD -> C tokens -> D logits",
                    "panel_size_cm": "x".join(f"{v:.2f}" for v in checks["panel"]["size_cm"]),
                }
            )
            mlflow.log_artifacts(str(OUT_DIR), artifact_path="arch_panel_a")
            mlflow.log_artifact(__file__)
            print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
