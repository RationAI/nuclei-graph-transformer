"""Thesis figure: what each nucleus embedding gives the model.

One nucleus of a PANDA Radboud test slide (in annotated carcinoma), as
(a) the H&E input with the polygon, centroid and the rotated texture window,
(b) the normalized polygon and its inverse-EFD reconstructions (shape embedding),
(c) the exact 48 × 48 patch the CNN receives (texture embedding).
The blank token carries no per-nucleus information, so it has no panel.

All model inputs come from the repository's own code: EFDs from
`NucleiFeatureExtractor.get_efd_features`, ψ from `get_orientations`, the patch from
`nucleus_patch`, and the checks compare against `BaseCropDataset.generate_embeddings`
on the full slide (the `full_slide` prediction path).

Usage:
    uv run -m visualization.thesis_embeddings_figure [--log]

`--log` logs the figure, README, checks and this script to MLflow (experiment 37).
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide
from scipy.spatial import KDTree

from nuclei_graph.data.datasets.crop.nuclei_classification import (
    NucleiClassificationDataset,
)
from nuclei_graph.data.datasets.nuclei_features import NucleiFeatureExtractor
from nuclei_graph.nuclei_graph_typing import (
    BBOX_MASK_MARGIN_UM,
    TARGET_BBOX_CONTEXT_UM,
    Box,
    SlideSize,
)


METADATA_URI = "mlflow-artifacts:/37/5b9c6f096026425fa2b04c4e74b7a6f3/artifacts/panda/slides_mapping_test.parquet"
SLIDE_ID = "4a3e6e77b42a7e8f8b2dec472667bdbe"  # PANDA Radboud test, Gleason 3+3
# row index into the slide's nuclei sorted by "id" (as in the datasets)
NUCLEI = {
    "carcinoma (Gleason 3)": 1867,
}

EFD_ORDER = 16  # as in training (configs/experiment/modeling/training/nuclei_level/radboud.yaml)
PATCH_SIZE = 48
RECON_ORDERS = (1, 16)
CROP_UM = 40.0
SCALE_BAR_UM = 10.0

COLUMNS = ("(a) Input", "(b) Shape", "(c) Texture")
FIG_W_IN = 16 / 2.54
PANEL_IN = 1.95
GAP_IN = 0.2
ROW_GAP_IN = 0.1
TOP_IN = 0.22

POLYGON_COLOR, POLYGON_LW = "#FFD700", 0.8  # pt
POLYGON_EDGE, POLYGON_EDGE_LW = "#1a1a1a", 0.3  # thin dark outline around the yellow line
RAY_LW, RAY_ALPHA = 0.3, 0.6
ORIGIN_COLOR, ORIGIN_MS = "#E6550D", 2.0  # the LSP-DETR point p
OTHER_LW, OTHER_ALPHA = 0.4, 0.6  # the other nuclei in the crop (white outlines)
MAX_RAY_ANGLE_DEV_DEG = 0.05  # vertices on the 64 evenly spaced rays from p
SCALE_BAR_LW = 1.5
TEXT_STROKE = 1.0  # black outline of the white scale bar and its label (half shows)
ACCENT_COLOR = "#E4572E"
SHAPE_COLORS = ("#86b6ef", "#1c5cab")  # one hue, light -> dark
POLYGON_GREY = "#000000"  # drawn at alpha 0.3
FRAME_GREY = "#b4b4b4"

OUT_DIR = Path("figures")

# how (b) can be displayed relative to (c): factors on (x, y) of the normalized frame
ORIENTATIONS = {
    "as computed": (1.0, 1.0),
    "rotated 180°": (-1.0, -1.0),
    "y flipped": (1.0, -1.0),
    "x flipped": (-1.0, 1.0),
}


def efd_centre(contour: NDArray[np.float64]) -> NDArray[np.float64]:
    """DC term (A0, C0) of a closed polyline: its arc-length-weighted mean."""
    closed = np.concatenate([contour, contour[:1]])
    dt = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    mid = (closed[1:] + closed[:-1]) / 2
    return (dt[:, None] * mid).sum(0) / dt.sum()


def ray_origin(polygon: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict]:
    """The LSP-DETR point p of a star-convex polygon, and how well the rays fit.

    `nuclei_standardization.py` builds each polygon as p + r_k (sin θ_k, cos θ_k) with
    θ_k = 2π k / n and stores the vertex mean as the centroid, not p. Every vertex lies
    on its ray from p, so (v_k - p) × (sin θ_k, cos θ_k) = 0: n linear equations in p,
    solved by least squares.
    """
    n = len(polygon)
    theta = 2 * np.pi * np.arange(n) / n
    a = np.stack([np.cos(theta), -np.sin(theta)], axis=1)
    b = polygon[:, 0] * np.cos(theta) - polygon[:, 1] * np.sin(theta)
    p, *_ = np.linalg.lstsq(a, b, rcond=None)

    def deviation(origin: NDArray[np.float64]) -> float:
        d = polygon - origin
        angle = np.arctan2(d[:, 0], d[:, 1])  # θ as in the standardization
        return float(np.degrees(np.abs(np.angle(np.exp(1j * (angle - theta))))).max())

    return p, {"deviation_around_p_deg": deviation(p), "theta_step_deg": 360.0 / n}


def inverse_efd(
    coeffs: NDArray[np.float64], order: int, n: int = 4096
) -> NDArray[np.float64]:
    """Curve of the first `order` harmonics (no DC term), `coeffs` is (order, 4)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    k = np.arange(1, order + 1)[:, None]
    cos, sin = np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)
    a, b, c, d = coeffs[:order].T
    x = (a[:, None] * cos + b[:, None] * sin).sum(0)
    y = (c[:, None] * cos + d[:, None] * sin).sum(0)
    return np.stack([x, y], axis=1)


def densify(polyline: NDArray[np.float64], per_segment: int = 64) -> NDArray[np.float64]:
    closed = np.concatenate([polyline, polyline[:1]])
    s = np.linspace(0.0, 1.0, per_segment, endpoint=False)[None, :, None]
    pts = closed[:-1, None] + s * (closed[1:, None] - closed[:-1, None])
    return pts.reshape(-1, 2)


def rms_distance(polygon: NDArray[np.float64], curve: NDArray[np.float64]) -> dict:
    """RMS nearest-point distances polygon -> curve and curve -> polygon."""
    poly_dense = densify(polygon)
    to_curve, _ = KDTree(curve).query(poly_dense)
    to_poly, _ = KDTree(poly_dense).query(curve)
    return {
        "polygon_to_curve": float(np.sqrt(np.mean(to_curve**2))),
        "curve_to_polygon": float(np.sqrt(np.mean(to_poly**2))),
    }


def polygon_iou(polygon_px: NDArray[np.float64], nucleus: NDArray[np.bool_]) -> float:
    """IoU of a patch-px polygon, dilated by the patch's mask margin, with `nucleus`."""
    mask = np.zeros(nucleus.shape, dtype=np.uint8)
    vertices = np.round(polygon_px * 16).astype(np.int32)  # 4-bit subpixel
    cv2.fillPoly(mask, [vertices], 255, lineType=cv2.LINE_8, shift=4)
    margin_px = round(BBOX_MASK_MARGIN_UM * PATCH_SIZE / TARGET_BBOX_CONTEXT_UM)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
    )
    filled = cv2.dilate(mask, kernel) > 0
    return float((filled & nucleus).sum() / (filled | nucleus).sum())


def patch_affine(
    box: Box, centroid: NDArray[np.float32], psi: float, mpp_x: float, mpp_y: float
) -> NDArray[np.float64]:
    """The window px -> patch px affine of `NucleiFeatureExtractor.nucleus_patch`."""
    origin = np.array([box.lx, box.ly], dtype=np.float32)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    rotation = np.array([[cos_psi, sin_psi], [-sin_psi, cos_psi]])
    linear = (PATCH_SIZE / TARGET_BBOX_CONTEXT_UM) * rotation @ np.diag([mpp_x, mpp_y])
    centre = np.full(2, (PATCH_SIZE - 1) / 2)
    offset = centre - linear @ (centroid - origin)
    return np.hstack([linear, offset[:, None]])


def patch_box(centroid: NDArray[np.float32], mpp_x: float, mpp_y: float) -> Box:
    """The read window of `NucleiFeatureExtractor.get_nuclei_bboxes` for one nucleus."""
    read_um = TARGET_BBOX_CONTEXT_UM * np.sqrt(2)
    size_x = int(np.ceil(read_um / mpp_x)) + 2
    size_y = int(np.ceil(read_um / mpp_y)) + 2
    lx = int(centroid[0]) - size_x // 2
    ly = int(centroid[1]) - size_y // 2
    return Box(lx, ly, lx + size_x, ly + size_y)


def prepare(slide: pd.Series) -> tuple[list[dict], dict]:
    extractor = NucleiFeatureExtractor()
    extractor.efd_order = EFD_ORDER
    extractor.patch_size = PATCH_SIZE

    # the dataset's own embeddings for the full slide (as with full_slide=True)
    dataset = NucleiClassificationDataset(
        metadata=slide.to_frame().T,
        embedding_mode="bbox",
        efd_order=EFD_ORDER,
        full_slide=True,
        patch_size=PATCH_SIZE,
    )
    nuclei = dataset.get_nuclei(slide.slide_nuclei_path)
    centroids_um = dataset.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)
    crop_indices = np.arange(len(nuclei), dtype=int)
    polygons, pos, _ = dataset.process_geometry(nuclei, crop_indices, centroids_um, slide)
    _, dataset_patches = dataset.generate_embeddings(
        polygons, pos, slide, nuclei, crop_indices
    )
    assert dataset_patches is not None

    rows, checks = [], {}
    with OpenSlide(slide.slide_path) as wsi:
        slide_size = SlideSize(*wsi.dimensions)
        for name, idx in NUCLEI.items():
            polygon = polygons[idx].astype(np.float32)  # level-0 px
            centroid = np.asarray(nuclei["centroid"].iloc[idx], dtype=np.float32)
            mpp_x, mpp_y = float(slide.mpp_x), float(slide.mpp_y)
            mpp = np.array([mpp_x, mpp_y])

            # (b) shape: the model's EFD features, then the polygon in the same frame
            features = extractor.get_efd_features(polygon[None], mpp_x, mpp_y)[0]
            coeffs = features[: EFD_ORDER * 4].reshape(EFD_ORDER, 4).astype(np.float64)
            log_s, cos_2psi, sin_2psi = features[EFD_ORDER * 4 :]
            psi = float(extractor.get_orientations(polygon[None], mpp_x, mpp_y)[0])
            assert np.isclose(np.cos(2 * psi), cos_2psi, atol=1e-5)
            assert np.isclose(np.sin(2 * psi), sin_2psi, atol=1e-5)
            # order 1 is the ellipse with its major semi-axis 1 along x
            assert np.isclose(coeffs[0, 0], 1.0, atol=1e-5)
            assert abs(coeffs[0, 1]) < 1e-5 and abs(coeffs[0, 2]) < 1e-5
            assert abs(coeffs[0, 3]) <= 1.0

            contour = polygon.astype(np.float64) * mpp
            rotation = np.array(
                [[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]]
            )
            poly_norm = (contour - efd_centre(contour)) @ rotation.T / np.exp(log_s)
            recons = {k: inverse_efd(coeffs, k) for k in RECON_ORDERS}
            rms = rms_distance(poly_norm, recons[EFD_ORDER])
            assert rms["polygon_to_curve"] < 0.05 and rms["curve_to_polygon"] < 0.05

            # (c) texture: one nucleus through nucleus_patch, as the dataset reads it
            box = patch_box(centroid, mpp_x, mpp_y)
            window = extractor.extract_patch(
                extractor.read_region(wsi, extractor.clip_box(box, slide_size)),
                box,
                slide_size,
            )
            patch = extractor.nucleus_patch(
                window, box, centroid, polygon, psi, mpp_x, mpp_y
            )
            dataset_patch = dataset_patches[idx].permute(1, 2, 0).numpy()
            patch_diff = float(
                np.abs(patch.astype(np.float64) - dataset_patch.astype(np.float64)).max()
            )
            assert patch_diff < 1e-6

            # the overlay affine must be nucleus_patch's: re-warping with it reproduces
            # the unmasked pixels exactly
            affine = patch_affine(box, centroid, psi, mpp_x, mpp_y)
            rewarped = cv2.warpAffine(
                window,
                affine,
                (PATCH_SIZE, PATCH_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            kept = (patch != 255).any(axis=2)
            assert np.array_equal(rewarped[kept], patch[kept])

            # model input normalization (Transformer.embed_patches), undone for display
            model_input = patch.astype(np.float32) / 127.5 - 1.0
            display_patch = np.round((model_input + 1.0) * 127.5).astype(np.uint8)
            assert np.array_equal(display_patch, patch)

            # texture window: the patch's outer pixel edges mapped back to the slide
            inverse = cv2.invertAffineTransform(affine)
            edges = np.array(
                [[-0.5, -0.5], [PATCH_SIZE - 0.5, -0.5], [PATCH_SIZE - 0.5, PATCH_SIZE - 0.5], [-0.5, PATCH_SIZE - 0.5]]
            )
            window_px = edges @ inverse[:, :2].T + inverse[:, 2] + [box.lx, box.ly]

            # polygon extent in the patch frame (half-width of the 16 µm window is 8 µm)
            poly_patch_um = (contour - centroid * mpp) @ rotation.T
            extent_um = float(np.abs(poly_patch_um).max())

            # (b) vs (c) orientation: the order-16 reconstruction of (b), scaled back to µm
            # and drawn into the patch (about its EFD centre), under each way (b) could be
            # displayed; the one that covers the patch's nucleus best is the display of (b)
            centre_offset = (efd_centre(contour) - centroid * mpp) @ rotation.T
            nucleus = (patch != 255).any(axis=2)

            to_patch_px = PATCH_SIZE / TARGET_BBOX_CONTEXT_UM
            orientation_px = {
                name: (PATCH_SIZE - 1) / 2
                + (recons[EFD_ORDER] * np.array(factors) * np.exp(log_s) + centre_offset)
                * to_patch_px
                for name, factors in ORIENTATIONS.items()
            }
            orientation_iou = {
                name: polygon_iou(px, nucleus) for name, px in orientation_px.items()
            }
            orientation = max(orientation_iou, key=orientation_iou.get)
            assert orientation_iou[orientation] > 0.9, orientation_iou

            # (a) input crop, level 0
            half = int(np.ceil(CROP_UM / 2 / min(mpp_x, mpp_y))) + 2
            crop_box = Box(int(centroid[0]) - half, int(centroid[1]) - half,
                           int(centroid[0]) + half, int(centroid[1]) + half)
            crop = extractor.read_region(wsi, crop_box).array

            # the ray origin p (the stored centroid is the vertex mean, not p)
            origin, rays = ray_origin(polygon.astype(np.float64))
            assert rays["deviation_around_p_deg"] < MAX_RAY_ANGLE_DEV_DEG, rays
            d = polygon.astype(np.float64) - centroid
            angle = np.arctan2(d[:, 0], d[:, 1])
            theta = 2 * np.pi * np.arange(len(polygon)) / len(polygon)
            centroid_dev = float(np.degrees(np.abs(np.angle(np.exp(1j * (angle - theta))))).max())
            assert np.allclose(centroid, polygon.mean(axis=0), atol=1e-2)

            # the other nuclei with a vertex in the crop
            in_crop = (
                (polygons[..., 0] >= crop_box.lx) & (polygons[..., 0] < crop_box.rx)
                & (polygons[..., 1] >= crop_box.ly) & (polygons[..., 1] < crop_box.ry)
            ).any(axis=1)
            in_crop[idx] = False
            others = polygons[in_crop].astype(np.float64)

            rows.append(
                {
                    "name": name,
                    "idx": idx,
                    "centroid": centroid,
                    "mpp": mpp,
                    "polygon": polygon,
                    "crop": crop,
                    "crop_box": crop_box,
                    "origin": origin,
                    "others": others,
                    "window_px": window_px,
                    "poly_norm": poly_norm,
                    "recons": recons,
                    "log_s": float(log_s),
                    "psi": psi,
                    "patch": display_patch,
                    "orientation_px": orientation_px,
                    "orientation_iou": orientation_iou,
                    "b_display": np.array(ORIENTATIONS[orientation]),
                }
            )
            checks[name] = {
                "nucleus_index": idx,
                "nucleus_id": str(nuclei["id"].iloc[idx]),
                "centroid_px": [float(centroid[0]), float(centroid[1])],
                "centroid_um": [float(c) for c in centroids_um[idx]],
                "log_s": float(log_s),
                "scale_um": float(np.exp(log_s)),
                "psi_deg": float(np.degrees(psi)),
                "axis_ratio": float(abs(coeffs[0, 3])),
                "polygon_max_abs_coord_in_patch_frame_um": extent_um,
                "polygon_inside_16um_window": extent_um < TARGET_BBOX_CONTEXT_UM / 2,
                "patch_max_abs_diff_vs_dataset": patch_diff,
                "efd16_rms_polygon_to_curve": rms["polygon_to_curve"],
                "efd16_rms_curve_to_polygon": rms["curve_to_polygon"],
                "ray_origin_px": [float(v) for v in origin],
                "ray_origin_to_centroid_um": float(np.linalg.norm((origin - centroid) * mpp)),
                "max_ray_angle_deviation_around_p_deg": rays["deviation_around_p_deg"],
                "max_ray_angle_deviation_around_stored_centroid_deg": centroid_dev,
                "stored_centroid_is_vertex_mean": True,
                "other_nuclei_in_crop": int(in_crop.sum()),
                "orientation_iou_b_vs_c": orientation_iou,
                "b_displayed_as": orientation,
            }
    return rows, checks


def draw_input(ax, row: dict, scale_bar: bool) -> None:
    mpp, centroid, box = row["mpp"], row["centroid"], row["crop_box"]
    # pixel i is centred at coordinate lx + i (the convention nucleus_patch uses)
    left, top = (np.array([box.lx, box.ly]) - 0.5 - centroid) * mpp
    right, bottom = (np.array([box.rx, box.ry]) - 0.5 - centroid) * mpp
    ax.imshow(row["crop"], extent=(left, right, bottom, top), interpolation="none")

    def to_um(p: NDArray) -> NDArray:
        return (p - centroid) * mpp

    # the other nuclei: thin white outlines, no rays
    for other in row["others"]:
        ax.plot(*to_um(np.vstack([other, other[:1]])).T, color="white", lw=OTHER_LW,
                alpha=OTHER_ALPHA, zorder=2)
    # the highlighted nucleus as LSP-DETR describes it: 64 rays from p, the polygon
    # through their ends, and p
    origin = to_um(row["origin"])
    vertices = to_um(row["polygon"].astype(np.float64))
    for v in vertices:
        ax.plot(*np.stack([origin, v]).T, color="white", lw=RAY_LW, alpha=RAY_ALPHA, zorder=3)
    ax.plot(*np.vstack([vertices, vertices[:1]]).T, color=POLYGON_COLOR, lw=POLYGON_LW, zorder=4,
            path_effects=[pe.withStroke(linewidth=POLYGON_LW + 2 * POLYGON_EDGE_LW,
                                        foreground=POLYGON_EDGE)])
    win = to_um(np.vstack([row["window_px"], row["window_px"][:1]]))
    ax.plot(*win.T, color="black", lw=0.6, ls=(0, (3, 2)), zorder=5)
    ax.plot(*origin, "o", ms=ORIGIN_MS, mfc=ORIGIN_COLOR, mec="none", zorder=6)

    h = CROP_UM / 2
    ax.set_xlim(-h, h)
    ax.set_ylim(h, -h)
    ax.set_aspect("equal")
    ax.axis("off")

    if scale_bar:
        # white bar and label, each with a thin black outline; no box behind them
        outline = [pe.withStroke(linewidth=SCALE_BAR_LW + TEXT_STROKE, foreground="black")]
        x0, y0 = -h + 2.5, h - 2.5
        ax.plot([x0, x0 + SCALE_BAR_UM], [y0, y0], color="white", lw=SCALE_BAR_LW,
                solid_capstyle="butt", path_effects=outline)
        ax.text(x0 + SCALE_BAR_UM / 2, y0 - 1.0, f"{SCALE_BAR_UM:g} µm",
                ha="center", va="bottom", fontsize=8, color="white",
                path_effects=[pe.withStroke(linewidth=TEXT_STROKE, foreground="black")])


def draw_shape(ax, row: dict, lim: float, legend: bool) -> None:
    # displayed as the patch of (c) is oriented (see the orientation check in `prepare`)
    flip = row["b_display"]
    # normalized polygon, faint, underneath the curves
    poly = np.vstack([row["poly_norm"], row["poly_norm"][:1]]) * flip
    ax.plot(*poly.T, color=POLYGON_GREY, lw=0.6, alpha=0.3)
    for order, color in zip(RECON_ORDERS, SHAPE_COLORS, strict=True):
        curve = np.vstack([row["recons"][order], row["recons"][order][:1]]) * flip
        ax.plot(*curve.T, color=color, lw=0.9, label=f"{order}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)  # y down, as in the patch
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME_GREY)
        spine.set_linewidth(0.4)
    label = f"log s = {row['log_s']:.2f}, ψ = {np.degrees(row['psi']):.0f}°"
    ax.text(0.5, 0.04, label.replace("-", "−"), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8)
    if legend:
        ax.legend(title="EFD order", title_fontsize=8, loc="upper center", ncols=2,
                  fontsize=8, frameon=False, handlelength=1.0, handletextpad=0.4,
                  columnspacing=0.8, borderaxespad=0.1, labelspacing=0.2)


def draw_texture(ax, row: dict) -> None:
    # "none" embeds the pixels unresampled in the PDF; repeated so that viewers which
    # smooth embedded images keep them sharp (nearest-neighbour display)
    blocks = np.repeat(np.repeat(row["patch"], 8, axis=0), 8, axis=1)
    edge = PATCH_SIZE - 0.5
    ax.imshow(blocks, interpolation="none", extent=(-0.5, edge, edge, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME_GREY)
        spine.set_linewidth(0.4)


def draw(rows: list[dict]) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n = len(rows)
    fig_h = TOP_IN + n * PANEL_IN + (n - 1) * ROW_GAP_IN + 0.03
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    n_cols = len(COLUMNS)
    left0 = (FIG_W_IN - n_cols * PANEL_IN - (n_cols - 1) * GAP_IN) / 2

    def rect(col: int, row: int, x=0.0, y=0.0, w=1.0, h=1.0) -> list[float]:
        """Axes rect of a panel (fractions x, y, w, h of the panel, y from its bottom)."""
        px = left0 + col * (PANEL_IN + GAP_IN) + x * PANEL_IN
        py = fig_h - TOP_IN - (row + 1) * PANEL_IN - row * ROW_GAP_IN + y * PANEL_IN
        return [px / FIG_W_IN, py / fig_h, w * PANEL_IN / FIG_W_IN, h * PANEL_IN / fig_h]

    lim = 1.08 * max(
        np.abs(np.concatenate([r["poly_norm"], *r["recons"].values()])).max() for r in rows
    )
    lim = max(lim, 1.3)  # room for the legend and the label above / below the shape

    for i, row in enumerate(rows):
        draw_input(fig.add_axes(rect(0, i)), row, scale_bar=i == 0)
        draw_shape(fig.add_axes(rect(1, i)), row, lim, legend=i == 0)
        draw_texture(fig.add_axes(rect(2, i)), row)

    for col, label in enumerate(COLUMNS):
        x = left0 + col * (PANEL_IN + GAP_IN) + PANEL_IN / 2
        fig.text(x / FIG_W_IN, 1 - 0.06 / fig_h, label, ha="center", va="top", fontsize=9)
    return fig


def draw_orientation_check(rows: list[dict]) -> plt.Figure:
    """The patch of (c) with the order-16 curve of (b) drawn into it (check only).

    The curve is scaled back to µm and drawn as computed, rotated by 180° and mirrored.
    """
    styles = {
        "as computed": ("lime", "-"),
        "rotated 180°": ("red", "--"),
        "y flipped": ("cyan", ":"),
        "x flipped": ("yellow", "-."),
    }
    fig, axs = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.6), squeeze=False)
    for ax, row in zip(axs[0], rows, strict=True):
        ax.imshow(row["patch"], interpolation="nearest")
        for name, (color, ls) in styles.items():
            curve = np.vstack([row["orientation_px"][name], row["orientation_px"][name][:1]])
            ax.plot(*curve.T, color=color, lw=1, ls=ls,
                    label=f"{name}: IoU {row['orientation_iou'][name]:.3f}")
        ax.legend(loc="lower center", fontsize=6, framealpha=0.8)
        ax.set_title(f"{row['name']}: order-16 curve of (b) in the patch", fontsize=8)
    fig.tight_layout()
    return fig


def write_readme(path: Path, slide: pd.Series, checks: dict, commit: str) -> None:
    lines = [
        "# Thesis figure: embeddings",
        "",
        f"Generated by `visualization/thesis_embeddings_figure.py` at commit `{commit}` "
        f"(branch xai-sae, `TARGET_BBOX_CONTEXT_UM = {TARGET_BBOX_CONTEXT_UM:g}`).",
        "",
        f"Slide: PANDA Radboud test, `{slide.slide_id}` (Gleason {slide.gleason_score}), "
        f"mpp = {slide.mpp_x:.4f} µm/px.",
        "Nucleus index = row of the slide's nuclei parquet sorted by `id` (as in the datasets).",
        "",
        "| row | nucleus index | centroid (level-0 px) | log s | ψ (°) | polygon max abs coord in patch frame (µm) | patch max abs diff | EFD-16 RMS (poly→curve / curve→poly) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in checks.items():
        lines.append(
            f"| {name} | {c['nucleus_index']} | ({c['centroid_px'][0]:.2f}, {c['centroid_px'][1]:.2f}) "
            f"| {c['log_s']:.3f} | {c['psi_deg']:.1f} | {c['polygon_max_abs_coord_in_patch_frame_um']:.2f} "
            f"| {c['patch_max_abs_diff_vs_dataset']:.1e} | {c['efd16_rms_polygon_to_curve']:.4f} / {c['efd16_rms_curve_to_polygon']:.4f} |"
        )
    lines += [
        "",
        "## Panels",
        "",
        f"- (a) {CROP_UM:g} × {CROP_UM:g} µm level-0 H&E crop around the centroid; the highlighted nucleus "
        "as LSP-DETR represents it: a star-convex polygon (yellow) through the ends of 64 evenly "
        "spaced rays (white) from the point p (orange; recovered from the vertices, as the stored "
        "centroid is the vertex mean); other nuclei as thin white outlines; dashed: the 16 × 16 µm texture window, rotated by ψ (its corners are the "
        "patch's outer pixel edges mapped back through the inverse of `nucleus_patch`'s affine).",
        "- (b) Inverse-EFD reconstructions of orders 1 and 16 (light → dark blue) from the "
        "features of `NucleiFeatureExtractor.get_efd_features` (order 16: start point, rotation, "
        "scale normalized), over the polygon as a faint grey line, normalized the same way: "
        "centred on the EFD DC term, rotated by −ψ and divided by s. s is the "
        "first-harmonic semi-major axis in µm, so order 1 is the ellipse with semi-axis 1 "
        "along x. The y axis points down, as in the image and the patch.",
        "- (c) The 48 × 48 patch of `NucleiFeatureExtractor.nucleus_patch` (masked 1 µm beyond the "
        "polygon, rotated by −ψ, bilinear), shown with nearest-neighbour interpolation "
        "(each pixel drawn as an 8 × 8 block, so PDF viewers do not smooth it).",
        "- The blank token has no panel: it is the same learned vector for every nucleus, so "
        "only the relative positions of the k-NN neighbours (shared by all three models) remain.",
        "",
        "## Input normalization",
        "",
        "The dataset patch is a uint8 RGB tensor; the model normalizes it in "
        "`Transformer.embed_patches` as `x / 127.5 - 1` (to [-1, 1]). For display that "
        "normalization is undone (`(x + 1) * 127.5`), which gives back the uint8 patch exactly. "
        "No other normalization (e.g. ImageNet mean/std, stain normalization) is applied to the input.",
        "",
        "## Checks",
        "",
        "- Panel (c) equals the dataset's own patch: `NucleiClassificationDataset.generate_embeddings` "
        "in `bbox` mode on the full slide (max abs diff < 1e-6).",
        "- The overlay affine reproduces `nucleus_patch`'s warp exactly on the unmasked pixels.",
        "- Orientation of (b) vs (c): the order-16 reconstruction of (b), scaled back to µm and "
        "drawn into the patch (dilated by the 1 µm mask margin), as computed, rotated by 180°, "
        "y flipped and x flipped; (b) is displayed in the orientation whose IoU with the "
        "patch's nucleus is highest (> 0.9): "
        + ", ".join(f"{name}: {c['b_displayed_as']}" for name, c in checks.items())
        + " (`embeddings_orientation_check.png`).",
        "- EFD-16 reconstruction: RMS nearest-point distance to the normalized polygon < 0.05 "
        "in both directions.",
        "- Ray origin: the 64 vertices lie on rays at 360°/64 around p within "
        + ", ".join(f"{c['max_ray_angle_deviation_around_p_deg']:.4f}°" for c in checks.values())
        + " (around the stored centroid: "
        + ", ".join(f"{c['max_ray_angle_deviation_around_stored_centroid_deg']:.2f}°" for c in checks.values())
        + ").",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="log the outputs to MLflow")
    args = parser.parse_args()

    meta = pd.read_parquet(download_artifacts(METADATA_URI))
    slide = meta.set_index("slide_id", drop=False).loc[SLIDE_ID]

    rows, checks = prepare(slide)
    for name, c in checks.items():
        print(name, json.dumps(c, indent=1))

    OUT_DIR.mkdir(exist_ok=True)
    fig = draw(rows)
    fig.savefig(OUT_DIR / "embeddings.pdf")
    fig.savefig(OUT_DIR / "embeddings.png", dpi=300)
    draw_orientation_check(rows).savefig(OUT_DIR / "embeddings_orientation_check.png", dpi=200)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (OUT_DIR / "embeddings_checks.json").write_text(json.dumps(checks, indent=2))
    write_readme(OUT_DIR / "README.md", slide, checks, commit)

    if args.log:
        mlflow.set_experiment(experiment_id="37")
        with mlflow.start_run(run_name="Thesis figure: embeddings") as run:
            mlflow.set_tag("git_commit", commit)
            mlflow.log_params(
                {
                    "slide_id": SLIDE_ID,
                    "efd_order": EFD_ORDER,
                    "patch_size": PATCH_SIZE,
                    "target_bbox_context_um": TARGET_BBOX_CONTEXT_UM,
                    **{f"nucleus_index/{k.split(' (')[0]}": v for k, v in NUCLEI.items()},
                }
            )
            for name in [
                "embeddings.pdf",
                "embeddings.png",
                "embeddings_orientation_check.png",
                "README.md",
                "embeddings_checks.json",
            ]:
                mlflow.log_artifact(str(OUT_DIR / name))
            mlflow.log_artifact(__file__)
            print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
