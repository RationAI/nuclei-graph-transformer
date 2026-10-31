"""Thesis figure: what each nucleus embedding gives the model.

One nucleus of a PANDA Radboud test slide (the elongated one of the architecture
panel), as
(a) the H&E input with the polygon outline and the rotated texture window,
(b) the contour and its order-16 inverse-EFD reconstruction, normalised (shape embedding),
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
import functools
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
SLIDE_ID = "89b62c165838d36fd294a57624db8e4f"  # PANDA Radboud test, Gleason 4+3
# row index into the slide's nuclei sorted by "id" (as in the datasets): the elongated
# nucleus of the architecture panel (thesis_architecture_panel_a.choose_nucleus: axis
# ratio 2-3, |ψ| >= 15°, nucleus id a61d2eee...), annotated non-carcinoma
NUCLEI = {
    "non-carcinoma (elongated)": 12674,
}
NUCLEUS_IDS = {"non-carcinoma (elongated)": "a61d2eee54cf2e53422a184f16e42d5ac2b44603fe44b05a0fe39c4b81ae74c8"}
MAX_AXIS_ANGLE_DEG = 10.0  # major axis of the patch's nucleus to the horizontal

EFD_ORDER = 16  # as in training (configs/experiment/modeling/training/nuclei_level/radboud.yaml)
PATCH_SIZE = 48
RECON_ORDERS = (1, 16)
CROP_UM = 40.0  # read around the centroid; (a) shows the rotated window plus VIEW_MARGIN_UM
VIEW_MARGIN_UM = 0.6
HE_UPSAMPLE = 8  # (a) is shown bilinearly upsampled (display only)
CLOSEUP_UM = 12.0  # close-up preview of (a) around the centroid
SCALE_BAR_OPTIONS_UM = (10.0, 5.0)  # the first that keeps clear of the window

COLUMNS = ("(a) Input", "(b) Shape (normalised)", "(c) Texture")
FIG_W_IN = 16 / 2.54
PANEL_IN = 1.95
GAP_IN = 0.2
ROW_GAP_IN = 0.1
TOP_IN = 0.22
BOTTOM_IN = 0.17  # room for the patch size below (c)

POLYGON_COLOR, POLYGON_LW = "#FFD700", 0.8  # pt
POLYGON_EDGE, POLYGON_EDGE_LW = "#1a1a1a", 0.3  # thin dark outline around the yellow line
# contour in (c) to the mask edge, px: the margin is 2 slide px (0.97 µm = 2.9 patch px),
# ± the pixel staircase of the dilated mask and the bilinear edge
CONTOUR_GAP_RANGE_PX = (1.5, 4.5)
MAX_RAY_ANGLE_DEV_DEG = 0.05  # vertices on the 64 evenly spaced rays from p
WINDOW_LW, WINDOW_DASH = 0.6, (0, (3, 2))  # the 16 µm window in (a) and the frame of (c)
# orientation in (a): white, thin dark outline; lengths are fractions of s
ANNOT_LW, ANNOT_EDGE_LW = 0.7, 0.3  # pt
AXIS_LW = 0.8  # pt, the semi-major half-axis
PSI_NEAR_RATIO = 0.6  # ψ label to the arc <= this times its distance to the half-axis end
REFERENCE_LEN = 0.8
ARC_RADIUS = 0.5
LABEL_PAD_PT = 0.5  # labels to every line
SCALE_BAR_LW = 1.5
TEXT_STROKE = 1.0  # black outline of the white scale bar and its label (half shows)
ACCENT_COLOR = "#E4572E"
CONTOUR_B_LW = 1.4  # pt, the normalised contour in (b), under the EFD curve
EFD_COLOR, EFD_LW = "#1F5FA8", 0.8  # (b), the order-16 reconstruction

# the thesis font: no LaTeX (and no Latin Modern) here, so Computer Modern via cmr10 and
# mathtext "cm"; cmr10 has no µ, ×, →, ° glyphs, so those are set as math
THESIS_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}
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
            assert nuclei["id"].iloc[idx] == NUCLEUS_IDS[name]
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

            # the contour in the patch, through the same affine (patch px, pixel centres
            # at integers), and its gap to the edge of the unmasked region (the union of
            # the pixels that are not pure white), which the 1 µm mask margin should make
            # ≈ 1 µm = 3 px all around
            contour_px = (polygon.astype(np.float64) - [box.lx, box.ly]) @ affine[:, :2].T + affine[:, 2]
            unmasked = shapely.union_all(
                [shapely.box(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for y, x in zip(*np.nonzero(kept))]
            )
            ring = shapely.LinearRing(contour_px)
            assert unmasked.contains(ring), "contour leaves the unmasked region"
            points = shapely.points(densify(np.vstack([contour_px, contour_px[:1]])))
            gap_px = shapely.distance(points, unmasked.boundary)
            assert CONTOUR_GAP_RANGE_PX[0] <= gap_px.min() and gap_px.max() <= CONTOUR_GAP_RANGE_PX[1], (
                gap_px.min(), gap_px.max())

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

            # polygon extent in the patch frame: with the mask margin it has to stay inside
            # the 16 µm window (half-width 8 µm), or (c) is clipped
            poly_patch_um = (contour - centroid * mpp) @ rotation.T
            extent_um = float(np.abs(poly_patch_um).max())
            assert extent_um + BBOX_MASK_MARGIN_UM < TARGET_BBOX_CONTEXT_UM / 2, extent_um

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
            # (b) and (c) share ψ, so they can differ only by 180°; a mirror would be a bug
            assert orientation in ("as computed", "rotated 180°"), orientation_iou
            # major axis horizontal in (c): second moments of the patch's nucleus (in (b)
            # it is along x by construction, the order-1 ellipse's semi-major axis)
            yy, xx = np.nonzero(nucleus)
            dx, dy = xx - xx.mean(), yy - yy.mean()
            patch_axis_deg = float(np.degrees(0.5 * np.arctan2(2 * (dx * dy).mean(),
                                                               (dx**2).mean() - (dy**2).mean())))
            assert abs(patch_axis_deg) < MAX_AXIS_ANGLE_DEG, patch_axis_deg

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
                    "efd_centre_um": efd_centre(contour) - centroid * mpp,
                    "patch": display_patch,
                    "contour_patch_px": contour_px,
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
                "mask_max_abs_coord_in_patch_frame_um": extent_um + BBOX_MASK_MARGIN_UM,
                "mask_inside_16um_window": extent_um + BBOX_MASK_MARGIN_UM < TARGET_BBOX_CONTEXT_UM / 2,
                "patch_max_abs_diff_vs_dataset": patch_diff,
                "contour_in_patch_inside_unmasked_region": True,
                "contour_gap_to_mask_edge_px": {
                    "min": float(gap_px.min()), "max": float(gap_px.max()),
                    "mean": float(gap_px.mean()), "median": float(np.median(gap_px)),
                    "unmasked_region_parts": len(shapely.get_parts(unmasked)),
                    "unmasked_region_holes": int(sum(shapely.get_num_interior_rings(shapely.get_parts(unmasked)))),
                },
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
                "patch_major_axis_to_horizontal_deg": patch_axis_deg,
            }
    return rows, checks


@functools.cache
def ink_offsets(string: str, fontsize: float, ha: str, va: str) -> NDArray[np.float64]:
    """The glyphs' bounding box (x0, y0, x1, y1) in pt relative to the text's anchor,
    from a render of the text alone (the line box of mathtext is taller than the glyphs)."""
    dpi, size = 720, 72.0
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.axis("off")
    ax.text(size / 2, size / 2, string, fontsize=fontsize, ha=ha, va=va, color="black")
    fig.canvas.draw()
    ys, xs = np.nonzero(np.asarray(fig.canvas.buffer_rgba())[..., 0] < 128)
    plt.close(fig)
    k = dpi / 72
    return np.array([xs.min() / k, size - (ys.max() + 1) / k, (xs.max() + 1) / k, size - ys.min() / k]) - size / 2


def label_box(ax, text: plt.Text) -> shapely.Polygon:
    """The bounding box of the text's glyphs (not its line box) in data coordinates (µm)."""
    x0, y0, x1, y1 = ink_offsets(text.get_text(), text.get_fontsize(), text.get_ha(), text.get_va())
    anchor = ax.transData.transform(text.get_position())
    k = ax.figure.dpi / 72
    corners = anchor + np.array([[x0, y0], [x1, y1]]) * k
    (u0, v0), (u1, v1) = ax.transData.inverted().transform(corners)
    return shapely.box(min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1))


def place_label(ax, text: plt.Text, positions, obstacles: dict, pad: float,
                accept=lambda box: True) -> dict | None:
    """Moves the text to the first position whose box keeps `pad` (µm) plus each
    obstacle's half line width from every obstacle and passes `accept`; None (text
    removed) if none does."""
    for xy in positions:
        text.set_position(xy)
        box = label_box(ax, text)
        gaps = {name: float(box.distance(geom)) - half for name, (geom, half) in obstacles.items()}
        if min(gaps.values()) >= pad and accept(box):
            return {"position_um": [float(v) for v in xy], "box": box,
                    "clearance_um": {k: round(v, 3) for k, v in gaps.items()}}
    text.remove()
    return None


def draw_input(ax, row: dict, scale_bar: bool) -> dict:
    mpp, centroid, box = row["mpp"], row["centroid"], row["crop_box"]
    # pixel i is centred at coordinate lx + i (the convention nucleus_patch uses)
    left, top = (np.array([box.lx, box.ly]) - 0.5 - centroid) * mpp
    right, bottom = (np.array([box.rx, box.ry]) - 0.5 - centroid) * mpp
    # display only: upsampled bilinearly (pixel centres kept), embedded unresampled
    crop = cv2.resize(row["crop"], None, fx=HE_UPSAMPLE, fy=HE_UPSAMPLE, interpolation=cv2.INTER_LINEAR)
    ax.imshow(crop, extent=(left, right, bottom, top), interpolation="none")

    def to_um(p: NDArray) -> NDArray:
        return (p - centroid) * mpp

    # the highlighted nucleus's outline, and the texture window
    vertices = to_um(row["polygon"].astype(np.float64))
    ax.plot(*np.vstack([vertices, vertices[:1]]).T, color=POLYGON_COLOR, lw=POLYGON_LW, zorder=4,
            path_effects=[pe.withStroke(linewidth=POLYGON_LW + 2 * POLYGON_EDGE_LW,
                                        foreground=POLYGON_EDGE)])
    win = to_um(np.vstack([row["window_px"], row["window_px"][:1]]))
    ax.plot(*win.T, color="black", lw=WINDOW_LW, ls=WINDOW_DASH, zorder=5)

    # the view: the rotated window plus a margin
    h = float(np.abs(win).max()) + VIEW_MARGIN_UM
    assert h <= CROP_UM / 2
    ax.set_xlim(-h, h)
    ax.set_ylim(h, -h)
    ax.set_aspect("equal")
    ax.axis("off")
    um_per_pt = 2 * h / (PANEL_IN * 72)

    # s and ψ: the order-1 ellipse's semi-major half-axis from the EFD centre (A0, C0) to
    # the ellipse (length s, along ψ), the horizontal reference and the arc between them.
    # ψ is modulo 180° in (-90°, 90°], so the half-axis (cos ψ, sin ψ) points right; y is
    # down, positive ψ is clockwise
    centre, s = row["efd_centre_um"], np.exp(row["log_s"])
    psi = 0.5 * np.arctan2(np.sin(2 * row["psi"]), np.cos(2 * row["psi"]))
    direction = np.array([np.cos(psi), np.sin(psi)])
    half_axis = np.array([centre, centre + s * direction])
    reference = np.array([centre, centre + [REFERENCE_LEN * s, 0.0]])
    t = np.linspace(0.0, psi, 64)
    arc = centre + ARC_RADIUS * s * np.stack([np.cos(t), np.sin(t)], axis=1)
    style = {"color": "white", "zorder": 6, "solid_capstyle": "butt",
             "path_effects": [pe.withStroke(linewidth=ANNOT_LW + 2 * ANNOT_EDGE_LW,
                                            foreground=POLYGON_EDGE)]}
    ax.plot(*half_axis.T, **(style | {"lw": AXIS_LW, "path_effects": [pe.withStroke(
        linewidth=AXIS_LW + 2 * ANNOT_EDGE_LW, foreground=POLYGON_EDGE)]}))
    ax.plot(*reference.T, **(style | {"lw": ANNOT_LW, "ls": (0, (1, 1.2))}))
    ax.plot(*arc.T, **(style | {"lw": ANNOT_LW}))

    # labels keep LABEL_PAD_PT plus their outline from every line (and its outline)
    pad = (LABEL_PAD_PT + TEXT_STROKE / 2) * um_per_pt
    annot_half = (ANNOT_LW / 2 + ANNOT_EDGE_LW) * um_per_pt
    nucleus = shapely.Polygon(vertices)
    obstacles = {
        "contour": (nucleus.exterior, (POLYGON_LW / 2 + POLYGON_EDGE_LW) * um_per_pt),
        "window": (shapely.LinearRing(win[:-1]), WINDOW_LW / 2 * um_per_pt),
        "half_axis": (shapely.LineString(half_axis), (AXIS_LW / 2 + ANNOT_EDGE_LW) * um_per_pt),
        "reference": (shapely.LineString(reference), annot_half),
        "arc": (shapely.LineString(arc), annot_half),
    }
    label_style = {"ha": "center", "va": "center", "fontsize": 8, "color": "white", "zorder": 7,
                   "path_effects": [pe.withStroke(linewidth=TEXT_STROKE, foreground="black")]}

    # ψ: just outside the arc at its angular midpoint, clearly closer to the arc than to
    # the end of the half-axis; if no such position is clear of the lines (the wedge is
    # narrow for small ψ), the clear position outside the arc nearest to its midpoint
    arc_line, axis_end = shapely.LineString(arc), shapely.Point(half_axis[1])

    def near_arc(box: shapely.Polygon) -> bool:
        return box.distance(arc_line) <= PSI_NEAR_RATIO * box.distance(axis_end)

    mid = np.array([np.cos(psi / 2), np.sin(psi / 2)])
    arc_mid = centre + ARC_RADIUS * s * mid
    psi_text = ax.text(0, 0, r"$\psi$", **label_style)
    radii = ARC_RADIUS * s + np.arange(0.0, 3.0, 0.02)
    psi_label = place_label(ax, psi_text, [centre + r * mid for r in radii], obstacles, pad, near_arc)
    psi_placement = "angular midpoint of the arc"
    if psi_label is None:
        psi_text = ax.text(0, 0, r"$\psi$", **label_style)
        angles = np.radians(np.arange(-180.0, 180.0, 2.0))
        ring = [centre + r * np.array([np.cos(a), np.sin(a)]) for r in radii for a in angles]
        ring.sort(key=lambda xy: float(np.linalg.norm(xy - arc_mid)))
        psi_label = place_label(ax, psi_text, ring, obstacles, pad, near_arc)
        psi_placement = "nearest clear position outside the arc (the midpoint has none)"
    assert psi_label is not None, "no clear position for the psi label"
    psi_box = psi_label.pop("box")
    psi_label |= {"placement": psi_placement,
                  "distance_to_arc_um": round(float(psi_box.distance(arc_line)), 3),
                  "distance_to_half_axis_end_um": round(float(psi_box.distance(axis_end)), 3)}

    scale = None
    if scale_bar:
        # white bar and label, each with a thin black outline; no box behind them; the
        # longest of SCALE_BAR_OPTIONS_UM that keeps clear of the window
        outline = [pe.withStroke(linewidth=SCALE_BAR_LW + TEXT_STROKE, foreground="black")]
        x0, y0 = -h + 0.12 * 72 * um_per_pt, h - 0.12 * 72 * um_per_pt
        for length in SCALE_BAR_OPTIONS_UM:
            bar = ax.plot([x0, x0 + length], [y0, y0], color="white", lw=SCALE_BAR_LW,
                          solid_capstyle="butt", path_effects=outline)[0]
            text = ax.text(x0 + length / 2, y0 - 2 * um_per_pt, rf"${length:g}\,\mu\mathrm{{m}}$",
                           ha="center", va="bottom", fontsize=8, color="white",
                           path_effects=[pe.withStroke(linewidth=TEXT_STROKE, foreground="black")])
            bar_geom = shapely.LineString([[x0, y0], [x0 + length, y0]]).buffer(
                (SCALE_BAR_LW + TEXT_STROKE) / 2 * um_per_pt, cap_style="flat")
            gap = min(bar_geom.distance(obstacles["window"][0]),
                      label_box(ax, text).distance(obstacles["window"][0])) - obstacles["window"][1]
            if gap >= pad:
                scale = {"length_um": length, "clearance_to_window_um": round(float(gap), 3)}
                break
            bar.remove()
            text.remove()
        assert scale is not None, "no scale bar fits beside the window"
    return {
        "view_um": 2 * h,
        "efd_centre_um": [float(v) for v in centre],
        "s_um": float(s),
        "psi_display_deg": float(np.degrees(psi)),
        "psi_label": psi_label,
        "scale_bar": scale,
    }


def draw_shape(ax, row: dict, lim: float, legend: bool) -> None:
    # displayed as the patch of (c) is oriented (see the orientation check in `prepare`)
    flip = row["b_display"]
    # the contour, normalised exactly like the EFD (centred on its DC term, rotated by -ψ,
    # divided by s), under the order-16 reconstruction
    contour = np.vstack([row["poly_norm"], row["poly_norm"][:1]]) * flip
    ax.plot(*contour.T, color=POLYGON_COLOR, lw=CONTOUR_B_LW, zorder=2, label="contour",
            path_effects=[pe.withStroke(linewidth=CONTOUR_B_LW + 2 * POLYGON_EDGE_LW,
                                        foreground=POLYGON_EDGE)])
    curve = np.vstack([row["recons"][EFD_ORDER], row["recons"][EFD_ORDER][:1]]) * flip
    ax.plot(*curve.T, color=EFD_COLOR, lw=EFD_LW, zorder=3, label=f"EFD, order {EFD_ORDER}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)  # y down, as in the patch
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME_GREY)
        spine.set_linewidth(0.4)
    if legend:
        ax.legend(loc="upper center", ncols=2, fontsize=8, frameon=False, handlelength=1.4,
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0.3)


def draw_texture(ax, row: dict) -> None:
    # "none" embeds the pixels unresampled in the PDF; repeated so that viewers which
    # smooth embedded images keep them sharp (nearest-neighbour display)
    blocks = np.repeat(np.repeat(row["patch"], 8, axis=0), 8, axis=1)
    edge = PATCH_SIZE - 0.5
    ax.imshow(blocks, interpolation="none", extent=(-0.5, edge, edge, -0.5))
    # the contour through nucleus_patch's affine, styled as in (a)
    contour = np.vstack([row["contour_patch_px"], row["contour_patch_px"][:1]])
    ax.plot(*contour.T, color=POLYGON_COLOR, lw=POLYGON_LW, zorder=4,
            path_effects=[pe.withStroke(linewidth=POLYGON_LW + 2 * POLYGON_EDGE_LW,
                                        foreground=POLYGON_EDGE)])
    ax.set_xlim(-0.5, edge)
    ax.set_ylim(edge, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # the frame is the texture window of (a), drawn the same way
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(WINDOW_LW)
        spine.set_linestyle(WINDOW_DASH)
    ax.annotate(rf"${TARGET_BBOX_CONTEXT_UM:g} \times {TARGET_BBOX_CONTEXT_UM:g}\,\mu\mathrm{{m}} \rightarrow "
                rf"{PATCH_SIZE} \times {PATCH_SIZE}\,\mathrm{{px}}$",
                xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -3), textcoords="offset points",
                ha="center", va="top", fontsize=8)


def draw(rows: list[dict]) -> plt.Figure:
    n = len(rows)
    fig_h = TOP_IN + n * PANEL_IN + (n - 1) * ROW_GAP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    n_cols = len(COLUMNS)
    left0 = (FIG_W_IN - n_cols * PANEL_IN - (n_cols - 1) * GAP_IN) / 2

    def rect(col: int, row: int, x=0.0, y=0.0, w=1.0, h=1.0) -> list[float]:
        """Axes rect of a panel (fractions x, y, w, h of the panel, y from its bottom)."""
        px = left0 + col * (PANEL_IN + GAP_IN) + x * PANEL_IN
        py = fig_h - TOP_IN - (row + 1) * PANEL_IN - row * ROW_GAP_IN + y * PANEL_IN
        return [px / FIG_W_IN, py / fig_h, w * PANEL_IN / FIG_W_IN, h * PANEL_IN / fig_h]

    lim = 1.08 * max(np.abs(np.concatenate([r["poly_norm"], r["recons"][EFD_ORDER]])).max() for r in rows)
    lim = max(lim, 1.3)  # room for the legend and the label above / below the shape

    for i, row in enumerate(rows):
        row["panel_a"] = draw_input(fig.add_axes(rect(0, i)), row, scale_bar=i == 0)
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


def write_readme(path: Path, slide: pd.Series, checks: dict, figure_checks: dict, commit: str) -> None:
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
        "| row | nucleus index | centroid (level-0 px) | log s | ψ (°) | polygon / mask (+1 µm) max abs coord in patch frame (µm) | patch max abs diff | EFD-16 RMS (poly→curve / curve→poly) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in checks.items():
        lines.append(
            f"| {name} | {c['nucleus_index']} | ({c['centroid_px'][0]:.2f}, {c['centroid_px'][1]:.2f}) "
            f"| {c['log_s']:.3f} | {c['psi_deg']:.1f} | {c['polygon_max_abs_coord_in_patch_frame_um']:.2f} / {c['mask_max_abs_coord_in_patch_frame_um']:.2f} "
            f"| {c['patch_max_abs_diff_vs_dataset']:.1e} | {c['efd16_rms_polygon_to_curve']:.4f} / {c['efd16_rms_curve_to_polygon']:.4f} |"
        )
    lines += [
        "",
        "## Panels",
        "",
        "- (a) Level-0 H&E around the centroid, cropped to the rotated texture window plus "
        f"{VIEW_MARGIN_UM:g} µm ("
        + ", ".join(f"{c['panel_a_orientation']['view_um']:.1f} µm" for c in checks.values())
        + "); the highlighted nucleus's outline (the LSP-DETR polygon, yellow); dashed: the "
        "16 × 16 µm texture window, rotated by ψ (its corners are the patch's outer pixel edges "
        "mapped back through the inverse of `nucleus_patch`'s affine). s and ψ (white, thin dark "
        "outline): the order-1 ellipse's semi-major half-axis from the EFD centre (A0, C0) to the "
        f"ellipse (length s, along ψ), the dotted horizontal reference ({REFERENCE_LEN:g} s) and "
        f"the arc between them ({ARC_RADIUS:g} s). Labels clear every line by ≥ {LABEL_PAD_PT:g} pt "
        "plus the outlines (measured on the glyphs' rendered extent); ψ outside the arc, at most "
        f"{PSI_NEAR_RATIO:g}× as far from the arc as from the half-axis end ("
        + ", ".join(c["panel_a_orientation"]["psi_label"]["placement"] for c in checks.values())
        + "); the half-axis is not labelled. The H&E is shown bilinearly upsampled "
        f"×{HE_UPSAMPLE} (display only). `embeddings_panel_a_closeup.png`: a "
        f"{CLOSEUP_UM:g} µm close-up of (a) at 1200 dpi.",
        "- (b) The order-16 inverse-EFD reconstruction (blue, 0.8 pt) from the features of "
        "`NucleiFeatureExtractor.get_efd_features` (start point, rotation, scale normalized), on "
        "top of the nucleus contour normalised the same way (yellow, 1.4 pt, thin dark outline): "
        "centred on the EFD DC term, rotated by −ψ and divided by s (the first-harmonic "
        "semi-major axis in µm). The y axis points down, as in the image and the patch.",
        "- (c) The 48 × 48 patch of `NucleiFeatureExtractor.nucleus_patch` (masked 1 µm beyond the "
        "polygon, rotated by −ψ, bilinear), shown with nearest-neighbour interpolation "
        "(each pixel drawn as an 8 × 8 block, so PDF viewers do not smooth it). Its dashed frame "
        "is the dashed window of (a). The yellow contour is the polygon mapped through the same "
        "affine (rotation by −ψ around the centroid, 48 px per 16 µm). It lies inside the unmasked "
        "region (the pixels that are not pure white); its gap to that region's edge is "
        + ", ".join(f"{c['contour_gap_to_mask_edge_px']['min']:.2f}–{c['contour_gap_to_mask_edge_px']['max']:.2f} px "
                    f"(mean {c['contour_gap_to_mask_edge_px']['mean']:.2f})" for c in checks.values())
        + f", checked to lie in {CONTOUR_GAP_RANGE_PX[0]:g}–{CONTOUR_GAP_RANGE_PX[1]:g} px "
        "(the 1 µm margin is 2 slide px = 0.97 µm = 2.9 patch px).",
        "- The blank token has no panel: it is the same learned vector for every nucleus, so "
        "only the relative positions of the k-NN neighbours (shared by all three models) remain.",
        "",
        "## Typesetting",
        "",
        f"Figure {figure_checks['pdf_width_cm']:.1f} × {figure_checks['pdf_height_cm']:.2f} cm (PDF, unscaled). "
        "All text in Computer Modern: cmr10 with the mathtext `cm` font set (no LaTeX or Latin "
        "Modern is installed here); µ, ×, → and ψ are set as math. PDF fonts: "
        + ", ".join(f"`{f}`" for f in figure_checks["pdf_fonts"]) + ".",
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
        + " (`embeddings_orientation_check.png`). As (b) and (c) share ψ, only 180° may differ; "
        "a mirror fails the check. The major axis of the patch's nucleus (second moments) is within "
        f"{MAX_AXIS_ANGLE_DEG:g}° of the horizontal: "
        + ", ".join(f"{c['patch_major_axis_to_horizontal_deg']:.1f}°" for c in checks.values())
        + ".",
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
    with mpl.rc_context(THESIS_RC):  # fonts resolve at draw time: save inside
        fig = draw(rows)
        fig.savefig(OUT_DIR / "embeddings.pdf")  # at print size, no bbox adjustment
        fig.savefig(OUT_DIR / "embeddings.png", dpi=300)
        # close-up of the nucleus in (a), from the print-size figure
        ax_a = fig.axes[0]
        corners = ax_a.transData.transform([[-CLOSEUP_UM / 2, CLOSEUP_UM / 2], [CLOSEUP_UM / 2, -CLOSEUP_UM / 2]])
        closeup = mpl.transforms.Bbox(fig.dpi_scale_trans.inverted().transform(corners))
        fig.savefig(OUT_DIR / "embeddings_panel_a_closeup.png", dpi=1200, bbox_inches=closeup)
        text_fonts = sorted({text.get_fontname() for text in fig.findobj(mpl.text.Text) if text.get_text()})
    for row in rows:
        checks[row["name"]]["panel_a_orientation"] = row["panel_a"]
    # the PDF: 16 cm wide, only Computer Modern fonts embedded
    pdf = (OUT_DIR / "embeddings.pdf").read_bytes()
    media = [float(v) for v in re.search(rb"/MediaBox \[\s*([\d.\s]+)\]", pdf).group(1).split()]
    width_cm = (media[2] - media[0]) / 72 * 2.54
    pdf_fonts = sorted({f.decode() for f in re.findall(rb"/BaseFont /([\w+-]+)", pdf)})
    assert abs(width_cm - 16.0) < 0.01, width_cm
    assert text_fonts == ["cmr10"], text_fonts
    assert all(f.split("+")[-1].lower().startswith("cm") for f in pdf_fonts), pdf_fonts
    figure_checks = {"pdf_width_cm": round(width_cm, 3),
                        "pdf_height_cm": round((media[3] - media[1]) / 72 * 2.54, 3),
                        "text_font": "cmr10 (mathtext fontset cm); no LaTeX/Latin Modern available",
                        "pdf_fonts": pdf_fonts}
    draw_orientation_check(rows).savefig(OUT_DIR / "embeddings_orientation_check.png", dpi=200)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (OUT_DIR / "embeddings_checks.json").write_text(json.dumps(checks | {"figure": figure_checks}, indent=2))
    write_readme(OUT_DIR / "README.md", slide, checks, figure_checks, commit)

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
                "embeddings_panel_a_closeup.png",
                "README.md",
                "embeddings_checks.json",
            ]:
                mlflow.log_artifact(str(OUT_DIR / name))
            mlflow.log_artifact(__file__)
            print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
