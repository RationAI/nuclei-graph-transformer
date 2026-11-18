r"""Input perturbations of the stable concepts (causal test of their morphometry).

Does changing one geometric property of the input change the concept activations as the
morphometric interpretation predicts?

Objects: model run c8f26853 (frozen; PANDA Radboud, polygons + positions, RoVE, k-NN
attention, k = 24), the k = 16 seed-1 L1 SAE (SAE training run 4cf1db8f; deck key: A = #0,
B = #1, C = #8, D = #12, E = #3, F = #6, G = #2, H = #4) and the PANDA Radboud test slides.

Targets, per stable concept j (κ_j = the stored whole-slide codes `codes_test.npy`):
"top" = its `N_TARGETS` most activated test nuclei, at most `MAX_PER_SLIDE` per slide (as
`xai.concept_material.top_nuclei`); controls "active" = `N_TARGETS` random nuclei with
κ_j > 0 and "inactive" = `N_TARGETS` random nuclei with κ_j = 0. A nucleus chosen for
several (concept, group) pairs is run once.

Context: the model runs on a crop of the `CROP_SIZE` nuclei nearest to the target (the
target included; centroid distance), the same nuclei for the unperturbed and every perturbed
run, built as the prediction dataset builds a slide (`PredictionDataset`: centroids of the
nuclei table in µm, EFD features of the polygons by `NucleiFeatureExtractor.
get_efd_features`, positions centred, `GraphCollator(predict=True)` with the training
k = 24 and block size 128, k-NN block mask built on the GPU, TF32 matmuls; all runs of a
target are centred on its unperturbed crop's mean, as RoPE only sees differences). h of the target
(output of `final_norm`) is encoded by the SAE (float32, on the GPU); z = class_head(h).

Perturbations (each alone, `RADIUS_UM` = 50 µm around the target, "neighbours" exclude the
target; centroid distances in the unperturbed crop):
  spread / compress   (P1) offsets of all nuclei within 50 µm from the target × 1.5 / × 0.75
                      (centroids and polygons translated, shapes kept)
  nbr_ellipse         (P2) neighbours' contours → their first-harmonic ellipse
  own_ellipse / own_circle (P3) the target's contour → its first-harmonic ellipse / a circle
  nbr_rotate          (P4) each neighbour rotated about its centroid by a uniform random angle
  remove_8nn          (P5) the 8 nearest neighbours deleted
The ellipse has the first EFD harmonic's axis ratio and orientation, the polygon's area and is
centred on the nucleus centroid; the circle has the polygon's area. Both have the polygon's
vertex count and winding. Two properties of the model's features matter here (see the tests):
the EFD is arc-length based, so the first harmonic of an elongated nucleus is rounder than its
second-moment ellipse (a 2:1 ellipse's major axis shrinks by ~8 %) — "→ ellipse" also makes
elongated nuclei a little rounder; the target's moment axis ratio before and after is
recorded (`target_axis_ratio`). And the EFD normalisation leaves a 180° ambiguity that
depends on the absolute orientation, so rotating a nucleus (P4) can flip the sign of its even
harmonics besides changing (cos 2ψ, sin 2ψ). Features of every changed nucleus are recomputed from its new
polygon with `get_efd_features` (the features are per-nucleus, see the tests).

Block partition. The k-NN block mask lets a block of 128 nuclei attend to every block that
holds a k-NN of one of its members, and the blocks come from a spatial sort of the positions.
Moving or deleting nuclei could re-sort the whole crop, so the perturbed runs reuse the
unperturbed crop's order (P5: without the deleted nuclei); the k-NN graph is recomputed from
the perturbed positions. As a check, P1 and P5 are also run with the crop re-sorted
(`*_resort`).

Statistics per (concept, group, perturbation), over targets, with 95 % percentile intervals
from a bootstrap over slides (`N_BOOT` resamples): mean Δκ_j = κ_j(perturbed) − κ_j(unperturbed
crop); relative change = Σ Δκ_j / Σ κ_j (ratio of means; undefined for "inactive"); mean Δz;
switch-off share = share of the targets with κ_j > 0 in the unperturbed crop that have κ_j = 0
after the perturbation; switch-on share likewise for κ_j = 0 before. Hypotheses (from the
thesis signature, fixed before the run) are judged on the "top" targets by the interval of the
mean Δκ_j: "holds" if it lies entirely on the predicted side, "opposite" if entirely on the
other, else "not supported".

Check: on `N_CHECK` random targets the unperturbed crop's codes are compared with the stored
whole-slide ones (Pearson r over all 16 codes and of the logits; same for all targets).

Outputs (no slide IDs; slides are numbered by their position among the test slides):
`OUT_DIR/{targets.parquet, raw/, raw.parquet, perturbation_summary.csv, hypotheses.csv,
check.json, tab_res_concept_perturbation.tex, note.md}`, `figures/res_concept_perturbation
.{pdf,png}` and `figures/res_concept_perturbation_stats.json`. Not logged to MLflow.

    PYTHONPATH=. python xai/concept_perturbation.py --stage check   # 50 targets, gate
    PYTHONPATH=. python xai/concept_perturbation.py --stage run     # all targets (resumable)
    PYTHONPATH=. python xai/concept_perturbation.py --stage report  # tables, figure, note
"""

import argparse
import json
import math
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from numpy.typing import NDArray

from nuclei_graph.data.block_mask import (
    BlockMask,
    _MaskMod,
    pack_and_shift_knn_indices,
)
from nuclei_graph.data.datamodules.collator import GraphCollator
from nuclei_graph.data.datasets.crop.base import BaseCropDataset
from nuclei_graph.nuclei_graph_typing import Sample


# --- inputs ---

MODEL_RUN = "c8f268536be8416f9fe395eea95c458e"
SAE_TRAINING_RUN = "4cf1db8f2cff4fef844426e57a387b99"
XAI_DIR = Path(f"/mnt/public_projects/nuclei_based_wsi_analysis/xai/{MODEL_RUN}")
CONCEPT_DIR = XAI_DIR / "concepts" / "k16_seed1"
SAE_MODEL = XAI_DIR / "sae" / "sae_k16_seed1_l13.pt"
DECK_KEY = XAI_DIR / "session" / "key.csv"
# downloaded from the public MLflow (experiment 37) into the gitignored mlflow_artifacts/
ARTIFACTS = Path("mlflow_artifacts/c8f26853")
CHECKPOINT = ARTIFACTS / "checkpoints" / "epoch=179-step=39240" / "checkpoint.ckpt"
MODEL_CONFIG = ARTIFACTS / "configs" / "config-resolved.yaml"
# the slide table holds the public cluster's paths; mounted read-only here
PATH_REMAP = ("/mnt/projects/", "/mnt/public_projects/")

OUT_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/concepts/perturbation"
)
FIG = Path("figures/res_concept_perturbation.pdf")

STABLE = {"A": 0, "B": 1, "C": 8, "D": 12, "E": 3, "F": 6, "G": 2, "H": 4}
GROUPS = ("top", "active", "inactive")

# as the activations (configs/experiment/modeling/prediction/nuclei_level/radboud_activations)
EXPECTED_DATA = {"k": 24, "block_size": 128, "context": "knn"}
EXPECTED_DATASET = {"embedding_mode": "efd", "efd_order": 16}

# run the model with the k-NN block mask it was trained and extracted with (see
# `legacy_knn_block_mask`); False = the current (fixed) mask
LEGACY_MASK = True

N_TARGETS = 300
MAX_PER_SLIDE = 3
CROP_SIZE = 2000
RADIUS_UM = 50.0
SPREAD, COMPRESS = 1.5, 0.75
N_REMOVE = 8
N_CHECK = 50
N_BOOT = 2000
SEED = 42

PERTURBATIONS = (
    "spread",
    "compress",
    "nbr_ellipse",
    "own_ellipse",
    "own_circle",
    "nbr_rotate",
    "remove_8nn",
)
RESORT = ("spread_resort", "compress_resort", "remove_8nn_resort")
VARIANTS = ("none", *PERTURBATIONS, *RESORT)
LABELS = {
    "spread": "P1 spread x1.5",
    "compress": "P1 compress x0.75",
    "nbr_ellipse": "P2 neighbours → ellipse",
    "own_ellipse": "P3 own → ellipse",
    "own_circle": "P3 own → circle",
    "nbr_rotate": "P4 neighbours rotated",
    "remove_8nn": "P5 8-NN removed",
    "spread_resort": "P1 spread x1.5 (re-sorted)",
    "compress_resort": "P1 compress x0.75 (re-sorted)",
    "remove_8nn_resort": "P5 8-NN removed (re-sorted)",
}
SHORT = {
    "spread": "spread\n$\\times$1.5",
    "compress": "compress\n$\\times$0.75",
    "nbr_ellipse": "nbrs $\\to$\nellipse",
    "own_ellipse": "own $\\to$\nellipse",
    "own_circle": "own $\\to$\ncircle",
    "nbr_rotate": "nbrs\nrotated",
    "remove_8nn": "8-NN\nremoved",
}
# (concept, perturbation, predicted sign of Δκ) — the thesis signature; tested, not tuned
HYPOTHESES = (
    ("A", "spread", -1),
    ("A", "nbr_rotate", -1),
    ("C", "own_ellipse", -1),
    ("C", "own_circle", -1),
    ("D", "own_ellipse", -1),
    ("D", "own_circle", -1),
    ("F", "spread", 1),
    ("F", "compress", -1),
    ("G", "spread", 1),
    ("G", "compress", -1),
    ("H", "spread", 1),
    ("H", "compress", -1),
)


# --- targets ---


def top_rows(
    kappa: NDArray, slide_codes: NDArray, n: int, max_per_slide: int
) -> NDArray:
    """The `n` rows with the highest κ > 0, at most `max_per_slide` per slide, descending."""
    order = np.argsort(-kappa, kind="stable")
    per_slide: dict[int, int] = {}
    rows = []
    for i in order:
        if kappa[i] <= 0 or len(rows) >= n:
            break
        s = int(slide_codes[i])
        if per_slide.get(s, 0) >= max_per_slide:
            continue
        per_slide[s] = per_slide.get(s, 0) + 1
        rows.append(i)
    return np.asarray(rows, dtype=np.int64)


def select_targets(
    codes: NDArray,
    slide_codes: NDArray,
    concepts: dict[str, int],
    n: int,
    max_per_slide: int,
    seed: int,
) -> pd.DataFrame:
    """(concept, group, row) of the targets; `codes` (n, k) are the whole-slide κ."""
    rng = np.random.default_rng(seed)
    frames = []
    for letter, j in concepts.items():
        kappa = np.asarray(codes[:, j])
        active = np.flatnonzero(kappa > 0)
        inactive = np.flatnonzero(kappa == 0)
        groups = {
            "top": top_rows(kappa, slide_codes, n, max_per_slide),
            "active": np.sort(rng.choice(active, n, replace=False)),
            "inactive": np.sort(rng.choice(inactive, n, replace=False)),
        }
        for group, rows in groups.items():
            assert len(rows) == n, f"{letter} {group}: {len(rows)} targets"
            frames.append(
                pd.DataFrame(
                    {
                        "concept": letter,
                        "sae_index": j,
                        "group": group,
                        "row": rows,
                        "kappa_slide": kappa[rows],
                    }
                )
            )
    targets = pd.concat(frames, ignore_index=True)
    targets["slide"] = slide_codes[targets["row"]]
    return targets


# --- geometry (any unit; polygons (n, v, 2), centroids (n, 2)) ---


def polygon_area(polygons: NDArray) -> NDArray:
    """Signed shoelace area of each polygon (positive counter-clockwise in x-right, y-up)."""
    x, y = polygons[..., 0], polygons[..., 1]
    return 0.5 * (x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y).sum(-1)


def first_harmonic(polygons: NDArray) -> NDArray:
    """The first EFD harmonic (a, b, c, d) of each polygon, start-point normalised.

    Computed by `nuclei_graph.data.efd.elliptic_fourier_descriptors`, as the model's
    features: (a, c) is the semi-major axis, (b, d) the semi-minor one.
    """
    from nuclei_graph.data.efd import elliptic_fourier_descriptors

    return elliptic_fourier_descriptors(np.asarray(polygons, np.float64), order=1)[:, 0]


def ellipse_polygons(polygons: NDArray, centroids: NDArray) -> NDArray:
    """Each polygon's first-harmonic ellipse, scaled to its area, centred on `centroids`.

    Same vertex count and winding as the polygon; the orientation and axis ratio are the
    first harmonic's. The area matched is the polygon's (shoelace) area.
    """
    a, b, c, d = first_harmonic(polygons).T
    t = np.linspace(0, 2 * np.pi, polygons.shape[1], endpoint=False)
    x = a[:, None] * np.cos(t) + b[:, None] * np.sin(t)
    y = c[:, None] * np.cos(t) + d[:, None] * np.sin(t)
    return scaled_to_area(np.stack([x, y], -1), polygons, centroids)


def circle_polygons(polygons: NDArray, centroids: NDArray) -> NDArray:
    """A circle of each polygon's area around `centroids` (same vertex count and winding)."""
    t = np.linspace(0, 2 * np.pi, polygons.shape[1], endpoint=False)
    unit = np.broadcast_to(np.stack([np.cos(t), np.sin(t)], -1), polygons.shape)
    return scaled_to_area(unit.astype(np.float64), polygons, centroids)


def scaled_to_area(shapes: NDArray, polygons: NDArray, centroids: NDArray) -> NDArray:
    """`shapes` (centred at 0) scaled to the polygons' areas, moved to `centroids`."""
    target = np.abs(polygon_area(np.asarray(polygons, np.float64)))
    current = np.abs(polygon_area(shapes))
    scale = np.sqrt(target / np.maximum(current, np.finfo(np.float64).tiny))
    out = shapes * scale[:, None, None] + np.asarray(centroids, np.float64)[:, None]
    return match_winding(out, polygons)


def match_winding(new: NDArray, old: NDArray) -> NDArray:
    flip = np.sign(polygon_area(new)) != np.sign(
        polygon_area(np.asarray(old, np.float64))
    )
    new[flip] = new[flip, ::-1]
    return new


def moment_axis_ratio(polygons: NDArray) -> NDArray:
    """Major / minor axis of each polygon's second-moment (inertia) ellipse, >= 1."""
    p = np.asarray(polygons, np.float64)
    x, y = p[..., 0], p[..., 1]
    xn, yn = np.roll(x, -1, axis=-1), np.roll(y, -1, axis=-1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum(-1)
    cx = ((x + xn) * cross).sum(-1) / (6 * area)
    cy = ((y + yn) * cross).sum(-1) / (6 * area)
    sxx = ((x**2 + x * xn + xn**2) * cross).sum(-1) / (12 * area) - cx**2
    syy = ((y**2 + y * yn + yn**2) * cross).sum(-1) / (12 * area) - cy**2
    sxy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cross).sum(-1) / (
        24 * area
    ) - cx * cy
    half_trace, det = (sxx + syy) / 2, sxx * syy - sxy**2
    root = np.sqrt(np.maximum(half_trace**2 - det, 0))
    return np.sqrt((half_trace + root) / np.maximum(half_trace - root, 1e-300))


def rotate_polygons(polygons: NDArray, centroids: NDArray, angles: NDArray) -> NDArray:
    """Each polygon rotated about its centroid by its angle (radians)."""
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    p = np.asarray(polygons, np.float64) - np.asarray(centroids)[:, None]
    x = cos * p[..., 0] - sin * p[..., 1]
    y = sin * p[..., 0] + cos * p[..., 1]
    return np.stack([x, y], -1) + np.asarray(centroids)[:, None]


def within_radius(centroids: NDArray, target: int, radius: float) -> NDArray:
    """Indices of the nuclei with centroid distance < `radius` from `target`, without it."""
    d = np.linalg.norm(centroids - centroids[target], axis=1)
    idx = np.flatnonzero(d < radius)
    return idx[idx != target]


def nearest(centroids: NDArray, target: int, n: int) -> NDArray:
    """The `n` nearest nuclei to `target` (without it), nearest first."""
    d = np.linalg.norm(centroids - centroids[target], axis=1)
    d[target] = np.inf
    n = min(n, len(d) - 1)
    idx = np.argpartition(d, n - 1)[:n] if n > 0 else np.empty(0, np.int64)
    return idx[np.argsort(d[idx], kind="stable")]


def crop_around(centroids: NDArray, target: int, size: int) -> NDArray:
    """The `size` nuclei nearest to `target` (it included), in ascending index order."""
    if len(centroids) <= size:
        return np.arange(len(centroids))
    d = np.linalg.norm(centroids - centroids[target], axis=1)
    d[target] = -1.0  # first, also among nuclei at the same centroid
    return np.sort(np.argpartition(d, size - 1)[:size])


# --- crops and variants ---


@dataclass
class Crop:
    """One target's crop. Local indices; µm centroids, px polygons (as the nuclei table)."""

    target: int  # local index of the target
    centroids_um: NDArray[np.float32]  # (n, 2)
    centroids_px: NDArray[np.float64]  # (n, 2)
    polygons_px: NDArray[np.float32]  # (n, v, 2)
    features: NDArray[np.float32]  # (n, f)
    mpp: float


@dataclass
class Variant:
    keep: NDArray[np.int64]  # local indices of the nuclei in the run, ascending
    centroids_um: NDArray[np.float32]  # (n, 2), of all crop nuclei
    polygons: dict[int, NDArray] = field(default_factory=dict)  # local index -> new px
    resort: bool = False
    n_changed: int = 0


def make_variants(crop: Crop, rng: np.random.Generator) -> dict[str, Variant]:
    t, c_um, c_px, mpp = crop.target, crop.centroids_um, crop.centroids_px, crop.mpp
    n = len(c_um)
    all_idx = np.arange(n)
    near = within_radius(c_um.astype(np.float64), t, RADIUS_UM)
    out: dict[str, Variant] = {"none": Variant(all_idx, c_um)}

    for name, factor in (("spread", SPREAD), ("compress", COMPRESS)):
        new_um = c_um.astype(np.float64).copy()
        new_um[near] = c_um[t] + factor * (c_um[near] - c_um[t].astype(np.float64))
        shift_px = (new_um[near] - c_um[near]) / mpp
        polys = {
            int(i): crop.polygons_px[i] + shift_px[k][None] for k, i in enumerate(near)
        }
        for resort in (False, True):
            key = f"{name}_resort" if resort else name
            out[key] = Variant(
                all_idx, new_um.astype(np.float32), polys, resort, len(near)
            )

    if len(near):
        ell = ellipse_polygons(crop.polygons_px[near], c_px[near])
        out["nbr_ellipse"] = Variant(
            all_idx,
            c_um,
            dict(zip(near.tolist(), ell, strict=True)),
            n_changed=len(near),
        )
        angles = rng.uniform(0, 2 * np.pi, len(near))
        rot = rotate_polygons(crop.polygons_px[near], c_px[near], angles)
        out["nbr_rotate"] = Variant(
            all_idx,
            c_um,
            dict(zip(near.tolist(), rot, strict=True)),
            n_changed=len(near),
        )
    else:  # nothing to change: the unperturbed run
        out["nbr_ellipse"] = Variant(all_idx, c_um)
        out["nbr_rotate"] = Variant(all_idx, c_um)

    own = crop.polygons_px[[t]]
    out["own_ellipse"] = Variant(
        all_idx, c_um, {t: ellipse_polygons(own, c_px[[t]])[0]}, n_changed=1
    )
    out["own_circle"] = Variant(
        all_idx, c_um, {t: circle_polygons(own, c_px[[t]])[0]}, n_changed=1
    )

    removed = nearest(c_um.astype(np.float64), t, N_REMOVE)
    keep = np.setdiff1d(all_idx, removed)
    out["remove_8nn"] = Variant(keep, c_um, n_changed=len(removed))
    out["remove_8nn_resort"] = Variant(keep, c_um, resort=True, n_changed=len(removed))
    assert set(out) == set(VARIANTS)
    return out


def variant_inputs(
    crop: Crop,
    variant: Variant,
    base_order: NDArray[np.int64],
    centre: NDArray,
    extractor: "SlideNuclei",
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64] | None, int]:
    """(features, centred positions, fixed order or None, target position) of a run.

    Rows follow `variant.keep`; the fixed order is `base_order` (the unperturbed crop's
    spatial sort, local indices) restricted to `keep`, None to re-sort.
    """
    features = crop.features.copy()
    if variant.polygons:
        idx = np.fromiter(variant.polygons, np.int64)
        polys = np.stack([variant.polygons[i] for i in idx.tolist()]).astype(np.float32)
        features[idx] = extractor.get_efd_features(polys, crop.mpp, crop.mpp)
    keep = variant.keep
    pos = (variant.centroids_um[keep] - centre).astype(np.float32)
    target_pos = int(np.searchsorted(keep, crop.target))
    assert keep[target_pos] == crop.target
    order = None
    if not variant.resort:
        rank = np.full(len(crop.features), -1)
        rank[keep] = np.arange(len(keep))
        order = rank[base_order]
        order = order[order >= 0]
    return features[keep], pos, order, target_pos


# --- model ---


def legacy_knn_block_mask(
    neighbor_indices_list: list[torch.Tensor], block_size: int, total_seq_len: int
) -> BlockMask:
    """The k-NN block mask as the model was trained and its activations extracted with.

    `create_ragged_block_quantized_knn_mask` before 752565f2 ("fix: block mask"; here as
    at c6b9aff1, the commit of the activation extraction): every adjacent block pair is
    in the partial list, and pairs of pure blocks are in the full list as well, so
    flex_attention counts their keys twice (keys of a block that also holds padding or
    another sample count once). Model run c8f26853 (Oct 2026) predates the fix; with the
    current mask its logits on whole slides differ from the stored ones by ~0.13 on
    average, with this one by TF32 noise only.
    """
    device = neighbor_indices_list[0].device
    n_real = sum(neighbors.shape[0] for neighbors in neighbor_indices_list)
    k = neighbor_indices_list[0].shape[1]
    packed = pack_and_shift_knn_indices(neighbor_indices_list, total_seq_len)
    doc_ids = torch.cat(
        [
            torch.full((n.shape[0],), doc, dtype=torch.int32, device=device)
            for doc, n in enumerate(neighbor_indices_list)
        ]
        + [torch.full((total_seq_len - n_real,), -1, dtype=torch.int32, device=device)]
    )
    num_blocks = math.ceil(total_seq_len / block_size)

    q_block_ids = (
        (torch.arange(total_seq_len, device=device) // block_size)
        .unsqueeze(1)
        .expand(total_seq_len, k)
    )
    kv_block_ids = packed // block_size
    valid = packed >= 0
    adj = torch.zeros((num_blocks, num_blocks), dtype=torch.bool, device=device)
    adj[q_block_ids[valid], kv_block_ids[valid]] = True

    def compress(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cols = torch.arange(num_blocks, dtype=torch.int32, device=device).expand(
            num_blocks, num_blocks
        )
        ordered, _ = torch.where(matrix, cols, num_blocks + 1).sort(dim=-1)
        indices = torch.where(
            ordered > num_blocks,
            torch.tensor(-1, dtype=torch.int32, device=device),
            ordered.to(torch.int32),
        )
        return matrix.sum(dim=-1, dtype=torch.int32), indices

    starts = torch.arange(num_blocks, device=device) * block_size
    ends = torch.clamp(starts + block_size - 1, max=total_seq_len - 1)
    pure = doc_ids[starts] == doc_ids[ends]
    kv_num_blocks, kv_indices = compress(adj)
    full_num_blocks, full_indices = compress(adj & pure[:, None] & pure[None, :])
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.view(1, 1, num_blocks),
        kv_indices=kv_indices.view(1, 1, num_blocks, num_blocks),
        full_kv_num_blocks=full_num_blocks.view(1, 1, num_blocks),
        full_kv_indices=full_indices.view(1, 1, num_blocks, num_blocks),
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=_MaskMod(doc_ids),
        seq_lengths=(total_seq_len, total_seq_len),
    )


class SlideNuclei(BaseCropDataset):
    """The crop datasets' nucleus loading and feature code, without a dataset behind it."""

    def __init__(self, efd_order: int) -> None:
        self.efd_order = efd_order
        self.patch_size = None
        self.embedding_mode = "efd"
        self.augmentations = None

    def __getitem__(self, idx: int) -> Sample:  # type: ignore[override]
        raise NotImplementedError


class FixedOrderCollator(GraphCollator):
    """`GraphCollator` for one sample whose order may be given (`order`) instead of sorted."""

    order: NDArray[np.int64] | None = None

    def sort_order(self, sample: Sample, global_offset: int) -> NDArray[np.int64]:
        if self.order is None:
            return super().sort_order(sample, global_offset)
        assert global_offset == 0 and len(self.order) == len(sample["pos"])
        return np.asarray(self.order)


def make_sample(features: NDArray, pos: NDArray) -> Sample:
    n = len(pos)
    return Sample(
        {
            "features": torch.from_numpy(np.ascontiguousarray(features)),
            "bboxes": None,
            "labels": {"nuclei": None, "graph": None},
            "pos": torch.from_numpy(np.ascontiguousarray(pos)),
            "sup_mask": torch.ones(n, dtype=torch.bool),
            "seq_len": torch.tensor(n, dtype=torch.int32),
            "metadata": None,
        }
    )


def collate(
    collator: FixedOrderCollator, features: NDArray, pos: NDArray, order: NDArray | None
) -> tuple[dict, NDArray[np.int64]]:
    """The model inputs of one run and the order its rows were sorted into."""
    sample = make_sample(features, pos)
    collator.order = order
    used = collator.sort_order(sample, 0)
    batch = collator([sample])
    collator.order = None
    return {k: v for k, v in batch.items() if k != "metadata"}, used


def load_model(device: str) -> tuple[torch.nn.Module, dict]:
    import yaml
    from hydra.utils import instantiate

    config = yaml.safe_load(MODEL_CONFIG.read_text())
    data = config["data"]
    for key, value in EXPECTED_DATA.items():
        assert data[key] == value, f"data.{key} = {data[key]}, expected {value}"
    for key, value in EXPECTED_DATASET.items():
        assert data["dataset"][key] == value, f"dataset.{key} = {data['dataset'][key]}"
    model = instantiate(config["model"])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    assert model.net.self_head is None
    return model.eval().to(device), data


class Runner:
    """Runs crops through the frozen model and reads h, z and κ of the target."""

    def __init__(self, device: str) -> None:
        from nuclei_graph.xai.sae import load_autoencoder

        self.device = device
        self.model, data = load_model(device)
        self.sae = load_autoencoder(str(SAE_MODEL))[0].to(device)
        self.captured: torch.Tensor | None = None
        self.model.net.final_norm.register_forward_hook(self._hook)
        self.block_size, self.k, self.context = (
            data["block_size"],
            data["k"],
            data["context"],
        )
        self.efd_order = data["dataset"]["efd_order"]

    def collator(self) -> FixedOrderCollator:
        return FixedOrderCollator(
            block_size=self.block_size,
            k=self.k,
            predict=True,
            context=self.context,
            build_block_mask=False,  # built on the GPU, as in prediction
        )

    def _hook(
        self, module: torch.nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        self.captured = output.detach()

    @torch.inference_mode()
    def __call__(self, inputs: dict, position: int) -> tuple[NDArray, float]:
        """κ (16,) and z of the row at `position` of the sorted sequence."""
        from lightning.fabric.utilities.apply_func import move_data_to_device

        inputs = move_data_to_device(inputs, self.device)
        if LEGACY_MASK:
            inputs["block_mask"] = legacy_knn_block_mask(
                inputs["all_knns"],
                inputs["block_size"],
                total_seq_len=len(inputs["global_neighbor_idx"]),
            )
        out = self.model(inputs)
        assert self.captured is not None
        h = self.captured[0, position].float()
        z = self.model.net.class_head(h).squeeze(-1)
        assert torch.allclose(z, out["nuclei"][position].squeeze(-1), atol=1e-4)
        kappa = self.sae.encode(h[None])[0]
        return kappa.cpu().numpy(), float(z)


# --- slides ---


def test_slides() -> pd.DataFrame:
    slides = pd.read_parquet(XAI_DIR / "slides.parquet")
    test = (
        slides[slides["split"] == "test"].sort_values("offset").reset_index(drop=True)
    )
    test["slide_nuclei_path"] = test["slide_nuclei_path"].str.replace(*PATH_REMAP, n=1)
    return test


@dataclass
class SlideJob:
    slide: int  # position among the test slides
    rows: NDArray[np.int64]  # test-index rows of the slide's targets
    nucleus_ids: NDArray  # their nucleus ids (checked against the nuclei table)


def slide_crops(
    job: SlideJob, slide: pd.Series, extractor: SlideNuclei
) -> Iterator[tuple[int, Crop, int]]:
    """(row, crop, number of nuclei within `RADIUS_UM`) of each target of the slide."""
    assert np.isclose(slide.mpp_x, slide.mpp_y)
    mpp = float(slide.mpp_x)
    nuclei = extractor.get_nuclei(
        slide.slide_nuclei_path, ["id", "centroid", "polygon"]
    )
    assert len(nuclei) == slide.n_nuclei
    centroids_um = extractor.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)
    centroids_px = np.stack(nuclei["centroid"].tolist()).astype(np.float64)
    polygons = rearrange(np.array(nuclei["polygon"].tolist()), "n (v c) -> n v c", c=2)
    for row, nucleus_id in zip(job.rows.tolist(), job.nucleus_ids, strict=True):
        t = row - int(slide.offset)
        assert str(nuclei["id"].iloc[t]) == str(nucleus_id), "nucleus id mismatch"
        idx = crop_around(centroids_um, t, CROP_SIZE)
        local_t = int(np.searchsorted(idx, t))
        polys = polygons[idx]
        crop = Crop(
            target=local_t,
            centroids_um=centroids_um[idx],
            centroids_px=centroids_px[idx],
            polygons_px=polys,
            features=extractor.get_efd_features(polys, slide.mpp_x, slide.mpp_y),
            mpp=mpp,
        )
        n_near = len(
            within_radius(crop.centroids_um.astype(np.float64), local_t, RADIUS_UM)
        )
        yield row, crop, n_near


def prepared_runs(
    job: SlideJob,
    slide: pd.Series,
    variants: tuple[str, ...],
    collator: FixedOrderCollator,
    extractor: SlideNuclei,
    seed: int,
) -> list[dict]:
    """All runs of a slide's targets, collated (CPU; a DataLoader worker calls it)."""
    runs = []
    for row, crop, n_near in slide_crops(job, slide, extractor):
        rng = np.random.default_rng([seed, row])
        built = make_variants(crop, rng)
        centre = crop.centroids_um.mean(0)
        base_pos = (crop.centroids_um - centre).astype(np.float32)
        base_order = collator.sort_order(make_sample(crop.features, base_pos), 0)
        own = crop.polygons_px[[crop.target]]
        ratio = {
            "none": float(moment_axis_ratio(own)[0]),
            "own_ellipse": float(
                moment_axis_ratio(built["own_ellipse"].polygons[crop.target][None])[0]
            ),
        }
        for name in variants:
            features, pos, order, target_pos = variant_inputs(
                crop, built[name], base_order, centre, extractor
            )
            inputs, used = collate(collator, features, pos, order)
            position = int(np.flatnonzero(used == target_pos)[0])
            runs.append(
                {
                    "row": row,
                    "slide": job.slide,
                    "variant": name,
                    "inputs": inputs,
                    "position": position,
                    "crop_n": len(pos),
                    "n_within_radius": n_near,
                    "n_changed": built[name].n_changed,
                    "target_axis_ratio": ratio.get(name, np.nan),
                }
            )
    return runs


class SlideRuns(torch.utils.data.Dataset):
    """The prepared runs of one slide per item (built in the DataLoader workers)."""

    def __init__(
        self,
        jobs: list[SlideJob],
        slides: pd.DataFrame,
        variants: tuple[str, ...],
        collator_args: dict,
        efd_order: int,
        seed: int,
    ) -> None:
        self.jobs, self.slides, self.variants = jobs, slides, variants
        self.collator_args, self.efd_order, self.seed = collator_args, efd_order, seed

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i: int) -> list[dict]:
        torch.set_num_threads(1)
        job = self.jobs[i]
        collator = FixedOrderCollator(
            predict=True, build_block_mask=False, **self.collator_args
        )
        return prepared_runs(
            job,
            self.slides.iloc[job.slide],
            self.variants,
            collator,
            SlideNuclei(self.efd_order),
            self.seed,
        )


def run_targets(
    rows: NDArray[np.int64],
    variants: tuple[str, ...],
    out_dir: Path,
    device: str,
    num_workers: int,
) -> pd.DataFrame:
    """κ and z of every (row, variant); one parquet per slide in `out_dir` (resumable)."""
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    slides = test_slides()
    index = pd.read_parquet(XAI_DIR / "activations" / "test" / "index.parquet")
    slide_of_row = np.searchsorted(slides["offset"].to_numpy(), rows, side="right") - 1
    jobs = []
    for s in np.unique(slide_of_row):
        if (out_dir / f"slide_{s:04d}.parquet").exists():
            continue
        r = np.sort(rows[slide_of_row == s])
        jobs.append(SlideJob(int(s), r, index["nucleus_id"].to_numpy()[r]))

    torch.set_float32_matmul_precision("high")  # as nuclei_graph.__main__
    runner = Runner(device)
    args = {"block_size": runner.block_size, "k": runner.k, "context": runner.context}
    dataset = SlideRuns(jobs, slides, variants, args, runner.efd_order, SEED)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        prefetch_factor=2 if num_workers else None,
        persistent_workers=False,
    )
    for job, runs in tqdm(
        zip(jobs, loader, strict=True), total=len(jobs), desc="slides"
    ):
        records = []
        for run in runs:
            kappa, z = runner(run.pop("inputs"), run["position"])
            records.append({**run, "z": z, **{f"k{j}": kappa[j] for j in range(16)}})
        frame = pd.DataFrame(records)
        tmp = out_dir / f".slide_{job.slide:04d}.parquet"
        frame.to_parquet(tmp, index=False)
        tmp.rename(out_dir / f"slide_{job.slide:04d}.parquet")
    return pd.concat(
        [pd.read_parquet(p) for p in sorted(out_dir.glob("slide_*.parquet"))],
        ignore_index=True,
    )


# --- statistics ---


def slide_bootstrap(
    numer: NDArray,
    denom: NDArray,
    slides: NDArray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Σ numer / Σ denom and its 95 % percentile interval from resampling slides."""
    codes, inv = np.unique(slides, return_inverse=True)
    num = np.bincount(inv, weights=numer, minlength=len(codes))
    den = np.bincount(inv, weights=denom, minlength=len(codes))
    total = den.sum()
    estimate = num.sum() / total if total > 0 else np.nan
    if total == 0:
        return np.nan, np.nan, np.nan
    draws = rng.integers(0, len(codes), (n_boot, len(codes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num[draws].sum(1) / den[draws].sum(1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(estimate), float(lo), float(hi)


def summarise(
    raw: pd.DataFrame, targets: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    """One row per (concept, group, perturbation); see the module docstring."""
    rng = np.random.default_rng(seed)
    base = raw[raw["variant"] == "none"].set_index("row")
    rows = []
    for (letter, group), tg in targets.groupby(["concept", "group"], sort=False):
        j = int(tg["sae_index"].iloc[0])
        r = tg["row"].to_numpy()
        slides = tg["slide"].to_numpy()
        k0 = base.loc[r, f"k{j}"].to_numpy(np.float64)
        z0 = base.loc[r, "z"].to_numpy(np.float64)
        on0 = k0 > 0
        for variant in (*PERTURBATIONS, *RESORT):
            pert = raw[raw["variant"] == variant].set_index("row").loc[r]
            k1 = pert[f"k{j}"].to_numpy(np.float64)
            dk = k1 - k0
            dz = pert["z"].to_numpy(np.float64) - z0
            ones = np.ones(len(r))
            stat = {
                "d_kappa": slide_bootstrap(dk, ones, slides, n_boot, rng),
                "rel_change": slide_bootstrap(dk, k0, slides, n_boot, rng),
                "d_z": slide_bootstrap(dz, ones, slides, n_boot, rng),
                "off_share": slide_bootstrap(
                    (on0 & (k1 == 0)).astype(float),
                    on0.astype(float),
                    slides,
                    n_boot,
                    rng,
                ),
                "on_share": slide_bootstrap(
                    (~on0 & (k1 > 0)).astype(float),
                    (~on0).astype(float),
                    slides,
                    n_boot,
                    rng,
                ),
            }
            row = {
                "concept": letter,
                "sae_index": j,
                "group": group,
                "perturbation": variant,
                "n_targets": len(r),
                "n_slides": len(np.unique(slides)),
                "n_active_crop": int(on0.sum()),
                "kappa0_crop_mean": float(k0.mean()),
                "kappa_slide_mean": float(tg["kappa_slide"].mean()),
                "kappa1_mean": float(k1.mean()),
                "z0_mean": float(z0.mean()),
                "n_changed_mean": float(pert["n_changed"].mean()),
            }
            for name, (est, lo, hi) in stat.items():
                row |= {name: est, f"{name}_lo": lo, f"{name}_hi": hi}
            rows.append(row)
    return pd.DataFrame(rows)


def verdict(lo: float, hi: float, sign: int) -> str:
    """The verdict of a hypothesis from the interval (lo, hi) of its effect.

    `holds` if the interval lies on the predicted side of 0, `opposite` if on the other.
    """
    if sign * lo > 0 and sign * hi > 0:
        return "holds"
    if sign * lo < 0 and sign * hi < 0:
        return "opposite"
    return "not supported"


def judge(summary: pd.DataFrame) -> pd.DataFrame:
    top = summary[summary["group"] == "top"].set_index(["concept", "perturbation"])
    rows = []
    for letter, pert, sign in HYPOTHESES:
        s = top.loc[(letter, pert)]
        rows.append(
            {
                "concept": letter,
                "perturbation": pert,
                "predicted": "rise" if sign > 0 else "drop",
                "d_kappa": s["d_kappa"],
                "d_kappa_lo": s["d_kappa_lo"],
                "d_kappa_hi": s["d_kappa_hi"],
                "rel_change": s["rel_change"],
                "rel_change_lo": s["rel_change_lo"],
                "rel_change_hi": s["rel_change_hi"],
                "off_share": s["off_share"],
                "verdict": verdict(s["d_kappa_lo"], s["d_kappa_hi"], sign),
            }
        )
    return pd.DataFrame(rows)


def check_crops(raw: pd.DataFrame, codes: NDArray, logits: NDArray) -> dict:
    """Unperturbed crop vs. whole slide: Pearson r of all 16 codes and of the logits."""
    base = raw[raw["variant"] == "none"]
    rows = base["row"].to_numpy()
    crop_k = base[[f"k{j}" for j in range(16)]].to_numpy(np.float64)
    slide_k = np.asarray(codes[rows], np.float64)
    crop_z, slide_z = (
        base["z"].to_numpy(np.float64),
        np.asarray(logits[rows], np.float64),
    )
    stable = list(STABLE.values())
    active_agree = ((crop_k > 0) == (slide_k > 0))[:, stable].mean()
    return {
        "n_targets": len(rows),
        "pearson_kappa_all16": float(
            np.corrcoef(crop_k.ravel(), slide_k.ravel())[0, 1]
        ),
        "pearson_kappa_stable8": float(
            np.corrcoef(crop_k[:, stable].ravel(), slide_k[:, stable].ravel())[0, 1]
        ),
        "pearson_kappa_per_concept": {
            letter: float(np.corrcoef(crop_k[:, j], slide_k[:, j])[0, 1])
            for letter, j in STABLE.items()
        },
        "mean_abs_diff_kappa": float(np.abs(crop_k - slide_k).mean()),
        "active_state_agreement_stable8": float(active_agree),
        "pearson_logit": float(np.corrcoef(crop_z, slide_z)[0, 1]),
        "mean_abs_diff_logit": float(np.abs(crop_z - slide_z).mean()),
        "max_abs_diff_logit": float(np.abs(crop_z - slide_z).max()),
    }


# --- report ---


def fmt(v: float, lo: float, hi: float, scale: float = 100.0, digits: int = 0) -> str:
    return (
        f"{v * scale:+.{digits}f} [{lo * scale:+.{digits}f}, {hi * scale:+.{digits}f}]"
    )


def tex_pct(value: float) -> str:
    """A signed whole percentage, with a math minus for LaTeX."""
    return f"{value * 100:+.0f}".replace("-", "$-$")


def latex_table(summary: pd.DataFrame, verdicts: pd.DataFrame) -> str:
    """Relative change of κ_j of the top targets (%), CI below; hypotheses marked."""
    top = summary[summary["group"] == "top"].set_index(["concept", "perturbation"])
    marks = {(r.concept, r.perturbation): r for r in verdicts.itertuples()}
    head = " & ".join(
        r"\makecell{" + SHORT[p].replace("\n", r" \\ ") + "}" for p in PERTURBATIONS
    )
    lines = [
        r"\begin{tabular}{l" + "r" * len(PERTURBATIONS) + "}",
        r"\toprule",
        r"Concept & " + head + r" \\",
        r"\midrule",
    ]
    for letter in STABLE:
        cells = []
        for p in PERTURBATIONS:
            s = top.loc[(letter, p)]
            est = tex_pct(s["rel_change"])
            ci = f"{tex_pct(s['rel_change_lo'])}, {tex_pct(s['rel_change_hi'])}"
            m = marks.get((letter, p))
            if m is not None:
                arrow = r"$\uparrow$" if m.predicted == "rise" else r"$\downarrow$"
                est = (r"\textbf{" + est + "}" if m.verdict == "holds" else est) + arrow
            cells.append(r"\makecell{" + est + r" \\ {\scriptsize [" + ci + "]}}")
        lines.append(letter + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def figure(summary: pd.DataFrame, verdicts: pd.DataFrame, path: Path) -> None:
    """Δκ_j in % of the top targets' unperturbed mean κ_j, per concept and perturbation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xai.concept_morphometry import FIG_W_CM, THESIS_RC

    style = {  # identity: colour and marker shape
        "top": ("#2a78d6", "o", "full", r"top 300 ($\leq$3 per slide)"),
        "active": ("#eb6834", "^", "none", "random active"),
        "inactive": ("#7a7a7a", "s", "none", "random inactive"),
    }
    offsets = {"top": -0.22, "active": 0.0, "inactive": 0.22}
    predicted = {(c, p): s for c, p, s in HYPOTHESES}
    held = {(r.concept, r.perturbation): r.verdict for r in verdicts.itertuples()}
    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(
            4,
            2,
            figsize=(FIG_W_CM / 2.54, 19 / 2.54),
            sharex=True,
            layout="constrained",
        )
        x = np.arange(len(PERTURBATIONS))
        for ax, letter in zip(axes.ravel(), STABLE, strict=True):
            sub = summary[(summary["concept"] == letter)].set_index(
                ["group", "perturbation"]
            )
            scale = sub.loc[("top", PERTURBATIONS[0]), "kappa0_crop_mean"]
            for k, p in enumerate(PERTURBATIONS):
                if (letter, p) in predicted:
                    ax.axvspan(k - 0.45, k + 0.45, color="#ececec", lw=0, zorder=0)
                    mark = r"\uparrow" if predicted[(letter, p)] > 0 else r"\downarrow"
                    ok = r"\checkmark" if held[(letter, p)] == "holds" else r"\times"
                    ax.text(
                        k,
                        1.0,
                        f"${mark}{ok}$",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#000000",
                    )
            ax.axhline(0, color="#9a9a9a", lw=0.6, zorder=1)
            for group, (color, marker, fill, _) in style.items():
                rows = sub.loc[[(group, p) for p in PERTURBATIONS]]
                y = rows["d_kappa"].to_numpy() / scale * 100
                lo = rows["d_kappa_lo"].to_numpy() / scale * 100
                hi = rows["d_kappa_hi"].to_numpy() / scale * 100
                ax.errorbar(
                    x + offsets[group],
                    y,
                    yerr=[y - lo, hi - y],
                    fmt=marker,
                    color=color,
                    mfc=color if fill == "full" else "white",
                    ms=3.6,
                    mew=0.9,
                    elinewidth=0.9,
                    capsize=0,
                    zorder=3,
                )
            ax.set_title(f"Concept {letter}", loc="left", fontsize=8, pad=11)
            ax.set_xlim(-0.6, len(PERTURBATIONS) - 0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, labelsize=7)
            ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        for ax in axes[-1]:
            ticks = [SHORT[p] for p in PERTURBATIONS]
            ax.set_xticks(x, ticks, fontsize=6.5)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\Delta\kappa_j$ (% of top mean $\kappa_j$)", fontsize=7)
        handles = [
            plt.Line2D(
                [],
                [],
                ls="",
                marker=m,
                color=c,
                mfc=c if f == "full" else "white",
                ms=4,
                mew=0.9,
                label=lab,
            )
            for c, m, f, lab in style.values()
        ]
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color="#ececec",
                label=r"hypothesis: $\uparrow\downarrow$ predicted, $\checkmark$ holds",
            )
        )
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=4,
            frameon=False,
            fontsize=7,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)


def note(summary: pd.DataFrame, verdicts: pd.DataFrame, check: dict) -> str:
    lines = [
        "# Input perturbations of the stable concepts — note",
        "",
        (
            f"Crop check ({check['n_targets']} random targets): Pearson r of the 16 "
            f"codes, crop vs. whole slide = {check['pearson_kappa_all16']:.3f} (logit "
            f"r = {check['pearson_logit']:.3f}, mean |Δz| = "
            f"{check['mean_abs_diff_logit']:.3f})."
        ),
        "",
        (
            "Hypotheses (top targets; 95 % slide-bootstrap CI; relative change = "
            "Σ Δκ_j / Σ κ_j):"
        ),
        "",
        "| Concept | Perturbation | Predicted | Relative change | Switch-off | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in verdicts.itertuples():
        lines.append(
            f"| {r.concept} | {LABELS[r.perturbation]} | {r.predicted} | "
            f"{fmt(r.rel_change, r.rel_change_lo, r.rel_change_hi)} % | "
            f"{r.off_share * 100:.0f} % | {r.verdict} |"
        )
    return "\n".join(lines) + "\n"


def code_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("check", "run", "report"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = np.load(CONCEPT_DIR / "codes_test.npy", mmap_mode="r")
    logits = np.load(XAI_DIR / "activations" / "test" / "logits.npy", mmap_mode="r")
    key = pd.read_csv(DECK_KEY).set_index("letter")["concept"].astype(int)
    assert all(key[letter] == j for letter, j in STABLE.items()), "deck key changed"

    targets_path = OUT_DIR / "targets.parquet"
    if targets_path.exists():
        targets = pd.read_parquet(targets_path)
    else:
        slides = test_slides()
        n = int(slides["n_nuclei"].sum())
        assert n == len(codes)
        slide_codes = np.repeat(np.arange(len(slides)), slides["n_nuclei"].to_numpy())
        targets = select_targets(
            np.asarray(codes), slide_codes, STABLE, N_TARGETS, MAX_PER_SLIDE, SEED
        )
        targets.to_parquet(targets_path, index=False)
    unique_rows = np.unique(targets["row"].to_numpy())

    if args.stage == "check":
        rng = np.random.default_rng(SEED)
        rows = np.sort(rng.choice(unique_rows, N_CHECK, replace=False))
        start = time.time()
        raw = run_targets(rows, ("none",), OUT_DIR / "check", args.device, args.workers)
        result = check_crops(raw, codes, logits)
        result["seconds"] = time.time() - start
        (OUT_DIR / "check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    if args.stage == "run":
        start = time.time()
        run_targets(unique_rows, VARIANTS, OUT_DIR / "raw", args.device, args.workers)
        print(f"done in {(time.time() - start) / 60:.1f} min")
        return

    raw = pd.concat(
        [pd.read_parquet(p) for p in sorted((OUT_DIR / "raw").glob("slide_*.parquet"))],
        ignore_index=True,
    )
    assert set(raw["row"]) == set(unique_rows), "run incomplete"
    assert raw.groupby("row")["variant"].nunique().eq(len(VARIANTS)).all()
    raw.to_parquet(OUT_DIR / "raw.parquet", index=False)
    summary = summarise(raw, targets, N_BOOT, SEED)
    verdicts = judge(summary)
    summary.to_csv(OUT_DIR / "perturbation_summary.csv", index=False)
    verdicts.to_csv(OUT_DIR / "hypotheses.csv", index=False)
    check = json.loads((OUT_DIR / "check.json").read_text())
    check_all = check_crops(raw, codes, logits)
    (OUT_DIR / "tab_res_concept_perturbation.tex").write_text(
        latex_table(summary, verdicts)
    )
    (OUT_DIR / "note.md").write_text(note(summary, verdicts, check))
    figure(summary, verdicts, FIG)
    stats = {
        "runs": {"model_training": MODEL_RUN, "sae_training": SAE_TRAINING_RUN},
        "sae_model": SAE_MODEL.name,
        "checkpoint": str(CHECKPOINT.relative_to(ARTIFACTS)),
        "code_commit": code_commit(),
        "n_targets_per_group": N_TARGETS,
        "max_per_slide_top": MAX_PER_SLIDE,
        "crop_size": CROP_SIZE,
        "radius_um": RADIUS_UM,
        "n_unique_targets": len(unique_rows),
        "n_boot": N_BOOT,
        "seed": SEED,
        "crop_check_50": check,
        "crop_check_all_targets": check_all,
        "neighbours_within_radius": raw[raw["variant"] == "none"]["n_within_radius"]
        .describe()
        .to_dict(),
        "hypotheses": verdicts.to_dict(orient="records"),
    }
    FIG.with_name(FIG.stem + "_stats.json").write_text(json.dumps(stats, indent=2))
    print(verdicts.to_string())


if __name__ == "__main__":
    main()
