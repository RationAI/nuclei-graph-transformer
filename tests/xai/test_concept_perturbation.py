"""xai.concept_perturbation: geometry of the perturbations, crops, order, statistics."""

import numpy as np
import pandas as pd
import pytest
import torch

from nuclei_graph.data.datamodules.collator import GraphCollator
from xai.concept_perturbation import (
    HYPOTHESES,
    PERTURBATIONS,
    RESORT,
    VARIANTS,
    Crop,
    FixedOrderCollator,
    SlideNuclei,
    circle_polygons,
    collate,
    crop_around,
    ellipse_polygons,
    first_harmonic,
    judge,
    make_sample,
    make_variants,
    moment_axis_ratio,
    nearest,
    polygon_area,
    rotate_polygons,
    select_targets,
    slide_bootstrap,
    summarise,
    variant_inputs,
    verdict,
    within_radius,
)


V = 64  # vertices per polygon, as the nuclei tables


def irregular_polygons(n: int, rng: np.random.Generator) -> np.ndarray:
    """Star-shaped nuclei-like polygons (ellipse + higher harmonics), centred at 0."""
    t = np.linspace(0, 2 * np.pi, V, endpoint=False)
    a, b = rng.uniform(4, 8, (n, 1)), rng.uniform(2, 4, (n, 1))
    psi = rng.uniform(0, np.pi, (n, 1))
    wobble = 1 + 0.08 * np.sin(3 * t + rng.uniform(0, 6, (n, 1))) + 0.05 * np.cos(5 * t)
    x, y = a * np.cos(t) * wobble, b * np.sin(t) * wobble
    xr = np.cos(psi) * x - np.sin(psi) * y
    yr = np.sin(psi) * x + np.cos(psi) * y
    return np.stack([xr, yr], -1)


def orientation(polygons: np.ndarray) -> np.ndarray:
    return SlideNuclei(16).get_orientations(polygons.astype(np.float32), 1.0, 1.0)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Difference of axis orientations (period π), in [0, π/2]."""
    d = np.mod(a - b, np.pi)
    return np.minimum(d, np.pi - d)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --- features ---


def test_features_are_per_nucleus(rng):
    """Recomputing the features of a subset gives the rows of the full computation."""
    polygons = (irregular_polygons(50, rng) + rng.uniform(0, 500, (50, 1, 2))).astype(
        np.float32
    )
    extractor = SlideNuclei(16)
    full = extractor.get_efd_features(polygons, 0.4862, 0.4862)
    idx = np.array([3, 17, 41])
    part = extractor.get_efd_features(polygons[idx], 0.4862, 0.4862)
    np.testing.assert_array_equal(part, full[idx])


# --- geometry ---


def test_ellipse_keeps_area_orientation_centroid(rng):
    polygons = irregular_polygons(40, rng) + rng.uniform(-100, 100, (40, 1, 2))
    centroids = rng.uniform(-100, 100, (40, 2))
    ellipses = ellipse_polygons(polygons, centroids)
    assert ellipses.shape == polygons.shape
    np.testing.assert_allclose(
        np.abs(polygon_area(ellipses)), np.abs(polygon_area(polygons)), rtol=1e-9
    )
    np.testing.assert_allclose(ellipses.mean(1), centroids, atol=1e-9)
    assert angle_diff(orientation(ellipses), orientation(polygons)).max() < 1e-2
    # every vertex lies on the ellipse spanned by the polygon's first harmonic (the
    # model's EFD is arc-length based, so its higher harmonics are not zero)
    harmonic = first_harmonic(polygons)  # (n, 4): a, b, c, d
    m = harmonic[:, [0, 1, 2, 3]].reshape(-1, 2, 2)  # [[a, b], [c, d]]
    unit = np.einsum("nij,nvj->nvi", np.linalg.inv(m), ellipses - centroids[:, None])
    radius = np.linalg.norm(unit, axis=-1)
    np.testing.assert_allclose(radius, radius[:, :1].repeat(V, 1), rtol=1e-9)
    # same winding
    assert np.all(np.sign(polygon_area(ellipses)) == np.sign(polygon_area(polygons)))


def test_moment_axis_ratio():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ellipse = np.stack([6 * np.cos(t) + 3, 3 * np.sin(t) - 1], -1)
    assert moment_axis_ratio(ellipse[None])[0] == pytest.approx(2.0, rel=1e-3)
    assert moment_axis_ratio(ellipse[None, ::-1])[0] == pytest.approx(2.0, rel=1e-3)


def test_ellipse_of_an_ellipse_is_rounder_same_orientation():
    """The arc-length first harmonic of an ellipse is a slightly rounder ellipse."""
    t = np.linspace(0, 2 * np.pi, V, endpoint=False)
    psi = 0.7
    x, y = 6 * np.cos(t), 3 * np.sin(t)
    poly = np.stack(
        [np.cos(psi) * x - np.sin(psi) * y, np.sin(psi) * x + np.cos(psi) * y], -1
    )
    out = ellipse_polygons(poly[None], np.zeros((1, 2)))[0]
    assert angle_diff(orientation(out[None]), np.array([psi]))[0] < 1e-6
    before, after = moment_axis_ratio(poly[None])[0], moment_axis_ratio(out[None])[0]
    assert 0.8 * before < after < before
    assert polygon_area(out) == pytest.approx(polygon_area(poly), rel=1e-9)


def test_circle_keeps_area_and_is_round(rng):
    polygons = irregular_polygons(20, rng)
    centroids = rng.uniform(0, 10, (20, 2))
    circles = circle_polygons(polygons, centroids)
    np.testing.assert_allclose(
        np.abs(polygon_area(circles)), np.abs(polygon_area(polygons)), rtol=1e-9
    )
    radii = np.linalg.norm(circles - centroids[:, None], axis=-1)
    np.testing.assert_allclose(radii, radii[:, :1].repeat(V, 1), rtol=1e-12)
    assert np.all(np.sign(polygon_area(circles)) == np.sign(polygon_area(polygons)))


def test_clockwise_winding_is_kept(rng):
    polygons = irregular_polygons(5, rng)[:, ::-1]
    assert np.all(polygon_area(polygons) < 0)
    assert np.all(polygon_area(ellipse_polygons(polygons, np.zeros((5, 2)))) < 0)
    assert np.all(polygon_area(circle_polygons(polygons, np.zeros((5, 2)))) < 0)


def test_rotation_turns_orientation_keeps_shape(rng):
    polygons = irregular_polygons(30, rng) + 50
    centroids = polygons.mean(1)
    angles = rng.uniform(0, 2 * np.pi, 30)
    rotated = rotate_polygons(polygons, centroids, angles)
    np.testing.assert_allclose(polygon_area(rotated), polygon_area(polygons), rtol=1e-9)
    np.testing.assert_allclose(rotated.mean(1), centroids, atol=1e-9)
    assert angle_diff(orientation(rotated), orientation(polygons) + angles).max() < 1e-3
    # the normalised shape features are unchanged up to the sign of the even harmonics
    # (the EFD normalisation's 180° ambiguity depends on the absolute orientation)
    f0 = SlideNuclei(16).get_efd_features(polygons.astype(np.float32), 1.0, 1.0)
    f1 = SlideNuclei(16).get_efd_features(rotated.astype(np.float32), 1.0, 1.0)
    np.testing.assert_allclose(np.abs(f1[:, :-2]), np.abs(f0[:, :-2]), atol=1e-5)
    odd = np.arange(64).reshape(16, 4)[::2].ravel()  # harmonics 1, 3, 5, ...
    np.testing.assert_allclose(f1[:, odd], f0[:, odd], atol=1e-5)


def test_neighbourhoods():
    centroids = np.array(
        [[0, 0], [10, 0], [0, 49.9], [50, 0], [30, 30], [100, 100]], float
    )
    assert set(within_radius(centroids, 0, 50.0)) == {1, 2, 4}
    np.testing.assert_array_equal(nearest(centroids, 0, 3), [1, 4, 2])
    np.testing.assert_array_equal(crop_around(centroids, 0, 3), [0, 1, 4])
    np.testing.assert_array_equal(crop_around(centroids, 5, 10), np.arange(6))
    # the target is in its crop even among nuclei at the same centroid
    same = np.zeros((10, 2))
    assert 7 in crop_around(same, 7, 3)


# --- variants ---


def grid_crop(
    rng: np.random.Generator, n_side: int = 30, spacing: float = 12.0
) -> Crop:
    """A jittered grid of nuclei (µm = px · mpp), the target near the centre."""
    mpp = 0.5
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1).reshape(-1, 2)
    c_um = (g * spacing + rng.uniform(-2, 2, g.shape)).astype(np.float32)
    c_px = c_um.astype(np.float64) / mpp
    polys = (irregular_polygons(len(g), rng) / mpp + c_px[:, None]).astype(np.float32)
    target = int(np.argmin(np.linalg.norm(c_um - c_um.mean(0), axis=1)))
    features = SlideNuclei(16).get_efd_features(polys, mpp, mpp)
    return Crop(target, c_um, c_px, polys, features, mpp)


def test_variants(rng):
    crop = grid_crop(rng)
    t, c = crop.target, crop.centroids_um.astype(np.float64)
    variants = make_variants(crop, np.random.default_rng(1))
    assert set(variants) == set(VARIANTS)
    near = within_radius(c, t, 50.0)
    far = np.setdiff1d(np.arange(len(c)), np.append(near, t))

    for name, factor in (("spread", 1.5), ("compress", 0.75)):
        v = variants[name]
        new = v.centroids_um.astype(np.float64)
        np.testing.assert_allclose(
            new[near] - c[t], factor * (c[near] - c[t]), atol=1e-4
        )
        np.testing.assert_array_equal(new[far], c[far])
        np.testing.assert_array_equal(new[t], c[t])
        # polygons moved with their centroids, shapes kept
        for i in near[:5]:
            shift = (new[i] - c[i]) / crop.mpp
            np.testing.assert_allclose(
                v.polygons[int(i)], crop.polygons_px[i] + shift, atol=1e-3
            )
        assert not v.resort and variants[f"{name}_resort"].resort

    for name in ("nbr_ellipse", "nbr_rotate"):
        assert set(variants[name].polygons) == set(near.tolist())
        np.testing.assert_array_equal(variants[name].centroids_um, crop.centroids_um)
    for name in ("own_ellipse", "own_circle"):
        assert set(variants[name].polygons) == {t}

    removed = nearest(c, t, 8)
    keep = variants["remove_8nn"].keep
    assert len(keep) == len(c) - 8 and not set(removed) & set(keep) and t in keep


def test_variant_inputs_order_and_features(rng):
    crop = grid_crop(rng)
    variants = make_variants(crop, np.random.default_rng(1))
    collator = FixedOrderCollator(block_size=128, k=24, predict=True)
    centre = crop.centroids_um.mean(0)
    base_pos = (crop.centroids_um - centre).astype(np.float32)
    base_order = collator.sort_order(make_sample(crop.features, base_pos), 0)
    extractor = SlideNuclei(16)

    feats, _, order, tpos = variant_inputs(
        crop, variants["none"], base_order, centre, extractor
    )
    np.testing.assert_array_equal(feats, crop.features)
    np.testing.assert_array_equal(order, base_order)
    assert tpos == crop.target

    # own ellipse: only the target's features change
    feats, _, order, tpos = variant_inputs(
        crop, variants["own_ellipse"], base_order, centre, extractor
    )
    changed = np.flatnonzero(np.any(feats != crop.features, axis=1))
    np.testing.assert_array_equal(changed, [crop.target])
    np.testing.assert_array_equal(order, base_order)

    # removal: the base order without the removed nuclei, reindexed
    v = variants["remove_8nn"]
    feats, _, order, tpos = variant_inputs(crop, v, base_order, centre, extractor)
    assert sorted(order.tolist()) == list(range(len(v.keep)))
    np.testing.assert_array_equal(
        v.keep[order], base_order[np.isin(base_order, v.keep)]
    )
    assert v.keep[tpos] == crop.target
    np.testing.assert_array_equal(feats, crop.features[v.keep])

    # re-sorted runs leave the order to the collator
    assert (
        variant_inputs(crop, variants["spread_resort"], base_order, centre, extractor)[
            2
        ]
        is None
    )


# --- collation ---


def test_graph_collator_sort_unchanged(rng):
    from nuclei_graph.data.block_mask import block_spatial_sort

    pos = rng.uniform(0, 500, (700, 2)).astype(np.float32)
    sample = make_sample(rng.normal(size=(700, 67)).astype(np.float32), pos)
    np.testing.assert_array_equal(
        GraphCollator(block_size=128, k=24).sort_order(sample, 0),
        block_spatial_sort(pos, 128, global_offset=0),
    )


def test_fixed_order_collator(rng):
    pos = rng.uniform(0, 500, (700, 2)).astype(np.float32)
    features = rng.normal(size=(700, 67)).astype(np.float32)
    plain = GraphCollator(block_size=128, k=24, predict=True, build_block_mask=False)
    fixed = FixedOrderCollator(
        block_size=128, k=24, predict=True, build_block_mask=False
    )

    # no order: identical to GraphCollator
    a = plain([make_sample(features, pos)])
    b, used = collate(fixed, features, pos, None)
    for key in ("pos", "features", "global_neighbor_idx"):
        torch.testing.assert_close(a[key], b[key])
    assert fixed.order is None

    # a given order is used as is; the target is found at its position
    order = rng.permutation(700)
    batch, used = collate(fixed, features, pos, order)
    np.testing.assert_array_equal(used, order)
    torch.testing.assert_close(batch["pos"][:700], torch.from_numpy(pos[order]))
    target = 123
    position = int(np.flatnonzero(used == target)[0])
    torch.testing.assert_close(
        batch["features"][position], torch.from_numpy(features[target])
    )


# --- targets and statistics ---


def test_select_targets(rng):
    n, k = 5000, 16
    codes = np.where(rng.random((n, k)) < 0.3, rng.exponential(1.0, (n, k)), 0.0)
    slide_codes = np.repeat(np.arange(50), n // 50)
    concepts = {"A": 0, "B": 5}
    targets = select_targets(codes, slide_codes, concepts, 40, 3, seed=0)
    assert len(targets) == 2 * 3 * 40
    for letter, j in concepts.items():
        top = targets[(targets["concept"] == letter) & (targets["group"] == "top")]
        assert top.groupby("slide").size().max() <= 3
        assert (np.diff(top["kappa_slide"].to_numpy()) <= 0).all()
        # nothing above the selection's minimum was skipped, except by the slide cap
        rows = set(top["row"])
        better = np.flatnonzero(codes[:, j] > top["kappa_slide"].min())
        skipped = [r for r in better if r not in rows]
        counts = top.groupby("slide").size()
        assert all(counts.get(slide_codes[r], 0) == 3 for r in skipped)
        sub = targets[targets["concept"] == letter].set_index("group")
        assert (codes[sub.loc["active", "row"], j] > 0).all()
        assert (codes[sub.loc["inactive", "row"], j] == 0).all()
        np.testing.assert_array_equal(
            targets["slide"], slide_codes[targets["row"].to_numpy()]
        )


def test_slide_bootstrap():
    rng = np.random.default_rng(0)
    slides = np.repeat(np.arange(30), 4)
    values = rng.normal(1.0, 1.0, 120)
    est, lo, hi = slide_bootstrap(values, np.ones(120), slides, 2000, rng)
    assert est == pytest.approx(values.mean())
    assert lo < est < hi
    # one slide: no variation between resamples
    est, lo, hi = slide_bootstrap(values[:4], np.ones(4), slides[:4], 200, rng)
    assert lo == pytest.approx(est) and hi == pytest.approx(est)
    # ratio of sums; zero denominator is undefined
    est, *_ = slide_bootstrap(
        np.array([1.0, 3.0]), np.array([2.0, 2.0]), np.array([0, 1]), 10, rng
    )
    assert est == pytest.approx(1.0)
    assert np.isnan(slide_bootstrap(np.ones(3), np.zeros(3), np.arange(3), 10, rng)[0])


def test_verdict():
    assert verdict(-3.0, -1.0, -1) == "holds"
    assert verdict(1.0, 3.0, -1) == "opposite"
    assert verdict(-1.0, 1.0, 1) == "not supported"
    assert verdict(0.5, 2.0, 1) == "holds"


def test_summarise():
    """Δκ, relative change and switch-off share against a hand computation."""
    rows = np.arange(6)
    targets = pd.DataFrame(
        {
            "concept": "A",
            "sae_index": 0,
            "group": "top",
            "row": rows,
            "kappa_slide": 1.0,
            "slide": [0, 0, 1, 1, 2, 2],
        }
    )
    k0 = np.array([1.0, 2.0, 0.0, 4.0, 1.0, 2.0])
    k1 = np.array([0.0, 1.0, 0.5, 4.0, 0.0, 3.0])
    frames = []
    for variant in VARIANTS:
        kk = k0 if variant == "none" else k1
        frame = pd.DataFrame(
            {"row": rows, "variant": variant, "z": kk * 2, "n_changed": 1}
        )
        for j in range(16):
            frame[f"k{j}"] = kk if j == 0 else 0.0
        frames.append(frame)
    out = summarise(pd.concat(frames), targets, 200, 0)
    assert set(out["perturbation"]) == set(PERTURBATIONS) | set(RESORT)
    s = out[out["perturbation"] == "spread"].iloc[0]
    assert s["d_kappa"] == pytest.approx((k1 - k0).mean())
    assert s["rel_change"] == pytest.approx((k1 - k0).sum() / k0.sum())
    assert s["d_z"] == pytest.approx(2 * (k1 - k0).mean())
    assert s["off_share"] == pytest.approx(2 / 5)  # active before: 5, off after: 2
    assert s["on_share"] == pytest.approx(1.0)
    assert s["n_active_crop"] == 5 and s["n_slides"] == 3


def test_hypotheses_name_known_perturbations():
    assert {p for _, p, _ in HYPOTHESES} <= set(PERTURBATIONS)


def test_legacy_mask_counts_pure_pairs_twice(rng):
    """Same block pairs as the current mask; pure pairs in both lists, mixed ones once."""
    from nuclei_graph.data.block_mask import create_ragged_block_quantized_knn_mask
    from xai.concept_perturbation import legacy_knn_block_mask

    pos = rng.uniform(0, 500, (700, 2)).astype(np.float32)
    collator = GraphCollator(block_size=128, k=24, predict=True, build_block_mask=False)
    batch = collator([make_sample(rng.normal(size=(700, 67)).astype(np.float32), pos)])
    total = len(batch["global_neighbor_idx"])  # 4096: blocks 5.. hold padding only
    legacy = legacy_knn_block_mask(batch["all_knns"], 128, total)
    current = create_ragged_block_quantized_knn_mask(batch["all_knns"], 128, total)

    def dense(num: torch.Tensor, idx: torch.Tensor) -> np.ndarray:
        n = idx.shape[-2]
        out = np.zeros((n, n), bool)
        for q in range(n):
            out[q, idx[0, 0, q, : num[0, 0, q]].numpy()] = True
        return out

    partial = dense(legacy.kv_num_blocks, legacy.kv_indices)
    full = dense(legacy.full_kv_num_blocks, legacy.full_kv_indices)
    np.testing.assert_array_equal(partial, current.to_dense()[0, 0].numpy())
    assert (full <= partial).all() and full.any()
    # block 5 (rows 640-767) holds the last 60 nuclei and padding: never in the full list
    assert not full[5].any() and not full[:, 5].any()
    assert partial[5].any()


def test_judge_both_groups():
    rows = []
    for group in ("top", "active", "inactive"):
        for letter, pert, sign in HYPOTHESES:
            # top: everything drops; active: as predicted
            d = -1.0 if group == "top" else float(sign)
            rows.append(
                {
                    "group": group,
                    "concept": letter,
                    "perturbation": pert,
                    "off_share": 0.0,
                    **{
                        f"{n}{e}": d + {"": 0, "_lo": -0.5, "_hi": 0.5}[e]
                        for n in ("d_kappa", "rel_change", "d_z")
                        for e in ("", "_lo", "_hi")
                    },
                }
            )
    out = judge(pd.DataFrame(rows))
    assert set(out["group"]) == {"top", "active"}
    assert (out[out["group"] == "active"]["verdict"] == "holds").all()
    top = out[out["group"] == "top"]
    assert (top[top["predicted"] == "drop"]["verdict"] == "holds").all()
    assert (top[top["predicted"] == "rise"]["verdict"] == "opposite").all()
