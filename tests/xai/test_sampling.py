import numpy as np
import pytest

from nuclei_graph.xai.sampling import (
    WeightedIndexSampler,
    balanced_slide_capped_weights,
)


def test_balanced_and_capped() -> None:
    rng = np.random.default_rng(0)
    # 300 slides of very different sizes, 80% carcinoma slides with mixed labels
    sizes = rng.lognormal(4, 1.0, size=300).astype(int) + 20
    slide_codes = np.searchsorted(
        np.cumsum(sizes), np.arange(sizes.sum()), side="right"
    )
    carcinoma_slide = rng.random(300) < 0.8
    labels = np.concatenate(
        [
            (rng.random(n) < (0.6 if c else 0.0)).astype(int)
            for n, c in zip(sizes, carcinoma_slide, strict=True)
        ]
    )

    weights, stats = balanced_slide_capped_weights(labels, slide_codes, 0.01)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[labels == 1].sum() == pytest.approx(0.5, abs=0.005)
    assert np.bincount(slide_codes, weights=weights).max() <= 0.01 * 1.011
    assert stats["effective_slides"] > 100

    # uniform within slide and label
    for s in (0, 1, 2):
        for label in (0, 1):
            w = weights[(slide_codes == s) & (labels == label)]
            if len(w):
                np.testing.assert_allclose(w, w[0])


def test_sampler_follows_weights_and_is_deterministic() -> None:
    weights = np.array([0.1, 0.0, 0.6, 0.3])
    draws = WeightedIndexSampler(weights, seed=0).sample(50_000)
    np.testing.assert_allclose(
        np.bincount(draws, minlength=4) / len(draws), weights, atol=0.01
    )
    np.testing.assert_array_equal(
        draws[:100], WeightedIndexSampler(weights, seed=0).sample(100)
    )
