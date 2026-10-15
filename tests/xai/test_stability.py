import numpy as np

from nuclei_graph.xai.sae import PCAAutoencoder, explained_variance
from nuclei_graph.xai.stability import matched_cosines


def test_hungarian_recovers_permutation() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(8, 32))
    perm = rng.permutation(8)
    other = reference[perm] + 0.01 * rng.normal(size=(8, 32))
    match, cosine = matched_cosines(reference, other)
    np.testing.assert_array_equal(perm[match], np.arange(8))
    assert (cosine > 0.99).all()


def test_explained_variance_matches_pca() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 10)) @ rng.normal(size=(10, 10))
    cumulative = explained_variance(z)
    assert cumulative[-1] == np.float64(1.0) or abs(cumulative[-1] - 1) < 1e-12
    eig = np.linalg.eigvalsh(np.cov(z.T, bias=True))[::-1]
    np.testing.assert_allclose(cumulative, np.cumsum(eig) / eig.sum(), rtol=1e-10)
    assert PCAAutoencoder(10, 3).fit(z).decoder_weight.shape == (3, 10)
