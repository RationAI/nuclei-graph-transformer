import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression, Ridge

from nuclei_graph.xai.metrics import (
    CompletenessAccumulator,
    LinearHead,
    RidgeHead,
    RidgeStatistics,
    completeness_metrics,
    fit_ols,
    logit_agreement,
)
from nuclei_graph.xai.sae import PCAAutoencoder, SparseAutoencoder, encode, train_sae


DIM, N = 24, 3000


def random_problem(seed: int) -> tuple[np.ndarray, LinearHead]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(N, 6)) @ rng.normal(size=(6, DIM))  # low-rank + noise
    z = latent + 0.3 * rng.normal(size=(N, DIM)) + rng.normal(size=DIM)
    return z, LinearHead(rng.normal(size=DIM), float(rng.normal()))


def models(z: np.ndarray, k: int, seed: int) -> list:
    zt = torch.from_numpy(z).float()
    weights = np.full(N, 1.0 / N)
    trained = train_sae(
        zt,
        weights,
        zt[:500],
        weights[:500],
        k,
        l1_coeff=0.1,
        seed=seed,
        samples_per_epoch=4096,
        batch_size=512,
        max_epochs=10,
    )
    return [
        SparseAutoencoder(DIM, k, decoder_bias=zt.mean(0)),  # untrained
        trained.model,
        PCAAutoencoder(DIM, k).fit(zt),
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [2, 8])
def test_mce_le_fe_le_bound(seed: int, k: int) -> None:
    z, head = random_problem(seed)
    f = head(z)

    for model in models(z, k, seed):
        U = model.decoder_weight.detach().double().numpy()
        b_d = model.decoder_bias.detach().double().numpy()
        codes = encode(model, z).double().numpy()

        # least squares on the evaluation set is the infimum over affine heads,
        # which contains gamma = g ∘ D, hence MCE ≤ FE
        lstsq = LinearRegression().fit(codes, f)
        m = completeness_metrics(f, z, codes, U, b_d, head, lstsq)
        assert m["MCE"] <= m["FE"] + 1e-9
        assert m["FE"] <= m["L_g_RE"] + 1e-9
        assert m["ATE"] == pytest.approx(m["FE"], abs=1e-9)


def test_streaming_equals_one_shot() -> None:
    z, head = random_problem(0)
    pca = PCAAutoencoder(DIM, 4).fit(z)
    U = pca.decoder_weight.double().numpy()
    b_d = pca.decoder_bias.double().numpy()
    codes = encode(pca, z).double().numpy()
    eta = LinearRegression().fit(codes, head(z))

    accumulator = CompletenessAccumulator(U, b_d, head, eta)
    for i in range(0, N, 700):
        accumulator.update(head(z[i : i + 700]), z[i : i + 700], codes[i : i + 700])
    streamed = accumulator.compute()
    one_shot = completeness_metrics(head(z), z, codes, U, b_d, head, eta)
    assert streamed == pytest.approx(one_shot, rel=1e-10)
    assert len(accumulator.reconstructed) == N


@pytest.mark.parametrize("alpha", [1e-6, 1.0, 100.0])
def test_ridge_from_statistics_matches_sklearn(alpha: float) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 5)) + 2.0
    y = x @ rng.normal(size=5) + 0.1 * rng.normal(size=1000) + 3.0
    stats = RidgeStatistics(5)
    for i in range(0, 1000, 300):
        stats.update(x[i : i + 300], y[i : i + 300])
    ours = RidgeHead.from_statistics(stats, alpha)
    ref = Ridge(alpha=alpha).fit(x, y)
    np.testing.assert_allclose(ours.coef, ref.coef_, rtol=1e-8, atol=1e-10)
    assert ours.intercept == pytest.approx(ref.intercept_, rel=1e-8)


def test_weighted_pca_matches_replicated_data() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 6)) @ rng.normal(size=(6, 6))
    counts = rng.integers(1, 4, size=200)
    weighted = PCAAutoencoder(6, 3).fit(z, counts.astype(np.float64))
    replicated = PCAAutoencoder(6, 3).fit(np.repeat(z, counts, axis=0))
    np.testing.assert_allclose(
        weighted.decoder_bias, replicated.decoder_bias, atol=1e-10
    )
    np.testing.assert_allclose(
        weighted.decoder_weight, replicated.decoder_weight, atol=1e-8
    )


def test_relative_l1_error_h() -> None:
    z, head = random_problem(0)
    pca = PCAAutoencoder(DIM, 4).fit(z)
    U, b_d = pca.decoder_weight.double().numpy(), pca.decoder_bias.double().numpy()
    codes = encode(pca, z).double().numpy()
    m = completeness_metrics(
        head(z), z, codes, U, b_d, head, LinearRegression().fit(codes, head(z))
    )
    z_hat = codes @ U + b_d
    assert m["relative_l1_error_h"] == pytest.approx(
        np.abs(z - z_hat).mean() / np.abs(z).mean()
    )


def test_ols_matches_sklearn_with_dead_code() -> None:
    rng = np.random.default_rng(0)
    x = np.maximum(rng.normal(size=(2000, 4)), 0)
    x[:, 2] = 0.0  # a dead concept
    y = x @ np.array([1.0, -2.0, 0.0, 0.5]) + 0.1 * rng.normal(size=2000)
    ours = fit_ols(x, y)
    ref = LinearRegression().fit(x, y)
    np.testing.assert_allclose(ours.predict(x), ref.predict(x), atol=1e-8)


def test_logit_agreement() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(size=5000) * 3 + 1
    shrunk = 0.5 * (f - 1) + 1  # perfectly correlated, half the spread
    agreement = logit_agreement(f, shrunk)
    assert agreement["pearson_f_fA"] == pytest.approx(1.0)
    assert agreement["spearman_f_fA"] == pytest.approx(1.0)
    assert agreement["shrinkage_slope"] == pytest.approx(2.0)
    assert agreement["shrinkage_intercept"] == pytest.approx(-1.0)
    assert agreement["std_ratio_fA_f"] == pytest.approx(0.5)
