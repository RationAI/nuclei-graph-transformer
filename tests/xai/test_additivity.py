import numpy as np
import pytest
import torch

from nuclei_graph.xai.metrics import (
    LinearHead,
    baseline,
    contributions,
    grad_times_input,
    insertion,
    occlusion,
    reconstructed_logits,
)
from nuclei_graph.xai.sae import PCAAutoencoder, SparseAutoencoder, encode


DIM, K, N = 32, 8, 500


@pytest.fixture
def data() -> tuple[torch.Tensor, LinearHead]:
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.normal(size=(N, DIM)) + 3.0).float()
    head = LinearHead(rng.normal(size=DIM), float(rng.normal()))
    return z, head


def autoencoders(z: torch.Tensor) -> list:
    torch.manual_seed(0)
    sae = SparseAutoencoder(DIM, K, decoder_bias=z.mean(0))
    with torch.no_grad():
        sae.encoder.bias.normal_()  # make some, not all, concepts active
    return [sae, PCAAutoencoder(DIM, K).fit(z)]


@pytest.mark.parametrize("which", [0, 1], ids=["sae", "pca"])
def test_additivity_and_attribution_equivalence(data, which: int) -> None:
    z, head = data
    model = autoencoders(z)[which]
    U = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    codes = encode(model, z).double().numpy()
    assert (codes != 0).any()

    phi = contributions(codes, U, head)
    f_a = reconstructed_logits(codes, U, b_d, head)
    np.testing.assert_allclose(baseline(b_d, head) + phi.sum(1), f_a, atol=1e-10)

    def gamma(c: np.ndarray) -> np.ndarray:
        return reconstructed_logits(c, U, b_d, head)

    w, beta = torch.from_numpy(head.w), head.beta
    U_t, b_t = torch.from_numpy(U), torch.from_numpy(b_d)

    def gamma_torch(c: torch.Tensor) -> torch.Tensor:
        return (c @ U_t + b_t) @ w + beta

    np.testing.assert_allclose(insertion(gamma, codes), phi, atol=1e-10)
    np.testing.assert_allclose(occlusion(gamma, codes), phi, atol=1e-10)
    np.testing.assert_allclose(grad_times_input(gamma_torch, codes), phi, atol=1e-10)


def test_sae_decoder_rows_unit_norm(data) -> None:
    z, _ = data
    sae = SparseAutoencoder(DIM, K, decoder_bias=z.mean(0))
    np.testing.assert_allclose(sae.decoder_weight.norm(dim=1).detach(), 1.0, atol=1e-6)
    assert (sae.encode(z) >= 0).all()
