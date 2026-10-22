import numpy as np
import torch

from nuclei_graph.xai.metrics import (
    LinearHead,
    baseline,
    contributions,
    reconstructed_logits,
)
from nuclei_graph.xai.sae import encode
from nuclei_graph.xai.topk import (
    TopKAutoencoder,
    dense_codes,
    encode_topk,
    input_scale,
    load_topk,
    save_topk,
    train_topk_sae,
)


DIM, M, K, N = 32, 64, 4, 2000


def data() -> tuple[torch.Tensor, LinearHead]:
    rng = np.random.default_rng(0)
    z = torch.from_numpy(5.0 * rng.normal(size=(N, DIM)) + 3.0).float()
    head = LinearHead(rng.normal(size=DIM), float(rng.normal()))
    return z, head


def test_input_scale() -> None:
    z, _ = data()
    scale = input_scale(z)
    assert np.isclose((z.double() / scale).pow(2).sum(1).mean().item(), DIM)


def test_topk_codes_and_additivity_in_original_units(tmp_path) -> None:
    z, head = data()
    torch.manual_seed(0)
    model = TopKAutoencoder(DIM, M, K, scale=input_scale(z), decoder_bias=z.mean(0) / 4)
    assert torch.allclose(model.w_dec.norm(dim=1), torch.ones(M))
    codes = encode(model, z).double().numpy()
    assert ((codes != 0).sum(1) <= K).all() and (codes >= 0).all()

    indices, values = encode_topk(model, z)
    np.testing.assert_allclose(dense_codes(indices, values, M), codes, atol=1e-6)

    U = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    z_hat = model.decode(torch.from_numpy(codes).float()).detach()
    z_hat_n = torch.from_numpy(codes).float() @ model.w_dec + model.b_dec
    torch.testing.assert_close(z_hat, z_hat_n.detach() * model.scale.float())
    phi = contributions(codes, U, head)
    f_a = reconstructed_logits(codes, U, b_d, head)
    np.testing.assert_allclose(baseline(b_d, head) + phi.sum(1), f_a, atol=1e-9)

    save_topk(model, str(tmp_path / "m.pt"), seed=0)
    loaded, meta = load_topk(str(tmp_path / "m.pt"))
    assert meta == {"seed": 0} and loaded.k == K
    torch.testing.assert_close(encode(loaded, z), encode(model, z))


def test_training_reduces_holdout_mse() -> None:
    z, _ = data()
    weights = np.full(N, 1.0 / N)
    result = train_topk_sae(
        z,
        weights,
        z[:500],
        weights[:500],
        num_latents=M,
        k=K,
        seed=0,
        lr=1e-2,
        warmup_steps=5,
        batch_size=256,
        samples_per_epoch=4096,
        max_epochs=5,
        patience=5,
        k_aux=16,
        dead_after=2048,
    )
    history = result.history
    assert history[-1]["val_mse"] < history[0]["val_mse"]
    assert torch.allclose(result.model.w_dec.norm(dim=1), torch.ones(M), atol=1e-5)


def test_sparse_ols_head_equals_fit_ols() -> None:
    from nuclei_graph.xai.metrics import fit_ols
    from xai.evaluate_topk_sae import ols_head

    z, head = data()
    torch.manual_seed(0)
    model = TopKAutoencoder(DIM, M, K, scale=input_scale(z), decoder_bias=z.mean(0) / 4)
    indices, values = encode_topk(model, z)
    f = head(z.double().numpy())
    sparse = ols_head((indices, values), M, f)
    dense = fit_ols(dense_codes(indices, values, M), f)
    codes = dense_codes(indices, values, M)
    np.testing.assert_allclose(sparse.predict(codes), dense.predict(codes), atol=1e-8)
