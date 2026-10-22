"""Overcomplete TopK sparse autoencoder (Gao et al., Scaling and evaluating sparse autoencoders, 2024).

The SAE works on normalized inputs h_n = h / c, with one scalar c such that
E‖h_n‖² = d on the training data:

    z = TopK(ReLU(W_enc (h_n - b_dec) + b_enc)),   ĥ_n = W_dec z + b_dec,

with unit-norm decoder columns (rows of `w_dec`, shape (m, d)). As a
`ConceptAutoencoder` it maps unnormalized h to h: `decoder_weight` = c · W_decᵀ and
`decoder_bias` = c · b_dec, so ĥ = c · ĥ_n and every completeness identity of
`nuclei_graph.xai.metrics` holds in the original units.

Training: MSE ‖h_n - ĥ_n‖² (summed over dimensions, as in `sae.per_sample_loss`) plus
`aux_coeff` times the AuxK loss: the `k_aux` dead latents with the largest
pre-activations reconstruct the residual e = h_n - ĥ_n (detached), ‖e - W_dec z_aux‖².
A latent is dead if it has not fired in the last `dead_after` training samples.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from nuclei_graph.xai.sae import CHUNK_SIZE, ConceptAutoencoder, weighted_mean
from nuclei_graph.xai.sampling import WeightedIndexSampler


def input_scale(
    z: Tensor | NDArray, weights: NDArray[np.float64] | None = None
) -> float:
    """C with E‖z / c‖² = d under `weights` (uniform if None), in float64 chunks."""
    n = len(z)
    weights = np.full(n, 1.0 / n) if weights is None else weights / weights.sum()
    sq_norm = 0.0
    for i in range(0, n, CHUNK_SIZE):
        zb = z[i : i + CHUNK_SIZE]
        zb = (
            zb.double().cpu().numpy()
            if isinstance(zb, Tensor)
            else np.asarray(zb, np.float64)
        )
        sq_norm += float(weights[i : i + CHUNK_SIZE] @ np.square(zb).sum(1))
    return float(np.sqrt(sq_norm / z.shape[1]))


def sparse_decode(indices: Tensor, values: Tensor, w_dec: Tensor) -> Tensor:
    """Σ_k values_k · w_dec[indices_k] per row, without the dense (n, m) codes."""
    return F.embedding_bag(indices, w_dec, per_sample_weights=values, mode="sum")


class TopKAutoencoder(ConceptAutoencoder):
    def __init__(
        self,
        dim: int,
        num_latents: int,
        k: int,
        scale: float = 1.0,
        decoder_bias: Tensor | None = None,
    ) -> None:
        """`decoder_bias` is b_dec in normalized units (the mean of h_n)."""
        super().__init__()
        self.k = k
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))
        self.encoder = nn.Linear(dim, num_latents)
        self.w_dec = nn.Parameter(torch.randn(num_latents, dim))
        self.b_dec = nn.Parameter(
            torch.zeros(dim) if decoder_bias is None else decoder_bias.clone().float()
        )
        self.normalize_decoder()
        with torch.no_grad():  # W_enc = W_decᵀ
            self.encoder.weight.copy_(self.w_dec)
            self.encoder.bias.zero_()

    @property
    def decoder_weight(self) -> Tensor:  # (m, d), in units of h
        return self.w_dec * self.scale.float()

    @property
    def decoder_bias(self) -> Tensor:  # (d,), in units of h
        return self.b_dec * self.scale.float()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.w_dec.div_(self.w_dec.norm(dim=1, keepdim=True))

    def pre_activation(self, z_n: Tensor) -> Tensor:
        return torch.relu(self.encoder(z_n - self.b_dec))

    def topk(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Indices and values of the K active latents of unnormalized z."""
        values, indices = self.pre_activation(z / self.scale.float()).topk(
            self.k, dim=1
        )
        return indices, values

    def encode(self, z: Tensor) -> Tensor:
        indices, values = self.topk(z)
        codes = torch.zeros(len(z), self.num_concepts, device=z.device)
        return codes.scatter_(1, indices, values)


@dataclass
class TopKTrainingResult:
    model: TopKAutoencoder
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


@torch.no_grad()
def weighted_mse(
    model: TopKAutoencoder, z: Tensor, weights: Tensor, batch_size: int = 65536
) -> float:
    """`weights`-weighted ‖h_n - ĥ_n‖² (normalized units)."""
    device = model.b_dec.device
    scale = model.scale.float()
    total = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, len(z), batch_size):
        zb = z[i : i + batch_size].to(device).float() / scale
        values, indices = model.pre_activation(zb).topk(model.k, dim=1)
        z_hat = sparse_decode(indices, values, model.w_dec) + model.b_dec
        sq_err = (zb - z_hat).pow(2).sum(-1).double()
        total += sq_err @ weights[i : i + batch_size].to(device)
    return (total / weights.sum()).item()


def train_topk_sae(
    z_fit: Tensor,
    weights_fit: NDArray[np.float64],
    z_holdout: Tensor,
    weights_holdout: NDArray[np.float64],
    num_latents: int,
    k: int,
    seed: int,
    lr: float = 2e-4,
    warmup_steps: int = 1000,
    batch_size: int = 4096,
    samples_per_epoch: int = 2**21,
    max_epochs: int = 100,
    patience: int = 10,
    k_aux: int = 256,
    aux_coeff: float = 1 / 32,
    dead_after: int = 1_000_000,
    device: str = "cpu",
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> TopKTrainingResult:
    """Trains a TopK SAE with Adam on minibatches drawn by `weights_fit`.

    As `sae.train_sae`: `z_fit` may be float16; c and b_dec (the mean of h_n) come
    from `z_fit` under `weights_fit`; decoder columns are renormalized after every
    step; early stopping uses the `weights_holdout`-weighted MSE on `z_holdout`
    (without the AuxK term). The learning rate rises linearly over `warmup_steps`.
    `on_epoch` receives every epoch's history entry (e.g. for live logging).
    """
    torch.manual_seed(seed)
    sampler = WeightedIndexSampler(weights_fit, seed)
    holdout_weights = torch.from_numpy(np.asarray(weights_holdout, dtype=np.float64))

    scale = input_scale(z_fit, weights_fit)
    model = TopKAutoencoder(
        z_fit.shape[1],
        num_latents,
        k,
        scale=scale,
        decoder_bias=torch.from_numpy(weighted_mean(z_fit, weights_fit) / scale),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    since_fired = torch.zeros(num_latents, dtype=torch.long, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss, best_epoch, bad_epochs = float("inf"), -1, 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        indices = torch.from_numpy(sampler.sample(samples_per_epoch))
        sums = dict.fromkeys(("mse", "aux", "dead"), 0.0)
        steps = 0
        for i in range(0, samples_per_epoch, batch_size):
            zb = z_fit[indices[i : i + batch_size].to(z_fit.device)].to(device).float()
            zb = zb / scale
            pre = model.pre_activation(zb)
            values, top = pre.topk(k, dim=1)
            z_hat = sparse_decode(top, values, model.w_dec) + model.b_dec
            mse = (zb - z_hat).pow(2).sum(-1).mean()

            fired = torch.zeros(num_latents, dtype=torch.bool, device=device)
            fired[top.flatten()] = True
            since_fired += len(zb)
            since_fired[fired] = 0
            dead = since_fired >= dead_after
            num_dead = int(dead.sum())
            aux = torch.zeros((), device=device)
            if num_dead:
                aux_values, aux_top = pre.masked_fill(~dead, 0.0).topk(
                    min(k_aux, num_dead), dim=1
                )
                residual = (zb - z_hat).detach()
                aux_hat = sparse_decode(aux_top, aux_values, model.w_dec)
                aux = (residual - aux_hat).pow(2).sum(-1).mean()

            loss = mse + aux_coeff * aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            warmup.step()
            model.normalize_decoder()
            sums["mse"] += mse.item()
            sums["aux"] += aux.item()
            sums["dead"] += num_dead / num_latents
            steps += 1

        model.eval()
        val_loss = weighted_mse(model, z_holdout, holdout_weights)
        history.append(
            {
                "epoch": epoch,
                "train_mse": sums["mse"] / steps,
                "train_aux": sums["aux"] / steps,
                "train_dead_fraction": sums["dead"] / steps,
                "dead_fraction_end": float(dead.float().mean()),
                "val_mse": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if on_epoch is not None:
            on_epoch(history[-1])

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TopKTrainingResult(model.cpu(), best_epoch, best_val_loss, history)


@torch.no_grad()
def encode_topk(
    model: TopKAutoencoder,
    z: Tensor | NDArray,
    batch_size: int = 65536,
    device: str | None = None,
) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
    """Sparse codes of unnormalized `z`: latent indices (n, K) and values (n, K)."""
    device = device or str(model.b_dec.device)
    assert model.num_concepts <= np.iinfo(np.int16).max
    indices, values = [], []
    for i in range(0, len(z), batch_size):
        zb = z[i : i + batch_size]
        zb = zb if isinstance(zb, Tensor) else torch.from_numpy(np.asarray(zb))
        top, vals = model.topk(zb.to(device).float())
        indices.append(top.cpu().numpy().astype(np.int16))
        values.append(vals.cpu().numpy())
    return np.concatenate(indices), np.concatenate(values)


def dense_codes(
    indices: NDArray[np.integer], values: NDArray[np.floating], num_latents: int
) -> NDArray[np.float64]:
    codes = np.zeros((len(indices), num_latents))
    np.put_along_axis(codes, indices.astype(np.int64), values, axis=1)
    return codes


def save_topk(model: TopKAutoencoder, path: str, **meta: object) -> None:
    torch.save(
        {
            "kind": "topk",
            "dim": model.w_dec.shape[1],
            "num_latents": model.num_concepts,
            "k": model.k,
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        path,
    )


def load_topk(path: str) -> tuple[TopKAutoencoder, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["kind"] == "topk", path
    model = TopKAutoencoder(
        checkpoint["dim"], checkpoint["num_latents"], checkpoint["k"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["meta"]
