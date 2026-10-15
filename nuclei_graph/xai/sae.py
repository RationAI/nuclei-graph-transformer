"""Concept autoencoders with an affine decoder (Kůr et al., unifying concept-based XAI).

Both autoencoders map a nucleus representation z ∈ R^d (output of `final_norm`) to
concept codes c ∈ R^k and back with an affine decoder D(c) = Uᵀc + b_D, where the
rows u_j of `decoder_weight` (k, d) are the concept directions.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from nuclei_graph.xai.sampling import WeightedIndexSampler


CHUNK_SIZE = 262144


def as_float64(z: Tensor | NDArray) -> NDArray[np.float64]:
    if isinstance(z, Tensor):
        return z.double().cpu().numpy()
    return np.asarray(z, dtype=np.float64)


def weighted_mean(
    z: Tensor | NDArray, weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    total = sum(
        weights[i : i + CHUNK_SIZE] @ as_float64(z[i : i + CHUNK_SIZE])
        for i in range(0, len(z), CHUNK_SIZE)
    )
    return total / weights.sum()


def covariance(
    z: Tensor | NDArray, weights: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(Weighted) mean and covariance, accumulated in float64 chunks."""
    n = len(z)
    weights = np.full(n, 1.0 / n) if weights is None else weights / weights.sum()
    mean = weighted_mean(z, weights)
    cov = np.zeros((z.shape[1], z.shape[1]))
    for i in range(0, n, CHUNK_SIZE):
        centered = as_float64(z[i : i + CHUNK_SIZE]) - mean
        cov += (centered * weights[i : i + CHUNK_SIZE, None]).T @ centered
    return mean, cov


def explained_variance(
    z: Tensor | NDArray, weights: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Cumulative explained variance ratio of the principal components."""
    eigvals = np.linalg.eigvalsh(covariance(z, weights)[1])[::-1]
    return np.cumsum(eigvals) / eigvals.sum()


class ConceptAutoencoder(nn.Module):
    """Interface: encoder E: R^d -> R^k and affine decoder D(c) = c @ U + b_D."""

    decoder_weight: Tensor  # (k, d), rows u_j
    decoder_bias: Tensor  # (d,), b_D

    def encode(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, c: Tensor) -> Tensor:
        return c @ self.decoder_weight + self.decoder_bias

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        c = self.encode(z)
        return self.decode(c), c

    @property
    def num_concepts(self) -> int:
        return self.decoder_weight.shape[0]


class SparseAutoencoder(ConceptAutoencoder):
    """c = ReLU(W_e (z - b_D) + b_e), ẑ = Uᵀc + b_D, with unit-norm decoder rows."""

    def __init__(
        self, dim: int, num_concepts: int, decoder_bias: Tensor | None = None
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(dim, num_concepts)
        self.decoder_weight = nn.Parameter(torch.randn(num_concepts, dim))
        self.decoder_bias = nn.Parameter(
            torch.zeros(dim) if decoder_bias is None else decoder_bias.clone().float()
        )
        self.normalize_decoder()
        with torch.no_grad():  # tied initialization
            self.encoder.weight.copy_(self.decoder_weight)
            self.encoder.bias.zero_()

    def encode(self, z: Tensor) -> Tensor:
        return torch.relu(self.encoder(z - self.decoder_bias))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_weight.div_(self.decoder_weight.norm(dim=1, keepdim=True))


class PCAAutoencoder(ConceptAutoencoder):
    """PCA baseline as an affine autoencoder: E(z) = V(z - μ), D(c) = Vᵀc + μ.

    The codes are not non-negative; the completeness identities still hold because
    the decoder is affine.
    """

    def __init__(self, dim: int, num_concepts: int) -> None:
        super().__init__()
        self.register_buffer("decoder_weight", torch.zeros(num_concepts, dim))
        self.register_buffer("decoder_bias", torch.zeros(dim))

    def fit(
        self, z: Tensor | NDArray, weights: NDArray[np.float64] | None = None
    ) -> "PCAAutoencoder":
        """(Weighted) PCA from the covariance, accumulated in float64 chunks."""
        mean, cov = covariance(z, weights)
        _, eigvecs = np.linalg.eigh(cov)
        components = eigvecs[:, ::-1][:, : self.num_concepts].T
        # deterministic sign: the largest-magnitude entry of each direction is positive
        signs = np.sign(
            components[np.arange(len(components)), np.abs(components).argmax(1)]
        )
        self.decoder_weight.copy_(torch.from_numpy(components * signs[:, None]))
        self.decoder_bias.copy_(torch.from_numpy(mean))
        return self

    def encode(self, z: Tensor) -> Tensor:
        return (z - self.decoder_bias) @ self.decoder_weight.T


@dataclass
class TrainingResult:
    model: SparseAutoencoder
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


def per_sample_loss(z: Tensor, z_hat: Tensor, c: Tensor, l1_coeff: float) -> Tensor:
    """‖z - ẑ‖² + λ‖c‖₁ for every sample."""
    return (z - z_hat).pow(2).sum(-1) + l1_coeff * c.abs().sum(-1)


@torch.no_grad()
def weighted_loss(
    model: SparseAutoencoder,
    z: Tensor,
    weights: Tensor,
    l1_coeff: float,
    batch_size: int,
) -> float:
    device = model.decoder_bias.device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, len(z), batch_size):
        zb = z[i : i + batch_size].to(device).float()
        z_hat, c = model(zb)
        loss = per_sample_loss(zb, z_hat, c, l1_coeff).double()
        total += loss @ weights[i : i + batch_size].to(device)
    return (total / weights.sum()).item()


def train_sae(
    z_fit: Tensor,
    weights_fit: NDArray[np.float64],
    z_holdout: Tensor,
    weights_holdout: NDArray[np.float64],
    num_concepts: int,
    l1_coeff: float,
    seed: int,
    lr: float = 1e-3,
    batch_size: int = 4096,
    samples_per_epoch: int = 2**21,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> TrainingResult:
    """Trains an SAE with Adam on minibatches drawn by `weights_fit`.

    `z_fit` may be float16 (it is cast per batch). `b_D` is initialized to the
    weighted mean of `z_fit`; decoder rows are renormalized to unit norm after every
    step. Early stopping uses the `weights_holdout`-weighted loss on `z_holdout`.
    """
    torch.manual_seed(seed)
    sampler = WeightedIndexSampler(weights_fit, seed)
    holdout_weights = torch.from_numpy(np.asarray(weights_holdout, dtype=np.float64))

    model = SparseAutoencoder(
        z_fit.shape[1],
        num_concepts,
        decoder_bias=torch.from_numpy(weighted_mean(z_fit, weights_fit)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss, best_epoch, bad_epochs = float("inf"), -1, 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        indices = torch.from_numpy(sampler.sample(samples_per_epoch))
        train_loss = 0.0
        for i in range(0, samples_per_epoch, batch_size):
            zb = z_fit[indices[i : i + batch_size].to(z_fit.device)].to(device).float()
            z_hat, c = model(zb)
            loss = per_sample_loss(zb, z_hat, c, l1_coeff).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            train_loss += loss.item() * len(zb)

        model.eval()
        val_loss = weighted_loss(model, z_holdout, holdout_weights, l1_coeff, 65536)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / samples_per_epoch,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model.cpu(), best_epoch, best_val_loss, history)


@torch.no_grad()
def encode(
    model: ConceptAutoencoder,
    z: Tensor | NDArray,
    batch_size: int = CHUNK_SIZE,
    device: str | None = None,
) -> Tensor:
    """Codes of `z` (any float dtype, tensor or array), float32 on the CPU."""
    device = device or str(model.decoder_bias.device)
    out = []
    for i in range(0, len(z), batch_size):
        zb = z[i : i + batch_size]
        zb = zb if isinstance(zb, Tensor) else torch.from_numpy(np.asarray(zb))
        out.append(model.encode(zb.to(device).float()).cpu())
    return torch.cat(out)


def save_autoencoder(model: ConceptAutoencoder, path: str, **meta: object) -> None:
    kind = "pca" if isinstance(model, PCAAutoencoder) else "sae"
    dim = model.decoder_weight.shape[1]
    torch.save(
        {"kind": kind, "dim": dim, "state_dict": model.state_dict(), "meta": meta},
        path,
    )


def load_autoencoder(path: str) -> tuple[ConceptAutoencoder, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    num_concepts = checkpoint["state_dict"]["decoder_weight"].shape[0]
    cls = PCAAutoencoder if checkpoint["kind"] == "pca" else SparseAutoencoder
    model = cls(checkpoint["dim"], num_concepts)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["meta"]
