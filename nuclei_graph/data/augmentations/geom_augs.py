import torch

from nuclei_graph.nuclei_graph_typing import Batch


def apply_jitter(batch: Batch, jitter_std: float = 0.02) -> Batch:
    batch = batch.copy()
    batch["pos"][:, :2] += torch.randn_like(batch["pos"][:, :2]) * jitter_std
    return batch


def apply_rotation(batch: Batch) -> Batch:
    batch = batch.copy()
    angle_shift = torch.rand(1, device=batch["pos"].device) * 2 * torch.pi
    batch["pos"][:, 2] = (batch["pos"][:, 2] + angle_shift) % (2 * torch.pi)
    return batch


def apply_feature_noise(batch: Batch, noise_std: float = 0.01) -> Batch:
    batch = batch.copy()
    batch["x"] += torch.randn_like(batch["x"]) * noise_std
    return batch


@torch.no_grad()
def apply_augmentations(batch: Batch) -> Batch:
    batch = apply_jitter(batch)
    batch = apply_rotation(batch)
    batch = apply_feature_noise(batch)
    return batch
