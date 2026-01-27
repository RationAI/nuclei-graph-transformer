import torch

from nuclei_graph.nuclei_graph_typing import Batch


def apply_jitter(batch: Batch, jitter_std: float = 0.03) -> Batch:
    batch = batch.copy()
    pos = batch["pos"].clone()  # avoid in-place modification
    pos[:, :2] += torch.randn_like(pos[:, :2]) * jitter_std
    batch["pos"] = pos
    return batch


def apply_rotation(batch: Batch) -> Batch:
    batch = batch.copy()
    pos = batch["pos"].clone()  # avoid in-place modification
    angle_shift = torch.rand(1, device=pos.device) * 2 * torch.pi
    pos[:, 2] = (pos[:, 2] + angle_shift) % (2 * torch.pi)
    batch["pos"] = pos
    return batch


def apply_feature_noise(batch: Batch, noise_std: float = 0.02) -> Batch:
    batch = batch.copy()
    x = batch["x"].clone()  # avoid in-place modification
    x += torch.randn_like(x) * noise_std
    batch["x"] = x
    return batch


@torch.no_grad()
def apply_augmentations(batch: Batch) -> Batch:
    batch = apply_jitter(batch)
    batch = apply_rotation(batch)
    batch = apply_feature_noise(batch)
    return batch
