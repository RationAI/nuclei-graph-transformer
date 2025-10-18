"""PyEFD library functions for per-nuclei normalization.

The functions work the same way as the original PyEFD library, except the calculation
is optimized using PyTorch tensors (in batches). The original library computes them one by one,
which is too slow large datasets.

Link to PyEFD: https://pyefd.readthedocs.io/en/latest/

Adjusted from the Nuclei Graph repository by Lukáš Hudec.
"""

import torch
from einops import rearrange
from torch import Tensor


def normalize_efd(
    coeffs: Tensor, return_angles: bool = False, size_invariant: bool = True
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Normalizes the coefficients of the Elliptical Fourier Descriptors.

    The coefficients are normalized to have the same phase as the first major axis,
    orientation and scale.

    Args:
        coeffs (Tensor): The coefficients of the Elliptical Fourier Descriptors.
        return_angles (bool, optional): Whether to return phase angle theta_1 and `axis' rotation angle psi_1. Default is False.
        size_invariant (bool, optional): If size invariance normalizing should be done as well. Default is True.

    Returns:
        tuple[Tensor, Tensor, Tensor] | Tensor: The normalised coefficients, and optionally
        the angles theta_1 and psi_1.
    """
    coeffs = rearrange(coeffs, "b (order coeffs) -> b order coeffs", coeffs=4)

    # phase rotation angle
    theta_1 = 0.5 * torch.arctan2(
        2 * ((coeffs[:, 0, 0] * coeffs[:, 0, 1]) + (coeffs[:, 0, 2] * coeffs[:, 0, 3])),
        (
            torch.pow(coeffs[:, 0, 0], 2)
            - torch.pow(coeffs[:, 0, 1], 2)
            + torch.pow(coeffs[:, 0, 2], 2)
            - torch.pow(coeffs[:, 0, 3], 2)
        ),
    )

    coeffs = rearrange(coeffs, "b order (d1 d2) -> b order d1 d2", d1=2, d2=2)
    harmonic_indices = torch.arange(
        1, coeffs.shape[1] + 1, device=coeffs.device
    ).float()
    rotation_angles = harmonic_indices * rearrange(theta_1, "b -> b 1")

    cos_t = torch.cos(rotation_angles)
    sin_t = torch.sin(rotation_angles)

    rotation_matrix = torch.stack(
        [
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t, cos_t], dim=-1),
        ],
        dim=-2,
    )

    coeffs = torch.matmul(coeffs, rotation_matrix)

    first_harmonic_rotated = coeffs[:, 0, :, :]
    psi_1 = torch.arctan2(
        first_harmonic_rotated[:, 1, 0], first_harmonic_rotated[:, 0, 0]
    )

    cos_p = torch.cos(psi_1)
    sin_p = torch.sin(psi_1)

    # rotation angle that makes the first major axis horizontal to the x axis
    psi_rotation_matrix = torch.stack(
        [
            torch.stack([cos_p, sin_p], dim=-1),
            torch.stack([-sin_p, cos_p], dim=-1),
        ],
        dim=-2,
    )
    psi_rotation_matrix = rearrange(psi_rotation_matrix, "b d1 d2 -> b 1 d1 d2")

    coeffs = torch.matmul(psi_rotation_matrix, coeffs)

    if size_invariant:
        scaling_factor = coeffs[:, 0, 0, 0]  # first element of the first harmonic
        scaling_factor = torch.abs(scaling_factor) + 1e-6
        divisor = rearrange(scaling_factor, "b -> b 1 1 1")
        coeffs = coeffs / divisor

    coeffs = rearrange(coeffs, "b h d1 d2 -> b (h d1 d2)")

    if return_angles:
        return theta_1, psi_1, coeffs

    return coeffs
