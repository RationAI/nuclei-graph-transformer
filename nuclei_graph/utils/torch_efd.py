"""PyEFD library functions rewritten by Lukáš Hudec (from the Nuclei Graph repository) to use torch for faster computation.

The functions work the same way as the original PyEFD library, but they are optimized
to compute the coefficients using PyTorch tensors. The original
library computes them one by one, which is too slow large datasets.

The docstrings are inspired by PyEFD.
Link to PyEFD: https://pyefd.readthedocs.io/en/latest/
"""

import torch
from torch import Tensor


def normalize_efd(
    coeffs: Tensor, return_angles: bool = False
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Normalises the coefficients of the Elliptical Fourier Descriptors.

    The coefficients are normalised to have the same phase as the first major axis
    and orientation. The is not normalised in this function.

    Args:
        coeffs (Tensor): The coefficients of the Elliptical Fourier Descriptors.
        return_angles (bool, optional): Whether to return phase angle theta_1 and `axis' rotation angle psi_1. Defaults to False.

    Returns:
        tuple[Tensor, Tensor, Tensor] | Tensor: The normalised coefficients, and optionally
        the angles theta_1 and psi_1.
    """
    coeffs = coeffs.view(coeffs.shape[0], -1, 4)

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

    coeffs = coeffs.view(-1, coeffs.shape[1], 2, 2)
    rotation_matrix = torch.arange(1, coeffs.shape[1] + 1) * theta_1.view(-1, 1)
    rotation_matrix = torch.hstack(
        [
            torch.cos(rotation_matrix).view(-1, 1),
            -torch.sin(rotation_matrix).view(-1, 1),
            torch.sin(rotation_matrix).view(-1, 1),
            torch.cos(rotation_matrix).view(-1, 1),
        ]
    ).view(-1, coeffs.shape[1], 2, 2)

    coeffs = torch.matmul(coeffs, rotation_matrix)

    # rotation angle that makes the first major axis horizontal to the x axis
    psi_1 = torch.arctan2(coeffs[:, 0, 1, 0], coeffs[:, 0, 0, 0])
    psi_rotation_matrix = torch.hstack(
        [
            torch.cos(psi_1).view(-1, 1),
            torch.sin(psi_1).view(-1, 1),
            -torch.sin(psi_1).view(-1, 1),
            torch.cos(psi_1).view(-1, 1),
        ]
    ).view(-1, 1, 2, 2)

    coeffs = torch.matmul(psi_rotation_matrix, coeffs)

    if return_angles:
        return theta_1, psi_1, coeffs.view(coeffs.shape[0], -1, 4)

    return coeffs.view(coeffs.shape[0], -1, 4)
