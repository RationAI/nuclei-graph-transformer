"""Node normalization transform from the Nuclei Graph repository by Lukáš Hudec.

`Normalize`
    Rotates the EFD coefficients such that the semi-major axis of the first harmonic is parallel to the x axis.
    The angle by which this rotation is done is optionally added as a node feature using a cosine embedding.
    It also scales the coefficients by the mean of the semi-major axis of the training set.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from nuclei_graph.utils import normalize_efd


class Normalize(BaseTransform):  # type: ignore[misc]
    def __init__(self, mean: float, use_angle: bool = False) -> None:
        self.mean = mean
        self.use_angle = use_angle
        super().__init__()

    def forward(self, data: Data) -> Data:
        _, angles, coeffs = normalize_efd(data.x, return_angles=True)
        coeffs = coeffs.view(-1, 40)
        coeffs = coeffs / self.mean
        if self.use_angle:
            angles = torch.cos(angles % torch.pi).view(-1, 1)
            data.x = torch.cat([angles, coeffs], dim=1)
        else:
            data.x = coeffs
        return data
