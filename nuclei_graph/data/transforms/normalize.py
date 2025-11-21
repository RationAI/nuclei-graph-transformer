"""Node normalization transforms."""

from torch_geometric.transforms import BaseTransform

from nuclei_graph.features import normalize_efd
from nuclei_graph.nuclei_graph_typing import FeatureDict


class NormalizeEFD(BaseTransform):  # type: ignore[misc]
    """Transform that normalizes EFD coefficients per nucleus.

    Performs phase, rotation, and scale normalization to make the descriptors
    invariant to the contour's starting point, orientation, and size.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: FeatureDict) -> FeatureDict:
        normalized_efd_features = normalize_efd(data["x"])
        if isinstance(normalized_efd_features, tuple):  # mypy
            normalized_efd_features = normalized_efd_features[2]  # take coeffs
        data["x"] = normalized_efd_features
        return data
