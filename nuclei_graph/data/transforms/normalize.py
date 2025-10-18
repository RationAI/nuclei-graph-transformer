"""Node normalization transforms."""

import torch
from mlflow.artifacts import download_artifacts
from torch_geometric.transforms import BaseTransform

from nuclei_graph.typing import FeatureDict
from nuclei_graph.utils import normalize_efd


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


class Normalize(BaseTransform):  # type: ignore[misc]
    """Standard Z-Score normalization transform for EFD features.

    Applies: x' = (x - mean) / std, where mean and std are calculated globally
    (after `NormalizeEFD`) for each feature dimension across the training dataset.
    """

    def __init__(self, mean_uri: str, std_uri: str) -> None:
        """Initialize the Normalization transform.

        Args:
            mean_uri (str): MLflow artifact URI to the mean .pt file.
            std_uri (str): MLflow artifact URI to the std .pt file.
        """
        super().__init__()
        self.mean_uri = mean_uri
        self.std_uri = std_uri
        self.setup()

    def setup(self) -> None:
        self.mean = torch.load(download_artifacts(self.mean_uri))
        self.std = torch.load(download_artifacts(self.std_uri))

    def forward(self, data: FeatureDict) -> FeatureDict:
        assert data["x"] is not None
        data["x"] = (data["x"] - self.mean) / (self.std + 1e-6)
        return data
