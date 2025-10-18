from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Compose(BaseTransform):  # type: ignore[misc]
    """Apply multiple transforms sequentially."""

    def __init__(self, transforms: list[BaseTransform]) -> None:
        """Initialize the transform.

        Args:
            transforms (list): List of transforms (callables) to apply.
        """
        self.transforms = transforms

    def forward(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data
