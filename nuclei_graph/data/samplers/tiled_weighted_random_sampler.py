from collections.abc import Sequence

import pandas as pd
from torch.utils.data import WeightedRandomSampler


class TileWeightedRandomSampler(WeightedRandomSampler):
    """Weighted Random Sampler for Tile-Level classification."""

    def __init__(
        self,
        tiles_df: pd.DataFrame,
        target_col: str = "carcinoma",
        replacement: bool = True,
    ) -> None:
        super().__init__(
            weights=self._get_weights(tiles_df, target_col),
            num_samples=len(tiles_df),
            replacement=replacement,
        )

    def _get_weights(self, df: pd.DataFrame, target_col: str) -> Sequence[float]:
        value_counts = df[target_col].value_counts()
        weights = 1 / df[target_col].map(value_counts)
        return weights.tolist()
