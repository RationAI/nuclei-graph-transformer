from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from nuclei_graph.data.datasets import NucleiDataset


class AutoWeightedRandomSampler(WeightedRandomSampler):
    """Weighted Random Sampler with weights based on the class distribution.

    It assigns weights to each sample in the dataset based on the inverse of the class frequency.

    Arguments:
        dataset: Torch dataset class (NucleiDataset).
        slides_positivity: Dictionary mapping slide IDs to their positivity scores.
        positivity_threshold: Threshold for considering a graph to be positive in the sampler.
        replacement: If True, samples are drawn with replacement. Default is True.
    """

    def __init__(
        self,
        dataset: NucleiDataset,
        slides_positivity: dict[str, float],
        positivity_threshold: float = 0.0,
        replacement: bool = True,
    ):
        weights = self._get_weights(dataset, slides_positivity, positivity_threshold)
        super().__init__(weights, num_samples=len(dataset), replacement=replacement)

    def _get_weights(
        self,
        dataset: NucleiDataset,
        slides_positivity: dict[str, float],
        positivity_threshold: float,
    ) -> Sequence[float]:
        slide_ids = dataset.df_metadata["slide_id"].values
        labels = torch.tensor(
            [
                1.0 if slides_positivity[slide_id] > positivity_threshold else 0.0
                for slide_id in slide_ids
            ]
        )
        positive = labels.sum()
        negative = len(labels) - positive
        weights = torch.zeros_like(labels)
        weights[labels == 0] = 1.0 / negative
        weights[labels == 1] = 1.0 / positive
        return weights.tolist()
